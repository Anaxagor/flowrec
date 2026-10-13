"""Optimizer-boundary resume for expensive exp7 epochs.

The data and collator must be deterministic for a given index, as the exp7
indexed reward loader is. Worker-local random data augmentation is not replayed.
Model/method randomness (Python, NumPy, Torch and CUDA) is restored after the
already processed batches have been skipped. Shuffle uses a separate generator
from worker initialization, including with persistent DataLoader workers.
"""
from __future__ import annotations

import json
from dataclasses import replace
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import RandomSampler

from flowrec.training import Trainer
from flowrec.training.trainer import FitResult, trainer_state_to_dict


def capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class _RemainingBatches:
    def __init__(self, loader, trainer, offset: int, limit: int, rng) -> None:
        self.loader, self.trainer = loader, trainer
        self.offset, self.limit, self.rng = offset, limit, rng

    def __len__(self) -> int:
        return self.limit - self.offset

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.offset):
            next(iterator)
        if self.rng is not None:
            restore_rng(self.rng)
        for position in range(self.offset, self.limit):
            batch = next(iterator)
            # The trainer only saves after this batch's optimizer step succeeds.
            self.trainer._completed_batches = position + 1
            yield batch


class EpochSeededTrainer(Trainer):
    def __init__(self, *, seed: int = 42, checkpoint_every_updates: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.checkpoint_every_updates = int(checkpoint_every_updates)
        if self.checkpoint_every_updates < 0:
            raise ValueError("checkpoint_every_updates must be nonnegative")
        if self.cfg.validate_every_steps is not None:
            raise ValueError("Resumable exp7 training requires epoch-level validation")
        self._completed_batches = 0
        self._epoch_batch_limit: int | None = None
        self._epoch_complete = False
        self._in_epoch = False
        self._epoch_original_config = None
        self._resume_rng = None
        self._resume_epoch: int | None = None
        self._last_completed_epoch: int | None = None

    @classmethod
    def from_optimizer_config(cls, *, seed: int = 42, checkpoint_every_updates: int = 10, **kwargs):
        if int(checkpoint_every_updates) < 0:
            raise ValueError("checkpoint_every_updates must be nonnegative")
        trainer = super().from_optimizer_config(**kwargs)
        trainer.seed = int(seed)
        trainer.checkpoint_every_updates = int(checkpoint_every_updates)
        return trainer

    def restore_progress(self, payload: Mapping[str, Any]) -> None:
        """Call after load_checkpoint; old epoch-boundary files remain readable."""
        progress = payload.get("epoch_progress", {"epoch_complete": True})
        if int(progress.get("seed", self.seed)) != self.seed:
            raise ValueError("Checkpoint shuffle seed differs from this trainer")
        if int(progress.get("gradient_accumulation_steps", self.cfg.gradient_accumulation_steps)) != self.cfg.gradient_accumulation_steps:
            raise ValueError("Checkpoint gradient accumulation differs from this trainer")
        self._epoch_complete = bool(progress["epoch_complete"])
        self._completed_batches = int(progress.get("completed_batches", 0))
        self._epoch_batch_limit = progress.get("epoch_batch_limit")
        if self._completed_batches < 0:
            raise ValueError("Checkpoint completed_batches must be nonnegative")
        self._resume_rng = None
        self._resume_epoch = None
        if self._epoch_complete:
            self._last_completed_epoch = self.state.epoch
            self.state.epoch += 1
        else:
            if "rng_state" not in progress:
                raise ValueError("An unfinished epoch checkpoint requires RNG state")
            self._resume_rng = progress["rng_state"]
            self._resume_epoch = self.state.epoch

    def fit(self, train_loader, val_loader=None, test_loader=None):
        # Generic Trainer checks should_stop only after another epoch. A saved
        # early-stop result must go straight to final evaluation on resubmission.
        if self.state.should_stop or self.state.epoch >= self.cfg.max_epochs:
            if self._last_completed_epoch is not None:
                self.state.epoch = self._last_completed_epoch
            root = Path(self.cfg.checkpoint_dir) if self.cfg.checkpoint_dir else None
            best = root / "best.pt" if root else None
            last = root / "last.pt" if root else None
            return FitResult(
                state=self.state,
                best_checkpoint_path=str(best) if best and best.exists() else None,
                last_checkpoint_path=str(last) if last and last.exists() else None,
            )
        return super().fit(train_loader, val_loader, test_loader)

    def train_one_epoch(self, train_loader, *, epoch=None, val_loader=None):
        epoch = self.state.epoch if epoch is None else int(epoch)
        self.state.epoch = epoch
        epoch_seed = self.seed + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % 2**32)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        generator = getattr(train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(epoch_seed)
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, RandomSampler):
            sampler.generator = torch.Generator().manual_seed(epoch_seed)
        elif hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        limit = len(train_loader)
        original_max_batches = self.cfg.max_train_batches
        if original_max_batches is not None:
            limit = min(limit, original_max_batches)
        resuming = self._resume_epoch == epoch
        offset = self._completed_batches if resuming else 0
        if resuming and self._epoch_batch_limit is not None and limit != self._epoch_batch_limit:
            raise ValueError("Training loader length or max_train_batches changed during an unfinished epoch")
        if offset > limit:
            raise ValueError("Checkpoint offset exceeds the training epoch length")
        rng = self._resume_rng if resuming else None
        self._epoch_batch_limit = limit
        self._completed_batches = offset
        self._epoch_complete = False
        self._in_epoch = True
        self._resume_epoch = None
        self._resume_rng = None
        original_config = self.cfg
        self._epoch_original_config = original_config
        self.cfg = replace(original_config, max_train_batches=limit - offset)
        try:
            result = super().train_one_epoch(
                _RemainingBatches(train_loader, self, offset, limit, rng),
                epoch=epoch, val_loader=val_loader,
            )
        finally:
            self._in_epoch = False
            self.cfg = original_config
            self._epoch_original_config = None
        self._epoch_complete = True
        self._last_completed_epoch = epoch
        return result

    def _optimizer_step(self) -> None:
        super()._optimizer_step()
        if (self._in_epoch and self.checkpoint_every_updates > 0
                and self.state.optimizer_step % self.checkpoint_every_updates == 0
                and self.cfg.save_last and self.cfg.checkpoint_dir is not None):
            self.save_checkpoint("last.pt")

    def save_checkpoint(self, filename: str | Path) -> str:
        if self.cfg.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set before saving checkpoints")
        if self._in_epoch and any(parameter.grad is not None for parameter in self._trainable_parameters()):
            raise RuntimeError("Intra-epoch checkpoints require cleared accumulated gradients")
        path = Path(filename)
        if not path.is_absolute():
            path = Path(self.cfg.checkpoint_dir) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        # The loop temporarily caps the remaining suffix of a resumed epoch.
        config = (self._epoch_original_config or self.cfg).to_dict()
        progress = {
            "version": 1, "seed": self.seed, "epoch_complete": self._epoch_complete,
            "completed_batches": self._completed_batches, "epoch_batch_limit": self._epoch_batch_limit,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "rng_state": capture_rng(),
        }
        metadata = {
            "trainer_config": config, "trainer_state": trainer_state_to_dict(self.state),
            "model_type": self.model.model_type, "model_class": self.model.__class__.__name__,
            "model_config": self.model.config_dict(),
            "method_name": getattr(self.method, "method_name", self.method.__class__.__name__),
            "method_class": self.method.__class__.__name__,
        }
        payload = {
            **metadata, "epoch_progress": progress,
            "model_state_dict": self.model.state_dict(), "method_state_dict": self.method.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None and hasattr(self.scheduler, "state_dict") else None,
            "amp_scaler_state_dict": self._scaler.state_dict() if self._use_amp else None,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # This diagnostic is not used for resume; the .pt contains all state.
        metadata["epoch_progress"] = {key: value for key, value in progress.items() if key != "rng_state"}
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)
