"""Epoch-boundary recovery for the opt-in matched GPT-2 tuning protocol.

Checkpoint continuation restores optimizer/scaler and stopping state. RNG and
DataLoader generator states are not recorded by Trainer: shuffling restarts,
so this is deliberately not a claim of bitwise reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch

from flowrec.training.trainer import Trainer


def _best_snapshot(directory: Path, epoch: int) -> Path:
    return directory / f"resume_best_epoch_{epoch}.pt"


class AtomicTrialTrainer(Trainer):
    """Keep the previous complete checkpoint if a worker dies during a save."""

    def save_checkpoint(self, filename: str | Path) -> str:
        path = Path(filename)
        if not path.is_absolute():
            if self.cfg.checkpoint_dir is None:
                raise ValueError("checkpoint_dir must be set before saving checkpoints")
            path = Path(self.cfg.checkpoint_dir) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary)
        sidecar = temporary.with_suffix(temporary.suffix + ".json")
        try:
            super().save_checkpoint(temporary)
            if path.name == "best.pt" and self.state.best_epoch is not None:
                # last.pt must retain access to ITS best weights if the next
                # validation saves a new best and dies before saving last.pt.
                snapshot = _best_snapshot(path.parent, self.state.best_epoch)
                try:
                    os.link(temporary, snapshot)
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(temporary, snapshot)
            os.replace(temporary, path)
            os.replace(sidecar, path.with_suffix(path.suffix + ".json"))
            if path.name == "last.pt":
                retained = _best_snapshot(path.parent, self.state.best_epoch) if self.state.best_epoch is not None else None
                for snapshot in path.parent.glob("resume_best_epoch_*.pt"):
                    if snapshot != retained:
                        snapshot.unlink(missing_ok=True)
            return str(path)
        finally:
            temporary.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)


@dataclass(frozen=True)
class ResumeCheckpoint:
    last_path: Path
    best_path: Path | None
    completed_epochs: int


def inspect_resume_checkpoint(directory: Path, identity: Mapping[str, Any]) -> ResumeCheckpoint | None:
    """Verify local provenance and integrity before queuing a checkpoint retry."""
    last = directory / "last.pt"
    if not last.exists():
        return None
    manifest_path = directory / "resolved_config.json"
    if not manifest_path.is_file():
        raise ValueError(f"Cannot resume {last}: missing resolved_config.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("resume_identity") != dict(identity):
        raise ValueError(f"Cannot resume {last}: checkpoint protocol, data, parameters or seed differ")
    payload = torch.load(last, map_location="cpu", weights_only=False)
    if any(payload.get(key) is None for key in (
        "trainer_state", "model_state_dict", "optimizer_state_dict", "trainer_config", "model_config",
    )):
        raise ValueError(f"Cannot resume {last}: incomplete training checkpoint")
    normalize = lambda value: json.loads(json.dumps(value, default=str))
    if normalize(payload["model_config"]) != manifest.get("model_config"):
        raise ValueError(f"Cannot resume {last}: model configuration differs from its manifest")
    trainer_config = normalize(payload["trainer_config"])
    expected_config = dict(manifest["trainer"])
    trainer_config.pop("checkpoint_dir", None)
    expected_config.pop("checkpoint_dir", None)
    if trainer_config != expected_config:
        raise ValueError(f"Cannot resume {last}: trainer configuration differs from its manifest")
    for group in payload["optimizer_state_dict"].get("param_groups", []):
        for key in ("lr", "weight_decay"):
            if group[key] != identity["hparams"][key]:
                raise ValueError(f"Cannot resume {last}: optimizer {key} differs")
    state = payload["trainer_state"]
    completed = int(state["epoch"]) + 1
    best_epoch, best_metric = state.get("best_epoch"), state.get("best_metric")
    if completed < 1 or int(state.get("global_step", 0)) < 1:
        raise ValueError(f"Cannot resume {last}: no completed training epoch")
    del payload
    best = None
    if best_metric is not None:
        if best_epoch is None:
            raise ValueError(f"Cannot resume {last}: missing best epoch")
        snapshot = _best_snapshot(directory, int(best_epoch))
        best = snapshot if snapshot.exists() else directory / "best.pt"
        if not best.is_file():
            raise ValueError(f"Cannot resume {last}: missing best checkpoint")
        best_payload = torch.load(best, map_location="cpu", weights_only=False)
        best_state = best_payload.get("trainer_state", {})
        if best_state.get("best_epoch") != best_epoch or best_state.get("best_metric") != best_metric:
            raise ValueError(f"Cannot resume {last}: best checkpoint does not match saved stopping state")
        if "model_state_dict" not in best_payload:
            raise ValueError(f"Cannot resume {last}: incomplete best checkpoint")
    return ResumeCheckpoint(last, best, completed)


def restore_trial_checkpoint(
    trainer: Trainer, source: ResumeCheckpoint, *, logger: Any = None,
) -> bool:
    """Restore into a new trial directory; return whether training is finished."""
    destination = Path(trainer.cfg.checkpoint_dir)
    if source.last_path.parent.resolve() == destination.resolve():
        raise ValueError("Checkpoint recovery must use a new trial directory")
    trainer.load_checkpoint(source.last_path, load_optimizer=True, load_scheduler=True)
    destination.mkdir(parents=True, exist_ok=True)
    if source.best_path is not None:
        shutil.copy2(source.best_path, destination / "best.pt")
        shutil.copy2(source.best_path, _best_snapshot(destination, int(trainer.state.best_epoch)))
    if logger is not None and hasattr(logger, "epoch"):
        cadence = int(trainer.cfg.validate_every_epochs or 1)
        logger.epoch = (source.completed_epochs // cadence) * cadence
    finished = bool(
        trainer.state.should_stop or source.completed_epochs >= trainer.cfg.max_epochs
        or (trainer.cfg.max_steps is not None and trainer.state.global_step >= trainer.cfg.max_steps)
    )
    # Trainer saves the COMPLETED epoch index, whereas fit starts at state.epoch.
    # Keep completed-state accounting intact when no further epoch should run.
    if not finished:
        trainer.state.epoch = source.completed_epochs
    return finished
