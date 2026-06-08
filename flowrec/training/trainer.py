from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, MutableMapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import Batch, FlowRecModel


DeviceLike = str | torch.device
MonitorMode = Literal["min", "max"]
SchedulerStepOn = Literal["none", "batch", "epoch", "metric"]


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Configuration for the generic FlowRec trainer.

    The trainer is deliberately method-agnostic. It does not know whether the
    loss is SFT, GFlowGR, DPO, or anything else. It only asks the method for a
    scalar loss and then performs the usual optimization steps.
    """

    max_epochs: int = 10

    # Optimization helpers.
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    skip_non_finite_loss: bool = False

    # Automatic mixed precision. AMP is enabled only on CUDA devices.
    use_amp: bool = False
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Scheduler stepping policy.
    scheduler_step_on: SchedulerStepOn = "none"

    # Logging.
    log_every_steps: int = 50
    progress_bar: bool = True

    # Evaluation.
    validate_every_epochs: int | None = 1
    validate_every_steps: int | None = None
    run_validation_at_start: bool = False

    # Checkpointing.
    checkpoint_dir: str | Path | None = None
    save_best: bool = True
    save_last: bool = True
    save_every_epochs: int | None = None

    # Early stopping.
    monitor: str = "val/loss"
    monitor_mode: MonitorMode = "min"
    early_stopping_patience: int | None = None
    min_delta: float = 0.0

    # If True, method metrics are prefixed again by stage, e.g.
    # train/sft/loss, val/sft/loss. If False, method metrics remain as-is and
    # only trainer-created metrics such as val/loss are stage-prefixed.
    prefix_method_metrics_with_stage: bool = True

    # Optional cap for debugging.
    max_steps: int | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrainerConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        if "checkpoint_dir" in normalized and normalized["checkpoint_dir"] is not None:
            normalized["checkpoint_dir"] = Path(normalized["checkpoint_dir"])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if out["checkpoint_dir"] is not None:
            out["checkpoint_dir"] = str(out["checkpoint_dir"])
        return out


@dataclass(slots=True)
class TrainerState:
    """Mutable state tracked during training."""

    epoch: int = 0
    global_step: int = 0
    optimizer_step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    should_stop: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FitResult:
    """Return object from ``Trainer.fit``."""

    state: TrainerState
    best_checkpoint_path: str | None = None
    last_checkpoint_path: str | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)


class MetricAverager:
    """Weighted mean aggregation for scalar metrics."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    def update(self, metrics: Mapping[str, Any], *, weight: int | float = 1) -> None:
        weight = float(weight)
        if weight <= 0:
            return
        for key, value in metrics.items():
            scalar = _to_float(value)
            if scalar is None or not math.isfinite(scalar):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + scalar * weight
            self._weights[key] = self._weights.get(key, 0.0) + weight

    def compute(self) -> dict[str, float]:
        return {
            key: self._sums[key] / max(self._weights[key], 1e-12)
            for key in sorted(self._sums)
        }


class GenerationCoverageAccumulator:
    """Dataset-level catalog coverage for semantic-ID generation metrics."""

    def __init__(self, model: FlowRecModel) -> None:
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if torch.is_tensor(item_semantic_ids) and item_semantic_ids.numel() > 0:
            rows = item_semantic_ids.detach().cpu().long().tolist()
            self._catalog = {tuple(int(x) for x in row) for row in rows}
        else:
            self._catalog = set()

        self._seen: dict[int, set[tuple[int, ...]]] = {}
        self._metric_prefix = getattr(model, "method_name", "sft")

    def update(self, extra: Mapping[str, Any]) -> None:
        if not self._catalog:
            return

        generated = extra.get("generated_token_ids")
        if not torch.is_tensor(generated):
            return

        if generated.ndim == 2:
            generated = generated.unsqueeze(1)
        if generated.ndim != 3:
            return

        topks = extra.get("generation_topk")
        if topks is None:
            topks = (generated.size(1),)
        topks = tuple(sorted({max(1, min(int(k), generated.size(1))) for k in topks}))
        if not topks:
            return

        metric_prefix = extra.get("generation_metric_prefix")
        if isinstance(metric_prefix, str) and metric_prefix:
            self._metric_prefix = metric_prefix

        generated_cpu = generated.detach().cpu().long()
        for k in topks:
            seen_k = self._seen.setdefault(k, set())
            rows = generated_cpu[:, :k, :].reshape(-1, generated_cpu.size(-1)).tolist()
            for row in rows:
                key = tuple(int(x) for x in row)
                if key in self._catalog:
                    seen_k.add(key)

    def compute(self) -> dict[str, float]:
        if not self._catalog:
            return {}
        denominator = float(len(self._catalog))
        prefix = self._metric_prefix.rstrip("/") or "sft"
        return {
            f"{prefix}/generation_coverage@{k}": len(seen) / denominator
            for k, seen in sorted(self._seen.items())
        }


class Trainer:
    """Generic FlowRec trainer.

    This class owns the *optimization loop*:

    1. move batch to device,
    2. ask the method for a loss,
    3. call ``backward()``,
    4. clip gradients if configured,
    5. call ``optimizer.step()``,
    6. evaluate and checkpoint.

    The trainer intentionally does **not** know how an SFT, GFlowGR, DPO, or RL
    loss is computed. That is delegated to ``FlowRecMethod``.
    """

    def __init__(
        self,
        *,
        model: FlowRecModel,
        method: FlowRecMethod,
        optimizer: Optimizer,
        cfg: TrainerConfig | None = None,
        device: DeviceLike | None = None,
        scheduler: Any | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.method = method
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg or TrainerConfig()
        self.device = torch.device(device) if device is not None else _default_device()
        self.logger = logger
        self.state = TrainerState()

        self.model.to(self.device)
        self.method.to(self.device)

        self._use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        self._amp_dtype = _resolve_amp_dtype(self.cfg.amp_dtype)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        if self.cfg.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        if self.cfg.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")

        if self.cfg.checkpoint_dir is not None:
            Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_optimizer_config(
        cls,
        *,
        model: FlowRecModel,
        method: FlowRecMethod,
        cfg: TrainerConfig | None = None,
        device: DeviceLike | None = None,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        scheduler: Any | None = None,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> "Trainer":
        """Convenience constructor using AdamW."""

        optimizer = torch.optim.AdamW(
            [p for p in list(model.parameters()) + list(method.parameters()) if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        if callable(scheduler):
            scheduler = scheduler(optimizer)
        return cls(
            model=model,
            method=method,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            scheduler=scheduler,
            logger=logger,
        )

    def fit(
        self,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
    ) -> FitResult:
        """Run full training."""

        best_ckpt: str | None = None
        last_ckpt: str | None = None
        last_train_metrics: dict[str, float] = {}
        last_val_metrics: dict[str, float] = {}

        if self.cfg.run_validation_at_start and val_loader is not None:
            last_val_metrics = self.evaluate(val_loader, stage="val")
            self._update_best_and_maybe_stop(last_val_metrics)

        for epoch in range(self.state.epoch, self.cfg.max_epochs):
            if self.cfg.max_steps is not None and self.state.global_step >= self.cfg.max_steps:
                break

            self.state.epoch = epoch
            epoch_start = time.time()

            last_train_metrics, step_val_metrics = self.train_one_epoch(
                train_loader, epoch=epoch, val_loader=val_loader
            )
            record: dict[str, Any] = {
                "epoch": epoch,
                "global_step": self.state.global_step,
                "optimizer_step": self.state.optimizer_step,
                "train": last_train_metrics,
            }
            if step_val_metrics:
                last_val_metrics = step_val_metrics
                record["val"] = last_val_metrics

            epoch_validate = (
                self.cfg.validate_every_steps is None  # epoch-mode is active
                and val_loader is not None
                and self.cfg.validate_every_epochs is not None
                and self.cfg.validate_every_epochs > 0
                and (epoch + 1) % self.cfg.validate_every_epochs == 0
            )
            if epoch_validate:
                last_val_metrics = self.evaluate(val_loader, stage="val")
                record["val"] = last_val_metrics
                improved = self._update_best_and_maybe_stop(last_val_metrics)
                if improved and self.cfg.save_best and self.cfg.checkpoint_dir is not None:
                    best_ckpt = self.save_checkpoint("best.pt")
                    record["best_checkpoint"] = best_ckpt
            else:
                improved = False
                if self.scheduler is not None and self.cfg.scheduler_step_on == "epoch":
                    self.scheduler.step()

            if (
                self.cfg.save_every_epochs is not None
                and self.cfg.checkpoint_dir is not None
                and (epoch + 1) % self.cfg.save_every_epochs == 0
            ):
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            if self.cfg.save_last and self.cfg.checkpoint_dir is not None:
                last_ckpt = self.save_checkpoint("last.pt")
                record["last_checkpoint"] = last_ckpt

            epoch_time_sec = time.time() - epoch_start
            record["epoch_time_sec"] = epoch_time_sec
            record["epoch_time_min"] = epoch_time_sec / 60.0
            self.state.history.append(record)
            self._log({"type": "epoch_end", **record})

            if self.state.should_stop:
                break

            if self.cfg.max_steps is not None and self.state.global_step >= self.cfg.max_steps:
                break

        return FitResult(
            state=self.state,
            best_checkpoint_path=best_ckpt,
            last_checkpoint_path=last_ckpt,
            train_metrics=last_train_metrics,
            val_metrics=last_val_metrics,
        )

    def train_one_epoch(
        self,
        train_loader: Iterable[Batch],
        *,
        epoch: int | None = None,
        val_loader: Iterable[Batch] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Train for one epoch.

        Returns (train_metrics, last_step_val_metrics).  The second element is
        non-empty only when validate_every_steps is set and at least one
        step-level validation ran during this epoch.
        """

        self.model.train()
        self.method.train()
        self.optimizer.zero_grad(set_to_none=True)

        metrics = MetricAverager()
        accum_steps = self.cfg.gradient_accumulation_steps
        micro_step_since_update = 0
        last_step_val_metrics: dict[str, float] = {}

        num_batches = _safe_len(train_loader)
        max_batches = self.cfg.max_train_batches

        for batch_idx, raw_batch in enumerate(train_loader):
            if self.cfg.max_steps is not None and self.state.global_step >= self.cfg.max_steps:
                break

            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = move_to_device(raw_batch, self.device)
            batch_size = infer_batch_size(batch)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                method_out = self.method.training_step(
                    self.model,
                    batch,
                    step=self.state.global_step,
                )
                self._validate_method_output(method_out)
                raw_loss = method_out.loss
                loss = raw_loss / accum_steps

            if not torch.isfinite(raw_loss.detach()):
                if self.cfg.skip_non_finite_loss:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._log(
                        {
                            "type": "non_finite_loss_skipped",
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": self.state.global_step,
                            "loss": float(raw_loss.detach().cpu().item()),
                        }
                    )
                    continue
                raise FloatingPointError(
                    f"Non-finite loss at step {self.state.global_step}: "
                    f"{float(raw_loss.detach().cpu().item())}"
                )

            if self._use_amp:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step_since_update += 1
            self.state.global_step += 1

            step_metrics = method_out.detached_metrics()
            step_metrics.setdefault("loss", float(raw_loss.detach().cpu().item()))
            step_metrics = self._stage_prefix_metrics(step_metrics, stage="train")
            metrics.update(step_metrics, weight=batch_size)

            should_step = micro_step_since_update >= accum_steps
            is_last_known_batch = num_batches is not None and (batch_idx + 1) >= num_batches
            if should_step or is_last_known_batch:
                self._optimizer_step()
                micro_step_since_update = 0

            if (
                self.cfg.validate_every_steps is not None
                and val_loader is not None
                and self.state.global_step > 0
                and self.state.global_step % self.cfg.validate_every_steps == 0
            ):
                step_val = self.evaluate(val_loader, stage="val")
                last_step_val_metrics = step_val
                improved = self._update_best_and_maybe_stop(step_val)
                if improved and self.cfg.save_best and self.cfg.checkpoint_dir is not None:
                    self.save_checkpoint("best.pt")
                self._log(
                    {
                        "type": "step_val",
                        "epoch": epoch,
                        "global_step": self.state.global_step,
                        **step_val,
                    }
                )
                self.model.train()
                self.method.train()
                if self.state.should_stop:
                    break

            if self.cfg.log_every_steps > 0 and self.state.global_step % self.cfg.log_every_steps == 0:
                log_metrics = metrics.compute()
                log_metrics.update(
                    {
                        "type": "train_step",
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": self.state.global_step,
                        "optimizer_step": self.state.optimizer_step,
                    }
                )
                self._log(log_metrics)

        # If the loader length is unknown and gradients remain accumulated, step
        # once at the end. For known DataLoaders this is usually handled above.
        if micro_step_since_update > 0:
            self._optimizer_step()

        return metrics.compute(), last_step_val_metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Batch],
        *,
        stage: Literal["val", "test"] = "val",
    ) -> dict[str, float]:
        """Evaluate using ``method.validation_step`` or ``method.test_step``."""

        if stage not in {"val", "test"}:
            raise ValueError("stage must be 'val' or 'test'")

        was_training = self.model.training
        was_method_training = self.method.training
        self.model.eval()
        self.method.eval()
        metrics = MetricAverager()
        coverage = GenerationCoverageAccumulator(self.model)

        max_batches = self.cfg.max_eval_batches

        try:
            for batch_idx, raw_batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch = move_to_device(raw_batch, self.device)
                batch_size = infer_batch_size(batch)

                if stage == "val":
                    method_out = self.method.validation_step(
                        self.model,
                        batch,
                        step=self.state.global_step,
                    )
                else:
                    method_out = self.method.test_step(
                        self.model,
                        batch,
                        step=self.state.global_step,
                    )

                self._validate_method_output(method_out)

                step_metrics = method_out.detached_metrics()
                step_metrics.setdefault(
                    "loss",
                    float(method_out.loss.detach().cpu().item()),
                )
                step_metrics = self._stage_prefix_metrics(step_metrics, stage=stage)
                metrics.update(step_metrics, weight=batch_size)
                coverage.update(method_out.extra)
        finally:
            self.model.train(was_training)
            self.method.train(was_method_training)

        out = metrics.compute()
        coverage_metrics = coverage.compute()
        if coverage_metrics:
            out.update(self._stage_prefix_metrics(coverage_metrics, stage=stage))
        self._log({"type": f"{stage}_end", **out, "global_step": self.state.global_step})
        return out

    def test(self, test_loader: Iterable[Batch]) -> dict[str, float]:
        return self.evaluate(test_loader, stage="test")

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            p
            for p in list(self.model.parameters()) + list(self.method.parameters())
            if p.requires_grad
        ]

    def _optimizer_step(self) -> None:
        if self._use_amp:
            if self.cfg.max_grad_norm is not None:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.cfg.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimizer_step += 1

        if self.scheduler is not None and self.cfg.scheduler_step_on == "batch":
            self.scheduler.step()

    def _update_best_and_maybe_stop(self, metrics: Mapping[str, float]) -> bool:
        monitor = self.cfg.monitor
        if monitor not in metrics:
            # Common fallback: if the monitor is val/loss but only val/sft/loss
            # exists, use that. This keeps the trainer usable with prefixed
            # method metrics.
            fallback_keys = [key for key in metrics if key.endswith("/loss")]
            if monitor == "val/loss" and fallback_keys:
                monitor = sorted(fallback_keys)[0]
            else:
                raise KeyError(
                    f"Monitor metric {self.cfg.monitor!r} not found in metrics: "
                    f"{sorted(metrics.keys())}"
                )

        current = float(metrics[monitor])
        best = self.state.best_metric

        if best is None:
            improved = True
        elif self.cfg.monitor_mode == "min":
            improved = current < best - self.cfg.min_delta
        else:
            improved = current > best + self.cfg.min_delta

        if improved:
            self.state.best_metric = current
            self.state.best_epoch = self.state.epoch
            self.state.bad_epochs = 0
        else:
            self.state.bad_epochs += 1

        if self.scheduler is not None:
            if self.cfg.scheduler_step_on == "epoch":
                self.scheduler.step()
            elif self.cfg.scheduler_step_on == "metric":
                self.scheduler.step(current)

        if (
            self.cfg.early_stopping_patience is not None
            and self.state.bad_epochs >= self.cfg.early_stopping_patience
        ):
            self.state.should_stop = True

        return improved

    def _stage_prefix_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        stage: str,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in metrics.items():
            if self.cfg.prefix_method_metrics_with_stage:
                out[f"{stage}/{key}"] = value
            else:
                out[key] = value
        # Ensure a simple monitor key exists.
        if f"{stage}/loss" not in out:
            # Prefer explicit method loss if present.
            for key, value in out.items():
                if key.endswith("/loss"):
                    out[f"{stage}/loss"] = value
                    break
        return out

    def _validate_method_output(self, output: MethodOutput) -> None:
        if not isinstance(output, MethodOutput):
            raise TypeError(f"Method must return MethodOutput, got {type(output)!r}")
        if output.loss.ndim != 0:
            raise ValueError(f"Method loss must be scalar, got shape {tuple(output.loss.shape)}")
        if not output.loss.requires_grad and self.model.training:
            # Validation/test steps run under no_grad(), so they may not require
            # gradients. Training losses should.
            raise RuntimeError("Training loss does not require gradients.")

    def _log(self, payload: Mapping[str, Any]) -> None:
        if self.logger is not None:
            self.logger(dict(payload))

    def save_checkpoint(self, filename: str | Path) -> str:
        if self.cfg.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be set before saving checkpoints")

        path = Path(filename)
        if not path.is_absolute():
            path = Path(self.cfg.checkpoint_dir) / path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "trainer_config": self.cfg.to_dict(),
            "trainer_state": trainer_state_to_dict(self.state),
            "model_type": self.model.model_type,
            "model_class": self.model.__class__.__name__,
            "model_config": self.model.config_dict(),
            "model_state_dict": self.model.state_dict(),
            "method_name": getattr(self.method, "method_name", self.method.__class__.__name__),
            "method_class": self.method.__class__.__name__,
            "method_state_dict": self.method.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None and hasattr(self.scheduler, "state_dict")
            else None,
            "amp_scaler_state_dict": self._scaler.state_dict() if self._use_amp else None,
        }
        torch.save(payload, path)

        # Human-readable sidecar for quick inspection.
        sidecar = path.with_suffix(path.suffix + ".json")
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trainer_config": self.cfg.to_dict(),
                    "trainer_state": trainer_state_to_dict(self.state),
                    "model_type": self.model.model_type,
                    "model_class": self.model.__class__.__name__,
                    "model_config": self.model.config_dict(),
                    "method_name": getattr(self.method, "method_name", self.method.__class__.__name__),
                    "method_class": self.method.__class__.__name__,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        return str(path)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        map_location: DeviceLike | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        map_location = map_location or self.device
        payload = torch.load(Path(path), map_location=map_location)

        if "model_state_dict" not in payload:
            raise KeyError("Checkpoint does not contain model_state_dict")

        self.model.load_state_dict(payload["model_state_dict"], strict=strict)

        if payload.get("method_state_dict") is not None:
            self.method.load_state_dict(payload["method_state_dict"], strict=strict)

        if load_optimizer and payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        if (
            load_scheduler
            and self.scheduler is not None
            and payload.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        if self._use_amp and payload.get("amp_scaler_state_dict") is not None:
            self._scaler.load_state_dict(payload["amp_scaler_state_dict"])

        if payload.get("trainer_state") is not None:
            self.state = trainer_state_from_dict(payload["trainer_state"])

        return payload


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to ``device``."""

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(val, device) for val in value)
    if isinstance(value, list):
        return [move_to_device(val, device) for val in value]
    return value


def infer_batch_size(batch: Any) -> int:
    """Infer batch size from the first tensor/list-like object in a batch."""

    if torch.is_tensor(batch):
        return int(batch.shape[0]) if batch.ndim > 0 else 1
    if isinstance(batch, Mapping):
        for value in batch.values():
            size = infer_batch_size(value)
            if size > 0:
                return size
    if isinstance(batch, (list, tuple)) and batch:
        first = batch[0]
        if torch.is_tensor(first):
            return len(batch)
        if isinstance(first, (str, int, float)):
            return len(batch)
    return 1


def trainer_state_to_dict(state: TrainerState) -> dict[str, Any]:
    return {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "optimizer_step": state.optimizer_step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "bad_epochs": state.bad_epochs,
        "should_stop": state.should_stop,
        "history": state.history,
    }


def trainer_state_from_dict(data: Mapping[str, Any]) -> TrainerState:
    return TrainerState(
        epoch=int(data.get("epoch", 0)),
        global_step=int(data.get("global_step", 0)),
        optimizer_step=int(data.get("optimizer_step", 0)),
        best_metric=data.get("best_metric"),
        best_epoch=data.get("best_epoch"),
        bad_epochs=int(data.get("bad_epochs", 0)),
        should_stop=bool(data.get("should_stop", False)),
        history=list(data.get("history", [])),
    )


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")


def _to_float(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_len(obj: Any) -> int | None:
    try:
        return len(obj)  # type: ignore[arg-type]
    except TypeError:
        return None
