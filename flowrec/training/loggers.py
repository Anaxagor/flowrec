from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch


LOGGER = logging.getLogger("flowrec.training.loggers")

ExperimentLogger = Callable[[dict[str, Any]], None]


class CompositeExperimentLogger:
    """Fan out trainer events to multiple experiment loggers."""

    def __init__(self, loggers: Iterable[ExperimentLogger]) -> None:
        self.loggers = list(loggers)

    def __call__(self, payload: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger(payload)

    def close(self) -> None:
        for logger in self.loggers:
            close = getattr(logger, "close", None)
            if callable(close):
                close()

    def log_asset(self, path: str | Path, *, file_name: str | None = None, step: int | None = None) -> None:
        for logger in self.loggers:
            log_asset = getattr(logger, "log_asset", None)
            if callable(log_asset):
                log_asset(path, file_name=file_name, step=step)


class JsonlExperimentLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, payload: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
        event_type = payload.get("type", "metric")
        brief = {k: v for k, v in payload.items() if k != "type" and isinstance(v, (int, float, str))}
        LOGGER.info("%s %s", event_type, brief)


class CometExperimentLogger:
    """Log FlowRec trainer events to Comet ML.

    The Comet SDK is imported lazily so the repository remains usable without
    comet_ml installed unless a run explicitly enables Comet logging.
    """

    _METADATA_KEYS = {
        "type",
        "epoch",
        "batch_idx",
        "global_step",
        "optimizer_step",
        "best_checkpoint",
        "last_checkpoint",
    }

    def __init__(
        self,
        *,
        experiment: Any,
        log_artifacts: bool = True,
    ) -> None:
        self.experiment = experiment
        self.log_artifacts = log_artifacts
        self._closed = False

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        run_name: str,
        run_dir: Path,
        config_path: Path,
        resolved_config: Mapping[str, Any],
        cli_args: Mapping[str, Any],
    ) -> "CometExperimentLogger | None":
        if not bool(cfg.get("enabled", False)):
            return None

        try:
            import comet_ml
        except ImportError as exc:
            raise RuntimeError(
                "Comet logging is enabled, but comet_ml is not installed. "
                "Install it with `pip install comet_ml` or set comet.enabled=false."
            ) from exc

        experiment_name = str(cfg.get("experiment_name") or run_name)
        tags = [str(tag) for tag in cfg.get("tags", [])]
        experiment_config = comet_ml.ExperimentConfig(
            name=experiment_name,
            tags=tags or None,
            disabled=_optional_bool(cfg.get("disabled")),
            log_code=_optional_bool(cfg.get("log_code")),
            log_graph=_optional_bool(cfg.get("log_graph")),
            parse_args=_optional_bool(cfg.get("parse_args")),
            display_summary_level=int(cfg.get("display_summary_level", 1)),
            log_git_metadata=_optional_bool(cfg.get("log_git_metadata")),
            log_git_patch=_optional_bool(cfg.get("log_git_patch")),
            log_env_details=_optional_bool(cfg.get("log_env_details")),
            log_env_gpu=_optional_bool(cfg.get("log_env_gpu")),
            log_env_host=_optional_bool(cfg.get("log_env_host")),
            log_env_cpu=_optional_bool(cfg.get("log_env_cpu")),
            log_env_network=_optional_bool(cfg.get("log_env_network")),
            log_env_disk=_optional_bool(cfg.get("log_env_disk")),
            auto_output_logging=cfg.get("auto_output_logging", "simple"),
            auto_param_logging=bool(cfg.get("auto_param_logging", False)),
            auto_metric_logging=bool(cfg.get("auto_metric_logging", False)),
            auto_log_co2=bool(cfg.get("auto_log_co2", False)),
            offline_directory=(
                str(_resolve_relative_path(cfg["offline_directory"], run_dir=run_dir))
                if cfg.get("offline_directory") is not None
                else None
            ),
            distributed_node_identifier=cfg.get("distributed_node_identifier"),
        )

        experiment = comet_ml.start(
            api_key=cfg.get("api_key"),
            workspace=cfg.get("workspace"),
            project_name=cfg.get("project_name"),
            experiment_key=cfg.get("experiment_key"),
            mode=cfg.get("mode", "create"),
            online=cfg.get("online"),
            source="flowrec",
            experiment_config=experiment_config,
        )

        logger = cls(experiment=experiment, log_artifacts=bool(cfg.get("log_artifacts", True)))
        logger.log_metadata(
            run_dir=run_dir,
            config_path=config_path,
            resolved_config=resolved_config,
            cli_args=cli_args,
        )
        return logger

    def __call__(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type", "metric"))
        step = _optional_int(payload.get("global_step"))
        epoch = _optional_int(payload.get("epoch"))

        metrics = _scalar_metrics(payload, exclude=self._METADATA_KEYS)
        if metrics:
            self.experiment.log_metrics(metrics, step=step, epoch=epoch)

        if event_type == "epoch_end":
            epoch_metrics = _epoch_metrics(payload)
            if epoch_metrics:
                self.experiment.log_metrics(epoch_metrics, step=step, epoch=epoch)

    def log_metadata(
        self,
        *,
        run_dir: Path,
        config_path: Path,
        resolved_config: Mapping[str, Any],
        cli_args: Mapping[str, Any],
    ) -> None:
        params = flatten_parameters(
            {
                "config": resolved_config,
                "cli": cli_args,
            }
        )
        if params:
            self.experiment.log_parameters(params)

        self.experiment.log_other("flowrec_run_dir", str(run_dir))
        self.experiment.log_other("flowrec_config_path", str(config_path))

        if self.log_artifacts:
            self.log_asset(config_path, file_name=f"source_config/{config_path.name}")

    def log_asset(self, path: str | Path, *, file_name: str | None = None, step: int | None = None) -> None:
        if not self.log_artifacts:
            return
        path = Path(path)
        if not path.exists() or not path.is_file():
            return
        self.experiment.log_asset(
            file_data=str(path),
            file_name=file_name or path.name,
            step=step,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.experiment.end()
        self._closed = True


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_safe_markdown_summary(
    path: Path, summary: Mapping[str, Any], *, model_name: str
) -> None:
    """Write scientific results without machine- or run-specific paths."""
    known_datasets = {
        "beauty", "instruments", "beauty_2023", "yambda_50m_listens",
        "yambda_500m_listens",
    }
    dataset = summary.get("dataset")
    candidates = [summary.get("dataset_name")]
    if isinstance(dataset, Mapping):
        candidates.append(dataset.get("name"))
    dataset_name = next(
        (value for value in candidates if isinstance(value, str) and value in known_datasets),
        None,
    )

    def safe_number(value: Any) -> int | float | None:
        value = to_jsonable(value)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def safe_metrics(group: str) -> str:
        metrics = to_jsonable(summary.get(group) or {})
        if not isinstance(metrics, Mapping):
            return "{}"
        result: dict[str, int | float] = {}
        for name, value in metrics.items():
            # Metric names are identifiers such as val/sft/ndcg@10, never paths.
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_/@.-]{0,127}", name):
                continue
            if ".." in name or "//" in name or re.search(
                r"(?i)(?:^|[/_])(?:slurm|job|pid|account|user_id|username|project_id|hostname|ip_address|password|secret|api_key)(?:$|[/_0-9])",
                name,
            ):
                continue
            number = safe_number(value)
            if number is not None:
                result[name] = number
        return json.dumps(result, indent=2, sort_keys=True)

    lines = [f"# {model_name} Run Summary", ""]
    if dataset_name is not None:
        lines.append(f"- Dataset: {dataset_name}")
    for key, label in (
        ("seed", "Seed"),
        ("epochs_completed", "Epochs completed"),
        ("steps_completed", "Steps completed"),
        ("best_metric", "Best metric"),
    ):
        number = safe_number(summary.get(key))
        if number is not None:
            lines.append(f"- {label}: {number}")
    lines.extend([
        "",
        "Run and checkpoint details are available in final_summary.json.",
        "",
        "## Validation Metrics",
        safe_metrics("validation_metrics"),
        "",
        "## Test Metrics",
        safe_metrics("test_metrics"),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def flatten_parameters(data: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in to_jsonable(data).items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_parameters(value, prefix=name))
        elif _is_scalar_parameter(value):
            out[name] = value
        elif value is not None:
            out[name] = json.dumps(value, sort_keys=True)
    return out


def _scalar_metrics(payload: Mapping[str, Any], *, exclude: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        if key in exclude:
            continue
        scalar = _to_finite_float(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out


def _epoch_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in ("train", "val", "test"):
        values = payload.get(group)
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            scalar = _to_finite_float(value)
            if scalar is not None:
                out[f"epoch/{key}"] = scalar
    return out


def _to_finite_float(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _is_scalar_parameter(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_relative_path(value: str | Path, *, run_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()
