from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


def _as_positive_int(value: Any, *, field_name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return number


def _as_positive_float(value: Any, *, field_name: str) -> float:
    number = float(value)
    if not (number > 0):
        raise ValueError(f"{field_name} must be positive.")
    return number


def _normalize_calibrate_sft_weight(data: dict[str, Any]) -> dict[str, Any]:
    """Accept a bool or a nested calibration block.

    Calibrate ``sft_weight`` against unweighted SDPO by gradient-norm ratio
    (the default)::

        {"calibrate_sft_weight": true}
        {"calibrate_sft_weight": {"num_batches": 8, "target_ratio": 1.0, "seed": 42}}

    Fixed weight instead::

        {"sft_weight": 1.0, "calibrate_sft_weight": false}
    """
    normalized = dict(data)
    raw = normalized.pop("calibrate_sft_weight", True)
    if raw is None or raw is False:
        normalized["calibrate_sft_weight"] = False
        return normalized
    if raw is True:
        normalized["calibrate_sft_weight"] = True
        return normalized
    if not isinstance(raw, dict):
        raise ValueError(
            "calibrate_sft_weight must be a bool or a dict with keys "
            "num_batches, target_ratio, seed."
        )
    unknown = sorted(set(raw) - {"num_batches", "target_ratio", "seed"})
    if unknown:
        raise ValueError(
            "Unknown calibrate_sft_weight keys: " + ", ".join(unknown)
        )
    normalized["calibrate_sft_weight"] = True
    if "num_batches" in raw:
        normalized["calibration_num_batches"] = raw["num_batches"]
    if "target_ratio" in raw:
        normalized["calibration_target_ratio"] = raw["target_ratio"]
    if "seed" in raw:
        normalized["calibration_seed"] = raw["seed"]
    return normalized


@dataclass(frozen=True, slots=True)
class SDPOConfig:
    """Configuration for Softmax-DPO / S-DPO on semantic item IDs."""

    name: str = "sdpo"
    beta: float = 0.1
    num_negatives: int = 3
    negative_sampler: str = "random"
    curriculum_steps: int = 0
    sampling_temperature: float = 1.0
    exclude_history_in_sampling: bool = False
    reference_model_strategy: str = "frozen_init"
    reference_model_checkpoint: str | None = None
    loss_reduction: str = "mean"
    sft_weight: float = 0.0
    # Default is gradient-norm calibration of sft_weight against unweighted
    # SDPO. Mutually exclusive with a non-zero sft_weight in the same config.
    calibrate_sft_weight: bool = True
    calibration_num_batches: int = 8
    calibration_target_ratio: float = 1.0
    calibration_seed: int | None = None
    generation_eval_every_n_steps: int | None = None

    metric_prefix: str = "sdpo"
    compute_train_metrics: bool = True
    compute_eval_metrics: bool = True
    compute_token_metrics: bool = True
    evaluate_generation: bool = True
    generation_topk: tuple[int, ...] = (5, 10, 20)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrain_valid_sequences: bool | None = None

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("SDPOConfig.beta must be positive.")
        if self.num_negatives < 1:
            raise ValueError("SDPOConfig.num_negatives must be >= 1.")
        if self.negative_sampler not in {"random", "curriculum"}:
            raise ValueError("SDPOConfig.negative_sampler must be 'random' or 'curriculum'.")
        if self.negative_sampler == "curriculum" and self.curriculum_steps < 1:
            raise ValueError("SDPOConfig.curriculum_steps must be >= 1 for curriculum sampling.")
        if self.sampling_temperature <= 0:
            raise ValueError("SDPOConfig.sampling_temperature must be positive.")
        if self.reference_model_strategy not in {"frozen_init", "checkpoint"}:
            raise ValueError(
                "SDPOConfig.reference_model_strategy must be 'frozen_init' or 'checkpoint'."
            )
        if self.reference_model_strategy == "checkpoint" and self.reference_model_checkpoint is None:
            raise ValueError(
                "SDPOConfig.reference_model_checkpoint is required when "
                "reference_model_strategy='checkpoint'."
            )
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("SDPOConfig.loss_reduction must be 'mean' or 'sum'.")
        if self.sft_weight < 0:
            raise ValueError("SDPOConfig.sft_weight must be >= 0.")
        if self.calibrate_sft_weight and self.sft_weight != 0.0:
            raise ValueError(
                "Set either method.sft_weight or method.calibrate_sft_weight, not both."
            )
        if self.calibration_num_batches < 1:
            raise ValueError("SDPOConfig.calibration_num_batches must be >= 1.")
        if self.calibration_target_ratio <= 0:
            raise ValueError("SDPOConfig.calibration_target_ratio must be positive.")
        if self.generation_eval_every_n_steps is not None and self.generation_eval_every_n_steps < 1:
            raise ValueError("SDPOConfig.generation_eval_every_n_steps must be >= 1 when set.")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SDPOConfig":
        if data is None:
            return cls()
        normalized = _normalize_calibrate_sft_weight(data)
        if "generation_topk" in normalized and isinstance(normalized["generation_topk"], list):
            normalized["generation_topk"] = tuple(int(x) for x in normalized["generation_topk"])
        if "calibration_num_batches" in normalized:
            normalized["calibration_num_batches"] = _as_positive_int(
                normalized["calibration_num_batches"],
                field_name="calibration_num_batches",
            )
        if "calibration_target_ratio" in normalized:
            normalized["calibration_target_ratio"] = _as_positive_float(
                normalized["calibration_target_ratio"],
                field_name="calibration_target_ratio",
            )
        if "calibration_seed" in normalized and normalized["calibration_seed"] is not None:
            normalized["calibration_seed"] = int(normalized["calibration_seed"])

        valid_keys = {field.name for field in fields(cls)}
        unknown = sorted(set(normalized) - valid_keys)
        if unknown:
            raise ValueError("Unknown SDPO configuration keys: " + ", ".join(unknown))
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "beta": self.beta,
            "num_negatives": self.num_negatives,
            "negative_sampler": self.negative_sampler,
            "curriculum_steps": self.curriculum_steps,
            "sampling_temperature": self.sampling_temperature,
            "exclude_history_in_sampling": self.exclude_history_in_sampling,
            "reference_model_strategy": self.reference_model_strategy,
            "reference_model_checkpoint": self.reference_model_checkpoint,
            "loss_reduction": self.loss_reduction,
            "sft_weight": self.sft_weight,
            "calibrate_sft_weight": self.calibrate_sft_weight,
            "calibration_num_batches": self.calibration_num_batches,
            "calibration_target_ratio": self.calibration_target_ratio,
            "calibration_seed": self.calibration_seed,
            "generation_eval_every_n_steps": self.generation_eval_every_n_steps,
            "metric_prefix": self.metric_prefix,
            "compute_train_metrics": self.compute_train_metrics,
            "compute_eval_metrics": self.compute_eval_metrics,
            "compute_token_metrics": self.compute_token_metrics,
            "evaluate_generation": self.evaluate_generation,
            "generation_topk": list(self.generation_topk),
            "generation_num_beams": self.generation_num_beams,
            "generation_num_return_sequences": self.generation_num_return_sequences,
            "generation_constrain_valid_sequences": self.generation_constrain_valid_sequences,
        }
