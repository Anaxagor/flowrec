надо как тут from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Configuration for Direct Preference Optimization on semantic item IDs."""

    name: str = "dpo"
    beta: float = 0.1
    num_negatives: int = 1
    negative_sampler: str = "random"
    reference_model_strategy: str = "frozen_init"
    reference_model_checkpoint: str | None = None
    loss_reduction: str = "mean"
    sft_weight: float = 0.0
    generation_eval_every_n_steps: int | None = None

    metric_prefix: str = "dpo"
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
            raise ValueError("DPOConfig.beta must be positive.")
        if self.num_negatives < 1:
            raise ValueError("DPOConfig.num_negatives must be >= 1.")
        if self.negative_sampler not in {"random"}:
            raise ValueError("DPOConfig.negative_sampler must be 'random'.")
        if self.reference_model_strategy not in {"frozen_init", "checkpoint"}:
            raise ValueError(
                "DPOConfig.reference_model_strategy must be 'frozen_init' or 'checkpoint'."
            )
        if self.reference_model_strategy == "checkpoint" and self.reference_model_checkpoint is None:
            raise ValueError(
                "DPOConfig.reference_model_checkpoint is required when "
                "reference_model_strategy='checkpoint'."
            )
        if self.loss_reduction not in {"mean", "sum", "none"}:
            raise ValueError("DPOConfig.loss_reduction must be 'mean', 'sum', or 'none'.")
        if self.sft_weight < 0:
            raise ValueError("DPOConfig.sft_weight must be >= 0.")
        if self.generation_eval_every_n_steps is not None and self.generation_eval_every_n_steps < 1:
            raise ValueError("DPOConfig.generation_eval_every_n_steps must be >= 1 when set.")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DPOConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        if "generation_topk" in normalized and isinstance(normalized["generation_topk"], list):
            normalized["generation_topk"] = tuple(int(x) for x in normalized["generation_topk"])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "beta": self.beta,
            "num_negatives": self.num_negatives,
            "negative_sampler": self.negative_sampler,
            "reference_model_strategy": self.reference_model_strategy,
            "reference_model_checkpoint": self.reference_model_checkpoint,
            "loss_reduction": self.loss_reduction,
            "sft_weight": self.sft_weight,
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