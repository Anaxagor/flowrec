from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


LossReduction = Literal["mean", "sum"]


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization."""

    name: str = "grpo"
    metric_prefix: str = "grpo"

    sft_weight: float = 1.0
    beta: float = 0.05
    clip_epsilon: float = 0.2
    loss_reduction: LossReduction = "mean"

    reward_epsilon: float = 1e-6
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-6

    num_rollouts: int = 4
    rollout_do_sample: bool = True
    rollout_num_beams: int | None = None
    rollout_num_return_sequences: int | None = None
    rollout_oversample_factor: int = 4
    rollout_temperature: float = 1.0
    rollout_top_k: int | None = 50
    rollout_top_p: float | None = None
    rollout_constrain_valid_sequences: bool = True
    rollout_exclude_seen_items: bool = False
    rollout_deduplicate: bool = True

    reference_model_checkpoint: str | None = None

    compute_train_metrics: bool = False
    compute_eval_metrics: bool = True
    compute_token_metrics: bool = True
    evaluate_generation: bool = False
    generation_topk: tuple[int, ...] = (5, 10)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrain_valid_sequences: bool = False
    generation_exclude_seen_items: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GRPOConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        if "generation_topk" in normalized and isinstance(normalized["generation_topk"], list):
            normalized["generation_topk"] = tuple(int(x) for x in normalized["generation_topk"])
        for key, default in (
            ("rollout_constrain_valid_sequences", True),
            ("rollout_exclude_seen_items", False),
            ("rollout_deduplicate", True),
            ("rollout_do_sample", True),
            ("generation_constrain_valid_sequences", False),
            ("generation_exclude_seen_items", False),
            ("normalize_advantages", True),
        ):
            if normalized.get(key) is None:
                normalized[key] = default
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_prefix": self.metric_prefix,
            "sft_weight": self.sft_weight,
            "beta": self.beta,
            "clip_epsilon": self.clip_epsilon,
            "loss_reduction": self.loss_reduction,
            "reward_epsilon": self.reward_epsilon,
            "normalize_advantages": self.normalize_advantages,
            "advantage_epsilon": self.advantage_epsilon,
            "num_rollouts": self.num_rollouts,
            "rollout_do_sample": self.rollout_do_sample,
            "rollout_num_beams": self.rollout_num_beams,
            "rollout_num_return_sequences": self.rollout_num_return_sequences,
            "rollout_oversample_factor": self.rollout_oversample_factor,
            "rollout_temperature": self.rollout_temperature,
            "rollout_top_k": self.rollout_top_k,
            "rollout_top_p": self.rollout_top_p,
            "rollout_constrain_valid_sequences": self.rollout_constrain_valid_sequences,
            "rollout_exclude_seen_items": self.rollout_exclude_seen_items,
            "rollout_deduplicate": self.rollout_deduplicate,
            "reference_model_checkpoint": self.reference_model_checkpoint,
            "compute_train_metrics": self.compute_train_metrics,
            "compute_eval_metrics": self.compute_eval_metrics,
            "compute_token_metrics": self.compute_token_metrics,
            "evaluate_generation": self.evaluate_generation,
            "generation_topk": list(self.generation_topk),
            "generation_num_beams": self.generation_num_beams,
            "generation_num_return_sequences": self.generation_num_return_sequences,
            "generation_constrain_valid_sequences": self.generation_constrain_valid_sequences,
            "generation_exclude_seen_items": self.generation_exclude_seen_items,
        }
