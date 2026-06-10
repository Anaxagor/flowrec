from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization on semantic item IDs.

    This implementation uses one observed positive item as the high-reward completion and
    random semantic-ID negatives as the low-reward completions. Rewards are normalized inside
    each user group, so no value model is required.
    """

    name: str = "grpo"

    # KL coefficient against the frozen reference model.
    beta: float = 0.04

    # PPO/GRPO ratio clipping.
    clip_epsilon: float = 0.2

    # Group construction: [chosen positive + num_negatives sampled negatives].
    num_negatives: int = 4
    negative_sampler: str = "random"

    # Simple offline reward: positive target gets reward_positive, negatives get reward_negative.
    reward_positive: float = 1.0
    reward_negative: float = 0.0
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-6

    # Frozen policy used both as reference and as the old policy denominator.
    reference_model_strategy: str = "frozen_init"
    reference_model_checkpoint: str | None = None

    loss_reduction: str = "mean"
    sft_weight: float = 0.0
    generation_eval_every_n_steps: int | None = None

    metric_prefix: str = "grpo"
    compute_train_metrics: bool = True
    compute_eval_metrics: bool = True
    compute_token_metrics: bool = True
    evaluate_generation: bool = True
    generation_topk: tuple[int, ...] = (5, 10, 20)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrained: bool | None = None

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError("GRPOConfig.beta must be non-negative.")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("GRPOConfig.clip_epsilon must be in (0, 1).")
        if self.num_negatives < 1:
            raise ValueError("GRPOConfig.num_negatives must be >= 1.")
        if self.negative_sampler not in {"random"}:
            raise ValueError("GRPOConfig.negative_sampler must be 'random'.")
        if self.advantage_epsilon <= 0:
            raise ValueError("GRPOConfig.advantage_epsilon must be positive.")
        if self.reference_model_strategy not in {"frozen_init", "checkpoint"}:
            raise ValueError(
                "GRPOConfig.reference_model_strategy must be 'frozen_init' or 'checkpoint'."
            )
        if self.reference_model_strategy == "checkpoint" and self.reference_model_checkpoint is None:
            raise ValueError(
                "GRPOConfig.reference_model_checkpoint is required when "
                "reference_model_strategy='checkpoint'."
            )
        if self.loss_reduction not in {"mean", "sum", "none"}:
            raise ValueError("GRPOConfig.loss_reduction must be 'mean', 'sum', or 'none'.")
        if self.sft_weight < 0:
            raise ValueError("GRPOConfig.sft_weight must be >= 0.")
        if self.generation_eval_every_n_steps is not None and self.generation_eval_every_n_steps < 1:
            raise ValueError("GRPOConfig.generation_eval_every_n_steps must be >= 1 when set.")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GRPOConfig":
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
            "clip_epsilon": self.clip_epsilon,
            "num_negatives": self.num_negatives,
            "negative_sampler": self.negative_sampler,
            "reward_positive": self.reward_positive,
            "reward_negative": self.reward_negative,
            "normalize_advantages": self.normalize_advantages,
            "advantage_epsilon": self.advantage_epsilon,
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
            "generation_constrained": self.generation_constrained,
        }
