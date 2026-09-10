from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal


LossReduction = Literal["mean", "sum"]


@dataclass(frozen=True, slots=True)
class MiniOneRecConfig:
    """Configuration for GRPO-style mini-one-rec fine-tuning."""

    name: str = "mini-one-rec"
    metric_prefix: str = "grpo"

    sft_weight: float = 1.0
    grpo_weight: float = 1.0
    beta: float = 0.05
    clip_epsilon: float = 0.2
    loss_reduction: LossReduction = "mean"

    reward_epsilon: float = 1e-6
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-6
    reward_type: str = "rule_rank"
    rank_reward_weight: float = 0.1
    normalize_rank_reward: bool = False
    force_ground_truth_in_group: bool = False

    num_rollouts: int = 4
    rollout_do_sample: bool = True
    rollout_num_beams: int | None = None
    rollout_num_return_sequences: int | None = None
    rollout_oversample_factor: int = 4
    rollout_max_new_tokens: int | None = None
    rollout_temperature: float = 1.0
    rollout_top_k: int | None = 50
    rollout_top_p: float | None = None
    rollout_constrain_valid_sequences: bool = True
    rollout_exclude_seen_items: bool = False
    rollout_deduplicate: bool = True

    reference_model_checkpoint: str | None = None
    sync_ref_every_steps: int | None = None

    compute_train_metrics: bool = True
    compute_eval_metrics: bool = True
    compute_token_metrics: bool = True
    evaluate_generation: bool = True
    generation_topk: tuple[int, ...] = (5, 10, 20)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrain_valid_sequences: bool = False
    generation_exclude_seen_items: bool = False

    def __post_init__(self) -> None:
        if self.sft_weight < 0:
            raise ValueError("MiniOneRecConfig.sft_weight must be >= 0.")
        if self.grpo_weight < 0:
            raise ValueError("MiniOneRecConfig.grpo_weight must be >= 0.")
        if self.beta < 0:
            raise ValueError("MiniOneRecConfig.beta must be >= 0.")
        if not (0.0 <= self.clip_epsilon):
            raise ValueError("MiniOneRecConfig.clip_epsilon must be >= 0.")
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("MiniOneRecConfig.loss_reduction must be 'mean' or 'sum'.")
        if self.reward_epsilon <= 0:
            raise ValueError("MiniOneRecConfig.reward_epsilon must be positive.")
        if self.advantage_epsilon <= 0:
            raise ValueError("MiniOneRecConfig.advantage_epsilon must be positive.")
        if self.rank_reward_weight < 0:
            raise ValueError("MiniOneRecConfig.rank_reward_weight must be >= 0.")
        if self.num_rollouts < 2:
            raise ValueError("MiniOneRecConfig.num_rollouts must be >= 2.")
        if self.rollout_num_beams is not None and self.rollout_num_beams < 1:
            raise ValueError("MiniOneRecConfig.rollout_num_beams must be >= 1 when set.")
        if self.rollout_num_return_sequences is not None and self.rollout_num_return_sequences < 1:
            raise ValueError(
                "MiniOneRecConfig.rollout_num_return_sequences must be >= 1 when set."
            )
        if self.rollout_oversample_factor < 1:
            raise ValueError("MiniOneRecConfig.rollout_oversample_factor must be >= 1.")
        if self.rollout_max_new_tokens is not None and self.rollout_max_new_tokens < 1:
            raise ValueError("MiniOneRecConfig.rollout_max_new_tokens must be >= 1 when set.")
        if self.rollout_temperature <= 0:
            raise ValueError("MiniOneRecConfig.rollout_temperature must be positive.")
        if self.rollout_top_k is not None and self.rollout_top_k < 1:
            raise ValueError("MiniOneRecConfig.rollout_top_k must be >= 1 when set.")
        if self.rollout_top_p is not None and not (0.0 < self.rollout_top_p <= 1.0):
            raise ValueError("MiniOneRecConfig.rollout_top_p must be in (0, 1].")
        if self.sync_ref_every_steps is not None and self.sync_ref_every_steps < 1:
            raise ValueError("MiniOneRecConfig.sync_ref_every_steps must be >= 1 when set.")
        if self.generation_num_beams is not None and self.generation_num_beams < 1:
            raise ValueError("MiniOneRecConfig.generation_num_beams must be >= 1 when set.")
        if (
            self.generation_num_return_sequences is not None
            and self.generation_num_return_sequences < 1
        ):
            raise ValueError(
                "MiniOneRecConfig.generation_num_return_sequences must be >= 1 when set."
            )

    @property
    def num_generations(self) -> int:
        return int(self.num_rollouts)

    @property
    def sample_num_beams(self) -> int:
        return int(self.rollout_num_beams or 1)

    @property
    def sample_max_new_tokens(self) -> int | None:
        return self.rollout_max_new_tokens

    @property
    def sample_temperature(self) -> float:
        return float(self.rollout_temperature)

    @property
    def sample_top_k(self) -> int | None:
        return self.rollout_top_k

    @property
    def sample_top_p(self) -> float | None:
        return self.rollout_top_p

    @property
    def sample_constrain_valid_sequences(self) -> bool:
        return bool(self.rollout_constrain_valid_sequences)

    @property
    def sample_exclude_seen_items(self) -> bool:
        return bool(self.rollout_exclude_seen_items)

    @property
    def normalize_rewards(self) -> bool:
        return bool(self.normalize_advantages)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MiniOneRecConfig":
        if data is None:
            return cls()
        normalized = dict(data)

        if "generation_topk" in normalized and isinstance(normalized["generation_topk"], list):
            normalized["generation_topk"] = tuple(int(x) for x in normalized["generation_topk"])

        alias_pairs = {
            "sft_loss_weight": "sft_weight",
            "grpo_loss_weight": "grpo_weight",
            "num_generations": "num_rollouts",
            "sample_num_beams": "rollout_num_beams",
            "sample_max_new_tokens": "rollout_max_new_tokens",
            "sample_temperature": "rollout_temperature",
            "sample_top_k": "rollout_top_k",
            "sample_top_p": "rollout_top_p",
            "sample_constrain_valid_sequences": "rollout_constrain_valid_sequences",
            "sample_exclude_seen_items": "rollout_exclude_seen_items",
            "normalize_rewards": "normalize_advantages",
        }
        for alias_key, canonical_key in alias_pairs.items():
            if normalized.get(canonical_key) is None and normalized.get(alias_key) is not None:
                normalized[canonical_key] = normalized[alias_key]

        for key, default in (
            ("rollout_constrain_valid_sequences", True),
            ("rollout_exclude_seen_items", False),
            ("rollout_deduplicate", True),
            ("rollout_do_sample", True),
            ("generation_constrain_valid_sequences", False),
            ("generation_exclude_seen_items", False),
            ("normalize_advantages", True),
            ("normalize_rank_reward", False),
            ("force_ground_truth_in_group", False),
        ):
            if normalized.get(key) is None:
                normalized[key] = default

        valid_keys = {field.name for field in fields(cls)}
        normalized = {key: value for key, value in normalized.items() if key in valid_keys}
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_prefix": self.metric_prefix,
            "sft_weight": self.sft_weight,
            "grpo_weight": self.grpo_weight,
            "beta": self.beta,
            "clip_epsilon": self.clip_epsilon,
            "loss_reduction": self.loss_reduction,
            "reward_epsilon": self.reward_epsilon,
            "normalize_advantages": self.normalize_advantages,
            "advantage_epsilon": self.advantage_epsilon,
            "reward_type": self.reward_type,
            "rank_reward_weight": self.rank_reward_weight,
            "normalize_rank_reward": self.normalize_rank_reward,
            "force_ground_truth_in_group": self.force_ground_truth_in_group,
            "num_rollouts": self.num_rollouts,
            "rollout_do_sample": self.rollout_do_sample,
            "rollout_num_beams": self.rollout_num_beams,
            "rollout_num_return_sequences": self.rollout_num_return_sequences,
            "rollout_oversample_factor": self.rollout_oversample_factor,
            "rollout_max_new_tokens": self.rollout_max_new_tokens,
            "rollout_temperature": self.rollout_temperature,
            "rollout_top_k": self.rollout_top_k,
            "rollout_top_p": self.rollout_top_p,
            "rollout_constrain_valid_sequences": self.rollout_constrain_valid_sequences,
            "rollout_exclude_seen_items": self.rollout_exclude_seen_items,
            "rollout_deduplicate": self.rollout_deduplicate,
            "reference_model_checkpoint": self.reference_model_checkpoint,
            "sync_ref_every_steps": self.sync_ref_every_steps,
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
