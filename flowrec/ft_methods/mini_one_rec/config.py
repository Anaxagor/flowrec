from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal


LossReduction = Literal["mean", "sum"]
RolloutStrategy = Literal["beam", "sample", "dynamic"]

_REWARD_ALIASES = {
    "rule_rank": "rule_rank", "none": "rule_rank", "null": "rule_rank", "internal": "rule_rank",
    "rating": "binary", "ratings": "binary", "binary": "binary", "positive": "binary",
    "item_prior": "item_prior", "rating_prior": "item_prior",
    "similarity": "similarity", "semantic_similarity": "similarity",
    "sasrec": "sasrec", "svd": "svd",
    "interaction": "interaction", "interaction_log": "interaction", "amazon_log": "interaction",
    "amazonlog": "interaction", "yambda_log": "interaction",
    "mix": "mix", "mixed": "mix", "mixture": "mix", "sum": "mix",
    "multiply": "multiply", "multiplicative": "multiply", "product": "multiply",
}


def normalize_reward_type(value: str) -> str:
    name = str(value).lower()
    if name not in _REWARD_ALIASES:
        choices = sorted(set(_REWARD_ALIASES.values()))
        raise ValueError(f"Unknown MiniOneRec reward {value!r}; expected one of {choices}.")
    return _REWARD_ALIASES[name]


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

    Fixed weight (default 1)::

        {"sft_weight": 1.0}

    Calibrate ``sft_weight`` against unweighted GRPO by gradient-norm ratio::

        {"calibrate_sft_weight": true}
        {"calibrate_sft_weight": {"num_batches": 8, "target_ratio": 1.0, "seed": 42}}
    """
    normalized = dict(data)
    raw = normalized.pop("calibrate_sft_weight", False)
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
class MiniOneRecConfig:
    """Source-faithful GRPO/ReRe configuration for TIGER fine-tuning.

    Defaults follow the attached MiniOneRec/ReRe implementations where the
    papers provide explicit settings: constrained beam search, G=16, ReRe
    beta=1e-3 and no extra target-SFT term in the RL objective.
    """

    name: str = "mini-one-rec"
    metric_prefix: str = "grpo"

    # Optional auxiliary next-SID SFT regularizer. Default is a fixed weight of
    # 1. Set calibrate_sft_weight to fill sft_weight from ||g_sft|| / ||g_grpo||
    # instead. Mutually exclusive with a non-zero sft_weight in the same config.
    sft_weight: float = 1.0
    calibrate_sft_weight: bool = False
    calibration_num_batches: int = 8
    calibration_target_ratio: float = 1.0
    calibration_seed: int | None = None
    grpo_weight: float = 1.0

    # GRPO / ReRe objective.
    beta: float = 1e-3
    clip_epsilon: float = 0.1
    loss_reduction: LossReduction = "mean"
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-6

    # Rule+rank is the source baseline; external modes reuse GFlow rewards.
    reward_type: str = "rule_rank"
    reward_power: float = 1.0
    rank_reward_weight: float = 1.0
    normalize_rank_reward: bool = True

    # Kept only for external reward-model plumbing in run_tiger.py. It is NOT
    # used to clamp the signed rule+rank reward.
    reward_epsilon: float = 1e-6

    # Rollout generation.
    num_rollouts: int = 16
    rollout_strategy: RolloutStrategy = "beam"
    rollout_num_beams: int | None = None
    rollout_max_new_tokens: int | None = None
    rollout_temperature: float = 1.0
    rollout_top_k: int | None = None
    rollout_top_p: float | None = None
    rollout_constrain_valid_sequences: bool = True
    rollout_exclude_seen_items: bool = False
    dynamic_oversample_ratio: float = 1.5

    # TIGER-specific positive augmentation. With exact-match SID rewards the
    # ground-truth item can be absent from all G beams, in which case the
    # binary rule reward provides no positive supervision. When enabled, keep
    # the generated group if it already contains the target; otherwise replace
    # the lowest-scoring generated candidate by the target before computing
    # behavior-policy log-probabilities and rewards. This is a deliberate
    # positive-augmented adaptation of GRPO rather than a strictly on-policy
    # rollout distribution.
    force_ground_truth_in_group: bool = False

    # Frozen KL reference. A checkpoint is optional; otherwise initialization
    # is copied from the policy at the start of RL.
    reference_model_checkpoint: str | None = None

    # SFT/generation metrics used by the existing trainer during validation.
    compute_train_metrics: bool = True
    compute_eval_metrics: bool = True
    compute_token_metrics: bool = True
    evaluate_generation: bool = True
    generation_topk: tuple[int, ...] = (5, 10, 20)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrain_valid_sequences: bool = True
    generation_exclude_seen_items: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_type", normalize_reward_type(self.reward_type))
        if not math.isfinite(self.reward_power) or self.reward_power <= 0:
            raise ValueError("MiniOneRecConfig.reward_power must be finite and positive.")
        if self.reward_type == "rule_rank" and self.reward_power != 1.0:
            raise ValueError("reward_power applies only to external positive rewards.")
        if self.sft_weight < 0:
            raise ValueError("MiniOneRecConfig.sft_weight must be >= 0.")
        if self.calibrate_sft_weight and self.sft_weight != 0.0:
            raise ValueError(
                "Set either method.sft_weight or method.calibrate_sft_weight, not both."
            )
        if self.calibration_num_batches < 1:
            raise ValueError("MiniOneRecConfig.calibration_num_batches must be >= 1.")
        if self.calibration_target_ratio <= 0:
            raise ValueError("MiniOneRecConfig.calibration_target_ratio must be positive.")
        if self.grpo_weight < 0:
            raise ValueError("MiniOneRecConfig.grpo_weight must be >= 0.")
        if self.beta < 0:
            raise ValueError("MiniOneRecConfig.beta must be >= 0.")
        if self.clip_epsilon < 0:
            raise ValueError("MiniOneRecConfig.clip_epsilon must be >= 0.")
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("MiniOneRecConfig.loss_reduction must be 'mean' or 'sum'.")
        if self.advantage_epsilon <= 0:
            raise ValueError("MiniOneRecConfig.advantage_epsilon must be positive.")
        if self.rank_reward_weight < 0:
            raise ValueError("MiniOneRecConfig.rank_reward_weight must be >= 0.")
        if self.reward_type == "rule_rank" and not self.normalize_rank_reward:
            raise ValueError(
                "The attached MiniOneRec/ReRe rank reward is normalized across the group; "
                "normalize_rank_reward=False is not source-faithful."
            )
        if self.reward_epsilon <= 0:
            raise ValueError("MiniOneRecConfig.reward_epsilon must be positive.")
        if self.num_rollouts < 2:
            raise ValueError("MiniOneRecConfig.num_rollouts must be >= 2.")
        if self.rollout_strategy not in {"beam", "sample", "dynamic"}:
            raise ValueError("rollout_strategy must be 'beam', 'sample', or 'dynamic'.")
        if self.rollout_num_beams is not None and self.rollout_num_beams < 1:
            raise ValueError("rollout_num_beams must be >= 1 when set.")
        if self.rollout_max_new_tokens is not None and self.rollout_max_new_tokens < 1:
            raise ValueError("rollout_max_new_tokens must be >= 1 when set.")
        if self.rollout_temperature <= 0:
            raise ValueError("rollout_temperature must be positive.")
        if self.rollout_top_k is not None and self.rollout_top_k < 1:
            raise ValueError("rollout_top_k must be >= 1 when set.")
        if self.rollout_top_p is not None and not (0.0 < self.rollout_top_p <= 1.0):
            raise ValueError("rollout_top_p must be in (0, 1].")
        if self.dynamic_oversample_ratio < 1.0:
            raise ValueError("dynamic_oversample_ratio must be >= 1.")
        if self.generation_num_beams is not None and self.generation_num_beams < 1:
            raise ValueError("generation_num_beams must be >= 1 when set.")
        if self.generation_num_return_sequences is not None and self.generation_num_return_sequences < 1:
            raise ValueError("generation_num_return_sequences must be >= 1 when set.")

    @property
    def num_generations(self) -> int:
        return int(self.num_rollouts)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MiniOneRecConfig":
        if data is None:
            return cls()
        normalized = _normalize_calibrate_sft_weight(data)

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
            if canonical_key not in normalized and alias_key in normalized:
                normalized[canonical_key] = normalized.pop(alias_key)
            else:
                normalized.pop(alias_key, None)

        if normalized.get("calibrate_sft_weight"):
            if "sft_weight" in normalized and float(normalized["sft_weight"]) != 0.0:
                raise ValueError(
                    "Set either method.sft_weight or method.calibrate_sft_weight, not both."
                )
            normalized["sft_weight"] = 0.0
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

        # Legacy sampling flags become an explicit strategy. An explicit
        # rollout_strategy always wins.
        explicit_strategy = "rollout_strategy" in normalized
        legacy_do_sample = normalized.pop("do_sample", None)
        if legacy_do_sample is None:
            legacy_do_sample = normalized.pop("rollout_do_sample", None)
        if not explicit_strategy and legacy_do_sample is not None:
            normalized["rollout_strategy"] = "sample" if bool(legacy_do_sample) else "beam"

        # Old implementation fields that were either dead or non-source-faithful.
        # Null sync is harmless; non-null hard syncing is rejected.
        sync_ref = normalized.pop("sync_ref_every_steps", None)
        if sync_ref is not None:
            raise ValueError(
                "sync_ref_every_steps is not used: periodic hard reference copies do not match "
                "the attached MiniOneRec/ReRe descriptions."
            )
        normalized.pop("rollout_num_return_sequences", None)
        normalized.pop("rollout_deduplicate", None)
        normalized.pop("rollout_oversample_factor", None)

        valid_keys = {field.name for field in fields(cls)}
        unknown = sorted(set(normalized) - valid_keys)
        if unknown:
            raise ValueError(
                "Unknown MiniOneRec configuration keys (previous code silently ignored these): "
                + ", ".join(unknown)
            )
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_prefix": self.metric_prefix,
            "sft_weight": self.sft_weight,
            "calibrate_sft_weight": self.calibrate_sft_weight,
            "calibration_num_batches": self.calibration_num_batches,
            "calibration_target_ratio": self.calibration_target_ratio,
            "calibration_seed": self.calibration_seed,
            "grpo_weight": self.grpo_weight,
            "beta": self.beta,
            "clip_epsilon": self.clip_epsilon,
            "loss_reduction": self.loss_reduction,
            "normalize_advantages": self.normalize_advantages,
            "advantage_epsilon": self.advantage_epsilon,
            "reward_type": self.reward_type,
            "reward_power": self.reward_power,
            "rank_reward_weight": self.rank_reward_weight,
            "normalize_rank_reward": self.normalize_rank_reward,
            "reward_epsilon": self.reward_epsilon,
            "num_rollouts": self.num_rollouts,
            "rollout_strategy": self.rollout_strategy,
            "rollout_num_beams": self.rollout_num_beams,
            "rollout_max_new_tokens": self.rollout_max_new_tokens,
            "rollout_temperature": self.rollout_temperature,
            "rollout_top_k": self.rollout_top_k,
            "rollout_top_p": self.rollout_top_p,
            "rollout_constrain_valid_sequences": self.rollout_constrain_valid_sequences,
            "rollout_exclude_seen_items": self.rollout_exclude_seen_items,
            "dynamic_oversample_ratio": self.dynamic_oversample_ratio,
            "force_ground_truth_in_group": self.force_ground_truth_in_group,
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
