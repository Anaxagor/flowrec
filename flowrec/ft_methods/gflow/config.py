from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from flowrec.ft_methods.gflow.flow import LogZContext
from flowrec.rewards.rating import RewardTransform


GFlowObjective = Literal["tb", "db"]
CandidateSamplerName = Literal["random", "generated"]


@dataclass(frozen=True, slots=True)
class GFlowConfig:
    """Configuration for GFlowGR fine-tuning.

    The initial implementation follows the paper's trajectory-balance setup:
    SFT on the observed positive item plus a GFlowNet loss over sampled item
    semantic-ID trajectories.
    """

    name: str = "gflow"
    objective: GFlowObjective = "tb"
    metric_prefix: str = "gflow"

    sft_weight: float = 1.0
    gflow_weight: float = 1.0
    logz_init: float = 0.0
    logz_context: LogZContext = "semantic_history"
    logz_vocab_size: int | None = None
    logz_vector_size: int = 1
    logz_embedding_dim: int = 64
    logz_hidden_dim: int = 128
    logz_dropout: float = 0.0

    num_negative_samples: int = 1
    candidate_sampler: CandidateSamplerName = "random"
    include_positive: bool = True
    avoid_positive: bool = True
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True
    max_resample_attempts: int = 3
    generated_refresh_every_steps: int = 1
    generated_num_beams: int = 10
    generated_num_return_sequences: int | None = None
    generated_oversample_factor: int = 4
    generated_constrained: bool | None = True

    reward_key: str = "target_ratings"
    reward_epsilon: float = 1e-6
    negative_reward: float = 1e-3
    reward_transform: RewardTransform = "identity"

    compute_train_metrics: bool = False
    compute_eval_metrics: bool = True
    compute_token_metrics: bool = True
    evaluate_generation: bool = False
    generation_topk: tuple[int, ...] = (5, 10)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrained: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GFlowConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        if "generation_topk" in normalized and isinstance(normalized["generation_topk"], list):
            normalized["generation_topk"] = tuple(int(x) for x in normalized["generation_topk"])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "metric_prefix": self.metric_prefix,
            "sft_weight": self.sft_weight,
            "gflow_weight": self.gflow_weight,
            "logz_init": self.logz_init,
            "logz_context": self.logz_context,
            "logz_vocab_size": self.logz_vocab_size,
            "logz_vector_size": self.logz_vector_size,
            "logz_embedding_dim": self.logz_embedding_dim,
            "logz_hidden_dim": self.logz_hidden_dim,
            "logz_dropout": self.logz_dropout,
            "num_negative_samples": self.num_negative_samples,
            "candidate_sampler": self.candidate_sampler,
            "include_positive": self.include_positive,
            "avoid_positive": self.avoid_positive,
            "pad_token_id": self.pad_token_id,
            "semantic_ids_are_fixed_length": self.semantic_ids_are_fixed_length,
            "max_resample_attempts": self.max_resample_attempts,
            "generated_refresh_every_steps": self.generated_refresh_every_steps,
            "generated_num_beams": self.generated_num_beams,
            "generated_num_return_sequences": self.generated_num_return_sequences,
            "generated_oversample_factor": self.generated_oversample_factor,
            "generated_constrained": self.generated_constrained,
            "reward_key": self.reward_key,
            "reward_epsilon": self.reward_epsilon,
            "negative_reward": self.negative_reward,
            "reward_transform": self.reward_transform,
            "compute_train_metrics": self.compute_train_metrics,
            "compute_eval_metrics": self.compute_eval_metrics,
            "compute_token_metrics": self.compute_token_metrics,
            "evaluate_generation": self.evaluate_generation,
            "generation_topk": list(self.generation_topk),
            "generation_num_beams": self.generation_num_beams,
            "generation_num_return_sequences": self.generation_num_return_sequences,
            "generation_constrained": self.generation_constrained,
        }
