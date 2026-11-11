from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

GFlowObjective = Literal["tb", "db", "subtb", "prefix_ce", "q_distill"]
TBLossWeighting = Literal["uniform", "reward"]
CandidateSamplerName = Literal["random", "sasrec", "policy"]
CandidatePoolStrategy = Literal["full_catalog", "random_pool"]
GFLOW_REWARD_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class GFlowConfig:
    """Configuration for GFlowGR fine-tuning.

    The initial implementation follows the paper's trajectory-balance setup:
    SFT on the observed positive item plus a GFlowNet loss over sampled item
    semantic-ID trajectories.
    """

    objective: GFlowObjective = "tb"

    sft_weight: float = 1.0
    gflow_weight: float = 1.0

    reward_cache_path: str | None = None
    reward_cache_power: float = 1.0
    tb_loss_weighting: TBLossWeighting = "uniform"
    subtb_lambda: float = 1.0
    subtb_max_length: int | None = None

    candidate_sampler: CandidateSamplerName = "random"
    num_negative_samples: int = 1
    num_random_negatives: int = 0
    easy_negative_steps: int = 0
    candidate_sampling_temperature: float = 1.0
    candidate_pool_size: int = 1024
    score_chunk_size: int = 50_000
    pool_strategy: CandidatePoolStrategy = "random_pool"
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GFlowConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        if str(normalized.pop("name", "gflow")).lower() != "gflow":
            raise ValueError("GFlowConfig requires method.name='gflow'.")
        if normalized.pop("db_flow_source", "exact") != "exact":
            raise ValueError("Only analytical exact prefix flows are supported.")
        if normalized.get("candidate_sampler", "random") not in {"random", "sasrec", "policy"}:
            raise ValueError("candidate_sampler must be 'random', 'sasrec' or 'policy'.")
        if normalized.get("tb_loss_weighting", "uniform") not in {"uniform", "reward"}:
            raise ValueError("tb_loss_weighting must be 'uniform' or 'reward'.")
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "sft_weight": self.sft_weight,
            "gflow_weight": self.gflow_weight,
            "reward_cache_path": self.reward_cache_path,
            "reward_cache_power": self.reward_cache_power,
            "tb_loss_weighting": self.tb_loss_weighting,
            "subtb_lambda": self.subtb_lambda,
            "subtb_max_length": self.subtb_max_length,
            "candidate_sampler": self.candidate_sampler,
            "num_negative_samples": self.num_negative_samples,
            "num_random_negatives": self.num_random_negatives,
            "easy_negative_steps": self.easy_negative_steps,
            "candidate_sampling_temperature": self.candidate_sampling_temperature,
            "candidate_pool_size": self.candidate_pool_size,
            "score_chunk_size": self.score_chunk_size,
            "pool_strategy": self.pool_strategy,
        }
