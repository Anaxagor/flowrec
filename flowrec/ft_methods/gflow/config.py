from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

from flowrec.ft_methods.gflow.flow import LogZContext


GFlowObjective = Literal["tb", "db", "subtb"]
CandidateSamplerName = Literal["random", "generated"]
TBLogZSource = Literal["learned", "reward_cache"]
TBLossWeighting = Literal["uniform", "reward"]
RewardCacheTransform = Literal["none", "power", "exp", "softplus", "rank"]
RewardCacheScoreSource = Literal["inverse_softplus", "reward"]
CandidateSampleStrategy = Literal["exact", "reward_proportional", "stratified", "legacy"]


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

    exact_terminal_distribution: bool = False
    exact_distribution_loss_weight: float = 0.0
    exact_distribution_chunk_size: int = 256
    exact_distribution_score_chunk_size: int | None = None
    exact_distribution_max_items: int | None = None
    reward_cache_path: str | None = None
    reward_cache_top_m: int = 1024
    reward_cache_probability: Literal["top", "full"] = "full"
    reward_cache_transform: RewardCacheTransform = "none"
    reward_cache_power: float = 1.0
    reward_cache_temperature: float = 1.0
    reward_cache_rank_alpha: float = 1.0
    reward_cache_score_source: RewardCacheScoreSource = "inverse_softplus"
    tb_logz_source: TBLogZSource = "learned"
    tb_loss_weighting: TBLossWeighting = "uniform"
    tb_catalog_sampling: Literal["legacy", "hybrid", "uniform"] = "legacy"
    tb_head_sample_size: int = 0
    tb_tail_sample_size: int = 0
    tb_candidate_weights_normalize: bool = True
    db_score_chunk_size: int = 32
    db_activation_checkpointing: bool = True
    db_reuse_sft_encoder: bool = False
    candidate_sample_size: int = 0
    candidate_sample_top_k: int = 256
    candidate_sample_strategy: CandidateSampleStrategy = "exact"
    subtb_lambda: float = 1.0
    subtb_max_length: int | None = None

    num_negative_samples: int = 1
    candidate_sampler: CandidateSamplerName = "random"
    include_positive: bool = True
    # Built-in fallback sampler only. It has no item-id-to-semantic-row
    # mapping, so history exclusion is off by default; wire an external
    # SemanticIDCandidateSampler (scripts/run_tiger.py does) to exclude
    # history items from negative sampling.
    exclude_history: bool = False
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True
    generated_refresh_every_steps: int = 1
    generated_num_beams: int = 10
    generated_num_return_sequences: int | None = None
    generated_oversample_factor: int = 4
    generated_constrain_valid_sequences: bool = False
    generated_exclude_seen_items: bool = False

    # Default reward is the binary PositiveRewardModel: the positive target
    # gets positive_reward, everything else negative_reward. Data-driven
    # rewards (flowrec/rewards/interaction.py) are wired externally.
    positive_reward: float = 1.0
    reward_epsilon: float = 1e-6
    negative_reward: float = 1e-3

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
    def from_dict(cls, data: dict[str, Any] | None) -> "GFlowConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        if normalized.pop("db_flow_source", "exact") != "exact":
            raise ValueError("Only analytical exact prefix flows are supported.")
        removed = [
            key
            for key in ("reward_key", "reward_transform", "avoid_positive", "max_resample_attempts")
            if normalized.pop(key, None) is not None
        ]
        if removed:
            warnings.warn(
                f"GFlowConfig keys {removed} were removed and are ignored: the "
                "positive target is always excluded from negatives and the "
                "default reward is the binary PositiveRewardModel.",
                stacklevel=2,
            )
        candidate_sample_strategy = normalized.get("candidate_sample_strategy", "exact")
        if candidate_sample_strategy not in {
            "exact",
            "reward_proportional",
            "stratified",
            "legacy",
        }:
            raise ValueError("candidate_sample_strategy must be 'exact', 'reward_proportional', " "or 'stratified'.")
        if "generation_topk" in normalized and isinstance(normalized["generation_topk"], list):
            normalized["generation_topk"] = tuple(int(x) for x in normalized["generation_topk"])
        for key in (
            "generated_constrain_valid_sequences",
            "generated_exclude_seen_items",
            "generation_constrain_valid_sequences",
            "generation_exclude_seen_items",
        ):
            if normalized.get(key) is None:
                normalized[key] = False
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
            "exact_terminal_distribution": self.exact_terminal_distribution,
            "exact_distribution_loss_weight": self.exact_distribution_loss_weight,
            "exact_distribution_chunk_size": self.exact_distribution_chunk_size,
            "exact_distribution_score_chunk_size": self.exact_distribution_score_chunk_size,
            "exact_distribution_max_items": self.exact_distribution_max_items,
            "reward_cache_path": self.reward_cache_path,
            "reward_cache_top_m": self.reward_cache_top_m,
            "reward_cache_probability": self.reward_cache_probability,
            "reward_cache_transform": self.reward_cache_transform,
            "reward_cache_power": self.reward_cache_power,
            "reward_cache_temperature": self.reward_cache_temperature,
            "reward_cache_rank_alpha": self.reward_cache_rank_alpha,
            "reward_cache_score_source": self.reward_cache_score_source,
            "tb_logz_source": self.tb_logz_source,
            "tb_loss_weighting": self.tb_loss_weighting,
            "tb_catalog_sampling": self.tb_catalog_sampling,
            "tb_head_sample_size": self.tb_head_sample_size,
            "tb_tail_sample_size": self.tb_tail_sample_size,
            "tb_candidate_weights_normalize": self.tb_candidate_weights_normalize,
            "db_score_chunk_size": self.db_score_chunk_size,
            "db_activation_checkpointing": self.db_activation_checkpointing,
            "db_reuse_sft_encoder": self.db_reuse_sft_encoder,
            "candidate_sample_size": self.candidate_sample_size,
            "candidate_sample_top_k": self.candidate_sample_top_k,
            "candidate_sample_strategy": self.candidate_sample_strategy,
            "subtb_lambda": self.subtb_lambda,
            "subtb_max_length": self.subtb_max_length,
            "num_negative_samples": self.num_negative_samples,
            "candidate_sampler": self.candidate_sampler,
            "include_positive": self.include_positive,
            "exclude_history": self.exclude_history,
            "pad_token_id": self.pad_token_id,
            "semantic_ids_are_fixed_length": self.semantic_ids_are_fixed_length,
            "generated_refresh_every_steps": self.generated_refresh_every_steps,
            "generated_num_beams": self.generated_num_beams,
            "generated_num_return_sequences": self.generated_num_return_sequences,
            "generated_oversample_factor": self.generated_oversample_factor,
            "generated_constrain_valid_sequences": self.generated_constrain_valid_sequences,
            "generated_exclude_seen_items": self.generated_exclude_seen_items,
            "positive_reward": self.positive_reward,
            "reward_epsilon": self.reward_epsilon,
            "negative_reward": self.negative_reward,
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
