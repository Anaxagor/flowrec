from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from flowrec.ft_methods.gflow.flow import LogZContext
from flowrec.rewards.canonical import CanonicalRewardSpec
from flowrec.rewards.rating import RewardTransform


GFlowObjective = Literal["tb", "db", "subtb"]
GFlowEnvironment = Literal["acyclic", "cyclic"]
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
    environment: GFlowEnvironment = "acyclic"
    metric_prefix: str = "gflow"

    cyclic_flow_ratio: float = 0.25
    cyclic_num_trajectories: int = 32
    cyclic_max_steps: int = 64
    cyclic_delete_prob_cap: float = 0.49
    cyclic_subtb_estimator: Literal["exact", "mc"] = "exact"
    cyclic_subtb_num_samples: int = 32

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
    reward_pipeline: Literal["legacy", "canonical_v1"] = "legacy"
    reward_type: Literal["sasrec", "mix"] = "sasrec"
    sasrec_adapter: Literal["softplus"] = "softplus"
    component_normalization: Literal["row_max"] = "row_max"
    mix_weights: tuple[tuple[str, float], ...] = (
        ("sasrec", 1.0),
        ("rating", 1.0),
        ("similarity", 1.0),
    )
    tb_logz_source: TBLogZSource = "learned"
    tb_loss_weighting: TBLossWeighting = "uniform"
    tb_catalog_sampling: Literal["legacy", "hybrid", "uniform"] = "legacy"
    tb_head_sample_size: int = 0
    tb_tail_sample_size: int = 0
    tb_candidate_weights_normalize: bool = True
    db_score_chunk_size: int = 32
    db_activation_checkpointing: bool = True
    db_reuse_sft_encoder: bool = False
    tb_reuse_sft_encoder: bool = False
    candidate_sample_size: int = 0
    candidate_sample_top_k: int = 256
    candidate_sample_strategy: CandidateSampleStrategy = "exact"
    subtb_lambda: float = 1.0
    subtb_max_length: int | None = None

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
    generated_constrain_valid_sequences: bool = False
    generated_exclude_seen_items: bool = False

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
    generation_constrain_valid_sequences: bool = False
    generation_exclude_seen_items: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GFlowConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        pipeline = normalized.get("reward_pipeline", "legacy")
        if pipeline not in {"legacy", "canonical_v1"}:
            raise ValueError("reward_pipeline must be 'legacy' or 'canonical_v1'.")
        if pipeline == "canonical_v1":
            canonical = CanonicalRewardSpec.from_mapping(normalized)
            normalized.update(
                reward_type=canonical.reward_type,
                sasrec_adapter=canonical.sasrec_adapter,
                component_normalization=canonical.component_normalization,
                mix_weights=canonical.mix_weights,
            )
        elif "mix_weights" in normalized and isinstance(normalized["mix_weights"], dict):
            normalized["mix_weights"] = tuple(
                (str(name), float(weight)) for name, weight in normalized["mix_weights"].items()
            )
        if normalized.get("environment", "acyclic") not in {"acyclic", "cyclic"}:
            raise ValueError("environment must be 'acyclic' or 'cyclic'.")
        if normalized.get("cyclic_subtb_estimator", "exact") not in {"exact", "mc"}:
            raise ValueError("cyclic_subtb_estimator must be 'exact' or 'mc'.")
        if normalized.pop("db_flow_source", "exact") != "exact":
            raise ValueError("Only analytical exact prefix flows are supported.")
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
            "environment": self.environment,
            "metric_prefix": self.metric_prefix,
            "cyclic_flow_ratio": self.cyclic_flow_ratio,
            "cyclic_num_trajectories": self.cyclic_num_trajectories,
            "cyclic_max_steps": self.cyclic_max_steps,
            "cyclic_delete_prob_cap": self.cyclic_delete_prob_cap,
            "cyclic_subtb_estimator": self.cyclic_subtb_estimator,
            "cyclic_subtb_num_samples": self.cyclic_subtb_num_samples,
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
            "reward_pipeline": self.reward_pipeline,
            "reward_type": self.reward_type,
            "sasrec_adapter": self.sasrec_adapter,
            "component_normalization": self.component_normalization,
            "mix_weights": dict(self.mix_weights),
            "tb_logz_source": self.tb_logz_source,
            "tb_loss_weighting": self.tb_loss_weighting,
            "tb_catalog_sampling": self.tb_catalog_sampling,
            "tb_head_sample_size": self.tb_head_sample_size,
            "tb_tail_sample_size": self.tb_tail_sample_size,
            "tb_candidate_weights_normalize": self.tb_candidate_weights_normalize,
            "db_score_chunk_size": self.db_score_chunk_size,
            "db_activation_checkpointing": self.db_activation_checkpointing,
            "db_reuse_sft_encoder": self.db_reuse_sft_encoder,
            "tb_reuse_sft_encoder": self.tb_reuse_sft_encoder,
            "candidate_sample_size": self.candidate_sample_size,
            "candidate_sample_top_k": self.candidate_sample_top_k,
            "candidate_sample_strategy": self.candidate_sample_strategy,
            "subtb_lambda": self.subtb_lambda,
            "subtb_max_length": self.subtb_max_length,
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
            "generated_constrain_valid_sequences": self.generated_constrain_valid_sequences,
            "generated_exclude_seen_items": self.generated_exclude_seen_items,
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
            "generation_constrain_valid_sequences": self.generation_constrain_valid_sequences,
            "generation_exclude_seen_items": self.generation_exclude_seen_items,
        }

    def canonical_reward_manifest(self) -> dict[str, object] | None:
        if self.reward_pipeline != "canonical_v1":
            return None
        return CanonicalRewardSpec(
            reward_type=self.reward_type,
            sasrec_adapter=self.sasrec_adapter,
            component_normalization=self.component_normalization,
            mix_weights=self.mix_weights,
            reward_epsilon=self.reward_epsilon,
        ).to_manifest()
