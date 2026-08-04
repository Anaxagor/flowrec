from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flowrec.candidates.generated import (
    GeneratedSemanticCandidateConfig,
    GeneratedSemanticIDCandidateSampler,
)
from flowrec.candidates.random import (
    RandomSemanticCandidateConfig,
    RandomSemanticIDCandidateSampler,
)
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.cyclic import (
    APPEND_ACTION,
    DELETE_ACTION,
    STOP_ACTION,
    CyclicSemanticTrajectoryBatch,
    analytical_delete_probabilities,
    sample_analytical_cyclic_trajectories,
)
from flowrec.ft_methods.gflow.exact_flows import (
    CatalogPrefixFlowIndex,
    reward_proportional_sample,
    stratified_catalog_sample,
)
from flowrec.ft_methods.gflow.flow import (
    GlobalLogZ,
    SemanticHistoryLogZ,
    SemanticHistoryLogZConfig,
    VectorLogZ,
)
from flowrec.ft_methods.gflow.losses.detailed_balance import detailed_balance_loss
from flowrec.ft_methods.gflow.losses.cyclic import (
    cyclic_detailed_balance_loss,
    cyclic_subtrajectory_balance_loss,
    cyclic_trajectory_balance_loss,
)
from flowrec.ft_methods.gflow.losses.subtrajectory_balance import (
    subtrajectory_balance_loss,
)
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.rewards.rating import RatingRewardConfig, RatingRewardModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class _PrecomputedRewardCache:
    def __init__(
        self,
        path: str | Path,
        *,
        top_m: int,
        probability: str,
        transform: str = "none",
        power: float = 1.0,
        temperature: float = 1.0,
        rank_alpha: float = 1.0,
        score_source: str = "inverse_softplus",
        load_full_catalog: bool = False,
        expected_canonical_reward: dict[str, object] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Reward cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_canonical_reward is not None:
            actual = self.manifest.get("canonical_reward")
            if actual != expected_canonical_reward:
                raise ValueError(
                    "Reward cache canonical specification does not match the active GFlow config: "
                    f"expected={expected_canonical_reward!r}, actual={actual!r}."
                )
        self.top_m = min(int(top_m), int(self.manifest.get("stored_top_m", top_m)))
        self.probability = str(probability).lower()
        if self.probability not in {"top", "full"}:
            raise ValueError("reward_cache_probability must be 'top' or 'full'.")
        self.transform = str(transform or "none").lower()
        if self.transform == "identity":
            self.transform = "none"
        if self.transform not in {"none", "power", "exp", "softplus", "rank"}:
            raise ValueError("reward_cache_transform must be one of none, power, exp, softplus, rank.")
        self.power = float(power)
        self.temperature = float(temperature)
        self.rank_alpha = float(rank_alpha)
        self.score_source = str(score_source or "inverse_softplus").lower()
        if self.score_source not in {"inverse_softplus", "reward"}:
            raise ValueError("reward_cache_score_source must be 'inverse_softplus' or 'reward'.")
        if self.transform != "none" and self.probability != "top":
            raise ValueError("Transformed reward-cache runs require reward_cache_probability='top'.")
        (
            self.rows,
            self.probs,
            self.rewards,
            self.reward_sum,
            self.mass,
            self.target_in_top,
            self.prefix_log_flows,
        ) = self._load_tensors()
        self.full_rewards: torch.Tensor | None = None
        self.full_reward_sum: torch.Tensor | None = None
        if load_full_catalog:
            self.full_rewards, self.full_reward_sum = self._load_full_catalog_tensors()
        if self.transform != "none":
            # Cached prefix flows are tied to the stored reward scale. If rewards
            # are transformed on load, fall back to exact recomputation.
            self.prefix_log_flows = None

    @property
    def num_examples(self) -> int:
        return int(self.rows.size(0))

    def select(
        self,
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._checked_indices(indices)
        rows = self.rows.index_select(0, idx).to(device=device, non_blocking=True).long()
        probs = self.probs.index_select(0, idx).to(device=device, non_blocking=True).float()
        mass = self.mass.index_select(0, idx).to(device=device, non_blocking=True).float()
        target_in_top = self.target_in_top.index_select(0, idx).to(device=device, non_blocking=True).float()
        return rows, probs, mass, target_in_top

    def select_tb(
        self,
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        idx = self._checked_indices(indices)
        rows = self.rows.index_select(0, idx).to(device=device, non_blocking=True).long()
        rewards = self.rewards.index_select(0, idx).to(device=device, non_blocking=True).float()
        reward_sum = self.reward_sum.index_select(0, idx).to(device=device, non_blocking=True).float()
        candidate_weights = self.probs.index_select(0, idx).to(device=device, non_blocking=True).float()
        mass = self.mass.index_select(0, idx).to(device=device, non_blocking=True).float()
        target_in_top = self.target_in_top.index_select(0, idx).to(device=device, non_blocking=True).float()
        prefix_log_flows = None
        if self.prefix_log_flows is not None:
            prefix_log_flows = self.prefix_log_flows.index_select(0, idx).to(device=device, non_blocking=True).float()
        return (
            rows,
            rewards,
            reward_sum,
            candidate_weights,
            mass,
            target_in_top,
            prefix_log_flows,
        )

    def select_full_catalog(
        self,
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.full_rewards is None or self.full_reward_sum is None:
            raise ValueError(
                "Full-catalogue sampling requires a cache built with --store-full-rewards "
                "and the matching --store-power-sums value."
            )
        idx = self._checked_indices(indices)
        rewards = self.full_rewards.index_select(0, idx).to(device=device, non_blocking=True)
        reward_sum = self.full_reward_sum.index_select(0, idx).to(device=device, non_blocking=True).float()
        return rewards, reward_sum

    def transform_candidate_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._transform_rewards(rewards).float()

    @staticmethod
    def _power_sum_key(power: float) -> str:
        slug = format(float(power), ".12g").replace("-", "m").replace(".", "p")
        return f"reward_sum_power_{slug}"

    def _checked_indices(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.detach().cpu().long()
        if idx.numel() == 0:
            raise ValueError("Empty example_indices tensor for reward cache lookup.")
        if int(idx.min()) < 0 or int(idx.max()) >= self.num_examples:
            raise IndexError(
                f"Reward cache index out of range: min={int(idx.min())} max={int(idx.max())} "
                f"num_examples={self.num_examples}"
            )
        return idx

    def _load_tensors(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        rows_parts: list[torch.Tensor] = []
        probs_parts: list[torch.Tensor] = []
        rewards_parts: list[torch.Tensor] = []
        reward_sum_parts: list[torch.Tensor] = []
        mass_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        prefix_flow_parts: list[torch.Tensor] = []
        has_prefix_flows: bool | None = None
        for shard in self.manifest.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.is_absolute():
                shard_path = self.path / shard_path
            payload = torch.load(shard_path, map_location="cpu")
            rows = payload["top_item_row_ids"][:, : self.top_m].long().contiguous()
            raw_rewards = self._raw_rewards(payload).float().contiguous()
            rewards = self._transform_rewards(raw_rewards).float().contiguous()
            probs = self._probabilities(payload, rewards).float().contiguous()
            mass = self._mass(payload, probs, rewards).float().contiguous()
            reward_sum = self._reward_sum(payload, rewards, mass).float().contiguous()
            target_in_top = self._target_in_top(payload).float().contiguous()
            prefix_log_flows = payload.get("top_prefix_log_flows")
            shard_has_prefix_flows = prefix_log_flows is not None
            if has_prefix_flows is None:
                has_prefix_flows = shard_has_prefix_flows
            elif bool(has_prefix_flows) != bool(shard_has_prefix_flows):
                raise ValueError("Reward cache mixes shards with and without top_prefix_log_flows.")
            if prefix_log_flows is not None:
                prefix_log_flows = prefix_log_flows[:, : self.top_m, :].float().contiguous()
                if prefix_log_flows.size(0) != rows.size(0) or prefix_log_flows.size(1) != rows.size(1):
                    raise ValueError("top_prefix_log_flows must align with top_item_row_ids.")
                prefix_flow_parts.append(prefix_log_flows)
            rows_parts.append(rows)
            probs_parts.append(probs)
            rewards_parts.append(rewards)
            reward_sum_parts.append(reward_sum)
            mass_parts.append(mass)
            target_parts.append(target_in_top)
        if not rows_parts:
            raise ValueError(f"Reward cache {self.path} has no shards.")
        return (
            torch.cat(rows_parts, dim=0),
            torch.cat(probs_parts, dim=0),
            torch.cat(rewards_parts, dim=0),
            torch.cat(reward_sum_parts, dim=0),
            torch.cat(mass_parts, dim=0),
            torch.cat(target_parts, dim=0),
            torch.cat(prefix_flow_parts, dim=0) if has_prefix_flows else None,
        )

    def _load_full_catalog_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards_parts: list[torch.Tensor] = []
        reward_sum_parts: list[torch.Tensor] = []
        if self.transform == "power":
            reward_sum_key = self._power_sum_key(self.power)
        elif self.transform == "none":
            reward_sum_key = "reward_sum"
        else:
            raise ValueError("Full-catalogue sampling currently supports reward_cache_transform " "'none' or 'power'.")
        for shard in self.manifest.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.is_absolute():
                shard_path = self.path / shard_path
            payload = torch.load(shard_path, map_location="cpu")
            if "full_rewards" not in payload:
                raise KeyError(f"Reward cache shard lacks full_rewards: {shard_path}")
            if reward_sum_key not in payload:
                raise KeyError(f"Reward cache shard lacks {reward_sum_key!r}: {shard_path}")
            rewards_parts.append(payload["full_rewards"].contiguous())
            reward_sum_parts.append(payload[reward_sum_key].reshape(-1).float().contiguous())
        if not rewards_parts:
            raise ValueError(f"Reward cache {self.path} has no full-catalogue shards.")
        return torch.cat(rewards_parts, dim=0), torch.cat(reward_sum_parts, dim=0)

    def _raw_rewards(self, payload: dict[str, Any]) -> torch.Tensor:
        if "top_rewards" not in payload:
            raise KeyError("Reward cache shard lacks top_rewards required for reward-cache trajectory balance.")
        return payload["top_rewards"][:, : self.top_m]

    def _transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.float().clamp_min(1e-6)
        if self.transform == "none":
            return rewards
        if self.transform == "power":
            if self.power <= 0.0:
                raise ValueError("reward_cache_power must be positive.")
            return rewards.pow(self.power).clamp_min(1e-6)
        if self.transform == "rank":
            if self.rank_alpha <= 0.0:
                raise ValueError("reward_cache_rank_alpha must be positive.")
            ranks = torch.arange(1, rewards.size(1) + 1, dtype=rewards.dtype, device=rewards.device).unsqueeze(0)
            return ranks.pow(-self.rank_alpha).expand_as(rewards).contiguous().clamp_min(1e-6)

        if self.temperature <= 0.0:
            raise ValueError("reward_cache_temperature must be positive.")
        scores = self._scores_from_rewards(rewards)
        if self.transform == "exp":
            scaled = (scores - scores.max(dim=1, keepdim=True).values) / self.temperature
            return torch.exp(scaled).clamp_min(1e-6)
        if self.transform == "softplus":
            return F.softplus(scores / self.temperature).clamp_min(1e-6)
        raise ValueError(f"Unsupported reward_cache_transform: {self.transform}")

    def _scores_from_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.score_source == "reward":
            return rewards
        return torch.where(rewards > 20.0, rewards, torch.log(torch.expm1(rewards).clamp_min(1e-12)))

    def _probabilities(self, payload: dict[str, Any], rewards: torch.Tensor) -> torch.Tensor:
        if self.transform != "none":
            return rewards / rewards.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if self.probability == "full":
            if "top_full_probs" not in payload:
                raise KeyError("Reward cache shard lacks top_full_probs required by reward_cache_probability='full'.")
            return payload["top_full_probs"][:, : self.top_m]
        key = "top_probs" if self.top_m == int(payload["top_item_row_ids"].size(1)) else f"top_probs_{self.top_m}"
        if key in payload:
            return payload[key][:, : self.top_m]
        return rewards / rewards.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _mass(self, payload: dict[str, Any], probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        if self.transform != "none":
            return torch.ones(probs.size(0), dtype=torch.float32)
        key = (
            "top_reward_mass"
            if self.top_m == int(payload["top_item_row_ids"].size(1))
            else f"top_reward_mass_{self.top_m}"
        )
        if key in payload:
            return payload[key].reshape(-1)
        if self.probability == "full":
            return probs.sum(dim=1)
        return torch.ones(probs.size(0), dtype=torch.float32)

    def _reward_sum(self, payload: dict[str, Any], rewards: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        if self.transform != "none" or self.probability == "top":
            return rewards.sum(dim=1)
        if "reward_sum" in payload:
            return payload["reward_sum"].reshape(-1)
        return rewards.sum(dim=1) / mass.clamp_min(1e-12)

    def _target_in_top(self, payload: dict[str, Any]) -> torch.Tensor:
        key = (
            "target_in_top_m"
            if self.top_m == int(payload["top_item_row_ids"].size(1))
            else f"target_in_top_{self.top_m}"
        )
        if key in payload:
            return payload[key].reshape(-1)
        target_rows = payload.get("target_semantic_row_ids")
        if target_rows is None:
            return torch.zeros(payload["top_item_row_ids"].size(0), dtype=torch.bool)
        return payload["top_item_row_ids"][:, : self.top_m].eq(target_rows.reshape(-1, 1)).any(dim=1)


class GFlowMethod(FlowRecMethod):
    """GFlowGR fine-tuning with rating rewards and trajectory balance.

    The method follows the paper's clean GR specialization:
      - SFT keeps the observed next-item supervision.
      - Candidate semantic-ID trajectories receive terminal rewards.
      - The TIGER model supplies forward token probabilities.
      - Backward probabilities are one for left-to-right generation.
    """

    method_name = "gflow"

    def __init__(
        self,
        cfg: GFlowConfig | None = None,
        *,
        candidate_sampler: RandomSemanticIDCandidateSampler | None = None,
        reward_model: RatingRewardModel | None = None,
        sft_method: SFTMethod | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GFlowConfig()
        if self.cfg.environment not in {"acyclic", "cyclic"}:
            raise ValueError("environment must be 'acyclic' or 'cyclic'.")
        if self.cfg.tb_logz_source == "reward_cache" and self.cfg.objective not in {
            "tb",
            "db",
            "subtb",
        }:
            raise ValueError("tb_logz_source='reward_cache' is only supported for objective='tb', 'db', or 'subtb'.")
        if self.cfg.objective in {"db", "subtb"} and self.cfg.tb_logz_source != "reward_cache":
            raise ValueError("Analytical DB/SubTB requires tb_logz_source='reward_cache'.")
        if self.cfg.tb_loss_weighting not in {"uniform", "reward"}:
            raise ValueError("tb_loss_weighting must be uniform or reward.")
        if self.cfg.candidate_sample_strategy not in {
            "exact",
            "reward_proportional",
            "stratified",
            "legacy",
        }:
            raise ValueError("candidate_sample_strategy must be exact, reward_proportional, " "or stratified.")
        if self.cfg.objective in {"db", "subtb"}:
            sample_size = int(self.cfg.candidate_sample_size)
            if sample_size < 0:
                raise ValueError("candidate_sample_size must be non-negative.")
            if int(self.cfg.candidate_sample_top_k) < 0:
                raise ValueError("candidate_sample_top_k must be non-negative.")
            if self.cfg.candidate_sample_strategy == "exact" and sample_size > 0:
                raise ValueError("Exact candidate evaluation requires candidate_sample_size=0.")
            if (
                self.cfg.candidate_sample_strategy
                in {
                    "reward_proportional",
                    "stratified",
                }
                and sample_size <= 0
            ):
                raise ValueError(
                    f"{self.cfg.candidate_sample_strategy} sampling requires " "candidate_sample_size > 0."
                )
            if (
                self.cfg.candidate_sample_strategy == "stratified"
                and int(self.cfg.candidate_sample_top_k) >= sample_size
            ):
                raise ValueError("Stratified sampling requires candidate_sample_top_k " "< candidate_sample_size.")
        if self.cfg.tb_catalog_sampling not in {"legacy", "hybrid", "uniform"}:
            raise ValueError("tb_catalog_sampling must be legacy, hybrid, or uniform.")
        if self.cfg.tb_catalog_sampling != "legacy":
            if self.cfg.objective != "tb" or self.cfg.tb_logz_source != "reward_cache":
                raise ValueError(
                    "Full-catalogue sampling requires objective='tb' and " "tb_logz_source='reward_cache'."
                )
            if self.cfg.tb_loss_weighting != "reward":
                raise ValueError("Full-catalogue sampling requires tb_loss_weighting='reward'.")
            if bool(self.cfg.tb_candidate_weights_normalize):
                raise ValueError("Importance contribution weights must not be renormalized.")
        if self.cfg.environment == "cyclic":
            if not self.cfg.reward_cache_path:
                raise ValueError("Cyclic GFlow requires a reward cache with full rewards.")
            if self.cfg.reward_cache_transform not in {"none", "power"}:
                raise ValueError("Cyclic GFlow supports reward_cache_transform='none' or 'power'.")
            if self.cfg.tb_logz_source != "reward_cache":
                raise ValueError("Cyclic GFlow requires tb_logz_source='reward_cache'.")
            if not self.cfg.semantic_ids_are_fixed_length:
                raise ValueError("Cyclic GFlow requires fixed-length semantic IDs.")
            if self.cfg.cyclic_flow_ratio <= 0 or self.cfg.cyclic_num_trajectories <= 0:
                raise ValueError("cyclic_flow_ratio and cyclic_num_trajectories must be positive.")
            if self.cfg.cyclic_max_steps <= 0 or self.cfg.cyclic_subtb_num_samples <= 0:
                raise ValueError("cyclic_max_steps and cyclic_subtb_num_samples must be positive.")
            internal, leaf = analytical_delete_probabilities(
                self.cfg.cyclic_flow_ratio,
                delete_probability_cap=self.cfg.cyclic_delete_prob_cap,
            )
            cap = float(self.cfg.cyclic_delete_prob_cap)
            targets = torch.tensor([internal / cap, leaf / cap]).clamp(1e-6, 1 - 1e-6)
            self.cyclic_delete_logits = torch.nn.Parameter(torch.logit(targets))
        self.log_z_model = (
            torch.nn.Identity()
            if self.cfg.tb_logz_source == "reward_cache" or self.cfg.objective in {"db", "subtb"}
            else self._build_log_z_model()
        )
        self._reward_cache: _PrecomputedRewardCache | None = None
        self._exact_flow_index: CatalogPrefixFlowIndex | None = None

        self.random_candidate_sampler = RandomSemanticIDCandidateSampler(
            RandomSemanticCandidateConfig(
                num_negative_samples=self.cfg.num_negative_samples,
                include_positive=self.cfg.include_positive,
                avoid_positive=self.cfg.avoid_positive,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids_are_fixed_length=self.cfg.semantic_ids_are_fixed_length,
                max_resample_attempts=self.cfg.max_resample_attempts,
            )
        )
        self.generated_candidate_sampler = GeneratedSemanticIDCandidateSampler(
            GeneratedSemanticCandidateConfig(
                num_negative_samples=self.cfg.num_negative_samples,
                include_positive=self.cfg.include_positive,
                num_beams=self.cfg.generated_num_beams,
                num_return_sequences=self.cfg.generated_num_return_sequences,
                oversample_factor=self.cfg.generated_oversample_factor,
                constrain_valid_sequences=self.cfg.generated_constrain_valid_sequences,
                exclude_seen_items=self.cfg.generated_exclude_seen_items,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids_are_fixed_length=self.cfg.semantic_ids_are_fixed_length,
                max_resample_attempts=self.cfg.max_resample_attempts,
            )
        )
        self.candidate_sampler = candidate_sampler
        self.reward_model = reward_model or RatingRewardModel(
            RatingRewardConfig(
                reward_key=self.cfg.reward_key,
                reward_epsilon=self.cfg.reward_epsilon,
                negative_reward=self.cfg.negative_reward,
                transform=self.cfg.reward_transform,
            )
        )
        self.sft_method = sft_method or SFTMethod(
            SFTConfig(
                compute_train_metrics=self.cfg.compute_train_metrics,
                compute_eval_metrics=self.cfg.compute_eval_metrics,
                compute_token_metrics=self.cfg.compute_token_metrics,
                compute_item_metrics=False,
                evaluate_generation=self.cfg.evaluate_generation,
                generation_topk=self.cfg.generation_topk,
                generation_num_beams=self.cfg.generation_num_beams,
                generation_num_return_sequences=self.cfg.generation_num_return_sequences,
                generation_constrain_valid_sequences=self.cfg.generation_constrain_valid_sequences,
                generation_exclude_seen_items=self.cfg.generation_exclude_seen_items,
            )
        )

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        sft_out = self.sft_method.compute_loss(model, batch, stage=stage, step=step)
        candidate_scoring_batch = self._candidate_scoring_batch(batch, sft_out.model_output)
        if self.cfg.environment == "cyclic":
            return self._compute_cyclic_loss(
                model,
                batch,
                candidate_scoring_batch,
                sft_out,
                stage=stage,
                step=step,
            )

        zero = sft_out.loss.new_zeros(())
        gflow_loss = zero
        gflow_metrics: dict[str, torch.Tensor] = {}
        trajectories = None
        rewards = None
        sequence_logprobs = None
        token_logprobs = None
        log_z = zero
        tb_candidate_weights = None

        use_reward_cache_tb = self.cfg.tb_logz_source == "reward_cache"
        compute_sampled_gflow = (
            not bool(self.cfg.exact_terminal_distribution) or float(self.cfg.gflow_weight) != 0.0
        ) and (not use_reward_cache_tb or stage == "train")
        if compute_sampled_gflow:
            if use_reward_cache_tb:
                (
                    trajectories,
                    rewards,
                    log_z,
                    tb_candidate_weights,
                    cache_tb_metrics,
                ) = self._precomputed_reward_cache_tb_inputs(model, batch)
            else:
                trajectories = self._sample_trajectories(model, batch, step=step)
                rewards = self.reward_model(batch, trajectories)
                log_z = self.log_z_model(batch) if self.cfg.objective == "tb" else zero
                cache_tb_metrics = {}

            if self.cfg.objective == "tb":
                sequence_logprobs = model.score_items(
                    candidate_scoring_batch,
                    candidate_token_ids=trajectories.candidate_token_ids,
                    candidate_token_mask=trajectories.candidate_token_mask,
                )
                if self.cfg.tb_loss_weighting == "reward" and tb_candidate_weights is None:
                    denom = rewards.sum(dim=1, keepdim=True).clamp_min(float(self.cfg.reward_epsilon))
                    tb_candidate_weights = rewards / denom
                gflow_loss, gflow_metrics = trajectory_balance_loss(
                    sequence_logprobs=sequence_logprobs,
                    rewards=rewards,
                    log_z=log_z,
                    reward_epsilon=self.cfg.reward_epsilon,
                    candidate_weights=(tb_candidate_weights if self.cfg.tb_loss_weighting == "reward" else None),
                    normalize_candidate_weights=bool(self.cfg.tb_candidate_weights_normalize),
                )
                gflow_metrics.update(cache_tb_metrics)
            elif self.cfg.objective == "db":
                token_logprobs, prefix_log_flows = self._candidate_token_logprobs_and_prefix_flows(
                    model,
                    candidate_scoring_batch,
                    trajectories,
                    rewards,
                )
                sequence_logprobs = (
                    token_logprobs
                    * trajectories.candidate_token_mask.to(token_logprobs.device).to(token_logprobs.dtype)
                ).sum(dim=-1)
                active = trajectories.candidate_token_mask.to(prefix_log_flows.device).bool()
                parent_log_flows = prefix_log_flows[..., :-1].masked_fill(~active, 0.0)
                child_log_flows = prefix_log_flows[..., 1:].masked_fill(~active, 0.0)
                gflow_loss, gflow_metrics = detailed_balance_loss(
                    token_logprobs=token_logprobs,
                    parent_log_flows=parent_log_flows,
                    child_log_flows=child_log_flows,
                    candidate_token_mask=trajectories.candidate_token_mask,
                    rewards=rewards,
                    reward_epsilon=self.cfg.reward_epsilon,
                    candidate_weights=tb_candidate_weights,
                )
                gflow_metrics.update(cache_tb_metrics)
            elif self.cfg.objective == "subtb":
                token_logprobs, prefix_log_flows = self._candidate_token_logprobs_and_prefix_flows(
                    model,
                    candidate_scoring_batch,
                    trajectories,
                    rewards,
                )
                sequence_logprobs = (
                    token_logprobs
                    * trajectories.candidate_token_mask.to(token_logprobs.device).to(token_logprobs.dtype)
                ).sum(dim=-1)
                subtb_candidate_weights = tb_candidate_weights
                if self.cfg.tb_loss_weighting == "uniform":
                    subtb_candidate_weights = torch.ones_like(rewards)
                gflow_loss, gflow_metrics = subtrajectory_balance_loss(
                    token_logprobs=token_logprobs,
                    prefix_log_flows=prefix_log_flows,
                    candidate_token_mask=trajectories.candidate_token_mask,
                    rewards=rewards,
                    reward_epsilon=self.cfg.reward_epsilon,
                    candidate_weights=subtb_candidate_weights,
                    subtb_lambda=self.cfg.subtb_lambda,
                    max_subtrajectory_length=self.cfg.subtb_max_length,
                )
                gflow_metrics.update(cache_tb_metrics)
            else:
                raise ValueError(f"Unknown GFlow objective: {self.cfg.objective}")
        elif use_reward_cache_tb:
            gflow_metrics = {
                "reward_cache_tb_active": torch.tensor(0.0, device=sft_out.loss.device),
                "fixed_log_z_active": torch.tensor(0.0, device=sft_out.loss.device),
            }

        exact_loss = zero
        exact_metrics: dict[str, torch.Tensor] = {}
        if bool(self.cfg.exact_terminal_distribution) and float(self.cfg.exact_distribution_loss_weight) != 0.0:
            if self.cfg.reward_cache_path:
                if stage == "train":
                    exact_loss, exact_metrics = self._precomputed_reward_cache_loss(model, batch)
                else:
                    exact_metrics = {
                        "reward_cache_active": torch.tensor(0.0, device=sft_out.loss.device),
                    }
            else:
                exact_loss, exact_metrics = self._exact_terminal_distribution_loss(model, batch)

        loss = (
            float(self.cfg.sft_weight) * sft_out.loss
            + float(self.cfg.gflow_weight) * gflow_loss
            + float(self.cfg.exact_distribution_loss_weight) * exact_loss
        )

        if trajectories is not None:
            metric_device = trajectories.candidate_token_ids.device
            num_candidates = trajectories.num_candidates
            num_negative_samples = max(num_candidates - int(self.cfg.include_positive), 0)
            candidate_unique_ratio = self._candidate_unique_ratio(trajectories.candidate_token_ids)
            generated_sampler_active = 1.0 if trajectories.extra.get("sampler") == "generated_semantic_id" else 0.0
            score_sampler_active = (
                1.0
                if str(trajectories.extra.get("sampler", "")).endswith("_semantic_id")
                and trajectories.extra.get("sampler") not in {"random_semantic_id", "generated_semantic_id"}
                else 0.0
            )
            easy_negative_active = 1.0 if trajectories.extra.get("sampler_phase") == "easy" else 0.0
        else:
            metric_device = sft_out.loss.device
            num_candidates = 0
            num_negative_samples = 0
            candidate_unique_ratio = zero
            generated_sampler_active = 0.0
            score_sampler_active = 0.0
            easy_negative_active = 0.0

        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "sft_loss": sft_out.loss.detach(),
            "gflow_loss": gflow_loss.detach(),
            "exact_terminal_loss": exact_loss.detach(),
            "log_z_mean": log_z.detach().mean(),
            "log_z_std": log_z.detach().reshape(-1).std(unbiased=False),
            "num_candidates": torch.tensor(num_candidates, device=metric_device, dtype=torch.float32),
            "num_negative_samples": torch.tensor(num_negative_samples, device=metric_device, dtype=torch.float32),
            "candidate_unique_ratio": candidate_unique_ratio.detach(),
            "generated_sampler_active": torch.tensor(
                generated_sampler_active, device=metric_device, dtype=torch.float32
            ),
            "score_sampler_active": torch.tensor(score_sampler_active, device=metric_device, dtype=torch.float32),
            "easy_negative_active": torch.tensor(easy_negative_active, device=metric_device, dtype=torch.float32),
        }
        metrics.update(gflow_metrics)
        metrics.update(exact_metrics)

        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        method_metrics.update(sft_out.metrics)

        extra = dict(sft_out.extra)
        extra.update({"stage": stage, "step": step})
        if trajectories is not None:
            extra.update(
                {
                    "trajectories": trajectories,
                    "rewards": rewards,
                    "sequence_logprobs": sequence_logprobs,
                }
            )
            if token_logprobs is not None:
                extra["token_logprobs"] = token_logprobs

        return MethodOutput(
            loss=loss,
            metrics=method_metrics,
            model_output=sft_out.model_output,
            extra=extra,
        )

    def _compute_cyclic_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        scoring_batch: Batch,
        sft_out: MethodOutput,
        *,
        stage: str,
        step: int | None,
    ) -> MethodOutput:
        zero = sft_out.loss.new_zeros(())
        gflow_loss = zero
        gflow_metrics: dict[str, torch.Tensor] = {
            "cyclic_active": torch.tensor(0.0, device=zero.device)
        }
        trajectories: CyclicSemanticTrajectoryBatch | None = None
        forward_logprobs = None
        sequence_logprobs = None
        log_z = zero

        if stage == "train" and float(self.cfg.gflow_weight) != 0.0:
            trajectories, log_z, cache_metrics = self._sample_cyclic_trajectories(model, batch)
            trajectories.is_positive = self._cached_positive_mask(batch, trajectories.terminal_token_ids)
            forward_logprobs, policy_metrics = self._cyclic_forward_logprobs(
                model, scoring_batch, trajectories
            )
            sequence_logprobs = (
                forward_logprobs * trajectories.step_mask.to(forward_logprobs.dtype)
            ).sum(dim=-1)
            if self.cfg.objective == "tb":
                gflow_loss, gflow_metrics = cyclic_trajectory_balance_loss(
                    forward_logprobs,
                    trajectories.backward_logprobs,
                    trajectories.step_mask,
                    trajectories.terminal_rewards,
                    log_z,
                    trajectories.root_backward_logprobs,
                    reward_epsilon=float(self.cfg.reward_epsilon),
                )
            elif self.cfg.objective == "db":
                gflow_loss, gflow_metrics = cyclic_detailed_balance_loss(
                    forward_logprobs,
                    trajectories.backward_logprobs,
                    trajectories.state_log_flows,
                    trajectories.step_mask,
                )
            elif self.cfg.objective == "subtb":
                gflow_loss, gflow_metrics = cyclic_subtrajectory_balance_loss(
                    forward_logprobs,
                    trajectories.backward_logprobs,
                    trajectories.state_log_flows,
                    trajectories.step_mask,
                    subtb_lambda=float(self.cfg.subtb_lambda),
                    max_subtrajectory_length=self.cfg.subtb_max_length,
                    estimator=self.cfg.cyclic_subtb_estimator,
                    num_mc_samples=int(self.cfg.cyclic_subtb_num_samples),
                )
            else:
                raise ValueError(f"Unknown GFlow objective: {self.cfg.objective}")
            gflow_metrics["cyclic_active"] = torch.tensor(1.0, device=zero.device)
            gflow_metrics.update(cache_metrics)
            gflow_metrics.update(policy_metrics)
            gflow_metrics.update(self._cyclic_trajectory_metrics(trajectories))

        exact_loss = zero
        exact_metrics: dict[str, torch.Tensor] = {}
        if bool(self.cfg.exact_terminal_distribution) and float(self.cfg.exact_distribution_loss_weight) != 0.0:
            if stage == "train":
                exact_loss, exact_metrics = self._precomputed_reward_cache_loss(model, batch)
            else:
                exact_metrics = {"reward_cache_active": torch.tensor(0.0, device=zero.device)}
        loss = (
            float(self.cfg.sft_weight) * sft_out.loss
            + float(self.cfg.gflow_weight) * gflow_loss
            + float(self.cfg.exact_distribution_loss_weight) * exact_loss
        )
        metric_device = zero.device if trajectories is None else trajectories.terminal_token_ids.device
        num_candidates = 0 if trajectories is None else trajectories.num_candidates
        unique_ratio = zero if trajectories is None else self._candidate_unique_ratio(trajectories.terminal_token_ids)
        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "sft_loss": sft_out.loss.detach(),
            "gflow_loss": gflow_loss.detach(),
            "exact_terminal_loss": exact_loss.detach(),
            "log_z_mean": log_z.detach().mean(),
            "log_z_std": log_z.detach().reshape(-1).std(unbiased=False),
            "num_candidates": torch.tensor(num_candidates, device=metric_device, dtype=torch.float32),
            "num_negative_samples": torch.tensor(0.0, device=metric_device),
            "candidate_unique_ratio": unique_ratio.detach(),
            "generated_sampler_active": torch.tensor(0.0, device=metric_device),
            "score_sampler_active": torch.tensor(0.0, device=metric_device),
            "easy_negative_active": torch.tensor(0.0, device=metric_device),
        }
        metrics.update(gflow_metrics)
        metrics.update(exact_metrics)
        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        method_metrics.update(sft_out.metrics)
        extra = dict(sft_out.extra)
        extra.update({"stage": stage, "step": step})
        if trajectories is not None:
            extra.update(
                {
                    "trajectories": trajectories,
                    "rewards": trajectories.terminal_rewards,
                    "sequence_logprobs": sequence_logprobs,
                    "token_logprobs": forward_logprobs,
                }
            )
        return MethodOutput(loss=loss, metrics=method_metrics, model_output=sft_out.model_output, extra=extra)

    def _sample_cyclic_trajectories(
        self, model: FlowRecModel, batch: Batch
    ) -> tuple[CyclicSemanticTrajectoryBatch, torch.Tensor, dict[str, torch.Tensor]]:
        if "example_indices" not in batch:
            raise KeyError("Cyclic GFlow requires batch['example_indices'].")
        item_tokens = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_tokens) or item_tokens.numel() == 0:
            raise ValueError("Cyclic GFlow requires model.item_semantic_ids.")
        device = batch["example_indices"].device
        cache = self._get_reward_cache()
        raw_rewards, cached_reward_sum = cache.select_full_catalog(batch["example_indices"], device)
        dense_rewards = cache.transform_candidate_rewards(raw_rewards).clamp_min(float(self.cfg.reward_epsilon))
        reward_sum = dense_rewards.double().sum(dim=1).float()
        trajectories = sample_analytical_cyclic_trajectories(
            flow_index=self._get_exact_flow_index(model),
            dense_rewards=dense_rewards,
            item_tokens=item_tokens,
            num_trajectories=int(self.cfg.cyclic_num_trajectories),
            cycle_flow_ratio=float(self.cfg.cyclic_flow_ratio),
            max_steps=int(self.cfg.cyclic_max_steps),
            reward_epsilon=float(self.cfg.reward_epsilon),
            delete_probability_cap=float(self.cfg.cyclic_delete_prob_cap),
        )
        relative_error = (reward_sum - cached_reward_sum.to(reward_sum)).abs() / reward_sum.clamp_min(
            float(self.cfg.reward_epsilon)
        )
        metrics = {
            "cyclic_active": torch.tensor(1.0, device=device),
            "fixed_log_z_active": torch.tensor(1.0, device=device),
            "reward_cache_tb_active": torch.tensor(1.0, device=device),
            "reward_cache_tb_reward_sum_mean": reward_sum.detach().mean(),
            "cyclic_cached_reward_sum_relative_error": relative_error.detach().mean(),
            "cyclic_full_catalog_items": torch.tensor(dense_rewards.size(1), device=device, dtype=torch.float32),
        }
        return trajectories, reward_sum.clamp_min(float(self.cfg.reward_epsilon)).log(), metrics

    def _cyclic_forward_logprobs(
        self,
        model: FlowRecModel,
        batch: Batch,
        trajectories: CyclicSemanticTrajectoryBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        append_mask = trajectories.step_mask & trajectories.action_types.eq(APPEND_ACTION)
        bsz, _, num_steps = append_mask.shape
        sid_len = int(trajectories.state_token_ids.size(-1))
        append_counts = append_mask.sum(dim=(1, 2))
        if int(append_counts.max()) <= 0:
            raise RuntimeError("Cyclic trajectories contain no APPEND actions.")

        device = trajectories.state_token_ids.device
        active_batch, active_flat = append_mask.reshape(bsz, -1).nonzero(as_tuple=True)
        active_paths = torch.div(active_flat, num_steps, rounding_mode="floor")
        active_steps = active_flat.remainder(num_steps)
        active_depths = trajectories.state_depths[active_batch, active_paths, active_steps]
        child_tokens = trajectories.state_token_ids[active_batch, active_paths, active_steps + 1]
        rows = torch.cat(
            [active_batch[:, None], (active_depths + 1)[:, None], child_tokens], dim=1
        )
        unique_rows, inverse = torch.unique(rows, dim=0, sorted=True, return_inverse=True)
        unique_batch = unique_rows[:, 0]
        unique_depths = unique_rows[:, 1] - 1
        unique_tokens = unique_rows[:, 2:]
        unique_counts = torch.bincount(unique_batch, minlength=bsz)
        width = int(unique_counts.max())
        starts = unique_counts.cumsum(0) - unique_counts
        slots = torch.arange(unique_rows.size(0), device=device) - starts[unique_batch]
        packed_tokens = torch.zeros((bsz, width, sid_len), device=device, dtype=torch.long)
        packed_mask = torch.zeros_like(packed_tokens, dtype=torch.bool)
        packed_depths = torch.zeros((bsz, width), device=device, dtype=torch.long)
        packed_tokens[unique_batch, slots] = unique_tokens
        packed_mask[unique_batch, slots] = torch.arange(sid_len, device=device)[None, :] < (unique_depths + 1)[:, None]
        packed_depths[unique_batch, slots] = unique_depths
        packed = SemanticTrajectoryBatch(
            candidate_token_ids=packed_tokens,
            candidate_token_mask=packed_mask,
            is_positive=torch.zeros((bsz, width), device=device, dtype=torch.bool),
            extra={"sampler": "packed_cyclic_append_actions"},
        )
        token_logprobs = self._candidate_token_logprobs(model, batch, packed)
        selected = token_logprobs.gather(-1, packed_depths.unsqueeze(-1)).squeeze(-1)
        forward = selected.new_zeros(trajectories.action_types.shape)
        forward[active_batch, active_paths, active_steps] = selected[unique_batch, slots][inverse]

        internal_delete, leaf_delete = self._cyclic_delete_probabilities()
        depths = trajectories.state_depths[..., :-1]
        internal_append = append_mask & depths.gt(0)
        delete_mask = trajectories.step_mask & trajectories.action_types.eq(DELETE_ACTION)
        leaf_delete_mask = delete_mask & depths.eq(sid_len)
        internal_delete_mask = delete_mask & ~leaf_delete_mask
        stop_mask = trajectories.step_mask & trajectories.action_types.eq(STOP_ACTION)
        forward = forward + internal_append.to(forward.dtype) * torch.log1p(-internal_delete)
        forward = torch.where(internal_delete_mask, torch.log(internal_delete), forward)
        forward = torch.where(leaf_delete_mask, torch.log(leaf_delete), forward)
        forward = torch.where(stop_mask, torch.log1p(-leaf_delete), forward)
        forward = forward.masked_fill(~trajectories.step_mask, 0.0)
        return forward, {
            "cyclic_delete_prob_internal": internal_delete.detach(),
            "cyclic_delete_prob_leaf": leaf_delete.detach(),
            "cyclic_append_actions_scored_mean": append_counts.float().mean(),
            "cyclic_unique_append_actions_mean": unique_counts.float().mean(),
            "cyclic_append_dedup_ratio": forward.new_tensor(unique_rows.size(0)) / append_counts.sum().clamp_min(1),
            "cyclic_append_pack_width": forward.new_tensor(width, dtype=torch.float32),
        }

    def _cyclic_delete_probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = float(self.cfg.cyclic_delete_prob_cap) * torch.sigmoid(self.cyclic_delete_logits.float())
        return probabilities[0], probabilities[1]

    def _cyclic_trajectory_metrics(
        self, trajectories: CyclicSemanticTrajectoryBatch
    ) -> dict[str, torch.Tensor]:
        mask = trajectories.step_mask
        lengths = mask.sum(dim=-1).float()
        deletes = (mask & trajectories.action_types.eq(DELETE_ACTION)).sum(dim=-1).float()
        kappa = float(self.cfg.cyclic_flow_ratio)
        sid_len = int(trajectories.state_token_ids.size(-1))
        reversible = mask & (
            trajectories.action_types.eq(APPEND_ACTION) | trajectories.action_types.eq(DELETE_ACTION)
        )
        backward_probs = trajectories.backward_logprobs[reversible].float().exp()
        metrics = {
            "cyclic_flow_ratio": torch.tensor(kappa, device=mask.device),
            "cyclic_trajectory_length_mean": lengths.mean(),
            "cyclic_trajectory_length_max": lengths.max(),
            "cyclic_delete_count_mean": deletes.mean(),
            "cyclic_expected_trajectory_length": torch.tensor((1 + 2 * kappa) * sid_len + 1, device=mask.device),
            "cyclic_expected_delete_count": torch.tensor(kappa * sid_len, device=mask.device),
            "cyclic_watchdog_rate": torch.tensor(0.0, device=mask.device),
        }
        if backward_probs.numel():
            metrics.update(
                {
                    "cyclic_backward_prob_mean": backward_probs.mean(),
                    "cyclic_backward_prob_std": backward_probs.std(unbiased=False),
                    "cyclic_backward_entropy_mean": -trajectories.backward_logprobs[reversible].float().mean(),
                }
            )
        return metrics

    def _precomputed_reward_cache_tb_inputs(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> tuple[
        SemanticTrajectoryBatch,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if "example_indices" not in batch:
            raise KeyError("Reward-cache trajectory balance requires batch['example_indices'].")
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
            raise ValueError("Reward-cache trajectory balance requires model.item_semantic_ids.")

        device = batch["example_indices"].device
        cache = self._get_reward_cache()
        (
            rows,
            rewards,
            reward_sum,
            candidate_weights,
            top_mass,
            target_in_top,
            cached_prefix_log_flows,
        ) = cache.select_tb(batch["example_indices"], device=device)
        item_tokens = item_semantic_ids.to(device).long()
        safe_rows = rows.clamp_min(0).clamp_max(max(int(item_tokens.size(0)) - 1, 0))
        rewards = rewards.to(device=device, dtype=torch.float32).clamp_min(float(self.cfg.reward_epsilon))
        sampler_name = "reward_cache_tb"
        if self.cfg.tb_catalog_sampling != "legacy":
            (
                safe_rows,
                rewards,
                reward_sum,
                candidate_weights,
                top_mass,
                sampler_name,
            ) = self._sample_full_catalog_tb(cache, batch["example_indices"], safe_rows, rewards, device)
        candidate_weights = candidate_weights.to(device=device, dtype=torch.float32).clamp_min(0.0)
        log_z = reward_sum.to(device=device, dtype=torch.float32).clamp_min(float(self.cfg.reward_epsilon)).log()
        extra: dict[str, Any] = {"sampler": sampler_name, "sampler_phase": "fixed_logz"}
        needs_exact_flows = self.cfg.objective in {"db", "subtb"}
        sample_size = int(self.cfg.candidate_sample_size)
        sample_strategy = str(self.cfg.candidate_sample_strategy)
        if needs_exact_flows or sample_size > 0:
            extra["cache_top_rows"] = safe_rows
            extra["cache_top_rewards"] = rewards
        if sample_size > 0:
            if self.cfg.objective == "tb":
                raise ValueError("candidate_sample_size is only supported for objective='db' or 'subtb'.")
            if sample_strategy == "reward_proportional":
                if self.cfg.objective == "subtb" and self.cfg.tb_loss_weighting != "reward":
                    raise ValueError("Reward-proportional SubTB sampling requires reward weighting.")
                sampled_indices, candidate_weights = reward_proportional_sample(
                    candidate_weights,
                    sample_size=sample_size,
                )
                safe_rows = safe_rows.gather(1, sampled_indices).contiguous()
                rewards = rewards.gather(1, sampled_indices).contiguous()
                if cached_prefix_log_flows is not None:
                    flow_indices = sampled_indices.unsqueeze(-1).expand(-1, -1, cached_prefix_log_flows.size(-1))
                    cached_prefix_log_flows = cached_prefix_log_flows.gather(1, flow_indices).contiguous()
                extra["sampler"] = "reward_cache_reward_proportional"
            elif sample_strategy == "legacy" and (
                cached_prefix_log_flows is not None and int(self.cfg.candidate_sample_top_k) >= sample_size
            ):
                safe_rows = safe_rows[:, :sample_size].contiguous()
                rewards = rewards[:, :sample_size].contiguous()
                candidate_weights = rewards / reward_sum.unsqueeze(1).clamp_min(float(self.cfg.reward_epsilon))
                log_z = reward_sum.clamp_min(float(self.cfg.reward_epsilon)).log()
                cached_prefix_log_flows = cached_prefix_log_flows[:, :sample_size, :].contiguous()
                extra["sampler"] = "reward_cache_top_head"
            elif sample_strategy in {"stratified", "legacy"}:
                index = self._get_exact_flow_index(model)
                dense = index.dense_rewards(safe_rows, rewards, reward_epsilon=float(self.cfg.reward_epsilon))
                sampled_rows, sampled_weights = stratified_catalog_sample(
                    dense,
                    safe_rows,
                    sample_size=sample_size,
                    top_k=int(self.cfg.candidate_sample_top_k),
                )
                safe_rows = sampled_rows
                rewards = dense.gather(1, sampled_rows)
                candidate_weights = sampled_weights
                log_z = dense.sum(dim=1).clamp_min(float(self.cfg.reward_epsilon)).log()
                cached_prefix_log_flows = None
                extra["sampler"] = "reward_cache_stratified"
            else:
                raise ValueError(f"Unsupported candidate sampling strategy: {sample_strategy!r}")
        if needs_exact_flows and cached_prefix_log_flows is not None:
            extra["cache_prefix_log_flows"] = cached_prefix_log_flows
        candidate_tokens = item_tokens[safe_rows]
        candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
        is_positive = self._cached_positive_mask(batch, candidate_tokens)
        trajectories = SemanticTrajectoryBatch(
            candidate_token_ids=candidate_tokens,
            candidate_token_mask=candidate_mask,
            is_positive=is_positive,
            candidate_item_row_ids=safe_rows,
            extra=extra,
        )
        metrics = {
            "reward_cache_tb_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "fixed_log_z_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "reward_cache_tb_top_m": torch.tensor(trajectories.num_candidates, device=device, dtype=torch.float32),
            "reward_cache_tb_reward_sum_mean": reward_sum.detach().mean(),
            "reward_cache_tb_top_mass_mean": top_mass.detach().mean(),
            "reward_cache_tb_weight_mass_mean": candidate_weights.detach().sum(dim=1).mean(),
            "reward_cache_tb_target_in_top_mean": target_in_top.detach().mean(),
            "reward_cache_candidate_sample_size": torch.tensor(sample_size, device=device, dtype=torch.float32),
            "reward_proportional_sample_active": torch.tensor(
                float(sample_size > 0 and sample_strategy == "reward_proportional"),
                device=device,
            ),
            "stratified_sample_active": torch.tensor(
                float(sample_size > 0 and sample_strategy in {"stratified", "legacy"}),
                device=device,
            ),
            "exact_candidate_evaluation_active": torch.tensor(float(sample_size <= 0), device=device),
        }
        return trajectories, rewards, log_z, candidate_weights, metrics

    def _sample_full_catalog_tb(
        self,
        cache: _PrecomputedRewardCache,
        example_indices: torch.Tensor,
        head_rows: torch.Tensor,
        head_rewards: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        full_raw_rewards, full_reward_sum = cache.select_full_catalog(example_indices, device)
        bsz, num_items = full_raw_rewards.shape
        head_count = int(self.cfg.tb_head_sample_size)
        tail_count = int(self.cfg.tb_tail_sample_size)
        if head_count < 0 or tail_count < 0 or head_count + tail_count <= 0:
            raise ValueError("TB catalogue sample sizes must be non-negative and sum to > 0.")

        if self.cfg.tb_catalog_sampling == "uniform":
            if head_count != 0:
                raise ValueError("Uniform catalogue sampling requires tb_head_sample_size=0.")
            rows = torch.randint(num_items, (bsz, tail_count), device=device)
            rewards = cache.transform_candidate_rewards(full_raw_rewards.gather(1, rows))
            weights = (
                rewards
                * float(num_items)
                / full_reward_sum.unsqueeze(1).clamp_min(float(self.cfg.reward_epsilon))
                / float(tail_count)
            )
            top_mass = torch.zeros(bsz, device=device, dtype=torch.float32)
            return (
                rows,
                rewards,
                full_reward_sum,
                weights,
                top_mass,
                "reward_cache_full_uniform",
            )

        if head_count <= 0 or tail_count <= 0:
            raise ValueError("Hybrid catalogue sampling requires positive head and tail sizes.")
        if head_rows.size(1) >= num_items:
            raise ValueError("Hybrid catalogue sampling requires at least one tail item.")

        head_sum = head_rewards.sum(dim=1).clamp_min(float(self.cfg.reward_epsilon))
        head_probs = head_rewards / head_sum.unsqueeze(1)
        head_indices = torch.multinomial(head_probs, head_count, replacement=True)
        sampled_head_rows = head_rows.gather(1, head_indices)
        sampled_head_rewards = head_rewards.gather(1, head_indices)
        head_mass = head_sum / full_reward_sum.clamp_min(float(self.cfg.reward_epsilon))
        head_weights = head_mass.unsqueeze(1).expand(-1, head_count) / float(head_count)

        sampled_tail_rows = self._sample_uniform_excluding(head_rows, tail_count, num_items)
        sampled_tail_rewards = cache.transform_candidate_rewards(full_raw_rewards.gather(1, sampled_tail_rows))
        tail_weights = (
            sampled_tail_rewards
            * float(num_items - head_rows.size(1))
            / full_reward_sum.unsqueeze(1).clamp_min(float(self.cfg.reward_epsilon))
            / float(tail_count)
        )

        return (
            torch.cat([sampled_head_rows, sampled_tail_rows], dim=1),
            torch.cat([sampled_head_rewards, sampled_tail_rewards], dim=1),
            full_reward_sum,
            torch.cat([head_weights, tail_weights], dim=1),
            head_mass,
            f"reward_cache_hybrid_{head_count}_{tail_count}",
        )

    @staticmethod
    def _sample_uniform_excluding(
        excluded_rows: torch.Tensor,
        sample_size: int,
        num_items: int,
    ) -> torch.Tensor:
        excluded = excluded_rows.long().sort(dim=1).values
        if excluded.size(1) >= num_items:
            raise ValueError("Cannot sample outside an excluded full catalogue.")
        ranks = torch.randint(
            num_items - excluded.size(1),
            (excluded.size(0), int(sample_size)),
            device=excluded.device,
        )
        rows = ranks
        for column in excluded.unbind(dim=1):
            rows = rows + rows.ge(column.unsqueeze(1)).long()
        return rows

    def _cached_positive_mask(self, batch: Batch, candidate_tokens: torch.Tensor) -> torch.Tensor:
        target = batch.get("target_token_ids")
        if not torch.is_tensor(target):
            return torch.zeros(
                candidate_tokens.shape[:2],
                device=candidate_tokens.device,
                dtype=torch.bool,
            )
        target = target.to(candidate_tokens.device).long()
        target_mask = batch.get("target_token_mask")
        if torch.is_tensor(target_mask):
            target_mask = target_mask.to(candidate_tokens.device).bool()
        else:
            target_mask = torch.ones_like(target, dtype=torch.bool, device=candidate_tokens.device)
        matches = candidate_tokens.eq(target.unsqueeze(1)) | ~target_mask.unsqueeze(1)
        return matches.all(dim=-1)

    def _precomputed_reward_cache_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "example_indices" not in batch:
            raise KeyError("Precomputed reward cache loss requires batch['example_indices'].")
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
            raise ValueError("Precomputed reward cache loss requires model.item_semantic_ids.")

        device = batch["example_indices"].device
        cache = self._get_reward_cache()
        rows, probs, top_mass, target_in_top = cache.select(batch["example_indices"], device=device)
        item_tokens = item_semantic_ids.to(device).long()
        safe_rows = rows.clamp_min(0).clamp_max(max(int(item_tokens.size(0)) - 1, 0))
        candidate_tokens = item_tokens[safe_rows]
        candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)

        score_chunk_size = self.cfg.exact_distribution_score_chunk_size
        if score_chunk_size is None:
            score_chunk_size = self.cfg.exact_distribution_chunk_size
        score_chunk_size = max(int(score_chunk_size), 1)

        bsz, num_candidates, _ = candidate_tokens.shape
        loss_per_example = probs.new_zeros((bsz,), dtype=torch.float32)
        model_mass_terms = probs.new_zeros((bsz,), dtype=torch.float32)
        for start in range(0, num_candidates, score_chunk_size):
            end = min(start + score_chunk_size, num_candidates)
            score_tokens = candidate_tokens[:, start:end, :]
            score_mask = candidate_mask[:, start:end, :]

            def score_fn(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return model.score_items(
                    batch,
                    candidate_token_ids=tokens,
                    candidate_token_mask=mask,
                )

            log_q = checkpoint(score_fn, score_tokens, score_mask, use_reentrant=False)
            expected = (bsz, end - start)
            if log_q.shape != expected:
                raise ValueError(
                    "Model score shape mismatch in precomputed reward cache loss: "
                    f"log_q={tuple(log_q.shape)} expected={expected}"
                )
            p = probs[:, start:end].to(log_q.device).to(log_q.dtype)
            loss_per_example = loss_per_example + (-(p * log_q).sum(dim=1)).float()
            model_mass_terms = model_mass_terms + torch.exp(log_q.detach()).float().sum(dim=1)

        loss = loss_per_example.mean()
        prob_sum = probs.sum(dim=1)
        metrics = {
            "reward_cache_loss": loss.detach(),
            "reward_cache_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "reward_cache_top_m": torch.tensor(num_candidates, device=device, dtype=torch.float32),
            "reward_cache_prob_mass_mean": prob_sum.detach().mean(),
            "reward_cache_top_mass_mean": top_mass.detach().mean(),
            "reward_cache_target_in_top_mean": target_in_top.detach().mean(),
            "reward_cache_model_top_mass_mean": model_mass_terms.detach().mean(),
        }
        return loss, metrics

    def _candidate_scoring_batch(
        self,
        batch: Batch,
        model_output: Any,
    ) -> Batch:
        reuse_cyclic = self.cfg.environment == "cyclic"
        reuse_tb = self.cfg.objective == "tb" and bool(self.cfg.tb_reuse_sft_encoder)
        reuse_db = self.cfg.objective in {"db", "subtb"} and bool(self.cfg.db_reuse_sft_encoder)
        if not (reuse_cyclic or reuse_tb or reuse_db):
            return batch
        if reuse_db and bool(self.cfg.db_activation_checkpointing):
            raise ValueError("db_reuse_sft_encoder requires db_activation_checkpointing=false.")
        extra = getattr(model_output, "extra", None)
        if not isinstance(extra, dict):
            raise ValueError("Encoder reuse requires model output with encoder state metadata.")
        encoder_hidden = extra.get("encoder_last_hidden_state")
        attention_mask = extra.get("encoder_attention_mask")
        if not torch.is_tensor(encoder_hidden) or not torch.is_tensor(attention_mask):
            raise ValueError(
                "Encoder reuse is enabled, but the model did not expose "
                "encoder_last_hidden_state and encoder_attention_mask."
            )
        scoring_batch = dict(batch)
        scoring_batch["_tiger_encoder_last_hidden_state"] = encoder_hidden
        scoring_batch["_tiger_encoder_attention_mask"] = attention_mask
        return scoring_batch

    def _candidate_token_logprobs_and_prefix_flows(
        self,
        model: FlowRecModel,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_tokens = trajectories.candidate_token_ids
        candidate_mask = trajectories.candidate_token_mask
        if candidate_tokens.ndim != 3:
            raise ValueError("candidate_token_ids must have shape [B, N, L].")
        if candidate_mask.shape != candidate_tokens.shape:
            raise ValueError("candidate_token_mask must match candidate_token_ids.")
        if rewards.shape != candidate_tokens.shape[:2]:
            raise ValueError(
                "rewards must have shape [B, N]. "
                f"Got {tuple(rewards.shape)} for candidates {tuple(candidate_tokens.shape)}."
            )
        top_rows = trajectories.extra.get("cache_top_rows")
        top_rewards = trajectories.extra.get("cache_top_rewards")
        if not torch.is_tensor(top_rows) or not torch.is_tensor(top_rewards):
            raise ValueError(
                "Analytical DB/SubTB flows require reward-cache candidates "
                "providing cache_top_rows and cache_top_rewards."
            )

        token_logprobs = self._candidate_token_logprobs(model, batch, trajectories)
        cached_prefix_log_flows = trajectories.extra.get("cache_prefix_log_flows")
        if torch.is_tensor(cached_prefix_log_flows):
            expected = (
                int(candidate_tokens.size(0)),
                int(candidate_tokens.size(1)),
                int(candidate_tokens.size(2)) + 1,
            )
            if tuple(cached_prefix_log_flows.shape) != expected:
                raise ValueError(
                    "Cached prefix flow shape mismatch: "
                    f"got {tuple(cached_prefix_log_flows.shape)}; expected {expected}."
                )
            return token_logprobs, cached_prefix_log_flows.to(
                device=token_logprobs.device,
                dtype=token_logprobs.dtype,
            )

        index = self._get_exact_flow_index(model)
        dense = index.dense_rewards(
            top_rows,
            top_rewards,
            reward_epsilon=float(self.cfg.reward_epsilon),
        )
        prefix_log_flows = index.prefix_log_flows(
            candidate_tokens,
            dense,
            reward_epsilon=float(self.cfg.reward_epsilon),
        ).to(dtype=token_logprobs.dtype)
        return token_logprobs, prefix_log_flows

    def _get_exact_flow_index(self, model: FlowRecModel) -> CatalogPrefixFlowIndex:
        if self._exact_flow_index is None:
            item_semantic_ids = getattr(model, "item_semantic_ids", None)
            if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
                raise ValueError("Exact prefix flows require model.item_semantic_ids.")
            self._exact_flow_index = CatalogPrefixFlowIndex(item_semantic_ids)
        return self._exact_flow_index

    def _candidate_token_logprobs(
        self,
        model: FlowRecModel,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        candidate_tokens = trajectories.candidate_token_ids
        candidate_mask = trajectories.candidate_token_mask
        if candidate_tokens.ndim != 3:
            raise ValueError("candidate_token_ids must have shape [B, N, L] for detailed balance.")
        if candidate_mask.shape != candidate_tokens.shape:
            raise ValueError("candidate_token_mask must match candidate_token_ids for detailed balance.")

        num_candidates = int(candidate_tokens.size(1))
        score_chunk_size = max(int(self.cfg.db_score_chunk_size), 1)
        token_scorer = getattr(model, "score_item_tokens_valid_actions", None)
        if not callable(token_scorer):
            token_scorer = getattr(model, "score_item_tokens", None)
        if callable(token_scorer):
            chunks: list[torch.Tensor] = []
            for start in range(0, num_candidates, score_chunk_size):
                end = min(start + score_chunk_size, num_candidates)
                score_tokens = candidate_tokens[:, start:end, :]
                score_mask = candidate_mask[:, start:end, :]

                def score_fn(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                    return token_scorer(
                        batch,
                        candidate_token_ids=tokens,
                        candidate_token_mask=mask,
                    )

                reuse_encoder = "_tiger_encoder_last_hidden_state" in batch
                if bool(self.cfg.db_activation_checkpointing) and not reuse_encoder:
                    chunks.append(checkpoint(score_fn, score_tokens, score_mask, use_reentrant=False))
                else:
                    chunks.append(score_fn(score_tokens, score_mask))
            stacked = torch.cat(chunks, dim=1)
            return stacked.masked_fill(~candidate_mask.to(stacked.device).bool(), 0.0)

        # Generic fallback for models without token-level candidate scoring.
        seq_len = int(candidate_tokens.size(-1))
        positions = torch.arange(seq_len, device=candidate_tokens.device).view(1, 1, seq_len)
        prefix_scores: list[torch.Tensor] = []
        for end in range(1, seq_len + 1):
            prefix_mask = candidate_mask & positions.lt(end)
            score_chunks: list[torch.Tensor] = []
            for start in range(0, num_candidates, score_chunk_size):
                chunk_end = min(start + score_chunk_size, num_candidates)
                score_chunks.append(
                    model.score_items(
                        batch,
                        candidate_token_ids=candidate_tokens[:, start:chunk_end, :],
                        candidate_token_mask=prefix_mask[:, start:chunk_end, :],
                    )
                )
            prefix_scores.append(torch.cat(score_chunks, dim=1))

        token_logprobs = []
        previous = None
        for score in prefix_scores:
            token_logprobs.append(score if previous is None else score - previous)
            previous = score
        stacked = torch.stack(token_logprobs, dim=-1)
        return stacked.masked_fill(~candidate_mask.to(stacked.device).bool(), 0.0)

    def _get_reward_cache(self) -> _PrecomputedRewardCache:
        if self._reward_cache is None:
            if not self.cfg.reward_cache_path:
                raise ValueError("reward_cache_path is required for precomputed reward cache loss.")
            self._reward_cache = _PrecomputedRewardCache(
                self.cfg.reward_cache_path,
                top_m=int(self.cfg.reward_cache_top_m),
                probability=str(self.cfg.reward_cache_probability),
                transform=str(self.cfg.reward_cache_transform),
                power=float(self.cfg.reward_cache_power),
                temperature=float(self.cfg.reward_cache_temperature),
                rank_alpha=float(self.cfg.reward_cache_rank_alpha),
                score_source=str(self.cfg.reward_cache_score_source),
                load_full_catalog=(self.cfg.tb_catalog_sampling != "legacy" or self.cfg.environment == "cyclic"),
                expected_canonical_reward=self.cfg.canonical_reward_manifest(),
            )
        return self._reward_cache

    def _exact_terminal_distribution_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
            raise ValueError("exact_terminal_distribution requires model.item_semantic_ids.")
        if "target_token_ids" not in batch:
            raise KeyError("exact_terminal_distribution requires batch['target_token_ids'].")

        target_token_ids = batch["target_token_ids"]
        if not torch.is_tensor(target_token_ids):
            raise TypeError("batch['target_token_ids'] must be a torch.Tensor.")
        device = target_token_ids.device
        bsz = int(target_token_ids.size(0))

        item_tokens = item_semantic_ids.to(device).long()
        max_items = self.cfg.exact_distribution_max_items
        if max_items is not None:
            item_tokens = item_tokens[: max(int(max_items), 1)]
        num_items = int(item_tokens.size(0))
        if num_items <= 0:
            raise ValueError("exact_terminal_distribution has an empty item catalog.")

        chunk_size = max(int(self.cfg.exact_distribution_chunk_size), 1)
        score_chunk_size = self.cfg.exact_distribution_score_chunk_size
        if score_chunk_size is None:
            score_chunk_size = chunk_size
        score_chunk_size = max(int(score_chunk_size), 1)
        reward_epsilon = float(self.cfg.reward_epsilon)

        # Exact CE target without materializing [B, num_items, sid_len] or [B, num_items].
        # CE(P_R || P_model) = - sum_i R_i log P_model(i) / sum_i R_i.
        reward_sum = target_token_ids.new_zeros((bsz,), dtype=torch.float32)
        reward_log_sum = target_token_ids.new_zeros((bsz,), dtype=torch.float32)
        reward_max = target_token_ids.new_full((bsz,), reward_epsilon, dtype=torch.float32)
        weighted_logprob_sum = target_token_ids.new_zeros((bsz,), dtype=torch.float32)

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            chunk_tokens = item_tokens[start:end]
            chunk_len = int(chunk_tokens.size(0))
            chunk_mask = torch.ones_like(chunk_tokens, dtype=torch.bool, device=device)
            candidate_tokens = chunk_tokens.unsqueeze(0).expand(bsz, -1, -1)
            candidate_mask = chunk_mask.unsqueeze(0).expand(bsz, -1, -1)
            candidate_rows = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            trajectories = SemanticTrajectoryBatch(
                candidate_token_ids=candidate_tokens,
                candidate_token_mask=candidate_mask,
                is_positive=self._catalog_positive_mask(batch, chunk_tokens),
                candidate_item_row_ids=candidate_rows,
            )

            with torch.no_grad():
                rewards = self.reward_model(batch, trajectories).float().clamp_min(reward_epsilon)

            if rewards.shape != (bsz, chunk_len):
                raise ValueError(
                    "Reward shape mismatch in exact_terminal_distribution: "
                    f"rewards={tuple(rewards.shape)} expected={(bsz, chunk_len)}"
                )

            for score_start in range(0, chunk_len, score_chunk_size):
                score_end = min(score_start + score_chunk_size, chunk_len)
                score_tokens = chunk_tokens[score_start:score_end]
                score_mask = chunk_mask[score_start:score_end]
                score_candidate_tokens = score_tokens.unsqueeze(0).expand(bsz, -1, -1)
                score_candidate_mask = score_mask.unsqueeze(0).expand(bsz, -1, -1)

                def score_fn(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                    return model.score_items(
                        batch,
                        candidate_token_ids=tokens,
                        candidate_token_mask=mask,
                    )

                log_q = checkpoint(
                    score_fn,
                    score_candidate_tokens,
                    score_candidate_mask,
                    use_reentrant=False,
                )
                expected_shape = (bsz, score_end - score_start)
                if log_q.shape != expected_shape:
                    raise ValueError(
                        "Model score shape mismatch in exact_terminal_distribution: "
                        f"log_q={tuple(log_q.shape)} expected={expected_shape}"
                    )
                rewards_for_grad = rewards[:, score_start:score_end].to(log_q.device).to(log_q.dtype)
                weighted_logprob_sum = weighted_logprob_sum + (rewards_for_grad * log_q).sum(dim=1).float()

            reward_sum = reward_sum + rewards.sum(dim=1)
            reward_log_sum = reward_log_sum + (rewards * rewards.clamp_min(reward_epsilon).log()).sum(dim=1)
            reward_max = torch.maximum(reward_max, rewards.amax(dim=1))

        reward_sum = reward_sum.clamp_min(reward_epsilon)
        loss = -(weighted_logprob_sum / reward_sum.to(weighted_logprob_sum.device)).mean()
        entropy = reward_sum.log() - reward_log_sum / reward_sum
        metrics = {
            "exact_terminal_loss": loss.detach(),
            "exact_catalog_items": torch.tensor(num_items, device=device, dtype=torch.float32),
            "exact_reward_sum_mean": reward_sum.detach().mean(),
            "exact_reward_max_mean": reward_max.detach().mean(),
            "exact_target_top_prob_mean": (reward_max / reward_sum).detach().mean(),
            "exact_target_entropy_mean": entropy.detach().mean(),
            "exact_target_effective_items_mean": entropy.detach().exp().mean(),
        }
        return loss, metrics

    def _catalog_positive_mask(self, batch: Batch, item_tokens: torch.Tensor) -> torch.Tensor:
        target = batch["target_token_ids"].to(item_tokens.device).long()
        target_mask = batch.get("target_token_mask")
        if torch.is_tensor(target_mask):
            target_mask = target_mask.to(item_tokens.device).bool()
        else:
            target_mask = torch.ones_like(target, dtype=torch.bool, device=item_tokens.device)
        matches = item_tokens.unsqueeze(0).eq(target.unsqueeze(1)) | ~target_mask.unsqueeze(1)
        return matches.all(dim=-1)

    def _build_log_z_model(self) -> torch.nn.Module:
        if self.cfg.logz_context == "global":
            return GlobalLogZ(init_value=self.cfg.logz_init)
        if self.cfg.logz_context == "vector":
            return VectorLogZ(
                vector_size=int(self.cfg.logz_vector_size),
                init_value=float(self.cfg.logz_init),
            )
        if self.cfg.logz_context == "semantic_history":
            if self.cfg.logz_vocab_size is None:
                raise ValueError("GFlowConfig.logz_vocab_size is required when " "logz_context='semantic_history'.")
            return SemanticHistoryLogZ(
                SemanticHistoryLogZConfig(
                    vocab_size=int(self.cfg.logz_vocab_size),
                    embedding_dim=int(self.cfg.logz_embedding_dim),
                    hidden_dim=int(self.cfg.logz_hidden_dim),
                    dropout=float(self.cfg.logz_dropout),
                    pad_token_id=int(self.cfg.pad_token_id),
                    init_value=float(self.cfg.logz_init),
                )
            )
        raise ValueError(f"Unknown logZ context: {self.cfg.logz_context}")

    def _sample_trajectories(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None,
    ):
        if self.candidate_sampler is not None:
            return self.candidate_sampler.sample(batch, model=model, step=step)

        if self.cfg.candidate_sampler == "random":
            return self.random_candidate_sampler.sample(batch, model=model)

        if self.cfg.candidate_sampler == "generated":
            refresh = max(int(self.cfg.generated_refresh_every_steps), 1)
            if step is not None and step % refresh != 0:
                trajectories = self.random_candidate_sampler.sample(batch, model=model)
                trajectories.extra["sampler"] = "random_semantic_id_scheduled"
                return trajectories
            return self.generated_candidate_sampler.sample(batch, model=model)

        raise ValueError(f"Unknown candidate sampler: {self.cfg.candidate_sampler}")

    def _candidate_unique_ratio(self, candidate_token_ids: torch.Tensor) -> torch.Tensor:
        flat = candidate_token_ids.reshape(-1, candidate_token_ids.size(-1))
        if flat.numel() == 0:
            return torch.zeros((), device=candidate_token_ids.device)
        unique_count = torch.unique(flat, dim=0).size(0)
        return torch.tensor(
            unique_count / max(flat.size(0), 1),
            device=candidate_token_ids.device,
            dtype=torch.float32,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GFlowMethod":
        return cls(cfg=GFlowConfig.from_dict(data))
