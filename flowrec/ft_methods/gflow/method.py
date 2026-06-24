from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flowrec.candidates.generated import GeneratedSemanticCandidateConfig, GeneratedSemanticIDCandidateSampler
from flowrec.candidates.random import RandomSemanticCandidateConfig, RandomSemanticIDCandidateSampler
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.flow import (
    GlobalLogZ,
    SemanticHistoryLogZ,
    SemanticHistoryLogZConfig,
    VectorLogZ,
)
#from flowrec.ft_methods.gflow.losses.detailed_balance import detailed_balance_loss
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
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Reward cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        self.rows, self.probs, self.rewards, self.reward_sum, self.mass, self.target_in_top = self._load_tensors()

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._checked_indices(indices)
        rows = self.rows.index_select(0, idx).to(device=device, non_blocking=True).long()
        rewards = self.rewards.index_select(0, idx).to(device=device, non_blocking=True).float()
        reward_sum = self.reward_sum.index_select(0, idx).to(device=device, non_blocking=True).float()
        candidate_weights = self.probs.index_select(0, idx).to(device=device, non_blocking=True).float()
        mass = self.mass.index_select(0, idx).to(device=device, non_blocking=True).float()
        target_in_top = self.target_in_top.index_select(0, idx).to(device=device, non_blocking=True).float()
        return rows, rewards, reward_sum, candidate_weights, mass, target_in_top

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

    def _load_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows_parts: list[torch.Tensor] = []
        probs_parts: list[torch.Tensor] = []
        rewards_parts: list[torch.Tensor] = []
        reward_sum_parts: list[torch.Tensor] = []
        mass_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
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
        )

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
        key = "top_reward_mass" if self.top_m == int(payload["top_item_row_ids"].size(1)) else f"top_reward_mass_{self.top_m}"
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
        key = "target_in_top_m" if self.top_m == int(payload["top_item_row_ids"].size(1)) else f"target_in_top_{self.top_m}"
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
        if self.cfg.tb_logz_source == "reward_cache" and self.cfg.objective != "tb":
            raise ValueError("tb_logz_source='reward_cache' is only supported for objective='tb'.")
        if self.cfg.tb_loss_weighting not in {"uniform", "reward"}:
            raise ValueError("tb_loss_weighting must be uniform or reward.")
        self.log_z_model = torch.nn.Identity() if self.cfg.tb_logz_source == "reward_cache" else self._build_log_z_model()
        self._reward_cache: _PrecomputedRewardCache | None = None

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

        zero = sft_out.loss.new_zeros(())
        gflow_loss = zero
        gflow_metrics: dict[str, torch.Tensor] = {}
        trajectories = None
        rewards = None
        sequence_logprobs = None
        log_z = zero
        tb_candidate_weights = None

        use_reward_cache_tb = self.cfg.tb_logz_source == "reward_cache"
        compute_sampled_gflow = (
            (not bool(self.cfg.exact_terminal_distribution) or float(self.cfg.gflow_weight) != 0.0)
            and (not use_reward_cache_tb or stage == "train")
        )
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
                log_z = self.log_z_model(batch)
                cache_tb_metrics = {}

            sequence_logprobs = model.score_items(
                batch,
                candidate_token_ids=trajectories.candidate_token_ids,
                candidate_token_mask=trajectories.candidate_token_mask,
            )

            if self.cfg.objective == "tb":
                if self.cfg.tb_loss_weighting == "reward" and tb_candidate_weights is None:
                    denom = rewards.sum(dim=1, keepdim=True).clamp_min(float(self.cfg.reward_epsilon))
                    tb_candidate_weights = rewards / denom
                gflow_loss, gflow_metrics = trajectory_balance_loss(
                    sequence_logprobs=sequence_logprobs,
                    rewards=rewards,
                    log_z=log_z,
                    reward_epsilon=self.cfg.reward_epsilon,
                    candidate_weights=tb_candidate_weights if self.cfg.tb_loss_weighting == "reward" else None,
                )
                gflow_metrics.update(cache_tb_metrics)
            elif self.cfg.objective == "db":
                gflow_loss, gflow_metrics = detailed_balance_loss()
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
            "generated_sampler_active": torch.tensor(generated_sampler_active, device=metric_device, dtype=torch.float32),
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

        return MethodOutput(
            loss=loss,
            metrics=method_metrics,
            model_output=sft_out.model_output,
            extra=extra,
        )

    def _precomputed_reward_cache_tb_inputs(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> tuple[SemanticTrajectoryBatch, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if "example_indices" not in batch:
            raise KeyError("Reward-cache trajectory balance requires batch['example_indices'].")
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
            raise ValueError("Reward-cache trajectory balance requires model.item_semantic_ids.")

        device = batch["example_indices"].device
        cache = self._get_reward_cache()
        rows, rewards, reward_sum, candidate_weights, top_mass, target_in_top = cache.select_tb(
            batch["example_indices"], device=device
        )
        item_tokens = item_semantic_ids.to(device).long()
        safe_rows = rows.clamp_min(0).clamp_max(max(int(item_tokens.size(0)) - 1, 0))
        candidate_tokens = item_tokens[safe_rows]
        candidate_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
        is_positive = self._cached_positive_mask(batch, candidate_tokens)
        rewards = rewards.to(device=device, dtype=torch.float32).clamp_min(float(self.cfg.reward_epsilon))
        candidate_weights = candidate_weights.to(device=device, dtype=torch.float32).clamp_min(0.0)
        log_z = reward_sum.to(device=device, dtype=torch.float32).clamp_min(float(self.cfg.reward_epsilon)).log()
        trajectories = SemanticTrajectoryBatch(
            candidate_token_ids=candidate_tokens,
            candidate_token_mask=candidate_mask,
            is_positive=is_positive,
            candidate_item_row_ids=safe_rows,
            extra={"sampler": "reward_cache_tb", "sampler_phase": "fixed_logz"},
        )
        metrics = {
            "reward_cache_tb_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "fixed_log_z_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "reward_cache_tb_top_m": torch.tensor(trajectories.num_candidates, device=device, dtype=torch.float32),
            "reward_cache_tb_reward_sum_mean": reward_sum.detach().mean(),
            "reward_cache_tb_top_mass_mean": top_mass.detach().mean(),
            "reward_cache_tb_weight_mass_mean": candidate_weights.detach().sum(dim=1).mean(),
            "reward_cache_tb_target_in_top_mean": target_in_top.detach().mean(),
        }
        return trajectories, rewards, log_z, candidate_weights, metrics

    def _cached_positive_mask(self, batch: Batch, candidate_tokens: torch.Tensor) -> torch.Tensor:
        target = batch.get("target_token_ids")
        if not torch.is_tensor(target):
            return torch.zeros(candidate_tokens.shape[:2], device=candidate_tokens.device, dtype=torch.bool)
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
            candidate_rows = (
                torch.arange(start, end, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
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
                raise ValueError(
                    "GFlowConfig.logz_vocab_size is required when "
                    "logz_context='semantic_history'."
                )
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
