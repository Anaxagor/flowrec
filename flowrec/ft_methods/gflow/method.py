from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON, GFlowConfig
from flowrec.ft_methods.gflow.exact_flows import CatalogPrefixFlowIndex
from flowrec.ft_methods.gflow.losses.detailed_balance import detailed_balance_loss
from flowrec.ft_methods.gflow.losses.subtrajectory_balance import (
    subtrajectory_balance_loss,
)
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class CandidateSampler(Protocol):
    def sample(
        self,
        batch: Batch,
        *,
        model: FlowRecModel,
        step: int | None,
    ) -> SemanticTrajectoryBatch: ...


@dataclass(frozen=True, slots=True)
class _RewardCacheShard:
    start: int
    stop: int
    full_rewards: torch.Tensor
    reward_sum: torch.Tensor
    target_rows: torch.Tensor


class _PrecomputedRewardCache:
    def __init__(
        self,
        path: str | Path,
        *,
        power: float = 1.0,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Reward cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.power = float(power)
        if self.power <= 0.0:
            raise ValueError("reward_cache_power must be positive.")
        self.shards, self._num_examples, self._num_items = self._load_shards()

    @property
    def num_examples(self) -> int:
        return self._num_examples

    def select_candidates(
        self,
        indices: torch.Tensor,
        candidate_rows: torch.Tensor,
        device: torch.device,
        *,
        include_full: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        idx = self._checked_indices(indices)
        rows = candidate_rows.detach().cpu().long()
        if rows.ndim != 2 or rows.size(0) != idx.numel():
            raise ValueError("candidate_rows must have shape [B, N] matching indices.")

        selected = torch.empty(rows.shape, dtype=torch.float32)
        reward_sum = torch.empty(idx.numel(), dtype=torch.float32)
        transformed_full = (
            torch.empty((idx.numel(), self._num_items), dtype=torch.float32)
            if include_full
            else None
        )
        covered = torch.zeros(idx.numel(), dtype=torch.bool)

        for shard in self.shards:
            positions = ((idx >= shard.start) & (idx < shard.stop)).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            local_idx = idx.index_select(0, positions) - shard.start
            full = shard.full_rewards.index_select(0, local_idx)
            target_rows = shard.target_rows.index_select(0, local_idx).long()
            shard_rows = rows.index_select(0, positions)
            shard_rows = torch.where(shard_rows.ge(0), shard_rows, target_rows.unsqueeze(1))
            if shard_rows.numel() and (
                int(shard_rows.min()) < 0 or int(shard_rows.max()) >= self._num_items
            ):
                raise IndexError("Candidate item row is outside the cached catalogue.")
            selected.index_copy_(
                0,
                positions,
                self._transform_rewards(full.gather(1, shard_rows)),
            )
            reward_sum.index_copy_(
                0,
                positions,
                shard.reward_sum.index_select(0, local_idx).float(),
            )
            if transformed_full is not None:
                transformed_full.index_copy_(0, positions, self._transform_rewards(full))
            covered[positions] = True

        if not bool(covered.all()):
            missing = idx[~covered].tolist()
            raise IndexError(f"Reward cache has no shard for example indices: {missing[:8]}")

        return (
            selected.to(device=device, non_blocking=True),
            reward_sum.to(device=device, non_blocking=True),
            transformed_full.to(device=device, non_blocking=True)
            if transformed_full is not None
            else None,
        )

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

    def _load_shards(self) -> tuple[list[_RewardCacheShard], int, int]:
        shards: list[_RewardCacheShard] = []
        reward_sum_key = self._power_sum_key(self.power) if self.power != 1.0 else "reward_sum"
        num_items: int | None = None
        max_stop = 0
        for shard in self.manifest.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.is_absolute():
                shard_path = self.path / shard_path
            payload = torch.load(
                shard_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            for key in ("full_rewards", reward_sum_key, "target_semantic_row_ids"):
                if key not in payload:
                    raise KeyError(f"Reward cache shard lacks {key!r}: {shard_path}")
            full_rewards = payload["full_rewards"]
            reward_sum = payload[reward_sum_key].reshape(-1)
            target_rows = payload["target_semantic_row_ids"].reshape(-1)
            if full_rewards.ndim != 2:
                raise ValueError(f"full_rewards must have shape [N, num_items]: {shard_path}")
            start = int(shard.get("example_start", payload.get("example_start", 0)))
            count = int(shard.get("num_examples", payload.get("num_examples", full_rewards.size(0))))
            stop = start + count
            if full_rewards.size(0) != count or reward_sum.numel() != count or target_rows.numel() != count:
                raise ValueError(f"Reward cache shard metadata does not match tensor rows: {shard_path}")
            if num_items is None:
                num_items = int(full_rewards.size(1))
            elif int(full_rewards.size(1)) != num_items:
                raise ValueError(f"Reward cache shards have inconsistent catalogue widths: {shard_path}")
            shards.append(
                _RewardCacheShard(
                    start=start,
                    stop=stop,
                    full_rewards=full_rewards,
                    reward_sum=reward_sum,
                    target_rows=target_rows,
                )
            )
            max_stop = max(max_stop, stop)
        if not shards or num_items is None:
            raise ValueError(f"Reward cache {self.path} has no shards.")
        num_examples = int(self.manifest.get("num_examples", max_stop))
        return shards, num_examples, num_items

    def _transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.float().clamp_min(1e-6)
        if self.power == 1.0:
            return rewards
        return rewards.pow(self.power).clamp_min(1e-6)


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
        candidate_sampler: CandidateSampler,
        sft_method: SFTMethod,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GFlowConfig()
        if not self.cfg.reward_cache_path:
            raise ValueError("GFlow training requires reward_cache_path.")
        self._reward_cache: _PrecomputedRewardCache | None = None
        self._exact_flow_index: CatalogPrefixFlowIndex | None = None

        self.candidate_sampler = candidate_sampler
        self.sft_method = sft_method

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

        zero = sft_out.loss.new_zeros(())
        gflow_loss = zero
        gflow_metrics: dict[str, torch.Tensor] = {}
        trajectories = None
        rewards = None
        sequence_logprobs = None
        token_logprobs = None
        log_z = zero
        candidate_weights = None

        compute_sampled_gflow = stage == "train"
        if compute_sampled_gflow:
            (
                trajectories,
                rewards,
                log_z,
                cache_tb_metrics,
            ) = self._precomputed_reward_cache_tb_inputs(model, batch, step=step)
            if self.cfg.tb_loss_weighting == "reward":
                candidate_weights = rewards

            if self.cfg.objective == "tb":
                item_scorer = getattr(model, "score_items_valid_actions", None)
                if not callable(item_scorer):
                    item_scorer = model.score_items
                sequence_logprobs = item_scorer(
                    candidate_scoring_batch,
                    candidate_token_ids=trajectories.candidate_token_ids,
                    candidate_token_mask=trajectories.candidate_token_mask,
                )
                gflow_loss, gflow_metrics = trajectory_balance_loss(
                    sequence_logprobs=sequence_logprobs,
                    rewards=rewards,
                    log_z=log_z,
                    reward_epsilon=GFLOW_REWARD_EPSILON,
                    candidate_weights=candidate_weights,
                    normalize_candidate_weights=True,
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
                    reward_epsilon=GFLOW_REWARD_EPSILON,
                    candidate_weights=candidate_weights,
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
                gflow_loss, gflow_metrics = subtrajectory_balance_loss(
                    token_logprobs=token_logprobs,
                    prefix_log_flows=prefix_log_flows,
                    candidate_token_mask=trajectories.candidate_token_mask,
                    rewards=rewards,
                    reward_epsilon=GFLOW_REWARD_EPSILON,
                    candidate_weights=candidate_weights,
                    subtb_lambda=self.cfg.subtb_lambda,
                    max_subtrajectory_length=self.cfg.subtb_max_length,
                )
                gflow_metrics.update(cache_tb_metrics)
            else:
                raise ValueError(f"Unknown GFlow objective: {self.cfg.objective}")
        else:
            gflow_metrics = {
                "reward_cache_tb_active": torch.tensor(0.0, device=sft_out.loss.device),
                "fixed_log_z_active": torch.tensor(0.0, device=sft_out.loss.device),
            }

        loss = float(self.cfg.sft_weight) * sft_out.loss + float(self.cfg.gflow_weight) * gflow_loss

        if trajectories is not None:
            metric_device = trajectories.candidate_token_ids.device
            num_candidates = trajectories.num_candidates
            num_negative_samples = float(
                (~trajectories.is_positive).sum(dim=1).float().mean().item()
            )
            candidate_unique_ratio = self._candidate_unique_ratio(trajectories.candidate_token_ids)
            score_sampler_active = (
                1.0
                if str(trajectories.extra.get("sampler", "")).endswith("_semantic_id")
                and trajectories.extra.get("sampler") != "random_semantic_id"
                else 0.0
            )
            easy_negative_active = 1.0 if trajectories.extra.get("sampler_phase") == "easy" else 0.0
        else:
            metric_device = sft_out.loss.device
            num_candidates = 0
            num_negative_samples = 0
            candidate_unique_ratio = zero
            score_sampler_active = 0.0
            easy_negative_active = 0.0

        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "sft_loss": sft_out.loss.detach(),
            "gflow_loss": gflow_loss.detach(),
            "log_z_mean": log_z.detach().mean(),
            "log_z_std": log_z.detach().reshape(-1).std(unbiased=False),
            "num_candidates": torch.tensor(num_candidates, device=metric_device, dtype=torch.float32),
            "num_negative_samples": torch.tensor(num_negative_samples, device=metric_device, dtype=torch.float32),
            "candidate_unique_ratio": candidate_unique_ratio.detach(),
            "score_sampler_active": torch.tensor(score_sampler_active, device=metric_device, dtype=torch.float32),
            "easy_negative_active": torch.tensor(easy_negative_active, device=metric_device, dtype=torch.float32),
            "reward_weighting_active": torch.tensor(
                float(self.cfg.tb_loss_weighting == "reward"),
                device=metric_device,
                dtype=torch.float32,
            ),
        }
        metrics.update(gflow_metrics)

        method_metrics = prefix_metrics(metrics, prefix=self.method_name)
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

    def _precomputed_reward_cache_tb_inputs(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None,
    ) -> tuple[
        SemanticTrajectoryBatch,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if "example_indices" not in batch:
            raise KeyError("Reward-cache trajectory balance requires batch['example_indices'].")
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
            raise ValueError("Reward-cache trajectory balance requires model.item_semantic_ids.")

        trajectories = self._sample_trajectories(model, batch, step=step)
        rows = trajectories.candidate_item_row_ids
        if not torch.is_tensor(rows):
            raise ValueError("Cached rewards require candidate_item_row_ids from the sampler.")
        if bool((rows.lt(0) & ~trajectories.is_positive).any()):
            raise ValueError("The sampler returned a negative candidate without a catalogue row.")

        device = rows.device
        cache = self._get_reward_cache()
        needs_exact_flows = self.cfg.objective in {"db", "subtb"}
        rewards, reward_sum, full_rewards = cache.select_candidates(
            batch["example_indices"],
            rows,
            device=device,
            include_full=needs_exact_flows,
        )
        rewards = rewards.to(device=device, dtype=torch.float32).clamp_min(GFLOW_REWARD_EPSILON)
        log_z = reward_sum.to(device=device, dtype=torch.float32).clamp_min(GFLOW_REWARD_EPSILON).log()
        trajectories.extra["reward_cache_active"] = True
        if full_rewards is not None:
            trajectories.extra["cache_full_rewards"] = full_rewards
        metrics = {
            "reward_cache_tb_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "fixed_log_z_active": torch.tensor(1.0, device=device, dtype=torch.float32),
            "reward_cache_candidate_count": torch.tensor(trajectories.num_candidates, device=device, dtype=torch.float32),
            "reward_cache_tb_reward_sum_mean": reward_sum.detach().mean(),
        }
        return trajectories, rewards, log_z, metrics

    def _candidate_scoring_batch(
        self,
        batch: Batch,
        model_output: Any,
    ) -> Batch:
        extra = getattr(model_output, "extra", None)
        if not isinstance(extra, dict):
            raise ValueError("GFlow candidate scoring requires model output with encoder state metadata.")
        encoder_hidden = extra.get("encoder_last_hidden_state")
        attention_mask = extra.get("encoder_attention_mask")
        if not torch.is_tensor(encoder_hidden) or not torch.is_tensor(attention_mask):
            raise ValueError(
                "GFlow candidate scoring requires the model to expose "
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
        full_rewards = trajectories.extra.get("cache_full_rewards")
        if not torch.is_tensor(full_rewards):
            raise ValueError(
                "Analytical DB/SubTB flows require full-catalogue cached rewards."
            )

        token_logprobs = self._candidate_token_logprobs(model, batch, trajectories)
        index = self._get_exact_flow_index(model)
        prefix_log_flows = index.prefix_log_flows(
            candidate_tokens,
            full_rewards,
            reward_epsilon=GFLOW_REWARD_EPSILON,
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

        token_scorer = getattr(model, "score_item_tokens_valid_actions", None)
        if not callable(token_scorer):
            token_scorer = getattr(model, "score_item_tokens", None)
        if callable(token_scorer):
            stacked = token_scorer(
                batch,
                candidate_token_ids=candidate_tokens,
                candidate_token_mask=candidate_mask,
            )
            return stacked.masked_fill(~candidate_mask.to(stacked.device).bool(), 0.0)

        # Generic fallback for models without token-level candidate scoring.
        seq_len = int(candidate_tokens.size(-1))
        positions = torch.arange(seq_len, device=candidate_tokens.device).view(1, 1, seq_len)
        prefix_scores: list[torch.Tensor] = []
        for end in range(1, seq_len + 1):
            prefix_mask = candidate_mask & positions.lt(end)
            prefix_scores.append(
                model.score_items(
                    batch,
                    candidate_token_ids=candidate_tokens,
                    candidate_token_mask=prefix_mask,
                )
            )

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
                power=float(self.cfg.reward_cache_power),
            )
        return self._reward_cache

    def _sample_trajectories(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None,
    ):
        return self.candidate_sampler.sample(batch, model=model, step=step)

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
