from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.rewards.similarity import SimilarityRewardModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class MiniOneRecMethod(FlowRecMethod):
    """Minimal GRPO-style fine-tuning for generative recommendation."""

    method_name = "mini-one-rec"

    def __init__(
        self,
        cfg: MiniOneRecConfig | None = None,
        *,
        reward_model: nn.Module | None = None,
        reference_model: FlowRecModel | None = None,
        sft_method: SFTMethod | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or MiniOneRecConfig()
        reward_type = str(self.cfg.reward_type).lower()
        self.reward_model = reward_model
        if self.reward_model is None and reward_type == "similarity":
            self.reward_model = SimilarityRewardModel()
        self.ref_model = reference_model
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
        if self.ref_model is not None:
            self._freeze_reference_model()

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        self._maybe_sync_reference_model(model, stage=stage, step=step)
        sft_out = self.sft_method.compute_loss(model, batch, stage=stage, step=step)

        trajectories = self._sample_trajectories(model, batch)
        rewards, reward_components = self._compute_rewards(batch, trajectories)
        rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
        advantages = self._compute_advantages(rewards)

        policy_token_logprobs = self._candidate_token_logprobs(
            model,
            batch,
            trajectories.candidate_token_ids,
            trajectories.candidate_token_mask,
        )

        if self.ref_model is None:
            raise RuntimeError("Reference model was not initialized.")

        with torch.no_grad():
            ref_token_logprobs = self._candidate_token_logprobs(
                self.ref_model,
                batch,
                trajectories.candidate_token_ids,
                trajectories.candidate_token_mask,
            )

        candidate_mask = trajectories.candidate_token_mask.to(policy_token_logprobs.device).to(policy_token_logprobs.dtype)
        per_token_kl = (
            torch.exp(ref_token_logprobs - policy_token_logprobs)
            - (ref_token_logprobs - policy_token_logprobs)
            - 1.0
        )
        policy_weight = torch.exp(policy_token_logprobs - policy_token_logprobs.detach())
        per_token_objective = policy_weight * advantages.unsqueeze(-1) - float(self.cfg.beta) * per_token_kl
        per_token_loss = -per_token_objective

        if self.cfg.loss_reduction == "sum":
            grpo_loss = (per_token_loss * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)
        elif self.cfg.loss_reduction == "mean":
            sequence_loss = (per_token_loss * candidate_mask).sum(dim=-1) / candidate_mask.sum(dim=-1).clamp_min(1.0)
            grpo_loss = sequence_loss.mean()
        else:
            raise ValueError(f"Unknown MiniOneRec loss_reduction={self.cfg.loss_reduction!r}.")

        sft_loss = sft_out.loss
        loss = float(self.cfg.grpo_weight) * grpo_loss + float(self.cfg.sft_weight) * sft_loss

        metrics = self._build_metrics(
            loss=loss,
            grpo_loss=grpo_loss,
            sft_loss=sft_loss,
            rewards=rewards,
            advantages=advantages,
            trajectories=trajectories,
            per_token_kl=per_token_kl,
            candidate_mask=candidate_mask,
            reward_components=reward_components,
        )
        if stage == "train" and not self.cfg.compute_train_metrics:
            metrics = {"loss": loss.detach()}
            sft_metrics: dict[str, Any] = {}
        elif stage != "train" and not self.cfg.compute_eval_metrics:
            metrics = {"loss": loss.detach()}
            sft_metrics = {}
        else:
            sft_metrics = dict(sft_out.metrics)

        return MethodOutput(
            loss=loss,
            metrics=self._merge_metrics(
                prefix_metrics(metrics, prefix=self.cfg.metric_prefix),
                sft_metrics,
            ),
            model_output=sft_out.model_output,
            extra={
                **sft_out.extra,
                "stage": stage,
                "step": step,
                "trajectories": trajectories,
                "rewards": rewards,
                "advantages": advantages,
                # Unweighted components for gradient-norm weight calibration.
                LOSS_COMPONENTS_KEY: {"sft": sft_loss, "grpo": grpo_loss},
            },
        )

    def _maybe_sync_reference_model(
        self,
        model: FlowRecModel,
        *,
        stage: str,
        step: int | None,
    ) -> None:
        if self.ref_model is None:
            self.ref_model = copy.deepcopy(model)
            self._freeze_reference_model()
            return

        sync_every = self.cfg.sync_ref_every_steps
        if stage != "train" or sync_every is None or step is None or step <= 0:
            return
        if sync_every < 1:
            raise ValueError("sync_ref_every_steps must be >= 1 when provided.")
        if step % sync_every != 0:
            return

        self.ref_model.load_state_dict(model.state_dict())
        self._freeze_reference_model()

    def _freeze_reference_model(self) -> None:
        if self.ref_model is None:
            return
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _sample_trajectories(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> SemanticTrajectoryBatch:
        num_generations = int(self.cfg.num_generations)
        if num_generations < 2:
            raise ValueError("mini-one-rec requires num_generations >= 2.")

        was_training = model.training
        model.eval()
        try:
            candidate_token_ids = model.generate(
                batch,
                num_beams=max(int(self.cfg.sample_num_beams), 1),
                num_return_sequences=num_generations,
                max_new_tokens=self.cfg.sample_max_new_tokens,
                constrain_valid_sequences=bool(self.cfg.sample_constrain_valid_sequences),
                exclude_seen_items=bool(self.cfg.sample_exclude_seen_items),
                do_sample=bool(self.cfg.rollout_do_sample),
                temperature=float(self.cfg.sample_temperature),
                top_k=self.cfg.sample_top_k,
                top_p=self.cfg.sample_top_p,
            )
        finally:
            model.train(was_training)

        if candidate_token_ids.ndim != 3:
            raise ValueError(
                "model.generate() must return [B, num_generations, sid_len] for mini-one-rec. "
                f"Got shape {tuple(candidate_token_ids.shape)}."
            )

        target_token_ids = batch.get("target_token_ids")
        if target_token_ids is None or not torch.is_tensor(target_token_ids):
            raise KeyError("mini-one-rec requires batch['target_token_ids'].")
        target_token_ids = target_token_ids.to(candidate_token_ids.device).long()
        target_token_mask = batch.get("target_token_mask")
        if target_token_mask is None or not torch.is_tensor(target_token_mask):
            target_token_mask = torch.ones_like(target_token_ids, dtype=torch.bool)
        else:
            target_token_mask = target_token_mask.to(candidate_token_ids.device).bool()

        candidate_token_mask = torch.ones_like(candidate_token_ids, dtype=torch.bool)
        forced_gt = torch.zeros(
            candidate_token_ids.size(0),
            candidate_token_ids.size(1),
            device=candidate_token_ids.device,
            dtype=torch.bool,
        )

        if self.cfg.force_ground_truth_in_group:
            candidate_token_ids[:, 0, :] = target_token_ids
            candidate_token_mask[:, 0, :] = target_token_mask
            forced_gt[:, 0] = True

        candidate_scores = None
        if bool(self.cfg.rollout_do_sample):
            candidate_scores = model.score_items(
                batch,
                candidate_token_ids=candidate_token_ids,
                candidate_token_mask=candidate_token_mask,
            )
            sort_idx = candidate_scores.argsort(dim=1, descending=True)
            gather_idx = sort_idx.unsqueeze(-1).expand(-1, -1, candidate_token_ids.size(-1))
            candidate_token_ids = torch.gather(candidate_token_ids, dim=1, index=gather_idx)
            candidate_token_mask = torch.gather(candidate_token_mask, dim=1, index=gather_idx)
            forced_gt = torch.gather(forced_gt, dim=1, index=sort_idx)
            candidate_scores = torch.gather(candidate_scores, dim=1, index=sort_idx)

        is_positive = candidate_token_ids.eq(target_token_ids.unsqueeze(1)).all(dim=-1)
        candidate_item_row_ids = self._resolve_candidate_item_rows(model, candidate_token_ids)

        return SemanticTrajectoryBatch(
            candidate_token_ids=candidate_token_ids.long(),
            candidate_token_mask=candidate_token_mask,
            is_positive=is_positive,
            candidate_item_row_ids=candidate_item_row_ids,
            extra={
                "sampler": "policy_sampled_semantic_id",
                "forced_gt": forced_gt,
                "candidate_scores": candidate_scores,
                "rank_source": "policy_score" if bool(self.cfg.rollout_do_sample) else "generation_order",
            },
        )

    def _resolve_candidate_item_rows(
        self,
        model: FlowRecModel,
        candidate_token_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        item_semantic_ids = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(item_semantic_ids) or item_semantic_ids.numel() == 0:
            return None

        catalog = item_semantic_ids.to(candidate_token_ids.device).long()
        if catalog.ndim != 2 or catalog.size(-1) != candidate_token_ids.size(-1):
            return None

        flat = candidate_token_ids.reshape(-1, candidate_token_ids.size(-1)).long()
        matches = flat.unsqueeze(1).eq(catalog.unsqueeze(0)).all(dim=-1)
        has_match = matches.any(dim=1)
        rows = torch.full((flat.size(0),), -1, device=flat.device, dtype=torch.long)
        if has_match.any():
            rows[has_match] = matches[has_match].float().argmax(dim=1).long()
        return rows.view(candidate_token_ids.size(0), candidate_token_ids.size(1))

    def _compute_rewards(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward_type = str(self.cfg.reward_type).lower()
        if reward_type == "rule_rank":
            return self._compute_rule_rank_rewards(batch, trajectories)

        if self.reward_model is None:
            raise ValueError(
                f"mini-one-rec reward_type={self.cfg.reward_type!r} requires an external reward model."
            )

        rewards = self.reward_model(batch, trajectories).to(trajectories.candidate_token_ids.device).float()
        zero = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        return rewards, {
            "rule_reward_mean": zero,
            "rank_reward_mean": zero,
        }

    def _compute_rule_rank_rewards(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        candidate_token_ids = trajectories.candidate_token_ids
        target_token_ids = batch.get("target_token_ids")
        if target_token_ids is None or not torch.is_tensor(target_token_ids):
            raise KeyError("mini-one-rec rule_rank reward requires batch['target_token_ids'].")
        target_token_ids = target_token_ids.to(candidate_token_ids.device).long()

        rule_reward = candidate_token_ids.eq(target_token_ids.unsqueeze(1)).all(dim=-1).float()

        num_candidates = candidate_token_ids.size(1)
        rank = torch.arange(
            1,
            num_candidates + 1,
            device=candidate_token_ids.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(candidate_token_ids.size(0), -1)
        rank_penalty = 1.0 / torch.log(rank + 1.0)
        negative_rank_reward = -float(self.cfg.rank_reward_weight) * (1.0 - rule_reward) * rank_penalty

        if self.cfg.normalize_rank_reward:
            negative_mask = (1.0 - rule_reward)
            penalty_sum = (rank_penalty * negative_mask).sum(dim=1, keepdim=True).clamp_min(1e-12)
            num_negative = negative_mask.sum(dim=1, keepdim=True)
            negative_rank_reward = (
                -float(self.cfg.rank_reward_weight)
                * negative_mask
                * rank_penalty
                / penalty_sum
                * num_negative
            )

        rewards = rule_reward + negative_rank_reward
        return rewards, {
            "rule_reward_mean": rule_reward.mean(),
            "rank_reward_mean": negative_rank_reward.mean(),
        }

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.ndim != 2:
            raise ValueError(f"Rewards must have shape [B, G], got {tuple(rewards.shape)}.")
        if not self.cfg.normalize_rewards:
            return rewards
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        return (rewards - mean) / std.clamp_min(float(self.cfg.advantage_epsilon))

    def _candidate_token_logprobs(
        self,
        model: FlowRecModel,
        batch: Batch,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_candidates, sid_len = candidate_token_ids.shape
        flat_candidates = candidate_token_ids.reshape(bsz * num_candidates, sid_len)
        flat_mask = candidate_token_mask.reshape(bsz * num_candidates, sid_len)
        expanded_batch = self._expand_batch(batch, repeats=num_candidates)
        token_logprobs = model.token_logprobs(
            expanded_batch,
            target_token_ids=flat_candidates,
            target_token_mask=flat_mask,
        )
        return token_logprobs.reshape(bsz, num_candidates, sid_len)

    def _expand_batch(self, batch: Batch, *, repeats: int) -> dict[str, Any]:
        expanded: dict[str, Any] = {}
        batch_size = self._infer_batch_size(batch)
        for key, value in batch.items():
            if batch_size is not None and torch.is_tensor(value) and value.shape[:1] == (batch_size,):
                expanded[key] = value.repeat_interleave(repeats, dim=0)
            else:
                expanded[key] = value
        return expanded

    def _infer_batch_size(self, batch: Batch) -> int | None:
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim >= 1:
                return int(value.size(0))
        return None

    def _build_metrics(
        self,
        *,
        loss: torch.Tensor,
        grpo_loss: torch.Tensor,
        sft_loss: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        trajectories: SemanticTrajectoryBatch,
        per_token_kl: torch.Tensor,
        candidate_mask: torch.Tensor,
        reward_components: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        reward_std = rewards.std(dim=1, unbiased=False).mean()
        advantage_std = advantages.std(dim=1, unbiased=False).mean()
        mean_kl = (per_token_kl * candidate_mask).sum(dim=-1) / candidate_mask.sum(dim=-1).clamp_min(1.0)
        sample_hit_rate = trajectories.is_positive.float().mean()
        group_hit_rate = trajectories.is_positive.any(dim=1).float().mean()

        flat_candidates = trajectories.candidate_token_ids.reshape(-1, trajectories.candidate_token_ids.size(-1))
        unique_ratio = torch.tensor(
            torch.unique(flat_candidates, dim=0).size(0) / max(flat_candidates.size(0), 1),
            device=trajectories.candidate_token_ids.device,
            dtype=torch.float32,
        )
        forced_gt = trajectories.extra.get("forced_gt")
        forced_gt_fraction = torch.zeros(
            (),
            device=trajectories.candidate_token_ids.device,
            dtype=torch.float32,
        )
        if torch.is_tensor(forced_gt):
            forced_gt_fraction = forced_gt.float().mean()

        return {
            "loss": loss.detach(),
            "grpo_loss": grpo_loss.detach(),
            "sft_loss": sft_loss.detach(),
            "sft_loss_weight": torch.tensor(
                float(self.cfg.sft_weight),
                device=loss.device,
                dtype=loss.dtype,
            ).detach(),
            "grpo_loss_weight": torch.tensor(
                float(self.cfg.grpo_weight),
                device=loss.device,
                dtype=loss.dtype,
            ).detach(),
            "reward_mean": rewards.mean().detach(),
            "reward_std": reward_std.detach(),
            "advantage_std": advantage_std.detach(),
            "sample_hit_rate": sample_hit_rate.detach(),
            "group_hit_rate": group_hit_rate.detach(),
            "candidate_unique_ratio": unique_ratio.detach(),
            "forced_gt_fraction": forced_gt_fraction.detach(),
            "mean_kl": mean_kl.mean().detach(),
            "mean_completion_length": candidate_mask.sum(dim=-1).float().mean().detach(),
            "rule_reward_mean": reward_components["rule_reward_mean"].detach(),
            "rank_reward_mean": reward_components["rank_reward_mean"].detach(),
        }

    def _merge_metrics(
        self,
        grpo_metrics: dict[str, Any],
        sft_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(grpo_metrics)
        merged.update(sft_metrics)
        return merged

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MiniOneRecMethod":
        return cls(cfg=MiniOneRecConfig.from_dict(data))
