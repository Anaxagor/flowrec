from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.grpo.config import GRPOConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class GRPOMethod(FlowRecMethod):
    """On-policy GRPO with generated semantic-ID groups and frozen-reference KL."""

    method_name = "grpo"

    def __init__(
        self,
        cfg: GRPOConfig | None = None,
        *,
        reward_model: nn.Module | None = None,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
        sft_method: SFTMethod | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GRPOConfig()
        if reward_model is None:
            raise ValueError("GRPOMethod requires a reward model for generated rollouts.")
        self.reward_model = reward_model
        self.reference_model = reference_model or self._build_reference_model(model)
        self._freeze_reference_model()
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
        self._semantic_row_lookup: dict[tuple[int, ...], int] | None = None
        self._semantic_row_lookup_size: tuple[int, int] | None = None

    def train(self, mode: bool = True) -> "GRPOMethod":
        super().train(mode)
        self.reference_model.eval()
        self.reward_model.eval()
        return self

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        sft_out = self._compute_sft_output(model, batch, stage=stage, step=step)
        trajectories = self._sample_trajectories(model, batch)
        rewards = self.reward_model(batch, trajectories).to(trajectories.candidate_token_ids.device).float()
        advantages = self._compute_group_advantages(rewards).detach()

        with torch.no_grad():
            old_logprobs = self._score_candidates(model, batch, trajectories, force_eval=True).detach()

        policy_logprobs = self._score_candidates(model, batch, trajectories, force_eval=True)
        log_ratio = (policy_logprobs - old_logprobs).clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - float(self.cfg.clip_epsilon),
            max=1.0 + float(self.cfg.clip_epsilon),
        )
        pg_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        pg_loss = self._reduce_loss(-pg_objective)

        with torch.no_grad():
            ref_logprobs = self._score_candidates(
                self.reference_model,
                batch,
                trajectories,
                force_eval=True,
            ).detach()
        log_ref_over_policy = (ref_logprobs - policy_logprobs).clamp(min=-20.0, max=20.0)
        kl_per_candidate = torch.exp(log_ref_over_policy) - log_ref_over_policy - 1.0
        kl_loss = self._reduce_loss(kl_per_candidate)

        grpo_loss = pg_loss + float(self.cfg.beta) * kl_loss
        total_loss = grpo_loss
        if sft_out is not None:
            total_loss = total_loss + float(self.cfg.sft_weight) * sft_out.loss

        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "grpo_only_loss": grpo_loss.detach(),
            "policy_loss": pg_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "mean_reward": rewards.detach().mean(),
            "reward_std": rewards.detach().std(unbiased=False),
            "mean_advantage": advantages.detach().mean(),
            "advantage_std": advantages.detach().std(unbiased=False),
            "mean_logprob": policy_logprobs.detach().mean(),
            "old_logprob": old_logprobs.detach().mean(),
            "ref_logprob": ref_logprobs.detach().mean(),
            "ratio_mean": ratio.detach().mean(),
            "ratio_clip_fraction": (
                (ratio.detach() < 1.0 - float(self.cfg.clip_epsilon))
                | (ratio.detach() > 1.0 + float(self.cfg.clip_epsilon))
            ).float().mean(),
            "num_rollouts": trajectories.candidate_token_ids.new_tensor(
                float(trajectories.num_candidates),
                dtype=torch.float32,
            ),
            "candidate_unique_ratio": self._candidate_unique_ratio(trajectories.candidate_token_ids),
            "resolved_row_fraction": self._resolved_row_fraction(trajectories),
        }
        if sft_out is not None:
            metrics["sft_loss"] = sft_out.loss.detach()

        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        if sft_out is not None:
            method_metrics.update(sft_out.metrics)

        extra: dict[str, Any] = {
            "stage": stage,
            "step": step,
            "trajectories": trajectories,
            "rewards": rewards.detach(),
            "advantages": advantages.detach(),
            "policy_logprobs": policy_logprobs.detach(),
            "old_logprobs": old_logprobs.detach(),
            "ref_logprobs": ref_logprobs.detach(),
        }
        if sft_out is not None:
            extra.update(sft_out.extra)

        return MethodOutput(
            loss=total_loss,
            metrics=method_metrics,
            model_output=sft_out.model_output if sft_out is not None else None,
            extra=extra,
        )

    def _compute_sft_output(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str,
        step: int | None,
    ) -> MethodOutput | None:
        should_run = self.cfg.sft_weight > 0
        if stage == "train":
            should_run = should_run or self.cfg.compute_train_metrics
        else:
            should_run = should_run or self.cfg.compute_eval_metrics
        if not should_run:
            return None
        return self.sft_method.compute_loss(model, batch, stage=stage, step=step)

    def _sample_trajectories(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> SemanticTrajectoryBatch:
        desired = max(int(self.cfg.num_rollouts), 1)
        do_sample = bool(self.cfg.rollout_do_sample)
        requested = self.cfg.rollout_num_return_sequences
        if requested is None:
            requested = desired
            if self.cfg.rollout_deduplicate:
                requested = max(requested, desired * max(int(self.cfg.rollout_oversample_factor), 1))
        requested = max(int(requested), desired)
        num_beams = self.cfg.rollout_num_beams
        if num_beams is None:
            num_beams = 1 if do_sample else requested
        if do_sample:
            num_beams = max(int(num_beams), 1)
        else:
            num_beams = max(int(num_beams), requested)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                generated = model.generate(
                    batch,
                    num_beams=num_beams,
                    num_return_sequences=requested,
                    do_sample=do_sample,
                    temperature=float(self.cfg.rollout_temperature),
                    top_k=self.cfg.rollout_top_k,
                    top_p=self.cfg.rollout_top_p,
                    constrain_valid_sequences=self.cfg.rollout_constrain_valid_sequences,
                    exclude_seen_items=self.cfg.rollout_exclude_seen_items,
                )
        finally:
            model.train(was_training)

        if generated.ndim == 2:
            generated = generated.unsqueeze(1)
        selected = self._select_rollouts(generated.long(), desired=desired)
        candidate_mask = torch.ones_like(selected, dtype=torch.bool)
        candidate_rows = self._resolve_candidate_rows(model, selected)
        is_positive = torch.zeros(selected.shape[:2], device=selected.device, dtype=torch.bool)
        return SemanticTrajectoryBatch(
            candidate_token_ids=selected,
            candidate_token_mask=candidate_mask,
            is_positive=is_positive,
            candidate_item_row_ids=candidate_rows,
            extra={"sampler": "grpo_generated"},
        )

    def _select_rollouts(self, generated: torch.Tensor, *, desired: int) -> torch.Tensor:
        bsz, num_generated, sid_len = generated.shape
        if num_generated < desired:
            raise ValueError(
                f"GRPO requested {desired} rollouts but model.generate() returned only {num_generated}."
            )

        selected = torch.empty((bsz, desired, sid_len), device=generated.device, dtype=generated.dtype)
        for batch_idx in range(bsz):
            chosen: list[torch.Tensor] = []
            seen: set[tuple[int, ...]] = set()
            for cand in generated[batch_idx]:
                key = tuple(int(x) for x in cand.tolist())
                if self.cfg.rollout_deduplicate and key in seen:
                    continue
                chosen.append(cand)
                seen.add(key)
                if len(chosen) >= desired:
                    break
            if len(chosen) < desired:
                for cand in generated[batch_idx]:
                    chosen.append(cand)
                    if len(chosen) >= desired:
                        break
            selected[batch_idx] = torch.stack(chosen[:desired], dim=0)
        return selected

    def _resolve_candidate_rows(
        self,
        model: FlowRecModel,
        candidate_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        table = getattr(model, "item_semantic_ids", None)
        if not torch.is_tensor(table) or table.ndim != 2 or table.numel() == 0:
            return torch.full(candidate_token_ids.shape[:2], -1, device=candidate_token_ids.device, dtype=torch.long)

        table_cpu = table.detach().cpu().long()
        table_shape = (int(table_cpu.size(0)), int(table_cpu.size(1)))
        if self._semantic_row_lookup is None or self._semantic_row_lookup_size != table_shape:
            self._semantic_row_lookup = {
                tuple(int(x) for x in row.tolist()): idx for idx, row in enumerate(table_cpu)
            }
            self._semantic_row_lookup_size = table_shape

        rows = torch.full(candidate_token_ids.shape[:2], -1, device="cpu", dtype=torch.long)
        candidates_cpu = candidate_token_ids.detach().cpu().long()
        for batch_idx in range(candidates_cpu.size(0)):
            for cand_idx in range(candidates_cpu.size(1)):
                key = tuple(int(x) for x in candidates_cpu[batch_idx, cand_idx].tolist())
                rows[batch_idx, cand_idx] = self._semantic_row_lookup.get(key, -1)
        return rows.to(candidate_token_ids.device)

    def _score_candidates(
        self,
        model: FlowRecModel,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
        *,
        force_eval: bool = False,
    ) -> torch.Tensor:
        was_training = model.training
        if force_eval:
            model.eval()
        try:
            return model.score_items(
                batch,
                candidate_token_ids=trajectories.candidate_token_ids,
                candidate_token_mask=trajectories.candidate_token_mask,
            ).float()
        finally:
            if force_eval:
                model.train(was_training)

    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_advantages:
            return rewards
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        return (rewards - mean) / std.clamp_min(float(self.cfg.advantage_epsilon))

    def _reduce_loss(self, per_candidate_loss: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_reduction == "mean":
            return per_candidate_loss.mean()
        if self.cfg.loss_reduction == "sum":
            return per_candidate_loss.sum()
        raise ValueError(f"Unknown GRPO loss reduction: {self.cfg.loss_reduction}")

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

    def _resolved_row_fraction(self, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        rows = trajectories.candidate_item_row_ids
        if rows is None:
            return torch.zeros((), device=trajectories.candidate_token_ids.device)
        return rows.ge(0).float().mean()

    def _build_reference_model(self, model: FlowRecModel | None) -> FlowRecModel:
        if model is None:
            raise ValueError("GRPOMethod requires a model to build a frozen reference copy.")

        reference_model = copy.deepcopy(model)
        checkpoint = self.cfg.reference_model_checkpoint
        if checkpoint is None:
            return reference_model

        payload = torch.load(Path(checkpoint), map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        reference_model.load_state_dict(state_dict, strict=True)
        return reference_model

    def _freeze_reference_model(self) -> None:
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        model: FlowRecModel | None = None,
        reward_model: nn.Module | None = None,
        reference_model: FlowRecModel | None = None,
    ) -> "GRPOMethod":
        return cls(
            cfg=GRPOConfig.from_dict(data),
            reward_model=reward_model,
            model=model,
            reference_model=reference_model,
        )
