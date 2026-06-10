from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from flowrec.candidates.random import RandomSemanticCandidateConfig, RandomSemanticIDCandidateSampler
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.grpo.config import GRPOConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel



class GRPOMethod(FlowRecMethod):
    """Group Relative Policy Optimization for semantic-ID recommendation.

    The group for each user is built as [chosen positive + sampled negatives].
    Rewards are normalized inside each group, which removes the need for a value model.
    """

    method_name = "grpo"

    def __init__(
        self,
        cfg: GRPOConfig | None = None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
        sft_method: SFTMethod | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or GRPOConfig()
        self.negative_sampler = RandomSemanticIDCandidateSampler(
            RandomSemanticCandidateConfig(
                num_negative_samples=max(int(self.cfg.num_negatives), 1),
                include_positive=False,
                avoid_positive=True,
            )
        )

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
                generation_constrained=self.cfg.generation_constrained,
            )
        )

    def train(self, mode: bool = True) -> "GRPOMethod":
        super().train(mode)
        self.reference_model.eval()
        return self

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        chosen_ids, chosen_mask = self._resolve_chosen(batch)
        negative_ids, negative_mask = self._sample_negatives(batch, model=model)

        group_ids = torch.cat([chosen_ids[:, None, :], negative_ids], dim=1)
        group_mask = torch.cat([chosen_mask[:, None, :], negative_mask], dim=1)

        batch_size, group_size, seq_len = group_ids.shape
        flat_ids = group_ids.reshape(batch_size * group_size, seq_len)
        flat_mask = group_mask.reshape(batch_size * group_size, seq_len)
        repeated_batch = self._repeat_batch_for_group(batch, group_size=group_size, batch_size=batch_size)

        policy_logp = model.sequence_logprobs(
            repeated_batch,
            target_token_ids=flat_ids,
            target_token_mask=flat_mask,
        ).float().reshape(batch_size, group_size)

        self.reference_model.eval()
        with torch.no_grad():
            ref_logp = self.reference_model.sequence_logprobs(
                repeated_batch,
                target_token_ids=flat_ids,
                target_token_mask=flat_mask,
            ).to(policy_logp.device).float().reshape(batch_size, group_size)

        rewards = self._build_rewards(
            batch_size=batch_size,
            group_size=group_size,
            device=policy_logp.device,
            dtype=policy_logp.dtype,
        )
        advantages = self._compute_group_advantages(rewards).detach()

        # In this offline adaptation we use the frozen reference model as the old policy.
        log_ratio = policy_logp - ref_logp.detach()
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - float(self.cfg.clip_epsilon),
            max=1.0 + float(self.cfg.clip_epsilon),
        )

        pg_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        pg_loss_per_candidate = -pg_objective
        pg_loss = self._reduce_loss(pg_loss_per_candidate)

        # Sequence-level approximation of the GRPO KL estimator:
        # exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1.
        log_ref_over_policy = (ref_logp.detach() - policy_logp).clamp(min=-20.0, max=20.0)
        kl_per_candidate = torch.exp(log_ref_over_policy) - log_ref_over_policy - 1.0
        kl_loss = self._reduce_loss(kl_per_candidate)

        grpo_loss = pg_loss + float(self.cfg.beta) * kl_loss

        sft_out: MethodOutput | None = None
        total_loss = grpo_loss
        if self.cfg.sft_weight > 0:
            sft_out = self.sft_method.compute_loss(model, batch, stage=stage, step=step)
            total_loss = total_loss + float(self.cfg.sft_weight) * sft_out.loss

        chosen_policy_logp = policy_logp[:, 0]
        negative_policy_logp = policy_logp[:, 1:]
        chosen_reward = rewards[:, 0]
        negative_reward = rewards[:, 1:]

        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "grpo_only_loss": grpo_loss.detach(),
            "policy_loss": pg_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "mean_reward": rewards.detach().mean(),
            "mean_advantage": advantages.detach().mean(),
            "advantage_std": advantages.detach().std(unbiased=False),
            "chosen_reward": chosen_reward.detach().mean(),
            "negative_reward": negative_reward.detach().mean(),
            "chosen_logp": chosen_policy_logp.detach().mean(),
            "negative_logp": negative_policy_logp.detach().mean(),
            "ref_logp": ref_logp.detach().mean(),
            "ratio_mean": ratio.detach().mean(),
            "ratio_clip_fraction": (
                (ratio.detach() < 1.0 - float(self.cfg.clip_epsilon))
                | (ratio.detach() > 1.0 + float(self.cfg.clip_epsilon))
            ).float().mean(),
            "group_size": torch.tensor(float(group_size), device=policy_logp.device),
        }
        if self.cfg.loss_reduction == "none":
            metrics["loss_per_candidate_mean"] = grpo_loss.detach().mean()
        if sft_out is not None:
            metrics["sft_loss"] = sft_out.loss.detach()

        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        if sft_out is not None:
            method_metrics.update(sft_out.metrics)

        extra: dict[str, Any] = {
            "stage": stage,
            "step": step,
            "group_token_ids": group_ids.detach(),
            "group_token_mask": group_mask.detach(),
            "group_rewards": rewards.detach(),
            "group_advantages": advantages.detach(),
            "policy_logp": policy_logp.detach(),
            "ref_logp": ref_logp.detach(),
        }
        if sft_out is not None:
            extra.update(sft_out.extra)

        return MethodOutput(
            loss=total_loss,
            metrics=method_metrics,
            model_output=sft_out.model_output if sft_out is not None else None,
            extra=extra,
        )

    def _build_reference_model(self, model: FlowRecModel | None) -> FlowRecModel:
        if model is None:
            raise ValueError("GRPOMethod requires a model to build a frozen reference copy.")

        reference_model = copy.deepcopy(model)
        checkpoint = self.cfg.reference_model_checkpoint
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            payload = torch.load(checkpoint_path, map_location="cpu")
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

    def _resolve_chosen(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if "target_token_ids" not in batch or not torch.is_tensor(batch["target_token_ids"]):
            raise KeyError("GRPOMethod requires batch['target_token_ids'].")

        chosen_ids = batch["target_token_ids"].long()
        chosen_mask = batch.get("target_token_mask")
        if chosen_mask is None or not torch.is_tensor(chosen_mask):
            chosen_mask = torch.ones_like(chosen_ids, dtype=torch.bool, device=chosen_ids.device)
        else:
            chosen_mask = chosen_mask.to(chosen_ids.device).bool()
        return chosen_ids, chosen_mask

    def _sample_negatives(
        self,
        batch: Batch,
        *,
        model: FlowRecModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.negative_sampler != "random":
            raise ValueError(f"Unsupported GRPO negative sampler: {self.cfg.negative_sampler}")

        trajectories = self.negative_sampler.sample(batch, model=model)
        if trajectories.candidate_token_ids.ndim != 3:
            raise ValueError("Negative sampler must return candidate_token_ids with shape [B, N, L].")
        if trajectories.candidate_token_ids.size(1) < int(self.cfg.num_negatives):
            raise ValueError(
                "Negative sampler returned fewer candidates than GRPOConfig.num_negatives."
            )

        negative_ids = trajectories.candidate_token_ids[:, : int(self.cfg.num_negatives)].long()
        negative_mask = trajectories.candidate_token_mask[:, : int(self.cfg.num_negatives)].bool()
        return negative_ids, negative_mask

    def _build_rewards(
        self,
        *,
        batch_size: int,
        group_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rewards = torch.full(
            (batch_size, group_size),
            fill_value=float(self.cfg.reward_negative),
            device=device,
            dtype=dtype,
        )
        rewards[:, 0] = float(self.cfg.reward_positive)
        return rewards

    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_advantages:
            return rewards
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        return (rewards - mean) / std.clamp_min(float(self.cfg.advantage_epsilon))

    def _repeat_batch_for_group(
        self,
        batch: Batch,
        *,
        group_size: int,
        batch_size: int,
    ) -> Batch:
        repeated: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == batch_size:
                repeated[key] = value.repeat_interleave(group_size, dim=0)
            else:
                repeated[key] = value
        return repeated

    def _reduce_loss(self, per_candidate_loss: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_reduction == "mean":
            return per_candidate_loss.mean()
        if self.cfg.loss_reduction == "sum":
            return per_candidate_loss.sum()
        if self.cfg.loss_reduction == "none":
            return per_candidate_loss.mean()
        raise ValueError(f"Unknown GRPO loss reduction: {self.cfg.loss_reduction}")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
    ) -> "GRPOMethod":
        return cls(
            cfg=GRPOConfig.from_dict(data),
            model=model,
            reference_model=reference_model,
        )
