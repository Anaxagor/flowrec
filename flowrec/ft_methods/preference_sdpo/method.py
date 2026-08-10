from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from flowrec.candidates.random import RandomSemanticCandidateConfig, RandomSemanticIDCandidateSampler
from flowrec.ft_methods.base import MethodOutput, prefix_metrics
from flowrec.ft_methods.preference.method import DPOMethod
from flowrec.ft_methods.preference_sdpo.config import SDPOConfig
from flowrec.models.base import Batch, FlowRecModel


class SDPOMethod(DPOMethod):
    """Softmax-DPO / S-DPO for semantic-ID recommendation."""

    method_name = "sdpo"

    def __init__(
        self,
        cfg: SDPOConfig | None = None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
        sft_method=None,
    ) -> None:
        super().__init__(
            cfg=cfg or SDPOConfig(),
            model=model,
            reference_model=reference_model,
            sft_method=sft_method,
        )
        self.negative_sampler = RandomSemanticIDCandidateSampler(
            RandomSemanticCandidateConfig(
                num_negative_samples=max(int(self.cfg.num_negatives), 1),
                include_positive=False,
                avoid_positive=True,
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
        chosen_ids, chosen_mask = self._resolve_chosen(batch)
        rejected_ids, rejected_mask = self._sample_rejected(batch, model=model)

        chosen_logp = model.sequence_logprobs(
            batch,
            target_token_ids=chosen_ids,
            target_token_mask=chosen_mask,
        )
        rejected_logp = model.score_items(
            batch,
            candidate_token_ids=rejected_ids,
            candidate_token_mask=rejected_mask,
        )

        self.reference_model.eval()
        with torch.no_grad():
            ref_chosen_logp = self.reference_model.sequence_logprobs(
                batch,
                target_token_ids=chosen_ids,
                target_token_mask=chosen_mask,
            )
            ref_rejected_logp = self.reference_model.score_items(
                batch,
                candidate_token_ids=rejected_ids,
                candidate_token_mask=rejected_mask,
            )

        chosen_logp = chosen_logp.float()
        rejected_logp = rejected_logp.to(chosen_logp.device).float()
        ref_chosen_logp = ref_chosen_logp.to(chosen_logp.device).float()
        ref_rejected_logp = ref_rejected_logp.to(chosen_logp.device).float()

        reward_chosen = float(self.cfg.beta) * (chosen_logp - ref_chosen_logp)
        reward_rejected = float(self.cfg.beta) * (rejected_logp - ref_rejected_logp)
        reward_margin = reward_chosen.unsqueeze(1) - reward_rejected
        s = -reward_margin
        logsumexp_s = torch.logsumexp(s, dim=1)
        per_example_loss = -F.logsigmoid(-logsumexp_s)
        sdpo_loss = self._reduce_loss(per_example_loss)

        sft_out: MethodOutput | None = None
        total_loss = sdpo_loss
        if self.cfg.sft_weight > 0:
            sft_out = self.sft_method.compute_loss(model, batch, stage=stage, step=step)
            total_loss = total_loss + float(self.cfg.sft_weight) * sft_out.loss

        detached_reward_chosen = reward_chosen.detach()
        detached_reward_rejected = reward_rejected.detach()
        detached_reward_margin = reward_margin.detach()
        max_reward_rejected = detached_reward_rejected.max(dim=1).values

        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "sdpo_only_loss": sdpo_loss.detach(),
            "chosen_logp": chosen_logp.detach().mean(),
            "rejected_logp_mean": rejected_logp.detach().mean(),
            "rejected_logp_max": rejected_logp.detach().max(dim=1).values.mean(),
            "ref_chosen_logp": ref_chosen_logp.detach().mean(),
            "ref_rejected_logp_mean": ref_rejected_logp.detach().mean(),
            "ref_rejected_logp_max": ref_rejected_logp.detach().max(dim=1).values.mean(),
            "reward_chosen": detached_reward_chosen.mean(),
            "reward_rejected_mean": detached_reward_rejected.mean(),
            "reward_rejected_max": max_reward_rejected.mean(),
            "reward_margin_mean": detached_reward_margin.mean(),
            "reward_margin_min": detached_reward_margin.min(dim=1).values.mean(),
            "hard_negative_index_mean": detached_reward_rejected.argmax(dim=1).float().mean(),
            "preference_accuracy": (detached_reward_chosen > max_reward_rejected).float().mean(),
        }
        if sft_out is not None:
            metrics["sft_loss"] = sft_out.loss.detach()

        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        if sft_out is not None:
            method_metrics.update(sft_out.metrics)

        extra: dict[str, Any] = {
            "stage": stage,
            "step": step,
            "chosen_token_ids": chosen_ids.detach(),
            "rejected_token_ids": rejected_ids.detach(),
            "chosen_token_mask": chosen_mask.detach(),
            "rejected_token_mask": rejected_mask.detach(),
        }
        if sft_out is not None:
            extra.update(sft_out.extra)

        return MethodOutput(
            loss=total_loss,
            metrics=method_metrics,
            model_output=sft_out.model_output if sft_out is not None else None,
            extra=extra,
        )

    def _sample_rejected(
        self,
        batch: Batch,
        *,
        model: FlowRecModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.negative_sampler != "random":
            raise ValueError(f"Unsupported S-DPO negative sampler: {self.cfg.negative_sampler}")

        trajectories = self.negative_sampler.sample(batch, model=model)
        if trajectories.candidate_token_ids.ndim != 3 or trajectories.candidate_token_ids.size(1) < 1:
            raise ValueError("Negative sampler must return candidate_token_ids with shape [B, N, L].")
        return trajectories.candidate_token_ids, trajectories.candidate_token_mask

    def _reduce_loss(self, per_example_loss: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_reduction == "mean":
            return per_example_loss.mean()
        if self.cfg.loss_reduction == "sum":
            return per_example_loss.sum()
        raise ValueError(f"Unknown S-DPO loss reduction: {self.cfg.loss_reduction}")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
    ) -> "SDPOMethod":
        return cls(
            cfg=SDPOConfig.from_dict(data),
            model=model,
            reference_model=reference_model,
        )
