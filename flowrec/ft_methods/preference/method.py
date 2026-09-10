from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.preference.config import DPOConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel


class DPOMethod(FlowRecMethod):
    """Direct Preference Optimization for semantic-ID recommendation."""

    method_name = "dpo"

    def __init__(
        self,
        cfg: DPOConfig | None = None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
        sft_method: SFTMethod | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DPOConfig()
        self.negative_sampler = SemanticIDCandidateSampler(
            CandidateSamplerConfig(
                mode="random",
                num_negatives=1,
                include_positive=False,
                exclude_history=False,
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
                generation_constrain_valid_sequences=self.cfg.generation_constrain_valid_sequences,
            )
        )

    def train(self, mode: bool = True) -> "DPOMethod":
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
        rejected_ids, rejected_mask = self._sample_rejected(batch, model=model)

        chosen_logp = model.sequence_logprobs(
            batch,
            target_token_ids=chosen_ids,
            target_token_mask=chosen_mask,
        )
        rejected_logp = model.sequence_logprobs(
            batch,
            target_token_ids=rejected_ids,
            target_token_mask=rejected_mask,
        )

        self.reference_model.eval()
        with torch.no_grad():
            ref_chosen_logp = self.reference_model.sequence_logprobs(
                batch,
                target_token_ids=chosen_ids,
                target_token_mask=chosen_mask,
            )
            ref_rejected_logp = self.reference_model.sequence_logprobs(
                batch,
                target_token_ids=rejected_ids,
                target_token_mask=rejected_mask,
            )

        chosen_logp = chosen_logp.float()
        rejected_logp = rejected_logp.float()
        ref_chosen_logp = ref_chosen_logp.to(chosen_logp.device).float()
        ref_rejected_logp = ref_rejected_logp.to(chosen_logp.device).float()

        policy_margin = chosen_logp - rejected_logp
        ref_margin = ref_chosen_logp - ref_rejected_logp
        reward_margin = float(self.cfg.beta) * (policy_margin - ref_margin)
        per_example_loss = -F.logsigmoid(reward_margin)
        dpo_loss = self._reduce_loss(per_example_loss)

        sft_out: MethodOutput | None = None
        total_loss = dpo_loss
        if self.cfg.sft_weight > 0:
            sft_out = self.sft_method.compute_loss(model, batch, stage=stage, step=step)
            total_loss = total_loss + float(self.cfg.sft_weight) * sft_out.loss

        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "dpo_only_loss": dpo_loss.detach(),
            "reward_margin": reward_margin.detach().mean(),
            "chosen_logp": chosen_logp.detach().mean(),
            "rejected_logp": rejected_logp.detach().mean(),
            "ref_chosen_logp": ref_chosen_logp.detach().mean(),
            "ref_rejected_logp": ref_rejected_logp.detach().mean(),
            "preference_accuracy": (reward_margin.detach() > 0).float().mean(),
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
        # Unweighted components for gradient-norm weight calibration (SFT is
        # only present when sft_weight > 0, i.e. when it was computed).
        components: dict[str, torch.Tensor] = {"dpo": dpo_loss}
        if sft_out is not None:
            components["sft"] = sft_out.loss
        extra[LOSS_COMPONENTS_KEY] = components

        return MethodOutput(
            loss=total_loss,
            metrics=method_metrics,
            model_output=sft_out.model_output if sft_out is not None else None,
            extra=extra,
        )

    def _build_reference_model(self, model: FlowRecModel | None) -> FlowRecModel:
        if model is None:
            raise ValueError("DPOMethod requires a model to build a frozen reference copy.")

        reference_model = copy.deepcopy(model)
        checkpoint = self.cfg.reference_model_checkpoint
        if checkpoint is not None:
            reference_model.load_checkpoint(Path(checkpoint), map_location="cpu")
        return reference_model

    def _freeze_reference_model(self) -> None:
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    def _resolve_chosen(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if "target_token_ids" not in batch or not torch.is_tensor(batch["target_token_ids"]):
            raise KeyError("DPOMethod requires batch['target_token_ids'].")

        chosen_ids = batch["target_token_ids"].long()
        chosen_mask = batch.get("target_token_mask")
        if chosen_mask is None or not torch.is_tensor(chosen_mask):
            chosen_mask = torch.ones_like(chosen_ids, dtype=torch.bool, device=chosen_ids.device)
        else:
            chosen_mask = chosen_mask.to(chosen_ids.device).bool()
        return chosen_ids, chosen_mask

    def _sample_rejected(
        self,
        batch: Batch,
        *,
        model: FlowRecModel,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.negative_sampler != "random":
            raise ValueError(f"Unsupported DPO negative sampler: {self.cfg.negative_sampler}")

        trajectories = self.negative_sampler.sample(batch, model=model)
        if trajectories.candidate_token_ids.ndim != 3 or trajectories.candidate_token_ids.size(1) < 1:
            raise ValueError("Negative sampler must return candidate_token_ids with shape [B, N, L].")
        return trajectories.candidate_token_ids[:, 0], trajectories.candidate_token_mask[:, 0]

    def _reduce_loss(self, per_example_loss: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_reduction == "mean":
            return per_example_loss.mean()
        if self.cfg.loss_reduction == "sum":
            return per_example_loss.sum()
        raise ValueError(f"Unknown DPO loss reduction: {self.cfg.loss_reduction}")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
    ) -> "DPOMethod":
        return cls(
            cfg=DPOConfig.from_dict(data),
            model=model,
            reference_model=reference_model,
        )
