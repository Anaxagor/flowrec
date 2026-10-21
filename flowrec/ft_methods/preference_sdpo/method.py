from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import torch

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, MethodOutput, prefix_metrics
from flowrec.ft_methods.calibration import calibrate_component_weight
from flowrec.ft_methods.gpt2_history_cache import (
    run_candidate_chunks,
    score_item_token_logprobs,
    shared_history_cache_enabled,
)
from flowrec.ft_methods.preference.method import DPOMethod
from flowrec.ft_methods.preference_sdpo.config import SDPOConfig
from flowrec.models.base import Batch, FlowRecModel


def softmax_preference_loss(
    chosen_reward: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy for choosing the positive among one or more negatives."""
    logits = torch.cat((chosen_reward.unsqueeze(1), rejected_rewards), dim=1)
    return torch.logsumexp(logits, dim=1) - chosen_reward


class SDPOMethod(DPOMethod):
    """Multi-negative softmax preference optimization for semantic-ID recommendation.

    With ``num_negatives=1`` the softmax objective equals pairwise DPO.
    """

    method_name = "sdpo"

    def __init__(
        self,
        cfg: SDPOConfig | None = None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
        sft_method=None,
        negative_sampler: SemanticIDCandidateSampler | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg or SDPOConfig(),
            model=model,
            reference_model=reference_model,
            sft_method=sft_method,
        )
        if negative_sampler is None:
            if self.cfg.negative_sampler != "random":
                raise ValueError("Curriculum S-DPO requires a configured negative sampler.")
            negative_sampler = SemanticIDCandidateSampler(
                CandidateSamplerConfig(
                    mode="random",
                    num_negatives=self.cfg.num_negatives,
                    include_positive=False,
                    exclude_history=self.cfg.exclude_history_in_sampling,
                )
            )
        if negative_sampler.cfg.num_negatives != self.cfg.num_negatives:
            raise ValueError("S-DPO sampler num_negatives must match SDPOConfig.num_negatives.")
        if negative_sampler.cfg.mode != self.cfg.negative_sampler:
            raise ValueError("S-DPO sampler mode must match SDPOConfig.negative_sampler.")
        self.negative_sampler = negative_sampler
        self.sft_weight_calibration: dict[str, Any] | None = None

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        chosen_ids, chosen_mask = self._resolve_chosen(batch)
        rejected_ids, rejected_mask = self._sample_distinct_rejected(
            batch,
            model=model,
            num_negatives=int(self.cfg.num_negatives),
            step=step,
        )

        chosen_logp, rejected_logp = self._sequence_logprobs(
            model,
            batch,
            chosen_ids=chosen_ids,
            chosen_mask=chosen_mask,
            rejected_ids=rejected_ids,
            rejected_mask=rejected_mask,
        )

        self.reference_model.eval()
        with torch.no_grad():
            ref_chosen_logp, ref_rejected_logp = self._sequence_logprobs(
                self.reference_model,
                batch,
                chosen_ids=chosen_ids,
                chosen_mask=chosen_mask,
                rejected_ids=rejected_ids,
                rejected_mask=rejected_mask,
            )

        ref_chosen_logp = ref_chosen_logp.to(chosen_logp.device).float()
        ref_rejected_logp = ref_rejected_logp.to(chosen_logp.device).float()

        chosen_reward = float(self.cfg.beta) * (chosen_logp - ref_chosen_logp)
        rejected_reward = float(self.cfg.beta) * (rejected_logp - ref_rejected_logp)
        reward_margin = chosen_reward.unsqueeze(1) - rejected_reward

        per_example_loss = softmax_preference_loss(chosen_reward, rejected_reward)
        sdpo_loss = self._reduce_loss(per_example_loss)

        sft_out = self._maybe_compute_sft_output(
            model,
            batch,
            stage=stage,
            step=step,
        )
        total_loss = sdpo_loss
        if self.cfg.sft_weight > 0:
            if sft_out is None:
                raise RuntimeError("Internal error: SFT output is required when sft_weight > 0.")
            total_loss = total_loss + float(self.cfg.sft_weight) * sft_out.loss

        detached_chosen = chosen_reward.detach()
        detached_rejected = rejected_reward.detach()
        detached_margin = reward_margin.detach()
        max_rejected = detached_rejected.max(dim=1).values

        metrics: dict[str, Any] = {
            "loss": total_loss.detach(),
            "sdpo_only_loss": sdpo_loss.detach(),
            "num_negatives": torch.tensor(
                float(self.cfg.num_negatives),
                device=chosen_logp.device,
            ),
            "chosen_logp": chosen_logp.detach().mean(),
            "rejected_logp_mean": rejected_logp.detach().mean(),
            "rejected_logp_max": rejected_logp.detach().max(dim=1).values.mean(),
            "ref_chosen_logp": ref_chosen_logp.detach().mean(),
            "ref_rejected_logp_mean": ref_rejected_logp.detach().mean(),
            "ref_rejected_logp_max": ref_rejected_logp.detach().max(dim=1).values.mean(),
            "reward_chosen": detached_chosen.mean(),
            "reward_rejected_mean": detached_rejected.mean(),
            "reward_rejected_max": max_rejected.mean(),
            "reward_margin_mean": detached_margin.mean(),
            "reward_margin_min": detached_margin.min(dim=1).values.mean(),
            "preference_accuracy": (detached_chosen > max_rejected).float().mean(),
        }
        if sft_out is not None and self.cfg.sft_weight > 0:
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
        components: dict[str, torch.Tensor] = {"sdpo": sdpo_loss}
        if sft_out is not None:
            components["sft"] = sft_out.loss
        extra[LOSS_COMPONENTS_KEY] = components

        return MethodOutput(
            loss=total_loss,
            metrics=method_metrics,
            model_output=sft_out.model_output if sft_out is not None else None,
            extra=extra,
        )

    def _sequence_logprobs(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum log-probs over SID tokens, matching ``sequence_logprobs``.

        The shared-history path scores the chosen item and every negative in
        one prefill. The preference loss still sees the full candidate set, so
        its logsumexp is unchanged. Catalog-valid-action normalization is not
        used: legacy SDPO normalizes over the semantic vocabulary.
        """
        use_cache = shared_history_cache_enabled(
            model,
            enabled=bool(self.cfg.gpt2_shared_history_cache),
            allow_active_dropout=False,
        )
        chunk_size = self.cfg.gpt2_candidate_chunk_size
        checkpoint_chunks = bool(self.cfg.gpt2_checkpoint_candidates)
        if use_cache:
            all_ids = torch.cat((chosen_ids[:, None, :], rejected_ids), dim=1)
            all_mask = torch.cat((chosen_mask[:, None, :], rejected_mask), dim=1)
            # Concatenate every candidate before the preference logsumexp.
            # Chunk-wise losses would change the partition function.
            sequence_logp = score_item_token_logprobs(
                model,
                batch,
                all_ids,
                all_mask,
                valid_actions=False,
                chunk_size=chunk_size,
                checkpoint_chunks=checkpoint_chunks,
            ).float().sum(dim=-1)
            return sequence_logp[:, 0], sequence_logp[:, 1:]
        if chunk_size is not None or checkpoint_chunks:
            all_ids = torch.cat((chosen_ids[:, None, :], rejected_ids), dim=1)
            all_mask = torch.cat((chosen_mask[:, None, :], rejected_mask), dim=1)

            def score_legacy(
                _batch: Batch,
                ids: torch.Tensor,
                mask: torch.Tensor,
            ) -> torch.Tensor:
                return model.score_items(
                    _batch,
                    candidate_token_ids=ids,
                    candidate_token_mask=mask,
                )

            sequence_logp = run_candidate_chunks(
                score_legacy,
                model,
                all_ids,
                all_mask,
                batch,
                chunk_size=chunk_size,
                checkpoint_chunks=checkpoint_chunks,
            ).float()
            return sequence_logp[:, 0], sequence_logp[:, 1:]

        chosen_logp = model.sequence_logprobs(
            batch,
            target_token_ids=chosen_ids,
            target_token_mask=chosen_mask,
        ).float()
        rejected_logp = model.score_items(
            batch,
            candidate_token_ids=rejected_ids,
            candidate_token_mask=rejected_mask,
        ).float()
        return chosen_logp, rejected_logp

    def _sample_distinct_rejected(
        self,
        batch: Batch,
        *,
        model: FlowRecModel,
        num_negatives: int,
        step: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectories = self.negative_sampler.sample(batch, model=model, step=step)
        rejected_ids = trajectories.candidate_token_ids
        rejected_mask = trajectories.candidate_token_mask
        if rejected_ids.ndim != 3 or rejected_ids.size(1) != num_negatives:
            raise ValueError("S-DPO sampler must return exactly num_negatives candidates.")
        if not bool(trajectories.candidate_item_row_ids.ge(0).all()):
            raise ValueError("Not enough distinct valid negatives for an S-DPO batch.")
        return rejected_ids, rejected_mask

    def _maybe_compute_sft_output(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str,
        step: int | None,
    ) -> MethodOutput | None:
        if (
            self.cfg.sft_weight > 0
            or self.cfg.calibrate_sft_weight
            or (stage != "train" and self.cfg.compute_eval_metrics)
        ):
            return self.sft_method.compute_loss(model, batch, stage=stage, step=step)
        return None

    def maybe_calibrate_sft_weight(
        self,
        model: FlowRecModel,
        batches: Iterable[Batch],
        *,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any] | None:
        """Fill ``sft_weight`` from gradient-norm calibration, if requested.

        Uses unweighted ``L_sft`` vs ``L_sdpo`` on the first
        ``calibration_num_batches`` train batches so that
        ``sft_weight * ||g_sft|| ≈ target_ratio * ||g_sdpo||``. No-ops when
        ``calibrate_sft_weight`` is false (fixed ``sft_weight`` from config).
        """
        if not self.cfg.calibrate_sft_weight:
            return None
        if self.cfg.sft_weight != 0.0:
            raise RuntimeError(
                "Internal error: calibrate_sft_weight=True requires sft_weight=0 "
                "before calibration."
            )

        old_sft_cfg = self.sft_method.cfg
        self.sft_method.cfg = replace(
            old_sft_cfg,
            compute_train_metrics=False,
            compute_eval_metrics=False,
            evaluate_generation=False,
        )
        try:
            result = calibrate_component_weight(
                model,
                self,
                batches,
                component="sft",
                reference="sdpo",
                target_ratio=float(self.cfg.calibration_target_ratio),
                num_batches=int(self.cfg.calibration_num_batches),
                seed=self.cfg.calibration_seed if self.cfg.calibration_seed is not None else seed,
                step=0,
                stage="train",
                device=device,
            )
        finally:
            self.sft_method.cfg = old_sft_cfg

        self.cfg = replace(
            self.cfg,
            sft_weight=float(result["weight"]),
            calibrate_sft_weight=False,
        )
        self.sft_weight_calibration = result
        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        model: FlowRecModel | None = None,
        reference_model: FlowRecModel | None = None,
        negative_sampler: SemanticIDCandidateSampler | None = None,
    ) -> "SDPOMethod":
        return cls(
            cfg=SDPOConfig.from_dict(data),
            model=model,
            reference_model=reference_model,
            negative_sampler=negative_sampler,
        )
