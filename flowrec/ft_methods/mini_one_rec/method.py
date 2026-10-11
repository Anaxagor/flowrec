from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.calibration import calibrate_component_weight
from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class MiniOneRecMethod(FlowRecMethod):
    """GRPO/ReRe-style post-training for TIGER semantic-ID recommendation.

    The implementation follows the two attached papers in the parts they make
    explicit: constrained generation, group rewards, normalized advantages,
    token-level clipped GRPO ratio, and the reference-policy KL estimator.
    """

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
        self.reward_model = reward_model
        if self.cfg.reward_type == "rule_rank":
            if reward_model is not None:
                raise ValueError("reward_model was provided but reward_type is 'rule_rank'.")
        elif reward_model is None:
            raise ValueError(f"reward_type={self.cfg.reward_type!r} requires a reward_model.")
        else:
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad_(False)

        self.ref_model = reference_model
        if self.ref_model is not None:
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
        self.sft_weight_calibration: dict[str, Any] | None = None

    def train(self, mode: bool = True) -> "MiniOneRecMethod":
        super().train(mode)
        if self.ref_model is not None:
            self.ref_model.eval()
        if self.reward_model is not None:
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
        self._ensure_reference_model(model)
        if self.ref_model is None:
            raise RuntimeError("Reference model was not initialized.")

        # Roll out from the behavior/old policy and record its token
        # probabilities before any gradient update. This is π_{theta_old} in
        # the MiniOneRec GRPO ratio.
        trajectories, old_token_logprobs = self._sample_trajectories(model, batch)
        rewards, reward_components = self._compute_rewards(batch, trajectories)
        advantages = self._compute_advantages(rewards)

        # Score the same trajectories under the differentiable current policy.
        # eval() disables dropout so numerator and behavior denominator use the
        # same policy semantics; gradients remain enabled in eval mode.
        was_training = model.training
        model.eval()
        try:
            policy_token_logprobs = self._candidate_token_logprobs(
                model,
                batch,
                trajectories.candidate_token_ids,
                trajectories.candidate_token_mask,
            ).float()
        finally:
            model.train(was_training)

        self.ref_model.eval()
        with torch.no_grad():
            ref_token_logprobs = self._candidate_token_logprobs(
                self.ref_model,
                batch,
                trajectories.candidate_token_ids,
                trajectories.candidate_token_mask,
            ).float()

        old_token_logprobs = old_token_logprobs.to(policy_token_logprobs.device).float()
        ref_token_logprobs = ref_token_logprobs.to(policy_token_logprobs.device).float()
        candidate_mask = trajectories.candidate_token_mask.to(policy_token_logprobs.device).to(
            policy_token_logprobs.dtype
        )

        ratio = torch.exp(policy_token_logprobs - old_token_logprobs)
        eps = float(self.cfg.clip_epsilon)
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        token_advantage = advantages.to(policy_token_logprobs.device).float().unsqueeze(-1)
        surrogate = torch.minimum(ratio * token_advantage, clipped_ratio * token_advantage)

        # Eq. (20) in ReRe: k3-style non-negative KL estimator.
        ref_minus_policy = ref_token_logprobs - policy_token_logprobs
        per_token_kl = torch.exp(ref_minus_policy) - ref_minus_policy - 1.0
        per_token_objective = surrogate - float(self.cfg.beta) * per_token_kl

        token_count = candidate_mask.sum(dim=-1).clamp_min(1.0)
        sequence_objective = (per_token_objective * candidate_mask).sum(dim=-1) / token_count
        if self.cfg.loss_reduction == "mean":
            grpo_loss = -sequence_objective.mean()
        elif self.cfg.loss_reduction == "sum":
            grpo_loss = -sequence_objective.sum()
        else:
            raise ValueError(f"Unknown MiniOneRec loss_reduction={self.cfg.loss_reduction!r}.")

        sft_out = self._maybe_compute_sft_output(
            model,
            batch,
            stage=stage,
            step=step,
        )
        sft_loss = torch.zeros((), device=grpo_loss.device, dtype=grpo_loss.dtype)
        loss = float(self.cfg.grpo_weight) * grpo_loss
        if self.cfg.sft_weight > 0:
            if sft_out is None:
                raise RuntimeError("Internal error: SFT output is required when sft_weight > 0.")
            sft_loss = sft_out.loss.to(grpo_loss.device)
            loss = loss + float(self.cfg.sft_weight) * sft_loss

        metrics = self._build_metrics(
            loss=loss,
            grpo_loss=grpo_loss,
            sft_loss=sft_loss,
            rewards=rewards,
            advantages=advantages,
            trajectories=trajectories,
            per_token_kl=per_token_kl,
            candidate_mask=candidate_mask,
            ratio=ratio,
            reward_components=reward_components,
        )
        if stage == "train" and not self.cfg.compute_train_metrics:
            metrics = {"loss": loss.detach()}
        elif stage != "train" and not self.cfg.compute_eval_metrics:
            metrics = {"loss": loss.detach()}

        method_metrics = prefix_metrics(metrics, prefix=self.cfg.metric_prefix)
        if sft_out is not None:
            method_metrics.update(sft_out.metrics)

        extra: dict[str, Any] = {
            "stage": stage,
            "step": step,
            "trajectories": trajectories,
            "rewards": rewards.detach(),
            "advantages": advantages.detach(),
            "old_token_logprobs": old_token_logprobs.detach(),
        }
        if sft_out is not None:
            extra.update(sft_out.extra)
        components: dict[str, torch.Tensor] = {"grpo": grpo_loss}
        if sft_out is not None:
            components["sft"] = sft_out.loss
        extra[LOSS_COMPONENTS_KEY] = components

        return MethodOutput(
            loss=loss,
            metrics=method_metrics,
            model_output=sft_out.model_output if sft_out is not None else None,
            extra=extra,
        )

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

        Uses unweighted ``L_sft`` vs ``L_grpo`` on the first
        ``calibration_num_batches`` train batches so that
        ``sft_weight * ||g_sft|| ≈ target_ratio * ||g_grpo||``. No-ops when
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
                reference="grpo",
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

    def _ensure_reference_model(self, model: FlowRecModel) -> None:
        if self.ref_model is not None:
            return
        self.ref_model = copy.deepcopy(model)
        checkpoint = self.cfg.reference_model_checkpoint
        if checkpoint is not None:
            self.ref_model.load_checkpoint(Path(checkpoint), map_location="cpu")
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
    ) -> tuple[SemanticTrajectoryBatch, torch.Tensor]:
        g = int(self.cfg.num_generations)
        strategy = str(self.cfg.rollout_strategy).lower()
        if g < 2:
            raise ValueError("mini-one-rec requires num_generations >= 2.")

        if strategy == "dynamic":
            raw_count = max(g, int((3 * g) // 2))
        else:
            raw_count = g
        do_sample = strategy in {"sample", "dynamic"}

        if strategy == "beam":
            num_beams = max(int(self.cfg.rollout_num_beams or raw_count), raw_count)
        else:
            num_beams = max(int(self.cfg.rollout_num_beams or 1), 1)

        was_training = model.training
        model.eval()
        try:
            generated = model.generate(
                batch,
                num_beams=num_beams,
                num_return_sequences=raw_count,
                max_new_tokens=self.cfg.rollout_max_new_tokens,
                constrain_valid_sequences=bool(self.cfg.rollout_constrain_valid_sequences),
                exclude_seen_items=bool(self.cfg.rollout_exclude_seen_items),
                do_sample=do_sample,
                temperature=float(self.cfg.rollout_temperature) if do_sample else None,
                top_k=self.cfg.rollout_top_k if do_sample else None,
                top_p=self.cfg.rollout_top_p if do_sample else None,
            )
        finally:
            model.train(was_training)
        generated = generated.detach().clone()

        if generated.ndim != 3:
            raise ValueError(
                "model.generate() must return [B, num_generations, sid_len] for mini-one-rec. "
                f"Got shape {tuple(generated.shape)}."
            )
        if generated.size(1) != raw_count:
            raise ValueError(
                f"Expected {raw_count} generated candidates, got {generated.size(1)}."
            )

        target_ids = batch.get("target_token_ids")
        if target_ids is None or not torch.is_tensor(target_ids):
            raise KeyError("mini-one-rec requires batch['target_token_ids'].")
        target_ids = target_ids.to(generated.device).long()

        if strategy == "dynamic":
            candidate_token_ids = self._dynamic_select(generated, target_ids, g=g)
        else:
            candidate_token_ids = generated[:, :g]
        candidate_token_mask = torch.ones_like(candidate_token_ids, dtype=torch.bool)

        # Before any positive augmentation, record how often the behavior
        # policy naturally generated the exact target SID. This is the key
        # diagnostic for TIGER: if it is near zero, pure exact-match GRPO has
        # almost no positive learning signal.
        natural_positive = candidate_token_ids.eq(target_ids.unsqueeze(1)).all(dim=-1)
        natural_group_hit = natural_positive.any(dim=1)
        forced_positive = torch.zeros_like(natural_positive)

        if self.cfg.force_ground_truth_in_group:
            target_mask = batch.get("target_token_mask")
            if target_mask is None or not torch.is_tensor(target_mask):
                target_mask = torch.ones_like(target_ids, dtype=torch.bool)
            else:
                target_mask = target_mask.to(candidate_token_ids.device).bool()

            # Keep all naturally generated positives. Only groups with no target
            # are augmented, replacing the last candidate. For beam search the
            # incoming candidates are already ordered by generation score, so
            # this discards the weakest beam; all candidates are re-scored and
            # re-sorted below under the behavior policy.
            missing = ~natural_group_hit
            if missing.any():
                candidate_token_ids = candidate_token_ids.clone()
                candidate_token_mask = candidate_token_mask.clone()
                candidate_token_ids[missing, -1, :] = target_ids[missing]
                candidate_token_mask[missing, -1, :] = target_mask[missing]
                forced_positive[missing, -1] = True

        # The TIGER generator currently has a broad exception fallback from
        # constrained to unconstrained decoding. Post-validation makes the
        # paper's "only valid items" guarantee enforceable.
        candidate_item_row_ids = self._resolve_candidate_item_rows(model, candidate_token_ids)
        if self.cfg.rollout_constrain_valid_sequences:
            if candidate_item_row_ids is None:
                raise ValueError(
                    "Constrained MiniOneRec rollouts require model.item_semantic_ids for validation."
                )
            if (candidate_item_row_ids < 0).any():
                bad = int((candidate_item_row_ids < 0).sum().item())
                raise RuntimeError(
                    f"Constrained generation returned {bad} invalid semantic-ID sequence(s). "
                    "TIGER may have fallen back to unconstrained generation."
                )

        if strategy == "beam":
            for b in range(candidate_token_ids.size(0)):
                if torch.unique(candidate_token_ids[b], dim=0).size(0) != g:
                    raise RuntimeError(
                        "Beam rollout produced duplicate items, contradicting the distinct-beam "
                        "sampling assumption used by MiniOneRec/ReRe."
                    )

        # Rank candidates by behavior-policy generation probability. Under
        # constrained decoding use the valid-action normalization implemented
        # by TIGER so the ranking matches the actual constrained action space.
        was_training = model.training
        model.eval()
        try:
            behavior_token_logprobs = self._candidate_token_logprobs(
                model,
                batch,
                candidate_token_ids,
                candidate_token_mask,
            ).float()
        finally:
            model.train(was_training)

        behavior_scores = (behavior_token_logprobs * candidate_token_mask.float()).sum(dim=-1)
        sort_idx = behavior_scores.argsort(dim=1, descending=True)
        gather_tokens = sort_idx.unsqueeze(-1).expand(-1, -1, candidate_token_ids.size(-1))
        candidate_token_ids = torch.gather(candidate_token_ids, 1, gather_tokens)
        candidate_token_mask = torch.gather(candidate_token_mask, 1, gather_tokens)
        behavior_token_logprobs = torch.gather(behavior_token_logprobs, 1, gather_tokens)
        behavior_scores = torch.gather(behavior_scores, 1, sort_idx)
        forced_positive = torch.gather(forced_positive, 1, sort_idx)
        if candidate_item_row_ids is not None:
            candidate_item_row_ids = torch.gather(candidate_item_row_ids, 1, sort_idx)

        is_positive = candidate_token_ids.eq(target_ids.unsqueeze(1)).all(dim=-1)
        trajectories = SemanticTrajectoryBatch(
            candidate_token_ids=candidate_token_ids.long(),
            candidate_token_mask=candidate_token_mask.bool(),
            is_positive=is_positive,
            candidate_item_row_ids=candidate_item_row_ids,
            extra={
                "sampler": strategy,
                "candidate_scores": behavior_scores,
                "rank_source": "behavior_policy_constrained_logprob"
                if self.cfg.rollout_constrain_valid_sequences
                else "behavior_policy_logprob",
                "natural_positive": natural_positive,
                "natural_group_hit": natural_group_hit,
                "forced_positive": forced_positive,
            },
        )
        return trajectories, behavior_token_logprobs

    def _dynamic_select(
        self,
        generated: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        g: int,
    ) -> torch.Tensor:
        """Paper Appendix-A.1-style dynamic selection from floor(3G/2) samples.

        The target is prioritized only if it was actually generated. Remaining
        slots first maximize unique items; duplicates are used only when needed
        to fill G slots.
        """

        bsz, raw_count, _ = generated.shape
        selected_batches: list[torch.Tensor] = []
        for b in range(bsz):
            row = generated[b]
            selected: list[int] = []

            positives = row.eq(target_ids[b].unsqueeze(0)).all(dim=-1).nonzero(as_tuple=True)[0]
            if positives.numel() > 0:
                selected.append(int(positives[0].item()))

            for j in range(raw_count):
                if len(selected) >= g:
                    break
                candidate = row[j]
                if any(candidate.eq(row[k]).all().item() for k in selected):
                    continue
                selected.append(j)

            if len(selected) < g:
                for j in range(raw_count):
                    if len(selected) >= g:
                        break
                    if j in selected:
                        continue
                    selected.append(j)

            if len(selected) < g:
                raise RuntimeError(
                    f"Dynamic sampling produced only {len(selected)} candidates for G={g}."
                )
            idx = torch.tensor(selected[:g], device=row.device, dtype=torch.long)
            selected_batches.append(row[idx])
        return torch.stack(selected_batches, dim=0)

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

        # Chunked matching avoids allocating [B*G, num_items, sid_len] all at
        # once for large catalogs while remaining exact.
        flat = candidate_token_ids.reshape(-1, candidate_token_ids.size(-1)).long()
        rows = torch.full((flat.size(0),), -1, device=flat.device, dtype=torch.long)
        chunk = 256
        for start in range(0, flat.size(0), chunk):
            part = flat[start : start + chunk]
            matches = part.unsqueeze(1).eq(catalog.unsqueeze(0)).all(dim=-1)
            has_match = matches.any(dim=1)
            if has_match.any():
                local_rows = matches[has_match].float().argmax(dim=1).long()
                part_rows = rows[start : start + part.size(0)]
                part_rows[has_match] = local_rows
        return rows.view(candidate_token_ids.size(0), candidate_token_ids.size(1))

    def _compute_rewards(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.cfg.reward_type == "rule_rank":
            return self._compute_rule_rank_rewards(batch, trajectories)

        assert self.reward_model is not None
        with torch.no_grad():
            rewards = self.reward_model(batch, trajectories).to(
                trajectories.candidate_token_ids.device
            ).float()
            expected_shape = trajectories.candidate_token_ids.shape[:2]
            if rewards.shape != expected_shape:
                raise ValueError(
                    f"MiniOneRec reward must have shape {expected_shape}, got {tuple(rewards.shape)}."
                )
            if not bool(torch.isfinite(rewards).all()):
                raise ValueError("MiniOneRec reward contains non-finite values.")
            if not bool(rewards.ge(0).all()):
                raise ValueError("MiniOneRec external rewards must be non-negative.")
            rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
            if self.cfg.reward_power != 1.0:
                rewards = rewards.pow(float(self.cfg.reward_power)).clamp_min(
                    float(self.cfg.reward_epsilon)
                )
        zero = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        return rewards, {"rule_reward_mean": zero, "rank_reward_mean": zero, "rank_reward_group_sum": zero}

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
        negative_mask = 1.0 - rule_reward

        # Eq. (7): raw negative penalty -1/log(rho+1), where rho=1 is the
        # highest-probability candidate. Eq. (8) normalizes negative penalties
        # across the group so they sum to -1 whenever at least one negative is
        # present.
        num_candidates = candidate_token_ids.size(1)
        rank = torch.arange(
            1,
            num_candidates + 1,
            device=candidate_token_ids.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(candidate_token_ids.size(0), -1)
        penalty = (1.0 / torch.log(rank + 1.0)) * negative_mask
        penalty_sum = penalty.sum(dim=1, keepdim=True)
        normalized_rank_reward = torch.where(
            penalty_sum > 0,
            -penalty / penalty_sum.clamp_min(1e-12),
            torch.zeros_like(penalty),
        )
        rank_reward = float(self.cfg.rank_reward_weight) * normalized_rank_reward
        rewards = rule_reward + rank_reward

        return rewards, {
            "rule_reward_mean": rule_reward.mean(),
            "rank_reward_mean": rank_reward.mean(),
            "rank_reward_group_sum": rank_reward.sum(dim=1).mean(),
        }

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.ndim != 2:
            raise ValueError(f"Rewards must have shape [B, G], got {tuple(rewards.shape)}.")
        if not self.cfg.normalize_advantages:
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
        if self.cfg.rollout_constrain_valid_sequences:
            scorer = getattr(model, "score_item_tokens_valid_actions", None)
            if callable(scorer):
                return scorer(
                    batch,
                    candidate_token_ids=candidate_token_ids,
                    candidate_token_mask=candidate_token_mask,
                )

        scorer = getattr(model, "score_item_tokens", None)
        if callable(scorer):
            return scorer(
                batch,
                candidate_token_ids=candidate_token_ids,
                candidate_token_mask=candidate_token_mask,
            )

        # Generic fallback for non-TIGER FlowRec models.
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

    def _group_unique_ratio(self, candidate_token_ids: torch.Tensor) -> torch.Tensor:
        ratios = []
        g = candidate_token_ids.size(1)
        for row in candidate_token_ids:
            ratios.append(torch.unique(row, dim=0).size(0) / max(g, 1))
        return torch.tensor(
            ratios,
            device=candidate_token_ids.device,
            dtype=torch.float32,
        ).mean()

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
        ratio: torch.Tensor,
        reward_components: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        reward_std = rewards.std(dim=1, unbiased=False).mean()
        advantage_std = advantages.std(dim=1, unbiased=False).mean()
        mean_kl = (per_token_kl * candidate_mask).sum(dim=-1) / candidate_mask.sum(dim=-1).clamp_min(1.0)
        sample_hit_rate = trajectories.is_positive.float().mean()
        group_hit_rate = trajectories.is_positive.any(dim=1).float().mean()
        natural_group_hit = trajectories.extra.get("natural_group_hit")
        if torch.is_tensor(natural_group_hit):
            natural_group_hit_rate = natural_group_hit.float().mean()
        else:
            natural_group_hit_rate = group_hit_rate
        forced_positive = trajectories.extra.get("forced_positive")
        if torch.is_tensor(forced_positive):
            forced_positive_rate = forced_positive.any(dim=1).float().mean()
        else:
            forced_positive_rate = torch.zeros((), device=rewards.device)
        unique_ratio = self._group_unique_ratio(trajectories.candidate_token_ids)

        eps = float(self.cfg.clip_epsilon)
        clipped = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).to(candidate_mask.dtype)
        clip_fraction = (clipped * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)

        return {
            "loss": loss.detach(),
            "grpo_loss": grpo_loss.detach(),
            "sft_loss": sft_loss.detach(),
            "reward_mean": rewards.mean().detach(),
            "reward_std": reward_std.detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_std": advantage_std.detach(),
            "sample_hit_rate": sample_hit_rate.detach(),
            "group_hit_rate": group_hit_rate.detach(),
            "natural_group_hit_rate": natural_group_hit_rate.detach(),
            "forced_positive_rate": forced_positive_rate.detach(),
            "candidate_unique_ratio": unique_ratio.detach(),
            "mean_kl": mean_kl.mean().detach(),
            "policy_ratio_mean": (ratio * candidate_mask).sum().detach()
            / candidate_mask.sum().clamp_min(1.0).detach(),
            "clip_fraction": clip_fraction.detach(),
            "mean_completion_length": candidate_mask.sum(dim=-1).float().mean().detach(),
            "rule_reward_mean": reward_components["rule_reward_mean"].detach(),
            "rank_reward_mean": reward_components["rank_reward_mean"].detach(),
            "rank_reward_group_sum": reward_components["rank_reward_group_sum"].detach(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MiniOneRecMethod":
        return cls(cfg=MiniOneRecConfig.from_dict(data))
