"""GRPO and multi-negative softmax DPO for the missing RL experiments."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import Batch, FlowRecModel
from flowrec.models.catalog import _CatalogActionIndex
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


NUM_NEGATIVES = 16


class OnPolicyBackfillMethod(FlowRecMethod):
    """One observed positive and sixteen distinct negatives drawn from the current policy."""

    def __init__(
        self,
        model: FlowRecModel,
        *,
        algorithm: str,
        item_id_to_semantic_row: torch.Tensor,
        reward_model: nn.Module | None = None,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        if algorithm not in {"grpo", "sdpo"}:
            raise ValueError(f"Unknown backfill algorithm {algorithm!r}.")
        if (algorithm == "grpo") != (reward_model is not None):
            raise ValueError("GRPO requires a reward model; SDPO uses implicit preference scores.")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive.")
        table = model.item_semantic_ids
        mapping = item_id_to_semantic_row.detach().long()
        if mapping.ndim != 1 or mapping.numel() == 0 or int(mapping[0]) != -1:
            raise ValueError("Expected a dataset item-to-catalog-row mapping with padding item 0 unmapped.")
        valid_rows = mapping[mapping >= 0]
        if valid_rows.numel() <= NUM_NEGATIVES or int(valid_rows.max()) >= table.size(0):
            raise ValueError("On-policy training needs a valid item mapping and at least 17 catalog items.")
        self.method_name = algorithm
        self.beta = beta
        self.reference = copy.deepcopy(model).requires_grad_(False).eval()
        self.reward_model = reward_model
        if self.reward_model is not None:
            self.reward_model.requires_grad_(False).eval()

        self.sampler = SemanticIDCandidateSampler(
            CandidateSamplerConfig(
                mode="policy", num_negatives=NUM_NEGATIVES, strict_negatives=True,
                include_positive=True, exclude_history=False, policy_extra_beams=1,
            ),
            item_id_to_semantic_row=mapping,
        )
        # The sampler normalizes over children of *mapped* catalog items only.
        # A tokenizer's full table can also contain items removed by this split.
        self.policy_actions = (
            _CatalogActionIndex(table.index_select(0, valid_rows.to(table.device)))
            if algorithm == "grpo" else None
        )

    def train(self, mode: bool = True) -> OnPolicyBackfillMethod:
        super().train(mode)
        self.reference.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        return self

    def _token_logprobs(
        self, model: FlowRecModel, batch: Batch, trajectories: SemanticTrajectoryBatch
    ) -> torch.Tensor:
        ids = trajectories.candidate_token_ids
        mask = trajectories.candidate_token_mask
        # GPT2 can share one graph-preserving prefill across all 17 candidates.
        prefill = getattr(model, "cache_history_encoding_grad", None)
        if callable(prefill):
            batch = prefill(batch)
        if self.policy_actions is None:
            # rl_ablations SDPO compares complete SID likelihoods using the
            # model's vocabulary normalization, including for on-policy negatives.
            return model.score_item_tokens(batch, candidate_token_ids=ids, candidate_token_mask=mask).float()
        logits = model.candidate_token_logits(batch, candidate_token_ids=ids, candidate_token_mask=mask)
        return self.policy_actions.token_logprobs(
            logits.flatten(0, 1), ids.flatten(0, 1), mask.flatten(0, 1)
        ).reshape_as(ids)

    def compute_loss(
        self, model: FlowRecModel, batch: Batch, *, stage: str = "train", step: int | None = None,
    ) -> MethodOutput:
        trajectories = self.sampler.sample(batch, model=model, step=step)
        rows = trajectories.candidate_item_row_ids
        ids = trajectories.candidate_token_ids
        if rows is None or not bool(rows[:, 0].ge(0).all()) or not torch.equal(
            model.item_semantic_ids[rows[:, 0]], ids[:, 0]
        ):
            raise ValueError("Positive target SID differs from the model catalog entry for its dataset item ID.")
        # Dropout is off during both rollout and scoring. eval() leaves gradients
        # enabled for the current policy; the initial reference stays frozen.
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                reference = self._token_logprobs(self.reference, batch, trajectories)
            policy = self._token_logprobs(model, batch, trajectories)
        finally:
            model.train(was_training)

        if self.method_name == "sdpo":
            scores = self.beta * (policy - reference).sum(dim=-1)
            loss = (scores.logsumexp(dim=1) - scores[:, 0]).mean()
            return MethodOutput(loss=loss, metrics={"sdpo/loss": loss.detach()})

        with torch.no_grad():
            rewards = self.reward_model(batch, trajectories).float()
            if rewards.shape != ids.shape[:2]:
                raise ValueError(f"Expected one reward per candidate, got {tuple(rewards.shape)}.")
            centered = rewards - rewards.mean(dim=1, keepdim=True)
            advantages = centered / rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)

        # Each rollout is consumed once before the optimizer step, so pi_old
        # has exactly the current policy's values; its denominator is detached.
        ratio = (policy - policy.detach()).exp()
        delta = reference - policy
        kl = delta.expm1() - delta
        objective = ratio * advantages.unsqueeze(-1) - self.beta * kl
        mask = trajectories.candidate_token_mask
        loss = -((objective * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)).mean()
        return MethodOutput(loss=loss, metrics={
            "grpo/loss": loss.detach(),
            "grpo/reward_mean": rewards.mean(),
            "grpo/kl": kl.detach().masked_select(mask).mean(),
        })
