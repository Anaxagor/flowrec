from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


SVDRewardTransform = Literal[
    "softplus",
    "sigmoid",
    "identity",
    "exp",
    "minmax_batch",
    "rank_percentile",
]


@dataclass(frozen=True, slots=True)
class SVDRewardConfig:
    """Configuration for SVD-based GFlowGR rewards.

    The reward is computed from a trained SVD item-scoring model:

        raw_score = SVD(user_history, candidate_item)
        reward    = transform(raw_score)

    GFlowNet losses use log(reward), so the final reward is always clamped to
    be strictly positive.
    """

    reward_epsilon: float = 1e-6

    # SVD scores can be negative.  The default transform maps them to a
    # positive, monotonic reward while preserving score ordering.
    transform: SVDRewardTransform = "softplus"

    # Applied before nonlinear transforms:
    #     score_scaled = (score + score_shift) / temperature
    temperature: float = 1.0
    score_shift: float = 0.0
    reward_scale: float = 1.0

    # SVD is normally a fixed teacher/reward model for GFlowGR.
    freeze_svd_model: bool = True
    detach_scores: bool = True

    # Batch key containing the positive target item ID.  Used because the
    # random semantic sampler stores positive candidate row IDs as -1.
    target_item_key: str = "target_item_ids"

    # If candidate IDs cannot be resolved, invalid entries receive epsilon.
    allow_invalid_candidates: bool = True


class SVDRewardModel(nn.Module):
    """Reward model that uses scores from a trained SVD recommender.

    This class is designed for GFlowGR candidate trajectories.  It expects a
    ``SemanticTrajectoryBatch`` containing candidates.  To call SVD, semantic-ID
    candidates must be mapped back to internal atomic item IDs.

    Resolution order for candidate item IDs:
        1. ``trajectories.extra['candidate_item_ids']`` if present.
        2. ``trajectories.candidate_item_row_ids``.
           By default, row IDs are assumed to equal FlowRec internal item IDs.
        3. If ``item_id_by_semantic_row`` was provided, candidate row IDs are
           mapped through that tensor.
        4. Positive candidates are filled from ``batch[target_item_key]``.

    The output shape is ``[B, N]`` where N is the number of candidate
    trajectories.
    """

    def __init__(
        self,
        svd_model: FlowRecModel,
        cfg: SVDRewardConfig | None = None,
        *,
        item_id_by_semantic_row: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SVDRewardConfig()
        self.svd_model = svd_model

        if item_id_by_semantic_row is None:
            self.register_buffer("item_id_by_semantic_row", torch.empty(0, dtype=torch.long))
        else:
            self.register_buffer("item_id_by_semantic_row", item_id_by_semantic_row.long())

        if self.cfg.freeze_svd_model:
            self.svd_model.eval()
            for param in self.svd_model.parameters():
                param.requires_grad_(False)

    def forward(self, batch: Batch, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        candidate_item_ids = self.resolve_candidate_item_ids(batch, trajectories)
        invalid = candidate_item_ids.lt(0)

        if invalid.any() and not self.cfg.allow_invalid_candidates:
            raise ValueError(
                "Some SVD reward candidates could not be mapped to item IDs. "
                "Provide trajectories.candidate_item_row_ids, "
                "trajectories.extra['candidate_item_ids'], or item_id_by_semantic_row."
            )

        # Use a safe item ID for invalid entries.  Their rewards are reset to
        # epsilon after scoring.
        safe_candidate_item_ids = candidate_item_ids.clamp_min(0)

        with torch.set_grad_enabled(not self.cfg.detach_scores):
            raw_scores = self.svd_model.score_items(
                batch,
                candidate_item_ids=safe_candidate_item_ids,
            )

        if raw_scores.ndim == 1:
            raw_scores = raw_scores.unsqueeze(1)

        if raw_scores.shape != safe_candidate_item_ids.shape:
            raise ValueError(
                "SVD score shape does not match candidate IDs: "
                f"scores={tuple(raw_scores.shape)}, "
                f"candidate_item_ids={tuple(safe_candidate_item_ids.shape)}"
            )

        rewards = self.transform_scores(raw_scores)

        if invalid.any():
            rewards = rewards.masked_fill(invalid.to(rewards.device), float(self.cfg.reward_epsilon))

        rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
        return rewards.detach() if self.cfg.detach_scores else rewards

    def resolve_candidate_item_ids(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        """Resolve semantic trajectory candidates to atomic item IDs."""
        device = trajectories.candidate_token_ids.device
        bsz = trajectories.batch_size
        n_candidates = trajectories.num_candidates

        from_extra = trajectories.extra.get("candidate_item_ids")
        if torch.is_tensor(from_extra):
            candidate_item_ids = from_extra.to(device).long()
        elif trajectories.candidate_item_row_ids is not None:
            candidate_item_ids = trajectories.candidate_item_row_ids.to(device).long()

            # Optional row -> item-id mapping.  This is useful if the semantic-ID
            # table row is not the same as the internal atomic item ID.
            if self.item_id_by_semantic_row.numel() > 0:
                mapping = self.item_id_by_semantic_row.to(device)
                valid_rows = (candidate_item_ids >= 0) & (candidate_item_ids < mapping.numel())
                safe_rows = candidate_item_ids.clamp_min(0).clamp_max(mapping.numel() - 1)
                mapped = mapping[safe_rows]
                candidate_item_ids = torch.where(valid_rows, mapped, torch.full_like(mapped, -1))
        else:
            # If there are no row IDs, this only works when all candidates are
            # the positive target.
            candidate_item_ids = torch.full((bsz, n_candidates), -1, device=device, dtype=torch.long)

        if candidate_item_ids.shape != (bsz, n_candidates):
            raise ValueError(
                "candidate item IDs must have shape [B, N], got "
                f"{tuple(candidate_item_ids.shape)}; expected {(bsz, n_candidates)}"
            )

        # Fill positive candidates from batch[target_item_key].  The random
        # sampler currently stores positive row IDs as -1, so this is essential.
        target_item_ids = _get_target_item_ids(batch, self.cfg.target_item_key)
        if target_item_ids is not None:
            target_item_ids = target_item_ids.to(device).long().reshape(bsz, 1)
            candidate_item_ids = torch.where(
                trajectories.is_positive.to(device).bool(),
                target_item_ids.expand_as(candidate_item_ids),
                candidate_item_ids,
            )

        return candidate_item_ids

    def transform_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """Map arbitrary SVD scores to strictly positive GFlowNet rewards."""
        scores = raw_scores.float()
        scores = (scores + float(self.cfg.score_shift)) / max(float(self.cfg.temperature), 1e-12)

        if self.cfg.transform == "softplus":
            rewards = F.softplus(scores)
        elif self.cfg.transform == "sigmoid":
            rewards = torch.sigmoid(scores)
        elif self.cfg.transform == "identity":
            rewards = scores
        elif self.cfg.transform == "exp":
            rewards = torch.exp(scores.clamp(max=50.0))
        elif self.cfg.transform == "minmax_batch":
            rewards = _minmax_batch(scores)
        elif self.cfg.transform == "rank_percentile":
            rewards = _rank_percentile(scores)
        else:
            raise ValueError(f"Unknown SVD reward transform: {self.cfg.transform}")

        rewards = rewards * float(self.cfg.reward_scale)
        return rewards.clamp_min(float(self.cfg.reward_epsilon))




def _get_target_item_ids(batch: Batch, preferred_key: str) -> torch.Tensor | None:
    for key in (preferred_key, "target_item_ids", "target_item_id"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _minmax_batch(scores: torch.Tensor) -> torch.Tensor:
    """Per-user min-max transform over candidate scores."""
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))

    min_vals = torch.where(finite, scores, torch.full_like(scores, float("inf"))).amin(dim=1, keepdim=True)
    max_vals = torch.where(finite, scores, torch.full_like(scores, -float("inf"))).amax(dim=1, keepdim=True)

    span = (max_vals - min_vals).clamp_min(1e-12)
    rewards = (safe_scores - min_vals) / span
    rewards = torch.where(finite, rewards, torch.zeros_like(rewards))
    return rewards


def _rank_percentile(scores: torch.Tensor) -> torch.Tensor:
    """Convert candidate scores to percentile rewards per user.

    Lowest score gets roughly 1/N and highest gets 1.0.  This keeps rewards
    bounded and positive while preserving candidate order.
    """
    if scores.ndim != 2:
        raise ValueError("rank_percentile expects scores of shape [B, N]")
    n = scores.size(1)
    if n == 1:
        return torch.ones_like(scores)

    order = torch.argsort(scores, dim=1, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float32)
    rank_values = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32)
    ranks.scatter_(dim=1, index=order, src=rank_values.expand_as(ranks))
    return ranks / float(n)
