from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


SASRecRewardTransform = Literal[
    "softplus",
    "sigmoid",
    "identity",
    "exp",
    "minmax_batch",
    "rank_percentile",
]


@dataclass(frozen=True, slots=True)
class SASRecRewardConfig:
    """Reward model derived from SASRec-like next-item scores.

    Expected input:
      - a semantic candidate batch (``SemanticTrajectoryBatch``)
      - candidates are resolved to atomic item IDs (see notes below)

    Reward is computed from SASRec score_items(batch, candidate_item_ids):
      raw_scores = sasrec.score_items(batch, candidate_item_ids)
      rewards    = transform(raw_scores)

    Since GFlowNet losses use log(reward), rewards are always clamped to be
    strictly positive.
    """

    reward_epsilon: float = 1e-6
    transform: SASRecRewardTransform = "softplus"

    # Optional calibration before nonlinear transform:
    temperature: float = 1.0
    score_shift: float = 0.0
    reward_scale: float = 1.0

    # Key used to fetch the positive target item id from ``batch``.
    target_item_key: str = "target_item_ids"

    # If candidates cannot be resolved, either allow them (map to -1) or
    # raise.
    allow_invalid_candidates: bool = True



class SASRecRewardModel(nn.Module):
    """Reward model computed from SASRec scores over candidate items.

    Candidate resolution order (semantic -> atomic item id):
      1) ``trajectories.extra['candidate_item_ids']`` if present
      2) ``trajectories.candidate_item_row_ids`` if present
      3) fall back to positive target for all candidates (only works if sampler
         marks all candidates as positive)

    NOTE: by default we assume candidate row ids equal internal atomic item ids,
    which is consistent with how the SVDRewardModel is implemented in this repo.
    """

    def __init__(
        self,
        sasrec_model: FlowRecModel,
        cfg: SASRecRewardConfig | None = None,
        *,
        item_id_by_semantic_row: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SASRecRewardConfig()
        self.sasrec_model = sasrec_model

        if item_id_by_semantic_row is None:
            self.register_buffer("item_id_by_semantic_row", torch.empty(0, dtype=torch.long))
        else:
            self.register_buffer(
                "item_id_by_semantic_row",
                item_id_by_semantic_row.long(),
            )

        self.sasrec_model.eval()
        for p in self.sasrec_model.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        candidate_item_ids = self.resolve_candidate_item_ids(batch, trajectories)
        invalid = candidate_item_ids.lt(0)
        item_vocab_size = _sasrec_item_vocab_size(self.sasrec_model)
        if item_vocab_size is not None:
            invalid = invalid | candidate_item_ids.ge(item_vocab_size)

        if invalid.any() and not self.cfg.allow_invalid_candidates:
            raise ValueError(
                "Some SASRec reward candidates could not be resolved to atomic item ids. "
                "Provide trajectories.extra['candidate_item_ids'] or trajectories.candidate_item_row_ids."
            )

        safe_candidate_item_ids = candidate_item_ids.clamp_min(0)
        if item_vocab_size is not None:
            safe_candidate_item_ids = safe_candidate_item_ids.clamp_max(item_vocab_size - 1)

        # sasrec_model.score_items expects [B, N] item ids and returns [B, N] scores.
        self.sasrec_model.eval()
        with torch.no_grad():
            raw_scores = self.sasrec_model.score_items(
                batch,
                candidate_item_ids=safe_candidate_item_ids,
            )

        if raw_scores.ndim == 1:
            raw_scores = raw_scores.unsqueeze(1)

        if raw_scores.shape != safe_candidate_item_ids.shape:
            raise ValueError(
                "SASRec reward score shape mismatch: "
                f"raw_scores={tuple(raw_scores.shape)} vs candidate_item_ids={tuple(safe_candidate_item_ids.shape)}"
            )

        rewards = self.transform_scores(raw_scores)

        if invalid.any():
            rewards = rewards.masked_fill(invalid.to(rewards.device), float(self.cfg.reward_epsilon))

        rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
        return rewards.detach()

    def resolve_candidate_item_ids(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        """Resolve candidate semantic rows/ids to internal atomic item ids."""

        device = trajectories.candidate_token_ids.device
        bsz = trajectories.batch_size
        n_candidates = trajectories.num_candidates

        from_extra = trajectories.extra.get("candidate_item_ids")
        if torch.is_tensor(from_extra):
            candidate_item_ids = from_extra.to(device).long()
        elif trajectories.candidate_item_row_ids is not None:
            candidate_item_ids = trajectories.candidate_item_row_ids.to(device).long()

            if self.item_id_by_semantic_row.numel() > 0:
                mapping = self.item_id_by_semantic_row.to(device)
                valid_rows = (candidate_item_ids >= 0) & (candidate_item_ids < mapping.numel())
                safe_rows = candidate_item_ids.clamp_min(0).clamp_max(mapping.numel() - 1)
                mapped = mapping[safe_rows]
                candidate_item_ids = torch.where(valid_rows, mapped, torch.full_like(mapped, -1))
        else:
            candidate_item_ids = torch.full((bsz, n_candidates), -1, device=device, dtype=torch.long)

        if candidate_item_ids.shape != (bsz, n_candidates):
            raise ValueError(
                "candidate_item_ids must have shape [B, N], got "
                f"{tuple(candidate_item_ids.shape)} expected {(bsz, n_candidates)}"
            )

        # Many samplers store positive candidates as -1; fill them from batch.
        target_item_ids = _get_target_item_ids(batch, self.cfg.target_item_key)
        if target_item_ids is not None:
            target_item_ids = target_item_ids.to(device).long().reshape(bsz, 1)
            candidate_item_ids = torch.where(
                trajectories.is_positive.to(device).bool(),
                target_item_ids.expand_as(candidate_item_ids),
                candidate_item_ids,
            )

        return candidate_item_ids
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.sasrec_model.eval()
        return self

    def transform_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        return transform_sasrec_scores(raw_scores, self.cfg)


def transform_sasrec_scores(raw_scores: torch.Tensor, cfg: SASRecRewardConfig) -> torch.Tensor:
    """Map raw SASRec scores to strictly positive rewards (``cfg.transform``).

    Module-level so consumers that rebuild SASRec scores without the teacher
    (e.g. the factorized reward cache) apply exactly the reward model's math.
    """
    scores = raw_scores.float()
    scores = (scores + float(cfg.score_shift)) / max(float(cfg.temperature), 1e-12)

    if cfg.transform == "softplus":
        rewards = F.softplus(scores)
    elif cfg.transform == "sigmoid":
        rewards = torch.sigmoid(scores)
    elif cfg.transform == "identity":
        rewards = scores
    elif cfg.transform == "exp":
        rewards = torch.exp(scores.clamp(max=50.0))
    elif cfg.transform == "minmax_batch":
        rewards = _minmax_batch(scores)
    elif cfg.transform == "rank_percentile":
        rewards = _rank_percentile(scores)
    else:
        raise ValueError(f"Unknown SASRec reward transform: {cfg.transform}")

    rewards = rewards * float(cfg.reward_scale)
    return rewards.clamp_min(float(cfg.reward_epsilon))


def _sasrec_item_vocab_size(model: FlowRecModel) -> int | None:
    item_vocab_size = getattr(model, "item_vocab_size", None)
    if item_vocab_size is None:
        cfg = getattr(model, "cfg", None)
        item_vocab_size = getattr(cfg, "item_vocab_size", None)
    if item_vocab_size is None:
        return None
    return max(int(item_vocab_size), 1)


def _get_target_item_ids(batch: Batch, preferred_key: str) -> torch.Tensor | None:
    for key in (preferred_key, "target_item_id", "target_item_ids"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _minmax_batch(scores: torch.Tensor) -> torch.Tensor:
    """Per-row min-max normalization over candidate scores."""
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))

    min_vals = torch.where(
        finite, scores, torch.full_like(scores, float("inf"))
    ).amin(dim=1, keepdim=True)
    max_vals = torch.where(
        finite, scores, torch.full_like(scores, -float("inf"))
    ).amax(dim=1, keepdim=True)

    span = (max_vals - min_vals).clamp_min(1e-12)
    rewards = (safe_scores - min_vals) / span
    rewards = torch.where(finite, rewards, torch.zeros_like(rewards))
    return rewards


def _rank_percentile(scores: torch.Tensor) -> torch.Tensor:
    """Convert candidate scores to percentile rewards per row."""
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
