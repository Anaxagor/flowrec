from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


@dataclass(frozen=True, slots=True)
class SimilarityRewardConfig:
    """Terminal similarity reward for semantic-ID trajectories.

    If T is a trajectory (token sequence) and S is the correct target
    semantic-ID sequence, then reward equals the number of positions i such
    that T_i == S_i.

    All-zero / padding positions are masked out using
    ``trajectories.candidate_token_mask`` if provided.

    Output:
      rewards with shape [B, num_candidates] (float32, >= 0).
    """

    reward_epsilon: float = 1e-6

    # batch key for the correct semantic target sequence
    target_token_key: str = "target_token_ids"


class SimilarityRewardModel(nn.Module):
    def __init__(self, cfg: SimilarityRewardConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SimilarityRewardConfig()

    def forward(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        if self.cfg.target_token_key not in batch:
            raise KeyError(
                f"Batch is missing {self.cfg.target_token_key!r}; cannot compute similarity reward."
            )
        target = batch[self.cfg.target_token_key]
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)

        # trajectories.candidate_token_ids: [B, N, L]
        cand = trajectories.candidate_token_ids
        if cand.ndim != 3:
            raise ValueError(
                "candidate_token_ids must have shape [B, num_candidates, L], "
                f"got {tuple(cand.shape)}"
            )

        bsz, num_candidates, L = cand.shape

        # target: [B, L]
        if target.ndim != 2 or target.shape[:1] != (bsz,) or target.shape[1] != L:
            raise ValueError(
                "target_token_ids must have shape [B, L] matching trajectories. "
                f"target={tuple(target.shape)} cand={tuple(cand.shape)}"
            )

        target = target.to(cand.device).long().unsqueeze(1)  # [B, 1, L]

        mask = (
            trajectories.candidate_token_mask
            if getattr(trajectories, "candidate_token_mask", None) is not None
            else None
        )
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            if mask.shape != (bsz, num_candidates, L):
                raise ValueError(
                    "candidate_token_mask must have shape [B, num_candidates, L]; "
                    f"got {tuple(mask.shape)}"
                )
            mask = mask.to(cand.device).bool()

        eq = cand.long() == target  # [B, N, L]
        if mask is not None:
            eq = eq & mask

        rewards = eq.sum(dim=-1).to(torch.float32)  # [B, N]
        rewards = rewards.clamp_min(0.0)
        rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
        # make [0, 1] scaling
        rewards /= L;
        return rewards
