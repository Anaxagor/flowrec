from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class SemanticTrajectoryBatch:
    """Batch of semantic-ID trajectories used by GFlowGR.

    A trajectory is a semantic-ID token sequence for one candidate item.

    Shapes:
        candidate_token_ids:  [B, N, L]
        candidate_token_mask: [B, N, L]
        is_positive:          [B, N]
        candidate_item_row_ids: optional [B, N], rows into an item semantic-ID table.
    """

    candidate_token_ids: torch.Tensor
    candidate_token_mask: torch.Tensor
    is_positive: torch.Tensor
    candidate_item_row_ids: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.candidate_token_ids.size(0))

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_token_ids.size(1))

    @property
    def trajectory_length(self) -> int:
        return int(self.candidate_token_ids.size(2))

    def to(self, device: torch.device | str) -> "SemanticTrajectoryBatch":
        return SemanticTrajectoryBatch(
            candidate_token_ids=self.candidate_token_ids.to(device),
            candidate_token_mask=self.candidate_token_mask.to(device),
            is_positive=self.is_positive.to(device),
            candidate_item_row_ids=(
                self.candidate_item_row_ids.to(device)
                if self.candidate_item_row_ids is not None
                else None
            ),
            extra=self.extra,
        )
