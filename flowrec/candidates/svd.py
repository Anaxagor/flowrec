from __future__ import annotations

from dataclasses import dataclass

import torch

from flowrec.candidates.item_model import (
    ItemModelSemanticCandidateConfig,
    ItemModelSemanticIDCandidateSampler,
)
from flowrec.models.base import FlowRecModel


@dataclass(frozen=True, slots=True)
class SVDSemanticCandidateConfig:
    """Backward-compatible config for score-based SVD semantic-ID sampling."""

    num_negative_samples: int = 1
    include_positive: bool = True
    avoid_positive: bool = True
    oversample_factor: int = 4
    easy_negative_steps: int = 0
    exclude_history: bool = True
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True
    max_resample_attempts: int = 3


class SVDSemanticIDCandidateSampler(ItemModelSemanticIDCandidateSampler):
    """Backward-compatible wrapper around the generic score-based sampler."""

    def __init__(
        self,
        svd_model: FlowRecModel,
        *,
        item_semantic_ids: torch.Tensor,
        item_id_to_semantic_row: torch.Tensor,
        cfg: SVDSemanticCandidateConfig | None = None,
    ) -> None:
        cfg = cfg or SVDSemanticCandidateConfig()
        super().__init__(
            svd_model,
            item_semantic_ids=item_semantic_ids,
            item_id_to_semantic_row=item_id_to_semantic_row,
            cfg=ItemModelSemanticCandidateConfig(
                num_negative_samples=cfg.num_negative_samples,
                include_positive=cfg.include_positive,
                avoid_positive=cfg.avoid_positive,
                oversample_factor=cfg.oversample_factor,
                easy_negative_steps=cfg.easy_negative_steps,
                exclude_history=cfg.exclude_history,
                pad_token_id=cfg.pad_token_id,
                semantic_ids_are_fixed_length=cfg.semantic_ids_are_fixed_length,
                max_resample_attempts=cfg.max_resample_attempts,
                sampler_name="svd_semantic_id",
            ),
        )
