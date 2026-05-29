from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    keep_scorer_eval: bool = True
    sampling_temperature: float = 1.0
    candidate_pool_size: int = 1024
    score_chunk_size: int = 50_000
    pool_strategy: Literal["full_catalog", "random_pool"] = "random_pool"


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
                keep_scorer_eval=cfg.keep_scorer_eval,
                sampling_temperature=cfg.sampling_temperature,
                candidate_pool_size=cfg.candidate_pool_size,
                score_chunk_size=cfg.score_chunk_size,
                pool_strategy=cfg.pool_strategy,
                sampler_name="svd_semantic_id",
            ),
        )
