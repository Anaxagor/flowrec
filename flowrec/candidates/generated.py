from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


@dataclass(frozen=True, slots=True)
class GeneratedSemanticCandidateConfig:
    """Configuration for model-generated hard negative semantic IDs."""

    num_negative_samples: int = 1
    include_positive: bool = True
    num_beams: int = 10
    num_return_sequences: int | None = None
    oversample_factor: int = 4
    constrain_valid_sequences: bool = False
    exclude_seen_items: bool = False
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True


class GeneratedSemanticIDCandidateSampler:
    """Build positive + generated hard-negative semantic-ID trajectories.

    Generated negatives are model predictions that are not equal to the observed
    target. If generation yields too few usable negatives for a row, random
    negatives fill the remaining slots to keep a dense [B, N, L] tensor.
    """

    def __init__(self, cfg: GeneratedSemanticCandidateConfig | None = None) -> None:
        self.cfg = cfg or GeneratedSemanticCandidateConfig()
        # The fallback has no item-id mapping, so it cannot exclude history;
        # generation-side exclusion is handled by exclude_seen_items.
        self._fallback_sampler = SemanticIDCandidateSampler(
            CandidateSamplerConfig(
                mode="random",
                num_negatives=self.cfg.num_negative_samples,
                include_positive=False,
                exclude_history=False,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids_are_fixed_length=self.cfg.semantic_ids_are_fixed_length,
            )
        )

    def sample(
        self,
        batch: Batch,
        *,
        model: FlowRecModel | None = None,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> SemanticTrajectoryBatch:
        if model is None:
            raise ValueError("GeneratedSemanticIDCandidateSampler requires a model.")
        if "target_token_ids" not in batch:
            raise KeyError("GeneratedSemanticIDCandidateSampler requires batch['target_token_ids'].")

        target_token_ids = batch["target_token_ids"]
        if not torch.is_tensor(target_token_ids):
            raise TypeError("batch['target_token_ids'] must be a torch.Tensor.")
        target_token_ids = target_token_ids.long()
        device = target_token_ids.device
        bsz, sid_len = target_token_ids.shape

        target_token_mask = batch.get("target_token_mask")
        if target_token_mask is None or not torch.is_tensor(target_token_mask):
            target_token_mask = torch.ones_like(target_token_ids, dtype=torch.bool, device=device)
        else:
            target_token_mask = target_token_mask.to(device).bool()

        neg_k = int(self.cfg.num_negative_samples)
        if neg_k < 0:
            raise ValueError("num_negative_samples must be non-negative.")

        candidates: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        is_positive_parts: list[torch.Tensor] = []
        candidate_rows: list[torch.Tensor] = []

        if self.cfg.include_positive:
            candidates.append(target_token_ids.unsqueeze(1))
            masks.append(target_token_mask.unsqueeze(1))
            is_positive_parts.append(torch.ones((bsz, 1), device=device, dtype=torch.bool))
            candidate_rows.append(torch.full((bsz, 1), -1, device=device, dtype=torch.long))

        if neg_k > 0:
            fallback = self._fallback_sampler.sample(batch, model=model, generator=generator)
            negative_tokens = fallback.candidate_token_ids[:, :neg_k].clone()
            negative_rows = (
                fallback.candidate_item_row_ids[:, :neg_k].clone()
                if fallback.candidate_item_row_ids is not None
                else torch.full((bsz, neg_k), -1, device=device, dtype=torch.long)
            )

            generated = self._generate(model, batch, requested_negatives=neg_k)
            negative_tokens, negative_rows = self._fill_from_generated(
                negative_tokens=negative_tokens,
                negative_rows=negative_rows,
                generated=generated,
                target_token_ids=target_token_ids,
            )

            if self.cfg.semantic_ids_are_fixed_length:
                negative_mask = torch.ones_like(negative_tokens, dtype=torch.bool)
            else:
                negative_mask = negative_tokens.ne(int(self.cfg.pad_token_id))

            candidates.append(negative_tokens)
            masks.append(negative_mask)
            is_positive_parts.append(torch.zeros((bsz, neg_k), device=device, dtype=torch.bool))
            candidate_rows.append(negative_rows)

        if not candidates:
            raise ValueError("No candidates were built. Enable include_positive or num_negative_samples > 0.")

        return SemanticTrajectoryBatch(
            candidate_token_ids=torch.cat(candidates, dim=1),
            candidate_token_mask=torch.cat(masks, dim=1),
            is_positive=torch.cat(is_positive_parts, dim=1),
            candidate_item_row_ids=torch.cat(candidate_rows, dim=1) if candidate_rows else None,
            extra={"sampler": "generated_semantic_id"},
        )

    @torch.no_grad()
    def _generate(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        requested_negatives: int,
    ) -> torch.Tensor:
        num_return_sequences = self.cfg.num_return_sequences
        if num_return_sequences is None:
            num_return_sequences = max(
                requested_negatives,
                requested_negatives * int(self.cfg.oversample_factor),
            )
        num_beams = max(int(self.cfg.num_beams), int(num_return_sequences))

        was_training = model.training
        model.eval()
        try:
            return model.generate(
                batch,
                num_beams=num_beams,
                num_return_sequences=int(num_return_sequences),
                constrain_valid_sequences=self.cfg.constrain_valid_sequences,
                exclude_seen_items=self.cfg.exclude_seen_items,
            )
        finally:
            model.train(was_training)

    def _fill_from_generated(
        self,
        *,
        negative_tokens: torch.Tensor,
        negative_rows: torch.Tensor,
        generated: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, neg_k, sid_len = negative_tokens.shape
        if generated.ndim == 2:
            generated = generated.unsqueeze(1)
        if generated.ndim != 3 or generated.size(0) != bsz or generated.size(2) != sid_len:
            raise ValueError(
                "Generated candidates must have shape [B, K, L]. "
                f"Got {tuple(generated.shape)} for target shape {tuple(target_token_ids.shape)}."
            )

        for b in range(bsz):
            seen = {tuple(target_token_ids[b].tolist())}
            write_idx = 0
            for cand in generated[b]:
                key = tuple(cand.tolist())
                if key in seen:
                    continue
                negative_tokens[b, write_idx] = cand
                negative_rows[b, write_idx] = -1
                seen.add(key)
                write_idx += 1
                if write_idx >= neg_k:
                    break
        return negative_tokens, negative_rows
