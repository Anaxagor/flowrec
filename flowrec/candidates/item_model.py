from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flowrec.candidates.random import RandomSemanticCandidateConfig, RandomSemanticIDCandidateSampler
from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


@dataclass(frozen=True, slots=True)
class ItemModelSemanticCandidateConfig:
    """Configuration for item-scoring-model semantic-ID candidate sampling."""

    num_negative_samples: int = 1
    include_positive: bool = True
    avoid_positive: bool = True
    oversample_factor: int = 4
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True
    max_resample_attempts: int = 3
    sampler_name: str = "item_model_semantic_id"


class ItemModelSemanticIDCandidateSampler:
    """Build positive + item-model-selected semantic-ID trajectories."""

    def __init__(
        self,
        item_model: FlowRecModel,
        *,
        item_semantic_ids: torch.Tensor,
        item_id_to_semantic_row: torch.Tensor,
        cfg: ItemModelSemanticCandidateConfig | None = None,
    ) -> None:
        self.cfg = cfg or ItemModelSemanticCandidateConfig()
        self.item_model = item_model
        self.item_semantic_ids = item_semantic_ids.long()
        self.item_id_to_semantic_row = item_id_to_semantic_row.long()
        self._fallback_sampler = RandomSemanticIDCandidateSampler(
            RandomSemanticCandidateConfig(
                num_negative_samples=self.cfg.num_negative_samples,
                include_positive=False,
                avoid_positive=self.cfg.avoid_positive,
                pad_token_id=self.cfg.pad_token_id,
                semantic_ids_are_fixed_length=self.cfg.semantic_ids_are_fixed_length,
                max_resample_attempts=self.cfg.max_resample_attempts,
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
        if "target_token_ids" not in batch:
            raise KeyError("ItemModelSemanticIDCandidateSampler requires batch['target_token_ids'].")

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
        candidate_item_ids_parts: list[torch.Tensor] = []

        target_item_ids = self._target_item_ids(batch, bsz, device)
        if self.cfg.include_positive:
            candidates.append(target_token_ids.unsqueeze(1))
            masks.append(target_token_mask.unsqueeze(1))
            is_positive_parts.append(torch.ones((bsz, 1), device=device, dtype=torch.bool))
            candidate_rows.append(torch.full((bsz, 1), -1, device=device, dtype=torch.long))
            candidate_item_ids_parts.append(target_item_ids.reshape(bsz, 1))

        if neg_k > 0:
            fallback = self._fallback_sampler.sample(batch, model=model, generator=generator)
            negative_tokens = fallback.candidate_token_ids[:, :neg_k].clone()
            negative_rows = (
                fallback.candidate_item_row_ids[:, :neg_k].clone()
                if fallback.candidate_item_row_ids is not None
                else torch.full((bsz, neg_k), -1, device=device, dtype=torch.long)
            )
            negative_item_ids = self._row_to_item_ids(negative_rows, device=device)

            generated_item_ids = self._generate_item_ids(batch, requested_negatives=neg_k).to(device)
            negative_tokens, negative_rows, negative_item_ids = self._fill_from_item_model(
                negative_tokens=negative_tokens,
                negative_rows=negative_rows,
                negative_item_ids=negative_item_ids,
                generated_item_ids=generated_item_ids,
                target_token_ids=target_token_ids,
                target_item_ids=target_item_ids,
            )

            negative_mask = (
                torch.ones_like(negative_tokens, dtype=torch.bool)
                if self.cfg.semantic_ids_are_fixed_length
                else negative_tokens.ne(int(self.cfg.pad_token_id))
            )
            candidates.append(negative_tokens)
            masks.append(negative_mask)
            is_positive_parts.append(torch.zeros((bsz, neg_k), device=device, dtype=torch.bool))
            candidate_rows.append(negative_rows)
            candidate_item_ids_parts.append(negative_item_ids)

        if not candidates:
            raise ValueError("No candidates were built. Enable include_positive or num_negative_samples > 0.")

        return SemanticTrajectoryBatch(
            candidate_token_ids=torch.cat(candidates, dim=1),
            candidate_token_mask=torch.cat(masks, dim=1),
            is_positive=torch.cat(is_positive_parts, dim=1),
            candidate_item_row_ids=torch.cat(candidate_rows, dim=1),
            extra={
                "sampler": self.cfg.sampler_name,
                "candidate_item_ids": torch.cat(candidate_item_ids_parts, dim=1),
            },
        )

    @torch.no_grad()
    def _generate_item_ids(self, batch: Batch, *, requested_negatives: int) -> torch.Tensor:
        num_return_sequences = max(
            requested_negatives,
            requested_negatives * int(self.cfg.oversample_factor),
        )
        device = self._batch_device(batch)
        if device is not None and self.item_model.device != device:
            self.item_model.to(device)
        was_training = self.item_model.training
        self.item_model.eval()
        try:
            return self.item_model.generate(batch, num_return_sequences=int(num_return_sequences))
        finally:
            self.item_model.train(was_training)

    @staticmethod
    def _batch_device(batch: Batch) -> torch.device | None:
        for value in batch.values():
            if torch.is_tensor(value):
                return value.device
        return None

    def _fill_from_item_model(
        self,
        *,
        negative_tokens: torch.Tensor,
        negative_rows: torch.Tensor,
        negative_item_ids: torch.Tensor,
        generated_item_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table = self.item_semantic_ids.to(negative_tokens.device)
        item_to_row = self.item_id_to_semantic_row.to(negative_tokens.device)
        bsz, neg_k, sid_len = negative_tokens.shape
        if generated_item_ids.ndim == 1:
            generated_item_ids = generated_item_ids.unsqueeze(1)
        if generated_item_ids.ndim != 2 or generated_item_ids.size(0) != bsz:
            raise ValueError(
                "Generated item IDs must have shape [B, K]. "
                f"Got {tuple(generated_item_ids.shape)} for batch size {bsz}."
            )

        for b in range(bsz):
            seen_tokens = {tuple(target_token_ids[b].tolist())}
            seen_items = {int(target_item_ids[b].item())}
            write_idx = 0
            for item_id_t in generated_item_ids[b]:
                item_id = int(item_id_t.item())
                if item_id in seen_items or item_id < 0 or item_id >= item_to_row.numel():
                    continue
                row = int(item_to_row[item_id].item())
                if row < 0 or row >= table.size(0):
                    continue
                tokens = table[row]
                if tokens.numel() != sid_len:
                    continue
                token_key = tuple(tokens.tolist())
                if token_key in seen_tokens:
                    continue
                negative_tokens[b, write_idx] = tokens
                negative_rows[b, write_idx] = row
                negative_item_ids[b, write_idx] = item_id
                seen_tokens.add(token_key)
                seen_items.add(item_id)
                write_idx += 1
                if write_idx >= neg_k:
                    break

        return negative_tokens, negative_rows, negative_item_ids

    def _target_item_ids(self, batch: Batch, bsz: int, device: torch.device) -> torch.Tensor:
        for key in ("target_item_ids", "target_item_id"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value.to(device).long().reshape(bsz)
        return torch.full((bsz,), -1, device=device, dtype=torch.long)

    def _row_to_item_ids(self, rows: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        item_to_row = self.item_id_to_semantic_row.to(device)
        if item_to_row.numel() == 0:
            return torch.full_like(rows, -1)
        row_to_item = torch.full(
            (int(self.item_semantic_ids.size(0)),),
            -1,
            device=device,
            dtype=torch.long,
        )
        valid_items = (item_to_row >= 0) & (item_to_row < row_to_item.numel())
        item_ids = valid_items.nonzero(as_tuple=True)[0]
        row_to_item[item_to_row[valid_items]] = item_ids
        safe_rows = rows.clamp_min(0).clamp_max(row_to_item.numel() - 1)
        mapped = row_to_item[safe_rows]
        return torch.where(rows >= 0, mapped, torch.full_like(mapped, -1))
