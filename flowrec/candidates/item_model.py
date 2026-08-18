from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from flowrec.candidates.random import RandomSemanticCandidateConfig, RandomSemanticIDCandidateSampler
from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


@dataclass(frozen=True, slots=True)
class ItemModelSemanticCandidateConfig:
    """Configuration for item-scoring-model semantic-ID candidate sampling."""

    num_negative_samples: int = 1
    include_positive: bool = True
    easy_negative_steps: int = 0
    # Keep this many of the negatives purely random (uniform over the catalog)
    # instead of score-based. Mixing hard and random negatives preserves a
    # push-down signal across the whole catalog and protects coverage.
    num_random_negatives: int = 0
    candidate_sampling_temperature: float = 1.0
    candidate_pool_size: int = 1024
    score_chunk_size: int = 50_000
    pool_strategy: Literal["full_catalog", "random_pool"] = "random_pool"


class ItemModelSemanticIDCandidateSampler:
    """Build positive + score-based item-model semantic-ID trajectories.

    The sampler scores the valid semantic catalog with an external item model
    such as SASRec or SVD.  During the initial curriculum phase it selects
    low-scoring easy negatives; afterwards it selects high-scoring hard
    negatives.  Random negatives are used only as a fallback when a row has too
    few valid scored candidates.
    """

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
        self._pool_item_ids = self._build_candidate_pool(
            self.item_id_to_semantic_row,
            num_semantic_rows=int(self.item_semantic_ids.size(0)),
        )
        self._fallback_sampler = RandomSemanticIDCandidateSampler(
            RandomSemanticCandidateConfig(
                num_negative_samples=self.cfg.num_negative_samples,
                include_positive=False,
            )
        )

    def sample(
        self,
        batch: Batch,
        *,
        model: FlowRecModel | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
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
            fallback = self._fallback_sampler.sample(
                batch,
                model=model,
                item_semantic_ids=self.item_semantic_ids,
                generator=generator,
            )
            negative_tokens = fallback.candidate_token_ids[:, :neg_k].clone()
            negative_rows = (
                fallback.candidate_item_row_ids[:, :neg_k].clone()
                if fallback.candidate_item_row_ids is not None
                else torch.full((bsz, neg_k), -1, device=device, dtype=torch.long)
            )
            negative_item_ids = self._row_to_item_ids(negative_rows, device=device)

            num_random = min(max(int(self.cfg.num_random_negatives), 0), neg_k)
            num_scored = neg_k - num_random
            if num_scored > 0:
                # Scored picks overwrite the first slots; the remaining slots
                # keep the uniform-random negatives from the fallback sampler.
                scored_item_ids = self._score_and_select_item_ids(
                    batch,
                    target_item_ids=target_item_ids,
                    requested_negatives=num_scored,
                    step=step,
                    generator=generator,
                ).to(device)
                negative_tokens, negative_rows, negative_item_ids = self._fill_from_item_ids(
                    negative_tokens=negative_tokens,
                    negative_rows=negative_rows,
                    negative_item_ids=negative_item_ids,
                    selected_item_ids=scored_item_ids,
                    target_token_ids=target_token_ids,
                    target_item_ids=target_item_ids,
                )
            negative_tokens, negative_rows, negative_item_ids = self._deduplicate_negative_candidates(
                batch=batch,
                negative_tokens=negative_tokens,
                negative_rows=negative_rows,
                negative_item_ids=negative_item_ids,
                target_token_ids=target_token_ids,
                target_item_ids=target_item_ids,
                generator=generator,
            )

            negative_mask = torch.ones_like(negative_tokens, dtype=torch.bool)
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
                "sampler": "item_model_semantic_id",
                "sampler_phase": "easy" if self._use_easy_negatives(step) else "hard",
                "candidate_item_ids": torch.cat(candidate_item_ids_parts, dim=1),
            },
        )

    @torch.no_grad()
    def _score_and_select_item_ids(
        self,
        batch: Batch,
        *,
        target_item_ids: torch.Tensor,
        requested_negatives: int,
        step: int | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = self._batch_device(batch)
        if device is not None and self.item_model.device != device:
            self.item_model.to(device)
        if device is None:
            device = self.item_model.device

        pool_item_ids = self._sample_pool_item_ids(device=device, generator=generator)
        if pool_item_ids.numel() == 0:
            bsz = target_item_ids.numel()
            return torch.full((bsz, requested_negatives), -1, device=device, dtype=torch.long)

        self.item_model.eval()
        scores = self._score_pool(batch, pool_item_ids=pool_item_ids)

        if scores.ndim != 2 or scores.size(0) != target_item_ids.numel():
            raise ValueError(
                "Item model scores must have shape [B, num_candidates]. "
                f"Got {tuple(scores.shape)} for batch size {target_item_ids.numel()}."
            )
        if scores.size(1) != pool_item_ids.numel():
            raise ValueError(
                "Item model score width does not match candidate pool: "
                f"scores={scores.size(1)}, pool={pool_item_ids.numel()}."
            )

        scores = scores.to(device).float()
        scores = self._mask_invalid_scores(
            scores=scores,
            pool_item_ids=pool_item_ids,
            target_item_ids=target_item_ids.to(device),
            batch=batch,
            easy=self._use_easy_negatives(step),
        )
        return self._select_item_ids_from_scores(
            scores=scores,
            pool_item_ids=pool_item_ids,
            requested_negatives=requested_negatives,
            easy=self._use_easy_negatives(step),
            generator=generator,
        )

    def _sample_pool_item_ids(
        self,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        pool_item_ids = self._pool_item_ids.to(device)
        if self.cfg.pool_strategy == "full_catalog":
            return pool_item_ids
        if self.cfg.pool_strategy != "random_pool":
            raise ValueError(f"Unknown pool_strategy={self.cfg.pool_strategy!r}.")

        pool_size = int(self.cfg.candidate_pool_size)
        if pool_size <= 0 or pool_size >= pool_item_ids.numel():
            return pool_item_ids
        order = self._randperm(pool_item_ids.numel(), device=device, generator=generator)[:pool_size]
        return pool_item_ids[order]

    def _score_pool(self, batch: Batch, *, pool_item_ids: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.cfg.score_chunk_size)
        if chunk_size <= 0 or chunk_size >= pool_item_ids.numel():
            return self._score_shared_pool(batch, pool_item_ids)

        chunks: list[torch.Tensor] = []
        for start in range(0, pool_item_ids.numel(), chunk_size):
            chunk_ids = pool_item_ids[start : start + chunk_size]
            chunks.append(self._score_shared_pool(batch, chunk_ids))
        return torch.cat(chunks, dim=1)

    def _score_shared_pool(self, batch: Batch, item_ids: torch.Tensor) -> torch.Tensor:
        batch_size = int(batch["user_ids"].size(0))
        if item_ids.ndim == 1 and item_ids.numel() == batch_size:
            item_ids = item_ids.unsqueeze(0).expand(batch_size, -1)
        return self.item_model.score_items(batch, candidate_item_ids=item_ids)

    @staticmethod
    def _batch_device(batch: Batch) -> torch.device | None:
        for value in batch.values():
            if torch.is_tensor(value):
                return value.device
        return None

    def _fill_from_item_ids(
        self,
        *,
        negative_tokens: torch.Tensor,
        negative_rows: torch.Tensor,
        negative_item_ids: torch.Tensor,
        selected_item_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table = self.item_semantic_ids.to(negative_tokens.device)
        item_to_row = self.item_id_to_semantic_row.to(negative_tokens.device)
        bsz, neg_k, sid_len = negative_tokens.shape
        if selected_item_ids.ndim == 1:
            selected_item_ids = selected_item_ids.unsqueeze(1)
        if selected_item_ids.ndim != 2 or selected_item_ids.size(0) != bsz:
            raise ValueError(
                "Selected item IDs must have shape [B, K]. "
                f"Got {tuple(selected_item_ids.shape)} for batch size {bsz}."
            )

        for b in range(bsz):
            seen_tokens = {tuple(target_token_ids[b].tolist())}
            seen_items = {int(target_item_ids[b].item())}
            write_idx = 0
            for item_id_t in selected_item_ids[b]:
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

    def _deduplicate_negative_candidates(
        self,
        *,
        batch: Batch,
        negative_tokens: torch.Tensor,
        negative_rows: torch.Tensor,
        negative_item_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table = self.item_semantic_ids.to(negative_tokens.device)
        item_to_row = self.item_id_to_semantic_row.to(negative_tokens.device)
        pool_item_ids = self._pool_item_ids.to(negative_tokens.device)
        bsz, neg_k, sid_len = negative_tokens.shape

        for b in range(bsz):
            seen_tokens = {tuple(target_token_ids[b].tolist())}
            target_item_id = int(target_item_ids[b].item())
            seen_items = {target_item_id} if target_item_id >= 0 else set()
            history_items = self._history_items_for_row(batch, b, device=negative_tokens.device)

            for j in range(neg_k):
                item_id = int(negative_item_ids[b, j].item())
                token_key = tuple(negative_tokens[b, j].tolist())
                duplicate = item_id in seen_items or token_key in seen_tokens
                invalid = (
                    item_id < 0
                    or item_id >= item_to_row.numel()
                    or item_id in history_items
                    or duplicate
                )
                if not invalid:
                    row = int(item_to_row[item_id].item())
                    invalid = row < 0 or row >= table.size(0)

                if invalid:
                    replacement = self._replacement_item_for_slot(
                        pool_item_ids=pool_item_ids,
                        table=table,
                        item_to_row=item_to_row,
                        sid_len=sid_len,
                        seen_items=seen_items,
                        seen_tokens=seen_tokens,
                        history_items=history_items,
                        generator=generator,
                    )
                    if replacement is None:
                        negative_item_ids[b, j] = -1
                        negative_rows[b, j] = -1
                        negative_tokens[b, j] = 0
                        continue
                    item_id, row, tokens = replacement
                    negative_item_ids[b, j] = item_id
                    negative_rows[b, j] = row
                    negative_tokens[b, j] = tokens
                    token_key = tuple(tokens.tolist())

                seen_items.add(int(negative_item_ids[b, j].item()))
                seen_tokens.add(token_key)

        return negative_tokens, negative_rows, negative_item_ids

    def _replacement_item_for_slot(
        self,
        *,
        pool_item_ids: torch.Tensor,
        table: torch.Tensor,
        item_to_row: torch.Tensor,
        sid_len: int,
        seen_items: set[int],
        seen_tokens: set[tuple[int, ...]],
        history_items: set[int],
        generator: torch.Generator | None,
    ) -> tuple[int, int, torch.Tensor] | None:
        if pool_item_ids.numel() == 0:
            return None
        order = self._randperm(pool_item_ids.numel(), device=pool_item_ids.device, generator=generator)
        for idx in order.tolist():
            item_id = int(pool_item_ids[idx].item())
            if item_id in seen_items or item_id in history_items:
                continue
            if item_id < 0 or item_id >= item_to_row.numel():
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
            return item_id, row, tokens
        return None

    @staticmethod
    def _randperm(
        n: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        try:
            return torch.randperm(n, device=device, generator=generator)
        except RuntimeError:
            return torch.randperm(n, device=device)

    @staticmethod
    def _multinomial(
        probs: torch.Tensor,
        *,
        num_samples: int,
        replacement: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        try:
            return torch.multinomial(
                probs,
                num_samples=num_samples,
                replacement=replacement,
                generator=generator,
            )
        except RuntimeError:
            return torch.multinomial(
                probs,
                num_samples=num_samples,
                replacement=replacement,
            )

    def _history_items_for_row(self, batch: Batch, row: int, *, device: torch.device) -> set[int]:
        if "history_item_ids" not in batch:
            return set()
        history = batch["history_item_ids"]
        if not torch.is_tensor(history):
            return set()
        history = history.to(device).long()
        history_mask = batch.get("history_mask")
        if torch.is_tensor(history_mask):
            mask = history_mask.to(device).bool()
        else:
            mask = history.ne(0)
        values = history[row][mask[row] & (history[row] >= 0)]
        return {int(value) for value in values.tolist()}

    def _mask_invalid_scores(
        self,
        *,
        scores: torch.Tensor,
        pool_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
        batch: Batch,
        easy: bool,
    ) -> torch.Tensor:
        invalid_value = float("inf") if easy else -float("inf")
        scores = scores.clone()
        scores = scores.masked_fill(~torch.isfinite(scores), invalid_value)

        positive = pool_item_ids.unsqueeze(0).eq(target_item_ids.reshape(-1, 1))
        scores = scores.masked_fill(positive, invalid_value)

        if "history_item_ids" in batch:
            history = batch["history_item_ids"].to(scores.device).long()
            history_mask = batch.get("history_mask")
            if torch.is_tensor(history_mask):
                history_mask = history_mask.to(scores.device).bool()
            else:
                history_mask = history.ne(0)
            history = history.masked_fill(~history_mask, -1)
            for b in range(scores.size(0)):
                seen_items = history[b][history[b] >= 0]
                if seen_items.numel() == 0:
                    continue
                scores[b] = scores[b].masked_fill(torch.isin(pool_item_ids, seen_items), invalid_value)

        return scores

    def _select_item_ids_from_scores(
        self,
        *,
        scores: torch.Tensor,
        pool_item_ids: torch.Tensor,
        requested_negatives: int,
        easy: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        bsz = scores.size(0)
        selected = torch.full(
            (bsz, requested_negatives),
            -1,
            device=scores.device,
            dtype=torch.long,
        )
        for b in range(bsz):
            valid = torch.isfinite(scores[b])
            num_valid = int(valid.sum().item())
            if num_valid <= 0:
                continue
            k = min(int(requested_negatives), num_valid)
            row_scores = scores[b, valid]
            row_item_ids = pool_item_ids[valid]
            logits = -row_scores if easy else row_scores
            temperature = max(float(self.cfg.candidate_sampling_temperature), 1e-12)
            probs = torch.softmax(logits / temperature, dim=0)
            if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
                probs = torch.ones_like(row_scores) / row_scores.numel()
            sampled = self._multinomial(
                probs,
                num_samples=k,
                replacement=False,
                generator=generator,
            )
            selected[b, :k] = row_item_ids[sampled]
        return selected

    def _use_easy_negatives(self, step: int | None) -> bool:
        easy_steps = int(self.cfg.easy_negative_steps)
        return easy_steps > 0 and step is not None and int(step) < easy_steps

    @staticmethod
    def _build_candidate_pool(
        item_id_to_semantic_row: torch.Tensor,
        *,
        num_semantic_rows: int,
    ) -> torch.Tensor:
        item_id_to_semantic_row = item_id_to_semantic_row.long()
        valid = (item_id_to_semantic_row >= 0) & (item_id_to_semantic_row < int(num_semantic_rows))
        return valid.nonzero(as_tuple=True)[0].long()

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
