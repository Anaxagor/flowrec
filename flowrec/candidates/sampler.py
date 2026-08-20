"""Unified semantic-ID candidate sampler.

One sampler, exactly three modes, all operating on the full item catalog
(every item with a valid semantic-ID row):

    - ``curriculum``: score the catalog with an item model and sample
      negatives from a softmax whose difficulty ramps from easy (lowest
      scores) to hard (highest scores) over ``curriculum_steps`` training
      steps. ``curriculum_steps=0`` starts at full difficulty immediately.
    - ``top_n``: deterministic hardest negatives — the ``num_negatives``
      highest-scoring allowed items.
    - ``random``: uniform over all allowed items. Needs no scorer.

Every mode honors the same exclusion contract: the positive target is never
sampled as a negative, and with ``exclude_history=True`` (default) no item
from the user's history is sampled either. Rewards can therefore assume that
a non-positive candidate is outside the user's positive history whenever the
flag is set (see flowrec/rewards/interaction.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

SamplerMode = Literal["curriculum", "top_n", "random"]

_SCORE_CHUNK = 50_000


@dataclass(frozen=True, slots=True)
class CandidateSamplerConfig:
    mode: SamplerMode = "random"
    num_negatives: int = 1
    include_positive: bool = True
    exclude_history: bool = True
    # curriculum only: steps to ramp from easy to hard negatives.
    curriculum_steps: int = 0
    # curriculum only: softmax temperature over item-model scores.
    temperature: float = 1.0
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True


class SemanticIDCandidateSampler:
    """Build positive + negative semantic-ID trajectories for GFlowGR.

    ``item_semantic_ids`` is the [num_rows, sid_len] semantic-ID table (it can
    also be resolved from ``model.item_semantic_ids`` at sample time).
    ``item_id_to_semantic_row`` maps catalog item IDs to table rows; it is
    required for the scored modes and for history exclusion, and enables
    ``extra["candidate_item_ids"]`` for store-backed rewards.
    """

    def __init__(
        self,
        cfg: CandidateSamplerConfig | None = None,
        *,
        item_semantic_ids: torch.Tensor | None = None,
        item_id_to_semantic_row: torch.Tensor | None = None,
        scorer: FlowRecModel | None = None,
    ) -> None:
        self.cfg = cfg or CandidateSamplerConfig()
        if self.cfg.mode not in ("curriculum", "top_n", "random"):
            raise ValueError(
                f"Unknown sampler mode {self.cfg.mode!r}. "
                "Expected curriculum, top_n, or random."
            )
        if self.cfg.num_negatives < 0:
            raise ValueError("num_negatives must be non-negative.")
        if self.cfg.mode in ("curriculum", "top_n") and scorer is None:
            raise ValueError(f"mode={self.cfg.mode!r} requires a scorer item model.")
        self.scorer = scorer
        self.item_semantic_ids = (
            item_semantic_ids.long() if item_semantic_ids is not None else None
        )
        self.item_id_to_semantic_row = (
            item_id_to_semantic_row.long()
            if item_id_to_semantic_row is not None
            else None
        )
        if self.cfg.mode in ("curriculum", "top_n") and self.item_id_to_semantic_row is None:
            raise ValueError(
                f"mode={self.cfg.mode!r} requires item_id_to_semantic_row so scored "
                "items can be mapped onto semantic-ID rows."
            )

    # -- public API ----------------------------------------------------------

    def sample(
        self,
        batch: Batch,
        *,
        model: FlowRecModel | None = None,
        item_semantic_ids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
        **_: Any,
    ) -> SemanticTrajectoryBatch:
        target_tokens, target_mask = self._target_tokens(batch)
        device = target_tokens.device
        bsz, sid_len = target_tokens.shape

        table = self._resolve_table(model=model, item_semantic_ids=item_semantic_ids)
        table = table.to(device).long()
        if table.ndim != 2 or table.size(1) != sid_len:
            raise ValueError(
                "item_semantic_ids must have shape [num_rows, sid_len] matching "
                f"target_token_ids; got {tuple(table.shape)} vs sid_len={sid_len}."
            )
        num_rows = int(table.size(0))
        row_to_item = self._row_to_item(num_rows, device)
        target_items = self._target_item_ids(batch, bsz, device)

        parts_tokens: list[torch.Tensor] = []
        parts_mask: list[torch.Tensor] = []
        parts_positive: list[torch.Tensor] = []
        parts_rows: list[torch.Tensor] = []
        parts_items: list[torch.Tensor] = []

        if self.cfg.include_positive:
            parts_tokens.append(target_tokens.unsqueeze(1))
            parts_mask.append(target_mask.unsqueeze(1))
            parts_positive.append(torch.ones((bsz, 1), device=device, dtype=torch.bool))
            parts_rows.append(self._rows_for_items(target_items, num_rows).unsqueeze(1))
            parts_items.append(target_items.unsqueeze(1))

        neg_k = int(self.cfg.num_negatives)
        if neg_k > 0:
            allowed = self._allowed_mask(
                batch,
                table=table,
                row_to_item=row_to_item,
                target_tokens=target_tokens,
                target_items=target_items,
            )
            rows = self._sample_rows(
                batch,
                allowed=allowed,
                row_to_item=row_to_item,
                neg_k=neg_k,
                step=step,
                generator=generator,
            )
            filled = rows.ge(0)
            safe_rows = rows.clamp_min(0)
            tokens = table[safe_rows]
            # A drawn row can still collide with the positive when the target
            # item is unmapped (no item_id mapping); drop those slots.
            collides = tokens.eq(target_tokens.unsqueeze(1)).all(dim=-1) & filled
            rows = torch.where(collides, torch.full_like(rows, -1), rows)
            filled = filled & ~collides
            safe_rows = rows.clamp_min(0)
            tokens = torch.where(
                filled.unsqueeze(-1), tokens, torch.full_like(tokens, int(self.cfg.pad_token_id))
            )
            if self.cfg.semantic_ids_are_fixed_length:
                mask = filled.unsqueeze(-1).expand_as(tokens).clone()
            else:
                mask = tokens.ne(int(self.cfg.pad_token_id)) & filled.unsqueeze(-1)
            items = torch.where(filled, row_to_item[safe_rows], torch.full_like(rows, -1))

            parts_tokens.append(tokens)
            parts_mask.append(mask)
            parts_positive.append(torch.zeros((bsz, neg_k), device=device, dtype=torch.bool))
            parts_rows.append(rows)
            parts_items.append(items)

        if not parts_tokens:
            raise ValueError(
                "No candidates were built. Enable include_positive or num_negatives > 0."
            )

        extra: dict[str, Any] = {
            "sampler": "semantic_candidates",
            "sampler_mode": self.cfg.mode,
        }
        if self.item_id_to_semantic_row is not None:
            # Without the mapping, item ids would be all -1 and shadow the
            # reward models' candidate_item_row_ids fallback.
            extra["candidate_item_ids"] = torch.cat(parts_items, dim=1)
        if self.cfg.mode == "curriculum":
            extra["curriculum_alpha"] = self._curriculum_alpha(step)

        return SemanticTrajectoryBatch(
            candidate_token_ids=torch.cat(parts_tokens, dim=1),
            candidate_token_mask=torch.cat(parts_mask, dim=1),
            is_positive=torch.cat(parts_positive, dim=1),
            candidate_item_row_ids=torch.cat(parts_rows, dim=1),
            extra=extra,
        )

    # -- exclusion mask ------------------------------------------------------

    def _allowed_mask(
        self,
        batch: Batch,
        *,
        table: torch.Tensor,
        row_to_item: torch.Tensor,
        target_tokens: torch.Tensor,
        target_items: torch.Tensor,
    ) -> torch.Tensor:
        """[B, num_rows] mask of rows a negative may be drawn from."""
        device = table.device
        bsz = target_tokens.size(0)
        num_rows = int(table.size(0))

        if self.item_id_to_semantic_row is not None:
            valid = row_to_item.ge(0)
        elif self.cfg.semantic_ids_are_fixed_length:
            valid = torch.ones(num_rows, device=device, dtype=torch.bool)
        else:
            valid = table.ne(int(self.cfg.pad_token_id)).any(dim=-1)
        allowed = valid.unsqueeze(0).repeat(bsz, 1)

        # The positive target is never a valid negative.
        if self.item_id_to_semantic_row is not None and bool(target_items.ge(0).any()):
            allowed = self._scatter_exclude(
                allowed, self._rows_for_items(target_items, num_rows).unsqueeze(1)
            )
        else:
            for start in range(0, num_rows, _SCORE_CHUNK):
                block = table[start : start + _SCORE_CHUNK]
                hits = block.unsqueeze(0).eq(target_tokens.unsqueeze(1)).all(dim=-1)
                allowed[:, start : start + block.size(0)] &= ~hits

        if self.cfg.exclude_history:
            history = batch.get("history_item_ids")
            if torch.is_tensor(history):
                if self.item_id_to_semantic_row is None:
                    raise ValueError(
                        "exclude_history=True requires item_id_to_semantic_row so "
                        "history item IDs can be mapped onto semantic-ID rows."
                    )
                history = history.to(device).long()
                history_mask = batch.get("history_mask")
                if torch.is_tensor(history_mask):
                    history_mask = history_mask.to(device).bool()
                else:
                    history_mask = history.ne(int(self.cfg.pad_token_id))
                history = torch.where(history_mask, history, torch.full_like(history, -1))
                history_rows = self._rows_for_items(history, num_rows)
                allowed = self._scatter_exclude(allowed, history_rows)
        return allowed

    @staticmethod
    def _scatter_exclude(allowed: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Set allowed[b, rows[b, j]] = False, ignoring negative row ids."""
        num_rows = allowed.size(1)
        padded = torch.cat(
            [allowed, torch.zeros_like(allowed[:, :1])], dim=1
        )
        safe = torch.where(rows.ge(0), rows, torch.full_like(rows, num_rows))
        padded.scatter_(1, safe, False)
        return padded[:, :num_rows]

    # -- mode-specific selection ---------------------------------------------

    def _sample_rows(
        self,
        batch: Batch,
        *,
        allowed: torch.Tensor,
        row_to_item: torch.Tensor,
        neg_k: int,
        step: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.cfg.mode == "random":
            return self._draw(allowed.float(), neg_k, generator)

        scores = self._score_rows(batch, allowed=allowed, row_to_item=row_to_item)

        if self.cfg.mode == "top_n":
            masked = scores.masked_fill(~allowed, -torch.inf)
            k = min(neg_k, int(allowed.size(1)))
            top = masked.topk(k, dim=1)
            rows = torch.full(
                (allowed.size(0), neg_k), -1, device=allowed.device, dtype=torch.long
            )
            rows[:, :k] = torch.where(
                torch.isfinite(top.values), top.indices, torch.full_like(top.indices, -1)
            )
            return rows

        # curriculum: ramp the sign of the score logits from easy (-1) to hard (+1).
        alpha = self._curriculum_alpha(step)
        temperature = max(float(self.cfg.temperature), 1e-12)
        logits = (alpha / temperature) * scores
        logits = logits.masked_fill(~allowed, -torch.inf)
        row_max = logits.amax(dim=1, keepdim=True)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        logits = logits - row_max
        weights = torch.exp(logits)
        weights = torch.where(
            torch.isfinite(weights) & allowed, weights, torch.zeros_like(weights)
        )
        # Rows whose weights degenerate (all-zero) fall back to uniform-allowed.
        degenerate = weights.sum(dim=1, keepdim=True) <= 0.0
        weights = torch.where(degenerate, allowed.float(), weights)
        return self._draw(weights, neg_k, generator)

    def _curriculum_alpha(self, step: int | None) -> float:
        steps = int(self.cfg.curriculum_steps)
        if steps <= 0 or step is None:
            return 1.0
        return -1.0 + 2.0 * min(max(float(step) / float(steps), 0.0), 1.0)

    def _draw(
        self,
        weights: torch.Tensor,
        neg_k: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Multinomial without replacement; unfillable slots stay -1."""
        k = min(neg_k, int(weights.size(1)))
        rows = torch.full(
            (weights.size(0), neg_k), -1, device=weights.device, dtype=torch.long
        )
        if k == 0:
            return rows
        positive = weights.gt(0.0)
        mass = weights.sum(dim=1, keepdim=True).clamp_min(1e-38)
        # Epsilon-pad zero-weight entries so multinomial(replacement=False)
        # never fails on rows with fewer than k allowed items; epsilon picks
        # are rejected below, leaving those slots at -1.
        safe = weights / mass + torch.where(
            positive, torch.zeros_like(weights), torch.full_like(weights, 1e-12)
        )
        try:
            picks = torch.multinomial(safe, k, replacement=False, generator=generator)
        except RuntimeError:
            picks = torch.multinomial(safe, k, replacement=False)
        valid = positive.gather(1, picks)
        rows[:, :k] = torch.where(valid, picks, torch.full_like(picks, -1))
        return rows

    @torch.no_grad()
    def _score_rows(
        self,
        batch: Batch,
        *,
        allowed: torch.Tensor,
        row_to_item: torch.Tensor,
    ) -> torch.Tensor:
        """Item-model scores aligned to semantic rows; unscorable rows get -inf."""
        assert self.scorer is not None
        device = allowed.device
        pool_rows = row_to_item.ge(0).nonzero(as_tuple=True)[0]
        pool_items = row_to_item[pool_rows]
        scores = torch.full(allowed.shape, -torch.inf, device=device)
        if pool_items.numel() == 0:
            return scores

        scorer_device = getattr(self.scorer, "device", None)
        if scorer_device is not None and scorer_device != device:
            self.scorer.to(device)
        was_training = self.scorer.training
        self.scorer.eval()
        try:
            chunks: list[torch.Tensor] = []
            for start in range(0, pool_items.numel(), _SCORE_CHUNK):
                chunk = pool_items[start : start + _SCORE_CHUNK]
                chunks.append(self.scorer.score_items(batch, candidate_item_ids=chunk))
            pool_scores = torch.cat(chunks, dim=1).to(device).float()
        finally:
            self.scorer.train(was_training)

        if pool_scores.shape != (allowed.size(0), pool_items.numel()):
            raise ValueError(
                "Item model scores must have shape [B, num_pool_items]; got "
                f"{tuple(pool_scores.shape)} for pool of {pool_items.numel()}."
            )
        pool_scores = pool_scores.masked_fill(~torch.isfinite(pool_scores), -torch.inf)
        scores[:, pool_rows] = pool_scores
        return scores

    # -- plumbing --------------------------------------------------------------

    def _resolve_table(
        self,
        *,
        model: FlowRecModel | None,
        item_semantic_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if item_semantic_ids is not None:
            return item_semantic_ids
        if self.item_semantic_ids is not None:
            return self.item_semantic_ids
        if model is not None and torch.is_tensor(getattr(model, "item_semantic_ids", None)):
            return model.item_semantic_ids
        raise ValueError(
            "SemanticIDCandidateSampler needs item_semantic_ids (constructor, "
            "sample() argument, or model.item_semantic_ids)."
        )

    def _row_to_item(self, num_rows: int, device: torch.device) -> torch.Tensor:
        if self.item_id_to_semantic_row is None:
            return torch.full((num_rows,), -1, device=device, dtype=torch.long)
        mapping = self.item_id_to_semantic_row.to(device)
        row_to_item = torch.full((num_rows,), -1, device=device, dtype=torch.long)
        valid = (mapping >= 0) & (mapping < num_rows)
        item_ids = valid.nonzero(as_tuple=True)[0]
        row_to_item[mapping[valid]] = item_ids
        return row_to_item

    def _rows_for_items(self, item_ids: torch.Tensor, num_rows: int) -> torch.Tensor:
        if self.item_id_to_semantic_row is None:
            return torch.full_like(item_ids, -1)
        mapping = self.item_id_to_semantic_row.to(item_ids.device)
        safe = item_ids.clamp_min(0).clamp_max(max(mapping.numel() - 1, 0))
        rows = mapping[safe] if mapping.numel() else torch.full_like(item_ids, -1)
        rows = torch.where(item_ids.ge(0) & item_ids.lt(mapping.numel()), rows, torch.full_like(rows, -1))
        return torch.where(rows.lt(num_rows), rows, torch.full_like(rows, -1))

    def _target_tokens(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if "target_token_ids" not in batch:
            raise KeyError("SemanticIDCandidateSampler requires batch['target_token_ids'].")
        target = batch["target_token_ids"]
        if not torch.is_tensor(target):
            raise TypeError("batch['target_token_ids'] must be a torch.Tensor.")
        target = target.long()
        mask = batch.get("target_token_mask")
        if torch.is_tensor(mask):
            mask = mask.to(target.device).bool()
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
        return target, mask

    @staticmethod
    def _target_item_ids(batch: Batch, bsz: int, device: torch.device) -> torch.Tensor:
        for key in ("target_item_ids", "target_item_id"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value.to(device).long().reshape(bsz)
        return torch.full((bsz,), -1, device=device, dtype=torch.long)
