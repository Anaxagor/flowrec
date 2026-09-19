"""Unified semantic-ID candidate sampler.

One sampler, four modes, all operating on the full item catalog (every item
with a valid semantic-ID row):

    - ``curriculum``: score the catalog with an item model and sample
      negatives from a softmax whose difficulty ramps from easy (lowest
      scores) to hard (highest scores) over ``curriculum_steps`` training
      steps. ``curriculum_steps=0`` starts at full difficulty immediately.
    - ``top_n``: deterministic hardest negatives — the ``num_negatives``
      highest-scoring allowed items.
    - ``random``: uniform over all allowed items. Needs no scorer.
    - ``policy``: negatives drawn from the recommendation model being
      fine-tuned (the ``model`` passed to ``sample``), with no item-model
      teacher. ALWAYS on-policy: the model's own autoregressive distribution
      (its current weights, temperature 1, no curriculum) is sampled WITHOUT
      replacement by stochastic beam search (Gumbel-top-k; Kool et al., 2019).
      Decoding is constrained to VALID ITEMS at every level: a token can only
      be drawn if it extends the prefix towards the semantic ID of a catalog
      item (probabilities renormalized over those tokens), so every sampled
      sequence is an existing item and the negatives are distinct.

Every mode honors the same exclusion contract: the positive target is never
sampled as a negative, and with ``exclude_history=True`` (default) no item
from the batch's ``history_item_ids`` is sampled either. Note that batch
histories are truncated to the collator's ``max_history_len`` most recent
items, so a user's *older* positive-history items can still be drawn as
negatives even with the flag on — the interaction rewards handle them via
the untruncated positive-history store (see flowrec/rewards/interaction.py).
The sampler never reads the interaction stores: negatives come from the
catalog, not from logged negative events, so users without any negative
interactions simply get candidates the data reward scores at its neutral
``unobserved_reward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

SamplerMode = Literal["curriculum", "top_n", "random", "policy"]

_SCORE_CHUNK = 50_000


@dataclass(frozen=True, slots=True)
class CandidateSamplerConfig:
    mode: SamplerMode = "random"
    num_negatives: int = 1
    include_positive: bool = True
    exclude_history: bool = True
    # curriculum only: steps to ramp from easy to hard negatives.
    curriculum_steps: int = 0
    # curriculum only: softmax temperature over item-model scores. policy mode
    # is always on-policy: it takes neither a curriculum nor a temperature.
    temperature: float = 1.0
    # policy only: beams searched beyond num_negatives. The search returns the
    # model's top (num_negatives + extra) items in perturbed order; the target
    # and batch-history items among them are dropped, so the draw is an exact
    # without-replacement sample of the ALLOWED items whenever num_negatives
    # remain. None = num_negatives (twice as many beams as negatives). Examples
    # left short are CONTINUED by a second, wider search from which the items
    # already drawn are removed - still an exact on-policy sample, because
    # sampling without replacement is sequential (extra["policy_retry_fraction"]).
    # Uniform filling (extra["policy_fallback_fraction"]) only happens when the
    # catalog has fewer than num_negatives allowed items.
    policy_extra_beams: int | None = None
    pad_token_id: int = 0
    semantic_ids_are_fixed_length: bool = True


def _gumbel(shape: Any, *, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    try:
        uniform = torch.rand(shape, device=device, generator=generator)
    except RuntimeError:  # generator on another device
        uniform = torch.rand(shape, device=device)
    return -torch.log(-torch.log(uniform.clamp(1e-20, 1.0 - 1e-7)))


class _PolicyCatalog:
    """Prefix -> valid-children tables of the catalog's semantic IDs, and the
    leaf -> table-row lookup (policy sampling). Built from the rows that map to
    a catalog item, so every sampled leaf is an item the rewards can score."""

    def __init__(self, table: torch.Tensor, valid_rows: torch.Tensor) -> None:
        tokens = table.detach().long().cpu()
        self.num_rows = int(tokens.size(0))
        self.sid_len = int(tokens.size(1))
        rows = valid_rows.detach().cpu().nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            raise ValueError("mode='policy' found no semantic-ID row mapped to a catalog item.")
        tokens = tokens.index_select(0, rows)
        if int(tokens.min()) < 0:
            raise ValueError("semantic ids must be non-negative.")
        self.key_bits = max((int(tokens.max()) + 1).bit_length(), 1)
        if self.key_bits * self.sid_len > 62:
            raise ValueError(f"Cannot pack {self.sid_len} levels x {self.key_bits} bits into int64.")
        self._levels: list[tuple[torch.Tensor, torch.Tensor]] = []
        for depth in range(self.sid_len):
            edges = torch.unique(self.pack(tokens, depth + 1), sorted=True)
            node_keys, counts = torch.unique_consecutive(edges >> self.key_bits, return_counts=True)
            group = torch.repeat_interleave(torch.arange(node_keys.numel()), counts)
            offset = torch.arange(edges.numel()) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
            # int32: the last level is [#prefixes, max collisions] and mostly padding
            # (yambda_500m: 355k x 233), so the width of the dtype is the table's size.
            child_table = torch.full((node_keys.numel(), int(counts.max())), -1, dtype=torch.int32)
            child_table[group, offset] = ((edges & ((1 << self.key_bits) - 1)) - 1).to(torch.int32)
            self._levels.append((node_keys.contiguous(), child_table.contiguous()))
        # Items sharing a semantic ID resolve to the first such row.
        leaf_keys = self.pack(tokens, self.sid_len)
        order = torch.argsort(leaf_keys, stable=True)
        self._leaf_keys, self._leaf_rows = leaf_keys[order].contiguous(), rows[order].contiguous()
        self._on_device: dict[torch.device, tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]] = {}

    def pack(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        keys = torch.zeros(tokens.size(0), dtype=torch.int64, device=tokens.device)
        for level in range(depth):
            keys = (keys << self.key_bits) | (tokens[:, level].long() + 1)
        return keys

    def _device(self, device: torch.device):
        cached = self._on_device.get(device)
        if cached is None:
            cached = ([(k.to(device), c.to(device)) for k, c in self._levels],
                      self._leaf_keys.to(device), self._leaf_rows.to(device))
            self._on_device[device] = cached
        return cached

    def levels_on(self, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._device(device)[0]

    def rows_of(self, leaves: torch.Tensor) -> torch.Tensor:
        """Table row of each full semantic ID [M, L]; -1 when it is no catalog item."""
        _, leaf_keys, leaf_rows = self._device(leaves.device)
        keys = self.pack(leaves, self.sid_len)
        pos = torch.searchsorted(leaf_keys, keys).clamp_max(leaf_keys.numel() - 1)
        return torch.where(leaf_keys.index_select(0, pos).eq(keys), leaf_rows.index_select(0, pos),
                           torch.full_like(keys, -1))


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
        if self.cfg.mode not in ("curriculum", "top_n", "random", "policy"):
            raise ValueError(
                f"Unknown sampler mode {self.cfg.mode!r}. "
                "Expected curriculum, top_n, random, or policy."
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
        if self.cfg.mode in ("curriculum", "top_n", "policy") and self.item_id_to_semantic_row is None:
            raise ValueError(
                f"mode={self.cfg.mode!r} requires item_id_to_semantic_row so scored "
                "items can be mapped onto semantic-ID rows."
            )
        if self.cfg.mode == "policy" and (int(self.cfg.curriculum_steps) != 0 or float(self.cfg.temperature) != 1.0):
            raise ValueError(
                "mode='policy' is always on-policy: it samples the model's own distribution, so "
                "curriculum_steps must be 0 and temperature 1 "
                f"(got curriculum_steps={self.cfg.curriculum_steps}, temperature={self.cfg.temperature})."
            )
        self._policy_catalog: _PolicyCatalog | None = None
        self._policy_fallback_fraction = 0.0
        self._policy_retry_fraction = 0.0

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
                model=model,
                table=table,
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
        if self.cfg.mode == "policy" and neg_k > 0:
            extra["policy_fallback_fraction"] = self._policy_fallback_fraction
            extra["policy_retry_fraction"] = self._policy_retry_fraction

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
        model: FlowRecModel | None = None,
        table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.mode == "random":
            return self._draw(allowed.float(), neg_k, generator)
        if self.cfg.mode == "policy":
            return self._policy_rows(
                batch, model=model, table=table, allowed=allowed, row_to_item=row_to_item,
                neg_k=neg_k, step=step, generator=generator,
            )

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

    # -- policy mode: sample the recommendation model itself ---------------------

    @torch.no_grad()
    def _policy_rows(
        self,
        batch: Batch,
        *,
        model: FlowRecModel | None,
        table: torch.Tensor | None,
        allowed: torch.Tensor,
        row_to_item: torch.Tensor,
        neg_k: int,
        step: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """``neg_k`` distinct allowed rows per example, sampled without
        replacement from the model's own distribution over catalog items
        (on-policy: ``step`` plays no role)."""
        scorer = getattr(model, "candidate_token_logits", None)
        if model is None or not callable(scorer) or table is None:
            raise ValueError(
                "mode='policy' samples the model being trained: sample() needs model=... "
                "exposing candidate_token_logits(batch, candidate_token_ids=[B, N, L])."
            )
        device = allowed.device
        if self._policy_catalog is None or self._policy_catalog.num_rows != int(table.size(0)):
            self._policy_catalog = _PolicyCatalog(table, row_to_item.ge(0))
        catalog = self._policy_catalog
        extra = self.cfg.policy_extra_beams
        width = neg_k + (neg_k if extra is None else max(int(extra), 0))
        bsz = int(allowed.size(0))

        def search(sub_batch: Batch, sub_allowed: torch.Tensor, beams: int) -> tuple[torch.Tensor, torch.Tensor]:
            """Rows of the ``beams`` sampled items in perturbed-score order (-1 =
            no item) and which of them may be used as negatives."""
            prepare = getattr(model, "cache_history_encoding", None)
            work = prepare(sub_batch) if callable(prepare) else sub_batch
            leaves, alive = self._stochastic_beam_search(
                scorer, work, catalog, bsz=int(sub_allowed.size(0)), width=beams,
                device=device, generator=generator,
            )
            found = catalog.rows_of(leaves.reshape(-1, leaves.size(-1))).reshape(leaves.shape[:2])
            found = torch.where(alive, found, torch.full_like(found, -1))
            return found, found.ge(0) & sub_allowed.gather(1, found.clamp_min(0))

        def take(target: torch.Tensor, found: torch.Tensor, ok: torch.Tensor, offset: torch.Tensor) -> None:
            """Write the first allowed rows, in order, into target[b, offset[b]:]."""
            slot = offset.unsqueeze(1) + ok.long().cumsum(dim=1) - 1
            keep = ok & slot.lt(neg_k)
            b_idx, w_idx = keep.nonzero(as_tuple=True)
            target[b_idx, slot[b_idx, w_idx]] = found[b_idx, w_idx]

        was_training = model.training
        model.eval()
        try:
            # The sampled items come back in perturbed-score order: the first
            # neg_k allowed ones are the without-replacement sample of the
            # allowed items.
            rows = torch.full((bsz, neg_k), -1, device=device, dtype=torch.long)
            found, ok = search(batch, allowed, width)
            take(rows, found, ok, torch.zeros(bsz, dtype=torch.long, device=device))
            # The target and the history items can crowd the top of the beam
            # (repeat consumption). Sampling without replacement is sequential:
            # given the items drawn so far, the rest is a fresh sample of the
            # REMAINING items. So a short example is continued exactly by a
            # second, independent search from which the first search's items
            # are removed - wide enough to hold them, every excluded item and
            # the missing negatives.
            short = rows.lt(0).any(dim=1).nonzero(as_tuple=True)[0]
            self._policy_retry_fraction = float(short.numel()) / max(bsz, 1)
            if short.numel():
                excluded = (row_to_item.ge(0).unsqueeze(0) & ~allowed[short]).sum(dim=1)
                wide = neg_k + width + int(excluded.max().item())
                sub_batch = {
                    key: (value.index_select(0, short.to(value.device))
                          if torch.is_tensor(value) and value.ndim > 0 and int(value.size(0)) == bsz else value)
                    for key, value in batch.items()
                }
                found2, ok2 = search(sub_batch, allowed[short], wide)
                seen = found[short]
                fresh = ~(found2.unsqueeze(-1).eq(seen.unsqueeze(1)) & seen.ge(0).unsqueeze(1)).any(dim=-1)
                continued = rows[short]
                take(continued, found2, ok2 & fresh, continued.ge(0).sum(dim=1))
                rows[short] = continued
        finally:
            model.train(was_training)

        missing = rows.lt(0)
        self._policy_fallback_fraction = float(missing.float().mean().item()) if rows.numel() else 0.0
        if bool(missing.any()):
            # Only when fewer than neg_k allowed items exist (tiny catalogs):
            # uniform over the allowed items not drawn yet.
            short = missing.any(dim=1).nonzero(as_tuple=True)[0]
            remaining = self._scatter_exclude(allowed[short], rows[short])
            fill = self._draw(remaining.float(), neg_k, generator)
            slot = missing[short].long().cumsum(dim=1) - 1
            patched = torch.where(missing[short], fill.gather(1, slot.clamp_min(0)), rows[short])
            rows[short] = patched
        # Invariant, checked on every batch: a negative is a catalog item the
        # exclusion contract allows - never an invalid semantic ID, a row
        # without an item, the target or a batch-history item.
        filled = rows.ge(0)
        safe = rows.clamp_min(0)
        if bool((filled & (row_to_item[safe].lt(0) | ~allowed.gather(1, safe))).any()):
            raise RuntimeError("policy sampler drew a negative that is not an allowed catalog item.")
        return rows

    @staticmethod
    def _stochastic_beam_search(
        scorer: Any,
        batch: Batch,
        catalog: "_PolicyCatalog",
        *,
        bsz: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-top-``width`` over the catalog trie (Kool et al., 2019).

        Returns the ``width`` leaves [B, width, L] in decreasing perturbed
        score and their validity mask; they are an exact sample without
        replacement from the model's own sequence distribution under the
        valid-item constraint: at every level the token probabilities are the
        model's, renormalized over the tokens that continue the prefix towards
        a catalog item (every other token has probability 0).
        """
        levels = catalog.levels_on(device)
        prefix = torch.zeros((bsz, 1, 0), dtype=torch.long, device=device)
        phi = torch.zeros((bsz, 1), dtype=torch.float32, device=device)      # log-probability
        perturbed = torch.zeros((bsz, 1), dtype=torch.float32, device=device)  # G of the prefix
        alive = torch.ones((bsz, 1), dtype=torch.bool, device=device)

        for depth in range(catalog.sid_len):
            beams = int(prefix.size(1))
            probe = torch.cat([prefix, prefix.new_zeros((bsz, beams, 1))], dim=-1)
            logits = scorer(batch, candidate_token_ids=probe)[:, :, -1, :].float()

            node_keys, child_table = levels[depth]
            keys = catalog.pack(prefix.reshape(bsz * beams, depth), depth)
            node = torch.searchsorted(node_keys, keys).clamp_max(node_keys.numel() - 1)
            known = node_keys.index_select(0, node).eq(keys).reshape(bsz, beams)
            children = child_table.index_select(0, node).reshape(bsz, beams, -1).long()
            child_mask = children.ge(0) & (alive & known).unsqueeze(-1)

            child_logits = logits.gather(-1, children.clamp_min(0))
            child_logits = child_logits.masked_fill(~child_mask, -torch.inf)
            log_norm = torch.logsumexp(child_logits, dim=-1, keepdim=True)
            log_norm = torch.where(torch.isfinite(log_norm), log_norm, torch.zeros_like(log_norm))
            child_phi = phi.unsqueeze(-1) + (child_logits - log_norm)

            # Children's Gumbels conditioned on their maximum being the parent's.
            noise = _gumbel(child_phi.shape, device=device, generator=generator)
            g_phi = (child_phi + noise).masked_fill(~child_mask, -torch.inf)
            z = g_phi.amax(dim=-1, keepdim=True)
            z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            target = perturbed.unsqueeze(-1)
            v = target - g_phi + torch.log1p(-torch.exp(g_phi - z).clamp_max(1.0))
            child_g = target - torch.relu(v) - torch.log1p(torch.exp(-v.abs()))
            child_g = child_g.masked_fill(~child_mask, -torch.inf)

            fanout = int(children.size(-1))
            flat_g = child_g.reshape(bsz, beams * fanout)
            top = flat_g.topk(min(int(width), beams * fanout), dim=1)
            parent = top.indices // fanout
            token = children.reshape(bsz, beams * fanout).gather(1, top.indices)
            prefix = torch.cat(
                [prefix.gather(1, parent.unsqueeze(-1).expand(-1, -1, depth)), token.clamp_min(0).unsqueeze(-1)],
                dim=-1,
            )
            phi = child_phi.reshape(bsz, beams * fanout).gather(1, top.indices)
            perturbed = top.values
            alive = torch.isfinite(perturbed)
            phi = torch.where(alive, phi, torch.zeros_like(phi))
            perturbed = torch.where(alive, perturbed, torch.zeros_like(perturbed))
        return prefix, alive

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
