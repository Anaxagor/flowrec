"""Factorized full-catalog reward caches for GFlow fine-tuning.

The dense reward cache (``scripts/precompute_reward_cache.py`` read by
``GFlowMethod._PrecomputedRewardCache``) stores one float16 row per training
example over the whole catalog: ``num_examples x num_items`` values. Every
reward component in this repo is a cheap function of a small per-example
factor and a per-item table, so those rows can be rebuilt exactly on the
training device at lookup time instead:

    sasrec       R[e, i] = transform(h_e . t_i)
                 h_e: SASRec teacher final hidden state of example e
                 t_i: teacher item embedding of catalog row i
    similarity   hierarchical: prefix-gated codebook distance between the
                 target semantic ID and every catalog semantic ID
                 embedding_cosine: (z_target . z_i + 1) / 2
                 exact: mean token agreement with the target
    interaction  a constant fill value everywhere except the user's logged
                 before-t evidence (stored as a sparse CSR of the exceptions)
    positive     negative_reward everywhere, positive_reward at the target row

``scripts/precompute_reward_factors.py`` writes the factors (``factors.pt``
plus ``manifest.json``). :class:`FactorizedRewardCache` reproduces the dense
cache's ``select_candidates`` contract for one component (including the
target-row substitution for negative candidate rows and the exact
full-catalog reward sum used as logZ); :class:`ComposedFactorizedRewardCache`
reproduces the GPT-2 grid's composition (weights / power / multiply | mixed)
at lookup time and accepts any mix of factorized and dense component caches.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON
from flowrec.rewards.sasrec import SASRecRewardConfig, transform_sasrec_scores

FACTORS_FILENAME = "factors.pt"
CACHE_KIND = "factorized"
COMPOSITIONS = ("multiply", "mixed")


# --------------------------------------------------------------------------- #
# Row reconstruction kernels
# --------------------------------------------------------------------------- #
def hierarchical_similarity_rows(
    target_tokens: torch.Tensor,
    item_tokens: torch.Tensor,
    code_dist: torch.Tensor,
    *,
    token_offset: int,
    level_stride: int,
    level_weights: Sequence[float] | None = None,
    chunk_rows: int = 64,
) -> torch.Tensor:
    """Full-catalog form of ``SimilarityRewardModel._hierarchical_similarity``.

    ``target_tokens`` [B, L] and ``item_tokens`` [N, L] are semantic-ID tokens
    (offset / level-offset as in the batch), ``code_dist`` the model's
    ``[n_rq_levels, codebook_size, codebook_size]`` normalized codebook
    distances. Returns [B, N] float32 similarities in [0, 1] before sharpness
    and the epsilon floor. Levels are broadcast [B, 1] x [1, N] and the batch
    is processed in ``chunk_rows`` slices so the [B, N, L] candidate tensor of
    the model path is never materialized.
    """
    if target_tokens.ndim != 2 or item_tokens.ndim != 2:
        raise ValueError("target_tokens must be [B, L] and item_tokens [N, L].")
    bsz, seq_len = target_tokens.shape
    num_items = int(item_tokens.size(0))
    if int(item_tokens.size(1)) != seq_len:
        raise ValueError("target_tokens and item_tokens disagree on the SID length.")
    device = item_tokens.device
    n_rq_levels, codebook_size = int(code_dist.shape[0]), int(code_dist.shape[1])

    if level_weights is None:
        weights = torch.ones(seq_len, dtype=torch.float32, device=device)
    else:
        if len(level_weights) != seq_len:
            raise ValueError(
                f"level_weights has {len(level_weights)} entries but the SID "
                f"length is {seq_len}."
            )
        weights = torch.as_tensor(list(level_weights), dtype=torch.float32, device=device)
    weight_sum = weights.sum().clamp_min(1e-12)

    item_codes = item_tokens.long() - int(token_offset)
    target_codes = target_tokens.to(device).long() - int(token_offset)
    if int(level_stride) > 0:
        level_shift = torch.arange(seq_len, device=device, dtype=torch.long) * int(level_stride)
        item_codes = item_codes - level_shift.view(1, -1)
        target_codes = target_codes - level_shift.view(1, -1)

    out = torch.empty((bsz, num_items), dtype=torch.float32, device=device)
    chunk = max(int(chunk_rows), 1)
    for start in range(0, bsz, chunk):
        stop = min(start + chunk, bsz)
        tc = target_codes[start:stop]
        rows = int(tc.size(0))
        sim = torch.zeros((rows, num_items), dtype=torch.float32, device=device)
        active = torch.ones((rows, num_items), dtype=torch.float32, device=device)
        for level in range(seq_len):
            ca = item_codes[:, level].view(1, -1)
            ct = tc[:, level].view(-1, 1)
            valid = (ca >= 0) & (ct >= 0)
            if level < n_rq_levels:
                valid = valid & (ca < codebook_size) & (ct < codebook_size)
            eq = (ca == ct) & valid
            if level < n_rq_levels:
                safe_ca = ca.clamp(0, codebook_size - 1)
                safe_ct = ct.clamp(0, codebook_size - 1)
                dist = code_dist[level][safe_ca, safe_ct]  # broadcast -> [rows, N]
                grade = torch.where(eq, torch.ones_like(dist), 1.0 - dist)
                grade = grade * valid.float()
            else:
                grade = eq.float()
            sim = sim + weights[level] * active * grade
            active = active * eq.float()
        out[start:stop] = sim / weight_sum
    return out


def exact_similarity_rows(
    target_tokens: torch.Tensor,
    item_tokens: torch.Tensor,
    *,
    chunk_rows: int = 64,
) -> torch.Tensor:
    """Full-catalog form of the ``exact`` similarity: mean positionwise token
    agreement between the target SID and every catalog SID, [B, N]."""
    bsz, seq_len = target_tokens.shape
    num_items = int(item_tokens.size(0))
    device = item_tokens.device
    target = target_tokens.to(device).long()
    items = item_tokens.long()
    out = torch.empty((bsz, num_items), dtype=torch.float32, device=device)
    chunk = max(int(chunk_rows), 1)
    for start in range(0, bsz, chunk):
        stop = min(start + chunk, bsz)
        eq = items.unsqueeze(0).eq(target[start:stop].unsqueeze(1))  # [rows, N, L]
        out[start:stop] = eq.sum(dim=-1).to(torch.float32) / float(seq_len)
    return out


def _pack_sparse_rows(
    indptr: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather CPU CSR rows in request order, including repeated/empty rows."""
    idx = indices.detach().cpu().long()
    starts = indptr[idx].long()
    lengths = (indptr[idx + 1] - starts).long()
    total = int(lengths.sum())
    row_index = torch.repeat_interleave(
        torch.arange(idx.numel(), dtype=torch.long), lengths, output_size=total
    )
    # Each requested row contributes one contiguous range. Expand its offset
    # into the packed array instead of constructing B individual arange tensors.
    packed_starts = lengths.cumsum(0) - lengths
    flat = torch.arange(total, dtype=torch.long) + torch.repeat_interleave(
        starts - packed_starts, lengths, output_size=total
    )
    return {"row_index": row_index, "cols": cols[flat].long(), "vals": vals[flat].float()}


def _scatter_packed_rows(
    packed: Mapping[str, torch.Tensor],
    *,
    row_count: int,
    num_items: int,
    fill_value: float,
    device: torch.device,
) -> torch.Tensor:
    full = torch.full((row_count, num_items), fill_value, dtype=torch.float32, device=device)
    if packed["row_index"].numel():
        row_index = packed["row_index"].to(device=device, non_blocking=True)
        cols = packed["cols"].to(device=device, non_blocking=True)
        full[row_index, cols] = packed["vals"].to(device=device, non_blocking=True)
    return full


def scatter_sparse_rows(
    indptr: torch.Tensor,
    cols: torch.Tensor,
    vals: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_items: int,
    fill_value: float,
    device: torch.device,
) -> torch.Tensor:
    """Densify CSR rows ``indices`` of a (num_examples x num_items) matrix whose
    unstored entries equal ``fill_value``."""
    return _scatter_packed_rows(
        _pack_sparse_rows(indptr, cols, vals, indices), row_count=indices.numel(),
        num_items=int(num_items), fill_value=float(fill_value), device=device,
    )


def _check_prepared_rows(
    prepared: Mapping[str, Any], indices: torch.Tensor, cache_key: str,
) -> torch.Tensor:
    """Check the internal batch payload without copying CUDA indices to CPU.

    ``prepare_rows`` checks public indices on CPU before worker enqueueing.
    A mismatch after transfer to CUDA is an internal batch-routing error, so
    its invariant uses an asynchronous assertion (which can invalidate the
    CUDA context on failure), rather than synchronizing every training step.
    """
    if not isinstance(prepared, Mapping) or prepared.get("cache_key") != cache_key:
        raise ValueError("Prepared reward rows belong to a different cache.")
    idx = indices.detach().long()
    if idx.ndim != 1 or idx.numel() == 0:
        raise ValueError("Prepared reward indices must be a nonempty one-dimensional tensor.")
    packed_idx = prepared.get("example_indices")
    if (prepared.get("row_count") != idx.numel()
            or not isinstance(packed_idx, torch.Tensor)
            or packed_idx.shape != idx.shape or packed_idx.dtype != torch.long):
        raise ValueError("Prepared reward row count/indices do not match the batch.")
    compare_device = idx.device if idx.device.type == "cuda" else packed_idx.device
    actual = idx.to(device=compare_device, non_blocking=True)
    expected = packed_idx.to(device=compare_device, non_blocking=True)
    if compare_device.type == "cuda":
        torch._assert_async((actual == expected).all(), "Prepared reward indices do not match the batch.")
    elif not torch.equal(actual, expected):
        raise ValueError("Prepared reward indices do not match the batch.")
    return idx


# --------------------------------------------------------------------------- #
# Single-component factorized cache
# --------------------------------------------------------------------------- #
class FactorizedRewardCache:
    """Drop-in for ``GFlowMethod._PrecomputedRewardCache`` that rebuilds the
    full-catalog reward rows of ONE component from stored factors.

    ``select_candidates(indices, candidate_rows, device, include_full)``
    returns ``(selected [B, N], reward_sum [B], full [B, num_items] | None)``
    with the same semantics as the dense class: rows are clamped at 1e-6,
    raised to ``power`` and clamped again; negative candidate rows resolve to
    the example's target row; ``reward_sum`` is the exact full-catalog sum of
    the transformed rewards (logZ = log reward_sum).
    """

    # The sasrec factors (final hidden state + item table) are independent of
    # the score -> reward transform, so these keys may be replaced at open time
    # without rebuilding the cache (e.g. sigmoid -> exp at several temperatures).
    SASREC_OVERRIDABLE = ("transform", "temperature", "score_shift")

    def __init__(
        self,
        path: str | Path,
        *,
        power: float = 1.0,
        chunk_rows: int = 64,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Factorized reward cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("kind") != CACHE_KIND:
            raise ValueError(
                f"{manifest_path} is not a factorized reward cache "
                f"(kind={self.manifest.get('kind')!r}); dense caches are read by "
                "GFlowMethod._PrecomputedRewardCache."
            )
        self.power = float(power)
        if self.power <= 0.0:
            raise ValueError("power must be positive.")
        self.chunk_rows = max(int(chunk_rows), 1)

        component = dict(self.manifest["component"])
        self.component_name = str(component["name"])
        self.mode = component.get("mode")
        self.config: dict[str, Any] = dict(component.get("config", {}))
        self.config_overrides: dict[str, Any] = {}
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        self.reward_epsilon = float(self.manifest.get("reward_epsilon", GFLOW_REWARD_EPSILON))
        self._num_examples = int(self.manifest["num_examples"])
        self._num_items = int(self.manifest["num_catalog_items"])

        factors_path = self.path / str(self.manifest.get("factors_path", FACTORS_FILENAME))
        self.factors: dict[str, torch.Tensor] = torch.load(
            factors_path, map_location="cpu", mmap=True, weights_only=True
        )
        self._validate_factors()
        self._shared_cache: dict[torch.device, dict[str, torch.Tensor]] = {}
        # Identity describes the stored factors, not their lookup-time transform.
        self._cache_key = json.dumps([
            str(self.path.resolve()), self.num_examples, self.num_items,
            self.component_name, self.mode,
        ])

    # -- lookup-time reward transform ---------------------------------------

    def _apply_config_overrides(self, overrides: Mapping[str, Any]) -> None:
        if self.component_name != "sasrec":
            raise ValueError(
                f"config_overrides are only supported for the sasrec component (its factors are "
                f"transform-agnostic); {self.path} holds {self.component_name!r}."
            )
        unknown = sorted(set(overrides) - set(self.SASREC_OVERRIDABLE))
        if unknown:
            raise ValueError(
                f"Unsupported sasrec config_overrides {unknown}; allowed: {list(self.SASREC_OVERRIDABLE)}."
            )
        if "temperature" in overrides and float(overrides["temperature"]) <= 0.0:
            raise ValueError("sasrec temperature must be positive.")
        self.config_overrides = {**self.config_overrides, **dict(overrides)}
        self.config = {**self.config, **dict(overrides)}

    def with_config(self, **overrides: Any) -> "FactorizedRewardCache":
        """A view of this cache with another sasrec transform / temperature.

        Shares the memory-mapped factors and the per-device item table with the
        original, so several temperatures cost no extra host or GPU memory.
        """
        view = copy.copy(self)
        view.config = dict(self.config)
        view.config_overrides = dict(self.config_overrides)
        view._apply_config_overrides(overrides)
        return view

    # -- metadata ------------------------------------------------------------

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def num_items(self) -> int:
        return self._num_items

    def describe(self) -> dict[str, Any]:
        return {
            "kind": CACHE_KIND,
            "component": self.component_name,
            "mode": self.mode,
            "config_overrides": dict(self.config_overrides),
            "power": self.power,
            "num_examples": self.num_examples,
            "num_items": self.num_items,
            "path": str(self.path),
        }

    def _require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.factors]
        if missing:
            raise KeyError(f"Factorized cache {self.path} lacks factors {missing}.")

    def _validate_factors(self) -> None:
        self._require("target_semantic_row_ids")
        if int(self.factors["target_semantic_row_ids"].numel()) != self._num_examples:
            raise ValueError(f"target_semantic_row_ids does not cover num_examples: {self.path}")
        name, mode = self.component_name, self.mode
        if name == "sasrec":
            self._require("hidden", "item_table", "row_valid")
            if int(self.factors["hidden"].size(0)) != self._num_examples:
                raise ValueError(f"hidden rows != num_examples: {self.path}")
            if tuple(self.factors["item_table"].shape) != (self._num_items, int(self.factors["hidden"].size(1))):
                raise ValueError(f"item_table must be [num_items, hidden_dim]: {self.path}")
        elif name == "similarity":
            if mode == "hierarchical":
                self._require("target_token_ids", "item_tokens", "code_dist")
            elif mode == "embedding_cosine":
                self._require("target_item_ids", "item_embeddings", "row_item_ids")
            elif mode == "exact":
                self._require("target_token_ids", "item_tokens")
            else:
                raise ValueError(f"Unknown similarity mode {mode!r} in {self.path}")
            if "item_tokens" in self.factors and int(self.factors["item_tokens"].size(0)) != self._num_items:
                raise ValueError(f"item_tokens rows != num_catalog_items: {self.path}")
        elif name == "interaction":
            self._require("indptr", "cols", "vals")
            if int(self.factors["indptr"].numel()) != self._num_examples + 1:
                raise ValueError(f"indptr must have num_examples + 1 entries: {self.path}")
        elif name == "positive":
            pass
        else:
            raise ValueError(f"Unknown factorized component {name!r} in {self.path}")

    # -- lookups -------------------------------------------------------------

    def _checked_indices(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.detach().cpu().long()
        if idx.ndim != 1:
            raise ValueError("example_indices must be a one-dimensional tensor.")
        if idx.numel() == 0:
            raise ValueError("Empty example_indices tensor for reward cache lookup.")
        if int(idx.min()) < 0 or int(idx.max()) >= self.num_examples:
            raise IndexError(
                f"Reward cache index out of range: min={int(idx.min())} max={int(idx.max())} "
                f"num_examples={self.num_examples}"
            )
        return idx

    def target_rows(self, indices: torch.Tensor) -> torch.Tensor:
        idx = self._checked_indices(indices)
        return self.factors["target_semantic_row_ids"].index_select(0, idx).long()

    def prepare_rows(self, indices: torch.Tensor) -> dict[str, Any]:
        """Gather only this batch's factors into independent CPU tensors.

        Call from a DataLoader worker, then pin/transfer the nested payload
        with the rest of the batch. Reconstruction subsequently reads only
        this payload and the shared catalog tables, never random mmap rows.
        """
        idx = self._checked_indices(indices).clone()
        factors: dict[str, torch.Tensor] = {}
        if self.component_name == "sasrec":
            factors["hidden"] = self.factors["hidden"].index_select(0, idx)
        elif self.component_name == "similarity":
            key = "target_item_ids" if self.mode == "embedding_cosine" else "target_token_ids"
            factors[key] = self.factors[key].index_select(0, idx)
        elif self.component_name == "interaction":
            factors = _pack_sparse_rows(
                self.factors["indptr"], self.factors["cols"], self.factors["vals"], idx
            )
        return {
            "cache_key": self._cache_key, "row_count": idx.numel(), "example_indices": idx,
            "target_semantic_row_ids": self.factors["target_semantic_row_ids"].index_select(0, idx).long(),
            "factors": factors,
        }

    def make_row_reader(self) -> "RewardFactorRowReader":
        """A small, picklable worker reader with no catalog/GPU tensors."""
        return RewardFactorRowReader(self._reader_spec())

    def _reader_spec(self) -> dict[str, Any]:
        return {
            "kind": "component", "path": str(self.path), "power": self.power,
            "chunk_rows": self.chunk_rows, "config_overrides": dict(self.config_overrides),
        }

    def _checked_prepared(self, indices: torch.Tensor, prepared: Mapping[str, Any]) -> torch.Tensor:
        idx = _check_prepared_rows(prepared, indices, self._cache_key)
        targets = prepared.get("target_semantic_row_ids")
        if (not isinstance(targets, torch.Tensor) or targets.shape != idx.shape
                or targets.dtype != torch.long):
            raise ValueError("Prepared target rows must be a long tensor matching indices.")
        factors = prepared.get("factors")
        if not isinstance(factors, Mapping):
            raise ValueError("Prepared reward rows lack their factors.")
        name, mode = self.component_name, self.mode
        if name == "interaction":
            tensors = [factors.get(k) for k in ("row_index", "cols", "vals")]
            if (any(not isinstance(t, torch.Tensor) or t.ndim != 1 for t in tensors)
                    or len({t.numel() for t in tensors}) != 1
                    or tensors[0].dtype != torch.long or tensors[1].dtype != torch.long
                    or tensors[2].dtype != torch.float32):
                raise ValueError("Prepared CSR row_index, cols, and vals must be aligned vectors.")
        elif name in {"sasrec", "similarity"}:
            key = ("hidden" if name == "sasrec" else
                   "target_item_ids" if mode == "embedding_cosine" else "target_token_ids")
            value = factors.get(key)
            expected_shape = (idx.numel(), *self.factors[key].shape[1:])
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(f"Prepared {key} has the wrong batch/factor shape.")
        return idx

    def _shared(self, device: torch.device) -> dict[str, torch.Tensor]:
        cached = self._shared_cache.get(device)
        if cached is not None:
            return cached
        shared: dict[str, torch.Tensor] = {}
        name, mode = self.component_name, self.mode
        if name == "sasrec":
            shared["item_table"] = self.factors["item_table"].to(device=device, dtype=torch.float32)
            shared["row_valid"] = self.factors["row_valid"].to(device=device, dtype=torch.bool)
        elif name == "similarity":
            if mode in {"hierarchical", "exact"}:
                shared["item_tokens"] = self.factors["item_tokens"].to(device=device, dtype=torch.long)
            if mode == "hierarchical":
                shared["code_dist"] = self.factors["code_dist"].to(device=device, dtype=torch.float32)
            if mode == "embedding_cosine":
                table = self.factors["item_embeddings"].to(device=device, dtype=torch.float32)
                vocab = int(table.size(0))
                row_ids = self.factors["row_item_ids"].to(device=device, dtype=torch.long)
                shared["item_embeddings"] = table
                shared["row_valid"] = row_ids.ge(0) & row_ids.lt(vocab)
                shared["row_vecs"] = table[row_ids.clamp(0, vocab - 1)]
        self._shared_cache[device] = shared
        return shared

    def component_rows(
        self, indices: torch.Tensor, device: torch.device, *, prepared: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Raw component rewards [B, num_items] (float32, floored at the GFlow
        epsilon) — what the dense cache stores before its dtype rounding."""
        idx = self._checked_indices(indices) if prepared is None else self._checked_prepared(indices, prepared)
        return self._component_rows(idx, torch.device(device), prepared)

    def _component_rows(
        self, idx: torch.Tensor, device: torch.device, prepared: Mapping[str, Any] | None,
    ) -> torch.Tensor:
        device = torch.device(device)
        eps = self.reward_epsilon
        name, mode = self.component_name, self.mode
        shared = self._shared(device)

        def factor_rows(key: str) -> torch.Tensor:
            return (self.factors[key].index_select(0, idx) if prepared is None
                    else prepared["factors"][key])

        if name == "sasrec":
            hidden = factor_rows("hidden").to(device=device, dtype=torch.float32, non_blocking=True)
            scores = hidden @ shared["item_table"].transpose(0, 1)
            scores = scores.masked_fill(~shared["row_valid"].unsqueeze(0), -float("inf"))
            cfg = SASRecRewardConfig(**self.config)
            rewards = transform_sasrec_scores(scores, cfg).clamp_min(float(cfg.reward_epsilon))
            return rewards.clamp_min(eps)

        if name == "similarity":
            sharpness = float(self.config.get("sharpness", 1.0))
            model_eps = float(self.config.get("reward_epsilon", eps))
            if mode == "hierarchical":
                target = factor_rows("target_token_ids")
                weights = self.config.get("level_weights")
                sim = hierarchical_similarity_rows(
                    target, shared["item_tokens"], shared["code_dist"],
                    token_offset=int(self.config.get("token_offset", 1)),
                    level_stride=int(self.config.get("level_stride", 0)),
                    level_weights=tuple(weights) if weights is not None else None,
                    chunk_rows=self.chunk_rows,
                )
            elif mode == "exact":
                target = factor_rows("target_token_ids")
                sim = exact_similarity_rows(target, shared["item_tokens"], chunk_rows=self.chunk_rows)
            else:  # embedding_cosine
                table = shared["item_embeddings"]
                vocab = int(table.size(0))
                target_ids = factor_rows("target_item_ids").to(device=device, dtype=torch.long, non_blocking=True)
                target_vecs = table[target_ids.clamp(0, vocab - 1)]
                sim = (target_vecs @ shared["row_vecs"].transpose(0, 1) + 1.0) / 2.0
                sim = sim.masked_fill(~shared["row_valid"].unsqueeze(0), 0.0)
            if sharpness != 1.0:
                sim = sim.clamp_min(0.0).pow(sharpness)
            return sim.clamp_min(model_eps).clamp_min(eps)

        if name == "interaction":
            if prepared is None:
                full = scatter_sparse_rows(
                    self.factors["indptr"], self.factors["cols"], self.factors["vals"], idx,
                    num_items=self.num_items, fill_value=float(self.manifest["fill_value"]), device=device,
                )
            else:
                full = _scatter_packed_rows(
                    prepared["factors"], row_count=idx.numel(), num_items=self.num_items,
                    fill_value=float(self.manifest["fill_value"]), device=device,
                )
            return full.clamp_min(eps)

        if name == "positive":
            positive = float(self.config.get("positive_reward", 1.0))
            negative = float(self.config.get("negative_reward", 1e-3))
            model_eps = float(self.config.get("reward_epsilon", eps))
            full = torch.full((idx.numel(), self.num_items), negative, dtype=torch.float32, device=device)
            rows = (self.factors["target_semantic_row_ids"].index_select(0, idx).long()
                    if prepared is None else prepared["target_semantic_row_ids"])
            valid = rows.ge(0) & rows.lt(self.num_items)
            b = torch.arange(idx.numel(), dtype=torch.long, device=rows.device)[valid].to(device)
            full[b, rows[valid].to(device)] = positive
            return full.clamp_min(model_eps).clamp_min(eps)

        raise ValueError(f"Unknown factorized component {name!r}")

    def _transform_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.float().clamp_min(1e-6)
        if self.power == 1.0:
            return rewards
        return rewards.pow(self.power).clamp_min(1e-6)

    def select_candidates(
        self,
        indices: torch.Tensor,
        candidate_rows: torch.Tensor,
        device: torch.device,
        *,
        include_full: bool = False,
        prepared: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        idx = self._checked_indices(indices) if prepared is None else self._checked_prepared(indices, prepared)
        device = torch.device(device)
        rows = candidate_rows.detach().long()
        if rows.ndim != 2 or rows.size(0) != idx.numel():
            raise ValueError("candidate_rows must have shape [B, N] matching indices.")
        rows = rows.to(device)
        target_rows = (self.factors["target_semantic_row_ids"].index_select(0, idx).long()
                       if prepared is None else prepared["target_semantic_row_ids"])
        target_rows = target_rows.to(device=device, non_blocking=True)
        rows = torch.where(rows.ge(0), rows, target_rows.unsqueeze(1))
        if rows.numel():
            if prepared is not None and rows.device.type == "cuda":
                torch._assert_async(
                    (rows.ge(0) & rows.lt(self.num_items)).all(),
                    "Candidate item row is outside the cached catalogue.",
                )
            elif int(rows.min()) < 0 or int(rows.max()) >= self.num_items:
                raise IndexError("Candidate item row is outside the cached catalogue.")

        full = self._transform_rewards(self._component_rows(idx, device, prepared))
        selected = full.gather(1, rows)
        reward_sum = full.sum(dim=1)
        return selected, reward_sum, (full if include_full else None)


# --------------------------------------------------------------------------- #
# Composition at lookup time (weights / power / multiply | mixed)
# --------------------------------------------------------------------------- #
class ComposedFactorizedRewardCache:
    """Compose per-component caches at lookup time, exactly like the GPT-2
    grid's ``ComposedRewardCache``:

        multiply:  R = prod_i clamp(R_i) ** w_i
        mixed:     R = sum_i w_i * R_i / max_row(R_i)

    then ``R ** power`` and ``reward_sum = sum_items R ** power``, in float32
    on the requested device. Components are ``(name, cache, weight)`` where
    ``cache`` is a directory path (opened as a :class:`FactorizedRewardCache`)
    or any object exposing the ``select_candidates`` contract with power 1
    (a dense ``_PrecomputedRewardCache`` works too, so mixed layouts are
    possible while migrating).
    """

    def __init__(
        self,
        components: Sequence[tuple[str, Any, float]],
        *,
        composition: str,
        power: float,
        reward_epsilon: float = GFLOW_REWARD_EPSILON,
    ) -> None:
        if not components:
            raise ValueError("ComposedFactorizedRewardCache requires at least one component.")
        if composition not in COMPOSITIONS:
            raise ValueError(f"composition must be one of {COMPOSITIONS}, got {composition!r}")
        if float(power) <= 0.0:
            raise ValueError("power must be positive.")
        self.names = [str(name) for name, _, _ in components]
        self.caches = [
            FactorizedRewardCache(cache, power=1.0) if isinstance(cache, (str, Path)) else cache
            for _, cache, _ in components
        ]
        for cache in self.caches:
            if float(getattr(cache, "power", 1.0)) != 1.0:
                raise ValueError("Component caches must be opened with power=1.0; the composition applies the power.")
        self.weights = [float(w) for _, _, w in components]
        if any(w <= 0.0 for w in self.weights):
            raise ValueError(f"component weights must be positive, got {self.weights}")
        self.composition = str(composition)
        self.power = float(power)
        self.reward_epsilon = float(reward_epsilon)
        sizes = {int(c.num_examples) for c in self.caches}
        if len(sizes) != 1:
            raise ValueError(
                "Component caches disagree on num_examples: "
                f"{dict(zip(self.names, [int(c.num_examples) for c in self.caches]))}"
            )
        widths = {int(getattr(c, "num_items", getattr(c, "_num_items", -1))) for c in self.caches}
        if len(widths) != 1:
            raise ValueError(f"Component caches disagree on the catalogue width: {widths}")
        self._num_examples = sizes.pop()
        self._num_items = widths.pop()
        self._cache_key = json.dumps([
            "composed", list(zip(self.names, [getattr(c, "_cache_key", None) for c in self.caches]))
        ])

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def num_items(self) -> int:
        return self._num_items

    def describe(self) -> dict[str, Any]:
        terms = [f"{n}^{w:g}" if self.composition == "multiply" else f"{w:g}*rownorm({n})"
                 for n, w in zip(self.names, self.weights)]
        joiner = " * " if self.composition == "multiply" else " + "
        return {
            "kind": CACHE_KIND,
            "composition": self.composition,
            "components": self.names,
            "weights": self.weights,
            "power": self.power,
            "form": "(" + joiner.join(terms) + f") ** {self.power:g}",
        }

    def _reader_spec(self) -> dict[str, Any]:
        if not all(isinstance(c, (FactorizedRewardCache, ComposedFactorizedRewardCache)) for c in self.caches):
            raise TypeError("CPU row preparation requires factorized component caches.")
        return {
            "kind": "composed", "components": [
                (name, cache._reader_spec(), weight)
                for name, cache, weight in zip(self.names, self.caches, self.weights)
            ],
            "composition": self.composition, "power": self.power,
            "reward_epsilon": self.reward_epsilon,
        }

    def make_row_reader(self) -> "RewardFactorRowReader":
        return RewardFactorRowReader(self._reader_spec())

    def prepare_rows(self, indices: torch.Tensor) -> dict[str, Any]:
        """Prepare each component in order; names need not be unique."""
        self._reader_spec()  # Fail clearly for legacy dense component layouts.
        components = [cache.prepare_rows(indices) for cache in self.caches]
        return {
            "cache_key": self._cache_key,
            "row_count": components[0]["row_count"],
            "example_indices": components[0]["example_indices"],
            "components": components,
        }

    def select_candidates(
        self,
        indices: torch.Tensor,
        candidate_rows: torch.Tensor,
        device: torch.device,
        *,
        include_full: bool = False,
        prepared: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if prepared is not None:
            _check_prepared_rows(prepared, indices, self._cache_key)
            if not isinstance(prepared.get("components"), list) or len(prepared["components"]) != len(self.caches):
                raise ValueError("Prepared reward components do not match the composed cache.")
        eps = self.reward_epsilon
        combined_sel: torch.Tensor | None = None
        combined_full: torch.Tensor | None = None
        for component_index, (cache, weight) in enumerate(zip(self.caches, self.weights)):
            kwargs = {} if prepared is None else {"prepared": prepared["components"][component_index]}
            sel, _, full = cache.select_candidates(
                indices, candidate_rows, device=device, include_full=True, **kwargs
            )
            assert full is not None
            sel = sel.float().clamp_min(eps)
            full = full.float().clamp_min(eps)
            if self.composition == "multiply":
                term_sel = sel if weight == 1.0 else sel.pow(weight)
                term_full = full if weight == 1.0 else full.pow(weight)
                combined_sel = term_sel if combined_sel is None else combined_sel * term_sel
                combined_full = term_full if combined_full is None else combined_full * term_full
            else:  # mixed: per-example max normalization, weighted sum
                row_max = full.amax(dim=1, keepdim=True).clamp_min(eps)
                term_sel = (sel / row_max).clamp_min(eps) * weight
                term_full = (full / row_max).clamp_min(eps) * weight
                combined_sel = term_sel if combined_sel is None else combined_sel + term_sel
                combined_full = term_full if combined_full is None else combined_full + term_full
        assert combined_sel is not None and combined_full is not None
        combined_sel = combined_sel.clamp_min(eps)
        combined_full = combined_full.clamp_min(eps)
        if self.power != 1.0:
            combined_sel = combined_sel.pow(self.power).clamp_min(eps)
            combined_full = combined_full.pow(self.power).clamp_min(eps)
        reward_sum = combined_full.sum(dim=1)
        return combined_sel, reward_sum, (combined_full if include_full else None)


class RewardFactorRowReader:
    """Path/config-only DataLoader worker handle for factorized mmap caches.

    A worker lazily opens its own CPU mappings. Pickling never includes these
    mappings (or a training cache's shared CUDA tables), including after use
    with num_workers=0 followed by a loader using the spawn start method.
    """

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = copy.deepcopy(dict(spec))
        self._cache: FactorizedRewardCache | ComposedFactorizedRewardCache | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"spec": self.spec, "_cache": None}

    @staticmethod
    def _open(spec: Mapping[str, Any]) -> FactorizedRewardCache | ComposedFactorizedRewardCache:
        if spec["kind"] == "component":
            return FactorizedRewardCache(
                spec["path"], power=spec["power"], chunk_rows=spec["chunk_rows"],
                config_overrides=spec["config_overrides"],
            )
        return ComposedFactorizedRewardCache(
            [(name, RewardFactorRowReader._open(component), weight)
             for name, component, weight in spec["components"]],
            composition=spec["composition"], power=spec["power"], reward_epsilon=spec["reward_epsilon"],
        )

    def prepare_rows(self, indices: torch.Tensor) -> dict[str, Any]:
        if self._cache is None:
            self._cache = self._open(self.spec)
        return self._cache.prepare_rows(indices)


def open_reward_cache(path: str | Path, *, power: float = 1.0):
    """Open a component cache directory as the right reader: a factorized
    cache when its manifest says so, else the dense ``_PrecomputedRewardCache``."""
    manifest_path = Path(path) / "manifest.json"
    if manifest_path.exists():
        kind = json.loads(manifest_path.read_text(encoding="utf-8")).get("kind")
        if kind == CACHE_KIND:
            return FactorizedRewardCache(path, power=power)
    from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache

    return _PrecomputedRewardCache(path, power=power)


def manifest_component(manifest: Mapping[str, Any]) -> tuple[str, str | None]:
    component = manifest.get("component", {})
    return str(component.get("name")), component.get("mode")
