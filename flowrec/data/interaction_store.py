"""Shared positive/auxiliary partition and the auxiliary interaction store.

Every datamodule follows the same contract:

1. The dataset-specific normalizer produces one canonical interaction frame
   with all native payload columns (Yambda: event_type / is_organic /
   played_ratio_pct / track_length_seconds; Amazon: rating /
   verified_purchase).
2. ``positive_criterion`` — a pandas query expression evaluated on that frame —
   selects the *positive* interactions used to train sequence models. The
   positive frame is stripped to user / item / timestamp; the item vocabulary
   and user catalog are defined by it alone.
3. Everything else becomes an :class:`InteractionLogStore`: the complement of
   the criterion with its payload columns intact, remapped to catalog IDs.
   Reward models and negative samplers query it by (user_id, item_id).

The store keeps event timestamps. To avoid leakage, training-time rewards for
``global_time`` splits should use ``filter_by_time(val_cutoff)``; store-derived
features must never be fed as model inputs at evaluation time (an auxiliary
event on a held-out target can encode the answer).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Columns a positive interaction is allowed to carry. ``order_idx`` is internal
# plumbing for stable sorting; ``raw`` is the opt-in debugging payload.
# Columns the positive stream keeps after partitioning. The split labels are
# only present for datasets with a provided split (VK-LSVD) and must survive
# because the split builder reads them after k-core filtering.
_POSITIVE_COLUMNS = (
    "raw_user_id",
    "raw_item_id",
    "timestamp",
    "order_idx",
    "raw",
    "flowrec_split",
    "flowrec_week",
)
_STORE_ID_COLUMNS = ("user_id", "item_id", "event_type", "timestamp")

PayloadAgg = str  # "last" | "first" | "max" | "min" | "mean"
_ALLOWED_AGGS = ("last", "first", "max", "min", "mean")


def partition_interactions(
    frame: pd.DataFrame,
    positive_criterion: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a canonical interaction frame into (positives, auxiliary).

    ``positive_criterion`` is a pandas query expression, e.g.
    ``"event_type == 'listen' and played_ratio_pct >= 50"`` or
    ``"rating >= 4.0"``. ``None`` keeps the historical behavior: every
    interaction is positive (with all its columns) and the auxiliary frame is
    empty. With a criterion, positives are stripped to user / item / timestamp
    while the auxiliary frame keeps every payload column.

    Rows where the criterion evaluates to NA (e.g. a payload column is missing
    for that event type) are treated as non-positive.
    """
    if positive_criterion is None:
        return frame, frame.iloc[0:0].copy()

    try:
        mask = frame.eval(positive_criterion)
    except Exception as exc:  # noqa: BLE001 - surface pandas parse/name errors
        raise ValueError(
            f"positive_criterion {positive_criterion!r} failed to evaluate: {exc}. "
            f"Available columns: {sorted(frame.columns)}"
        ) from exc
    if not (pd.api.types.is_bool_dtype(mask) or pd.api.types.is_object_dtype(mask)):
        raise ValueError(
            f"positive_criterion {positive_criterion!r} must evaluate to a boolean "
            f"mask, got dtype {getattr(mask, 'dtype', type(mask))}"
        )
    mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)

    if not frame.empty and not bool(mask.any()):
        raise ValueError(
            f"positive_criterion {positive_criterion!r} matched no interactions "
            f"out of {len(frame):,}. {_criterion_column_diagnostics(frame, positive_criterion)}"
        )

    positive_columns = [c for c in _POSITIVE_COLUMNS if c in frame.columns]
    positives = frame.loc[mask, positive_columns].reset_index(drop=True)
    auxiliary = frame.loc[~mask].reset_index(drop=True)
    return positives, auxiliary


def _criterion_column_diagnostics(frame: pd.DataFrame, criterion: str) -> str:
    """Describe the columns a criterion references — an all-null column is the
    usual culprit when a criterion matches nothing (e.g. a dataset export that
    lacks the field entirely)."""
    referenced = [c for c in frame.columns if c in criterion]
    if not referenced:
        return f"Available columns: {sorted(frame.columns)}"
    parts = []
    for column in referenced:
        series = frame[column]
        null_share = float(series.isna().mean())
        sample = series.dropna().unique()[:3].tolist()
        parts.append(f"{column}: {null_share:.0%} null, sample values {sample}")
    return "Referenced columns — " + "; ".join(parts)


@dataclass(frozen=True, slots=True)
class InteractionStoreBuildStats:
    """How the auxiliary frame mapped onto the final training catalog."""

    num_events_total: int = 0
    num_events_kept: int = 0
    num_events_dropped_unknown_user: int = 0
    num_events_dropped_unknown_item: int = 0
    events_per_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_events_total": self.num_events_total,
            "num_events_kept": self.num_events_kept,
            "num_events_dropped_unknown_user": self.num_events_dropped_unknown_user,
            "num_events_dropped_unknown_item": self.num_events_dropped_unknown_item,
            "events_per_type": dict(self.events_per_type),
        }


class InteractionLogStore:
    """Auxiliary interactions keyed by catalog IDs, with payload columns.

    ``events`` columns: ``user_id`` / ``item_id`` (catalog integers),
    ``event_type`` (string), ``timestamp`` (int64), plus any payload columns.
    Lookup indices are built lazily per event type and are always derived from
    ``events``, so the frame itself stays the single source of truth.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        num_users: int,
        item_vocab_size: int,
        stats: InteractionStoreBuildStats | None = None,
        out_of_catalog_item_ids: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        missing = [c for c in _STORE_ID_COLUMNS if c not in events.columns]
        if missing:
            raise ValueError(f"InteractionLogStore events missing columns: {missing}")
        self.events = events.reset_index(drop=True)
        self.num_users = int(num_users)
        self.item_vocab_size = int(item_vocab_size)
        # Raw IDs of items that catalog users interacted with in auxiliary
        # events but that did not make it into the positive catalog. Their
        # events are not in ``events`` (the store is keyed by catalog IDs),
        # but the IDs are kept so item-scope consumers — e.g. semantic-ID
        # tokenizer training — can cover the full positive+auxiliary universe.
        self.out_of_catalog_item_ids = frozenset(str(x) for x in out_of_catalog_item_ids)
        self.stats = stats or InteractionStoreBuildStats(
            num_events_total=len(events),
            num_events_kept=len(events),
            events_per_type=events["event_type"].value_counts().to_dict() if len(events) else {},
        )
        self._type_cache: dict[str, pd.DataFrame] = {}
        self._pair_key_cache: dict[str, np.ndarray] = {}
        self._user_index_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._payload_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._before_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.events)

    @property
    def payload_columns(self) -> list[str]:
        return [c for c in self.events.columns if c not in _STORE_ID_COLUMNS]

    def event_types(self) -> list[str]:
        return sorted(self.events["event_type"].astype(str).unique().tolist())

    def counts_by_type(self) -> dict[str, int]:
        return self.events["event_type"].astype(str).value_counts().to_dict()

    # -- internal indices ---------------------------------------------------

    def _pair_key(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        return user_ids.astype(np.int64) * self.item_vocab_size + item_ids.astype(np.int64)

    def _events_of_type(self, event_type: str) -> pd.DataFrame:
        if event_type not in self._type_cache:
            subset = self.events.loc[self.events["event_type"].astype(str) == event_type]
            self._type_cache[event_type] = subset.sort_values(
                ["user_id", "item_id", "timestamp"], kind="stable"
            ).reset_index(drop=True)
        return self._type_cache[event_type]

    def _pair_keys_of_type(self, event_type: str) -> np.ndarray:
        if event_type not in self._pair_key_cache:
            subset = self._events_of_type(event_type)
            keys = self._pair_key(
                subset["user_id"].to_numpy(), subset["item_id"].to_numpy()
            )
            self._pair_key_cache[event_type] = np.unique(keys)
        return self._pair_key_cache[event_type]

    # -- lookups ------------------------------------------------------------

    def has_pair(self, event_type: str, user_ids, item_ids) -> np.ndarray:
        """Vectorized membership: does (user, item) have this event type?"""
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        keys = self._pair_key(user_ids, item_ids)
        sorted_keys = self._pair_keys_of_type(event_type)
        if sorted_keys.size == 0:
            return np.zeros(keys.shape, dtype=bool)
        pos = np.searchsorted(sorted_keys, keys)
        pos = np.clip(pos, 0, sorted_keys.size - 1)
        return sorted_keys[pos] == keys

    def items_for_user(self, event_type: str, user_id: int) -> np.ndarray:
        """Distinct item IDs this user has the given event type for."""
        if event_type not in self._user_index_cache:
            subset = self._events_of_type(event_type)
            users = subset["user_id"].to_numpy()
            items = subset["item_id"].to_numpy()
            indptr = np.zeros(self.num_users + 1, dtype=np.int64)
            if users.size:
                counts = np.bincount(users, minlength=self.num_users)
                indptr[1:] = np.cumsum(counts)
            self._user_index_cache[event_type] = (indptr, items)
        indptr, items = self._user_index_cache[event_type]
        if not 0 <= user_id < self.num_users:
            return np.empty(0, dtype=np.int64)
        return np.unique(items[indptr[user_id] : indptr[user_id + 1]])

    def item_counts(self, event_type: str, *, distinct_users: bool = True) -> np.ndarray:
        """Per-item event counts as an array of length ``item_vocab_size``."""
        subset = self._events_of_type(event_type)
        if distinct_users:
            keys = self._pair_keys_of_type(event_type)
            items = (keys % self.item_vocab_size).astype(np.int64)
        else:
            items = subset["item_id"].to_numpy()
        return np.bincount(items, minlength=self.item_vocab_size).astype(np.int64)

    def user_counts(self, event_type: str, *, distinct_items: bool = True) -> np.ndarray:
        """Per-user event counts as an array of length ``num_users``."""
        if distinct_items:
            keys = self._pair_keys_of_type(event_type)
            users = (keys // self.item_vocab_size).astype(np.int64)
        else:
            users = self._events_of_type(event_type)["user_id"].to_numpy()
        return np.bincount(users, minlength=self.num_users).astype(np.int64)

    def payload_for_pairs(
        self,
        event_type: str,
        user_ids,
        item_ids,
        column: str,
        *,
        agg: PayloadAgg = "last",
        default: float = np.nan,
    ) -> np.ndarray:
        """Payload value of the event for each (user, item), ``default`` if absent.

        With multiple events per pair (e.g. repeated listens), ``agg`` picks the
        value: "last"/"first" by timestamp, or "max"/"min"/"mean" over the pair.
        """
        if agg not in _ALLOWED_AGGS:
            raise ValueError(f"agg must be one of {_ALLOWED_AGGS}, got {agg!r}")
        if self.events.empty:
            # An empty store (e.g. positive_criterion=None, or a criterion the
            # data never violates) carries no payload columns at all; every
            # lookup is simply unobserved.
            return np.full(np.asarray(user_ids).shape, float(default), dtype=float)
        if column not in self.events.columns:
            raise KeyError(
                f"payload column {column!r} not in store; available: {self.payload_columns}"
            )
        cache_key = (event_type, column, agg)
        if cache_key not in self._payload_cache:
            subset = self._events_of_type(event_type)
            keys = self._pair_key(subset["user_id"].to_numpy(), subset["item_id"].to_numpy())
            values = pd.to_numeric(subset[column], errors="coerce")
            grouped = values.groupby(keys, sort=True)
            aggregated = grouped.agg(agg)
            self._payload_cache[cache_key] = (
                aggregated.index.to_numpy(dtype=np.int64),
                aggregated.to_numpy(dtype=float),
            )
        sorted_keys, sorted_values = self._payload_cache[cache_key]

        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        keys = self._pair_key(user_ids, item_ids)
        out = np.full(keys.shape, float(default), dtype=float)
        if sorted_keys.size:
            pos = np.searchsorted(sorted_keys, keys)
            pos_clipped = np.clip(pos, 0, sorted_keys.size - 1)
            found = sorted_keys[pos_clipped] == keys
            out[found] = sorted_values[pos_clipped[found]]
        return out

    # -- time-aware lookups (no future leakage) -----------------------------

    def _pair_ranges_before(
        self,
        event_type: str,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        timestamps: np.ndarray,
        *,
        strict: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per query, the [start, end) slice of this pair's events before t.

        Events of a type are sorted by (user_id, item_id, timestamp), so pair
        keys are non-decreasing and timestamps ascend within each pair block.
        The inner timestamp search loops only over queries whose pair has any
        events — sparse for sampled negatives.
        """
        if event_type not in self._before_cache:
            subset = self._events_of_type(event_type)
            keys_all = self._pair_key(
                subset["user_id"].to_numpy(), subset["item_id"].to_numpy()
            )
            self._before_cache[event_type] = (keys_all, subset["timestamp"].to_numpy())
        keys_all, ts_all = self._before_cache[event_type]

        keys = self._pair_key(user_ids, item_ids)
        left = np.searchsorted(keys_all, keys, side="left")
        right = np.searchsorted(keys_all, keys, side="right")
        end = left.copy()
        side = "left" if strict else "right"
        for q in np.nonzero(right > left)[0]:
            end[q] = left[q] + np.searchsorted(
                ts_all[left[q] : right[q]], timestamps[q], side=side
            )
        return left, end

    def has_pair_before(
        self,
        event_type: str,
        user_ids,
        item_ids,
        timestamps,
        *,
        strict: bool = True,
    ) -> np.ndarray:
        """Does (user, item) have this event type before each query timestamp?

        ``strict=True`` (default) excludes events at exactly the query time —
        the reward-safe convention: an event at the target's own timestamp is
        not past information.
        """
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        left, end = self._pair_ranges_before(
            event_type, user_ids, item_ids, timestamps, strict=strict
        )
        return end > left

    def count_pairs_before(
        self,
        event_type: str,
        user_ids,
        item_ids,
        timestamps,
        *,
        strict: bool = True,
    ) -> np.ndarray:
        """Number of events of this type for (user, item) before each query time."""
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        left, end = self._pair_ranges_before(
            event_type, user_ids, item_ids, timestamps, strict=strict
        )
        return (end - left).astype(np.int64)

    def payload_for_pairs_before(
        self,
        event_type: str,
        user_ids,
        item_ids,
        timestamps,
        column: str,
        *,
        agg: PayloadAgg = "last",
        strict: bool = True,
        default: float = np.nan,
    ) -> np.ndarray:
        """Payload aggregated over the pair's events before each query time.

        Like :meth:`payload_for_pairs`, but only events strictly before (or at,
        with ``strict=False``) the per-query timestamp contribute. ``column``
        may be any events column, including ``timestamp`` itself (useful for
        "when did the user last do this" lookups).
        """
        if agg not in _ALLOWED_AGGS:
            raise ValueError(f"agg must be one of {_ALLOWED_AGGS}, got {agg!r}")
        if self.events.empty:
            return np.full(np.asarray(user_ids).shape, float(default), dtype=float)
        if column not in self.events.columns:
            raise KeyError(
                f"payload column {column!r} not in store; available: "
                f"{self.payload_columns + list(_STORE_ID_COLUMNS)}"
            )
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        left, end = self._pair_ranges_before(
            event_type, user_ids, item_ids, timestamps, strict=strict
        )

        values = pd.to_numeric(self._events_of_type(event_type)[column], errors="coerce").to_numpy(
            dtype=float
        )
        out = np.full(user_ids.shape, float(default), dtype=float)
        for q in np.nonzero(end > left)[0]:
            block = values[left[q] : end[q]]
            if agg == "last":
                out[q] = block[-1]
            elif agg == "first":
                out[q] = block[0]
            elif agg == "max":
                out[q] = np.nanmax(block)
            elif agg == "min":
                out[q] = np.nanmin(block)
            else:
                out[q] = np.nanmean(block)
        return out

    def filter_by_time(self, max_timestamp: int) -> "InteractionLogStore":
        """New store with only events at or before ``max_timestamp``."""
        kept = self.events.loc[self.events["timestamp"] <= int(max_timestamp)]
        return InteractionLogStore(
            kept.reset_index(drop=True),
            num_users=self.num_users,
            item_vocab_size=self.item_vocab_size,
            out_of_catalog_item_ids=self.out_of_catalog_item_ids,
        )

    # -- reporting / persistence -------------------------------------------

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "num_events": len(self.events),
            "num_event_types": len(self.event_types()),
            "counts_by_type": self.counts_by_type(),
            "payload_columns": self.payload_columns,
            "num_out_of_catalog_items": len(self.out_of_catalog_item_ids),
            "build_stats": self.stats.to_dict(),
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.events.to_parquet(directory / "events.parquet", index=False)
        meta = {
            "num_users": self.num_users,
            "item_vocab_size": self.item_vocab_size,
            "stats": self.stats.to_dict(),
            "out_of_catalog_item_ids": sorted(self.out_of_catalog_item_ids),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "InteractionLogStore":
        directory = Path(directory)
        events = pd.read_parquet(directory / "events.parquet")
        meta = json.loads((directory / "meta.json").read_text())
        stats = InteractionStoreBuildStats(
            num_events_total=meta["stats"]["num_events_total"],
            num_events_kept=meta["stats"]["num_events_kept"],
            num_events_dropped_unknown_user=meta["stats"]["num_events_dropped_unknown_user"],
            num_events_dropped_unknown_item=meta["stats"]["num_events_dropped_unknown_item"],
            events_per_type=meta["stats"]["events_per_type"],
        )
        return cls(
            events,
            num_users=meta["num_users"],
            item_vocab_size=meta["item_vocab_size"],
            stats=stats,
            out_of_catalog_item_ids=frozenset(meta.get("out_of_catalog_item_ids", ())),
        )


POSITIVE_EVENT_TYPE = "positive"


def build_positive_history_store(
    interactions: pd.DataFrame,
    *,
    num_users: int,
    item_vocab_size: int,
) -> InteractionLogStore:
    """Store view of the positive training stream (catalog IDs, no payload).

    Lets reward models answer "did this user positively interact with this
    item strictly before t?" without the truncation of the batch history
    window, using the same leakage-safe ``*_before`` lookups as the
    auxiliary store. Events carry ``event_type="positive"``.
    """
    required = ("user_id", "item_id", "timestamp")
    missing = [c for c in required if c not in interactions.columns]
    if missing:
        raise ValueError(f"positive interactions frame missing columns: {missing}")
    events = pd.DataFrame(
        {
            "user_id": interactions["user_id"].astype("int64").to_numpy(),
            "item_id": interactions["item_id"].astype("int64").to_numpy(),
            "event_type": POSITIVE_EVENT_TYPE,
            "timestamp": pd.to_numeric(interactions["timestamp"], errors="coerce")
            .fillna(0)
            .astype("int64")
            .to_numpy(),
        }
    )
    return InteractionLogStore(
        events, num_users=num_users, item_vocab_size=item_vocab_size
    )


def build_interaction_store(
    auxiliary: pd.DataFrame,
    *,
    user2id: Mapping[str, int],
    item2id: Mapping[str, int],
    num_users: int,
    item_vocab_size: int,
    default_event_type: str = "interaction",
) -> InteractionLogStore:
    """Map an auxiliary frame onto the final training catalog.

    Events whose user or item did not survive the positive-stream pipeline
    (k-core, cold-item filter) are dropped and counted in the build stats —
    the catalog is defined by positives only and must not grow here.
    """
    if auxiliary.empty:
        empty = pd.DataFrame(
            {
                "user_id": pd.Series(dtype="int64"),
                "item_id": pd.Series(dtype="int64"),
                "event_type": pd.Series(dtype="object"),
                "timestamp": pd.Series(dtype="int64"),
            }
        )
        return InteractionLogStore(
            empty, num_users=num_users, item_vocab_size=item_vocab_size
        )

    user_ids = auxiliary["raw_user_id"].astype(str).map(user2id)
    item_ids = auxiliary["raw_item_id"].astype(str).map(item2id)
    known_user = user_ids.notna()
    known_item = item_ids.notna()
    kept_mask = known_user & known_item
    # Items catalog users touched only in auxiliary events; their raw IDs are
    # kept so item-scope consumers (e.g. tokenizer training) can include them.
    out_of_catalog_item_ids = frozenset(
        auxiliary.loc[known_user & ~known_item, "raw_item_id"].astype(str)
    )

    events = auxiliary.loc[kept_mask].drop(columns=["raw_user_id", "raw_item_id"]).copy()
    events.insert(0, "user_id", user_ids.loc[kept_mask].astype("int64").to_numpy())
    events.insert(1, "item_id", item_ids.loc[kept_mask].astype("int64").to_numpy())
    if "event_type" not in events.columns:
        events["event_type"] = default_event_type
    events["event_type"] = events["event_type"].astype(str)
    events["timestamp"] = (
        pd.to_numeric(events["timestamp"], errors="coerce").fillna(0).astype("int64")
    )
    events = events.drop(columns=[c for c in ("order_idx", "raw") if c in events.columns])
    events = events.reset_index(drop=True)

    stats = InteractionStoreBuildStats(
        num_events_total=int(len(auxiliary)),
        num_events_kept=int(kept_mask.sum()),
        num_events_dropped_unknown_user=int((~known_user).sum()),
        num_events_dropped_unknown_item=int((known_user & ~known_item).sum()),
        events_per_type=events["event_type"].value_counts().to_dict() if len(events) else {},
    )
    return InteractionLogStore(
        events,
        num_users=num_users,
        item_vocab_size=item_vocab_size,
        stats=stats,
        out_of_catalog_item_ids=out_of_catalog_item_ids,
    )
