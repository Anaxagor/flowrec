"""Indexed auxiliary interactions excluded from the positive training stream."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


PayloadAggregation = Literal["last", "max", "mean"]
_IDENTITY_COLUMNS = ("raw_user_id", "raw_item_id", "timestamp", "order_idx")


@dataclass(frozen=True, slots=True)
class InteractionStoreStats:
    num_events_total: int = 0
    num_events_stored: int = 0
    num_events_dropped_unknown_user: int = 0
    num_events_dropped_unknown_item: int = 0


def partition_interactions(
    interactions: pd.DataFrame,
    positive_criterion: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positive training events from payload-rich auxiliary events."""
    if positive_criterion is None:
        return interactions, interactions.iloc[0:0].copy()
    try:
        selected = interactions.eval(positive_criterion).fillna(False).astype(bool)
    except Exception as exc:
        raise ValueError(
            f"Invalid positive_criterion={positive_criterion!r}; available columns: "
            f"{sorted(interactions.columns)}"
        ) from exc
    if not bool(selected.any()):
        nulls = ", ".join(
            f"{column}: {interactions[column].isna().mean():.0%} null"
            for column in interactions.columns
            if interactions[column].isna().any()
        )
        detail = f" Null diagnostics: {nulls}." if nulls else ""
        raise ValueError(f"positive_criterion matched no interactions.{detail}")

    positive_columns = [column for column in _IDENTITY_COLUMNS if column in interactions]
    positives = interactions.loc[selected, positive_columns].reset_index(drop=True)
    auxiliary = interactions.loc[~selected].reset_index(drop=True)
    return positives, auxiliary


class InteractionLogStore:
    """Compact DataFrame-backed lookup store keyed by event type, user and item."""

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        num_users: int,
        item_vocab_size: int,
        stats: InteractionStoreStats | None = None,
        out_of_catalog_item_ids: Iterable[str] = (),
    ) -> None:
        required = {"user_id", "item_id", "event_type", "timestamp"}
        missing = required - set(events.columns)
        if missing and not events.empty:
            raise ValueError(f"InteractionLogStore events are missing columns: {sorted(missing)}")
        self.events = events.reset_index(drop=True).copy()
        self.num_users = int(num_users)
        self.item_vocab_size = int(item_vocab_size)
        self.stats = stats or InteractionStoreStats(
            num_events_total=len(events), num_events_stored=len(events)
        )
        self.out_of_catalog_item_ids = {str(item) for item in out_of_catalog_item_ids}

    def __len__(self) -> int:
        return len(self.events)

    @property
    def payload_columns(self) -> list[str]:
        keys = {"user_id", "item_id", "event_type", "timestamp"}
        return [column for column in self.events.columns if column not in keys]

    def event_types(self) -> list[str]:
        if self.events.empty:
            return []
        return sorted(self.events["event_type"].astype(str).unique().tolist())

    def counts_by_type(self) -> dict[str, int]:
        if self.events.empty:
            return {}
        return {
            str(key): int(value)
            for key, value in self.events["event_type"].value_counts().sort_index().items()
        }

    def _event_frame(self, event_type: str) -> pd.DataFrame:
        if self.events.empty:
            return self.events
        return self.events.loc[self.events["event_type"].astype(str).eq(str(event_type))]

    def has_pair(
        self,
        event_type: str,
        user_ids: Sequence[int] | np.ndarray,
        item_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        frame = self._event_frame(event_type)
        pairs = set(zip(frame.get("user_id", []), frame.get("item_id", [])))
        return np.asarray(
            [(int(user), int(item)) in pairs for user, item in zip(user_ids, item_ids)],
            dtype=bool,
        )

    def has_pair_before(
        self,
        event_type: str,
        user_ids: Sequence[int] | np.ndarray,
        item_ids: Sequence[int] | np.ndarray,
        timestamps: Sequence[int] | np.ndarray,
        *,
        strict: bool = True,
    ) -> np.ndarray:
        values = self.payload_for_pairs_before(
            event_type,
            user_ids,
            item_ids,
            timestamps,
            "timestamp",
            agg="last",
            strict=strict,
        )
        return ~np.isnan(values)

    def items_for_user(self, event_type: str, user_id: int) -> np.ndarray:
        frame = self._event_frame(event_type)
        values = frame.loc[frame["user_id"].eq(int(user_id)), "item_id"].unique()
        return np.sort(values.astype(np.int64))

    def item_counts(self, event_type: str) -> np.ndarray:
        counts = np.zeros(self.item_vocab_size, dtype=np.int64)
        frame = self._event_frame(event_type)
        if not frame.empty:
            ids, values = np.unique(frame["item_id"].to_numpy(dtype=np.int64), return_counts=True)
            valid = (ids >= 0) & (ids < self.item_vocab_size)
            counts[ids[valid]] = values[valid]
        return counts

    def user_counts(self, event_type: str) -> np.ndarray:
        counts = np.zeros(self.num_users, dtype=np.int64)
        frame = self._event_frame(event_type)
        if not frame.empty:
            ids, values = np.unique(frame["user_id"].to_numpy(dtype=np.int64), return_counts=True)
            valid = (ids >= 0) & (ids < self.num_users)
            counts[ids[valid]] = values[valid]
        return counts

    def payload_for_pairs(
        self,
        event_type: str,
        user_ids: Sequence[int] | np.ndarray,
        item_ids: Sequence[int] | np.ndarray,
        column: str,
        *,
        agg: PayloadAggregation = "last",
        default: float = np.nan,
    ) -> np.ndarray:
        frame = self._event_frame(event_type)
        return self._lookup_payload(
            frame, user_ids, item_ids, column, agg=agg, default=default
        )

    def payload_for_pairs_before(
        self,
        event_type: str,
        user_ids: Sequence[int] | np.ndarray,
        item_ids: Sequence[int] | np.ndarray,
        timestamps: Sequence[int] | np.ndarray,
        column: str,
        *,
        agg: PayloadAggregation = "last",
        strict: bool = True,
        default: float = np.nan,
    ) -> np.ndarray:
        frame = self._event_frame(event_type)
        output = np.full(len(user_ids), float(default), dtype=float)
        comparator = np.less if strict else np.less_equal
        for index, (user, item, cutoff) in enumerate(zip(user_ids, item_ids, timestamps)):
            selected = frame.loc[
                frame["user_id"].eq(int(user))
                & frame["item_id"].eq(int(item))
                & comparator(frame["timestamp"].to_numpy(), int(cutoff))
            ]
            output[index] = self._aggregate(selected, column, agg, default)
        return output

    def _lookup_payload(
        self,
        frame: pd.DataFrame,
        user_ids: Sequence[int] | np.ndarray,
        item_ids: Sequence[int] | np.ndarray,
        column: str,
        *,
        agg: PayloadAggregation,
        default: float,
    ) -> np.ndarray:
        output = np.full(len(user_ids), float(default), dtype=float)
        for index, (user, item) in enumerate(zip(user_ids, item_ids)):
            selected = frame.loc[
                frame["user_id"].eq(int(user)) & frame["item_id"].eq(int(item))
            ]
            output[index] = self._aggregate(selected, column, agg, default)
        return output

    @staticmethod
    def _aggregate(
        frame: pd.DataFrame,
        column: str,
        agg: PayloadAggregation,
        default: float,
    ) -> float:
        if frame.empty or column not in frame:
            return float(default)
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return float(default)
        if agg == "last":
            ordered = frame.assign(_value=pd.to_numeric(frame[column], errors="coerce"))
            ordered = ordered.dropna(subset=["_value"]).sort_values("timestamp", kind="stable")
            return float(ordered["_value"].iloc[-1])
        if agg == "max":
            return float(values.max())
        if agg == "mean":
            return float(values.mean())
        raise ValueError(f"Unsupported aggregation: {agg!r}")

    def filter_by_time(self, cutoff: int, *, strict: bool = False) -> "InteractionLogStore":
        mask = self.events["timestamp"].lt(cutoff) if strict else self.events["timestamp"].le(cutoff)
        return InteractionLogStore(
            self.events.loc[mask].reset_index(drop=True),
            num_users=self.num_users,
            item_vocab_size=self.item_vocab_size,
            stats=self.stats,
            out_of_catalog_item_ids=self.out_of_catalog_item_ids,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "num_events": len(self),
            "event_counts": self.counts_by_type(),
            "num_out_of_catalog_items": len(self.out_of_catalog_item_ids),
            **asdict(self.stats),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.events.to_pickle(path / "events.pkl")
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "num_users": self.num_users,
                    "item_vocab_size": self.item_vocab_size,
                    "stats": asdict(self.stats),
                    "out_of_catalog_item_ids": sorted(self.out_of_catalog_item_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "InteractionLogStore":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            pd.read_pickle(path / "events.pkl"),
            num_users=int(metadata["num_users"]),
            item_vocab_size=int(metadata["item_vocab_size"]),
            stats=InteractionStoreStats(**metadata["stats"]),
            out_of_catalog_item_ids=metadata.get("out_of_catalog_item_ids", ()),
        )


def build_interaction_store(
    auxiliary: pd.DataFrame,
    *,
    user2id: dict[str, int],
    item2id: dict[str, int],
    num_users: int,
    item_vocab_size: int,
    default_event_type: str | None = None,
) -> InteractionLogStore:
    """Map raw auxiliary events onto the final positive catalog."""
    total = len(auxiliary)
    if auxiliary.empty:
        columns = ["user_id", "item_id", "event_type", "timestamp"]
        return InteractionLogStore(
            pd.DataFrame(columns=columns),
            num_users=num_users,
            item_vocab_size=item_vocab_size,
            stats=InteractionStoreStats(),
        )

    frame = auxiliary.copy()
    raw_users = frame["raw_user_id"].astype(str)
    raw_items = frame["raw_item_id"].astype(str)
    mapped_users = raw_users.map(user2id)
    mapped_items = raw_items.map(item2id)
    unknown_users = mapped_users.isna()
    unknown_items = mapped_items.isna()
    out_of_catalog = set(raw_items.loc[unknown_items].tolist())
    keep = ~(unknown_users | unknown_items)
    frame = frame.loc[keep].copy()
    frame["user_id"] = mapped_users.loc[keep].astype(np.int64)
    frame["item_id"] = mapped_items.loc[keep].astype(np.int64)
    if "event_type" not in frame:
        frame["event_type"] = default_event_type or "interaction"
    elif default_event_type is not None:
        frame["event_type"] = frame["event_type"].fillna(default_event_type)
    frame = frame.drop(columns=["raw_user_id", "raw_item_id"], errors="ignore")
    stats = InteractionStoreStats(
        num_events_total=total,
        num_events_stored=len(frame),
        num_events_dropped_unknown_user=int(unknown_users.sum()),
        num_events_dropped_unknown_item=int(unknown_items.sum()),
    )
    return InteractionLogStore(
        frame,
        num_users=num_users,
        item_vocab_size=item_vocab_size,
        stats=stats,
        out_of_catalog_item_ids=out_of_catalog,
    )
