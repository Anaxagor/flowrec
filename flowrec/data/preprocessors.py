from __future__ import annotations

import gzip
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd

from .datasets import InteractionRecord

ItemKey = Literal["parent_asin", "asin"]
_JSONL_CHUNK_SIZE = 10000000


@contextmanager
def open_text(path: str | Path) -> Iterator[Any]:
    """Open plain-text or gzip-compressed files in text mode."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield f


def _is_jsonl(path: str | Path) -> bool:
    suffixes = "".join(Path(path).suffixes).lower()
    return suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz")


def _jsonl_chunks(path: str | Path, *, chunksize: int = _JSONL_CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Read strict JSONL into pandas chunks.

    This intentionally has one supported behavior: newline-delimited JSON records.
    Malformed rows, Python-literal Amazon dumps, or full JSON files raise directly
    instead of falling through to a slower parser.
    """
    yield from pd.read_json(
        path,
        lines=True,
        compression="infer",
        chunksize=chunksize,
    )


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield strict JSONL records using pandas as the only reader."""
    if not _is_jsonl(path):
        raise ValueError(f"Expected a .jsonl or .jsonl.gz file, got: {path}")
    for chunk in _jsonl_chunks(path):
        yield from chunk.to_dict(orient="records")


def load_json_or_jsonl(path: str | Path) -> Any:
    """Load strict JSON or strict JSONL.

    JSONL returns a list of record dictionaries. JSON returns the parsed JSON
    document. There is no legacy Python-literal compatibility branch.
    """
    path = Path(path)
    if _is_jsonl(path):
        records: list[dict[str, Any]] = []
        for chunk in _jsonl_chunks(path):
            records.extend(chunk.to_dict(orient="records"))
        return records
    with open_text(path) as f:
        return json.load(f)


def _first_series(frame: pd.DataFrame, *columns: str) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    out = frame[present[0]].copy()
    for column in present[1:]:
        out = out.where(out.notna(), frame[column])
    return out
def _coerce_optional_bool_series(series: pd.Series) -> pd.Series:
    """Normalize optional boolean fields without treating 'False' as true."""
    out = pd.Series([None] * len(series), index=series.index, dtype="object")
    notna = series.notna()
    if not notna.any():
        return out

    normalized = series.astype("string").str.strip().str.lower()
    true_mask = normalized.isin(["true", "t", "yes", "y", "1"])
    false_mask = normalized.isin(["false", "f", "no", "n", "0", ""])
    out.loc[true_mask] = True
    out.loc[false_mask] = False

    other_mask = notna & ~(true_mask | false_mask)
    if other_mask.any():
        out.loc[other_mask] = series.loc[other_mask].astype(bool)
    return out


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", ""}:
            return False
    return bool(value)


def extract_timestamp(record: dict[str, Any]) -> int:
    value = record.get("timestamp", record.get("sort_timestamp", record.get("unixReviewTime", 0)))
    return 0 if value is None else int(value)


def normalize_review_record(
    record: dict[str, Any],
    *,
    item_key: ItemKey = "parent_asin",
    use_verified_only: bool = False,
    keep_raw: bool = False,
    order_idx: int = 0,
) -> InteractionRecord | None:
    """Normalize one already-parsed Amazon row.

    Kept for callers/tests that normalize individual rows. The datamodule path
    uses the vectorized loader below.
    """
    verified_purchase = _coerce_optional_bool(record.get("verified_purchase", record.get("verified")))
    if use_verified_only and verified_purchase is not True:
        return None

    raw_user_id = record.get("user_id", record.get("reviewerID", record.get("reviewer_id")))
    raw_item_id = record.get(item_key) or record.get("parent_asin") or record.get("asin")
    if raw_user_id is None or raw_item_id is None:
        return None

    rating = record.get("rating", record.get("overall", 0.0))
    rating_float = pd.to_numeric(pd.Series([rating]), errors="coerce").fillna(0.0).iloc[0]
    return InteractionRecord(
        raw_user_id=str(raw_user_id),
        raw_item_id=str(raw_item_id),
        timestamp=extract_timestamp(record),
        rating=float(rating_float),
        verified_purchase=verified_purchase,
        order_idx=order_idx,
        raw=dict(record) if keep_raw else {},
    )


def _normalize_amazon_review_frame(
    frame: pd.DataFrame,
    *,
    item_key: ItemKey,
    order_offset: int,
    use_verified_only: bool,
    keep_raw: bool,
    allowed_items: set[str] | None,
) -> pd.DataFrame:
    raw_user = _first_series(frame, "user_id", "reviewerID", "reviewer_id")
    raw_item = _first_series(frame, item_key, "parent_asin", "asin")
    timestamp = pd.to_numeric(
        _first_series(frame, "timestamp", "sort_timestamp", "unixReviewTime"),
        errors="coerce",
    ).fillna(0).astype("int64")
    rating = pd.to_numeric(_first_series(frame, "rating", "overall"), errors="coerce").fillna(0.0)
    verified = _coerce_optional_bool_series(_first_series(frame, "verified_purchase", "verified"))

    mask = raw_user.notna() & raw_item.notna()
    if use_verified_only:
        mask &= verified.eq(True)
    if allowed_items is not None:
        mask &= raw_item.astype("string").isin(allowed_items)

    idx = np.arange(order_offset, order_offset + len(frame), dtype=np.int64)
    out = pd.DataFrame(
        {
            "raw_user_id": raw_user.loc[mask].astype("string").to_numpy(),
            "raw_item_id": raw_item.loc[mask].astype("string").to_numpy(),
            "timestamp": timestamp.loc[mask].to_numpy(dtype=np.int64),
            "rating": rating.loc[mask].to_numpy(dtype=float),
            "verified_purchase": verified.loc[mask].to_numpy(dtype=object),
            "order_idx": idx[mask.to_numpy()],
        }
    )
    if keep_raw:
        out["raw"] = frame.loc[mask].to_dict(orient="records")
    return out


def _interaction_frame_to_user_events(frame: pd.DataFrame, *, raw_column: str | None = None) -> dict[str, list[InteractionRecord]]:
    if frame.empty:
        return {}

    records = [
        InteractionRecord(
            raw_user_id=str(row.raw_user_id),
            raw_item_id=str(row.raw_item_id),
            timestamp=int(row.timestamp),
            rating=float(row.rating),
            verified_purchase=row.verified_purchase,
            order_idx=int(row.order_idx),
            raw=getattr(row, raw_column) if raw_column else {},
        )
        for row in frame.itertuples(index=False)
    ]
    tmp = pd.DataFrame(
        {
            "raw_user_id": frame["raw_user_id"].astype("string").to_numpy(),
            "event": records,
        }
    )
    return {
        str(raw_user_id): list(events)
        for raw_user_id, events in tmp.groupby("raw_user_id", sort=False)["event"].agg(list).items()
    }


def load_reviews_by_user(
    reviews_path: str | Path,
    *,
    item_key: ItemKey = "parent_asin",
    use_verified_only: bool = False,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
) -> dict[str, list[InteractionRecord]]:
    """Load Amazon review interactions from strict JSONL with a single vectorized path."""
    if not _is_jsonl(reviews_path):
        raise ValueError(
            "load_reviews_by_user expects strict .jsonl or .jsonl.gz input. "
            "Convert legacy JSON/Python-literal dumps once before training."
        )

    grouped_frames: list[pd.DataFrame] = []
    order_offset = 0
    for _, chunk in enumerate(_jsonl_chunks(reviews_path)):
        print(_)
        raw_len = len(chunk)
        frame = _normalize_amazon_review_frame(
            chunk,
            item_key=item_key,
            order_offset=order_offset,
            use_verified_only=use_verified_only,
            keep_raw=keep_raw,
            allowed_items=allowed_items,
        )
        order_offset += raw_len
        if not frame.empty:
            grouped_frames.append(frame)

    if not grouped_frames:
        return {}
    events = pd.concat(grouped_frames, ignore_index=True)
    return _interaction_frame_to_user_events(events, raw_column="raw" if keep_raw else None)


def drop_consecutive_repeats(
    data: pd.DataFrame,
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp",
) -> pd.DataFrame:
    """Remove immediate repeated items in each user's chronological sequence."""
    missing = [column for column in (user_id, item_id, timestamp) if column not in data.columns]
    if missing:
        raise KeyError(f"drop_consecutive_repeats missing columns: {missing}")

    data_sorted = data.sort_values([user_id, timestamp], kind="stable").copy()
    shifted_col = "__flowrec_shifted_item__"
    data_sorted[shifted_col] = data_sorted.groupby(user_id, sort=False)[item_id].shift(1)
    return (
        data_sorted.loc[data_sorted[item_id] != data_sorted[shifted_col]]
        .drop(columns=[shifted_col])
        .reset_index(drop=True)
    )


def _user_events_to_frame(user_events: dict[str, list[InteractionRecord]]) -> pd.DataFrame:
    records = [
        (
            event.raw_user_id,
            event.raw_item_id,
            int(event.timestamp),
            int(event.order_idx),
            event,
        )
        for events in user_events.values()
        for event in events
    ]
    return pd.DataFrame(
        records,
        columns=["raw_user_id", "raw_item_id", "timestamp", "order_idx", "event"],
    )


def _sort_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.sort_values("order_idx", kind="stable").copy()
    out["__user_code"] = out["raw_user_id"].factorize(sort=False)[0]
    out = out.sort_values(["__user_code", "timestamp", "order_idx"], kind="stable")
    return out.drop(columns=["__user_code"])


def _frame_to_user_events(
    frame: pd.DataFrame,
    *,
    empty_users: Iterable[str] = (),
) -> dict[str, list[InteractionRecord]]:
    if frame.empty:
        return {str(user): [] for user in empty_users}
    grouped = frame.groupby("raw_user_id", sort=False)["event"].agg(list).to_dict()
    out = {str(user): list(events) for user, events in grouped.items()}
    for user in empty_users:
        out.setdefault(str(user), [])
    return out


def sort_user_events(
    user_events: dict[str, list[InteractionRecord]],
) -> dict[str, list[InteractionRecord]]:
    """Sort events by user and timestamp using vectorized DataFrame operations."""
    empty_users = [raw_user_id for raw_user_id, events in user_events.items() if not events]
    frame = _user_events_to_frame(user_events)
    if frame.empty:
        return {raw_user_id: [] for raw_user_id in user_events}
    return _frame_to_user_events(_sort_event_frame(frame), empty_users=empty_users)


def _deduplicate_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sorted_frame = _sort_event_frame(frame)
    previous_item = sorted_frame.groupby("raw_user_id", sort=False)["raw_item_id"].shift(1)
    return sorted_frame.loc[sorted_frame["raw_item_id"].ne(previous_item)].copy()


def deduplicate_consecutive_items(
    user_events: dict[str, list[InteractionRecord]],
) -> dict[str, list[InteractionRecord]]:
    """Remove consecutive duplicate item interactions per user."""
    frame = _user_events_to_frame(user_events)
    if frame.empty:
        return {}
    return _frame_to_user_events(_deduplicate_event_frame(frame))


def _deduplicate_sorted_event_list(events: list[InteractionRecord]) -> list[InteractionRecord]:
    frame = _user_events_to_frame({"__single_user__": events})
    if frame.empty:
        return []
    return list(_deduplicate_event_frame(frame)["event"])


def apply_k_core_filter(
    user_events: dict[str, list[InteractionRecord]],
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 0,
    drop_consecutive_repeats: bool = False,
) -> dict[str, list[InteractionRecord]]:
    """Iterative user/item k-core filtering with vectorized group counts."""
    if min_user_interactions < 1:
        raise ValueError("min_user_interactions must be >= 1")
    if min_item_interactions < 0:
        raise ValueError("min_item_interactions must be >= 0")

    frame = _user_events_to_frame(user_events)
    if frame.empty:
        return {}

    if drop_consecutive_repeats:
        frame = _deduplicate_event_frame(frame)

    while True:
        before_users = int(frame["raw_user_id"].nunique())
        before_interactions = int(len(frame))

        user_counts = frame.groupby("raw_user_id", sort=False).size()
        keep_users = user_counts.index[user_counts >= int(min_user_interactions)]
        frame = frame.loc[frame["raw_user_id"].isin(keep_users)]

        if min_item_interactions > 0 and not frame.empty:
            item_counts = frame.groupby("raw_item_id", sort=False).size()
            keep_items = item_counts.index[item_counts >= int(min_item_interactions)]
            frame = frame.loc[frame["raw_item_id"].isin(keep_items)]

            if drop_consecutive_repeats and not frame.empty:
                frame = _deduplicate_event_frame(frame)

            if not frame.empty:
                user_counts = frame.groupby("raw_user_id", sort=False).size()
                keep_users = user_counts.index[user_counts >= int(min_user_interactions)]
                frame = frame.loc[frame["raw_user_id"].isin(keep_users)]

        after_users = int(frame["raw_user_id"].nunique()) if not frame.empty else 0
        after_interactions = int(len(frame))
        if (after_users, after_interactions) == (before_users, before_interactions):
            break
        if after_interactions == 0:
            break

    if frame.empty:
        return {}
    return _frame_to_user_events(frame)


def load_metadata(
    metadata_path: str | Path,
    *,
    item_key: ItemKey = "parent_asin",
) -> dict[str, dict[str, Any]]:
    """Load Amazon metadata from strict JSONL using pandas chunks."""
    if not _is_jsonl(metadata_path):
        raw = load_json_or_jsonl(metadata_path)
        if isinstance(raw, list):
            frame = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            frame = pd.DataFrame.from_dict(raw, orient="index").reset_index(names="__raw_key__")
            if item_key not in frame.columns and "parent_asin" not in frame.columns and "asin" not in frame.columns:
                frame[item_key] = frame["__raw_key__"]
        else:
            raise ValueError(f"Unsupported metadata JSON content: {type(raw)}")
        return _metadata_frame_to_dict(frame, item_key=item_key)

    metadata: dict[str, dict[str, Any]] = {}
    for chunk in _jsonl_chunks(metadata_path):
        metadata.update(_metadata_frame_to_dict(chunk, item_key=item_key))
    return metadata


def _metadata_frame_to_dict(frame: pd.DataFrame, *, item_key: ItemKey) -> dict[str, dict[str, Any]]:
    raw_item = _first_series(frame, item_key, "parent_asin", "asin", "raw_item_id", "item_id")
    mask = raw_item.notna()
    if not mask.any():
        return {}
    records = frame.loc[mask].to_dict(orient="records")
    keys = raw_item.loc[mask].astype("string").to_numpy()
    return {str(key): record for key, record in zip(keys, records, strict=False)}


def metadata_to_text(metadata: dict[str, Any]) -> str:
    """Build a robust text representation from Amazon metadata."""
    parts: list[str] = []

    for key in ["title", "main_category", "main_cat", "store", "brand"]:
        value = metadata.get(key)
        if value:
            parts.append(str(value))

    for key in ["features", "feature", "description", "categories", "category"]:
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value if x)
        elif value:
            parts.append(str(value))

    details = metadata.get("details")
    if isinstance(details, dict):
        parts.extend(f"{k}: {v}" for k, v in details.items() if k and v)

    price = metadata.get("price")
    if price not in (None, ""):
        parts.append(f"price: {price}")

    return " ".join(parts)


def load_semantic_ids(
    path: str | Path,
    *,
    offset_by_one: bool = True,
) -> tuple[dict[str, list[int]], int]:
    """Load item semantic IDs from strict JSON or JSONL."""
    raw = load_json_or_jsonl(path)
    semantic_ids: dict[str, list[int]] = {}

    def normalize_tokens(value: Any) -> list[int]:
        tokens = value.get("tokens", value.get("semantic_id", value.get("sid"))) if isinstance(value, dict) else value
        if not isinstance(tokens, list):
            raise ValueError(f"Semantic ID value must be a list, got: {type(tokens)}")
        int_tokens = [int(x) for x in tokens]
        return [x + 1 for x in int_tokens] if offset_by_one else int_tokens

    if isinstance(raw, dict):
        for raw_item_id, value in raw.items():
            semantic_ids[str(raw_item_id)] = normalize_tokens(value)
    elif isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                raise ValueError("JSONL semantic ID records must be objects.")
            raw_item_id = (
                record.get("item_id")
                or record.get("parent_asin")
                or record.get("asin")
                or record.get("raw_item_id")
            )
            if raw_item_id is None:
                raise ValueError("Semantic ID record is missing item_id/parent_asin/asin.")
            semantic_ids[str(raw_item_id)] = normalize_tokens(record)
    else:
        raise ValueError(f"Unsupported semantic ID file content: {type(raw)}")

    sid_len = max((len(tokens) for tokens in semantic_ids.values()), default=0)
    if sid_len == 0:
        raise ValueError(f"No semantic IDs found in {path}")
    return semantic_ids, sid_len


def collect_item_ids(user_events: dict[str, list[InteractionRecord]]) -> set[str]:
    return {event.raw_item_id for events in user_events.values() for event in events}


def count_interactions(user_events: dict[str, list[InteractionRecord]]) -> int:
    return sum(len(events) for events in user_events.values())
