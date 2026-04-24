from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from .datasets import InteractionRecord

ItemKey = Literal["parent_asin", "asin"]


@contextmanager
def open_text(path: str | Path) -> Iterator[Any]:
    """Open plain-text or gzip-compressed files in text mode."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield f


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from `.jsonl` or `.jsonl.gz`."""
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def load_json_or_jsonl(path: str | Path) -> Any:
    """Load `.json`, `.json.gz`, `.jsonl`, or `.jsonl.gz`."""
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        return list(iter_jsonl(path))
    with open_text(path) as f:
        return json.load(f)


def extract_timestamp(record: dict[str, Any]) -> int:
    """Extract a timestamp from Amazon Review records.

    Amazon Reviews 2023 uses `timestamp`. Some derived datasets use
    `sort_timestamp`; we support both.
    """
    value = record.get("timestamp", record.get("sort_timestamp", 0))
    if value is None:
        return 0
    return int(value)


def normalize_review_record(
    record: dict[str, Any],
    *,
    item_key: ItemKey = "parent_asin",
    use_verified_only: bool = False,
    keep_raw: bool = False,
    order_idx: int = 0,
) -> InteractionRecord | None:
    """Convert one Amazon review JSON object into an InteractionRecord.

    Returns None for unusable rows, such as missing user or item IDs.
    """
    if use_verified_only and not bool(record.get("verified_purchase", False)):
        return None

    raw_user_id = record.get("user_id")
    raw_item_id = record.get(item_key)

    if raw_user_id is None or raw_item_id is None:
        return None

    rating = record.get("rating", 0.0)
    try:
        rating_float = float(rating)
    except (TypeError, ValueError):
        rating_float = 0.0

    return InteractionRecord(
        raw_user_id=str(raw_user_id),
        raw_item_id=str(raw_item_id),
        timestamp=extract_timestamp(record),
        rating=rating_float,
        verified_purchase=record.get("verified_purchase"),
        order_idx=order_idx,
        raw=dict(record) if keep_raw else {},
    )


def load_reviews_by_user(
    reviews_path: str | Path,
    *,
    item_key: ItemKey = "parent_asin",
    use_verified_only: bool = False,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
) -> dict[str, list[InteractionRecord]]:
    """Load Amazon review interactions grouped by user."""
    grouped: dict[str, list[InteractionRecord]] = defaultdict(list)

    for order_idx, record in enumerate(iter_jsonl(reviews_path)):
        event = normalize_review_record(
            record,
            item_key=item_key,
            use_verified_only=use_verified_only,
            keep_raw=keep_raw,
            order_idx=order_idx,
        )
        if event is None:
            continue
        if allowed_items is not None and event.raw_item_id not in allowed_items:
            continue
        grouped[event.raw_user_id].append(event)

    return dict(grouped)


def sort_user_events(
    user_events: dict[str, list[InteractionRecord]],
) -> dict[str, list[InteractionRecord]]:
    """Sort every user's events chronologically with stable tie-breaking."""
    return {
        user: sorted(events, key=lambda e: (e.timestamp, e.order_idx))
        for user, events in user_events.items()
    }


def deduplicate_consecutive_items(
    user_events: dict[str, list[InteractionRecord]],
) -> dict[str, list[InteractionRecord]]:
    """Remove consecutive duplicate item interactions per user.

    This is optional. Some RecSys pipelines keep all repeated interactions; others
    collapse immediate duplicates that may be logging artifacts.
    """
    output: dict[str, list[InteractionRecord]] = {}
    for user, events in sort_user_events(user_events).items():
        kept: list[InteractionRecord] = []
        last_item: str | None = None
        for event in events:
            if event.raw_item_id == last_item:
                continue
            kept.append(event)
            last_item = event.raw_item_id
        output[user] = kept
    return output


def apply_k_core_filter(
    user_events: dict[str, list[InteractionRecord]],
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 0,
) -> dict[str, list[InteractionRecord]]:
    """Iterative user/item k-core filtering.

    `min_item_interactions=0` means only user-length filtering is applied.
    """
    if min_user_interactions < 1:
        raise ValueError("min_user_interactions must be >= 1")
    if min_item_interactions < 0:
        raise ValueError("min_item_interactions must be >= 0")

    filtered = {
        user: events
        for user, events in user_events.items()
        if len(events) >= min_user_interactions
    }

    if min_item_interactions == 0:
        return filtered

    changed = True
    while changed:
        changed = False
        item_counts = Counter(
            event.raw_item_id for events in filtered.values() for event in events
        )

        next_filtered: dict[str, list[InteractionRecord]] = {}
        for user, events in filtered.items():
            kept = [
                event
                for event in events
                if item_counts[event.raw_item_id] >= min_item_interactions
            ]
            if len(kept) >= min_user_interactions:
                next_filtered[user] = kept
            if len(kept) != len(events) or user not in next_filtered:
                changed = True

        filtered = next_filtered

    return filtered


def load_metadata(
    metadata_path: str | Path,
    *,
    item_key: ItemKey = "parent_asin",
) -> dict[str, dict[str, Any]]:
    """Load Amazon metadata keyed by item ID."""
    metadata: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(metadata_path):
        raw_item_id = record.get(item_key) or record.get("parent_asin") or record.get("asin")
        if raw_item_id is None:
            continue
        metadata[str(raw_item_id)] = record
    return metadata


def metadata_to_text(metadata: dict[str, Any]) -> str:
    """Build a robust text representation from Amazon metadata."""
    parts: list[str] = []

    for key in ["title", "main_category", "store"]:
        value = metadata.get(key)
        if value:
            parts.append(str(value))

    for key in ["features", "description", "categories"]:
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
    """Load item semantic IDs.

    Supported formats:

    1. JSON dictionary:
       {"ITEM": [1, 2, 3]}
       {"ITEM": {"tokens": [1, 2, 3]}}

    2. JSONL records:
       {"item_id": "ITEM", "tokens": [1, 2, 3]}
       {"parent_asin": "ITEM", "semantic_id": [1, 2, 3]}

    If offset_by_one=True, token IDs are shifted by +1 so 0 can be reserved for
    padding.
    """
    raw = load_json_or_jsonl(path)
    semantic_ids: dict[str, list[int]] = {}

    def normalize_tokens(value: Any) -> list[int]:
        if isinstance(value, dict):
            tokens = value.get("tokens", value.get("semantic_id", value.get("sid")))
        else:
            tokens = value
        if not isinstance(tokens, list):
            raise ValueError(f"Semantic ID value must be a list, got: {type(tokens)}")
        int_tokens = [int(x) for x in tokens]
        if offset_by_one:
            int_tokens = [x + 1 for x in int_tokens]
        return int_tokens

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
