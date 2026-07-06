from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal
import polars as pl

import numpy as np
import pandas as pd

ItemKey = Literal["parent_asin", "asin"]
_JSONL_CHUNK_SIZE = 500_000
_INTERACTION_COLUMNS = [
    "raw_user_id",
    "raw_item_id",
    "timestamp",
    "rating",
    "verified_purchase",
    "order_idx",
]


@contextmanager
def open_text(path: str | Path) -> Iterator[Any]:
    """Open plain-text or gzip-compressed files in text mode."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield f


def _suffixes(path: str | Path) -> str:
    return "".join(Path(path).suffixes).lower()


def _is_jsonl(path: str | Path) -> bool:
    suffixes = _suffixes(path)
    return suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz")


def _is_parquet(path: str | Path) -> bool:
    return _suffixes(path).endswith(".parquet")


def load_json_or_jsonl(path: str | Path) -> Any:
    """Load strict JSON or strict JSONL.

    JSONL returns a list of record dictionaries. JSON returns the parsed JSON
    document. There is no legacy Python-literal compatibility branch.
    """
    path = Path(path)
   
    chunk_size = _JSONL_CHUNK_SIZE
    chunks = []

    with open_text(path) as f:
        buffer = []

        for line in f:
            buffer.append(json.loads(line))

            if len(buffer) >= chunk_size:
                chunks.append(pd.DataFrame(buffer))
                buffer = []

        if buffer:
            chunks.append(pd.DataFrame(buffer))

    if not chunks:
        return []
    df = pd.concat(chunks, ignore_index=True)
    return df



def _first_series(frame: pd.DataFrame, *columns: str) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    out = frame[present[0]].copy()
    for column in present[1:]:
        out = out.where(out.notna(), frame[column])
    return out


def _coerce_optional_bool_series(series: pd.Series) -> pd.Series:
    """Normalize optional boolean fields without treating the string 'False' as true."""
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
    if value is None or value is pd.NA:
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


def normalize_review_record(
    record: dict[str, Any],
    *,
    item_key: ItemKey = "asin",
    use_verified_only: bool = False,
    keep_raw: bool = False,
    order_idx: int = 0,
) -> dict[str, Any] | None:
    """Normalize one parsed row for tests/edge-callers.

    The production path is normalize_interaction_frame/load_interactions_frame;
    this helper intentionally returns a plain dict instead of an InteractionRecord
    object so all downstream processing can remain DataFrame based.
    """
    verified_purchase = _coerce_optional_bool(record.get("verified_purchase", record.get("verified")))
    if use_verified_only and verified_purchase is not True:
        return None

    raw_user_id = record.get("user_id", record.get("reviewerID", record.get("reviewer_id")))
    raw_item_id = record.get(item_key) or record.get("parent_asin") or record.get("asin")
    if raw_user_id is None or raw_item_id is None:
        return None

    timestamp = record.get("timestamp", record.get("sort_timestamp", record.get("unixReviewTime", 0)))
    rating = record.get("rating", record.get("overall", 0.0))
    rating_float = pd.to_numeric(pd.Series([rating]), errors="coerce").fillna(0.0).iloc[0]
    out: dict[str, Any] = {
        "raw_user_id": str(raw_user_id),
        "raw_item_id": str(raw_item_id),
        "timestamp": 0 if timestamp is None else int(timestamp),
        "rating": float(rating_float),
        "verified_purchase": verified_purchase,
        "order_idx": int(order_idx),
    }
    if keep_raw:
        out["raw"] = dict(record)
    return out


def normalize_interaction_frame(
    frame: pd.DataFrame,
    *,
    item_key: ItemKey = "asin",
    order_offset: int = 0,
    use_verified_only: bool = False,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize an interaction DataFrame to the canonical vectorized schema.

    Expected parquet columns are: rating, asin, user_id, timestamp,
    verified_purchase.  For backward compatibility this also accepts Amazon
    JSONL aliases such as overall, parent_asin, reviewerID and unixReviewTime.
    """
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

    order_idx = np.arange(order_offset, order_offset + len(frame), dtype=np.int64)
    out = pd.DataFrame(
        {
            "raw_user_id": raw_user.loc[mask].astype("string").astype(str).to_numpy(),
            "raw_item_id": raw_item.loc[mask].astype("string").astype(str).to_numpy(),
            "timestamp": timestamp.loc[mask].to_numpy(dtype=np.int64),
            "rating": rating.loc[mask].to_numpy(dtype=float),
            "verified_purchase": verified.loc[mask].to_numpy(dtype=object),
            "order_idx": order_idx[mask.to_numpy()],
        }
    )
    if keep_raw:
        out["raw"] = frame.loc[mask].to_dict(orient="records")
    return out


def load_interactions_frame(
    reviews_path: str | Path,
    *,
    item_key: ItemKey = "asin",
    use_verified_only: bool = False,
    keep_raw: bool = False,
    allowed_items: set[str] | None = None,
) -> pd.DataFrame:
    """Load interactions from parquet, returning a normalized DataFrame.

    The parquet path is the primary fast path. Strict JSONL/JSON are retained as
    compatibility inputs and are normalized through the same DataFrame code.
    """
    path = Path(reviews_path)
    if _is_parquet(path):
        raw = pd.read_parquet(path)
        return normalize_interaction_frame(
            raw,
            item_key=item_key,
            use_verified_only=use_verified_only,
            keep_raw=keep_raw,
            allowed_items=allowed_items,
        )

    raw = load_json_or_jsonl(path)

    return normalize_interaction_frame(
        raw,
        item_key=item_key,
        use_verified_only=use_verified_only,
        keep_raw=keep_raw,
        allowed_items=allowed_items,
    )



def sort_interactions_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Sort interactions by first-seen user order, timestamp, and source order."""
    if data.empty:
        return data.copy()
    missing = [column for column in ("raw_user_id", "timestamp") if column not in data.columns]
    if missing:
        raise KeyError(f"sort_interactions_frame missing columns: {missing}")
    out = data.copy()
    out["__user_order"] = out["raw_user_id"].factorize(sort=False)[0]
    pl_frame = pl.from_pandas(out)
    pl_frame = pl_frame.sort(["__user_order", "timestamp"])
    out = pl_frame.to_pandas()
    return out.drop(columns=["__user_order"]).reset_index(drop=True)


def drop_consecutive_repeats(
    data: pd.DataFrame,
    user_id: str = "raw_user_id",
    item_id: str = "raw_item_id",
    timestamp: str = "timestamp",
) -> pd.DataFrame:
    """Remove immediate repeated items in each user's chronological sequence."""
    missing = [column for column in (user_id, item_id, timestamp) if column not in data.columns]
    if missing:
        raise KeyError(f"drop_consecutive_repeats missing columns: {missing}")

    data_sorted = sort_interactions_frame(data) if user_id == "raw_user_id" else data.sort_values([user_id, timestamp], kind="stable")
    shifted_col = "__flowrec_shifted_item__"
    data_sorted[shifted_col] = data_sorted.groupby(user_id, sort=False)[item_id].shift(1)
    return (
        data_sorted.loc[data_sorted[item_id] != data_sorted[shifted_col]]
        .drop(columns=[shifted_col])
        .reset_index(drop=True)
    )


def apply_k_core_filter_frame(
    interactions: pd.DataFrame,
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 0,
    drop_consecutive_repeats: bool = False,
) -> pd.DataFrame:
    """Iterative user/item k-core filtering over a DataFrame.

    This replaces the previous dict[str, list[InteractionRecord]] path. The only
    loop left is the k-core fixed-point iteration itself; each iteration uses
    vectorized group counts and boolean masks.
    """
    if min_user_interactions < 1:
        raise ValueError("min_user_interactions must be >= 1")
    if min_item_interactions < 0:
        raise ValueError("min_item_interactions must be >= 0")

    frame = interactions.copy()
    if frame.empty:
        return frame.reset_index(drop=True)
    if drop_consecutive_repeats:
        frame = globals()["drop_consecutive_repeats"](frame)

    while True:
        before_users = int(frame["raw_user_id"].nunique())
        before_interactions = int(len(frame))

        user_counts = frame.groupby("raw_user_id", sort=False, observed=True).size()
        keep_users = user_counts.index[user_counts >= int(min_user_interactions)]
        frame = frame.loc[frame["raw_user_id"].isin(keep_users)]

        if min_item_interactions > 0 and not frame.empty:
            item_counts = frame.groupby("raw_item_id", sort=False, observed=True).size()
            keep_items = item_counts.index[item_counts >= int(min_item_interactions)]
            frame = frame.loc[frame["raw_item_id"].isin(keep_items)]

            if drop_consecutive_repeats and not frame.empty:
                frame = globals()["drop_consecutive_repeats"](frame)

            if not frame.empty:
                user_counts = frame.groupby("raw_user_id", sort=False, observed=True).size()
                keep_users = user_counts.index[user_counts >= int(min_user_interactions)]
                frame = frame.loc[frame["raw_user_id"].isin(keep_users)]

        after_users = int(frame["raw_user_id"].nunique()) if not frame.empty else 0
        after_interactions = int(len(frame))
        if (after_users, after_interactions) == (before_users, before_interactions):
            break
        if after_interactions == 0:
            break

    return sort_interactions_frame(frame) if not frame.empty else frame.reset_index(drop=True)


def load_metadata_frame(
    metadata: str | Path | pd.DataFrame | None,
    *,
    item_key: ItemKey = "parent_asin",
) -> pd.DataFrame:
    """Load item metadata as a DataFrame with a normalized raw_item_id column.

    Expected metadata columns are: main_category, title, description, price,
    categories, parent_asin.  Extra columns are preserved.
    """
    if metadata is None:
        return pd.DataFrame()
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    else:
        path = Path(metadata)
        if _is_parquet(path):
            frame = pd.read_parquet(path, engine="pyarrow")
        else:
            frame = load_json_or_jsonl(path)

    if frame.empty:
        return pd.DataFrame(columns=["raw_item_id"])
    raw_item = _first_series(frame, item_key, "parent_asin", "asin", "raw_item_id", "item_id")
    out = frame.loc[raw_item.notna()].copy()
    out["raw_item_id"] = raw_item.loc[raw_item.notna()].astype("string").astype(str).to_numpy()
    out = out.drop_duplicates("raw_item_id", keep="first").reset_index(drop=True)
    return out


def metadata_frame_to_dict(metadata: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if metadata.empty or "raw_item_id" not in metadata.columns:
        return {}
    return {
        str(row["raw_item_id"]): {k: v for k, v in row.items() if k != "raw_item_id"}
        for row in metadata.to_dict(orient="records")
    }


def metadata_to_text(metadata: dict[str, Any] | pd.Series) -> str:
    """Build a robust text representation from Amazon metadata."""
    if isinstance(metadata, pd.Series):
        metadata = metadata.to_dict()
    parts: list[str] = []

    for key in ["title", "main_category", "main_cat", "store", "brand"]:
        value = metadata.get(key)
        if value is not None and value != "":
            parts.append(str(value))

    for key in ["features", "feature", "description", "categories", "category"]:
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value if x)
        elif value is not None and value != "":
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


def collect_item_ids(interactions: pd.DataFrame) -> set[str]:
    if interactions.empty:
        return set()
    return set(interactions["raw_item_id"].astype(str).unique())


def count_interactions(interactions: pd.DataFrame) -> int:
    return int(len(interactions))
