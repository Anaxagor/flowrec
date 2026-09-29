"""Single-target training examples with exactly the dense window's context."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .sequence_builder import SequenceSplitConfig, SequenceSplits, _SEQUENCE_COLUMNS


def dense_prefix_signature(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Cache identity: changing contexts invalidates rewards even for equal targets."""
    if not config.get("train_prefix_from_dense", False):
        return None
    window = config.get("train_window_len") or config["max_history_len"]
    return {"version": 1, "context": "dense_window", "window_len": int(window),
            "window_stride": int(config.get("train_window_stride") or window),
            "target_policy": config["train_target_policy"],
            "last_n": (int(config["train_last_n"])
                       if config["train_target_policy"] == "last_n" else None)}


def dense_reference_config(config: SequenceSplitConfig) -> SequenceSplitConfig:
    """Build the base dense splits before changing training examples only."""
    if config.train_example_layout != "prefix_target":
        raise ValueError("train_prefix_from_dense requires prefix_target layout")
    if config.train_target_policy not in {"all", "last_n"} or not config.train_all_prefixes:
        raise ValueError("train_prefix_from_dense supports all or last_n targets")
    if (config.train_window_len or config.max_history_len) > config.max_history_len:
        raise ValueError("dense window exceeds the prefix collator's max_history_len")
    return replace(config, train_example_layout="dense_causal")


def expand_dense_training_split(
    splits: SequenceSplits, interactions: pd.DataFrame, *,
    last_n: int | None, catalog_items: int,
) -> dict[str, Any]:
    """Expand training after catalog filtering, retaining complete held-out rows."""
    reference = {name: example_digest(getattr(splits, name)) for name in ("val", "test")}
    dense_rows = len(splits.train)
    splits.train = prefixes_from_dense_windows(splits.train, interactions, last_n=last_n)
    actual = {name: example_digest(getattr(splits, name)) for name in ("val", "test")}
    if actual != reference:
        raise RuntimeError("Dense-to-prefix conversion changed validation/test examples")
    return {
        "reference_layout": "dense_causal", "prefix_layout": "prefix_target",
        "catalog_source": "full dense training windows, before target selection",
        "dense_train_rows": dense_rows, "prefix_train_rows": len(splits.train),
        "catalog_items": catalog_items,
        **{label: {"num_examples": len(getattr(splits, name)),
                   "dense_sha256": reference[name], "prefix_sha256": actual[name], "equal": True}
           for label, name in (("validation", "val"), ("test", "test"))},
    }


def example_digest(examples: pd.DataFrame) -> str:
    """Hash ordered complete examples, including histories, IDs and timestamps."""
    digest = hashlib.sha256()
    for row in examples[_SEQUENCE_COLUMNS].itertuples(index=False, name=None):
        digest.update(json.dumps(row, separators=(",", ":"), ensure_ascii=True,
                                 default=lambda value: value.item()).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prefixes_from_dense_windows(
    dense: pd.DataFrame, interactions: pd.DataFrame, *, last_n: int | None = None,
) -> pd.DataFrame:
    """Expand active dense labels, optionally retaining only each user's tail.

    Call AFTER dense catalog filtering. Interactions must be the contiguous,
    chronological, factorized frame used by the sequence builder. Select targets
    without reanchoring windows; history always begins at its original window.
    Keeping the tail here avoids expanding all historical targets on Yambda.
    """
    if last_n is not None and last_n < 1:
        raise ValueError("last_n must be positive")
    if dense.empty:
        return pd.DataFrame(columns=_SEQUENCE_COLUMNS)
    last_positions = dense.groupby("user_id", sort=False)["target_position"].max().to_dict()
    # Factorization gives contiguous users and zero-based chronological positions.
    starts = np.flatnonzero(interactions["position"].to_numpy() == 0)
    user_starts = dict(zip(interactions["user_id"].to_numpy()[starts], starts))
    timestamps = interactions["timestamp"].to_numpy()
    rows = []
    for window in dense.itertuples(index=False):
        first_position = int(window.target_position) - len(window.label_mask) + 1
        lower = (last_positions[window.user_id] - last_n + 1) if last_n is not None else 0
        if int(window.target_position) < lower:
            continue
        for offset, active in enumerate(window.label_mask):
            position = first_position + offset
            if not active or position < lower:
                continue
            event_index = user_starts[window.user_id] + position
            end = offset + 1
            rows.append({
                "raw_user_id": window.raw_user_id, "user_id": window.user_id,
                "history_raw_items": window.history_raw_items[:end],
                "history_item_ids": window.history_item_ids[:end],
                "history_ratings": window.history_ratings[:end],
                "history_verified_purchase": window.history_verified_purchase[:end],
                "target_raw_item": window.label_raw_items[offset],
                "target_item_id": window.label_item_ids[offset],
                "target_rating": (window.history_ratings[end] if end < len(window.history_ratings)
                                  else window.target_rating),
                "target_timestamp": int(timestamps[event_index]),
                "target_verified_purchase": (window.history_verified_purchase[end]
                                             if end < len(window.history_verified_purchase)
                                             else window.target_verified_purchase),
                "target_position": position, "split": "train",
            })
    return (pd.DataFrame(rows, columns=_SEQUENCE_COLUMNS)
            .sort_values(["user_id", "target_position"], kind="stable").reset_index(drop=True))
