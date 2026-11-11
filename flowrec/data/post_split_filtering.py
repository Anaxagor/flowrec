from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .sequence_builder import SequenceSplits, is_dense_example_frame

ColdItemTrainSource = Literal["observed", "targets"]


def remap_interactions_to_catalog(
    interactions: pd.DataFrame, item2id: "dict[str, int]"
) -> pd.DataFrame:
    """Re-key an interactions frame onto the post-filter catalog.

    The cold-item filter compactly renumbers item ids, so every datamodule
    that keeps its interactions frame must re-key it or downstream consumers
    (e.g. positive-history stores) silently mix id spaces. Removed items get
    the pad id 0 — rows keep their place, so position/length-based train
    boundaries stay exact — and the pre-filter ids survive in
    ``item_id_prefilter`` for consumers bound to the unfiltered factorization
    (e.g. teacher models trained without the filter).
    """
    remapped = interactions["raw_item_id"].astype(str).map(item2id)
    frame = interactions.copy()
    frame["item_id_prefilter"] = frame["item_id"]
    frame["item_id"] = remapped.fillna(0).astype("int64")
    return frame


@dataclass(frozen=True, slots=True)
class PostSplitColdItemFilterStats:
    """Diagnostics for train-only catalog filtering after split construction."""

    enabled: bool
    train_item_source: ColdItemTrainSource
    min_history_len: int

    num_items_before: int
    num_items_after: int
    num_items_removed: int

    num_train_examples_before: int
    num_val_examples_before: int
    num_test_examples_before: int
    num_train_examples_after: int
    num_val_examples_after: int
    num_test_examples_after: int

    dropped_train_examples: int
    dropped_val_examples: int
    dropped_test_examples: int

    removed_train_history_items: int
    removed_val_history_items: int
    removed_test_history_items: int

    def to_dict(self) -> dict[str, int | bool | str]:
        return asdict(self)


@dataclass(slots=True)
class PostSplitColdItemFilterResult:
    """Filtered split DataFrames plus a compact train-only item catalog."""

    splits: SequenceSplits
    item2id: dict[str, int]
    id2item: list[str]
    train_raw_items: set[str]
    removed_raw_items: set[str]
    stats: PostSplitColdItemFilterStats


@dataclass(slots=True)
class _FilterCounters:
    dropped: int = 0
    removed_history_items: int = 0


def filter_splits_to_train_item_catalog(
    splits: SequenceSplits,
    *,
    old_id2item: list[str],
    train_item_source: ColdItemTrainSource = "observed",
    min_history_len: int = 1,
    pad_raw_item: str = "<PAD>",
) -> PostSplitColdItemFilterResult:
    """Remove post-split cold items and rebuild a compact train-only catalog.

    This is now a DataFrame transformation: train catalog collection, history
    filtering, target filtering, and ID remapping operate on split DataFrames
    instead of list[SequenceExample].
    """
    if min_history_len < 0:
        raise ValueError("min_history_len must be >= 0")
    if train_item_source not in {"observed", "targets"}:
        raise ValueError("train_item_source must be 'observed' or 'targets'")

    dense_train = is_dense_example_frame(splits.train)
    if dense_train and train_item_source != "observed":
        raise ValueError(
            "dense_causal train rows require cold_item_train_source='observed': "
            "with 'targets' some window inputs could fall outside the catalog, "
            "and dropping them would break the input/label alignment."
        )

    train_raw_items = _collect_train_raw_items(splits.train, source=train_item_source)
    old_raw_items = [str(item) for item in old_id2item[1:]]
    ordered_train_items = [item for item in old_raw_items if item in train_raw_items]
    missing_from_old_order = sorted(train_raw_items - set(ordered_train_items))
    ordered_train_items.extend(missing_from_old_order)

    id2item = [pad_raw_item] + ordered_train_items
    item2id = {raw_item: idx for idx, raw_item in enumerate(ordered_train_items, start=1)}

    if dense_train:
        # The dense catalog is collected from the dense rows themselves, so
        # every input and label item is present in item2id: remapping is a pure
        # renumbering that never drops events or breaks input/label alignment.
        train = _renumber_dense_examples(splits.train, item2id=item2id)
        train_counters = _FilterCounters()
    else:
        train, train_counters = _filter_and_remap_examples(
            splits.train,
            item2id=item2id,
            min_history_len=min_history_len,
        )
    val, val_counters = _filter_and_remap_examples(
        splits.val,
        item2id=item2id,
        min_history_len=min_history_len,
    )
    test, test_counters = _filter_and_remap_examples(
        splits.test,
        item2id=item2id,
        min_history_len=min_history_len,
    )

    filtered_splits = SequenceSplits(
        train=train,
        val=val,
        test=test,
    )

    old_catalog = set(old_raw_items)
    removed_raw_items = old_catalog - set(item2id)

    stats = PostSplitColdItemFilterStats(
        enabled=True,
        train_item_source=train_item_source,
        min_history_len=min_history_len,
        num_items_before=len(old_raw_items),
        num_items_after=len(ordered_train_items),
        num_items_removed=len(removed_raw_items),
        num_train_examples_before=len(splits.train),
        num_val_examples_before=len(splits.val),
        num_test_examples_before=len(splits.test),
        num_train_examples_after=len(train),
        num_val_examples_after=len(val),
        num_test_examples_after=len(test),
        dropped_train_examples=train_counters.dropped,
        dropped_val_examples=val_counters.dropped,
        dropped_test_examples=test_counters.dropped,
        removed_train_history_items=train_counters.removed_history_items,
        removed_val_history_items=val_counters.removed_history_items,
        removed_test_history_items=test_counters.removed_history_items,
    )

    return PostSplitColdItemFilterResult(
        splits=filtered_splits,
        item2id=item2id,
        id2item=id2item,
        train_raw_items=set(item2id),
        removed_raw_items=removed_raw_items,
        stats=stats,
    )


def disabled_post_split_filter_stats(
    *,
    splits: SequenceSplits,
    num_items: int,
    train_item_source: ColdItemTrainSource = "observed",
    min_history_len: int = 1,
) -> PostSplitColdItemFilterStats:
    """Return zero-effect diagnostics when filtering is disabled."""
    return PostSplitColdItemFilterStats(
        enabled=False,
        train_item_source=train_item_source,
        min_history_len=min_history_len,
        num_items_before=num_items,
        num_items_after=num_items,
        num_items_removed=0,
        num_train_examples_before=len(splits.train),
        num_val_examples_before=len(splits.val),
        num_test_examples_before=len(splits.test),
        num_train_examples_after=len(splits.train),
        num_val_examples_after=len(splits.val),
        num_test_examples_after=len(splits.test),
        dropped_train_examples=0,
        dropped_val_examples=0,
        dropped_test_examples=0,
        removed_train_history_items=0,
        removed_val_history_items=0,
        removed_test_history_items=0,
    )


def _collect_train_raw_items(train_examples: pd.DataFrame, *, source: ColdItemTrainSource) -> set[str]:
    if train_examples.empty:
        return set()
    items = set(train_examples["target_raw_item"].astype(str).unique())
    if source == "observed" and "history_raw_items" in train_examples.columns:
        exploded = train_examples["history_raw_items"].explode().dropna()
        if not exploded.empty:
            items.update(exploded.astype(str).unique())
    if source == "observed" and "label_raw_items" in train_examples.columns:
        exploded = train_examples["label_raw_items"].explode().dropna()
        if not exploded.empty:
            items.update(exploded.astype(str).unique())
    return items


def _renumber_dense_examples(examples: pd.DataFrame, *, item2id: dict[str, int]) -> pd.DataFrame:
    """Renumber dense train rows to a new catalog without dropping anything.

    Requires every item in the rows to be present in item2id (guaranteed when
    the catalog was collected from these rows with source='observed').
    """
    if examples.empty:
        return examples.copy()

    out = examples.reset_index(drop=True).copy()

    def _map_list(raw_items: list) -> list[int]:
        return [item2id[str(raw_item)] for raw_item in raw_items]

    out["history_item_ids"] = [_map_list(row) for row in out["history_raw_items"]]
    out["label_item_ids"] = [_map_list(row) for row in out["label_raw_items"]]
    out["target_item_id"] = [item2id[str(raw_item)] for raw_item in out["target_raw_item"]]
    return out


def _filter_and_remap_examples(
    examples: pd.DataFrame,
    *,
    item2id: dict[str, int],
    min_history_len: int,
) -> tuple[pd.DataFrame, _FilterCounters]:
    counters = _FilterCounters()
    if examples.empty:
        return examples.copy(), counters

    out = examples.reset_index(drop=True).copy()
    ex_idx = np.arange(len(out), dtype=np.int64)

    target_ids = out["target_raw_item"].astype(str).map(item2id)
    target_allowed = target_ids.notna()

    hist = pd.DataFrame(
        {
            "__ex_idx": ex_idx,
            "raw_item": out["history_raw_items"].tolist(),
            "rating": _history_feature_lists(out, "history_ratings", default=0.0),
            "verified": _history_feature_lists(out, "history_verified_purchase", default=None),
        }
    ).explode(["raw_item", "rating", "verified"], ignore_index=True)

    if hist.empty:
        keep = target_allowed.to_numpy() & (np.zeros(len(out), dtype=np.int64) >= int(min_history_len))
        counters.dropped = int(len(out) - int(keep.sum()))
        return out.loc[keep].copy().reset_index(drop=True), counters

    hist = hist.dropna(subset=["raw_item"])
    if hist.empty:
        keep = target_allowed.to_numpy() & (np.zeros(len(out), dtype=np.int64) >= int(min_history_len))
        counters.dropped = int(len(out) - int(keep.sum()))
        return out.loc[keep].copy().reset_index(drop=True), counters

    hist["raw_item"] = hist["raw_item"].astype(str)
    hist["new_id"] = hist["raw_item"].map(item2id)
    allowed = hist["new_id"].notna()
    counters.removed_history_items = int((~allowed).sum())

    kept_hist = hist.loc[allowed, ["__ex_idx", "raw_item", "new_id", "rating", "verified"]].copy()
    if kept_hist.empty:
        history_lengths = pd.Series(0, index=ex_idx, dtype=np.int64)
        grouped = pd.DataFrame(index=pd.Index([], name="__ex_idx"))
    else:
        kept_hist["new_id"] = kept_hist["new_id"].astype("int64")
        kept_hist["rating"] = pd.to_numeric(kept_hist["rating"], errors="coerce").fillna(0.0).astype(float)
        grouped = kept_hist.groupby("__ex_idx", sort=False).agg(
            history_raw_items=("raw_item", list),
            history_item_ids=("new_id", lambda s: [int(x) for x in s]),
            history_ratings=("rating", lambda s: [float(x) for x in s]),
            history_verified_purchase=("verified", list),
        )
        history_lengths = grouped["history_raw_items"].str.len().reindex(ex_idx, fill_value=0)

    keep = target_allowed.to_numpy() & (history_lengths.to_numpy(dtype=np.int64) >= int(min_history_len))
    counters.dropped = int(len(out) - int(keep.sum()))
    kept = out.loc[keep].copy()
    kept_idx = ex_idx[keep]
    if not kept.empty:
        mapped_target_ids = target_ids.loc[keep].astype("int64").to_numpy()
        kept["target_item_id"] = mapped_target_ids
        for column in ["history_raw_items", "history_item_ids", "history_ratings", "history_verified_purchase"]:
            series = grouped[column].reindex(kept_idx) if column in grouped.columns else pd.Series(index=kept_idx, dtype=object)
            kept[column] = [value if isinstance(value, list) else [] for value in series]
    return kept.reset_index(drop=True), counters


def _history_feature_lists(frame: pd.DataFrame, column: str, *, default) -> list[list]:
    """Return history-aligned feature lists from a split DataFrame column."""
    if column not in frame.columns:
        return [[default for _ in history] for history in frame["history_raw_items"]]

    lengths = frame["history_raw_items"].str.len().fillna(0).astype(int)
    values = frame[column].tolist()
    out: list[list] = []
    for value, expected_len in zip(values, lengths.tolist(), strict=False):
        if not isinstance(value, list) or len(value) == 0:
            out.append([default for _ in range(expected_len)])
        elif len(value) != expected_len:
            raise ValueError(
                f"History feature length does not match history item length for {column}: "
                f"feature_len={len(value)}, expected_len={expected_len}."
            )
        else:
            out.append(list(value))
    return out
