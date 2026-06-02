from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

from .datasets import SequenceExample
from .sequence_builder import SequenceSplits

ColdItemTrainSource = Literal["observed", "targets"]


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
    """Filtered splits plus a compact train-only item catalog."""

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

    Cold items are items that were assigned IDs while building splits but are not
    present in the chosen training item universe.  By default that universe is
    the set of all items observed in training examples, including both training
    targets and training histories.  Evaluation examples whose target item is not
    in that universe are dropped; history entries outside the universe are
    removed from every split; and item IDs are remapped to a compact catalog.
    """
    if min_history_len < 0:
        raise ValueError("min_history_len must be >= 0")
    if train_item_source not in {"observed", "targets"}:
        raise ValueError("train_item_source must be 'observed' or 'targets'")

    train_raw_items = _collect_train_raw_items(
        splits.train,
        source=train_item_source,
    )

    # Preserve the original deterministic item order while dropping cold items.
    old_raw_items = [str(item) for item in old_id2item[1:]]
    ordered_train_items = [item for item in old_raw_items if item in train_raw_items]
    missing_from_old_order = sorted(train_raw_items - set(ordered_train_items))
    ordered_train_items.extend(missing_from_old_order)

    id2item = [pad_raw_item] + ordered_train_items
    item2id = {raw_item: idx for idx, raw_item in enumerate(ordered_train_items, start=1)}

    train, train_counters = _filter_and_remap_examples(
        splits.train,
        item2id=item2id,
        allowed_raw_items=train_raw_items,
        min_history_len=min_history_len,
    )
    val, val_counters = _filter_and_remap_examples(
        splits.val,
        item2id=item2id,
        allowed_raw_items=train_raw_items,
        min_history_len=min_history_len,
    )
    test, test_counters = _filter_and_remap_examples(
        splits.test,
        item2id=item2id,
        allowed_raw_items=train_raw_items,
        min_history_len=min_history_len,
    )

    filtered_splits = SequenceSplits(
        train=train,
        val=val,
        test=test,
        val_cutoff=splits.val_cutoff,
        test_cutoff=splits.test_cutoff,
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


def _collect_train_raw_items(
    train_examples: list[SequenceExample],
    *,
    source: ColdItemTrainSource,
) -> set[str]:
    items = {str(ex.target_raw_item) for ex in train_examples}
    if source == "observed":
        for ex in train_examples:
            items.update(str(raw_item) for raw_item in ex.history_raw_items)
    return items


def _filter_and_remap_examples(
    examples: list[SequenceExample],
    *,
    item2id: dict[str, int],
    allowed_raw_items: set[str],
    min_history_len: int,
) -> tuple[list[SequenceExample], _FilterCounters]:
    filtered: list[SequenceExample] = []
    counters = _FilterCounters()

    for ex in examples:
        target_raw_item = str(ex.target_raw_item)
        target_item_id = item2id.get(target_raw_item)
        if target_item_id is None:
            counters.dropped += 1
            counters.removed_history_items += _count_disallowed_history_items(
                ex.history_raw_items,
                allowed_raw_items,
            )
            continue

        history_raw_items: list[str] = []
        history_item_ids: list[int] = []
        history_ratings: list[float] = []
        history_verified_purchase: list[bool | None] = []

        old_history_ratings = _aligned_history_feature(
            ex.history_ratings,
            len(ex.history_raw_items),
            default=0.0,
        )
        old_history_verified = _aligned_history_feature(
            ex.history_verified_purchase,
            len(ex.history_raw_items),
            default=None,
        )

        removed = 0
        for idx, raw_item in enumerate(ex.history_raw_items):
            raw_key = str(raw_item)
            new_id = item2id.get(raw_key)
            if new_id is None:
                removed += 1
                continue
            history_raw_items.append(raw_key)
            history_item_ids.append(new_id)
            history_ratings.append(float(old_history_ratings[idx]))
            history_verified_purchase.append(old_history_verified[idx])

        counters.removed_history_items += removed
        if len(history_raw_items) < min_history_len:
            counters.dropped += 1
            continue

        filtered.append(
            replace(
                ex,
                history_raw_items=history_raw_items,
                history_item_ids=history_item_ids,
                history_ratings=history_ratings,
                history_verified_purchase=history_verified_purchase,
                target_raw_item=target_raw_item,
                target_item_id=target_item_id,
            )
        )

    return filtered, counters


def _count_disallowed_history_items(
    history_raw_items: list[str],
    allowed_raw_items: set[str],
) -> int:
    return sum(1 for raw_item in history_raw_items if str(raw_item) not in allowed_raw_items)

def _aligned_history_feature(
    values: list,
    expected_len: int,
    *,
    default,
) -> list:
    """Return a history-aligned feature list with length expected_len.

    Older cached SequenceExample objects may not have newly-added history
    features.  Missing values are filled with a neutral default, while mismatched
    non-empty lists are treated as an error because silently misaligning ratings
    and item IDs would corrupt rewards.
    """
    if not values:
        return [default for _ in range(expected_len)]
    if len(values) != expected_len:
        raise ValueError(
            "History feature length does not match history item length: "
            f"feature_len={len(values)}, expected_len={expected_len}."
        )
    return list(values)
