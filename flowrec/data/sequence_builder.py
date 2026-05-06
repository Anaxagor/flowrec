from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np

from .datasets import InteractionRecord, SequenceExample
from .preprocessors import sort_user_events

SplitStrategy = Literal["leave_one_out", "global_time"]
GlobalHistoryPolicy = Literal["rolling", "window_start"]
GlobalEvalTargetMode = Literal["all", "last_per_user"]


@dataclass(frozen=True, slots=True)
class SequenceSplitConfig:
    split_strategy: SplitStrategy = "leave_one_out"

    min_history_len: int = 1
    train_all_prefixes: bool = True

    # Global-time split options.
    global_val_timestamp: int | None = None
    global_test_timestamp: int | None = None
    global_val_ratio: float = 0.1
    global_test_ratio: float = 0.1
    global_history_policy: GlobalHistoryPolicy = "rolling"
    global_eval_target_mode: GlobalEvalTargetMode = "all"


IdGetter = Callable[[str], int]


@dataclass(slots=True)
class SequenceSplits:
    train: list[SequenceExample]
    val: list[SequenceExample]
    test: list[SequenceExample]
    val_cutoff: int | None = None
    test_cutoff: int | None = None



def build_sequence_splits(
    user_events: dict[str, list[InteractionRecord]],
    *,
    get_user_id: IdGetter,
    get_item_id: IdGetter,
    cfg: SequenceSplitConfig,
) -> SequenceSplits:
    """Build train/validation/test examples from normalized interactions."""
    if cfg.min_history_len < 1:
        raise ValueError("min_history_len must be >= 1")

    user_events = sort_user_events(user_events)

    if cfg.split_strategy == "leave_one_out":
        train, val, test = build_leave_one_out_examples(
            user_events,
            get_user_id=get_user_id,
            get_item_id=get_item_id,
            cfg=cfg,
        )
        return SequenceSplits(train=train, val=val, test=test)

    if cfg.split_strategy == "global_time":
        val_cutoff, test_cutoff = resolve_global_time_cutoffs(user_events, cfg=cfg)
        train, val, test = build_global_time_examples(
            user_events,
            get_user_id=get_user_id,
            get_item_id=get_item_id,
            cfg=cfg,
            val_cutoff=val_cutoff,
            test_cutoff=test_cutoff,
        )
        return SequenceSplits(
            train=train,
            val=val,
            test=test,
            val_cutoff=val_cutoff,
            test_cutoff=test_cutoff,
        )

    raise ValueError(f"Unknown split_strategy: {cfg.split_strategy}")



def build_leave_one_out_examples(
    user_events: dict[str, list[InteractionRecord]],
    *,
    get_user_id: IdGetter,
    get_item_id: IdGetter,
    cfg: SequenceSplitConfig,
) -> tuple[list[SequenceExample], list[SequenceExample], list[SequenceExample]]:
    train_examples: list[SequenceExample] = []
    val_examples: list[SequenceExample] = []
    test_examples: list[SequenceExample] = []

    for raw_user_id, events in user_events.items():
        if len(events) < cfg.min_history_len + 2:
            continue

        user_id = get_user_id(raw_user_id)
        raw_items = [event.raw_item_id for event in events]
        item_ids = [get_item_id(raw_item_id) for raw_item_id in raw_items]
        n = len(events)

        if cfg.train_all_prefixes:
            train_target_indices = list(range(cfg.min_history_len, n - 2))
        else:
            last_train_idx = n - 3
            train_target_indices = [last_train_idx] if last_train_idx >= cfg.min_history_len else []

        for target_idx in train_target_indices:
            train_examples.append(
                make_sequence_example(
                    raw_user_id=raw_user_id,
                    user_id=user_id,
                    raw_items=raw_items,
                    item_ids=item_ids,
                    events=events,
                    target_idx=target_idx,
                    split="train",
                )
            )

        val_idx = n - 2
        test_idx = n - 1
        if val_idx >= cfg.min_history_len:
            val_examples.append(
                make_sequence_example(
                    raw_user_id=raw_user_id,
                    user_id=user_id,
                    raw_items=raw_items,
                    item_ids=item_ids,
                    events=events,
                    target_idx=val_idx,
                    split="val",
                )
            )
        if test_idx >= cfg.min_history_len:
            test_examples.append(
                make_sequence_example(
                    raw_user_id=raw_user_id,
                    user_id=user_id,
                    raw_items=raw_items,
                    item_ids=item_ids,
                    events=events,
                    target_idx=test_idx,
                    split="test",
                )
            )

    return train_examples, val_examples, test_examples



def build_global_time_examples(
    user_events: dict[str, list[InteractionRecord]],
    *,
    get_user_id: IdGetter,
    get_item_id: IdGetter,
    cfg: SequenceSplitConfig,
    val_cutoff: int,
    test_cutoff: int,
) -> tuple[list[SequenceExample], list[SequenceExample], list[SequenceExample]]:
    train_examples: list[SequenceExample] = []
    val_examples: list[SequenceExample] = []
    test_examples: list[SequenceExample] = []

    make_ex = make_sequence_example

    for raw_user_id, events in user_events.items():
        n = len(events)
        if n < cfg.min_history_len + 1:
            continue

        user_id = get_user_id(raw_user_id)

        raw_items = [event.raw_item_id for event in events]
        item_ids = [get_item_id(raw_item_id) for raw_item_id in raw_items]
        timestamps = np.asarray([int(event.timestamp) for event in events], dtype=np.int64)

        first_target = cfg.min_history_len

        val_start = max(
            first_target,
            int(np.searchsorted(timestamps, val_cutoff, side="left")),
        )
        test_start = max(
            val_start,
            int(np.searchsorted(timestamps, test_cutoff, side="left")),
        )

        if cfg.train_all_prefixes:
            train_range = range(first_target, val_start)
        else:
            train_range = range(val_start - 1, val_start) if val_start > first_target else range(0)

        for target_idx in train_range:
            train_examples.append(
                make_ex(
                    raw_user_id=raw_user_id,
                    user_id=user_id,
                    raw_items=raw_items,
                    item_ids=item_ids,
                    events=events,
                    target_idx=target_idx,
                    split="train",
                    history_end_idx=target_idx,
                )
            )

        val_indices = _select_eval_targets(
            start=val_start,
            end=test_start,
            mode=cfg.global_eval_target_mode,
        )

        if val_indices.size > 0:
            if cfg.global_history_policy == "window_start":
                val_history_end = int(np.searchsorted(timestamps, val_cutoff, side="left"))
                if val_history_end >= cfg.min_history_len:
                    for target_idx in val_indices:
                        val_examples.append(
                            make_ex(
                                raw_user_id=raw_user_id,
                                user_id=user_id,
                                raw_items=raw_items,
                                item_ids=item_ids,
                                events=events,
                                target_idx=int(target_idx),
                                split="val",
                                history_end_idx=val_history_end,
                            )
                        )
            else:
                for target_idx in val_indices:
                    val_examples.append(
                        make_ex(
                            raw_user_id=raw_user_id,
                            user_id=user_id,
                            raw_items=raw_items,
                            item_ids=item_ids,
                            events=events,
                            target_idx=int(target_idx),
                            split="val",
                            history_end_idx=int(target_idx),
                        )
                    )

        test_indices = _select_eval_targets(
            start=test_start,
            end=n,
            mode=cfg.global_eval_target_mode,
        )

        if test_indices.size > 0:
            if cfg.global_history_policy == "window_start":
                test_history_end = int(np.searchsorted(timestamps, test_cutoff, side="left"))
                if test_history_end >= cfg.min_history_len:
                    for target_idx in test_indices:
                        test_examples.append(
                            make_ex(
                                raw_user_id=raw_user_id,
                                user_id=user_id,
                                raw_items=raw_items,
                                item_ids=item_ids,
                                events=events,
                                target_idx=int(target_idx),
                                split="test",
                                history_end_idx=test_history_end,
                            )
                        )
            else:
                for target_idx in test_indices:
                    test_examples.append(
                        make_ex(
                            raw_user_id=raw_user_id,
                            user_id=user_id,
                            raw_items=raw_items,
                            item_ids=item_ids,
                            events=events,
                            target_idx=int(target_idx),
                            split="test",
                            history_end_idx=int(target_idx),
                        )
                    )

    return train_examples, val_examples, test_examples
def _select_eval_targets(
    *,
    start: int,
    end: int,
    mode: GlobalEvalTargetMode,
) -> np.ndarray:
    if end <= start:
        return np.empty(0, dtype=np.int64)

    if mode == "all":
        return np.arange(start, end, dtype=np.int64)

    if mode == "last_per_user":
        return np.asarray([end - 1], dtype=np.int64)

    raise ValueError(f"Unknown global_eval_target_mode: {mode}")


def resolve_global_time_cutoffs(
    user_events: dict[str, list[InteractionRecord]],
    *,
    cfg: SequenceSplitConfig,
) -> tuple[int, int]:
    """Resolve validation/test cutoffs for global chronological splitting."""
    if cfg.global_val_timestamp is not None or cfg.global_test_timestamp is not None:
        if cfg.global_val_timestamp is None or cfg.global_test_timestamp is None:
            raise ValueError(
                "Both global_val_timestamp and global_test_timestamp are required "
                "when using explicit global time cutoffs."
            )
        val_cutoff = int(cfg.global_val_timestamp)
        test_cutoff = int(cfg.global_test_timestamp)
        if val_cutoff >= test_cutoff:
            raise ValueError("global_val_timestamp must be < global_test_timestamp")
        return val_cutoff, test_cutoff

    if cfg.global_val_ratio <= 0 or cfg.global_test_ratio <= 0:
        raise ValueError("global_val_ratio and global_test_ratio must be positive")
    if cfg.global_val_ratio + cfg.global_test_ratio >= 1:
        raise ValueError("global_val_ratio + global_test_ratio must be < 1")

    timestamps = sorted(event.timestamp for events in user_events.values() for event in events)
    if not timestamps:
        raise ValueError("Cannot infer cutoffs from an empty interaction set")

    n = len(timestamps)
    val_pos = min(max(int((1.0 - cfg.global_val_ratio - cfg.global_test_ratio) * n), 0), n - 1)
    test_pos = min(max(int((1.0 - cfg.global_test_ratio) * n), 0), n - 1)

    val_cutoff = int(timestamps[val_pos])
    test_cutoff = int(timestamps[test_pos])

    if val_cutoff >= test_cutoff:
        raise ValueError(
            "Inferred global-time cutoffs are not strictly ordered. This usually "
            "means many interactions share the same timestamp. Provide explicit "
            "global_val_timestamp and global_test_timestamp."
        )

    return val_cutoff, test_cutoff



def select_eval_indices(indices: list[int], *, mode: GlobalEvalTargetMode) -> list[int]:
    if mode == "all":
        return indices
    if mode == "last_per_user":
        return [indices[-1]] if indices else []
    raise ValueError(f"Unknown global_eval_target_mode: {mode}")



def global_history_end_idx(
    *,
    timestamps: list[int],
    target_idx: int,
    window_start: int,
    policy: GlobalHistoryPolicy,
) -> int:
    if policy == "rolling":
        return target_idx
    if policy == "window_start":
        return bisect_left(timestamps, window_start)
    raise ValueError(f"Unknown global_history_policy: {policy}")



def make_sequence_example(
    *,
    raw_user_id: str,
    user_id: int,
    raw_items: list[str],
    item_ids: list[int],
    events: list[InteractionRecord],
    target_idx: int,
    split: Literal["train", "val", "test"],
    history_end_idx: int | None = None,
) -> SequenceExample:
    if history_end_idx is None:
        history_end_idx = target_idx
    if history_end_idx > target_idx:
        raise ValueError("history_end_idx cannot exceed target_idx; this would leak future data.")

    target_event = events[target_idx]
    return SequenceExample(
        raw_user_id=raw_user_id,
        user_id=user_id,
        history_raw_items=raw_items[:history_end_idx],
        history_item_ids=item_ids[:history_end_idx],
        target_raw_item=raw_items[target_idx],
        target_item_id=item_ids[target_idx],
        target_rating=float(target_event.rating),
        target_timestamp=int(target_event.timestamp),
        split=split,
        target_position=target_idx,
    )
