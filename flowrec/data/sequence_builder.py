from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .preprocessors import sort_interactions_frame

SplitStrategy = Literal["leave_one_out"]
TrainTargetPolicy = Literal["all", "last", "last_n", "stride", "sampled", "hybrid"]
TrainExampleLayout = Literal["prefix_target", "dense_causal"]

_SEQUENCE_COLUMNS = [
    "raw_user_id",
    "user_id",
    "history_raw_items",
    "history_item_ids",
    "target_raw_item",
    "target_item_id",
    "target_rating",
    "target_timestamp",
    "target_verified_purchase",
    "history_ratings",
    "history_verified_purchase",
    "split",
    "target_position",
]

# Dense-causal train rows keep the prefix-target column names for the input
# side (history_*) and add aligned per-position labels. label_item_ids[i] is
# the item that follows history_item_ids[i] in the user's sequence; label_mask
# marks which positions contribute to the training loss.
_DENSE_SEQUENCE_COLUMNS = [
    "raw_user_id",
    "user_id",
    "history_raw_items",
    "history_item_ids",
    "label_raw_items",
    "label_item_ids",
    "label_mask",
    "target_raw_item",
    "target_item_id",
    "target_rating",
    "target_timestamp",
    "target_verified_purchase",
    "history_ratings",
    "history_verified_purchase",
    "split",
    "target_position",
]


@dataclass(frozen=True, slots=True)
class SequenceSplitConfig:
    split_strategy: SplitStrategy = "leave_one_out"
    min_history_len: int = 1

    # Backward-compatible flag. If train_target_policy == "all" and
    # train_all_prefixes is False, the effective policy becomes "last".
    train_all_prefixes: bool = True

    # Training target selection. This prevents all-prefix explosion for long
    # histories while keeping the output schema unchanged.
    train_target_policy: TrainTargetPolicy = "all"
    train_targets_per_user: int = 256
    train_stride: int = 10
    train_last_n: int = 512
    train_hybrid_recent_n: int = 256
    train_hybrid_random_n: int = 256

    # History is materialized as list columns. Keeping this bounded avoids the
    # old full-prefix memory blow-up and makes the vectorized history expansion
    # predictable.
    max_history_len: int | None = None

    # Training example materialization. "prefix_target" (default) slices each
    # user history into (prefix, single target) rows for encoder-decoder-style
    # training. "dense_causal" emits one row per user window with labels at
    # every position (inputs shifted by one) for decoder-only-style training;
    # train_target_policy is ignored in the default dense mode because
    # windowing already covers every eligible target exactly once.
    train_example_layout: TrainExampleLayout = "prefix_target"
    train_window_len: int | None = None  # defaults to max_history_len
    train_window_stride: int | None = None  # defaults to train_window_len
    # Opt-in comparison with prefix training: honor all/last_n target selection
    # and use a full preceding history for every selected target. Each dense
    # row supervises only its final item (stride 1), matching prefix contexts.
    train_dense_match_prefix: bool = False

    seed: int = 42


@dataclass(slots=True)
class SequenceSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def factorize_interaction_ids(
    interactions: pd.DataFrame,
    *,
    pad_raw_item: str = "<PAD>",
) -> tuple[pd.DataFrame, dict[str, int], list[str], dict[str, int], list[str]]:
    """Add compact integer user_id/item_id columns to an interaction DataFrame.

    User IDs are 0-based. Item IDs reserve 0 for padding and therefore start at
    1. Factorization follows first occurrence in chronological/user-stable order.
    """
    if interactions.empty:
        empty = interactions.copy()
        empty["user_id"] = pd.Series(dtype="int64")
        empty["item_id"] = pd.Series(dtype="int64")
        empty["position"] = pd.Series(dtype="int64")
        return empty, {}, [], {}, [pad_raw_item]

    frame = sort_interactions_frame(interactions)
    user_codes, user_uniques = pd.factorize(frame["raw_user_id"].astype(str), sort=False)
    item_codes, item_uniques = pd.factorize(frame["raw_item_id"].astype(str), sort=False)
    frame = frame.copy()
    frame["user_id"] = user_codes.astype(np.int64)
    frame["item_id"] = (item_codes + 1).astype(np.int64)
    frame["position"] = frame.groupby("user_id", sort=False).cumcount().astype(np.int64)

    id2user = [str(x) for x in user_uniques.tolist()]
    id2item = [pad_raw_item] + [str(x) for x in item_uniques.tolist()]
    user2id = {raw_user: idx for idx, raw_user in enumerate(id2user)}
    item2id = {raw_item: idx for idx, raw_item in enumerate(id2item[1:], start=1)}
    return frame.reset_index(drop=True), user2id, id2user, item2id, id2item


def build_sequence_splits(
    interactions: pd.DataFrame,
    *,
    cfg: SequenceSplitConfig,
) -> SequenceSplits:
    """Build train/validation/test examples using DataFrame operations."""
    if cfg.min_history_len < 1:
        raise ValueError("min_history_len must be >= 1")
    validate_dense_prefix_match_config(cfg)
    required = {"raw_user_id", "raw_item_id", "user_id", "item_id", "timestamp", "position"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise KeyError(f"build_sequence_splits missing columns: {missing}")

    events = _prepare_events(interactions)

    if cfg.split_strategy == "leave_one_out":
        train, val, test = build_leave_one_out_examples(events, cfg=cfg)
        return SequenceSplits(train=train, val=val, test=test)

    raise ValueError(f"Unknown split_strategy: {cfg.split_strategy}")


def _prepare_events(interactions: pd.DataFrame) -> pd.DataFrame:
    events = sort_interactions_frame(interactions)
    events = events.copy()
    if "rating" not in events.columns:
        events["rating"] = 0.0
    if "verified_purchase" not in events.columns:
        events["verified_purchase"] = None
    events["raw_user_id"] = events["raw_user_id"].astype(str)
    events["raw_item_id"] = events["raw_item_id"].astype(str)
    events["user_id"] = events["user_id"].astype("int64")
    events["item_id"] = events["item_id"].astype("int64")
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce").fillna(0).astype("int64")
    events["rating"] = pd.to_numeric(events["rating"], errors="coerce").fillna(0.0).astype(float)
    events["position"] = events.groupby("user_id", sort=False).cumcount().astype("int64")
    events["user_len"] = events.groupby("user_id", sort=False)["position"].transform("size").astype("int64")
    return events.reset_index(drop=True)


def effective_train_target_policy(cfg: SequenceSplitConfig | object) -> TrainTargetPolicy:
    policy = getattr(cfg, "train_target_policy", "all")
    train_all_prefixes = bool(getattr(cfg, "train_all_prefixes", True))
    if policy == "all" and not train_all_prefixes:
        return "last"
    return policy  # type: ignore[return-value]


def validate_dense_prefix_match_config(cfg: SequenceSplitConfig | object) -> None:
    """Reject settings that would silently break the opt-in prefix comparison."""
    if not getattr(cfg, "train_dense_match_prefix", False):
        return
    if getattr(cfg, "train_example_layout", None) != "dense_causal":
        raise ValueError("train_dense_match_prefix requires train_example_layout='dense_causal'")
    if getattr(cfg, "train_prefix_from_dense", False):
        raise ValueError("train_dense_match_prefix requires train_prefix_from_dense=False")
    history_len = getattr(cfg, "max_history_len", None)
    if history_len is None or int(history_len) < max(1, int(getattr(cfg, "min_history_len", 1))):
        raise ValueError("train_dense_match_prefix requires max_history_len >= min_history_len and >= 1")
    window_len = getattr(cfg, "train_window_len", None)
    if window_len is not None and int(window_len) != int(history_len):
        raise ValueError("train_dense_match_prefix requires train_window_len == max_history_len")
    stride = getattr(cfg, "train_window_stride", None)
    if stride is not None and int(stride) != 1:
        raise ValueError("train_dense_match_prefix requires train_window_stride=1")
    policy = effective_train_target_policy(cfg)
    if policy not in {"all", "last_n"}:
        raise ValueError("train_dense_match_prefix supports only effective train_target_policy 'all' or 'last_n'")
    if policy == "last_n" and int(getattr(cfg, "train_last_n", 0)) < 1:
        raise ValueError("train_dense_match_prefix requires train_last_n >= 1")


def select_train_targets(
    *,
    start: int,
    end: int,
    policy: TrainTargetPolicy,
    targets_per_user: int = 256,
    stride: int = 10,
    last_n: int = 512,
    hybrid_recent_n: int = 256,
    hybrid_random_n: int = 256,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compatibility helper for one user's target positions."""
    if end <= start:
        return np.empty(0, dtype=np.int64)
    if policy == "all":
        return np.arange(start, end, dtype=np.int64)
    if policy == "last":
        return np.asarray([end - 1], dtype=np.int64)
    if policy == "last_n":
        return np.arange(max(start, end - int(last_n)), end, dtype=np.int64)
    if policy == "stride":
        return np.arange(start, end, max(int(stride), 1), dtype=np.int64)
    if policy == "sampled":
        candidates = np.arange(start, end, dtype=np.int64)
        n = min(int(targets_per_user), candidates.size)
        sampled = rng.choice(candidates, size=n, replace=False)
        sampled.sort()
        return sampled.astype(np.int64, copy=False)
    if policy == "hybrid":
        recent_start = max(start, end - int(hybrid_recent_n))
        recent = np.arange(recent_start, end, dtype=np.int64)
        older = np.arange(start, recent_start, dtype=np.int64)
        n = min(int(hybrid_random_n), older.size)
        if n > 0:
            sampled_old = rng.choice(older, size=n, replace=False)
            targets = np.concatenate([sampled_old, recent])
            targets.sort()
            return targets.astype(np.int64, copy=False)
        return recent
    raise ValueError(f"Unknown train_target_policy: {policy}")


def build_leave_one_out_examples(
    events: pd.DataFrame,
    *,
    cfg: SequenceSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return _empty_sequence_frame(), _empty_sequence_frame(), _empty_sequence_frame()

    frame = events.copy()
    if cfg.train_example_layout == "dense_causal":
        train = build_dense_train_examples(events, cfg=cfg)
    else:
        frame["train_end_exclusive"] = frame["user_len"] - 2
        train_targets = _select_train_rows(frame, end_col="train_end_exclusive", cfg=cfg)
        train = _build_examples_from_targets(train_targets, events, split="train", cfg=cfg)

    eligible = frame["user_len"] >= cfg.min_history_len + 2
    val_targets = frame.loc[eligible & frame["position"].eq(frame["user_len"] - 2)].copy()
    val_targets["history_end_idx"] = val_targets["position"]
    val = _build_examples_from_targets(val_targets, events, split="val", cfg=cfg)

    test_targets = frame.loc[eligible & frame["position"].eq(frame["user_len"] - 1)].copy()
    test_targets["history_end_idx"] = test_targets["position"]
    test = _build_examples_from_targets(test_targets, events, split="test", cfg=cfg)
    return train, val, test


def _select_train_rows(frame: pd.DataFrame, *, end_col: str, cfg: SequenceSplitConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    policy = effective_train_target_policy(cfg)
    position = frame["position"].to_numpy(dtype=np.int64)
    end = frame[end_col].to_numpy(dtype=np.int64)
    base_mask = (position >= int(cfg.min_history_len)) & (position < end)
    candidates = frame.loc[base_mask].copy()
    if candidates.empty:
        return candidates

    pos = candidates["position"].to_numpy(dtype=np.int64)
    end = candidates[end_col].to_numpy(dtype=np.int64)
    if policy == "all":
        selected = candidates
    elif policy == "last":
        selected = candidates.loc[pos == (end - 1)]
    elif policy == "last_n":
        start = np.maximum(int(cfg.min_history_len), end - int(cfg.train_last_n))
        selected = candidates.loc[pos >= start]
    elif policy == "stride":
        step = max(int(cfg.train_stride), 1)
        selected = candidates.loc[((pos - int(cfg.min_history_len)) % step) == 0]
    elif policy == "sampled":
        selected = _sample_rows_per_user(
            candidates,
            n=int(cfg.train_targets_per_user),
            seed=int(cfg.seed),
        )
    elif policy == "hybrid":
        recent_start = np.maximum(int(cfg.min_history_len), end - int(cfg.train_hybrid_recent_n))
        recent = candidates.loc[pos >= recent_start]
        older = candidates.loc[pos < recent_start]
        sampled_old = _sample_rows_per_user(
            older,
            n=int(cfg.train_hybrid_random_n),
            seed=int(cfg.seed),
        )
        selected = pd.concat([sampled_old, recent], ignore_index=False).sort_values(
            ["user_id", "position"], kind="stable"
        )
    else:
        raise ValueError(f"Unknown train_target_policy: {policy}")

    selected = selected.copy()
    selected["history_end_idx"] = selected["position"]
    return selected


def _sample_rows_per_user(frame: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    out = frame.copy()
    rng = np.random.default_rng(seed)
    out["__rand"] = rng.random(len(out))
    out["__rank"] = out.groupby("user_id", sort=False)["__rand"].rank(method="first")
    sampled = out.loc[out["__rank"] <= int(n)].drop(columns=["__rand", "__rank"])
    return sampled.sort_values(["user_id", "position"], kind="stable")


def _build_examples_from_targets(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: Literal["train", "val", "test"],
    cfg: SequenceSplitConfig,
) -> pd.DataFrame:
    if targets.empty:
        return _empty_sequence_frame()

    targets = targets.copy().sort_values(["user_id", "position"], kind="stable")
    if "history_end_idx" not in targets.columns:
        targets["history_end_idx"] = targets["position"]
    targets["history_end_idx"] = targets["history_end_idx"].astype("int64")

    if cfg.max_history_len is not None and int(cfg.max_history_len) > 0:
        starts = np.maximum(0, targets["history_end_idx"].to_numpy(dtype=np.int64) - int(cfg.max_history_len))
    else:
        starts = np.zeros(len(targets), dtype=np.int64)
    targets["history_start_idx"] = starts.astype(np.int64)
    history_len = targets["history_end_idx"].to_numpy(dtype=np.int64) - starts
    targets = targets.loc[history_len >= int(cfg.min_history_len)].copy()
    if targets.empty:
        return _empty_sequence_frame()

    examples = pd.DataFrame(
        {
            "raw_user_id": targets["raw_user_id"].astype(str).to_numpy(),
            "user_id": targets["user_id"].to_numpy(dtype=np.int64),
            "target_raw_item": targets["raw_item_id"].astype(str).to_numpy(),
            "target_item_id": targets["item_id"].to_numpy(dtype=np.int64),
            "target_rating": targets["rating"].to_numpy(dtype=float),
            "target_timestamp": targets["timestamp"].to_numpy(dtype=np.int64),
            "target_verified_purchase": targets["verified_purchase"].to_numpy(dtype=object),
            "split": split,
            "target_position": targets["position"].to_numpy(dtype=np.int64),
            "history_start_idx": targets["history_start_idx"].to_numpy(dtype=np.int64),
            "history_end_idx": targets["history_end_idx"].to_numpy(dtype=np.int64),
        }
    )

    _attach_history_lists(examples, events)
    return examples[_SEQUENCE_COLUMNS].reset_index(drop=True)


def _attach_history_lists(examples: pd.DataFrame, events: pd.DataFrame) -> None:
    n = len(examples)
    empty_lists = [[] for _ in range(n)]
    for column in ["history_raw_items", "history_item_ids", "history_ratings", "history_verified_purchase"]:
        examples[column] = [list(x) for x in empty_lists]
    if n == 0:
        return

    starts = examples["history_start_idx"].to_numpy(dtype=np.int64)
    ends = examples["history_end_idx"].to_numpy(dtype=np.int64)
    lengths = ends - starts
    total = int(lengths.sum())
    if total <= 0:
        return

    max_len = int(lengths.max())
    offsets = np.arange(max_len, dtype=np.int64)
    valid = offsets[None, :] < lengths[:, None]
    positions = (starts[:, None] + offsets[None, :])[valid]
    ex_idx = np.repeat(np.arange(n, dtype=np.int64), lengths)
    hist_users = np.repeat(examples["user_id"].to_numpy(dtype=np.int64), lengths)

    hist = pd.DataFrame(
        {
            "__ex_idx": ex_idx,
            "user_id": hist_users,
            "position": positions,
        }
    )
    event_cols = events[["user_id", "position", "raw_item_id", "item_id", "rating", "verified_purchase"]]
    hist = hist.merge(event_cols, on=["user_id", "position"], how="left", sort=False)
    grouped = hist.groupby("__ex_idx", sort=False).agg(
        history_raw_items=("raw_item_id", lambda s: [str(x) for x in s]),
        history_item_ids=("item_id", lambda s: [int(x) for x in s]),
        history_ratings=("rating", lambda s: [float(x) for x in s]),
        history_verified_purchase=("verified_purchase", list),
    )
    for column in ["history_raw_items", "history_item_ids", "history_ratings", "history_verified_purchase"]:
        series = grouped[column].reindex(np.arange(n))
        examples[column] = [value if isinstance(value, list) else [] for value in series]


def build_dense_train_examples(
    events: pd.DataFrame,
    *,
    cfg: SequenceSplitConfig,
) -> pd.DataFrame:
    """Build dense-causal train rows: one row per user window, labels everywhere.

    The train region matches the prefix-target builders exactly.  For
    leave_one_out the region is positions [0, user_len - 2), so the last event
    that ever appears as a label is the one at position user_len - 3; the
    validation target (user_len - 2) and test target (user_len - 1) never enter
    a train row, neither as input nor as label.

    Within the region sequence s[0..n-1], inputs are s[0..n-2] and the label at
    input index i is s[i+1].  A label is masked out when its sequence index is
    below min_history_len, which makes the supervised target set identical to
    train_target_policy="all" in prefix mode.

    Long sequences are windowed with train_window_len inputs per row.  Windows
    are anchored at the end of the sequence so the most recent transitions
    always get full context.  With a stride smaller than the window length,
    consecutive windows overlap and only the trailing stride positions of each
    window keep active labels, so every eligible label is supervised exactly
    once.

    With train_dense_match_prefix=True, only targets selected by all/last_n
    are emitted. Each gets its full preceding max_history_len items and only
    the final label is active, so the targets, contexts, and observed catalog
    match prefix_target. This intentionally trades packing efficiency for an
    exact comparison; it does not materialize the discarded older prefixes.
    """
    validate_dense_prefix_match_config(cfg)
    if events.empty:
        return _empty_dense_frame()

    window_len = cfg.train_window_len if cfg.train_window_len is not None else cfg.max_history_len
    if window_len is not None and int(window_len) < 1:
        raise ValueError("train_window_len must be >= 1")
    stride = cfg.train_window_stride if cfg.train_window_stride is not None else (
        1 if cfg.train_dense_match_prefix else window_len
    )
    if stride is not None and int(stride) < 1:
        raise ValueError("train_window_stride must be >= 1")
    if window_len is not None and stride is not None and int(stride) > int(window_len):
        raise ValueError(
            "train_window_stride must be <= train_window_len, otherwise some "
            "training targets would be skipped."
        )

    if cfg.split_strategy != "leave_one_out":
        raise ValueError(f"Unknown split_strategy: {cfg.split_strategy}")
    region = events["position"].lt(events["user_len"] - 2)

    region_events = events.loc[region]
    if region_events.empty:
        return _empty_dense_frame()

    min_history_len = int(cfg.min_history_len)
    rows: list[dict] = []
    for _, group in region_events.groupby("user_id", sort=False):
        n = len(group)
        # Need at least one label at sequence index >= min_history_len.
        if n < max(min_history_len + 1, 2):
            continue

        raw_items = group["raw_item_id"].astype(str).tolist()
        item_ids = [int(x) for x in group["item_id"].tolist()]
        ratings = [float(x) for x in group["rating"].tolist()]
        verified = list(group["verified_purchase"].tolist())
        timestamps = [int(x) for x in group["timestamp"].tolist()]
        positions = [int(x) for x in group["position"].tolist()]
        raw_user_id = str(group["raw_user_id"].iloc[0])
        user_id = int(group["user_id"].iloc[0])

        num_inputs = n - 1
        w = int(window_len) if window_len is not None else num_inputs
        s = int(stride) if stride is not None else w

        if cfg.train_dense_match_prefix:
            first_target = min_history_len
            if effective_train_target_policy(cfg) == "last_n":
                first_target = max(first_target, n - int(cfg.train_last_n))
            ends = range(first_target, n)
        else:
            ends = range(num_inputs, 0, -s)
        for end in ends:
            begin = max(0, end - w)
            active_lo = max(begin, end - s)
            label_mask = [
                active_lo <= i and (i + 1) >= min_history_len for i in range(begin, end)
            ]
            if not any(label_mask):
                continue

            last_valid_input = begin + max(i for i, valid in enumerate(label_mask) if valid)
            target_idx = last_valid_input + 1
            rows.append(
                {
                    "raw_user_id": raw_user_id,
                    "user_id": user_id,
                    "history_raw_items": raw_items[begin : begin + len(label_mask)],
                    "history_item_ids": item_ids[begin : begin + len(label_mask)],
                    "label_raw_items": raw_items[begin + 1 : begin + 1 + len(label_mask)],
                    "label_item_ids": item_ids[begin + 1 : begin + 1 + len(label_mask)],
                    "label_mask": label_mask,
                    "target_raw_item": raw_items[target_idx],
                    "target_item_id": item_ids[target_idx],
                    "target_rating": ratings[target_idx],
                    "target_timestamp": timestamps[target_idx],
                    "target_verified_purchase": verified[target_idx],
                    "history_ratings": ratings[begin : begin + len(label_mask)],
                    "history_verified_purchase": verified[begin : begin + len(label_mask)],
                    "split": "train",
                    "target_position": positions[target_idx],
                }
            )

    if not rows:
        return _empty_dense_frame()
    return pd.DataFrame(rows, columns=_DENSE_SEQUENCE_COLUMNS)


def is_dense_example_frame(frame: pd.DataFrame) -> bool:
    """Return True when a split frame carries dense-causal label columns."""
    return "label_item_ids" in frame.columns


def validate_dense_layout_config(
    *,
    layout: TrainExampleLayout,
    history_padding: str,
) -> None:
    """Cheap dense-layout config validation, run before any data is loaded."""
    if layout != "dense_causal":
        return

    if history_padding not in {"left", "right"}:
        raise ValueError("history_padding must be 'left' or 'right'")


def _empty_sequence_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in _SEQUENCE_COLUMNS})


def _empty_dense_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in _DENSE_SEQUENCE_COLUMNS})
