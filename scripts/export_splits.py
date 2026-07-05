from __future__ import annotations

"""Export leave-one-out train/val/test splits as flat interaction CSVs.

Given the SASRec-Beauty data configuration in ``instruments_sasrec.py`` this
script writes five files, each a flat interaction table with the header::

    user_id,item_id,timestamp
    0,9450,1405296000
    ...

Files written:
  - train.csv              all-but-last-two interactions per user
  - validation_input.csv   history that precedes the validation target
  - validation_target.csv  the single validation target per user (2nd to last)
  - test_input.csv         history that precedes the test target
  - test_target.csv        the single test target per user (last interaction)

IDs match exactly what the model is trained on: ``user_id`` / ``item_id`` are the
integer IDs from the datamodule's (cold-item-filtered, compacted) catalog, and
``timestamp`` is the raw unix review time.  Interactions on items that were
dropped by post-split cold-item filtering are excluded, because they have no ID
in the training catalog.
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from flowrec.data.datasets import InteractionRecord
from flowrec.data.preprocessors import (
    apply_k_core_filter,
    load_metadata,
    load_reviews_by_user,
    sort_user_events,
)

# Reuse the exact data configuration the SASRec experiment trains on so the
# splits and integer IDs are identical to the model's.
from instruments_sasrec import build_datamodule

Row = tuple[int, int, int]  # (user_id, item_id, timestamp)


def load_sorted_user_events(cfg) -> dict[str, list[InteractionRecord]]:
    """Re-run the datamodule's preprocessing to recover per-event timestamps.

    ``SequenceExample`` objects do not retain history timestamps and cap history
    length, so we rebuild the full chronological interaction list per user using
    the same loading / k-core / sorting steps as ``AmazonReviewDataModule.setup``.
    """
    allowed_items: set[str] | None = None
    if cfg.metadata_path is not None:
        metadata = load_metadata(cfg.metadata_path, item_key=cfg.item_key)
        allowed_items = set(metadata.keys())

    user_events = load_reviews_by_user(
        cfg.reviews_path,
        item_key=cfg.item_key,
        use_verified_only=cfg.use_verified_only,
        keep_raw=cfg.keep_raw_reviews,
        allowed_items=allowed_items,
    )
    user_events = apply_k_core_filter(
        user_events,
        min_user_interactions=cfg.min_user_interactions,
        min_item_interactions=cfg.min_item_interactions,
        drop_consecutive_repeats=cfg.deduplicate_consecutive,
    )
    return sort_user_events(user_events)


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "timestamp"])
        writer.writerows(rows)


def build_split_rows(dm, user_events: dict[str, list[InteractionRecord]]):
    """Reconstruct the five interaction tables from full user sequences.

    Leave-one-out: for a user's chronological interactions ``x[0..N-1]`` the last
    item is the test target, the second-to-last is the validation target, and
    everything before that is training data.  Cold items (not in the training
    catalog) are dropped from every sequence.
    """
    user2id = dm.user2id
    item2id = dm.item2id
    min_history_len = dm.cfg.min_history_len

    train_rows: list[Row] = []
    val_input_rows: list[Row] = []
    val_target_rows: list[Row] = []
    test_input_rows: list[Row] = []
    test_target_rows: list[Row] = []

    n_users_val = 0
    n_users_test = 0

    for raw_user_id, events in user_events.items():
        # Users shorter than min_history_len + 2 never receive a val/test target
        # and are not assigned an id by the datamodule.
        if len(events) < min_history_len + 2:
            continue
        user_id = user2id.get(raw_user_id)
        if user_id is None:
            continue

        n = len(events)
        val_pos = n - 2
        test_pos = n - 1

        def warm_rows(events_slice: list[InteractionRecord]) -> list[Row]:
            out: list[Row] = []
            for ev in events_slice:
                item_id = item2id.get(ev.raw_item_id)
                if item_id is None:  # cold item removed by post-split filtering
                    continue
                out.append((user_id, item_id, int(ev.timestamp)))
            return out

        # Training data: everything up to (but excluding) the validation target.
        train_slice = warm_rows(events[:val_pos])
        train_rows.extend(train_slice)

        # Validation: input = history before the val target; target = x[N-2].
        val_target_id = item2id.get(events[val_pos].raw_item_id)
        val_input = train_slice  # history preceding the validation target
        if val_target_id is not None and len(val_input) >= min_history_len:
            val_input_rows.extend(val_input)
            val_target_rows.append(
                (user_id, val_target_id, int(events[val_pos].timestamp))
            )
            n_users_val += 1

        # Test: input = history before the test target (includes the val item);
        # target = x[N-1].
        test_target_id = item2id.get(events[test_pos].raw_item_id)
        test_input = warm_rows(events[:test_pos])
        if test_target_id is not None and len(test_input) >= min_history_len:
            test_input_rows.extend(test_input)
            test_target_rows.append(
                (user_id, test_target_id, int(events[test_pos].timestamp))
            )
            n_users_test += 1

    return {
        "train.csv": train_rows,
        "validation_input.csv": val_input_rows,
        "validation_target.csv": val_target_rows,
        "test_input.csv": test_input_rows,
        "test_target.csv": test_target_rows,
    }, n_users_val, n_users_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data/splits/beauty_14",
        help="Directory to write the CSV files into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to the datamodule (affects catalog determinism).",
    )
    args = parser.parse_args()

    print(f"Building datamodule (seed={args.seed}) ...")
    dm = build_datamodule(args.seed)

    print("Recovering per-event timestamps ...")
    user_events = load_sorted_user_events(dm.cfg)

    print("Reconstructing splits ...")
    files, n_users_val, n_users_test = build_split_rows(dm, user_events)

    out_dir: Path = args.out
    for name, rows in files.items():
        path = out_dir / name
        write_csv(path, rows)
        print(f"  wrote {len(rows):>9,} rows -> {path}")

    print(
        f"\nUsers with a validation example: {n_users_val:,} | "
        f"with a test example: {n_users_test:,}"
    )
    print(f"Done. Files in: {out_dir}")


if __name__ == "__main__":
    main()
