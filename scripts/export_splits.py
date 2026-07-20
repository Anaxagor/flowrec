from __future__ import annotations

"""Export leave-one-out train/val/test splits as flat interaction CSVs.

Given the data configuration in ``instruments_sasrec.py`` this script writes
five files, each a flat interaction table with the header::

    user_id,item_id,timestamp
    0,9450,1405296000
    ...

Files written:
  - train.csv              all-but-last-two interactions per user
  - validation_input.csv   history that precedes the validation target
  - validation_target.csv  the single validation target per user (2nd to last)
  - test_input.csv         history that precedes the test target
  - test_target.csv        the single test target per user (last interaction)

IDs match exactly what the model is trained on: ``user_id`` / ``item_id`` are
the integer IDs from the datamodule's (optionally cold-item-filtered) catalog,
and ``timestamp`` is the raw unix event time.  Interactions on items without a
catalog ID are excluded.

The export is layout-independent: both the prefix_target and dense_causal
training layouts consume exactly the train region written to train.csv (all
events before the validation target), so these CSVs describe either kind of
run.  Events are read from the datamodule's canonical interaction DataFrame,
which already carries per-event timestamps.
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the exact data configuration the SASRec experiment trains on so the
# splits and integer IDs are identical to the model's.
from instruments_sasrec import build_datamodule

Row = tuple[int, int, int]  # (user_id, item_id, timestamp)


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "timestamp"])
        writer.writerows(rows)


def build_split_rows(dm):
    """Reconstruct the five interaction tables from the events DataFrame.

    Leave-one-out: for a user's chronological interactions ``x[0..N-1]`` the
    last item is the test target, the second-to-last is the validation target,
    and everything before that is training data.  Items without a catalog ID
    (removed by post-split cold-item filtering) are dropped from every
    sequence.
    """
    item2id = dm.item2id
    min_history_len = int(dm.cfg.min_history_len)

    train_rows: list[Row] = []
    val_input_rows: list[Row] = []
    val_target_rows: list[Row] = []
    test_input_rows: list[Row] = []
    test_target_rows: list[Row] = []

    n_users_val = 0
    n_users_test = 0

    events = dm.interactions.sort_values(["user_id", "position"], kind="stable")
    for user_id, group in events.groupby("user_id", sort=False):
        n = len(group)
        # Users shorter than min_history_len + 2 never receive a val/test
        # target in the leave-one-out split.
        if n < min_history_len + 2:
            continue

        raw_items = group["raw_item_id"].astype(str).tolist()
        timestamps = [int(t) for t in group["timestamp"].tolist()]
        user_id = int(user_id)

        def warm_rows(start: int, end: int) -> list[Row]:
            out: list[Row] = []
            for idx in range(start, end):
                item_id = item2id.get(raw_items[idx])
                if item_id is None:  # cold item removed by post-split filtering
                    continue
                out.append((user_id, int(item_id), timestamps[idx]))
            return out

        val_pos = n - 2
        test_pos = n - 1

        # Training data: everything up to (but excluding) the validation target.
        train_slice = warm_rows(0, val_pos)
        train_rows.extend(train_slice)

        # Validation: input = history before the val target; target = x[N-2].
        val_target_id = item2id.get(raw_items[val_pos])
        if val_target_id is not None and len(train_slice) >= min_history_len:
            val_input_rows.extend(train_slice)
            val_target_rows.append((user_id, int(val_target_id), timestamps[val_pos]))
            n_users_val += 1

        # Test: input = history before the test target (includes the val item);
        # target = x[N-1].
        test_target_id = item2id.get(raw_items[test_pos])
        test_input = warm_rows(0, test_pos)
        if test_target_id is not None and len(test_input) >= min_history_len:
            test_input_rows.extend(test_input)
            test_target_rows.append((user_id, int(test_target_id), timestamps[test_pos]))
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
        default=PROJECT_ROOT / "data/splits/instruments_2018",
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

    print("Reconstructing splits ...")
    files, n_users_val, n_users_test = build_split_rows(dm)

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
