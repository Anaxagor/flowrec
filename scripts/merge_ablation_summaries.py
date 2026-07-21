#!/usr/bin/env python
"""Merge per-job summaries from a scheduler-array reward ablation.

Each array task writes its own timestamped ``<out>/<stamp>_rewards_ablation/``
directory with a ``summary.csv`` holding one (reward, seed) record. This script
globs them all, concatenates the records, and writes a merged ``summary.csv``,
``summary_aggregated.csv`` (mean +/- std over seeds), and ``summary.md`` at the
top level, reusing the ablation's own writer so the format is identical.

Usage:
    python scripts/merge_ablation_summaries.py experiments/rewards_ablation_cluster
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rewards_ablation import SUMMARY_METRICS, write_summary  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/rewards_ablation_cluster")
    if not root.exists():
        raise SystemExit(f"No such directory: {root}")

    numeric = set(SUMMARY_METRICS) | {"best_metric"}
    records: list[dict] = []
    for csv_path in sorted(root.glob("*_rewards_ablation/summary.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                record: dict = {}
                for key, value in row.items():
                    if value is None or value == "":
                        continue
                    if key in numeric:
                        record[key] = float(value)
                    elif key == "seed":
                        record[key] = int(float(value))
                    else:
                        record[key] = value
                records.append(record)

    if not records:
        raise SystemExit(f"No summary.csv files found under {root}/*_rewards_ablation/")

    # Deduplicate (dataset, reward, seed): keep the last occurrence, so re-run
    # array tasks override earlier failed attempts.
    unique: dict[tuple, dict] = {}
    for record in records:
        unique[(record.get("dataset"), record.get("reward"), record.get("seed"))] = record
    merged = list(unique.values())
    merged.sort(key=lambda r: (str(r.get("dataset")), str(r.get("reward")), str(r.get("seed"))))

    write_summary(root, merged)
    completed = sum(1 for r in merged if r.get("status") == "completed")
    print(f"merged {len(merged)} records ({completed} completed) from {root}")
    print(f"-> {root / 'summary.csv'}")
    print(f"-> {root / 'summary_aggregated.csv'}")
    print(f"-> {root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
