"""Convert tiger's inter.json to a flowrec-compatible interactions file.

Tiger stores interactions as integer-indexed sequences:
  inter.json:  {user_int: [item_int, item_int, ...]}

Flowrec expects a JSONL file where each line is a review record with
string user/item IDs and a timestamp for chronological sorting.

This script:
  1. Reconstructs the item integer → ASIN mapping using the same k-core
     filtering and first-occurrence ordering that tiger's DatasetProcessing
     notebook applied (matching convert_tiger_reference_semantic_ids.py).
  2. Writes a synthetic JSONL where each record carries the sequence position
     as its timestamp — preserving tiger's exact chronological ordering.
  3. The resulting file can be used as reviews_path in any flowrec config,
     paired with rqvae_beauty_reference/semantic_ids.json for SIDs.

Usage:
    python scripts/convert_tiger_interactions.py
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "dataframe_format" / "reviews_beauty_2014.parquet",
    )
    parser.add_argument(
        "--tiger-inter",
        type=Path,
        default=PROJECT_ROOT.parent / "tiger" / "data" / "Beauty" / "inter.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tiger_beauty_interactions.jsonl",
    )
    parser.add_argument("--min-interactions", type=int, default=5)
    return parser.parse_args()


def load_rows(path: Path) -> list[tuple[str, str, int]]:
    """Load (reviewer, asin, timestamp) rows in source order.

    Parquet (and strict JSONL) inputs go through the shared interaction
    normalizer — the repo's primary path. Legacy 2014-era ``.json.gz`` dumps
    with python-literal lines remain readable via the gzip fallback. Row order
    follows the source file either way, which matters because the downstream
    first-occurrence item numbering must replicate tiger's DatasetProcessing
    exactly.
    """
    if path.suffix != ".gz":
        from flowrec.data.preprocessors import load_interactions_frame

        frame = load_interactions_frame(path, item_key="asin")
        return [
            (str(user), str(item), int(ts))
            for user, item, ts in zip(
                frame["raw_user_id"], frame["raw_item_id"], frame["timestamp"], strict=True
            )
        ]

    rows: list[tuple[str, str, int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = ast.literal_eval(line)
            rows.append((
                str(record["reviewerID"]),
                str(record["asin"]),
                int(record["unixReviewTime"]),
            ))
    return rows


def apply_k_core(
    rows: list[tuple[str, str, int]],
    min_k: int,
) -> list[tuple[str, str, int]]:
    filtered = rows
    changed = True
    while changed:
        user_counts = Counter(u for u, _, _ in filtered)
        item_counts = Counter(i for _, i, _ in filtered)
        next_filtered = [
            r for r in filtered
            if user_counts[r[0]] >= min_k and item_counts[r[1]] >= min_k
        ]
        changed = len(next_filtered) != len(filtered)
        filtered = next_filtered
    return filtered


def main() -> None:
    args = parse_args()

    # Step 1: reconstruct item integer → ASIN mapping using tiger's exact ordering.
    # Tiger's DatasetProcessing assigns integer IDs by first-occurrence order after
    # k-core filtering, which is what setdefault() replicates here.
    print("Loading and filtering reviews...")
    rows = load_rows(args.reviews_path)
    filtered = apply_k_core(rows, args.min_interactions)

    item_int_to_asin: dict[int, str] = {}
    for _, asin, _ in filtered:
        item_int_to_asin.setdefault(len(item_int_to_asin), asin)
    # Invert: asin → int, then flip
    asin_to_int: dict[str, int] = {}
    for _, asin, _ in filtered:
        asin_to_int.setdefault(asin, len(asin_to_int))
    item_int_to_asin = {v: k for k, v in asin_to_int.items()}

    print(f"  Items in mapping: {len(item_int_to_asin)}")

    # Step 2: load tiger's interaction sequences.
    print("Loading tiger inter.json...")
    with open(args.tiger_inter, encoding="utf-8") as f:
        tiger_inter: dict[str, list[int]] = json.load(f)

    print(f"  Users in tiger inter.json: {len(tiger_inter)}")

    # Validate mapping coverage.
    all_item_ints = {item for seq in tiger_inter.values() for item in seq}
    missing = all_item_ints - set(item_int_to_asin.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} item integers in inter.json have no ASIN mapping. "
            "The reviews file and inter.json may not correspond to the same dataset."
        )

    # Step 3: write synthetic JSONL.
    # Each user's items are written in sequence order with position-as-timestamp,
    # so flowrec's chronological sort recovers tiger's exact ordering.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for user_int_str, item_ints in tiger_inter.items():
            user_id = f"tiger_user_{user_int_str}"
            for position, item_int in enumerate(item_ints):
                asin = item_int_to_asin[item_int]
                record = {
                    "reviewerID": user_id,
                    "asin": asin,
                    "unixReviewTime": position,
                    "overall": 5.0,
                }
                out.write(json.dumps(record) + "\n")
                records_written += 1

    print(f"Wrote {records_written} records to {args.output}")
    print(f"Users: {len(tiger_inter)}, Items: {len(item_int_to_asin)}")
    print()
    print("Next step: use this file as reviews_path in your config,")
    print("paired with artifacts/tokenizers/rqvae_beauty_reference/semantic_ids.json")


if __name__ == "__main__":
    main()
