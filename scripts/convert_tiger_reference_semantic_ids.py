from __future__ import annotations

import argparse
import ast
import gzip
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the reference TIGER Beauty semantic IDs to FlowRec raw-item keys."
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "reviews_Beauty.json.gz",
    )
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=PROJECT_ROOT.parent / "tiger" / "data" / "Beauty" / "index_rqvae.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tokenizers"
        / "rqvae_beauty_reference"
        / "semantic_ids.json",
    )
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument(
        "--level-offset-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offset level l by l * codebook_size for FlowRec TIGER training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.reviews_path)
    filtered = apply_k_core_filter(
        rows,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )

    item_to_reference_id: dict[str, int] = {}
    for _, item_id, _ in filtered:
        item_to_reference_id.setdefault(item_id, len(item_to_reference_id))

    with open(args.reference_index, "r", encoding="utf-8") as f:
        reference_ids = json.load(f)

    mapping: dict[str, list[int]] = {}
    for raw_item_id, reference_id in item_to_reference_id.items():
        sid = list(reference_ids[str(reference_id)])
        if args.level_offset_tokens:
            sid = [token + level * args.codebook_size for level, token in enumerate(sid)]
        mapping[raw_item_id] = sid

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"Saved {len(mapping)} semantic IDs to {args.output}")


def load_rows(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = ast.literal_eval(line)
            rows.append(
                (
                    str(record["reviewerID"]),
                    str(record["asin"]),
                    int(record["unixReviewTime"]),
                )
            )
    return rows


def apply_k_core_filter(
    rows: list[tuple[str, str, int]],
    *,
    min_user_interactions: int,
    min_item_interactions: int,
) -> list[tuple[str, str, int]]:
    filtered = rows
    changed = True
    while changed:
        user_counts = Counter(user_id for user_id, _, _ in filtered)
        item_counts = Counter(item_id for _, item_id, _ in filtered)
        next_filtered = [
            row
            for row in filtered
            if user_counts[row[0]] >= min_user_interactions
            and item_counts[row[1]] >= min_item_interactions
        ]
        changed = len(next_filtered) != len(filtered)
        filtered = next_filtered
    return filtered


if __name__ == "__main__":
    main()
