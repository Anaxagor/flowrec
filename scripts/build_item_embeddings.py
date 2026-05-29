#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.preprocessors import (
    apply_k_core_filter,
    collect_item_ids,
    count_interactions,
    load_reviews_by_user,
)
from flowrec.data.yambda import load_yambda_events_by_user
from flowrec.item_representation.content_embeddings import (
    AmazonItemTextBuilder,
    ContentEmbeddingBuilder,
    HashingTextEmbedder,
    HFTextEmbedder,
)
from flowrec.item_representation.yambda import (
    YambdaAudioEmbeddingBuilder,
    YambdaAudioEmbeddingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FlowRec item embedding stores from text metadata or "
            "Yambda audio embeddings."
        )
    )
    parser.add_argument("--source", choices=("metadata", "yambda-audio"), default="metadata")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--allowed-items-file", type=Path, default=None)

    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--item-key", default="asin")
    parser.add_argument("--reviews-path", type=Path, default=None)
    parser.add_argument("--review-item-key", default=None)
    parser.add_argument("--min-user-interactions", type=int, default=1)
    parser.add_argument("--min-item-interactions", type=int, default=1)
    parser.add_argument("--save-texts", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--embedder", choices=("hf", "hashing"), default="hf")
    parser.add_argument("--model-name", default="huggyllama/llama-7b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument(
        "--pooling",
        choices=("masked_mean", "padded_mean"),
        default="padded_mean",
    )
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hashing-dim", type=int, default=384)

    parser.add_argument("--include-title", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-main-category", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-categories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-store", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-description", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-details", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-price", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--separator", default=". ")
    parser.add_argument("--category-separator", default=" > ")
    parser.add_argument("--category-path-separator", default=" | ")
    parser.add_argument("--first-category-path-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--embedding-column", default="normalized_embed")
    parser.add_argument("--fallback-embedding-column", default="embed")
    parser.add_argument("--embedding-item-id-column", default="item_id")
    parser.add_argument("--max-embedding-items", type=int, default=None)

    parser.add_argument("--yambda-interactions-path", type=Path, default=None)
    parser.add_argument("--use-hf-yambda-interactions", action="store_true")
    parser.add_argument("--hf-dataset-name", default="yandex/yambda")
    parser.add_argument("--dataset-type", choices=("flat", "sequential"), default="sequential")
    parser.add_argument("--dataset-size", choices=("50m", "500m", "5b"), default="50m")
    parser.add_argument(
        "--event-file",
        choices=("listens", "likes", "dislikes", "unlikes", "undislikes", "multi_event"),
        default="listens",
    )
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-interaction-rows", type=int, default=None)
    parser.add_argument(
        "--organic-filter",
        choices=("any", "organic", "recommendation"),
        default="any",
    )
    parser.add_argument("--min-played-ratio-pct", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output_path)
    summary_path = (
        resolve_path(args.summary_path)
        if args.summary_path
        else output_path.with_suffix(".summary.json")
    )
    allowed_items = collect_allowed_items(args)

    if args.source == "metadata":
        result = build_metadata_embeddings(args, allowed_items=allowed_items)
        store = result.store
        summary = {
            "source": args.source,
            "output_path": output_path,
            "num_items": store.num_items,
            "embedding_dim": store.embedding_dim,
            "num_records_read": result.num_records_read,
            "num_empty_texts": result.num_empty_texts,
            "num_duplicate_items": result.num_duplicate_items,
            "num_filtered_items": result.num_filtered_items,
            "allowed_item_count": len(allowed_items) if allowed_items is not None else None,
            "metadata": store.metadata,
        }
    else:
        store = build_yambda_audio_embeddings(args, allowed_items=allowed_items)
        summary = {
            "source": args.source,
            "output_path": output_path,
            "num_items": store.num_items,
            "embedding_dim": store.embedding_dim,
            "allowed_item_count": len(allowed_items) if allowed_items is not None else None,
            "metadata": store.metadata,
        }

    store.save(output_path)
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def build_metadata_embeddings(args: argparse.Namespace, *, allowed_items: set[str] | None):
    if args.metadata_path is None:
        raise ValueError("--metadata-path is required when --source metadata")

    text_builder = AmazonItemTextBuilder(
        include_title=args.include_title,
        include_main_category=args.include_main_category,
        include_categories=args.include_categories,
        include_store=args.include_store,
        include_features=args.include_features,
        include_description=args.include_description,
        include_details=args.include_details,
        include_price=args.include_price,
        max_chars=args.max_chars,
        separator=args.separator,
        category_separator=args.category_separator,
        category_path_separator=args.category_path_separator,
        first_category_path_only=args.first_category_path_only,
    )
    embedder = build_embedder(args)
    builder = ContentEmbeddingBuilder(
        text_builder=text_builder,
        embedder=embedder,
        item_key=args.item_key,
    )
    return builder.build_from_metadata(
        metadata_path=resolve_path(args.metadata_path),
        batch_size=args.batch_size,
        save_texts=args.save_texts,
        allowed_item_ids=allowed_items,
    )


def build_embedder(args: argparse.Namespace):
    if args.embedder == "hashing":
        return HashingTextEmbedder(dim=args.hashing_dim, normalize=args.normalize)
    return HFTextEmbedder(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        normalize=args.normalize,
        pooling=args.pooling,
        torch_dtype=args.torch_dtype,
        show_progress=args.show_progress,
    )


def build_yambda_audio_embeddings(
    args: argparse.Namespace,
    *,
    allowed_items: set[str] | None,
):
    if args.embeddings_path is None:
        raise ValueError("--embeddings-path is required when --source yambda-audio")
    return YambdaAudioEmbeddingBuilder(
        YambdaAudioEmbeddingConfig(
            embeddings_path=resolve_path(args.embeddings_path),
            embedding_column=args.embedding_column,
            fallback_embedding_column=args.fallback_embedding_column,
            item_id_column=args.embedding_item_id_column,
            max_items=args.max_embedding_items,
            allowed_items=allowed_items,
        )
    ).build()


def collect_allowed_items(args: argparse.Namespace) -> set[str] | None:
    allowed_sets: list[set[str]] = []

    if args.allowed_items_file is not None:
        allowed_sets.append(read_item_ids(resolve_path(args.allowed_items_file)))

    if args.reviews_path is not None:
        user_events = load_reviews_by_user(
            resolve_path(args.reviews_path),
            item_key=args.review_item_key or args.item_key,
        )
        filtered = apply_k_core_filter(
            user_events,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
        print(
            "Review filter:",
            f"raw_interactions={count_interactions(user_events)}",
            f"filtered_interactions={count_interactions(filtered)}",
            f"items={len(collect_item_ids(filtered))}",
        )
        allowed_sets.append(collect_item_ids(filtered))

    if args.use_hf_yambda_interactions or args.yambda_interactions_path is not None:
        user_events = load_yambda_events_by_user(
            interactions_path=(
                None
                if args.yambda_interactions_path is None
                else resolve_path(args.yambda_interactions_path)
            ),
            use_hf=bool(args.use_hf_yambda_interactions),
            hf_dataset_name=args.hf_dataset_name,
            dataset_type=args.dataset_type,
            dataset_size=args.dataset_size,
            event_file=args.event_file,
            streaming=args.streaming,
            max_rows=args.max_interaction_rows,
            organic_filter=args.organic_filter,
            min_played_ratio_pct=args.min_played_ratio_pct,
        )
        items = collect_item_ids(user_events)
        print(
            "Yambda filter:",
            f"interactions={count_interactions(user_events)}",
            f"items={len(items)}",
        )
        allowed_sets.append(items)

    if not allowed_sets:
        return None
    allowed = set.intersection(*allowed_sets)
    if not allowed:
        raise ValueError("Allowed-item filters produced an empty item set.")
    return allowed


def read_item_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(key) for key in data}
        if isinstance(data, list):
            return {str(value) for value in data}
        raise ValueError("--allowed-items-file JSON must be a list or object")
    return {line.strip() for line in text.splitlines() if line.strip()}


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    return value


if __name__ == "__main__":
    main()
