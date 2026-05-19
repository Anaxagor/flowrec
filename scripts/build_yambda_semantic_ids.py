#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.yambda import load_yambda_events_by_user
from flowrec.item_representation.semantic_id import (
    RQVAEConfig,
    RQVAEItemTokenizer,
    RQVAEItemTokenizerConfig,
    RQVAETrainConfig,
)
from flowrec.item_representation.yambda import (
    YambdaAudioEmbeddingBuilder,
    YambdaAudioEmbeddingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FlowRec RQ-VAE semantic IDs from local Yambda audio embeddings."
    )
    parser.add_argument("--interactions-path", type=Path, default=None)
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/tokenizers/yambda_rqvae")
    parser.add_argument(
        "--embedding-store-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts/item_embeddings/yambda_audio_embeddings.pt",
    )
    parser.add_argument("--skip-save-embedding-store", action="store_true")

    parser.add_argument("--use-hf-interactions", action="store_true")
    parser.add_argument("--use-hf-embeddings", action="store_true")
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
    parser.add_argument("--max-embedding-items", type=int, default=None)
    parser.add_argument("--no-filter-to-interactions", action="store_true")
    parser.add_argument(
        "--organic-filter",
        choices=("any", "organic", "recommendation"),
        default="any",
    )
    parser.add_argument("--min-played-ratio-pct", type=float, default=None)

    parser.add_argument("--embedding-column", default="normalized_embed")
    parser.add_argument("--fallback-embedding-column", default="embed")
    parser.add_argument("--item-id-column", default="item_id")

    parser.add_argument("--num-codebooks", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dead-code-reset-interval",
        type=int,
        default=0,
        help="0 disables full-dataset dead-code resets, which is usually better for Yambda scale.",
    )
    parser.add_argument(
        "--collision-policy",
        choices=("none", "append", "random_append"),
        default="append",
    )
    parser.add_argument("--normalize", choices=("none", "l2", "standard"), default="l2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactions_path = resolve_optional_path(args.interactions_path)
    embeddings_path = resolve_optional_path(args.embeddings_path)
    output_dir = resolve_path(args.output_dir)
    embedding_store_path = resolve_path(args.embedding_store_path)

    if not args.use_hf_embeddings and embeddings_path is None:
        raise ValueError("--embeddings-path is required unless --use-hf-embeddings is set.")

    allowed_items = None
    if not args.no_filter_to_interactions:
        allowed_items = collect_interaction_items(args, interactions_path=interactions_path)
        if not allowed_items:
            raise ValueError("No Yambda item IDs were collected from interactions.")

    store = YambdaAudioEmbeddingBuilder(
        YambdaAudioEmbeddingConfig(
            embeddings_path=embeddings_path,
            use_hf=bool(args.use_hf_embeddings),
            hf_dataset_name=str(args.hf_dataset_name),
            embedding_column=str(args.embedding_column),
            fallback_embedding_column=str(args.fallback_embedding_column),
            item_id_column=str(args.item_id_column),
            max_items=args.max_embedding_items,
            allowed_items=allowed_items,
        )
    ).build()

    if not args.skip_save_embedding_store:
        store.save(embedding_store_path)

    dead_code_reset_interval = (
        None if int(args.dead_code_reset_interval) <= 0 else int(args.dead_code_reset_interval)
    )
    tokenizer = RQVAEItemTokenizer(
        RQVAEItemTokenizerConfig(
            rqvae=RQVAEConfig(
                input_dim=int(store.embeddings.shape[1]),
                latent_dim=int(args.latent_dim),
                num_codebooks=int(args.num_codebooks),
                codebook_size=int(args.codebook_size),
            ),
            train=RQVAETrainConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                device=args.device,
                seed=int(args.seed),
                dead_code_reset_interval=dead_code_reset_interval,
            ),
            normalize=args.normalize,
            collision_policy=args.collision_policy,
        )
    )
    tokenizer.fit(store.embeddings, store.item_ids)
    tokenizer.save(output_dir)

    summary = {
        "output_dir": output_dir,
        "embedding_store_path": None if args.skip_save_embedding_store else embedding_store_path,
        "num_allowed_items": len(allowed_items) if allowed_items is not None else None,
        "num_embedding_items": store.num_items,
        "embedding_dim": store.embedding_dim,
        "semantic_id_path": output_dir / "semantic_ids.json",
        "semantic_id_length": tokenizer.semantic_id_store.sid_length
        if tokenizer.semantic_id_store is not None
        else None,
        "unique_semantic_ids": tokenizer.semantic_id_store.unique_code_count
        if tokenizer.semantic_id_store is not None
        else None,
        "collision_count": tokenizer.semantic_id_store.collision_count
        if tokenizer.semantic_id_store is not None
        else None,
        "store_metadata": store.metadata,
        "rqvae": asdict(tokenizer.config.rqvae),
        "train": asdict(tokenizer.config.train),
    }
    write_json(output_dir / "yambda_prepare_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def collect_interaction_items(
    args: argparse.Namespace,
    *,
    interactions_path: Path | None,
) -> set[str]:
    if not args.use_hf_interactions and interactions_path is None:
        raise ValueError(
            "--interactions-path is required unless --use-hf-interactions or "
            "--no-filter-to-interactions is set."
        )

    user_events = load_yambda_events_by_user(
        interactions_path=interactions_path,
        use_hf=bool(args.use_hf_interactions),
        dataset_type=args.dataset_type,
        dataset_size=args.dataset_size,
        event_file=args.event_file,
        hf_dataset_name=args.hf_dataset_name,
        streaming=bool(args.streaming),
        max_rows=args.max_interaction_rows,
        organic_filter=args.organic_filter,
        min_played_ratio_pct=args.min_played_ratio_pct,
    )
    return {event.raw_item_id for events in user_events.values() for event in events}


def resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return resolve_path(path)


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    main()
