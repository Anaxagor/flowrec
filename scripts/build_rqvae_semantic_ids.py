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

from flowrec.item_representation.base import ItemEmbeddingStore
from flowrec.item_representation.semantic_id import (
    RQVAEConfig,
    RQVAEItemTokenizer,
    RQVAEItemTokenizerConfig,
    RQVAETrainConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RQ-VAE tokenizer and export FlowRec semantic item IDs."
    )
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=None)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--hidden-dims",
        default="input,32",
        help="Comma-separated hidden dims. Use 'input' for the embedding dimension.",
    )
    parser.add_argument("--num-codebooks", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-batch-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--init-codebooks-from-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dead-code-reset-interval",
        type=int,
        default=1,
        help="Use 0 to disable full-dataset dead-code resets.",
    )

    parser.add_argument("--normalize", choices=("none", "l2", "standard"), default="none")
    parser.add_argument(
        "--collision-policy",
        choices=("none", "append", "random_append"),
        default="random_append",
    )
    parser.add_argument("--level-offset-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-token-offset", type=int, default=None)
    parser.add_argument("--collision-suffix-size", type=int, default=512)
    parser.add_argument("--collision-seed", type=int, default=42)
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_path = resolve_path(args.embeddings_path)
    output_dir = resolve_path(args.output_dir)

    store = ItemEmbeddingStore.load(embeddings_path)
    if args.max_items is not None:
        store = ItemEmbeddingStore(
            item_ids=store.item_ids[: args.max_items],
            embeddings=store.embeddings[: args.max_items],
            texts=store.texts[: args.max_items] if store.texts is not None else None,
            metadata={**store.metadata, "max_items": args.max_items},
        )

    hidden_dims = parse_hidden_dims(args.hidden_dims, input_dim=store.embedding_dim)
    dead_code_reset_interval = (
        None if args.dead_code_reset_interval <= 0 else args.dead_code_reset_interval
    )
    gradient_clip_norm = args.gradient_clip_norm
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        gradient_clip_norm = None

    tokenizer = RQVAEItemTokenizer(
        RQVAEItemTokenizerConfig(
            rqvae=RQVAEConfig(
                input_dim=store.embedding_dim,
                latent_dim=args.latent_dim,
                hidden_dims=hidden_dims,
                num_codebooks=args.num_codebooks,
                codebook_size=args.codebook_size,
                commitment_weight=args.commitment_weight,
                dropout=args.dropout,
                use_batch_norm=args.use_batch_norm,
            ),
            train=RQVAETrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                optimizer=args.optimizer,
                val_ratio=args.val_ratio,
                num_workers=args.num_workers,
                drop_last=args.drop_last,
                seed=args.seed,
                gradient_clip_norm=gradient_clip_norm,
                device=args.device,
                init_codebooks_from_data=args.init_codebooks_from_data,
                dead_code_reset_interval=dead_code_reset_interval,
            ),
            normalize=args.normalize,
            collision_policy=args.collision_policy,
            level_offset_tokens=args.level_offset_tokens,
            collision_token_offset=args.collision_token_offset,
            collision_suffix_size=args.collision_suffix_size,
            collision_seed=args.collision_seed,
            encode_batch_size=args.encode_batch_size,
        )
    )
    tokenizer.fit(store.embeddings, store.item_ids)
    tokenizer.save(output_dir)

    if tokenizer.semantic_id_store is None:
        raise RuntimeError("Tokenizer did not produce semantic IDs.")

    summary = {
        "embeddings_path": embeddings_path,
        "output_dir": output_dir,
        "semantic_id_path": output_dir / "semantic_ids.json",
        "num_items": store.num_items,
        "embedding_dim": store.embedding_dim,
        "semantic_id_length": tokenizer.semantic_id_store.sid_length,
        "unique_semantic_ids": tokenizer.semantic_id_store.unique_code_count,
        "collision_count": tokenizer.semantic_id_store.collision_count,
        "collision_stats": (
            tokenizer.collision_stats.to_dict() if tokenizer.collision_stats is not None else None
        ),
        "rqvae": asdict(tokenizer.config.rqvae),
        "train": asdict(tokenizer.config.train),
        "normalize": tokenizer.config.normalize,
        "collision_policy": tokenizer.config.collision_policy,
    }
    write_json(output_dir / "semantic_id_summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def parse_hidden_dims(value: str, *, input_dim: int) -> tuple[int, ...]:
    dims: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part == "input":
            dims.append(input_dim)
        else:
            dims.append(int(part))
    if not dims:
        raise ValueError("--hidden-dims must contain at least one dimension")
    return tuple(dims)


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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    return value


if __name__ == "__main__":
    main()
