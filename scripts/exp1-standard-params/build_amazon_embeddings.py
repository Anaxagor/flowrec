"""Build content embeddings for the exp1 Amazon datasets.

Encodes item metadata text (title, categories, description, ...) for Beauty
2014, Instruments 2018, and Beauty 2023 with the embedder configured in
``configs/exp1-standard_params/semantic_ids.yaml`` and saves one
``ItemEmbeddingStore`` per dataset to
``datasets/embeddings/exp1-standard_params/<dataset>_content_embeddings.pt``.
Items are restricted to the interaction catalog (verified purchases, k-core
filtered with the dataset config's minimum user/item interactions), so the
stores feed straight into scripts/build_rqvae_semantic_ids.py.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.data.preprocessors import (
    apply_k_core_filter_frame,
    collect_item_ids,
    count_interactions,
    load_interactions_frame,
)
from flowrec.item_representation.content_embeddings import (
    AmazonItemTextBuilder,
    ContentEmbeddingBuilder,
    HashingTextEmbedder,
    HFTextEmbedder,
)
from scripts import sasrec_config_runner as runner


DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp1-standard_params"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "datasets" / "embeddings" / "exp1-standard_params"
)
SEMANTIC_CONFIG_NAME = "semantic_ids.yaml"
AMAZON_DATA_CONFIG_NAMES = ("dataset_amazon.yaml", "dataset_amazon_2023.yaml")
# The exp1 dataset configs express "verified purchases only" as this pandas
# query; load_interactions_frame implements the same filter natively.
VERIFIED_ONLY_CRITERION = "verified_purchase == True"


@dataclass(frozen=True)
class EmbeddingTarget:
    name: str
    reviews_path: Path
    metadata_path: Path
    reviews_item_key: str
    metadata_item_key: str
    min_user_interactions: int
    min_item_interactions: int
    verified_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build content-embedding stores for the exp1 Amazon datasets "
            "(Beauty 2014, Instruments 2018, Beauty 2023) from item metadata."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset keys (any mix of beauty, instruments, beauty_2023) or "
            "'all'; defaults to every configured Amazon dataset."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embedder",
        choices=("hf", "hashing"),
        default="hf",
        help="hashing is a dependency-free debug embedder; hf is the real one.",
    )
    parser.add_argument("--device", type=str, default=None, help="Runtime override.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to embedding_batch_size from semantic_ids.yaml.",
    )
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--hashing-dim", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    semantic_config = runner.load_yaml(config_dir / SEMANTIC_CONFIG_NAME)
    targets = collect_targets(config_dir, args.datasets)
    validate_inputs(targets)
    output_dir = runner.resolve_path(args.output_dir, base=PROJECT_ROOT)

    text_builder = build_text_builder(semantic_config)
    embedder = build_embedder(args, semantic_config)
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else semantic_config.get("embedding_batch_size", 512)
    )

    for target in targets:
        allowed_items = allowed_item_ids(target)
        builder = ContentEmbeddingBuilder(
            text_builder=text_builder,
            embedder=embedder,
            item_key=target.metadata_item_key,
        )
        result = builder.build_from_metadata(
            metadata_path=target.metadata_path,
            batch_size=batch_size,
            save_texts=False,
            allowed_item_ids=allowed_items,
        )
        store = result.store
        output_path = embeddings_output_path(output_dir, target.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(output_path)

        summary = {
            "dataset": target.name,
            "output_path": str(output_path),
            "metadata_path": str(target.metadata_path),
            "reviews_path": str(target.reviews_path),
            "num_items": store.num_items,
            "embedding_dim": store.embedding_dim,
            "allowed_item_count": len(allowed_items),
            "num_records_read": result.num_records_read,
            "num_empty_texts": result.num_empty_texts,
            "num_duplicate_items": result.num_duplicate_items,
            "num_filtered_items": result.num_filtered_items,
            "embedder": args.embedder,
            "metadata": store.metadata,
        }
        runner.write_json(output_path.with_suffix(".summary.json"), summary)
        print(
            f"[{target.name}] saved {store.num_items} x {store.embedding_dim} "
            f"embeddings to {output_path}"
        )


def collect_targets(
    config_dir: Path,
    requested: list[str] | None,
) -> list[EmbeddingTarget]:
    explicit: list[str] | None = None
    if requested is not None and "all" not in requested:
        explicit = list(dict.fromkeys(requested))

    targets: list[EmbeddingTarget] = []
    available: list[str] = []
    for config_name in AMAZON_DATA_CONFIG_NAMES:
        dataset_config = runner.load_yaml(config_dir / config_name)
        shared_min_user = int(dataset_config.get("min_user_interactions", 1))
        shared_min_item = int(dataset_config.get("min_item_interactions", 1))
        shared_criterion = dataset_config.get("positive_criterion")
        for name, entry in dict(dataset_config.get("datasets", {})).items():
            available.append(name)
            if explicit is not None and name not in explicit:
                continue
            if "metadata_path" not in entry:
                raise KeyError(
                    f"Dataset {name!r} does not define metadata_path; content "
                    "embeddings require item metadata."
                )
            item_key = str(entry.get("item_key", "asin"))
            criterion = entry.get("positive_criterion", shared_criterion)
            targets.append(
                EmbeddingTarget(
                    name=name,
                    reviews_path=runner.resolve_path(
                        entry["reviews_path"], base=PROJECT_ROOT
                    ),
                    metadata_path=runner.resolve_path(
                        entry["metadata_path"], base=PROJECT_ROOT
                    ),
                    reviews_item_key=item_key,
                    metadata_item_key=str(entry.get("metadata_item_key", item_key)),
                    min_user_interactions=shared_min_user,
                    min_item_interactions=shared_min_item,
                    verified_only=criterion == VERIFIED_ONLY_CRITERION,
                )
            )

    if explicit is not None:
        unknown = [name for name in explicit if name not in available]
        if unknown:
            raise KeyError(f"Unknown dataset(s) {unknown}; available: {available}")
    if not targets:
        raise ValueError("No datasets selected.")
    return targets


def validate_inputs(targets: list[EmbeddingTarget]) -> None:
    missing = [
        path
        for target in targets
        for path in (target.reviews_path, target.metadata_path)
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in dict.fromkeys(missing))
        raise FileNotFoundError(
            "Missing local Amazon input files. Download/copy them before "
            f"building embeddings:\n{formatted}"
        )


def allowed_item_ids(target: EmbeddingTarget) -> set[str]:
    """Items surviving the same filters that define the training catalog."""

    interactions = load_interactions_frame(
        target.reviews_path,
        item_key=target.reviews_item_key,
        use_verified_only=target.verified_only,
    )
    filtered = apply_k_core_filter_frame(
        interactions,
        min_user_interactions=target.min_user_interactions,
        min_item_interactions=target.min_item_interactions,
    )
    items = collect_item_ids(filtered)
    if not items:
        raise ValueError(
            f"Dataset {target.name!r} has no items left after filtering."
        )
    print(
        f"[{target.name}] interactions {count_interactions(interactions)} -> "
        f"{count_interactions(filtered)} after k-core; {len(items)} items"
    )
    return items


def build_text_builder(semantic_config: Mapping[str, Any]) -> AmazonItemTextBuilder:
    return AmazonItemTextBuilder(
        **dict(semantic_config.get("text_builder", {})),
        max_chars=semantic_config.get("text_max_chars"),
    )


def build_embedder(
    args: argparse.Namespace,
    semantic_config: Mapping[str, Any],
) -> Any:
    normalize = bool(semantic_config.get("hf_normalize", False))
    if args.embedder == "hashing":
        return HashingTextEmbedder(dim=args.hashing_dim, normalize=normalize)
    return HFTextEmbedder(
        model_name=str(
            semantic_config.get(
                "hf_model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
        ),
        device=args.device,
        max_length=int(semantic_config.get("hf_max_length", 256)),
        normalize=normalize,
        pooling=semantic_config.get("pooling", "masked_mean"),
        torch_dtype=args.torch_dtype,
        show_progress=True,
    )


def embeddings_output_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"{dataset_name}_content_embeddings.pt"


if __name__ == "__main__":
    main()
