"""Build Amazon content embeddings for the final global-time training catalogs.

Supports Beauty 2014, Instruments 2018 and Beauty 2023. Text construction and
the pretrained encoder settings are shared with exp1; only catalog selection
uses the global-time datamodule. Outputs never default to the leave-one-out
embedding directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.item_representation.content_embeddings import ContentEmbeddingBuilder
from scripts import global_time_representation as shared
from scripts import sasrec_config_runner as runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=shared.DEFAULT_CONFIG_DIR)
    parser.add_argument("--semantic-config", type=Path, default=shared.DEFAULT_SEMANTIC_CONFIG)
    parser.add_argument("--datasets", nargs="+", help="beauty, instruments, beauty_2023, or all (default)")
    parser.add_argument("--output-dir", type=Path, default=shared.DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=("hf", "hashing"), default="hf")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--hashing-dim", type=int, default=384)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    semantic_path = runner.resolve_path(args.semantic_config, base=PROJECT_ROOT)
    semantic_config = runner.load_yaml(semantic_path)
    output_dir = runner.resolve_path(args.output_dir, base=PROJECT_ROOT)
    # Reuse exp1's Amazon-only selection and input validation.
    amazon_targets = shared.embedding_script.collect_targets(config_dir, args.datasets)
    shared.embedding_script.validate_inputs(amazon_targets)
    targets = shared.collect_targets(config_dir, output_dir, [t.name for t in amazon_targets])
    text_builder = shared.embedding_script.build_text_builder(semantic_config)
    embedder = shared.embedding_script.build_embedder(args, semantic_config)
    batch_size = args.batch_size if args.batch_size is not None else int(semantic_config.get("embedding_batch_size", 512))

    for target in targets:
        data = shared.build_catalog_datamodule(target, seed=args.seed)
        allowed_items = data.item_ids_in_data()
        provenance = shared.catalog_metadata(data)
        metadata_path = data.cfg.metadata_path
        metadata_item_key = data.cfg.metadata_item_key
        del data  # Release sequence frames before encoding a large catalog.
        builder = ContentEmbeddingBuilder(
            text_builder=text_builder, embedder=embedder, item_key=metadata_item_key
        )
        result = builder.build_from_metadata(
            metadata_path=metadata_path,
            batch_size=batch_size,
            save_texts=False,
            allowed_item_ids=allowed_items,
        )
        store = shared.restrict_embeddings(result.store, allowed_items)
        store.metadata.update(provenance)
        output_path = shared.embedding_script.embeddings_output_path(output_dir, target.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(output_path)
        runner.write_json(output_path.with_suffix(".summary.json"), {
            "dataset": target.name,
            "output_path": str(output_path),
            "semantic_config_path": str(semantic_path),
            "semantic_config": dict(semantic_config),
            "num_items": store.num_items,
            "embedding_dim": store.embedding_dim,
            "allowed_item_count": len(allowed_items),
            "num_records_read": result.num_records_read,
            "num_empty_texts": result.num_empty_texts,
            "num_duplicate_items": result.num_duplicate_items,
            "num_filtered_items": result.num_filtered_items,
            "embedder": args.embedder,
            "metadata": store.metadata,
        })
        print(f"[{target.name}] saved {store.num_items} x {store.embedding_dim} embeddings to {output_path}")


if __name__ == "__main__":
    main()
