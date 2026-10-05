"""Train and save RQ-VAE tokenizers on the final temporal training catalogs.

Amazon uses the global-time content embedding stores; Yambda and VK-LSVD use
the existing audio/video embedding sources. All families derive their item
scope from a fully set-up datamodule. RQ-VAE, normalization, collision handling
and serialization are the same as exp1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.item_representation.base import ItemEmbeddingStore, SemanticIDStore
from flowrec.item_representation.semantic_id import RQVAEItemTokenizer
from scripts import global_time_representation as shared
from scripts import sasrec_config_runner as runner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=shared.DEFAULT_CONFIG_DIR)
    parser.add_argument("--semantic-config", type=Path, default=shared.DEFAULT_SEMANTIC_CONFIG)
    parser.add_argument("--datasets", nargs="+", help="Configured dataset keys, or all (default)")
    parser.add_argument("--embeddings-dir", type=Path, default=shared.DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=shared.DEFAULT_SEMANTIC_IDS_DIR)
    parser.add_argument("--report-path", type=Path, default=shared.DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--reuse-semantic-ids-dir", type=Path,
        help="Export the final catalog's existing SIDs instead of fitting tokenizers; codebooks retain their original training scope",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Debug override")
    parser.add_argument("--max-items", type=int, default=None, help="Debug: truncate only after catalog selection")
    args = parser.parse_args(argv)
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.reuse_semantic_ids_dir is not None and (args.max_items is not None or args.epochs is not None):
        parser.error("--reuse-semantic-ids-dir cannot be combined with --max-items or --epochs")
    return args


def load_embedding_store(target, args: argparse.Namespace) -> tuple[ItemEmbeddingStore, int]:
    data = shared.build_catalog_datamodule(target, seed=args.seed)
    item_ids = data.item_ids_in_data()
    provenance = shared.catalog_metadata(data)
    if target.family == "amazon":
        store = ItemEmbeddingStore.load(target.embeddings_path)
    elif target.family == "yambda":
        store = shared.tokenizer_script.load_yambda_audio_embedding_store_for_datamodule(data)
    else:
        store = shared.tokenizer_script.load_vk_lsvd_item_embedding_store_for_datamodule(data)
    del data
    store = shared.restrict_embeddings(store, item_ids)
    store.metadata.update(provenance)
    if args.max_items is not None:
        store = ItemEmbeddingStore(
            item_ids=store.item_ids[:args.max_items],
            embeddings=store.embeddings[:args.max_items],
            texts=store.texts[:args.max_items] if store.texts is not None else None,
            metadata={**store.metadata, "max_items": args.max_items},
        )
    return store, len(item_ids)


def export_existing_ids(target, *, source_dir: Path, output_dir: Path, seed: int) -> dict:
    source_path = source_dir / target.name / "semantic_ids.json"
    dataset_output_dir = output_dir / target.name
    if source_path.resolve() == (dataset_output_dir / "semantic_ids.json").resolve():
        raise ValueError("Reuse output must differ from the source SID directory")
    if (dataset_output_dir / "model.pt").exists():
        raise ValueError("Reuse output already contains a trained tokenizer; choose a separate --output-dir")
    source = SemanticIDStore.load_json(source_path)
    data = shared.build_catalog_datamodule(target, seed=seed)
    item_ids = data.item_ids_in_data()
    provenance = shared.catalog_metadata(data)
    del data
    missing = item_ids - set(source.semantic_ids)
    if missing:
        raise ValueError(f"{target.name}: existing SIDs miss {len(missing)} training items; examples: {sorted(missing)[:5]}")
    store = SemanticIDStore(
        {item: tokens for item, tokens in source.semantic_ids.items() if item in item_ids},
        metadata={
            "mode": "reuse_semantic_ids",
            "source_path": str(source_path),
            "tokenizer_fit_scope": "unchanged_from_source",
            **provenance,
        },
    )
    source_metadata = source_path.with_name("semantic_id_metadata.json")
    if source_metadata.is_file():
        store.metadata["source_metadata"] = json.loads(source_metadata.read_text())
    store.save_json(dataset_output_dir / "semantic_ids.json")
    store.save_pt(dataset_output_dir / "semantic_ids.pt")
    runner.write_json(dataset_output_dir / "semantic_id_metadata.json", store.metadata)
    print(f"[{target.name}] exported {store.num_items} existing SIDs to {dataset_output_dir}")
    return {
        "dataset": target.name,
        "family": target.family,
        "mode": "reuse_semantic_ids",
        "source_path": str(source_path),
        "output_dir": str(dataset_output_dir),
        "semantic_id_path": str(dataset_output_dir / "semantic_ids.json"),
        "num_items": store.num_items,
        "num_catalog_items": len(item_ids),
        "semantic_id_length": store.sid_length,
        "unique_semantic_ids": store.unique_code_count,
        "unresolved_collision_count": store.collision_count,
        "tokenizer_fit_scope": "unchanged_from_source",
        **provenance,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = runner.resolve_path(args.config_dir, base=PROJECT_ROOT)
    semantic_path = runner.resolve_path(args.semantic_config, base=PROJECT_ROOT)
    semantic_config = runner.load_yaml(semantic_path)
    embeddings_dir = runner.resolve_path(args.embeddings_dir, base=PROJECT_ROOT)
    output_dir = runner.resolve_path(args.output_dir, base=PROJECT_ROOT)
    report_path = runner.resolve_path(args.report_path, base=PROJECT_ROOT)
    targets = shared.collect_targets(config_dir, embeddings_dir, args.datasets)
    shared.validate_catalog_inputs(targets)
    source_dir = (
        runner.resolve_path(args.reuse_semantic_ids_dir, base=PROJECT_ROOT)
        if args.reuse_semantic_ids_dir is not None else None
    )
    if source_dir is None:
        shared.tokenizer_script.validate_inputs(targets)
    else:
        for target in targets:
            source_path = source_dir / target.name / "semantic_ids.json"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
    report = {
        "script": "scripts/global-time/train_tokenizer_all.py",
        "mode": "reuse_semantic_ids" if source_dir else "train_tokenizers",
        "config_dir": str(config_dir),
        "semantic_config_path": str(semantic_path),
        "semantic_config": dict(semantic_config),
        "seed": args.seed,
        "device": args.device,
        "epochs_override": args.epochs,
        "max_items": args.max_items,
        "started_at": shared.tokenizer_script.utc_now(),
        "datasets": [],
    }
    for target in targets:
        started = time.perf_counter()
        dataset_output_dir = output_dir / target.name
        if source_dir is not None:
            entry = export_existing_ids(target, source_dir=source_dir, output_dir=output_dir, seed=args.seed)
        else:
            store, num_catalog_items = load_embedding_store(target, args)
            print(f"[{target.name}] training RQ-VAE on {store.num_items} x {store.embedding_dim} embeddings")
            tokenizer = RQVAEItemTokenizer(shared.tokenizer_script.build_tokenizer_config(
                semantic_config, input_dim=store.embedding_dim,
                seed=args.seed, device=args.device, epochs_override=args.epochs,
            ))
            tokenizer.fit(store.embeddings, store.item_ids)
            tokenizer.save(dataset_output_dir)
            entry = shared.tokenizer_script.summarize_run(
                target, tokenizer, store=store, num_catalog_items=num_catalog_items,
                dataset_output_dir=dataset_output_dir, train_seconds=time.perf_counter() - started,
            )
            entry["mode"] = "train_tokenizers"
            entry.update({key: store.metadata[key] for key in ("catalog_scope", "catalog_config", "catalog_summary")})
            print(f"[{target.name}] saved tokenizer and semantic IDs to {dataset_output_dir}")
            del tokenizer, store
        # Keep provenance next to each tokenizer, even if a report path is reused.
        runner.write_json(dataset_output_dir / "catalog.json", entry)
        report["datasets"].append(entry)
        report["finished_at"] = shared.tokenizer_script.utc_now()
        runner.write_json(report_path, report)
    print(f"Saved semantic-ID training report to {report_path}")


if __name__ == "__main__":
    main()
