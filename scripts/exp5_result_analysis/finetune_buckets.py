#!/usr/bin/env python3
"""Matched popularity/history diagnostics for the workbook's exp4 checkpoints.

Stages: discover (metadata only), prepare (CPU), generate (GPU), analyze (CPU).
Array tasks select a dataset or a run by its stable manifest index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from bucket_manifest import build_manifest

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("beauty", "instruments", "yambda_50m_listens", "yambda_500m_listens")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("discover", "prepare", "generate", "analyze"), required=True)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Compatibility aliases for the same current primary SFT model, counted once")
    p.add_argument("--datasets", nargs="+", choices=DATASETS,
                   help="Defaults to all datasets for discover, otherwise datasets present in manifest")
    p.add_argument("--dataset-index", type=int, help="Zero-based index in --datasets (prepare/analyze arrays)")
    p.add_argument("--run-index", type=int, help="Zero-based index in the complete manifest (generate array)")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--num-beams", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--history-edges", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200])
    p.add_argument("--max-users", type=int, help="Shared deterministic smoke-test sample; omit for full test")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--bootstrap-samples", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    if min(args.batch_size, args.cpu_threads, args.top_k) < 1 or args.num_beams < args.top_k:
        p.error("Require positive batch-size/cpu-threads and num-beams >= top-k > 0")
    if not args.ks or min(args.ks) < 1 or max(args.ks) > args.top_k:
        p.error("ks must be between 1 and top-k")
    if args.bootstrap_samples < 0 or (args.max_users is not None and args.max_users < 1):
        p.error("bootstrap-samples must be nonnegative; max-users must be positive")
    if args.datasets is not None and len(set(args.datasets)) != len(args.datasets):
        p.error("datasets must be unique")
    if args.run_index is not None and args.mode != "generate":
        p.error("run-index is only used in generate mode")
    args.repo_root = args.repo_root.resolve()
    args.output_dir = (args.output_dir or args.repo_root / "artifacts/exp6_result_analysis/finetune_buckets").resolve()
    return args


def load_runs(args):
    manifest = args.manifest or args.output_dir / "runs.json"
    if args.mode == "discover":
        args.datasets = args.datasets or list(DATASETS)
        specs = build_manifest(args.repo_root, seeds=tuple(args.seeds))
        specs = [s for s in specs if s["dataset"] in args.datasets]
        write_json(manifest, specs)
    else:
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {manifest}; run --mode discover first")
        specs = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("Manifest must be a nonempty list")
    ids = [s["run_id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate run IDs")
    args.datasets = args.datasets or list(dict.fromkeys(s["dataset"] for s in specs))
    if args.dataset_index is not None and not 0 <= args.dataset_index < len(args.datasets):
        raise ValueError("dataset-index out of range")
    return specs


def analyze_dataset(specs, folder, args):
    import numpy as np
    import pandas as pd
    from bucket_metrics import analyze_buckets

    prepared = folder / "prepared"
    catalog = pd.read_csv(prepared / "catalog.csv", dtype={"item_id": str}, keep_default_na=False)
    catalog_hash = sha256(prepared / "catalog.csv")
    prepared_metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    prepared_hash = sha256(prepared / "metadata.json")
    bundles, provenance = {}, []
    for spec in specs:
        rid = spec["run_id"]
        path = folder / "cache" / f"{rid}.npz"
        sidecar = path.with_suffix(".json")
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing selected run {rid}; generate its cache before analysis")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in ("run_id", "dataset", "model", "stage", "seed", "checkpoint", "baseline_run_id", "sft_run_id",
                    "init_checkpoint", "data_config", "selection_labels"):
            if meta.get(key) != spec.get(key):
                raise ValueError(f"{rid}: manifest/cache mismatch for {key}")
        if meta.get("npz_sha256") != sha256(path) or meta.get("catalog_sha256") != catalog_hash:
            raise ValueError(f"{rid}: cache contents/catalog changed; regenerate")
        generation = meta.get("fingerprint_inputs", {})
        if (generation.get("prepared_fingerprint") != prepared_metadata["fingerprint"]
                or generation.get("prepared_metadata_sha256") != prepared_hash):
            raise ValueError(f"{rid}: cache comes from a different prepared test split; regenerate")
        with np.load(path, allow_pickle=False) as data:
            bundles[rid] = {key: data[key] for key in data.files}
        provenance.append(meta)
    settings = ("num_beams", "top_k", "constrain_valid_sequences", "exclude_seen_items", "dtype")
    for key in settings:
        values = [p["fingerprint_inputs"].get(key) for p in provenance]
        if any(v != values[0] for v in values[1:]):
            raise ValueError(f"Generation settings differ across runs: {key}")
    tables = analyze_buckets(specs, bundles, catalog, ks=tuple(args.ks),
                            history_edges=tuple(args.history_edges),
                            bootstrap_samples=args.bootstrap_samples, seed=args.sample_seed)
    # Full-test reference metrics are a diagnostic, never a substitution for rerun predictions.
    validation = []
    for spec, meta in zip(specs, provenance):
        if meta.get("num_users") != meta.get("total_test_users"):
            continue
        for row in tables["summary"].to_dict("records"):
            if row["run_id"] != spec["run_id"]:
                continue
            for field, metric in (("hr", "HR"), ("ndcg", "NDCG"), ("coverage", "Coverage"), ("arp", "ARP")):
                original = spec.get("recorded_test_metrics", {}).get(f"{metric}@{row['k']}")
                if original is not None:
                    validation.append(dict(run_id=spec["run_id"], k=row["k"], metric=metric,
                                           recorded=original, recomputed=row[field], delta=row[field] - original))
    tables["metric_validation"] = pd.DataFrame(validation, columns=["run_id", "k", "metric", "recorded", "recomputed", "delta"])
    table_dir = folder / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        suffix = ".csv.gz" if name in {"items", "item_deltas", "catalog_buckets"} else ".csv"
        table.to_csv(table_dir / (name + suffix), index=False)
    metadata = {
        "dataset": specs[0]["dataset"], "split": "test", "runs": specs,
        "ks": args.ks, "history_edges": args.history_edges,
        "bootstrap_samples": args.bootstrap_samples,
        "generation": {key: provenance[0]["fingerprint_inputs"].get(key) for key in settings},
        "provenance": provenance,
        "subsample": any(p["num_users"] < p["total_test_users"] for p in provenance),
        "notes": ["Popularity uses training interactions only; bucket boundaries are shared by all runs.",
                  "GFlow selections use seed42; comparisons against SFT use the same seed.",
                  "Coverage/entropy describe aggregate exposure, not semantic intra-list diversity.",
                  "Selection was inherited from the workbook; this is descriptive test analysis, not new unbiased model selection.",
                  "History includes all observed positive events before the test target, including validation; model input is clipped.",
                  "No items are removed or reranked to compute bucket-specific accuracy."],
    }
    write_json(folder / "analysis_metadata.json", metadata)
    if not args.no_plots:
        from bucket_report import render_report
        render_report(tables, folder, metadata)
    print(f"{specs[0]['dataset']}: {len(specs)} runs; tables: {table_dir}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    specs = load_runs(args)
    if args.mode == "discover":
        for index, spec in enumerate(specs):
            print(f"{index:2d} {spec['run_id']} available={spec['available']} {spec['checkpoint']}")
        print(f"total: {len(specs)}")
        return 0
    selected_datasets = args.datasets if args.dataset_index is None else [args.datasets[args.dataset_index]]
    if args.mode == "generate":
        selected = ([s for s in specs if s["dataset"] in selected_datasets] if args.run_index is None
                    else specs[args.run_index:args.run_index + 1])
        if not selected or (args.run_index is not None and args.run_index < 0):
            raise ValueError("run-index out of range")
        import torch
        from bucket_inference import generate_prepared_run
        torch.set_num_threads(args.cpu_threads)
        for spec in selected:
            if spec["dataset"] not in selected_datasets:
                raise ValueError("Selected manifest run is outside --datasets")
            folder = args.output_dir / spec["dataset"]
            generate_prepared_run(spec, repo_root=args.repo_root, prepared_dir=folder / "prepared",
                                  output_dir=folder / "cache", device=args.device,
                                  batch_size=args.batch_size, num_beams=args.num_beams,
                                  top_k=args.top_k, force=args.force)
        return 0
    for dataset in selected_datasets:
        runs = [s for s in specs if s["dataset"] == dataset]
        if not runs:
            raise ValueError(f"No manifest runs for {dataset}")
        folder = args.output_dir / dataset
        if args.mode == "prepare":
            import torch
            from bucket_inference import prepare_dataset
            torch.set_num_threads(args.cpu_threads)
            prepare_dataset(runs[0], repo_root=args.repo_root, output_dir=folder / "prepared",
                            max_users=args.max_users, sample_seed=args.sample_seed, force=args.force)
        else:
            analyze_dataset(runs, folder, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
