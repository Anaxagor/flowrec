#!/usr/bin/env python3
"""Compare Beauty recommendations from exp2 GPT-2/TIGER before and after SFT.

Examples (from the repository root):
  python scripts/exp6_result_analysis/beauty_sft_analysis.py --mode discover
  python scripts/exp6_result_analysis/beauty_sft_analysis.py --device cuda
  python scripts/exp6_result_analysis/beauty_sft_analysis.py --mode analyze

Generation is cached per checkpoint. Analyze mode only needs NumPy, pandas,
Matplotlib and the cache; it does not import PyTorch or reload models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from analysis import analyze_bundles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("discover", "generate", "analyze", "all"), default="all")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, help="Explicit JSON list of run specs; see --mode discover")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4, help="PyTorch intra-op threads on CPU")
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-users", type=int, default=None,
                        help="Deterministic shared user sample. Omit for full test; use a separate output directory.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=500,
                        help="Paired user bootstrap of delta NDCG, not training-seed uncertainty; 0 disables")
    parser.add_argument("--force", action="store_true", help="Regenerate caches even if their fingerprints match")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1 or args.top_k < 1 or args.num_beams < args.top_k:
        parser.error("Require batch-size > 0 and num-beams >= top-k > 0")
    if not args.ks or min(args.ks) < 1 or max(args.ks) > args.top_k:
        parser.error("Every k must be positive and <= top-k")
    if args.max_users is not None and args.max_users < 1:
        parser.error("max-users must be positive")
    if args.bootstrap_samples < 0:
        parser.error("bootstrap-samples must be nonnegative")
    args.repo_root = args.repo_root.resolve()
    args.output_dir = (args.output_dir or args.repo_root / "artifacts/exp6_result_analysis/beauty").resolve()
    return args


def _json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _export_tables(tables, output):
    folder = output / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if name in ("items", "users"):
            # CSV remains usable without a Parquet engine, and gzip keeps full
            # per-user/per-item diagnostics small enough to share.
            frame.to_csv(folder / f"{name}.csv.gz", index=False, compression="gzip")
        else:
            frame.to_csv(folder / f"{name}.csv", index=False)


def _hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _metric_validation(specs, tables, provenance):
    rows = []
    for spec, source in zip(specs, provenance):
        if source.get("num_users") != source.get("total_test_users"):
            continue
        for _, row in tables["summary"][tables["summary"].run_id == spec["run_id"]].iterrows():
            for column, original in (("ndcg", "NDCG"), ("hr", "HR"), ("coverage", "Coverage"), ("arp", "ARP")):
                expected = spec.get("recorded_test_metrics", {}).get(f"{original}@{row['k']}")
                if expected is not None:
                    rows.append({"run_id": spec["run_id"], "k": row["k"], "metric": original,
                                 "recorded": expected, "recomputed": row[column],
                                 "difference": row[column] - expected})
    return pd.DataFrame(rows, columns=["run_id", "k", "metric", "recorded", "recomputed", "difference"])


def _validate_cache_identity(spec, source):
    for key in ("run_id", "model", "stage", "seed", "baseline_run_id", "checkpoint",
                "init_checkpoint", "data_config", "best_trial", "recorded_test_metrics"):
        if source.get(key) != spec.get(key):
            raise ValueError(f"Manifest does not match cache provenance for {spec['run_id']}: {key}; regenerate with --force")


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    manifest = args.manifest or output / "runs.json"
    if args.mode == "analyze":
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {manifest}; run --mode generate first")
        specs = json.loads(manifest.read_text(encoding="utf-8"))
    elif args.manifest:
        specs = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        from inference import discover_runs
        specs = discover_runs(args.repo_root, args.seeds)
    if not isinstance(specs, list) or not specs:
        raise ValueError("Manifest must be a nonempty list of run specifications")
    _json_write(output / "runs.json", specs)
    for spec in specs:
        print(f"{spec['run_id']}: {spec['checkpoint']}", flush=True)
    if args.mode == "discover":
        print(f"\nManifest: {output / 'runs.json'}", flush=True)
        return 0
    if args.mode in ("generate", "all"):
        from inference import generate_run
        import torch
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(args.cpu_threads)
        print(f"Inference device: {device}", flush=True)
        for spec in specs:
            generate_run(spec, repo_root=args.repo_root, output_dir=cache,
                         device=device, batch_size=args.batch_size,
                         num_beams=args.num_beams, top_k=args.top_k,
                         max_users=args.max_users, sample_seed=args.sample_seed,
                         force=args.force)
        if args.mode == "generate":
            print(f"\nRecommendation cache: {cache}", flush=True)
            return 0
    bundles, provenance = {}, []
    for spec in specs:
        rid = spec["run_id"]
        with np.load(cache / f"{rid}.npz", allow_pickle=False) as data:
            bundles[rid] = {key: data[key] for key in data.files}
        sidecar = cache / f"{rid}.json"
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing cache provenance: {sidecar}")
        source = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_cache_identity(spec, source)
        if source.get("npz_sha256") != _hash(cache / f"{rid}.npz") or source.get("catalog_sha256") != _hash(cache / "catalog.csv"):
            raise ValueError(f"Cache contents changed since generation: {rid}")
        provenance.append(source)
    for key in ("num_beams", "top_k", "constrain_valid_sequences", "exclude_seen_items", "max_users", "sample_seed"):
        values = [p["fingerprint_inputs"][key] for p in provenance]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Incomparable generation settings across caches: {key}")
    catalog = pd.read_csv(cache / "catalog.csv", dtype={"item_id": str}, keep_default_na=False)
    print("Computing paired ranking, exposure and catalog diagnostics…", flush=True)
    tables = analyze_bundles(specs, bundles, catalog, ks=tuple(args.ks),
                            bootstrap_samples=args.bootstrap_samples, seed=args.sample_seed)
    tables["metric_validation"] = _metric_validation(specs, tables, provenance)
    _export_tables(tables, output)
    n_users = len(next(iter(bundles.values()))["user_ids"])
    metadata = {
        "dataset": "Beauty", "split": "test", "num_users": n_users, "catalog_size": len(catalog),
        "run_specs": specs, "provenance": provenance, "bootstrap_samples": args.bootstrap_samples,
        "ks": sorted(set(args.ks)),
        "warnings": [
            "Три SFT seed каждой модели используют один исходный чекпойнт: это не три независимых baseline.",
            "Coverage — совокупный охват каталога; он не измеряет разнообразие внутри списка пользователя.",
            "Энтропия exposure относится к частотам показов; вероятностная энтропия модели не измеряется.",
            "Связь NDCG и coverage описательная: различаются исходные модели и протоколы SFT; причинность не установлена.",
            "Bootstrap-интервалы отражают вариативность по тестовым пользователям при фиксированных чекпойнтах, не по seed обучения.",
            "Популярность рассчитана по тренировочным взаимодействиям; test используется только для оценки.",
            "Группы популярности фиксированы по процентилям тренировочной частоты; одинаковые частоты не разделяются, поэтому размеры групп могут отличаться от 50/30/20%.",
        ],
        "definitions": {
            "coverage": "unique recommended catalog items / catalog size",
            "arp": "mean training interaction count across valid recommendation slots",
            "mean_pairwise_overlap": "exact mean |topK(u) intersect topK(v)| / K over distinct users",
            "mean_list_overlap": "mean |topK_before(u) intersect topK_after(u)| / K",
            "new_item_share": "fraction of a user's slots occupied by items never recommended at this K by the baseline",
            "accuracy_attribution": "additive decomposition of mean delta NDCG by whether the held-out target was globally covered before SFT",
        },
    }
    # In analyze mode generation settings come from the cache, not CLI defaults.
    max_users_values = [p.get("fingerprint_inputs", {}).get("max_users") for p in provenance]
    metadata["subsample"] = any(p["num_users"] < p["total_test_users"] for p in provenance)
    metadata["max_users"] = next((v for v in max_users_values if v is not None), None) if metadata["subsample"] else None
    metadata["generation_settings"] = {key: provenance[0]["fingerprint_inputs"].get(key)
                                       for key in ("num_beams", "top_k", "device", "dtype", "constrain_valid_sequences", "exclude_seen_items")}
    if metadata["generation_settings"]["num_beams"] != 20:
        metadata["warnings"].append("Число beams отличается от исходной оценки (20); сравнение с исходными CSV включает изменение декодирования.")
    if metadata["subsample"]:
        metadata["warnings"].insert(0, f"Анализ подвыборки ({n_users} пользователей): coverage не сопоставим с полными CSV экспериментов.")
    _json_write(output / "analysis_metadata.json", metadata)
    from plots import render_report
    report = render_report(tables, output, metadata)
    print(tables["summary"][["run_id", "k", "ndcg", "coverage", "arp"]].to_string(index=False), flush=True)
    print(f"\nReport: {report}\nTables: {output / 'tables'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
