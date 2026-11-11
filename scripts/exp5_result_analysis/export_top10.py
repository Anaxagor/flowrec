#!/usr/bin/env python3
"""Export test top-10 rankings and train-only item counts for historical runs.

discover freezes selection; prepare builds inputs on CPU; generate decodes one
checkpoint per GPU task; verify checks every requested export without PyTorch.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
TOP_K = 10
RANK_COLUMNS = [f"item_{rank}" for rank in range(1, TOP_K + 1)]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_manifest(runs: list[dict], repo_root: Path, *, max_users=None, sample_seed=42,
                    selection_split="test", checkpoint_kinds=("best",)) -> dict:
    from bucket_inference import data_protocol, resolve_data_config
    from top10_manifest import missing_groups

    preparations = {}
    frozen = []
    for original in runs:
        spec = dict(original)
        config = resolve_data_config(spec, repo_root, check_inputs=False)
        protocol = {"dataset": spec["dataset"], "model": spec["model"],
                    "protocol": data_protocol(config)}
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        preparation_id = f"{spec['dataset']}_{spec['model']}_{digest}"
        spec["preparation_id"] = preparation_id
        # No path migration or substitution of missing historical weights.
        spec["preserve_legacy_checkpoint_path"] = True
        frozen.append(spec)
        preparations.setdefault(preparation_id, {"preparation_id": preparation_id,
            "dataset": spec["dataset"], "model": spec["model"], "spec": spec})
    return {"schema_version": SCHEMA_VERSION, "selection_split": selection_split,
            "evaluation_split": "test", "checkpoint_kinds": list(checkpoint_kinds), "top_k": TOP_K,
            "max_users": max_users, "sample_seed": sample_seed,
            "runs": frozen, "preparations": list(preparations.values()),
            "missing_groups": missing_groups(frozen),
            "popularity_definition": "Count original positive training interactions, excluding validation/test; no duplicated prefix targets.",
            "note": "All first/second SFT runs; GFlow/CE start from first SFT. Different original data protocols have separate popularity files."}


def validate_manifest(manifest: dict) -> None:
    from top10_manifest import missing_groups, validate_run_scope
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected a version-2 from-SFT manifest; regenerate the manifest")
    if manifest.get("selection_split") not in {"validation", "test"} or manifest.get("evaluation_split") != "test":
        raise ValueError("Expected validation/test selection and test evaluation")
    if manifest.get("top_k") != TOP_K:
        raise ValueError("Expected top_k=10")
    if not manifest.get("checkpoint_kinds") or not set(manifest["checkpoint_kinds"]) <= {"best", "last"}:
        raise ValueError("Expected checkpoint_kinds containing best and/or last")
    runs, preparations = manifest.get("runs", []), manifest.get("preparations", [])
    if not runs or not preparations:
        raise ValueError("Manifest must contain runs and preparations")
    ids = [r["run_id"] for r in runs]
    prep_ids = [p["preparation_id"] for p in preparations]
    if len(ids) != len(set(ids)) or len(prep_ids) != len(set(prep_ids)):
        raise ValueError("Duplicate run/preparation IDs")
    for name in ids + prep_ids:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Unsafe artifact ID: {name!r}")
    if any(r.get("preparation_id") not in prep_ids for r in runs):
        raise ValueError("Run refers to an unknown preparation")
    if manifest.get("missing_groups") != missing_groups(runs):
        raise ValueError("Manifest completeness does not match its runs; regenerate the manifest")
    for run in runs:
        validate_run_scope(run)
        if run.get("preserve_legacy_checkpoint_path") is not True:
            raise ValueError("Original checkpoint paths must be preserved")
        if run["stage"] in {"sft", "sft2"} and run.get("checkpoint_kind", "best") not in manifest["checkpoint_kinds"]:
            raise ValueError("SFT checkpoint kind differs from the frozen manifest")
        if run.get("stage") in {"gflow", "ce"} and run.get("selection_split") != manifest["selection_split"]:
            raise ValueError("Run selection split differs from the frozen manifest")
    by_id = {run["run_id"]: run for run in runs}
    for preparation in preparations:
        spec = preparation.get("spec", {})
        if (by_id.get(spec.get("run_id")) != spec
                or spec.get("preparation_id") != preparation["preparation_id"]):
            raise ValueError("Preparation must reference a matching frozen run")


def export_popularity(prepared_dir: Path, output: Path) -> Path:
    """One row per catalog item, including zero-count items; raw IDs stay strings."""
    source = prepared_dir / "catalog.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [(row["item_id"], int(row["popularity"])) for row in csv.DictReader(handle)]
    if not rows or len({item for item, _ in rows}) != len(rows) or any(n < 0 for _, n in rows):
        raise ValueError(f"Invalid training popularity catalog: {source}")
    total = sum(n for _, n in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "train_count", "train_fraction"])
        writer.writerows((item, n, n / total if total else 0.0) for item, n in rows)
    temporary.replace(output)
    write_json(output.with_suffix(".json"), {"catalog_sha256": sha256(source),
        "csv_sha256": sha256(output), "num_items": len(rows), "total_train_interactions": total,
        "prepared_metadata_sha256": sha256(prepared_dir / "metadata.json"),
        "definition": "Original positive train interactions only, before prefix expansion; fraction is count/sum(count)."})
    return output


def export_recommendations(npz_path: Path, spec: dict, output_dir: Path) -> Path:
    import numpy as np

    source_meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    if source_meta["npz_sha256"] != sha256(npz_path):
        raise ValueError(f"Recommendation cache changed: {npz_path}")
    popularity = output_dir / "popularity" / f"{spec['preparation_id']}.csv"
    pop_meta = json.loads(popularity.with_suffix(".json").read_text(encoding="utf-8"))
    if (pop_meta["catalog_sha256"] != source_meta["catalog_sha256"]
            or pop_meta["csv_sha256"] != sha256(popularity)
            or pop_meta["prepared_metadata_sha256"] != source_meta["fingerprint_inputs"]["prepared_metadata_sha256"]):
        raise ValueError("Popularity file does not match this run's prepared test inputs")
    with popularity.open(newline="", encoding="utf-8") as handle:
        catalog = {r["item_id"] for r in csv.DictReader(handle)}
    with np.load(npz_path, allow_pickle=False) as data:
        users, targets, recommendations = data["user_ids"], data["target_item_ids"], data["recommendations"]
        if recommendations.shape != (len(users), TOP_K) or len(targets) != len(users):
            raise ValueError("Expected one target and exactly 10 recommendations per test example")
        if len(set(map(str, users))) != len(users) or len(users) != source_meta["num_users"]:
            raise ValueError("Test examples are missing or duplicated")
        output = output_dir / "recommendations" / f"{spec['run_id']}.csv.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["example_index", "user_id", "target_item_id", *RANK_COLUMNS])
                for index, (user, target, recs) in enumerate(zip(users, targets, recommendations)):
                    items = list(map(str, recs))
                    if len(set(items)) != TOP_K or not set(items).issubset(catalog):
                        raise ValueError(f"Invalid top10 for test example {index}")
                    writer.writerow([index, str(user), str(target), *items])
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(output.with_suffix(".json"), {**spec, "num_examples": len(users),
        "total_test_examples": source_meta["total_test_users"], "top_k": TOP_K,
        "csv_sha256": sha256(output), "cache_sha256": source_meta["npz_sha256"],
        "popularity_file": str(popularity.relative_to(output_dir)),
        "popularity_sha256": pop_meta["csv_sha256"], "generation": source_meta["fingerprint_inputs"]})
    print(f"Exported {len(users)} test examples: {output}", flush=True)
    return output


def active_missing_groups(manifest: dict) -> list[str]:
    """Every requested dataset/model/stage must be represented."""
    if manifest.get("schema_version") == SCHEMA_VERSION:
        from top10_manifest import missing_groups
        return missing_groups(manifest["runs"])
    return list(manifest.get("missing_groups", []))


def verify_exports(manifest: dict, output_dir: Path) -> None:
    errors = [f"No source runs for {group}" for group in active_missing_groups(manifest)]
    for spec in manifest["runs"]:
        path = output_dir / "recommendations" / f"{spec['run_id']}.csv.gz"
        try:
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if any(meta.get(key) != value for key, value in spec.items()):
                raise ValueError("manifest/export metadata mismatch")
            popularity = output_dir / meta["popularity_file"]
            if sha256(path) != meta["csv_sha256"] or sha256(popularity) != meta["popularity_sha256"]:
                raise ValueError("export/popularity checksum mismatch")
            prepared_path = output_dir / "prepared" / spec["preparation_id"] / "metadata.json"
            prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
            if sha256(prepared_path) != meta["generation"]["prepared_metadata_sha256"]:
                raise ValueError("export uses a different test preparation")
            frozen = prepared["fingerprint_inputs"]
            if (frozen["max_users"] != manifest.get("max_users")
                    or frozen["sample_seed"] != manifest.get("sample_seed", 42)):
                raise ValueError("export test subset differs from the frozen manifest")
            if meta["num_examples"] != prepared["num_users"]:
                raise ValueError("export example count differs from test preparation")
            if manifest.get("max_users") is None and meta["num_examples"] != meta["total_test_examples"]:
                raise ValueError("incomplete full-test export")
            with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["example_index", "user_id", "target_item_id", *RANK_COLUMNS]:
                    raise ValueError("invalid CSV columns")
                count = sum(1 for _ in reader)
            if count != meta["num_examples"]:
                raise ValueError("CSV example count mismatch")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{spec['run_id']}: {exc}")
    if errors:
        raise ValueError("Export verification failed:\n" + "\n".join(errors))
    print(f"Verified all {len(manifest['runs'])} top10 files and their train popularity files")


def check_inputs(manifest: dict, repo_root: Path) -> None:
    """Check historical inputs before starting preparation and inference."""
    from bucket_inference import resolve_data_config
    from top10_manifest import portable

    errors = [f"No source runs for {group}" for group in active_missing_groups(manifest)]
    for spec in manifest["runs"]:
        checkpoint = repo_root / portable(spec["checkpoint"])
        if not checkpoint.is_file():
            errors.append(f"Missing original checkpoint: {checkpoint}")
    for preparation in manifest["preparations"]:
        try:
            resolve_data_config(preparation["spec"], repo_root, check_inputs=True)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise FileNotFoundError("Cannot start export:\n" + "\n".join(dict.fromkeys(errors)))
    print(f"All original input paths exist for {len(manifest['runs'])} checkpoints")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("discover", "check", "prepare", "generate", "verify"), required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/exp6_result_analysis/top10")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare-index", type=int)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-users", type=int, help="Discover only: freeze a smoke-test subset; default is the complete test")
    parser.add_argument("--sample-seed", type=int, help="Discover only: test subset seed (default: 42)")
    parser.add_argument("--selection-split", choices=("validation", "test"),
                        help="Discover only: metrics used to select GFlow/CE configurations (default: test); evaluation is always test")
    parser.add_argument("--checkpoint-kinds", nargs="+", choices=("best", "last"),
                        help="Discover only: first/second SFT checkpoints to export; selected GFlow/CE always use best")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.cpu_threads) < 1 or args.num_beams < TOP_K:
        parser.error("Require positive batch-size/cpu-threads and num-beams >= 10")
    if args.max_users is not None and (args.max_users < 1 or args.mode != "discover"):
        parser.error("max-users must be positive and is frozen in discover mode")
    if args.mode != "discover" and any(getattr(args, key) is not None for key in
                                       ("sample_seed", "selection_split", "checkpoint_kinds")):
        parser.error("sample-seed, selection-split, and checkpoint-kinds are frozen in discover mode")
    args.sample_seed = 42 if args.sample_seed is None else args.sample_seed
    args.selection_split = args.selection_split or "test"
    args.checkpoint_kinds = list(dict.fromkeys(args.checkpoint_kinds or ["best"]))
    for name, mode in (("run_index", "generate"), ("prepare_index", "prepare")):
        index = getattr(args, name)
        if index is not None and (index < 0 or args.mode != mode):
            parser.error(f"{name.replace('_', '-')} must be nonnegative and used only with {mode}")
    args.repo_root, args.output_dir = args.repo_root.resolve(), args.output_dir.resolve()
    args.manifest = (args.manifest or args.output_dir / "manifest.json").resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "discover":
        from top10_manifest import build_manifest
        manifest = freeze_manifest(build_manifest(args.repo_root, selection_split=args.selection_split,
                                                  checkpoint_kinds=args.checkpoint_kinds), args.repo_root,
                                   max_users=args.max_users, sample_seed=args.sample_seed,
                                   selection_split=args.selection_split, checkpoint_kinds=args.checkpoint_kinds)
        validate_manifest(manifest)
        if args.manifest.exists() and not args.force:
            raise FileExistsError(f"Frozen manifest already exists: {args.manifest}; choose a new path or --force")
        write_json(args.manifest, manifest)
        for index, spec in enumerate(manifest["runs"]):
            print(f"{index:3d} {spec['run_id']} {','.join(spec.get('selection_labels', []))}: {spec['checkpoint']}")
        print(f"{len(manifest['runs'])} checkpoints; {len(manifest['preparations'])} data preparations; {args.manifest}")
        if manifest["missing_groups"]:
            print("MISSING source groups: " + ", ".join(manifest["missing_groups"]))
        return 0
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if args.mode == "check":
        check_inputs(manifest, args.repo_root)
        return 0
    if args.mode == "verify":
        verify_exports(manifest, args.output_dir)
        return 0
    import torch
    from bucket_inference import generate_prepared_run, prepare_dataset
    torch.set_num_threads(args.cpu_threads)
    entries = manifest["preparations"] if args.mode == "prepare" else manifest["runs"]
    index = args.prepare_index if args.mode == "prepare" else args.run_index
    if index is not None:
        if index >= len(entries):
            raise ValueError(f"Array index {index} out of range for {len(entries)} entries")
        entries = [entries[index]]
    for entry in entries:
        prep_id = entry["preparation_id"]
        prepared = args.output_dir / "prepared" / prep_id
        if args.mode == "prepare":
            prepare_dataset(entry["spec"], repo_root=args.repo_root, output_dir=prepared,
                            max_users=manifest["max_users"], sample_seed=manifest["sample_seed"], force=args.force)
            export_popularity(prepared, args.output_dir / "popularity" / f"{prep_id}.csv")
        else:
            path = generate_prepared_run(entry, repo_root=args.repo_root, prepared_dir=prepared,
                output_dir=args.output_dir / "cache" / prep_id, device=args.device,
                batch_size=args.batch_size, num_beams=args.num_beams, top_k=TOP_K, force=args.force)
            export_recommendations(path, entry, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
