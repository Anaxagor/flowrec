"""Frozen source identities for continuing completed SFT-only experiments.

This module uses only the standard library. It never discovers a substitute
checkpoint, imports training code, or loads model tensors. Paths are relative
to a supplied repository root; recorded data paths are relocated.
The manifest records the actual checkpoint hash and every metadata dependency.
Use ``hash_checkpoint=False`` for cheap inventory only, not training identity.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = Path(__file__).with_name("source_protocols.json")
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    model: str
    dataset: str
    protocol: str
    default: bool = True


_SPECS = [
    SourceSpec("tiger_beauty", "tiger", "beauty", "finetuned"),
    SourceSpec("tiger_instruments", "tiger", "instruments", "finetuned"),
    # The selected weights exist, but their full SFT protocol is unavailable.
    SourceSpec("tiger_beauty_2023", "tiger", "beauty_2023", "finetuned_matched", default=False),
    SourceSpec("tiger_yambda50_matched", "tiger", "yambda_50m_listens", "finetuned_matched"),
    SourceSpec("tiger_yambda500_matched_r15", "tiger", "yambda_500m_listens", "finetuned_matched_r15"),
    SourceSpec("gpt2_beauty", "gpt2", "beauty", "dense_window_v1"),
    SourceSpec("gpt2_instruments", "gpt2", "instruments", "dense_window_v1"),
    SourceSpec("gpt2_beauty_2023", "gpt2", "beauty_2023", "dense_window_v1"),
    SourceSpec("gpt2_yambda50_trial36", "gpt2", "yambda_50m_listens", "matched_trial36"),
    SourceSpec("gpt2_yambda500_trial4_sampled50", "gpt2", "yambda_500m_listens", "matched_trial4_sampled50"),
]
SOURCE_SPECS = {spec.source_id: spec for spec in _SPECS}


def default_source_ids(models: Iterable[str] | str | None = None) -> list[str]:
    selected = {models} if isinstance(models, str) else set(models or ("tiger", "gpt2"))
    unknown = selected - {"tiger", "gpt2"}
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return [spec.source_id for spec in _SPECS if spec.default and spec.model in selected]


def source_paths(source_id: str, source_seed: int = 42,
                 project_root: Path | str = PROJECT_ROOT) -> dict[str, Path]:
    if source_id not in SOURCE_SPECS:
        raise ValueError(f"Unknown source_id {source_id!r}; choose one of {sorted(SOURCE_SPECS)}")
    if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
        raise ValueError("source_seed must be a nonnegative integer")
    root = Path(project_root).resolve()
    spec = SOURCE_SPECS[source_id]
    directory = _primary_directory(spec, root)
    checkpoint = directory / "best.pt"
    paths = {"run_dir": directory, "checkpoint": checkpoint,
             "sidecar": directory / "best.pt.json"}
    if spec.model == "tiger":
        paths.update(config=directory / "run_config.json", result=directory / "metrics.json",
                     resolved=directory / "resolved_config.json")
    else:
        paths.update(config=directory / "resolved_config.json", result=directory / "result.json",
                     input_record=directory / "input_record.json")
    return paths


def _primary_directory(spec: SourceSpec, root: Path) -> Path:
    directory = root / "checkpoints/exp2/loo_split/sft-finetuned" / spec.model / spec.dataset
    if spec.model == "gpt2" and spec.dataset.startswith("yambda_"):
        regime = {"matched_trial36": "last_128", "matched_trial4_sampled50": "sampled_50"}[spec.protocol]
        directory = directory / regime
    return directory


def primary_source_seed(source_id: str, requested_seed: int = 42,
                        project_root: Path | str = PROJECT_ROOT) -> int:
    """Resolve seed aliases to the actual selected model's recorded seed."""
    spec = SOURCE_SPECS[source_id]
    if isinstance(requested_seed, bool) or not isinstance(requested_seed, int) or requested_seed < 0:
        raise ValueError("source_seed must be a nonnegative integer")
    root = Path(project_root).resolve()
    directory = _primary_directory(spec, root)
    provenance = directory / "provenance.json"
    if not (directory / "best.pt").is_file() or not provenance.is_file():
        return requested_seed
    record = json.loads(provenance.read_text(encoding="utf-8"))
    seed = record.get("seed")
    if (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or any(record.get(key) != value for key, value in
                   (("stage", "sft-finetuned"), ("model", spec.model), ("dataset", spec.dataset)))):
        return requested_seed
    if _curated_checkpoint(spec, seed, root) is not None:
        return seed
    return requested_seed


def _curated_checkpoint(spec: SourceSpec, seed: int, root: Path) -> Path | None:
    """Check the selected file's canonical identity and recorded weight hash."""
    directory = _primary_directory(spec, root)
    checkpoint, provenance = directory / "best.pt", directory / "provenance.json"
    if not checkpoint.is_file() or not provenance.is_file():
        return None
    record = json.loads(provenance.read_text(encoding="utf-8"))
    expected = {"stage": "sft-finetuned", "split": "loo_split", "model": spec.model,
                "dataset": spec.dataset, "seed": seed,
                "checkpoint": checkpoint.relative_to(root).as_posix()}
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    if spec.model == "gpt2" and spec.dataset.startswith("yambda_"):
        if record.get("regime") != directory.name:
            return None
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return None
    return checkpoint


def available_source_seeds(source_id: str, project_root: Path | str = PROJECT_ROOT) -> list[int]:
    spec = SOURCE_SPECS[source_id]
    root = Path(project_root).resolve()
    directory = _primary_directory(spec, root)
    provenance = directory / "provenance.json"
    if (directory / "best.pt").is_file() and provenance.is_file():
        seed = json.loads(provenance.read_text(encoding="utf-8")).get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 and (
                _curated_checkpoint(spec, seed, root) is not None):
            return [seed]
    return []


def relocate_path(value: str | Path | None, project_root: Path | str = PROJECT_ROOT) -> str | None:
    """Relocate a recorded host prefix without opening historical files."""
    if value is None:
        return None
    root, path = Path(project_root).resolve(), Path(value)
    if not path.is_absolute():
        path = root / path
    elif not path.is_relative_to(root) and "/flowrec/" in str(path):
        path = root / str(path).split("/flowrec/", 1)[1]
    return str(path)


def _relocate_config(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = _relocate_config(value, root)
        elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir")):
            result[key] = relocate_path(value, root)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _snapshot_records(spec: SourceSpec, seed: int, config_path: Path) -> tuple[dict, list[dict]]:
    """Portable auxiliary records, bound to the selected source's actual config."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    entry = snapshot.get("sources", {}).get(spec.source_id)
    if not entry or str(seed) not in entry.get("seeds", {}):
        raise ValueError(f"{spec.source_id} seed {seed}: no frozen preprocessing record in {SNAPSHOT_PATH}")
    if any(entry.get(key) != getattr(spec, key) for key in ("model", "dataset", "protocol")):
        raise ValueError(f"{spec.source_id}: frozen preprocessing identity mismatch")
    saved = entry["seeds"][str(seed)]
    if _sha256(config_path) != saved["source_config_sha256"]:
        raise ValueError(f"{spec.source_id}: source config changed relative to frozen preprocessing record")
    records, metadata = {}, []
    for role, record in saved["records"].items():
        if _json_hash(record["content"]) != record["content_sha256"]:
            raise ValueError(f"{spec.source_id}: corrupt frozen {role} record")
        records[role] = copy.deepcopy(record["content"])
        pointer = f"scripts/exp7-from-sft/source_protocols.json#{spec.source_id}/seed_{seed}/{role}"
        metadata.append({"role": role, "path": str(SNAPSHOT_PATH), "relative_path": pointer,
                         "sha256": record["content_sha256"]})
    return records, metadata


def _validate_identity(spec: SourceSpec, seed: int, config: dict, result: dict,
                       sidecar: dict) -> None:
    if result.get("status") != "completed":
        raise ValueError(f"{spec.source_id}: source run is not completed")
    expected_variant = "sft_finetune" if spec.model == "tiger" else "sft_only"
    if result.get("variant") != expected_variant:
        raise ValueError(f"{spec.source_id}: source result is not {expected_variant}")
    for record in (result, config):
        if record.get("seed") != seed:
            raise ValueError(f"{spec.source_id}: recorded source seed does not match {seed}")
    if result.get("dataset") != spec.dataset:
        raise ValueError(f"{spec.source_id}: source dataset mismatch")
    if spec.model == "gpt2" and (config.get("dataset") != spec.dataset or config.get("variant") != "sft_only"):
        raise ValueError(f"{spec.source_id}: resolved source is not the selected dataset/SFT-only arm")
    if spec.model == "tiger" and config.get("method", {}).get("name") != "sft":
        raise ValueError(f"{spec.source_id}: run_config is not SFT-only")
    if sidecar.get("model_type") != spec.model or sidecar.get("method_name") != "sft":
        raise ValueError(f"{spec.source_id}: checkpoint sidecar model/method mismatch")
    if not sidecar.get("model_config"):
        raise ValueError(f"{spec.source_id}: checkpoint sidecar has no model_config")
    for key, value in config.get("model", {}).items():
        if key in sidecar["model_config"] and value != sidecar["model_config"][key]:
            raise ValueError(f"{spec.source_id}: recorded model disagrees with checkpoint on {key}")


def _counts(data: Mapping[str, Any]) -> dict[str, int]:
    return {key: int(data[key]) for key in ("num_items", "num_users", "num_train_examples",
            "num_finetune_examples", "num_val_examples", "num_test_examples") if data.get(key) is not None}


def _tiger_protocol(spec: SourceSpec, config: dict, resolved: dict, result: dict,
                    sidecar: dict, root: Path) -> dict[str, Any]:
    if resolved.get("dataset") != spec.dataset or resolved.get("seed") != result["seed"]:
        raise ValueError(f"{spec.source_id}: experiment resolved_config identity mismatch")
    subset_record = resolved.get("finetune_subset", "full train set")
    if isinstance(subset_record, dict):
        random_n, subset_seed = int(subset_record["per_user_random_n"]), int(subset_record["seed"])
        subset_size = int(subset_record["num_examples"])
    elif subset_record == "full train set":
        random_n, subset_seed, subset_size = 0, 0, int(resolved["num_train_examples"])
    else:
        raise ValueError(f"{spec.source_id}: unsupported recorded subset {subset_record!r}")
    expected_n = 15 if spec.protocol == "finetuned_matched_r15" else 0
    if random_n != expected_n or (random_n and subset_seed != 42):
        raise ValueError(f"{spec.source_id}: recorded subset does not match source family")
    source = _relocate_config(config, root)
    batch = int(source["dataloader"].get("train_batch_size", source["dataloader"]["batch_size"]))
    accum = int(source["training"]["gradient_accumulation_steps"])
    return {"source_config": source, "data_config": {key: copy.deepcopy(source[key])
            for key in ("data", "dataset", "dataloader", "model", "seed")},
            "model_config": copy.deepcopy(sidecar["model_config"]),
            "model_profile": {"model": copy.deepcopy(source["model"])},
            "subset": {"train_last_n": 0, "train_random_n": random_n,
                       "train_subset_seed": subset_seed, "train_pool_last_n": None},
            "optimizer": copy.deepcopy(source["optimizer"]), "scheduler": source.get("scheduler"),
            "trainer": copy.deepcopy(resolved.get("trainer", sidecar["trainer_config"])),
            "batching": {"batch_size": batch, "gradient_accumulation": accum,
                         "effective_batch": batch * accum,
                         "val_batch_size": source["dataloader"]["validation_batch_size"],
                         "test_batch_size": source["dataloader"]["validation_batch_size"]},
            "expected_counts": {"num_items": int(resolved["item_vocab_size"]) - 1,
                "num_train_examples": int(resolved["num_train_examples"]), "num_finetune_examples": subset_size,
                "num_val_examples": int(result["validation"]["num_examples"]),
                "num_test_examples": int(result["test"]["num_examples"])},
            "input_audit": None, "recorded_cli": {}, "preprocessing_origin": "source_run_config"}


def _gpt2_inputs(spec: SourceSpec, config: dict, recorded: dict, root: Path) -> tuple[dict, dict | None, str]:
    """Recover source inputs from saved JSON, never current YAML defaults."""
    if spec.protocol == "dense_window_v1":
        context = config["data"].get("training_context", {})
        if context != {"context": "dense_window", "last_n": None, "target_policy": "all",
                       "version": 1, "window_len": 20, "window_stride": 20}:
            raise ValueError(f"{spec.source_id}: expected the recorded dense-window v1 Amazon context")
        if "dataset" in recorded and isinstance(recorded["dataset"], dict):
            result = {key: copy.deepcopy(recorded[key]) for key in ("data", "dataset", "model")}
            origin = "source_data_reward_config"
        else:
            # Beauty 2023's parent fine-tuning JSON was not downloaded. Its
            # saved original data config + recorded dense-window v1 rule
            # fully specify the expansion used by its SFT-only checkpoint.
            values = copy.deepcopy(recorded["data"])
            data = {"source": "amazon"}
            for key in ("reviews_path", "metadata_path", "semantic_id_path"):
                data[key] = values.pop(key)
            user_hash = values.get("user_hash_mod")
            values.update(train_example_layout="prefix_target", train_prefix_from_dense=True,
                          train_dense_match_prefix=False, train_target_policy="all", train_all_prefixes=True,
                          history_padding="left", train_window_len=None, train_window_stride=None)
            result = {"data": data, "dataset": values,
                      "model": {"use_user_token": user_hash is not None, "user_ids_count": user_hash}}
            origin = "saved_pretraining_data_config+dense_window_v1"
        result["seed"] = int(config["seed"])
        audit = copy.deepcopy(context)
    else:
        if recorded.get("dataset") != spec.dataset:
            raise ValueError(f"{spec.source_id}: recorded pretraining dataset mismatch")
        values = copy.deepcopy(recorded["data_config"])
        for key, expected in {"train_dense_match_prefix": True, "train_example_layout": "dense_causal",
                              "train_target_policy": "last_n", "train_last_n": 128,
                              "max_history_len": 50, "train_window_stride": 1}.items():
            if values.get(key) != expected:
                raise ValueError(f"{spec.source_id}: recorded pretraining {key} differs from matched protocol")
        audit = copy.deepcopy((recorded.get("resume_identity") or {}).get("data_audit"))
        required_hashes = ("catalog_sha256", "semantic_ids_sha256", "train_targets_sha256",
                           "train_contexts_sha256", "validation_sha256", "test_sha256")
        if not audit or any(key not in audit for key in required_hashes):
            raise ValueError(f"{spec.source_id}: matched source lacks saved data fingerprints")
        # The source SFT-only runner partitioned the original positive stream
        # to retain reward evidence without changing examples or histories.
        predicates = []
        if values.get("positive_criterion") is not None:
            predicates.append(f"({values['positive_criterion']})")
        if values.get("allowed_event_types") is not None:
            predicates.append(f"(event_type in {list(values['allowed_event_types'])!r})")
        if values.get("min_played_ratio_pct") is not None:
            predicates.append(f"(event_type != 'listen' or played_ratio_pct >= {float(values['min_played_ratio_pct'])!r})")
        if predicates:
            values.update(positive_criterion=" and ".join(predicates), allowed_event_types=None, min_played_ratio_pct=None)
        data = {"source": "yambda"}
        for key in ("interactions_path", "semantic_id_path", "embeddings_path", "use_hf",
                    "hf_dataset_name", "streaming", "max_rows"):
            if key in values:
                data[key] = values.pop(key)
        user_hash = values.get("user_hash_mod")
        values.update(train_example_layout="prefix_target", train_dense_match_prefix=False,
                      train_prefix_from_dense=False, history_padding="left",
                      matched_tuning_data_audit=copy.deepcopy(audit))
        result = {"data": data, "dataset": values, "seed": int(recorded["data_config"]["seed"]),
                  "model": {"use_user_token": user_hash is not None, "user_ids_count": user_hash}}
        origin = "matched_pretraining_data_config+recorded_prefix_conversion"
    result = _relocate_config(result, root)
    for key, value in result["data"].items():
        if key.endswith("_path") and key in config["data"]:
            if value != relocate_path(config["data"][key], root):
                raise ValueError(f"{spec.source_id}: saved input record disagrees with source {key}")
    if result["dataset"].get("positive_criterion") != config["data"].get("positive_criterion"):
        raise ValueError(f"{spec.source_id}: saved positive stream differs from source SFT-only run")
    return result, audit, origin


def _gpt2_protocol(spec: SourceSpec, config: dict, recorded: dict, sidecar: dict, root: Path) -> dict[str, Any]:
    profile = copy.deepcopy(config["model_profile"])
    if "dropout" not in profile.get("model", {}):
        raise ValueError(f"{spec.source_id}: saved model_profile must include dropout")
    inputs, audit, origin = _gpt2_inputs(spec, config, recorded, root)
    cli, data = config["cli_args"], config["data"]
    batching = copy.deepcopy(config["batching"])
    for key in ("val_batch_size", "test_batch_size"):
        batching.setdefault(key, int(cli[key]))
    inputs["dataloader"] = {"train_batch_size": batching["batch_size"], "batch_size": batching["batch_size"],
        "validation_batch_size": batching["val_batch_size"], "num_workers": int(cli.get("num_workers", 0)),
        "persistent_workers": int(cli.get("num_workers", 0)) > 0, "pin_memory": True, "train_drop_last": False}
    subset = {"train_last_n": int(data.get("finetune_per_user_last_n") or 0),
              "train_random_n": int(data.get("finetune_per_user_random_n") or 0),
              "train_subset_seed": int(data.get("finetune_subset_seed") or 0),
              "train_pool_last_n": data.get("finetune_per_user_pool_last_n")}
    if spec.protocol == "matched_trial4_sampled50" and subset != {
            "train_last_n": 0, "train_random_n": 50, "train_subset_seed": 42, "train_pool_last_n": 128}:
        raise ValueError(f"{spec.source_id}: source is not the expected sampled50 subset")
    if spec.protocol == "matched_trial36" and (subset["train_random_n"] or subset["train_pool_last_n"] != 128):
        raise ValueError(f"{spec.source_id}: source is not the full last-128 matched pool")
    if spec.protocol == "dense_window_v1" and (subset["train_random_n"] or subset["train_last_n"]):
        raise ValueError(f"{spec.source_id}: source does not use all dense-window prefixes")
    if bool(cli.get("prefix_from_dense")) != (spec.protocol == "dense_window_v1"):
        raise ValueError(f"{spec.source_id}: source context construction disagrees with selected protocol")
    return {"source_config": _relocate_config(config, root), "data_config": inputs,
            "model_config": copy.deepcopy(sidecar["model_config"]), "model_profile": profile,
            "subset": subset, "optimizer": copy.deepcopy(config["optimizer"]),
            "scheduler": None, "trainer": copy.deepcopy(config["trainer"]), "batching": batching,
            "expected_counts": _counts(data), "input_audit": audit,
            "recorded_cli": copy.deepcopy(cli), "preprocessing_origin": origin}


def resolve_source(source_id: str, source_seed: int = 42, project_root: Path | str = PROJECT_ROOT,
                   *, hash_checkpoint: bool = True) -> dict[str, Any]:
    source_seed = primary_source_seed(source_id, source_seed, project_root)
    paths = source_paths(source_id, source_seed, project_root)
    spec, root = SOURCE_SPECS[source_id], Path(project_root).resolve()
    if source_id == "tiger_beauty_2023" and paths["checkpoint"].is_file():
        raise FileNotFoundError(
            f"{source_id}: primary checkpoint is available at {paths['checkpoint']}, but missing exact SFT-only source assets: "
            "the supplied checkpoint needs its original run config, completed result and frozen preprocessing protocol; "
            "the old matched protocol cannot be substituted")
    primary = ("checkpoint", "sidecar", "config", "result")
    missing = [str(paths[key]) for key in primary if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError(f"{source_id} seed {source_seed}: missing exact SFT-only source assets; "
                                "no fallback to another family/seed/initial checkpoint:\n" + "\n".join(missing))
    records, metadata_files = {}, []
    for key, path in paths.items():
        if key in ("sidecar", "config", "result"):
            records[key] = json.loads(path.read_text(encoding="utf-8"))
            metadata_files.append({"role": key, "path": str(path),
                                   "relative_path": path.relative_to(root).as_posix(),
                                   "sha256": _sha256(path)})
    _validate_identity(spec, source_seed, records["config"], records["result"], records["sidecar"])
    frozen_records, frozen_files = _snapshot_records(spec, source_seed, paths["config"])
    records.update(frozen_records)
    metadata_files.extend(frozen_files)
    protocol = (_tiger_protocol(spec, records["config"], records["resolved"], records["result"], records["sidecar"], root)
                if spec.model == "tiger" else
                _gpt2_protocol(spec, records["config"], records["input_record"], records["sidecar"], root))
    checkpoint_hash = _sha256(paths["checkpoint"]) if hash_checkpoint else None
    if _curated_checkpoint(spec, source_seed, root) is None:
        raise ValueError(f"{source_id}: primary provenance does not match the selected source protocol/seed")
    provenance = json.loads((paths["checkpoint"].parent / "provenance.json").read_text(encoding="utf-8"))
    if hash_checkpoint and checkpoint_hash != provenance["sha256"]:
        raise ValueError(f"{source_id}: curated checkpoint differs from its recorded source hash")
    metadata_hashes = provenance.get("metadata_sha256", {})
    for item in metadata_files:
        if item["role"] in ("sidecar", "config", "result"):
            if metadata_hashes.get(Path(item["path"]).name) != item["sha256"]:
                raise ValueError(f"{source_id}: source {item['role']} differs from its recorded metadata hash")
    fingerprint_payload = {"schema_version": SCHEMA_VERSION, "source_id": source_id, "source_seed": source_seed,
        "checkpoint_relative": paths["checkpoint"].relative_to(root).as_posix(), "checkpoint_sha256": checkpoint_hash,
        "metadata": [{key: item[key] for key in ("role", "relative_path", "sha256")} for item in metadata_files]}
    return {"schema_version": SCHEMA_VERSION, "source_id": source_id, "model": spec.model, "dataset": spec.dataset,
            "source_seed": source_seed, "protocol": spec.protocol, "checkpoint": str(paths["checkpoint"]),
            "run_dir": str(paths["run_dir"]), "checkpoint_sha256": checkpoint_hash,
            "source_fingerprint": _json_hash(fingerprint_payload),
            "fingerprint_kind": "checkpoint_and_metadata" if hash_checkpoint else "metadata_only",
            "metadata_files": metadata_files, "source_result": copy.deepcopy(records["result"]), **protocol}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-id", choices=list(SOURCE_SPECS))
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.source_id:
        result = resolve_source(args.source_id, args.source_seed, args.project_root,
                                hash_checkpoint=not args.metadata_only)
    else:
        result = [{"source_id": spec.source_id, "model": spec.model, "dataset": spec.dataset,
                   "default": spec.default, "available_seeds": available_source_seeds(spec.source_id, args.project_root)}
                  for spec in _SPECS]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
