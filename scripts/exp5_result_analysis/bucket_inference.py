"""Freeze evaluation inputs on CPU, then decode independent checkpoints on GPU.

Preparation deliberately uses the original datamodule, including its train
layout: cold-item filtering derives the catalog from those examples. Training
frames are never serialized or loaded by a GPU inference task.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from inference import _atomic_json, _json, _sha256, _signature, resolve_project_path


SCHEMA_VERSION = 1
_INPUT_KEYS = ("user_ids", "hashed_user_ids", "history_token_ids", "history_token_mask")


def _imports(repo_root: Path):
    for directory in (repo_root, repo_root / "scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return importlib.import_module("run_tiger")


def resolve_data_config(spec: dict[str, Any], repo_root: Path, *, check_inputs: bool = True) -> dict[str, Any]:
    """Resolve captured Amazon/Yambda input paths without downloading anything."""
    path = resolve_project_path(spec["data_config"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation data config: {path}")
    config = copy.deepcopy(_json(path))
    for key in spec.get("data_config_selector", []):
        if not isinstance(config, dict) or key not in config:
            raise ValueError(f"{path}: missing embedded evaluation configuration {key}")
        config = config[key]
    if not isinstance(config, dict) or not config.get("data"):
        raise ValueError(f"{path}: expected an evaluation config with a data section")
    data = config.setdefault("data", {})
    source = str(data.get("source", config.get("dataset", {}).get("source", "amazon"))).lower()
    if source not in {"amazon", "yambda"}:
        raise ValueError(f"Unsupported source: {source}")
    required = {"semantic_id_path", *( ["interactions_path"] if source == "yambda" else ["reviews_path", "metadata_path"] )}
    for name in required | {"embeddings_path"}:
        value = data.get(name)
        if not value:
            if name in required:
                raise ValueError(f"{path}: missing data.{name}")
            continue
        resolved = resolve_project_path(value, repo_root)
        if not resolved.is_file() and source == "amazon" and name in {"reviews_path", "metadata_path"}:
            local = repo_root / "data/raw/dataframe_format" / Path(value).name
            if local.is_file():
                resolved = local
        if check_inputs and not resolved.is_file():
            raise FileNotFoundError(f"Missing evaluation input {name}: {resolved}")
        data[name] = str(resolved)
    if data.get("use_hf"):
        raise ValueError("Frozen evaluation requires local inputs (data.use_hf must be false)")
    config.setdefault("dataloader", {}).update(
        validation_batch_size=512, num_workers=0, persistent_workers=False,
        pin_memory=False, include_raw_in_batch=False,
    )
    return config


def data_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint evaluation-affecting settings, ignoring optimization knobs."""
    dataset = copy.deepcopy(config.get("dataset", {}))
    data = copy.deepcopy(config.get("data", {}))
    # 50m GFlow retains non-positive events for rewards; its positive stream is
    # identical to the original SFT config's destructive listen>=50 filters.
    if str(data.get("source", dataset.get("source", "amazon"))).lower() == "yambda":
        if dataset.get("allowed_event_types") == ["listen"] and dataset.get("min_played_ratio_pct") == 50:
            if not dataset.get("positive_criterion"):
                dataset["positive_criterion"] = "event_type == 'listen' and played_ratio_pct >= 50"
                dataset["allowed_event_types"] = None
                dataset["min_played_ratio_pct"] = None
    model = config.get("model", {})
    return {"data": data, "dataset": dataset,
            "user_token": {k: model.get(k) for k in ("use_user_token", "user_ids_count")}}


def select_examples(frame, max_users: int | None, sample_seed: int):
    users = frame["raw_user_id"].astype(str).to_numpy(dtype=str)
    if not len(users) or len(set(users)) != len(users):
        raise ValueError("Expected a nonempty test split with one target per raw user")
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive")
    indices = list(range(len(users)))
    if max_users is not None and max_users < len(indices):
        indices.sort(key=lambda i: hashlib.sha256(f"{sample_seed}:{users[i]}".encode()).hexdigest())
        indices = indices[:max_users]
    indices.sort(key=lambda i: users[i])
    return indices


def example_arrays(frame, max_history_len: int) -> dict[str, np.ndarray]:
    histories = [list(map(str, h[-max_history_len:])) for h in frame["history_raw_items"]]
    clipped = np.asarray([len(h) for h in histories], dtype=np.int64)
    if "target_position" not in frame:
        raise ValueError("Full history cohorts require original target_position, not clipped histories")
    full = frame["target_position"].to_numpy(dtype=np.int64)
    if (full < clipped).any():
        raise ValueError("Full pre-target histories cannot be shorter than model inputs")
    return {"user_ids": frame["raw_user_id"].astype(str).to_numpy(dtype=str),
            "target_item_ids": frame["target_raw_item"].astype(str).to_numpy(dtype=str),
            "history_lengths": clipped, "full_history_lengths": full,
            "history_fingerprints": np.asarray([hashlib.sha256(json.dumps(h).encode()).hexdigest() for h in histories])}


def _code_signatures(repo_root: Path) -> dict[str, str]:
    paths = [Path(__file__), repo_root / "scripts/run_tiger.py", repo_root / "scripts/run_gpt2.py",
             repo_root / "scripts/sasrec_config_runner.py"]
    paths.extend(sorted((repo_root / "flowrec/data").glob("*.py")))
    paths.extend(sorted((repo_root / "flowrec/models").glob("*.py")))
    return {str(p.relative_to(repo_root)): _sha256(p) for p in paths if p.is_file()}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _library_versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "pandas", "torch", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _verify_files(directory: Path, hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        path = directory / filename
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Prepared/cache artifact is missing or changed: {path}")


def prepare_dataset(
    spec: dict[str, Any], *, repo_root: Path, output_dir: Path,
    max_users: int | None = None, sample_seed: int = 42, force: bool = False,
) -> Path:
    """Build exact original splits once on CPU and freeze small generation inputs.

    Full history means the number of positive interactions preceding the held
    out target after original preprocessing, before input clipping/cold-item
    removal. It includes the preceding validation item in leave-one-out.
    """
    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    config = resolve_data_config(spec, repo_root)
    sources = [_signature(resolve_project_path(spec["data_config"], repo_root), contents=True)]
    # Parquet can contain hundreds of millions of events. Stat signatures avoid
    # reading the full input twice; frozen output files get content hashes.
    sources.extend(_signature(Path(v)) for v in config["data"].values()
                   if isinstance(v, str) and Path(v).is_file())
    fingerprint_inputs = {"schema_version": SCHEMA_VERSION, "dataset": spec["dataset"],
                          "model": spec["model"], "sources": sources, "config": config,
                          "max_users": max_users, "sample_seed": sample_seed,
                          "code": _code_signatures(repo_root), "libraries": _library_versions()}
    fingerprint = _digest(fingerprint_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.json"
    previous = _json(meta_path)
    if not force and previous.get("fingerprint") == fingerprint:
        _verify_files(output_dir, previous["files_sha256"])
        print(f"Reuse verified prepared evaluation: {output_dir}", flush=True)
        return output_dir
    import torch
    from torch.utils.data import DataLoader, Subset
    runner = _imports(repo_root)
    cfg = runner.build_data_config(config, config_path=repo_root / "analysis_config.json")
    dm = runner.build_data_module(cfg)
    print(f"[{spec['dataset']}] build exact original splits on CPU (zero loader workers)", flush=True)
    dm.setup()
    frame = dm.test_dataset.examples
    indices = select_examples(frame, max_users, sample_seed)
    arrays = example_arrays(frame.iloc[indices], int(dm.cfg.max_history_len))
    catalog_ids = sorted(map(str, dm.item2id))
    if len(catalog_ids) != dm.num_items or not dm.semantic_ids or set(catalog_ids) != set(dm.semantic_ids):
        raise ValueError("Evaluation catalog and semantic IDs differ or contain padding")
    semantics = np.asarray([dm.semantic_ids[i] for i in catalog_ids], dtype=np.int64)
    if semantics.ndim != 2 or len(np.unique(semantics, axis=0)) != len(semantics):
        raise ValueError("Semantic IDs must be fixed length and unique")
    # Existing project function counts raw training interactions, never the
    # duplicated prefix labels and never validation/test targets.
    popularity = importlib.import_module("sasrec_config_runner").training_item_popularity(dm).cpu().numpy()
    files = []
    catalog_path = output_dir / "catalog.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "popularity"])
        writer.writerows((i, int(popularity[dm.item2id[i]])) for i in catalog_ids)
    files.append(catalog_path.name)
    for filename, values in (("examples.npz", arrays), ("semantics.npz", {"item_ids": np.asarray(catalog_ids), "semantic_ids": semantics})):
        np.savez_compressed(output_dir / filename, **values)
        files.append(filename)
    base_loader = dm.test_dataloader()
    # include_raw=False may select the tensorized index-yielding dataset. Its
    # collator consumes integer indices, so preserve that dataset/collator pair.
    loader = DataLoader(Subset(base_loader.dataset, indices), batch_size=512, shuffle=False,
                        num_workers=0, collate_fn=base_loader.collate_fn)
    tensors = {}
    cursor = 0
    for batch in loader:
        rows = len(batch["user_ids"])
        for key in _INPUT_KEYS:
            if key not in batch:
                continue
            value = batch[key].detach().cpu().numpy()
            if key not in tensors:
                filename = f"input_{key}.npy"
                tensors[key] = np.lib.format.open_memmap(output_dir / filename, mode="w+", dtype=value.dtype,
                                                       shape=(len(indices), *value.shape[1:]))
                files.append(filename)
            tensors[key][cursor:cursor + rows] = value
        cursor += rows
    if cursor != len(indices) or not {"user_ids", "history_token_ids", "history_token_mask"}.issubset(tensors):
        raise ValueError("Incomplete frozen generation inputs")
    tensor_keys = list(tensors)
    for value in tensors.values():
        value.flush()
    tensors.clear()
    metadata = {"schema_version": SCHEMA_VERSION, "dataset": spec["dataset"], "model": spec["model"],
                "fingerprint": fingerprint, "fingerprint_inputs": fingerprint_inputs,
                "data_protocol": data_protocol(config), "config": config,
                "num_users": len(indices), "total_test_users": len(frame), "catalog_size": len(catalog_ids),
                "sid_len": int(dm.sid_len), "max_history_len": int(dm.cfg.max_history_len),
                "train_window_len": dm.cfg.train_window_len, "tensor_keys": tensor_keys,
                "full_history_definition": "target_position: all preceding positive events after original preprocessing, before model clipping and cold-item history filtering; includes validation event",
                "test_context_sha256": _digest({k: v.tolist() for k, v in arrays.items()}),
                "files_sha256": {name: _sha256(output_dir / name) for name in files}}
    _atomic_json(meta_path, metadata)
    print(f"[{spec['dataset']}] prepared {len(indices)} test users, {len(catalog_ids)} catalog items", flush=True)
    return output_dir


def generate_prepared_run(
    spec: dict[str, Any], *, repo_root: Path, prepared_dir: Path, output_dir: Path,
    device: str, batch_size: int, num_beams: int, top_k: int, force: bool = False,
) -> Path:
    """Decode one checkpoint without reading raw datasets or building train splits."""
    if batch_size < 1 or top_k < 1 or num_beams < top_k:
        raise ValueError("Require batch_size >= 1 and num_beams >= top_k >= 1")
    repo_root, prepared_dir, output_dir = map(lambda p: Path(p).resolve(), (repo_root, prepared_dir, output_dir))
    meta_path = prepared_dir / "metadata.json"
    metadata = _json(meta_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Missing or incompatible CPU preparation: {meta_path}")
    if any(metadata[key] != spec[key] for key in ("dataset", "model")):
        raise ValueError("Prepared data does not match checkpoint model/dataset")
    config = resolve_data_config(spec, repo_root, check_inputs=False)
    if data_protocol(config) != metadata["data_protocol"]:
        raise ValueError("Checkpoint evaluation data protocol differs from the frozen preparation")
    if spec.get("stage") == "gflow" and spec.get("source_family"):
        original_spec = {**spec, "data_config": str(Path(spec["source_family"]) / "data_reward_config.json")}
        original = resolve_data_config(original_spec, repo_root, check_inputs=False)
        if data_protocol(original) != metadata["data_protocol"]:
            raise ValueError("Original GFlow run data protocol differs from frozen evaluation")
    if metadata["fingerprint_inputs"]["code"] != _code_signatures(repo_root):
        raise ValueError("Preprocessing/model code changed after preparation; run CPU preparation again")
    _verify_files(prepared_dir, metadata["files_sha256"])
    if spec.get("preserve_legacy_checkpoint_path"):
        # Explicit opt-in for historical exports: rebase only the repository
        # prefix, never redirect an old checkpoint to a promoted replacement.
        from checkpoint_locations import portable
        checkpoint = repo_root / portable(spec["checkpoint"])
    else:
        checkpoint = resolve_project_path(spec["checkpoint"], repo_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
    import torch
    runner = _imports(repo_root)
    inputs = {"prepared_fingerprint": metadata["fingerprint"], "prepared_metadata_sha256": _sha256(meta_path),
              "checkpoint": _signature(checkpoint, contents=True), "num_beams": num_beams,
              "top_k": top_k, "batch_size": batch_size, "device": str(device), "dtype": "float32",
              "constrain_valid_sequences": True, "exclude_seen_items": False,
              "torch_version": torch.__version__, "libraries": _library_versions(),
              "code": _code_signatures(repo_root), "config": config, "run_spec": spec}
    fingerprint = _digest(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A legitimate new preparation can change the catalog or its train-only
    # popularity. Publish it before loading a model; untouched old run metadata
    # keeps its original fingerprints, so analysis rejects mixed generations
    # until all selected runs have been regenerated.
    catalog = output_dir / "catalog.csv"
    if not catalog.is_file() or _sha256(catalog) != metadata["files_sha256"]["catalog.csv"]:
        temporary_catalog = catalog.with_name(f".catalog.{spec['run_id']}.{os.getpid()}.tmp")
        temporary_catalog.write_bytes((prepared_dir / "catalog.csv").read_bytes())
        temporary_catalog.replace(catalog)
    output = output_dir / f"{spec['run_id']}.npz"
    previous = _json(output.with_suffix(".json"))
    if not force and previous.get("fingerprint") == fingerprint and output.is_file():
        _verify_files(output_dir, {output.name: previous["npz_sha256"], "catalog.csv": previous["catalog_sha256"]})
        print(f"[{spec['run_id']}] reuse verified recommendations", flush=True)
        return output
    with np.load(prepared_dir / "semantics.npz", allow_pickle=False) as artifact:
        semantic_ids = {str(i): [int(x) for x in s] for i, s in zip(artifact["item_ids"], artifact["semantic_ids"])}
    dm = SimpleNamespace(semantic_ids=semantic_ids, sid_len=metadata["sid_len"],
                         cfg=SimpleNamespace(max_history_len=metadata["max_history_len"], train_window_len=metadata["train_window_len"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Expected a FlowRec Trainer checkpoint: {checkpoint}")
    saved = payload.get("model_config", {})
    # Several TIGER architecture options are absent from saved model_config.
    # Use this run's original configuration even when test tensors are shared.
    build = copy.deepcopy(config)
    build.setdefault("model", {}).update(saved)
    build["model"].pop("checkpoint_path", None)
    if spec["model"] == "tiger":
        model = runner.build_model(build, dm, config_path=repo_root / "analysis_config.json")
    elif spec["model"] == "gpt2":
        model = importlib.import_module("run_gpt2").build_model(build, dm)
    else:
        raise ValueError(f"Unknown model: {spec['model']}")
    expected_sids = model.item_semantic_ids.detach().cpu().clone()
    built = model.config_dict()
    mismatch = {k: (built[k], v) for k, v in saved.items()
                if k in built and k not in {"num_beams", "num_return_sequences"} and built[k] != v}
    if mismatch:
        raise ValueError(f"Checkpoint architecture differs from frozen model configuration: {mismatch}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    if not torch.equal(expected_sids, model.item_semantic_ids.cpu()):
        raise ValueError("Checkpoint semantic-ID row table differs from frozen evaluation catalog")
    model.to(torch.device(device)).eval()
    batches = {key: np.load(prepared_dir / f"input_{key}.npy", mmap_mode="r", allow_pickle=False)
               for key in metadata["tensor_keys"]}
    raw_by_sid = {tuple(tokens): raw for raw, tokens in semantic_ids.items()}
    recommendations = []
    start, last_print = time.monotonic(), time.monotonic()
    with torch.inference_mode():
        for offset in range(0, metadata["num_users"], batch_size):
            batch = {key: torch.from_numpy(np.array(value[offset:offset + batch_size], copy=True)).to(device)
                     for key, value in batches.items()}
            generated = model.generate(batch, num_beams=num_beams, num_return_sequences=top_k,
                                       constrain_valid_sequences=True, exclude_seen_items=False).detach().cpu().tolist()
            rows = [[raw_by_sid.get(tuple(tokens), "") for tokens in row] for row in generated]
            if any(len(row) != top_k or "" in row or len(set(row)) != top_k for row in rows):
                raise ValueError("Constrained generation produced invalid, duplicate, or missing items")
            recommendations.extend(rows)
            now = time.monotonic()
            if now - last_print >= 20 or len(recommendations) == metadata["num_users"]:
                print(f"[{spec['run_id']}] {len(recommendations)}/{metadata['num_users']} users; {now - start:.1f}s", flush=True)
                last_print = now
    with np.load(prepared_dir / "examples.npz", allow_pickle=False) as artifact:
        arrays = {key: artifact[key] for key in artifact.files}
    recs = np.asarray(recommendations, dtype=str)
    if recs.shape != (metadata["num_users"], top_k):
        raise ValueError(f"Unexpected recommendation shape: {recs.shape}")
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays, recommendations=recs)
    temporary.replace(output)
    _atomic_json(output.with_suffix(".json"), {**spec, "fingerprint": fingerprint, "fingerprint_inputs": inputs,
                 "npz_sha256": _sha256(output), "catalog_sha256": _sha256(catalog),
                 "num_users": metadata["num_users"], "total_test_users": metadata["total_test_users"],
                 "catalog_size": metadata["catalog_size"], "test_context_sha256": metadata["test_context_sha256"],
                 "checkpoint_model_config": saved, "elapsed_seconds": time.monotonic() - start,
                 "invalid_recommendations": 0, "prepared_dir": str(prepared_dir)})
    return output
