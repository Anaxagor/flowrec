"""Discover the Beauty controls and export matched, catalog-constrained rankings.

Heavy dependencies are imported only by ``generate_run``. Discovery therefore
works on a machine without PyTorch, transformers, or the model checkpoints.
Only local data and randomly initialized model constructors are used: there is
no model download, fitting, or modification of source experiments.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_DATA_CACHE: dict[str, Any] = {}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_project_path(value: str | Path, repo_root: Path) -> Path:
    """Rebase historical cluster paths at an explicit repository boundary."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return repo_root / path
    if path.is_file():
        return path
    parts = path.parts
    for marker in ("checkpoints", "experiments", "datasets", "configs", "data", "artifacts"):
        if marker in parts:
            rebased = repo_root.joinpath(*parts[parts.index(marker):])
            if rebased.is_file():
                return rebased
    return path


def _first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.is_file()), paths[0])


def discover_runs(repo_root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Return two shared initial checkpoints and their requested SFT controls.

    Deliberately select the dense-prefix GPT-2 arm used in the preceding
    discussion, and the exp2 TIGER control. Other legacy/fullhist experiments
    must never silently substitute for these runs. An initial checkpoint is
    represented once, since its weights are shared by all fine-tuning seeds.
    Missing weights are reported in ``available`` and fail only on inference.
    """
    repo_root = Path(repo_root).resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique integers")
    gpt_root = repo_root / "experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix/beauty"
    tiger_root = repo_root / "experiments/exp2-sft-finetune/beauty"
    initial_root = repo_root / "checkpoints/exp2-tuned_params/initial"
    gpt_data = gpt_root / "data_reward_config.json"
    tiger_data = repo_root / "configs/exp2-tuned-params/tiger_beauty_sft.json"
    gpt_csv = repo_root / "artifacts/exp4-reward_ablation/gflow_finetune_gpt2_amazon_fullhist_dense_prefix_metrics_summary.csv"
    tiger_csv = repo_root / "artifacts/exp2-tuned-params/tiger_sft_finetune_beauty_metrics_summary.csv"
    csv_metrics: dict[tuple[str, str, int], dict[str, float]] = {}
    for model, source in (("gpt2", gpt_csv), ("tiger", tiger_csv)):
        if source.is_file():
            with source.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("dataset") != "beauty" or row.get("split") != "test":
                        continue
                    variant = row.get("variant")
                    if variant not in {"init", "sft_only", "sft_finetune"}:
                        continue
                    stage = "init" if variant == "init" else "sft"
                    metrics = {key: float(value) for key, value in row.items()
                               if value and key.startswith(("HR@", "NDCG@", "Coverage@", "ARP@"))}
                    csv_metrics[(model, stage, int(row.get("seed") or 42))] = metrics

    runs = []
    for model in ("gpt2", "tiger"):
        initial = initial_root / model / "beauty" / ("best_model.pt" if model == "gpt2" else "best.pt")
        data_config = gpt_data if model == "gpt2" else tiger_data
        source_csv = gpt_csv if model == "gpt2" else tiger_csv
        family = "exp4-fullhist-dense-prefix-sft-only" if model == "gpt2" else "exp2-sft-finetune"
        baseline = f"{model}_init"
        for seed in [None, *seeds]:
            stage = "init" if seed is None else "sft"
            run_dir = None
            checkpoint = initial
            result_path = None
            resolved_path = None
            selected_config = data_config
            if seed is not None:
                if model == "gpt2":
                    run_dir = gpt_root / "sft_only" / f"seed_{seed}"
                    checkpoint = run_dir / "checkpoints/best.pt"
                    result_path = run_dir / "result.json"
                else:
                    run_dir = tiger_root / "sft_lr3e-05" / f"seed_{seed}"
                    promoted = repo_root / "checkpoints/exp2-tuned_params/finetuned/tiger/beauty/sft_only_lr3e-05" / f"seed_{seed}"
                    checkpoint = _first_existing(promoted / "best.pt", run_dir / "checkpoints/best.pt")
                    result_path = _first_existing(run_dir / "metrics.json", promoted / "metrics.json")
                    selected_config = _first_existing(run_dir / "run_config.json", promoted / "run_config.json", data_config)
                resolved_path = run_dir / "resolved_config.json"
            result = _json(result_path) if result_path else {}
            resolved = _json(resolved_path) if resolved_path else {}
            recorded_initial = resolved.get("init_checkpoint")
            if recorded_initial and resolve_project_path(recorded_initial, repo_root).resolve() != initial.resolve():
                raise ValueError(f"{resolved_path}: unexpected initial checkpoint {recorded_initial}")
            if result and result.get("status") not in (None, "completed"):
                raise ValueError(f"Incomplete SFT run: {result_path}")
            if model == "gpt2" and resolved and not resolved.get("cli_args", {}).get("prefix_from_dense", False):
                raise ValueError(f"Expected dense-prefix GPT-2 control: {resolved_path}")
            test_metrics = result.get("splits", {}).get("test", result.get("test", {}))
            test_metrics = test_metrics or csv_metrics.get((model, stage, 42 if seed is None else seed), {})
            runs.append({
                "run_id": baseline if seed is None else f"{model}_sft_seed{seed}",
                "model": model, "stage": stage, "seed": 42 if seed is None else seed,
                "baseline_run_id": baseline, "shared_initial_weights": True,
                "checkpoint": str(checkpoint), "available": checkpoint.is_file(),
                "init_checkpoint": str(initial), "dataset": "beauty", "split": "test",
                "data_config": str(selected_config),
                "best_trial": str(initial_root / "gpt2/beauty/best_trial.json") if model == "gpt2" else None,
                "resolved_config": str(resolved_path) if resolved_path else None,
                "result_path": str(result_path) if result_path else None,
                "source_metrics_csv": str(source_csv), "source_family": family,
                "run_dir": str(run_dir) if run_dir else None,
                "recorded_test_metrics": test_metrics,
                "protocol_note": (
                    "Shared Optuna winner best_model.pt; fine-tuning seeds are not independent base-training seeds."
                    if model == "gpt2" else
                    "Shared tuned best.pt; exp2 control validates every 5 epochs (GPT-2 every epoch)."
                ),
            })
    return runs


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _signature(path: Path, *, contents: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if contents:
        result["sha256"] = _sha256(path)
    return result


def _resolved_data_config(spec: dict[str, Any], repo_root: Path, batch_size: int) -> dict[str, Any]:
    path = resolve_project_path(spec["data_config"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing original evaluation data config: {path}")
    config = copy.deepcopy(_json(path))
    for name in ("reviews_path", "metadata_path", "semantic_id_path"):
        value = config.get("data", {}).get(name)
        if not value:
            raise ValueError(f"{path}: missing data.{name}")
        resolved = resolve_project_path(value, repo_root)
        if not resolved.is_file() and name in {"reviews_path", "metadata_path"}:
            resolved = repo_root / "data/raw/dataframe_format" / Path(value).name
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing local input {name}: {resolved}; no download is performed")
        config["data"][name] = str(resolved)
    config.setdefault("dataloader", {}).update(
        validation_batch_size=batch_size, num_workers=0, persistent_workers=False,
        pin_memory=False, include_raw_in_batch=True,
    )
    return config


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def generate_run(
    spec: dict[str, Any], *, repo_root: Path, output_dir: Path, device: str,
    batch_size: int, num_beams: int, top_k: int, max_users: int | None,
    sample_seed: int, force: bool = False,
) -> Path:
    """Export one FP32 eval-mode run, reusing only fingerprint-matched caches.

    ``history_lengths`` counts the actual clipped model input. Additional
    ``history_fingerprints`` permit exact cross-model context validation, not
    merely matching context lengths. The test subset is selected by a stable
    hash of raw user ID, independently of dataloader ordering or model family.
    """
    if batch_size < 1 or top_k < 1 or num_beams < top_k:
        raise ValueError("Require batch_size >= 1 and num_beams >= top_k >= 1")
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive or None")
    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    checkpoint = resolve_project_path(spec["checkpoint"], repo_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
    config = _resolved_data_config(spec, repo_root, batch_size)
    # Import no training runner until the complete local-input check succeeds.
    for directory in (repo_root, repo_root / "scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader, Subset
    tiger_runner = importlib.import_module("run_tiger")
    gpt_runner = importlib.import_module("run_gpt2")
    popularity_runner = importlib.import_module("sasrec_config_runner")

    sources = [_signature(checkpoint, contents=True),
               _signature(resolve_project_path(spec["data_config"], repo_root), contents=True)]
    if spec.get("best_trial"):
        sources.append(_signature(resolve_project_path(spec["best_trial"], repo_root), contents=True))
    sources.extend(_signature(Path(value), contents=True) for value in config["data"].values()
                   if isinstance(value, str) and Path(value).is_file())
    code_paths = [Path(__file__), repo_root / "scripts/run_tiger.py", repo_root / "scripts/run_gpt2.py",
                  repo_root / "scripts/sasrec_config_runner.py"]
    code_paths.extend(sorted((repo_root / "flowrec/data").glob("*.py")))
    code_paths.extend(sorted((repo_root / "flowrec/models").glob("*.py")))
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION, "run_id": spec["run_id"], "sources": sources,
        "code_sha256": {str(p.relative_to(repo_root)): _sha256(p) for p in code_paths},
        "config": config, "device": str(device), "dtype": "float32", "batch_size": batch_size,
        "num_beams": num_beams, "top_k": top_k, "constrain_valid_sequences": True,
        "exclude_seen_items": False, "max_users": max_users, "sample_seed": sample_seed,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{spec['run_id']}.npz"
    meta_path = output.with_suffix(".json")
    catalog_path = output_dir / "catalog.csv"
    previous = _json(meta_path)
    if not force and output.is_file() and catalog_path.is_file() and previous.get("fingerprint") == fingerprint:
        if previous.get("npz_sha256") == _sha256(output) and previous.get("catalog_sha256") == _sha256(catalog_path):
            print(f"[{spec['run_id']}] reuse verified recommendations: {output}", flush=True)
            return output

    data_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if data_key not in _DATA_CACHE:
        data_cfg = tiger_runner.build_data_config(config, config_path=repo_root / "analysis_config.json")
        dm = tiger_runner.build_data_module(data_cfg)
        dm.setup()
        _DATA_CACHE[data_key] = dm
    dm = _DATA_CACHE[data_key]
    popularity = popularity_runner.training_item_popularity(dm).cpu().numpy()
    catalog = sorted((str(raw), int(popularity[index])) for raw, index in dm.item2id.items())
    if len(catalog) != dm.num_items:
        raise ValueError("Catalog contains a padding entry or differs from the datamodule item count")
    if catalog_path.is_file():
        with catalog_path.open(newline="", encoding="utf-8") as handle:
            previous_catalog = [(row["item_id"], int(row["popularity"])) for row in csv.DictReader(handle)]
        if previous_catalog != catalog:
            raise ValueError(f"{spec['run_id']}: catalog/popularity differs from previous run; use a separate output directory")
    else:
        with catalog_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id", "popularity"])
            writer.writerows(catalog)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Expected a FlowRec Trainer checkpoint: {checkpoint}")
    saved_config = payload.get("model_config", {})
    if spec["model"] == "tiger":
        build_config = copy.deepcopy(config)
        build_config["model"].pop("checkpoint_path", None)
        model = tiger_runner.build_model(build_config, dm, config_path=repo_root / "analysis_config.json")
    elif spec["model"] == "gpt2":
        best = _json(resolve_project_path(spec["best_trial"], repo_root))
        values = {**best.get("model", {}), **saved_config, "user_ids_count": 1, "top_k": top_k}
        model = gpt_runner.build_model({"model": values}, dm)
    else:
        raise ValueError(f"Unknown model: {spec['model']}")
    expected_sids = model.item_semantic_ids.detach().cpu().clone()
    built_config = model.config_dict()
    mismatch = {key: (built_config.get(key), value) for key, value in saved_config.items()
                if key not in {"num_beams", "num_return_sequences"} and key in built_config
                and built_config[key] != value}
    if mismatch:
        raise ValueError(f"Rebuilt architecture does not match {checkpoint}: {mismatch}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    if not torch.equal(expected_sids, model.item_semantic_ids.cpu()):
        raise ValueError(f"{checkpoint}: checkpoint semantic IDs differ from the evaluation catalog")
    model.to(torch.device(device)).eval()
    raw_by_sid = {tuple(int(t) for t in tokens): str(raw) for raw, tokens in dm.semantic_ids.items()}
    if len(raw_by_sid) != len(dm.semantic_ids):
        raise ValueError("Semantic IDs are not unique; raw-item ranking analysis would be ambiguous")
    if set(raw_by_sid.values()) != {item for item, _ in catalog}:
        raise ValueError("Semantic-ID table and evaluation catalog differ")

    frame = dm.test_dataset.examples
    user_ids = frame["raw_user_id"].astype(str).tolist()
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("Expected exactly one held-out test target per raw user")
    indices = list(range(len(frame)))
    if max_users is not None and max_users < len(indices):
        indices.sort(key=lambda i: hashlib.sha256(f"{sample_seed}:{user_ids[i]}".encode()).hexdigest())
        indices = indices[:max_users]
    indices.sort(key=lambda i: user_ids[i])
    selected = frame.iloc[indices]
    histories = [list(map(str, h[-int(dm.cfg.max_history_len):])) for h in selected["history_raw_items"]]
    user_array = selected["raw_user_id"].astype(str).to_numpy(dtype=str)
    target_array = selected["target_raw_item"].astype(str).to_numpy(dtype=str)
    history_fingerprints = np.asarray([hashlib.sha256(json.dumps(h).encode()).hexdigest() for h in histories])
    base_loader = dm.test_dataloader()
    loader = DataLoader(Subset(dm.test_dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=base_loader.collate_fn)
    recommendations: list[list[str]] = []
    start = time.monotonic()
    last_print = start
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw_batch.items()}
            generated = model.generate(batch, num_beams=num_beams, num_return_sequences=top_k,
                                       constrain_valid_sequences=True).detach().cpu().tolist()
            recommendations.extend([[raw_by_sid.get(tuple(tokens), "") for tokens in row] for row in generated])
            now = time.monotonic()
            if now - last_print >= 20 or len(recommendations) == len(indices):
                print(f"[{spec['run_id']}] {len(recommendations)}/{len(indices)} users; {now - start:.1f}s", flush=True)
                last_print = now
    rec_array = np.asarray(recommendations, dtype=str)
    if rec_array.shape != (len(indices), top_k):
        raise ValueError(f"Unexpected recommendation shape: {rec_array.shape}")
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, user_ids=user_array, target_item_ids=target_array,
                        history_lengths=np.asarray(list(map(len, histories)), dtype=np.int64),
                        history_fingerprints=history_fingerprints, recommendations=rec_array)
    temporary.replace(output)
    metadata = {
        **spec, "fingerprint": fingerprint, "fingerprint_inputs": fingerprint_payload,
        "npz_sha256": _sha256(output), "catalog_sha256": _sha256(catalog_path),
        "num_users": len(indices), "total_test_users": len(frame), "catalog_size": len(catalog),
        "invalid_recommendations": int((rec_array == "").sum()),
        "elapsed_seconds": time.monotonic() - start,
        "test_context_sha256": hashlib.sha256(json.dumps(list(zip(user_array.tolist(), target_array.tolist(), history_fingerprints.tolist()))).encode()).hexdigest(),
        "checkpoint_model_config": saved_config,
    }
    _atomic_json(meta_path, metadata)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()
    return output
