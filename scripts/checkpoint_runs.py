"""Current primary checkpoint reads and flat primary-model publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


def _canonical_root(project_root: Path, checkpoint_root: Path) -> Path | None:
    root = Path(checkpoint_root)
    if not root.is_absolute():
        root = project_root / root
    try:
        relative = root.relative_to(project_root / "checkpoints")
    except ValueError:
        return None
    parts = relative.parts
    if (len(parts) == 2 and parts[0] == "exp1") or (
        len(parts) == 4 and parts[0] == "exp2"
        and parts[1] in {"loo_split", "global_split"} and parts[2] == "initial"
    ):
        return relative
    return None


def primary_checkpoint_dir(project_root: Path, checkpoint_root: Path, dataset: str) -> Path:
    """Selected dataset directory, including the recorded GPT-2 Yambda regime."""
    relative = _canonical_root(project_root, checkpoint_root)
    directory = Path(checkpoint_root) / dataset
    if relative is not None and relative.parts[0] == "exp2" and relative.name == "gpt2" and dataset.startswith("yambda_"):
        directory /= "last_128"
    return directory


def is_global_yambda_base(project_root: Path, checkpoint_root: Path, dataset: str) -> bool:
    relative = _canonical_root(project_root, checkpoint_root)
    return (
        relative is not None and relative.parts[:2] == ("exp2", "global_split")
        and relative.name in {"gpt2", "tiger"}
        and dataset in {"yambda_50m_listens", "yambda_500m_listens"}
    )


def validate_global_yambda_protocol(model: str, dataset: str, data: Mapping[str, Any]) -> None:
    """Check the actual training rows and temporal split of a selected Yambda base."""
    import yaml

    if not isinstance(data, Mapping) or not data:
        raise ValueError(f"{model}/{dataset}: missing global-time last_128 data protocol")
    config_path = Path(__file__).resolve().parents[1] / "configs/global-time/exp2-tuned-params/dataset_yambda.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {key: config[key] for key in (
        "split_strategy", "global_history_policy", "global_eval_target_mode",
        "filter_cold_items_after_split", "cold_item_train_source", "drop_items_missing_semantic_id",
        "max_history_len", "min_history_len", "hf_recent_months", "positive_criterion",
        "min_item_interactions", "min_user_interactions", "drop_items_missing_embedding",
        "deduplicate_consecutive", "organic_filter", "allowed_event_types", "min_played_ratio_pct",
    )}
    spec = config["datasets"][dataset]
    expected.update({key: spec[key] for key in ("dataset_size", "global_val_timestamp", "global_test_timestamp")})
    expected.update(train_target_policy="last_n", train_last_n=128, train_all_prefixes=True,
                    train_prefix_from_dense=False, max_rows=None)
    if model == "gpt2":
        expected.update(train_example_layout="dense_causal", train_dense_match_prefix=True,
                        train_window_len=expected["max_history_len"], train_window_stride=1)
    else:
        expected["train_example_layout"] = "prefix_target"
    mismatches = {key: (data.get(key), value) for key, value in expected.items() if data.get(key) != value}
    if mismatches:
        raise ValueError(f"{model}/{dataset}: different global-time last_128 data protocol: {mismatches}")


def recorded_training_data(directory: Path) -> Mapping[str, Any]:
    for name in ("resolved_config.json", "study_config.json"):
        path = directory / name
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            data = metadata.get("data", metadata.get("data_config"))
            if isinstance(data, Mapping) and data:
                return data
    return {}


def _selected_checkpoint(project_root: Path, checkpoint_root: Path, dataset: str) -> Path:
    path = primary_checkpoint_dir(project_root, checkpoint_root, dataset) / "best.pt"
    if path.is_file() and is_global_yambda_base(project_root, checkpoint_root, dataset):
        validate_global_yambda_protocol(Path(checkpoint_root).name, dataset, recorded_training_data(path.parent))
    return path


def checkpoint_run_root(project_root: Path, checkpoint_root: Path) -> Path:
    """Location for raw snapshots and multi-seed summaries."""
    relative = _canonical_root(project_root, checkpoint_root)
    if relative is not None and relative.parts[:2] == ("exp2", "global_split"):
        return project_root / "experiments/global-time/checkpoints/exp2-tuned_params" / relative.name
    return (project_root / "experiments/checkpoints" / relative
            if relative is not None else Path(checkpoint_root))


def training_checkpoint_dir(project_root: Path, checkpoint_root: Path,
                            dataset: str, seed: int) -> Path:
    """Keep per-seed best/last snapshots outside the curated model folders."""
    root = checkpoint_run_root(project_root, checkpoint_root)
    if is_global_yambda_base(project_root, checkpoint_root, dataset) and Path(checkpoint_root).name == "gpt2":
        root = root.with_name("gpt2_last_128")
    return root / dataset / f"seed_{seed}"


def seed_checkpoint_path(project_root: Path, checkpoint_root: Path,
                         dataset: str, seed: int, *, which: str = "best") -> Path:
    """Canonical best references always identify the current primary model."""
    relative = _canonical_root(project_root, checkpoint_root)
    if relative is not None and which == "best":
        return _selected_checkpoint(project_root, checkpoint_root, dataset)
    # Explicit last/custom-run requests use only the current experiment root.
    return training_checkpoint_dir(project_root, checkpoint_root, dataset, seed) / f"{which}.pt"


def incumbent_checkpoint_path(project_root: Path, checkpoint_root: Path,
                              dataset: str, *, legacy_name: str = "best_model.pt") -> Path:
    if _canonical_root(project_root, checkpoint_root) is not None:
        return _selected_checkpoint(project_root, checkpoint_root, dataset)
    return Path(checkpoint_root) / dataset / legacy_name


def publish_primary_checkpoint(project_root: Path, checkpoint_root: Path,
                               dataset: str, seed: int, checkpoint_path: str | Path | None,
                               *, run_dir: Path | None = None,
                               parameters: Mapping[str, Any] | None = None,
                               summary: Mapping[str, Any] | None = None) -> None:
    """Only seed42 publishes a primary model; explicit experiment roots stay private."""
    relative = _canonical_root(project_root, checkpoint_root)
    if relative is None or seed != 42 or checkpoint_path is None:
        return
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = primary_checkpoint_dir(project_root, checkpoint_root, dataset)
    global_yambda = is_global_yambda_base(project_root, checkpoint_root, dataset)
    if global_yambda or destination.name == "last_128":
        if run_dir is None or not (run_dir / "resolved_config.json").is_file():
            raise ValueError(f"Publishing {relative.name}/{dataset} requires the actual resolved training protocol")
        resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        data = resolved.get("data", resolved.get("data_config", {}))
        if global_yambda:
            validate_global_yambda_protocol(relative.name, dataset, data)
        else:
            required = {"split_strategy": "leave_one_out", "train_target_policy": "last_n",
                        "train_last_n": 128, "train_dense_match_prefix": True}
            if any(data.get(key) != value for key, value in required.items()):
                raise ValueError("Refusing to publish a different GPT-2 data protocol into selected last_128; use a separate experiment checkpoint root")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "best.pt"
    temporary = destination / ".best.pt.tmp"
    shutil.copy2(source, temporary)
    temporary.replace(target)
    sidecar = Path(str(source) + ".json")
    if sidecar.is_file():
        shutil.copy2(sidecar, destination / "best.pt.json")
    elif (destination / "best.pt.json").exists():
        (destination / "best.pt.json").unlink()
    # A newly trained primary must not retain a previous trial's metadata.
    for name in ("best_trial.json", "resolved_config.json", "study_config.json",
                 "data_audit.json", "data_summary.json", "training_summary.json"):
        stale = destination / name
        if stale.is_file():
            stale.unlink()
    if run_dir is not None and (run_dir / "resolved_config.json").is_file():
        shutil.copy2(run_dir / "resolved_config.json", destination / "resolved_config.json")
    if relative.parts[0] == "exp2" and parameters is not None:
        selection = {"dataset": dataset, "seed": seed, "checkpoint": str(target),
                     "best_params": dict(parameters),
                     "selection_reason": "Primary seed42 final training run"}
        (destination / "best_trial.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    if summary is not None:
        (destination / "training_summary.json").write_text(
            json.dumps({"datasets": [dataset], "seeds": [seed], "runs": [dict(summary)]},
                       indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    record = {"stage": "exp1" if relative.parts[0] == "exp1" else "initial",
              "split": relative.parts[1] if relative.parts[0] == "exp2" else None,
              "model": relative.name, "dataset": dataset, "seed": seed,
              "checkpoint": target.relative_to(project_root).as_posix(),
              "source_checkpoint": str(source), "sha256": digest.hexdigest()}
    if global_yambda:
        record["regime"] = "last_128"
    (destination / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
