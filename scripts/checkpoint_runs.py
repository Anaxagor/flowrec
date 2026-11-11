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
        and parts[1] == "loo_split" and parts[2] == "initial"
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


def checkpoint_run_root(project_root: Path, checkpoint_root: Path) -> Path:
    """Location for raw snapshots and multi-seed summaries."""
    relative = _canonical_root(project_root, checkpoint_root)
    return (project_root / "experiments/checkpoints" / relative
            if relative is not None else Path(checkpoint_root))


def training_checkpoint_dir(project_root: Path, checkpoint_root: Path,
                            dataset: str, seed: int) -> Path:
    """Keep per-seed best/last snapshots outside the curated model folders."""
    return checkpoint_run_root(project_root, checkpoint_root) / dataset / f"seed_{seed}"


def seed_checkpoint_path(project_root: Path, checkpoint_root: Path,
                         dataset: str, seed: int, *, which: str = "best") -> Path:
    """Canonical best references always identify the current primary model."""
    relative = _canonical_root(project_root, checkpoint_root)
    if relative is not None and which == "best":
        return primary_checkpoint_dir(project_root, checkpoint_root, dataset) / "best.pt"
    # Explicit last/custom-run requests use only the current experiment root.
    return training_checkpoint_dir(project_root, checkpoint_root, dataset, seed) / f"{which}.pt"


def incumbent_checkpoint_path(project_root: Path, checkpoint_root: Path,
                              dataset: str, *, legacy_name: str = "best_model.pt") -> Path:
    name = "best.pt" if _canonical_root(project_root, checkpoint_root) is not None else legacy_name
    return primary_checkpoint_dir(project_root, checkpoint_root, dataset) / name


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
    if destination.name == "last_128":
        if run_dir is None or not (run_dir / "resolved_config.json").is_file():
            raise ValueError("Publishing a regime-specific GPT-2 base requires the actual resolved training protocol")
        resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        data = resolved.get("data", resolved.get("data_config", {}))
        required = {"split_strategy": "leave_one_out", "train_target_policy": "last_n",
                    "train_last_n": 128, "train_dense_match_prefix": True}
        if any(data.get(key) != value for key, value in required.items()):
            raise ValueError(f"Refusing to publish a different GPT-2 data protocol into selected {destination.name}; use a separate experiment checkpoint root")
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
    (destination / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
