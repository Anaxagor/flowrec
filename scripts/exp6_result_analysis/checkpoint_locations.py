"""Locate the fixed analysis models without changing their run identities."""
from __future__ import annotations

import json
from pathlib import Path


def portable(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        positions = [path.parts.index(marker)
                     for marker in ("checkpoints", "experiments", "datasets", "configs", "artifacts", "data")
                     if marker in path.parts]
        if positions:
            return Path(*path.parts[min(positions):])
    return path


def selected_source(repo_root: Path, model: str, dataset: str) -> Path:
    """The primary path remains authoritative when its weights are missing."""
    return repo_root / "checkpoints/exp2/loo_split/sft-finetuned" / model / dataset / "best.pt"


def selected_seeds(repo_root: Path, model: str, dataset: str, requested) -> list[int]:
    """Represent the primary model once and label its actual recorded seed."""
    canonical = selected_source(repo_root, model, dataset)
    provenance = canonical.parent / "provenance.json"
    if not canonical.is_file() and not provenance.is_file():
        # Keep one missing-model row for the historical seed42 reference.
        # Inference will fail at the current canonical path.
        return [42]
    record = json.loads(provenance.read_text()) if provenance.is_file() else {}
    seed = record.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"{provenance}: primary checkpoint must record its actual seed for analysis")
    return [seed]


def initial_identity(value: str | Path) -> Path:
    """Compare current checkpoint paths independently of checkout location."""
    return portable(value)


def same_initial(left: str | Path, right: str | Path) -> bool:
    return initial_identity(left) == initial_identity(right)


def checkpoint_identity(value: str | Path) -> Path:
    """Keep explicitly selected current paths without substituting weights."""
    path = portable(value)
    if path.suffix == ".pt" and path.parts[:1] == ("checkpoints",):
        roots = (Path("checkpoints/exp1"), Path("checkpoints/exp2/loo_split"),
                 Path("checkpoints/exp2/global_split"))
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"Unsupported checkpoint path {value}; choose a model from the current checkpoints structure")
    return path
