"""Locate the fixed analysis models without changing their run identities."""
from __future__ import annotations

import json
from pathlib import Path


ARCHIVE = Path("artifacts/checkpoint_archive/2026-10-30")


def portable(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        positions = [path.parts.index(marker)
                     for marker in ("checkpoints", "experiments", "datasets", "configs", "artifacts", "data")
                     if marker in path.parts]
        if positions:
            return Path(*path.parts[min(positions):])
    return path


def _logical(value: str | Path) -> Path:
    path = portable(value)
    if path.is_relative_to(ARCHIVE):
        return Path("checkpoints") / path.relative_to(ARCHIVE)
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
    """Normalize renamed initial assets for historical evidence checks."""
    path = _logical(value)
    previous = Path("checkpoints/exp2/initial")
    if path.is_relative_to(previous):
        return Path("checkpoints/exp2/loo_split/initial") / path.relative_to(previous)
    old = Path("checkpoints/exp2-tuned_params/initial")
    if path.is_relative_to(old):
        parts = list(path.relative_to(old).parts)
        if len(parts) >= 3:
            if parts[0] == "tiger" and parts[1] == "yambda_50m":
                parts[1] = "yambda_50m_listens"
                if parts[2] == "checkpoints":
                    del parts[2]
            if parts[-1] == "best_model.pt":
                parts[-1] = "best.pt"
            return Path("checkpoints/exp2/loo_split/initial") / Path(*parts)
    return path


def same_initial(left: str | Path, right: str | Path) -> bool:
    return initial_identity(left) == initial_identity(right)


def checkpoint_identity(value: str | Path) -> Path:
    """Upgrade saved initial/first-SFT paths without reading old weights."""
    path = initial_identity(value)
    logical = _logical(path)
    if logical.suffix != ".pt":
        return path
    parts = logical.parts
    model = dataset = None
    if len(parts) >= 5 and parts[:2] == ("checkpoints", "exp2-tuned_params") and parts[2].startswith("finetuned"):
        model, dataset = parts[3:5]
    elif len(parts) >= 5 and parts[:2] == ("checkpoints", "exp2") and parts[2] == "sft-finetuned":
        model, dataset = parts[3:5]
    elif len(parts) >= 5 and parts[0] == "experiments":
        if parts[1].startswith("exp2-sft-finetune") and parts[3].startswith("sft_"):
            model, dataset = "tiger", parts[2]
        elif parts[1].startswith("exp4-gflow-gpt2-amazon") and parts[3] == "sft_only":
            model, dataset = "gpt2", parts[2]
    if model and dataset:
        return selected_source(Path(), model, dataset)
    if logical.parts[:2] in {
        ("checkpoints", "exp1-standard_params"),
        ("checkpoints", "exp2-tuned_params"),
        ("checkpoints", "global-time"),
    } or path.is_relative_to(ARCHIVE):
        raise ValueError(f"Obsolete checkpoint path {value}; specify a model from the current checkpoints structure")
    return path
