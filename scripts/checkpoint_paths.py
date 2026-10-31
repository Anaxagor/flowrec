"""Current asset paths, historical identity comparison, and primary publication."""
from pathlib import Path
import json
import math


# This prefix is recognized only in historical provenance strings, never read.
ARCHIVE_ROOT = Path("artifacts/checkpoint_archive/2026-10-30")
_OLD_INITIAL = Path("checkpoints/exp2-tuned_params/initial")
_INITIAL = Path("checkpoints/exp2/loo_split/initial")


def _canonical_initial_relative(relative: Path) -> Path:
    parts = list(relative.parts)
    if len(parts) < 3:
        return relative
    if parts[:2] == ["gpt2", "legacy"] and len(parts) == 4:
        parts = ["gpt2", parts[2], "full_history", parts[3]]
    elif parts[:2] == ["tiger", "yambda_50m"]:
        parts[1] = "yambda_50m_listens"
        if parts[2:3] == ["checkpoints"]:
            del parts[2]
    if len(parts) in (3, 4):
        parts[-1] = {"best_model.pt": "best.pt", "best_model.pt.json": "best.pt.json"}.get(parts[-1], parts[-1])
    if len(parts) == 3 and parts[0] == "gpt2" and parts[1].startswith("yambda_"):
        parts.insert(2, "last_128")
    return Path(*parts)


def initial_asset_path(project_root: Path, value: str | Path) -> Path:
    """Resolve the configured asset path; never search old layouts or archives."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def checkpoint_path_matches(project_root: Path, recorded: str | None, expected: str) -> bool:
    """Recognize either the historical or current path across cluster roots."""
    if not recorded:
        return False
    def identity(value: str | Path) -> str:
        path = Path(value)
        if path.is_absolute():
            if path.is_relative_to(project_root):
                path = path.relative_to(project_root)
            elif "/flowrec/" in path.as_posix():
                path = Path(path.as_posix().split("/flowrec/", 1)[1])
        if path.is_relative_to(ARCHIVE_ROOT):
            path = Path("checkpoints") / path.relative_to(ARCHIVE_ROOT)
        for prefix in (_OLD_INITIAL, Path("checkpoints/exp2/initial"), _INITIAL):
            if path.is_relative_to(prefix):
                path = _INITIAL / _canonical_initial_relative(path.relative_to(prefix))
                break
        return path.as_posix().rstrip("/")

    return identity(recorded) == identity(expected)


def sft_promotion_directory(project_root: Path, promote_root: Path, dataset: str,
                            lr: float, seed: int, source_checkpoint: Path) -> Path | None:
    """Protect the selected first-SFT family when publishing a flat primary model."""
    canonical = project_root / "checkpoints/exp2/loo_split/sft-finetuned/tiger"
    if promote_root != canonical:
        return promote_root / dataset / f"sft_only_lr{lr:g}" / f"seed_{seed}"
    if seed != 42 or not math.isclose(lr, 3e-5, rel_tol=1e-9):
        return None
    destination = canonical / dataset
    provenance = destination / "provenance.json"
    if provenance.is_file():
        selected = json.loads(provenance.read_text(encoding="utf-8"))

        def relative(value: str | Path) -> str:
            path = Path(value)
            if path.is_relative_to(project_root):
                return path.relative_to(project_root).as_posix()
            return path.as_posix().split("/flowrec/", 1)[-1]

        if selected.get("seed") != seed or relative(selected.get("source_checkpoint", "")) != relative(source_checkpoint):
            return None
    elif (destination / "best.pt").exists():
        # An existing model without provenance is not an invitation to replace it.
        return None
    return destination
