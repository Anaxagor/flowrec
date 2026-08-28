"""Path resolution with explicit precedence and checksum verification.

Precedence for every asset (spec Part I §3):
  1. explicit CLI override;
  2. environment variables ``FLOWREC_DATA_ROOT`` / ``FLOWREC_PROJECT_ROOT``;
  3. repository-relative path;
  4. otherwise fail with a clear error (never silently substitute a dataset).

All resolved paths are printed before computation and stored in run metadata
as (logical, resolved) pairs together with SHA-256 checksums.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Logical (repository-relative) asset paths — the single source of truth.
LOGICAL_ASSETS: dict[str, str] = {
    "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt",
    "sasrec_checkpoint": "checkpoints/exp2-tuned_params/initial/sasrec/beauty/best_model.pt",
    "semantic_ids": "datasets/semantic_ids/exp2-tuned-params/beauty/semantic_ids.json",
    "sft_config": "configs/exp2-tuned-params/tiger_beauty_sft.json",
    "reviews": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
    "metadata": "data/raw/dataframe_format/meta_beauty_2014.parquet",
}


def find_project_root() -> Path:
    env = os.environ.get("FLOWREC_PROJECT_ROOT")
    if env:
        root = Path(env).resolve()
        if not (root / "flowrec" / "__init__.py").exists():
            raise FileNotFoundError(
                f"FLOWREC_PROJECT_ROOT={root} does not contain the flowrec package")
        return root
    here = Path(__file__).resolve()
    for cand in here.parents:
        # scripts/flowrec also exists, so require repo markers next to the package
        if (cand / "flowrec" / "__init__.py").exists() and (
                (cand / ".git").exists() or (cand / "requirements.txt").exists()):
            return cand
    raise FileNotFoundError(
        "cannot locate the FlowRec project root; set FLOWREC_PROJECT_ROOT")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ResolvedAsset:
    logical: str          # repository-relative path (portable identifier)
    path: Path            # resolved absolute path on this machine
    sha256: str | None    # checksum (None until computed)
    source: str           # which precedence rule resolved it


@dataclass
class PathResolver:
    project_root: Path = field(default_factory=find_project_root)
    data_root: Path | None = field(
        default_factory=lambda: Path(os.environ["FLOWREC_DATA_ROOT"]).resolve()
        if os.environ.get("FLOWREC_DATA_ROOT") else None)
    cli_overrides: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str, *, must_exist: bool = True) -> tuple[Path, str]:
        logical = LOGICAL_ASSETS.get(name, name)
        override = self.cli_overrides.get(name)
        if override:
            path, source = Path(override).expanduser().resolve(), "cli"
        elif self.data_root is not None and (self.data_root / logical).exists():
            path, source = (self.data_root / logical).resolve(), "FLOWREC_DATA_ROOT"
        else:
            path, source = (self.project_root / logical).resolve(), "project_root"
        if must_exist and not path.exists():
            raise FileNotFoundError(
                f"asset '{name}' (logical '{logical}') not found at {path}.\n"
                f"Precedence tried: CLI override -> FLOWREC_DATA_ROOT "
                f"({self.data_root}) -> project root ({self.project_root}).\n"
                f"Provide --asset-{name.replace('_', '-')} or set the environment "
                f"variables; the pipeline never substitutes a different dataset.")
        return path, source

    def resolve_all(
        self, names: list[str] | None = None, *, checksum: bool = True
    ) -> dict[str, ResolvedAsset]:
        out: dict[str, ResolvedAsset] = {}
        for name in names or list(LOGICAL_ASSETS):
            path, source = self.resolve(name)
            digest = sha256_file(path) if checksum else None
            out[name] = ResolvedAsset(
                logical=LOGICAL_ASSETS.get(name, name), path=path,
                sha256=digest, source=source)
        return out

    def print_resolved(self, assets: dict[str, ResolvedAsset]) -> None:
        print("resolved assets:", flush=True)
        for name, a in assets.items():
            digest = (a.sha256[:12] + "...") if a.sha256 else "-"
            print(f"  {name:18s} [{a.source}] {a.path}  sha256={digest}", flush=True)


def verify_known_checksums(
    assets: dict[str, ResolvedAsset], known_path: Path
) -> list[str]:
    """Compare against a stored asset_checksums.json when available.

    Returns the list of mismatched asset names; empty when everything matches
    or no stored checksums exist yet.
    """
    if not known_path.exists():
        return []
    known = json.loads(known_path.read_text())
    mismatches = []
    for name, a in assets.items():
        want = known.get(name, {}).get("sha256")
        if want and a.sha256 and want != a.sha256:
            mismatches.append(name)
    return mismatches


def write_checksums(assets: dict[str, ResolvedAsset], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {"logical": a.logical, "resolved": str(a.path), "sha256": a.sha256,
               "source": a.source}
        for name, a in assets.items()
    }
    path.write_text(json.dumps(payload, indent=2))
