"""Run manifests, split manifests, and the immutable selection manifest.

The selection manifest (spec §14.1) is hashed at write time; the test stage
refuses to run when the stored hash no longer matches, and every test
prediction shard records the selection hash it was produced under.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .common import atomic_write_json, atomic_write_parquet, environment_info, json_safe

SPLIT_MANIFEST_COLUMNS = [
    "raw_user_id", "internal_user_id", "dataset_index", "split",
    "history_item_ids", "history_length", "target_item_id",
    "target_in_catalog", "target_has_sid", "eligible", "exclusion_reason",
]


def manifest_dir(root: Path) -> Path:
    return root / "manifest"


def write_run_manifest(root: Path, payload: Mapping[str, Any]) -> None:
    data = dict(payload)
    data.setdefault("environment", environment_info())
    data.setdefault("written_unix", time.time())
    atomic_write_json(data, manifest_dir(root) / "run_manifest.json")


def read_run_manifest(root: Path) -> dict[str, Any]:
    return json.loads((manifest_dir(root) / "run_manifest.json").read_text())


def split_manifest_path(root: Path, split: str) -> Path:
    return manifest_dir(root) / f"{split}_users.parquet"


def write_split_manifest(root: Path, split: str, df: pd.DataFrame) -> None:
    missing = [c for c in SPLIT_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"split manifest missing columns {missing}")
    atomic_write_parquet(df[SPLIT_MANIFEST_COLUMNS], split_manifest_path(root, split))


def read_split_manifest(root: Path, split: str, *, eligible_only: bool = True) -> pd.DataFrame:
    path = split_manifest_path(root, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run 01_build_split_manifest.py first")
    df = pd.read_parquet(path)
    if eligible_only:
        df = df[df["eligible"]].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Selection manifest (immutable after test predictions exist)
# --------------------------------------------------------------------------- #
def selection_manifest_path(root: Path) -> Path:
    return manifest_dir(root) / "selected_configs_full_validation.json"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode()


def write_selection_manifest(root: Path, payload: Mapping[str, Any]) -> str:
    data = dict(payload)
    data["selection_timestamp"] = time.time()
    digest = hashlib.sha256(_canonical_bytes(data)).hexdigest()
    data["_selection_manifest_sha256"] = digest
    path = selection_manifest_path(root)
    atomic_write_json(data, path)
    (path.with_suffix(".sha256")).write_text(digest + "\n")
    return digest


def read_selection_manifest(root: Path, *, verify: bool = True) -> tuple[dict[str, Any], str]:
    path = selection_manifest_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run 06_select_validation_configs.py first")
    data = json.loads(path.read_text())
    stored = data.pop("_selection_manifest_sha256", None)
    digest = hashlib.sha256(_canonical_bytes(data)).hexdigest()
    if verify:
        side = path.with_suffix(".sha256")
        want = side.read_text().strip() if side.exists() else stored
        if want != digest or (stored is not None and stored != digest):
            raise RuntimeError(
                "selection manifest was MODIFIED after freezing "
                f"(stored {want} != recomputed {digest}); refusing to run test "
                "evaluation (spec 14.1)")
    data["_selection_manifest_sha256"] = digest
    return data, digest


def summary_row_base(
    *, run_id: str, git_commit: str | None, split: str, n_users: int,
    policy_family: str, canonical_config_hash: str, reward_transform: str | None,
    seed: int | None, cache_mode: str,
) -> dict[str, Any]:
    """Mandatory fields for every summary row (spec Part V)."""
    return {
        "run_id": run_id, "git_commit": git_commit, "split": split,
        "n_users": int(n_users), "policy_family": policy_family,
        "canonical_config_hash": canonical_config_hash,
        "reward_transform": reward_transform if reward_transform else "none",
        "seed": seed, "cache_mode": cache_mode,
    }
