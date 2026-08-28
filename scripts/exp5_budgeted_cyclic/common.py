"""Shared infrastructure: CLI, config, sharding, atomic IO, asset loading.

Every expensive stage supports the same portable arguments:
``--split --shard-id --num-shards --user-start --user-end --resume
--overwrite --dry-run --device --batch-size --num-workers --output-dir``.
Shard assignment is deterministic (contiguous chunks of the manifest order),
writes are atomic (temp file + rename), and every artifact records the device
and library versions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from .paths import PathResolver, ResolvedAsset, find_project_root

PROJECT_ROOT = find_project_root()
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common_args(p: argparse.ArgumentParser, *, needs_split: bool = True) -> None:
    p.add_argument("--config", default=None,
                   help="experiment YAML (default: configs/beauty2014_full.yaml)")
    p.add_argument("--run-id", default=None,
                   help="versioned run directory name (default: from config or 'dev')")
    p.add_argument("--output-dir", default=None,
                   help="override artifact root (default: artifacts/exp5_budgeted_cyclic/beauty2014/<run_id>)")
    if needs_split:
        p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--user-start", type=int, default=None,
                   help="evaluate manifest rows [user-start, user-end) instead of shard math")
    p.add_argument("--user-end", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip shards whose outputs already exist and validate")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print resolved paths + workload estimate, then exit")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    for name in ("tiger-checkpoint", "sasrec-checkpoint", "semantic-ids",
                 "sft-config", "reviews", "metadata"):
        p.add_argument(f"--asset-{name}", default=None, help=f"CLI override for asset {name}")


def resolver_from_args(args: argparse.Namespace) -> PathResolver:
    overrides = {}
    for name in ("tiger_checkpoint", "sasrec_checkpoint", "semantic_ids",
                 "sft_config", "reviews", "metadata"):
        v = getattr(args, f"asset_{name}", None)
        if v:
            overrides[name] = v
    return PathResolver(cli_overrides=overrides)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = Path(args.config) if args.config else (
        Path(__file__).parent / "configs" / "beauty2014_full.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["_config_path"] = str(cfg_path)
    if args.run_id:
        cfg["run_id"] = args.run_id
    cfg.setdefault("run_id", "dev")
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    cfg.setdefault("seed", 42)
    return cfg


def run_root(cfg: Mapping[str, Any], args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return (PROJECT_ROOT / "artifacts" / "exp5_budgeted_cyclic" / "beauty2014"
            / str(cfg["run_id"]))


# --------------------------------------------------------------------------- #
# Sharding / atomic IO
# --------------------------------------------------------------------------- #
def shard_rows(n_rows: int, args: argparse.Namespace) -> np.ndarray:
    """Deterministic contiguous shard of manifest row positions."""
    if args.user_start is not None or args.user_end is not None:
        lo = int(args.user_start or 0)
        hi = int(args.user_end if args.user_end is not None else n_rows)
        return np.arange(lo, min(hi, n_rows))
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard-id {args.shard_id} out of range [0, {args.num_shards})")
    chunks = np.array_split(np.arange(n_rows), args.num_shards)
    return chunks[args.shard_id]


def shard_name(shard_id: int) -> str:
    return f"shard_{shard_id:05d}"


def atomic_write_bytes(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a temp file in the same directory and atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    atomic_write_bytes(path, lambda tmp: df.to_parquet(tmp, index=False))


def atomic_write_json(payload: Any, path: Path) -> None:
    atomic_write_bytes(
        path, lambda tmp: tmp.write_text(json.dumps(json_safe(payload), indent=2)))


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
    atomic_write_bytes(path, _write)


def json_safe(obj: Any) -> Any:
    """Booleans stay booleans, numpy scalars become Python scalars (spec §11.3)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def environment_info(device: str = "cpu") -> dict[str, Any]:
    import torch
    import transformers

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "device": str(device),
        "gpu_model": gpu,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timestamp_unix": time.time(),
    }


def print_workload_estimate(stage: str, n_units: int, sec_per_unit: float) -> None:
    est = n_units * sec_per_unit
    print(f"[{stage}] workload estimate: {n_units} units x ~{sec_per_unit:.2f}s "
          f"= ~{est / 60:.1f} min on this shard", flush=True)


# --------------------------------------------------------------------------- #
# Frozen assets: datamodule + TIGER + SASRec + trie
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentAssets:
    resolver: PathResolver
    resolved: dict[str, ResolvedAsset]
    dm: Any
    tiger: Any
    sasrec: Any
    trie: Any
    row_to_item: np.ndarray
    item_to_row: np.ndarray
    device: Any


def build_run_config_dict(resolved: dict[str, ResolvedAsset], seed: int) -> dict[str, Any]:
    """The checkpoint's own training config with resolver-supplied data paths."""
    config = copy.deepcopy(json.loads(resolved["sft_config"].path.read_text()))
    config["seed"] = int(seed)
    config["comet"] = {"enabled": False}
    data = dict(config.get("data", {}))
    data["reviews_path"] = str(resolved["reviews"].path)
    data["metadata_path"] = str(resolved["metadata"].path)
    data["semantic_id_path"] = str(resolved["semantic_ids"].path)
    config["data"] = data
    dataloader = dict(config.get("dataloader", {}))
    dataloader.update(num_workers=0, persistent_workers=False, pin_memory=False)
    config["dataloader"] = dataloader
    return config


def load_experiment_assets(
    args: argparse.Namespace, cfg: Mapping[str, Any], *,
    need_models: bool = True, checksum: bool = False,
) -> ExperimentAssets:
    import dataclasses as _dc

    import torch

    import run_tiger as tiger_runner
    import sft_finetune_control as sft_control
    from flowrec.models.sasrec import SASRec, SASRecConfig
    from flowrec.trajectories.nonacyclic import SemanticIDTrie

    resolver = resolver_from_args(args)
    resolved = resolver.resolve_all(checksum=checksum)
    resolver.print_resolved(resolved)
    device = torch.device(args.device)

    run_config = build_run_config_dict(resolved, int(cfg.get("seed", 42)))
    cfg_dir = run_root(cfg, args) / "manifest"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg_dir / "resolved_run_config.json"
    atomic_write_json(run_config, config_path)

    data_cfg = tiger_runner.build_data_config(dict(run_config), config_path=config_path)
    if hasattr(data_cfg, "processed_cache_dir") and not data_cfg.processed_cache_dir:
        data_cfg = _dc.replace(
            data_cfg,
            processed_cache_dir=str(PROJECT_ROOT / "data" / "processed_cache"))
    dm = tiger_runner.build_data_module(data_cfg)
    dm.setup()

    tiger = sasrec = None
    if need_models:
        tiger = tiger_runner.build_model(dict(run_config), dm, config_path=config_path)
        sft_control.load_tiger_checkpoint(tiger, resolved["tiger_checkpoint"].path)
        tiger.to(device)
        _freeze(tiger)

        payload = torch.load(resolved["sasrec_checkpoint"].path, map_location="cpu")
        sasrec = SASRec(SASRecConfig(**payload["model_config"]))
        sasrec.load_state_dict(payload["model_state_dict"], strict=True)
        if int(sasrec.cfg.item_vocab_size) != int(dm.item_vocab_size):
            raise RuntimeError(
                f"SASRec vocab {sasrec.cfg.item_vocab_size} != datamodule {dm.item_vocab_size}")
        sasrec.to(device)
        _freeze(sasrec)

    row_to_item_t = tiger_runner.item_id_by_semantic_row(dm)
    if row_to_item_t is None:
        raise RuntimeError("datamodule has no semantic-row -> item mapping")
    row_to_item = row_to_item_t.cpu().numpy().astype(np.int64)
    item_to_row = tiger_runner.item_id_to_semantic_row(dm).cpu().numpy().astype(np.int64)

    if need_models:
        sids = tiger.item_semantic_ids.detach().cpu().numpy().astype(np.int64)
    else:
        sids = np.asarray([v for _, v in sorted(dm.semantic_ids.items())], dtype=np.int64)
    trie = SemanticIDTrie(sids)
    assert trie.num_items == len(row_to_item) == dm.num_items

    return ExperimentAssets(
        resolver=resolver, resolved=resolved, dm=dm, tiger=tiger, sasrec=sasrec,
        trie=trie, row_to_item=row_to_item, item_to_row=item_to_row, device=device)


def _freeze(model: Any) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def single_user_batch(dm: Any, split: str, index: int) -> dict[str, Any]:
    ds = dm.val_dataset if split == "validation" else dm.test_dataset
    ds_t, collate = dm._loader_parts(ds, dm._make_collator())
    batch = collate([ds_t[int(index)]])
    if int(batch["user_ids"].size(0)) != 1:
        raise AssertionError("expected a single-user batch")
    return batch


def user_batches(dm: Any, split: str, indices: list[int], batch_size: int):
    """Yield (indices_chunk, batch) over a fixed list of dataset indices."""
    ds = dm.val_dataset if split == "validation" else dm.test_dataset
    ds_t, collate = dm._loader_parts(ds, dm._make_collator())
    for i in range(0, len(indices), batch_size):
        chunk = [int(x) for x in indices[i: i + batch_size]]
        yield chunk, collate([ds_t[j] for j in chunk])


# --------------------------------------------------------------------------- #
# Sharded numpy caches (spec §18): atomic writes, resume, completeness checks
# --------------------------------------------------------------------------- #
def write_cache_shard(
    cache_dir: Path, shard_id: int, *, arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> Path:
    from . import SCHEMA_VERSION

    path = cache_dir / f"{shard_name(shard_id)}.npz"
    meta = dict(meta)
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta["shard_id"] = int(shard_id)
    meta["array_keys"] = sorted(arrays)
    atomic_write_npz(path, **arrays)
    atomic_write_json(meta, cache_dir / f"{shard_name(shard_id)}.meta.json")
    return path


def cache_shard_done(cache_dir: Path, shard_id: int) -> bool:
    return ((cache_dir / f"{shard_name(shard_id)}.npz").exists()
            and (cache_dir / f"{shard_name(shard_id)}.meta.json").exists())


class ShardCacheReader:
    """Reader over a directory of shard_XXXXX.npz caches keyed by dataset index."""

    def __init__(self, cache_dir: Path, *, expected_schema: str | None = None):
        from . import SCHEMA_VERSION

        self.cache_dir = Path(cache_dir)
        self.metas: list[dict[str, Any]] = []
        self.index: dict[int, tuple[Path, int]] = {}
        for meta_path in sorted(self.cache_dir.glob("shard_*.meta.json")):
            meta = json.loads(meta_path.read_text())
            want = expected_schema or SCHEMA_VERSION
            if meta.get("schema_version") != want:
                raise RuntimeError(
                    f"stale cache shard {meta_path}: schema "
                    f"{meta.get('schema_version')} != {want}; rebuild the cache")
            npz_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".npz"))
            if not npz_path.exists():
                raise RuntimeError(f"cache meta without data file: {npz_path}")
            for row, ds_idx in enumerate(meta["dataset_indices"]):
                key = int(ds_idx)
                if key in self.index:
                    raise RuntimeError(
                        f"duplicate user dataset_index {key} across cache shards")
                self.index[key] = (npz_path, row)
            self.metas.append(meta)
        self._loaded_path: Path | None = None
        self._loaded: Any = None

    def validate_complete(self, expected_indices: list[int]) -> None:
        missing = [i for i in expected_indices if int(i) not in self.index]
        if missing:
            raise RuntimeError(
                f"cache {self.cache_dir} is missing {len(missing)} users "
                f"(first: {missing[:5]}); rebuild the missing shards")

    def meta_field(self, key: str) -> Any:
        vals = {json.dumps(json_safe(m.get(key))) for m in self.metas}
        if len(vals) > 1:
            raise RuntimeError(f"inconsistent '{key}' across cache shards")
        return self.metas[0].get(key) if self.metas else None

    def arrays_for(self, dataset_index: int) -> tuple[Any, int]:
        path, row = self.index[int(dataset_index)]
        if self._loaded_path != path:
            self._loaded = np.load(path)
            self._loaded_path = path
        return self._loaded, row
