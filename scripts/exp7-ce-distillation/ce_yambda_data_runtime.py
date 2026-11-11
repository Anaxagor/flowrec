"""Prepare the fixed CE target subset once, outside GPU training jobs.

Only metadata helpers are imported by --count/--list. The compact snapshot
implementation and ML dependencies are loaded at runtime on compute nodes.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATASET = "yambda_500m_listens"
DEFAULT_ROOT = ROOT / "artifacts/exp7-ce-distillation-prepared"
OPERATIONAL = frozenset({
    "batch_size", "eval_batch_size", "test_batch_size", "num_workers",
    "pin_memory", "persistent_workers", "train_drop_last", "tensorized_batches",
    "seed",  # Under this fixed last_n/leave_one_out protocol, only loader shuffle.
})


def configure_parser(parser):
    parser.add_argument("--prepare-data-only", action="store_true",
                        help="500M only: CPU preparation of compact sampled train/val/test arrays")
    parser.add_argument("--prepared-data-root", type=Path,
                        help="Shared compact data parent; default artifacts/exp7-ce-distillation-prepared")
    parser.add_argument("--require-prepared-data", action="store_true",
                        help="500M training: refuse to rebuild the full dataset if CPU preparation is missing")


def validate_args(args):
    if args.prepare_data_only:
        if args.datasets != [DATASET] or args.cache_only or args.index is not None:
            raise SystemExit("--prepare-data-only requires --datasets yambda_500m_listens and no --cache-only/--index")
        if args.device != "cpu" and not (args.count or args.list):
            raise SystemExit("--prepare-data-only requires --device cpu; submit the CPU preparation stage")


def directory(args, model):
    root = args.prepared_data_root or DEFAULT_ROOT
    root = root if root.is_absolute() else ROOT / root
    if args.smoke:
        root = args.output_root / "prepared_data" if args.prepared_data_root is None else root / "smoke"
    return root / model / DATASET / f"sampled{args.train_random_n}_seed{args.train_subset_seed}"


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def _file_identity(path):
    path = Path(path)
    stat = path.stat()
    if not path.is_file():
        raise RuntimeError(f"Prepared CE requires a local input file: {path}")
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _code_identity():
    files = [*sorted((ROOT / "flowrec/data").rglob("*.py")),
             ROOT / "scripts/run_tiger.py", ROOT / "scripts/precompute_reward_cache.py",
             ROOT / "scripts/gpt2_matched_finetune.py", Path(__file__).resolve(),
             Path(__file__).with_name("ce_yambda_prepared_data.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def identity(args, model, cfg, config):
    # These conditions make ignoring the model/shuffle seed safe. The fixed
    # sampling seed is explicitly included below; random split policies are
    # intentionally unsupported by this cache.
    if (cfg.split_strategy != "leave_one_out" or cfg.train_target_policy != "last_n"
            or cfg.train_last_n != 128 or cfg.train_example_layout != "prefix_target"
            or cfg.train_prefix_from_dense or cfg.train_dense_match_prefix):
        raise RuntimeError("Compact CE data require the audited Yambda last128 prefix pool")
    if cfg.use_hf or cfg.include_raw_in_batch:
        raise RuntimeError("Compact CE data require local files and include_raw_in_batch=False")
    assets = {str(getattr(cfg, key)): _file_identity(getattr(cfg, key))
              for key in ("interactions_path", "semantic_id_path", "embeddings_path")
              if getattr(cfg, key, None) is not None}
    return _json({
        "version": 1, "model": model, "dataset": DATASET,
        "data_config": {k: v for k, v in asdict(cfg).items() if k not in OPERATIONAL},
        "subset": {"random_n": args.train_random_n, "seed": args.train_subset_seed, "pool_last_n": 128},
        "source_trial": config.get("initial_model", {}).get("trial"),
        "source_audit": config.get("dataset", {}).get("matched_tuning_data_audit"),
        "assets": assets, "code": _code_identity(),
        "runtime": {"python": list(sys.version_info[:2]),
                    **{name: version(name) for name in ("numpy", "pandas", "polars", "torch")}},
    })


def prepare_once(args, model, base, b, config, config_path, audit_fn=None):
    import ce_yambda_prepared_data as prepared
    cfg = base.tiger_runner.build_data_config(config, config_path=config_path)
    expected = identity(args, model, cfg, config)
    destination = directory(args, model)
    destination.mkdir(parents=True, exist_ok=True)
    # This CPU-stage lock is acquired BEFORE loading any raw data. GPU workers
    # never wait for it: they require a committed manifest and fail fast.
    with base.control.file_lock(destination / "preparation.lock"):
        if prepared.check(destination, expected) is not None:
            print(f"[prepared CE] already ready: {destination}", flush=True)
            return
        started = time.perf_counter()
        base.tiger_runner.seed_everything(42)
        dm = base.tiger_runner.build_data_module(cfg)
        dm.setup()
        audit = audit_fn(base, dm, b) if audit_fn is not None else {
            "source_summary": dm.summary(), "target_selection": f"sampled{args.train_random_n}"}
        indices = base.finetune_indices(dm.train_dataset, b)
        if indices is None:
            raise RuntimeError("500M preparation must select a fixed sampled subset")
        if identity(args, model, cfg, config) != expected:
            raise RuntimeError("Input data/code changed during CPU preparation")
        manifest = prepared.prepare(destination, dm, indices, expected, audit=audit)
        print(f"[prepared CE] ready in {time.perf_counter() - started:.1f}s: {destination}", flush=True)
        return manifest


def load_or_build(args, model, base, b, config, config_path, dataset):
    cfg = base.tiger_runner.build_data_config(config, config_path=config_path)
    if dataset == DATASET:
        destination = directory(args, model)
        if (args.require_prepared_data or args.prepared_data_root is not None
                or (destination / "manifest.json").is_file()):
            import ce_yambda_prepared_data as prepared
            dm = prepared.load(destination, cfg, identity(args, model, cfg, config))
            if dm is None:
                raise RuntimeError(f"Prepared CE data are missing: {destination}; run the CPU --prepare-data-only stage first")
            print(f"[prepared CE] using {len(dm.train_dataset):,} fixed train rows via mmap: {destination}", flush=True)
            return dm
    dm = base.tiger_runner.build_data_module(cfg)
    dm.setup()
    return dm
