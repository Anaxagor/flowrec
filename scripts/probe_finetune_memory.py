#!/usr/bin/env python
"""Measure the HOST memory one fine-tuning task needs for a dataset.

A fine-tuning task keeps the whole dataset in RAM (pandas frames with list
columns) and every dataloader worker gradually acquires its own copy of what it
reads (fork is copy-on-write, but touching a Python object rewrites its
reference count and so copies the page). The peak is therefore either the
datamodule build or, later, main process + workers. Slurm must be told this
number (--mem), or it packs array tasks onto a node that cannot hold them.

The probe builds the datamodule exactly as the base-matched TIGER runs do
(sft_finetune_control_matched.py: the checkpoint's captured config, the
protocol's batch sizes), then drives the train and validation loaders with the
requested number of workers and reports

  peak_build_gb      peak RSS of the main process during the build
  main_gb            main process after the build (PSS where available)
  workers_gb         sum over dataloader workers (PSS on Linux = shared pages
                     counted once; RSS elsewhere = an upper bound)
  recommended_mem_gb 1.3 x max(peak_build, main + workers), rounded up to 8 GB

Worker copies keep growing until a worker has touched every example it can
draw, so use --max-batches 0 (one full pass) for the number that matters; the
default stops early and extrapolates nothing.

  python scripts/probe_finetune_memory.py --dataset yambda_500m_listens --num-workers 8 --max-batches 0

Put the result into configs/exp2-tuned-params/finetune_base_matched.json
(cluster_resources.datasets.<name>.mem_gb). The cluster resource planner and
Slurm launcher are optional local tools and are not part of the repository.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "scripts" / "exp2-tuned-params")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _gb(num_bytes: float) -> float:
    return round(float(num_bytes) / 1e9, 2)


def _resident(proc) -> int:
    """PSS where the platform has it (Linux: shared copy-on-write pages are
    split between the processes that map them), else RSS."""
    try:
        info = proc.memory_full_info()
        return int(getattr(info, "pss", 0) or info.rss)
    except Exception:  # noqa: BLE001 - AccessDenied / platform without full info
        return int(proc.memory_info().rss)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024  # Linux reports KB


def recommended_mem_gb(peak_build_gb: float, main_gb: float, workers_gb: float, margin: float = 1.3) -> int:
    need = max(float(peak_build_gb), float(main_gb) + float(workers_gb)) * float(margin)
    return int(math.ceil(need / 8.0) * 8)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Host-memory probe of one base-matched fine-tuning task.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-workers", type=int, default=8, help="Dataloader workers per loader (the launchers pass 8)")
    p.add_argument("--max-batches", type=int, default=2000,
                   help="Train batches to draw before measuring the workers; 0 = one full pass (worst case)")
    p.add_argument("--val-batches", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import psutil

    import run_tiger as tiger_runner
    import sft_finetune_control as control
    import sft_finetune_control_matched as matched

    protocol = matched.load_protocol()
    control.DATASETS = matched.matched_datasets(protocol)
    if args.dataset not in control.DATASETS:
        raise SystemExit(f"--dataset must be one of {sorted(control.DATASETS)}")
    run_args = control.parse_args([*matched.matched_defaults(args.dataset, protocol), "--datasets", args.dataset,
                                   "--num-workers", str(int(args.num_workers))])
    config = control.build_run_config(args.dataset, 3e-5, 42, run_args)

    me = psutil.Process()
    started = time.time()
    dm = tiger_runner.build_data_module(
        tiger_runner.build_data_config(dict(config), config_path=PROJECT_ROOT / "probe.json"))
    dm.setup()
    gc.collect()
    report = {
        "dataset": args.dataset, "num_workers": int(args.num_workers),
        "train_examples": len(dm.train_dataset), "val_examples": len(dm.val_dataset),
        "items": int(dm.item_vocab_size), "build_sec": round(time.time() - started),
        "peak_build_gb": _gb(_peak_rss_bytes()), "main_after_build_gb": _gb(_resident(me)),
    }
    print(json.dumps(report), flush=True)

    # Keep both worker pools alive, as a run does (persistent workers).
    val_iter = iter(dm.val_dataloader())
    for _ in range(int(args.val_batches)):
        if next(val_iter, None) is None:
            break
    batches = 0
    for batches, _ in enumerate(dm.train_dataloader(), start=1):
        if args.max_batches and batches >= int(args.max_batches):
            break
    children = me.children(recursive=True)
    workers = sum(_resident(child) for child in children)
    report.update({
        "train_batches_drawn": batches, "full_pass": not args.max_batches,
        "worker_processes": len(children), "main_gb": _gb(_resident(me)), "workers_gb": _gb(workers),
        "memory_metric": "pss" if sys.platform.startswith("linux") else "rss (upper bound)",
    })
    report["recommended_mem_gb"] = recommended_mem_gb(
        report["peak_build_gb"], report["main_gb"], report["workers_gb"])
    print("RESULT " + json.dumps(report), flush=True)
    print(f"-> cluster_resources.datasets.{args.dataset}.mem_gb = {report['recommended_mem_gb']}", flush=True)


if __name__ == "__main__":
    main()
