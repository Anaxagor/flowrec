#!/usr/bin/env python3
"""Continue seed-42 SFT-only checkpoints with SFT, GFlow or sampled CE.

Metadata commands deliberately use only the standard library. One training
array element is one complete experiment, not one epoch. Preparation is split
into CPU reward factors and GPU teacher factors/baselines.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import sft_source_protocol as protocol

DATASETS = ("beauty", "instruments", "beauty_2023", "yambda_50m_listens", "yambda_500m_listens")
COMBOS = (("sasrec",), ("interaction",), ("similarity",), ("sasrec", "interaction", "similarity"))
POWERS = (0.25, 1.0, 16.0)
RHOS = (0.1, 1.0)
LOSSES = ("tb", "db", "subtb")
SAMPLINGS = ("reward", "curriculum_smooth_is")


def resource_profile(dataset):
    return "small" if dataset in ("beauty", "instruments") else "large" if dataset == "beauty_2023" else "yambda"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--models", nargs="+", choices=("gpt2", "tiger"), default=["gpt2", "tiger"])
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--source-ids", nargs="+", choices=list(protocol.SOURCE_SPECS))
    p.add_argument("--source-seeds", nargs="+", type=int, default=[42],
                   help="Source seed aliases use the current primary model's recorded seed")
    p.add_argument("--seeds", nargs="+", type=int, choices=[42], default=[42])
    p.add_argument("--arms", nargs="+", choices=("sft", "gflow", "ce"), default=["sft", "gflow", "ce"])
    p.add_argument("--resource-profile", choices=("small", "large", "yambda"))
    p.add_argument("--phase", choices=("prepare", "train"), default="train")
    p.add_argument("--prepare-kind", choices=("cpu", "gpu", "all"), default="all")
    p.add_argument("--reuse-tiger-gflow-caches", action="store_true",
                   help="Validate/import existing exp4 TIGER Beauty/Instruments reward factors; never rebuild them")
    p.add_argument("--run-index", type=int)
    p.add_argument("--count", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--check-sources", action="store_true")
    p.add_argument("--output-root", type=Path, default=ROOT / "experiments/exp8-from-sft")
    p.add_argument("--cache-root", type=Path, default=ROOT / "artifacts/exp8-from-sft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--cache-batch-size", type=int, default=256)
    p.add_argument("--cache-verify-batches", type=int, default=1)
    p.add_argument("--checkpoint-every-updates", type=int, default=100)
    p.add_argument("--traceback-interval", type=int, default=0,
                   help="Python stack snapshot interval in seconds (0 disables periodic snapshots; stage logs remain)")
    p.add_argument("--history-build-chunk-size", type=int, default=0,
                   help="Bound intermediate history expansion during preparation (0 keeps the original implementation)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    for name in ("models", "datasets", "source_ids", "source_seeds", "seeds", "arms"):
        value = getattr(a, name)
        if value and len(set(value)) != len(value):
            p.error(f"--{name.replace('_', '-')} contains duplicates")
    if any(seed < 0 for seed in a.source_seeds):
        p.error("--source-seeds must be nonnegative")
    for name in ("batch_size", "grad_accum", "max_epochs", "cache_batch_size"):
        if getattr(a, name) is not None and getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if min(a.num_workers, a.cache_verify_batches, a.checkpoint_every_updates) < 0:
        p.error("workers, verification and checkpoint intervals must be nonnegative")
    if a.traceback_interval < 0:
        p.error("--traceback-interval must be nonnegative")
    if a.history_build_chunk_size < 0:
        p.error("--history-build-chunk-size must be nonnegative")
    for name in ("output_root", "cache_root"):
        path = getattr(a, name)
        path = path if path.is_absolute() else ROOT / path
        setattr(a, name, path / "smoke" if a.smoke else path)
    if not (a.count or a.list or a.check_sources) and a.run_index is None:
        p.error("select one --run-index (see --list)")
    if a.reuse_tiger_gflow_caches:
        if a.smoke:
            p.error("--reuse-tiger-gflow-caches requires the full experiment, without --smoke")
        if any(job["source_id"] not in ("tiger_beauty", "tiger_instruments") for job in source_jobs(a)):
            p.error("--reuse-tiger-gflow-caches supports only --source-ids tiger_beauty tiger_instruments")
    return a


def source_jobs(args):
    ids = args.source_ids or protocol.default_source_ids()
    jobs = []
    selected = set()
    for source_id in ids:
        spec = protocol.SOURCE_SPECS[source_id]
        if spec.model not in args.models or spec.dataset not in args.datasets:
            continue
        if args.resource_profile and resource_profile(spec.dataset) != args.resource_profile:
            continue
        for seed in args.source_seeds:
            seed = protocol.primary_source_seed(source_id, seed)
            if (source_id, seed) in selected:
                continue
            selected.add((source_id, seed))
            jobs.append(dict(source_id=source_id, model=spec.model, dataset=spec.dataset, source_seed=seed))
    return jobs


def build_grid(args):
    cells = []
    for job, seed in itertools.product(source_jobs(args), args.seeds):
        common = {**job, "seed": seed}
        if "sft" in args.arms:
            cells.append({**common, "arm": "sft"})
        for combo, power, rho in itertools.product(COMBOS, POWERS, RHOS):
            if "gflow" in args.arms:
                cells.extend({**common, "arm": "gflow", "combo": combo, "power": power,
                              "rho": rho, "objective": objective, "num_negatives": 3}
                             for objective in LOSSES)
            if "ce" in args.arms:
                cells.extend({**common, "arm": "ce", "combo": combo, "power": power,
                              "rho": rho, "sampling": sampling, "num_samples": 3}
                             for sampling in SAMPLINGS)
    return cells


def cell_key(cell):
    return hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()[:20]


@contextmanager
def execution_diagnostics(args, job):
    """Log stages; optionally inspect retained Python frames under the GIL.

    Do not use faulthandler.dump_traceback_later here: its native watchdog
    walks a running thread's frames without the GIL and can itself segfault
    on CPython 3.11. Diagnostics must not introduce that crash path.
    """
    import time
    started = time.monotonic()

    def stage(name):
        print(json.dumps({"type": "exp8_stage", "source_id": job["source_id"],
                          "phase": args.phase, "prepare_kind": args.prepare_kind,
                          "stage": name, "elapsed_sec": round(time.monotonic() - started, 3)},
                         sort_keys=True), flush=True)

    if not args.traceback_interval:
        yield stage
        return

    import threading
    import traceback
    target_thread = threading.get_ident()
    stopped = threading.Event()
    output = sys.stderr

    def monitor():
        while not stopped.wait(args.traceback_interval):
            try:
                # _current_frames returns owned frame references. Only keep
                # source locations between snapshots, never frames/locals
                # that could retain the multi-gigabyte datamodule.
                frame = sys._current_frames().get(target_thread)
                if frame is None:
                    return
                try:
                    stack = traceback.extract_stack(frame, limit=100)
                finally:
                    del frame
                print(f"[exp8 diagnostic stack] elapsed_sec={time.monotonic() - started:.3f}\n"
                      + "".join(traceback.format_list(stack)), file=output, flush=True)
            except Exception:
                # A closed log or unavailable frame must not fail training.
                return

    thread = threading.Thread(target=monitor, name="exp8-diagnostics", daemon=True)
    thread.start()
    try:
        yield stage
    finally:
        stopped.set()
        thread.join(timeout=1.0)


def main(argv=None):
    args = parse_args(argv)
    jobs = source_jobs(args) if args.phase == "prepare" else build_grid(args)
    if args.count:
        print(len(jobs))
        return
    if args.list:
        for index, job in enumerate(jobs):
            print(json.dumps({"index": index, **job}, sort_keys=True))
        return
    if args.check_sources:
        for job in source_jobs(args):
            source = protocol.resolve_source(job["source_id"], job["source_seed"], hash_checkpoint=False)
            print(json.dumps({**job, "checkpoint": str(source["checkpoint"]),
                              "subset": source["subset"], "status": "verified"}, default=str))
        return
    if args.run_index < 0 or args.run_index >= len(jobs):
        raise SystemExit(f"--run-index must be in 0..{len(jobs) - 1}")
    job = jobs[args.run_index]
    with execution_diagnostics(args, job) as stage:
        stage("verify_source_and_hash_checkpoint")
        source = protocol.resolve_source(job["source_id"], job["source_seed"])
        stage("import_runtime")
        from from_sft_runtime import open_source, prepare, run_cell
        stage("open_source_session")
        with open_source(source, args, job.get("seed", 42)) as session:
            stage("prepare_" + args.prepare_kind if args.phase == "prepare" else "train")
            if args.phase == "prepare":
                history_context = nullcontext()
                if args.history_build_chunk_size:
                    from bounded_preparation import bounded_history_build
                    history_context = bounded_history_build(args.history_build_chunk_size)
                with history_context:
                    if args.reuse_tiger_gflow_caches:
                        from legacy_tiger_cache_import import prepare_reusing_legacy
                        prepare_reusing_legacy(session, args.prepare_kind)
                    else:
                        prepare(session, args.prepare_kind)
            else:
                run_cell(session, job)
        stage("completed")


if __name__ == "__main__":
    main()
