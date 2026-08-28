#!/usr/bin/env python
"""Stage 03: TIGER prefix-score cache (spec §18.2) + model-based validation.

Encodes each user history once and sweeps the trie prefixes in decoder
batches; stores per-user flat edge log-prob vectors (float32) plus the raw
valid-token mass diagnostic. Every shard validates random cached item
log-probabilities against ``score_items_valid_actions`` (spec §18.3).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic.compute_counters import ComputeCounters  # noqa: E402
from exp5_budgeted_cyclic.tiger_scoring import (  # noqa: E402
    compute_user_edge_logprobs,
    renormalized_edge_logprob,
    validate_cache_against_model,
)
from exp5_budgeted_cyclic.trie import trie_hash  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p)
    p.add_argument("--validate-items", type=int, default=16,
                   help="random items per probe user checked against the model")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    users = manifest.read_split_manifest(root, args.split)
    rows = common.shard_rows(len(users), args)
    cache_dir = root / "cache" / "tiger" / args.split
    common.print_workload_estimate("03_tiger", len(rows), 0.5)
    if args.dry_run:
        return
    if args.resume and not args.overwrite and common.cache_shard_done(cache_dir, args.shard_id):
        print(f"shard {args.shard_id} already complete; --resume skip", flush=True)
        return

    assets = common.load_experiment_assets(args, cfg, need_models=True)
    trie = assets.trie
    n_internal = int(trie.level_offsets[trie.sid_len])
    counters = ComputeCounters(cache_mode="online_serving")
    indices = [int(i) for i in users["dataset_index"].iloc[rows]]
    edge_lp = np.empty((len(indices), trie.num_nodes), dtype=np.float32)
    raw_mass = np.empty((len(indices), n_internal), dtype=np.float32)
    validation_stats = []
    rng = np.random.default_rng(cfg["seed"] + args.shard_id)
    t0 = time.time()
    for k, ds_idx in enumerate(indices):
        batch = common.single_user_batch(assets.dm, args.split, ds_idx)
        lp32, rvm = compute_user_edge_logprobs(
            assets.tiger, batch, trie,
            batch_size=args.batch_size or 512, counters=counters)
        edge_lp[k], raw_mass[k] = lp32, rvm
        if k < 2 or (k == len(indices) - 1):
            lp64 = renormalized_edge_logprob(trie, lp32)
            probe_rows = rng.integers(0, trie.num_items, size=args.validate_items)
            validation_stats.append(validate_cache_against_model(
                assets.tiger, batch, trie, lp64, rows=probe_rows))
        if (k + 1) % 25 == 0:
            print(f"  swept {k + 1}/{len(indices)} "
                  f"({(k + 1) / (time.time() - t0):.2f} users/s)", flush=True)
    counters.record_peaks()
    common.write_cache_shard(
        cache_dir, args.shard_id,
        arrays={"edge_logprob": edge_lp, "raw_valid_mass": raw_mass},
        meta={
            "kind": "tiger_prefix_scores", "split": args.split,
            "num_shards": args.num_shards,
            "dataset_indices": indices,
            "manifest_rows": [int(r) for r in rows],
            "trie_hash": trie_hash(trie),
            "num_internal_nodes": n_internal,
            "model_validation": validation_stats,
            "counters": counters.to_dict(),
            "environment": common.environment_info(args.device),
        })
    print(f"stage 03 shard {args.shard_id}/{args.num_shards} complete "
          f"({len(indices)} users, {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
