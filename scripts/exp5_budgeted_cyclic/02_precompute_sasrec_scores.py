#!/usr/bin/env python
"""Stage 02: SASRec full-catalog raw-score cache (spec §18.1).

Sharded float32 arrays [n_shard_users, num_items] in semantic-row order with
the item-order hash in every shard meta. Raw scores, not transformed rewards.
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
from exp5_budgeted_cyclic.sasrec_scoring import (  # noqa: E402
    item_order_hash,
    score_user_batch,
    validate_batch_vs_single,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p)
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    users = manifest.read_split_manifest(root, args.split)
    rows = common.shard_rows(len(users), args)
    cache_dir = root / "cache" / "sasrec" / args.split
    common.print_workload_estimate("02_sasrec", len(rows), 0.01)
    if args.dry_run:
        return
    if args.resume and not args.overwrite and common.cache_shard_done(cache_dir, args.shard_id):
        print(f"shard {args.shard_id} already complete; --resume skip", flush=True)
        return

    assets = common.load_experiment_assets(args, cfg, need_models=True)
    counters = ComputeCounters(cache_mode="online_serving")
    indices = [int(i) for i in users["dataset_index"].iloc[rows]]
    batch_size = args.batch_size or 256

    # batched-vs-single consistency probe FIRST (spec 29): a failure must not
    # discard a fully scored shard
    probe = indices[: min(4, len(indices))]
    err = validate_batch_vs_single(
        assets.sasrec, assets.dm, args.split, probe, assets.row_to_item, assets.device)
    print(f"  batch-vs-single probe OK (max err {err:.2e})", flush=True)

    out = np.empty((len(indices), assets.trie.num_items), dtype=np.float32)
    t0 = time.time()
    pos = 0
    for chunk, batch in common.user_batches(assets.dm, args.split, indices, batch_size):
        out[pos: pos + len(chunk)] = score_user_batch(
            assets.sasrec, batch, assets.row_to_item, assets.device, counters)
        pos += len(chunk)
        if pos % (batch_size * 8) < batch_size:
            print(f"  scored {pos}/{len(indices)} ({pos / (time.time() - t0):.1f}/s)",
                  flush=True)
    counters.record_peaks()
    common.write_cache_shard(
        cache_dir, args.shard_id,
        arrays={"raw_scores": out},
        meta={
            "kind": "sasrec_raw_scores", "split": args.split,
            "num_shards": args.num_shards,
            "dataset_indices": indices,
            "manifest_rows": [int(r) for r in rows],
            "item_order_hash": item_order_hash(assets.row_to_item),
            "batch_vs_single_max_err": err,
            "counters": counters.to_dict(),
            "environment": common.environment_info(args.device),
        })
    print(f"stage 02 shard {args.shard_id}/{args.num_shards} complete "
          f"({len(indices)} users, {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
