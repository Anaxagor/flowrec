#!/usr/bin/env python
"""Stage 04: flow-oracle construction, numerical diagnostics, oracle regression.

For every (user, reward-target) pair, builds the log-space target and global
flow oracle from the stage-02/03 caches and records the spec §11 diagnostics
(reward stats, support/underflow/saturation classification, scale-aware flow
identity errors, infinite-oracle cost). This is the corrected
``oracle_regression`` evidence for the FULL split, and its per-user
``E_del_inf`` feeds the heavy-tail analysis (report Q10).

Flow quantities are cheap to rebuild (~ms per user from the caches), so this
stage stores diagnostics parquet shards rather than dense d* arrays; pass
``--store-d-star`` to additionally materialize d* (float32) per target.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic.compute_counters import ComputeCounters  # noqa: E402
from exp5_budgeted_cyclic.policies import user_context_from_caches  # noqa: E402
from exp5_budgeted_cyclic.rewards import hybrid_target_name  # noqa: E402


def target_names(cfg: dict) -> list[str]:
    out = list(cfg.get("transforms", []))
    for h in cfg.get("hybrid_targets", []):
        out.append(hybrid_target_name(float(h["alpha"]), float(h["T"])))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p)
    p.add_argument("--store-d-star", action="store_true")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    users = manifest.read_split_manifest(root, args.split)
    rows = common.shard_rows(len(users), args)
    out_dir = root / "cache" / "flow" / args.split
    out_path = out_dir / f"{common.shard_name(args.shard_id)}.parquet"
    common.print_workload_estimate("04_flow", len(rows) * 4, 0.02)
    if args.dry_run:
        return
    if args.resume and not args.overwrite and out_path.exists():
        print(f"shard {args.shard_id} already complete; --resume skip", flush=True)
        return

    assets = common.load_experiment_assets(args, cfg, need_models=False)
    trie = assets.trie
    tiger_reader = common.ShardCacheReader(root / "cache" / "tiger" / args.split)
    sasrec_reader = common.ShardCacheReader(root / "cache" / "sasrec" / args.split)
    idx = users["dataset_index"].iloc[rows]
    tiger_reader.validate_complete([int(i) for i in idx])
    sasrec_reader.validate_complete([int(i) for i in idx])

    names = target_names(cfg)
    counters = ComputeCounters(cache_mode="offline_cache")
    diag_rows = []
    d_star_store: dict[str, list[np.ndarray]] = {n: [] for n in names}
    t0 = time.time()
    for pos, (_, urow) in enumerate(users.iloc[rows].iterrows()):
        ctx, _ = user_context_from_caches(
            trie, urow, tiger_reader, sasrec_reader, assets.item_to_row,
            target_counters=counters)
        for name in names:
            tgt = ctx.get_target(name)
            row = {
                "user": ctx.raw_user_id, "dataset_index": ctx.dataset_index,
                "split": args.split, "target": name,
                "oracle_ok": bool(tgt.oracle is not None),
                "oracle_error": tgt.oracle_error,
            }
            row.update({f"reward_{k}": v for k, v in tgt.log_reward.stats.items()})
            if tgt.oracle is not None:
                row.update(tgt.oracle.diagnostics)
                row["E_del_inf"] = tgt.oracle.E_del_inf
                row["E_n_inf"] = tgt.oracle.E_n_inf
                for lvl in range(1, trie.sid_len + 1):
                    row[f"E_del_inf_L{lvl}"] = float(tgt.oracle.E_del_inf_by_level[lvl])
                row.update(tgt.clip_info)
                if args.store_d_star:
                    d_star_store[name].append(tgt.d_dp.astype(np.float32))
            diag_rows.append(row)
        if (pos + 1) % 100 == 0:
            print(f"  {pos + 1}/{len(rows)} users "
                  f"({(pos + 1) / (time.time() - t0):.1f}/s)", flush=True)

    df = pd.DataFrame(diag_rows)
    common.atomic_write_parquet(df, out_path)
    counters.record_peaks()
    common.atomic_write_json(
        {"kind": "flow_diagnostics", "split": args.split, "shard_id": args.shard_id,
         "num_shards": args.num_shards, "targets": names,
         "dataset_indices": [int(i) for i in idx],
         "counters": counters.to_dict(),
         "environment": common.environment_info(args.device)},
        out_dir / f"{common.shard_name(args.shard_id)}.meta.json")
    if args.store_d_star:
        for name, mats in d_star_store.items():
            if mats:
                common.atomic_write_npz(
                    out_dir / f"{common.shard_name(args.shard_id)}.d_star.{name}.npz",
                    d_star=np.stack(mats))
    ok = df["oracle_ok"].mean()
    print(f"stage 04 shard {args.shard_id} complete: {len(df)} rows, "
          f"oracle_ok rate {ok:.3f}", flush=True)


if __name__ == "__main__":
    main()
