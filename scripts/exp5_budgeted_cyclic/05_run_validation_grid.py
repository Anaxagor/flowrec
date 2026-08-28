#!/usr/bin/env python
"""Stage 05: exact evaluation of the predeclared grid on the FULL validation set.

Reads the stage-02/03 caches, evaluates every grid configuration for every
shard user, and writes per-config prediction shards plus per-config compute
counters. ``--fit-controller-only`` fits the shared budget-aware controllers
(spec §23.1) on a predeclared validation subsample and exits; run it once
before the evaluation array (submit_pipeline.sh encodes that dependency).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic import finite_budget_dp as dp  # noqa: E402
from exp5_budgeted_cyclic.compute_counters import ComputeCounters  # noqa: E402
from exp5_budgeted_cyclic.gate import user_gate_features  # noqa: E402
from exp5_budgeted_cyclic.policies import (  # noqa: E402
    PolicyConfig,
    build_validation_grid,
    evaluate_policy,
    user_context_from_caches,
)


def load_grid_cfg(args, cfg) -> dict:
    path = Path(args.grid) if args.grid else (
        Path(__file__).parent / "configs" / "validation_grid.yaml")
    grid = yaml.safe_load(path.read_text())
    grid.setdefault("transforms", cfg.get("transforms", []))
    grid.setdefault("hybrid", cfg.get("hybrid_grid", {}))
    return grid


def controllers_dir(root: Path) -> Path:
    return root / "manifest" / "controllers"


def controller_path(root: Path, target: str, budget: int) -> Path:
    return controllers_dir(root) / f"controller__{target}__B{budget}.npz"


def fit_controllers(args, cfg, grid, root) -> None:
    ctrl = grid.get("controller")
    if not ctrl:
        print("no controller block in the grid; nothing to fit", flush=True)
        return
    users = manifest.read_split_manifest(root, "validation")
    assets = common.load_experiment_assets(args, cfg, need_models=False)
    trie = assets.trie
    tiger_reader = common.ShardCacheReader(root / "cache" / "tiger" / "validation")
    sasrec_reader = common.ShardCacheReader(root / "cache" / "sasrec" / "validation")
    # fit only on users whose caches exist (a smoke run caches a small subset)
    cached = users[users["dataset_index"].astype(int).isin(tiger_reader.index)]
    cached = cached.reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    fit_n = min(int(ctrl.get("fit_users", 256)), len(cached))
    fit_rows = np.sort(rng.choice(len(cached), size=fit_n, replace=False))
    budget = int(ctrl["budget"])
    for target in ctrl.get("transforms", grid.get("transforms", [])[:1]):
        user_data = []
        for _, urow in cached.iloc[fit_rows].iterrows():
            ctx, _ = user_context_from_caches(
                trie, urow, tiger_reader, sasrec_reader, assets.item_to_row)
            tgt = ctx.get_target(target)
            if tgt.d_dp is None:
                continue
            feats = dp.controller_features(trie, tgt.d_dp, tgt.oracle.log_C, ctx.entropy)
            user_data.append({
                "edge_prob": ctx.edge_prob, "features": feats,
                "p_target": tgt.p_target, "dataset_index": ctx.dataset_index})
        print(f"fitting controller target={target} B={budget} on "
              f"{len(user_data)} users ...", flush=True)
        art = dp.fit_budget_controller(
            trie, user_data, budget=budget, target_name=target,
            steps=int(ctrl.get("steps", 200)), lr=float(ctrl.get("lr", 0.05)),
            lambda_steps=float(ctrl.get("lambda_steps", 0.0)),
            seed=cfg["seed"], device=args.device)
        controllers_dir(root).mkdir(parents=True, exist_ok=True)
        art.save(controller_path(root, target, budget))
        print(f"  saved; final mean JS {art.history[-1]['mean_js']:.4f}", flush=True)


def load_controllers(root, grid) -> dict:
    out = {}
    ctrl = grid.get("controller")
    if not ctrl:
        return out
    budget = int(ctrl["budget"])
    for target in ctrl.get("transforms", grid.get("transforms", [])[:1]):
        path = controller_path(root, target, budget)
        if path.exists():
            out[(target, budget)] = dp.ControllerArtifact.load(path)
    return out


def run_grid(args, cfg, grid, root, *, split: str,
             configs: list[PolicyConfig] | None = None,
             extra_meta: dict | None = None) -> None:
    users = manifest.read_split_manifest(root, split)
    rows = common.shard_rows(len(users), args)
    configs = configs if configs is not None else build_validation_grid(grid)
    print(f"{len(configs)} configurations x {len(rows)} shard users", flush=True)
    common.print_workload_estimate("policy_grid", len(rows) * len(configs), 0.01)
    if args.dry_run:
        for c in configs:
            print(f"  {c.family:45s} {c.name}")
        return

    first_out = _pred_path(root, split, configs[0], args.shard_id)
    if args.resume and not args.overwrite and first_out.exists():
        print(f"shard {args.shard_id} predictions exist; --resume skip", flush=True)
        return

    assets = common.load_experiment_assets(args, cfg, need_models=False)
    trie = assets.trie
    tiger_reader = common.ShardCacheReader(root / "cache" / "tiger" / split)
    sasrec_reader = common.ShardCacheReader(root / "cache" / "sasrec" / split)
    idx = [int(i) for i in users["dataset_index"].iloc[rows]]
    tiger_reader.validate_complete(idx)
    sasrec_reader.validate_complete(idx)
    controllers = load_controllers(root, grid)

    target_counters = ComputeCounters(cache_mode="offline_cache")
    per_cfg_rows: dict[str, list[dict]] = defaultdict(list)
    per_cfg_counters: dict[str, ComputeCounters] = defaultdict(
        lambda: ComputeCounters(cache_mode="offline_cache"))
    gate_rows = []
    gate_ref = cfg.get("gate", {}).get("reference_target", "exp_T1")
    t0 = time.time()
    for pos, (_, urow) in enumerate(users.iloc[rows].iterrows()):
        ctx, raw_valid_mass = user_context_from_caches(
            trie, urow, tiger_reader, sasrec_reader, assets.item_to_row,
            target_counters=target_counters)
        for c in configs:
            row = evaluate_policy(c, ctx, controllers=controllers,
                                  counters=per_cfg_counters[c.canonical_hash()])
            per_cfg_rows[c.canonical_hash()].append(row)
        try:
            feats = user_gate_features(
                ctx, raw_valid_mass, int(urow["history_length"]),
                ctx.get_target(gate_ref))
            feats.update(user=ctx.raw_user_id, dataset_index=ctx.dataset_index)
            gate_rows.append(feats)
        except Exception as exc:  # gate features are best-effort diagnostics
            gate_rows.append({"user": ctx.raw_user_id,
                              "dataset_index": ctx.dataset_index,
                              "feature_error": str(exc)})
        if (pos + 1) % 50 == 0:
            print(f"  user {pos + 1}/{len(rows)} "
                  f"({(pos + 1) / (time.time() - t0):.2f}/s)", flush=True)

    for c in configs:
        df = pd.DataFrame(per_cfg_rows[c.canonical_hash()])
        common.atomic_write_parquet(df, _pred_path(root, split, c, args.shard_id))
    gate_dir = root / "predictions" / split / "gate_features"
    common.atomic_write_parquet(pd.DataFrame(gate_rows),
                                gate_dir / f"{common.shard_name(args.shard_id)}.parquet")

    comp_rows = []
    base = dict(run_id=cfg["run_id"], git_commit=common.git_commit(), split=split,
                shard_id=args.shard_id, **(extra_meta or {}))
    for c in configs:
        cc = per_cfg_counters[c.canonical_hash()]
        cc.record_peaks()
        comp_rows.append({**base, "policy_family": c.family, "config_name": c.name,
                          "canonical_config_hash": c.canonical_hash(),
                          "reward_transform": c.transform or "none",
                          **cc.to_dict()})
    target_counters.record_peaks()
    comp_rows.append({**base, "policy_family": "target_construction",
                      "config_name": "shared_target_and_oracle_build",
                      "canonical_config_hash": "-",
                      "reward_transform": "all", **target_counters.to_dict()})
    common.atomic_write_parquet(
        pd.DataFrame(comp_rows),
        root / "predictions" / split / "compute" / f"{common.shard_name(args.shard_id)}.parquet")
    print(f"stage complete: shard {args.shard_id} ({len(rows)} users, "
          f"{time.time() - t0:.1f}s)", flush=True)


def _pred_path(root: Path, split: str, c: PolicyConfig, shard_id: int) -> Path:
    return (root / "predictions" / split / c.family_dir / c.name
            / f"{common.shard_name(shard_id)}.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p)
    p.add_argument("--grid", default=None, help="validation_grid.yaml override")
    p.add_argument("--fit-controller-only", action="store_true")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    grid = load_grid_cfg(args, cfg)
    if args.fit_controller_only:
        fit_controllers(args, cfg, grid, root)
        return
    run_grid(args, cfg, grid, root, split=args.split)


if __name__ == "__main__":
    main()
