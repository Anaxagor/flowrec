#!/usr/bin/env python
"""Stage 07: test evaluation of the FROZEN validation selection (spec §14.2).

Refuses to run when the selection manifest hash no longer matches its frozen
value; every prediction shard records the selection hash it was produced
under. Only the append-only reference, one selected configuration per primary
family, fixed sanity references, and (when trained) the frozen gate are
evaluated — transform-independent methods exactly once.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic.compute_counters import ComputeCounters  # noqa: E402
from exp5_budgeted_cyclic.gate import load_gate, user_gate_features  # noqa: E402
from exp5_budgeted_cyclic.policies import (  # noqa: E402
    PolicyConfig,
    assert_no_duplicate_configs,
    evaluate_policy,
    user_context_from_caches,
)


def _load_stage05():
    spec = importlib.util.spec_from_file_location(
        "exp5_budgeted_cyclic.stage05",
        Path(__file__).parent / "05_run_validation_grid.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_test_configs(selection: dict, tsel: dict) -> list[PolicyConfig]:
    configs: list[PolicyConfig] = []
    for ref in tsel.get("fixed_references", []):
        configs.append(PolicyConfig.from_canonical(ref))
    for fam, sel in (selection.get("selections") or {}).items():
        can = sel.get("canonical_config")
        if can is None:
            print(f"  WARNING: no canonical config stored for {fam}; skipping",
                  flush=True)
            continue
        c = PolicyConfig.from_canonical(can)
        if c.canonical_hash() != sel["canonical_config_hash"]:
            raise RuntimeError(
                f"reconstructed config hash mismatch for {fam}: "
                f"{c.canonical_hash()} != {sel['canonical_config_hash']}")
        configs.append(c)
    # dedup by canonical hash (a selected config may equal a fixed reference)
    seen: set[str] = set()
    unique = []
    for c in configs:
        if c.canonical_hash() not in seen:
            seen.add(c.canonical_hash())
            unique.append(c)
    assert_no_duplicate_configs(unique)
    return unique


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p)
    p.add_argument("--grid", default=None)
    p.add_argument("--test-config", default=None)
    args = p.parse_args()
    args.split = "test"
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    selection, sel_hash = manifest.read_selection_manifest(root, verify=True)
    print(f"frozen selection verified: sha256 {sel_hash[:16]}...", flush=True)
    tc_path = Path(args.test_config) if args.test_config else (
        Path(__file__).parent / "configs" / "test_selected.yaml")
    tsel = yaml.safe_load(tc_path.read_text())
    configs = build_test_configs(selection, tsel)
    print(f"test configurations ({len(configs)}):", flush=True)
    for c in configs:
        print(f"  {c.family:45s} {c.name}")
    if args.dry_run:
        return

    stage05 = _load_stage05()
    grid = stage05.load_grid_cfg(args, cfg)
    users = manifest.read_split_manifest(root, "test")
    rows = common.shard_rows(len(users), args)
    assets = common.load_experiment_assets(args, cfg, need_models=False)
    trie = assets.trie
    tiger_reader = common.ShardCacheReader(root / "cache" / "tiger" / "test")
    sasrec_reader = common.ShardCacheReader(root / "cache" / "sasrec" / "test")
    idx = [int(i) for i in users["dataset_index"].iloc[rows]]
    tiger_reader.validate_complete(idx)
    sasrec_reader.validate_complete(idx)
    controllers = stage05.load_controllers(root, grid)
    gate = load_gate(root / "manifest" / "gate")
    gate_ref = cfg.get("gate", {}).get("reference_target", "exp_T1")
    gate_candidates = cfg.get("gate", {}).get("candidate_families", [])

    per_cfg_rows: dict[str, list[dict]] = defaultdict(list)
    counters: dict[str, ComputeCounters] = defaultdict(
        lambda: ComputeCounters(cache_mode="offline_cache"))
    gate_rows: list[dict] = []
    t0 = time.time()
    for pos, (_, urow) in enumerate(users.iloc[rows].iterrows()):
        ctx, raw_valid_mass = user_context_from_caches(
            trie, urow, tiger_reader, sasrec_reader, assets.item_to_row)
        rows_by_family: dict[str, dict] = {}
        for c in configs:
            row = evaluate_policy(c, ctx, controllers=controllers,
                                  counters=counters[c.canonical_hash()])
            row["selection_manifest_sha256"] = sel_hash
            per_cfg_rows[c.canonical_hash()].append(row)
            rows_by_family[c.family] = row
        if gate is not None:
            model, meta = gate
            feats = user_gate_features(
                ctx, raw_valid_mass, int(urow["history_length"]),
                ctx.get_target(gate_ref))
            X = np.asarray([[feats.get(f, np.nan) for f in meta.feature_names]])
            choice = int(model.predict(X)[0])
            fam = gate_candidates[choice] if choice < len(gate_candidates) else None
            chosen = rows_by_family.get(fam)
            if chosen is not None:
                g = dict(chosen)
                g.update(policy_family="gated", config_name="gated_selection",
                         canonical_config_hash="gated",
                         gate_choice_family=fam)
                gate_rows.append(g)
        if (pos + 1) % 100 == 0:
            print(f"  test user {pos + 1}/{len(rows)} "
                  f"({(pos + 1) / (time.time() - t0):.1f}/s)", flush=True)

    for c in configs:
        common.atomic_write_parquet(
            pd.DataFrame(per_cfg_rows[c.canonical_hash()]),
            root / "predictions" / "test" / c.family_dir / c.name
            / f"{common.shard_name(args.shard_id)}.parquet")
    if gate_rows:
        common.atomic_write_parquet(
            pd.DataFrame(gate_rows),
            root / "predictions" / "test" / "gated" / "gated_selection"
            / f"{common.shard_name(args.shard_id)}.parquet")
    comp_rows = []
    for c in configs:
        cc = counters[c.canonical_hash()]
        cc.record_peaks()
        comp_rows.append({
            "run_id": cfg["run_id"], "git_commit": common.git_commit(),
            "split": "test", "shard_id": args.shard_id,
            "policy_family": c.family, "config_name": c.name,
            "canonical_config_hash": c.canonical_hash(),
            "reward_transform": c.transform or "none", **cc.to_dict()})
    common.atomic_write_parquet(
        pd.DataFrame(comp_rows),
        root / "predictions" / "test" / "compute"
        / f"{common.shard_name(args.shard_id)}.parquet")
    print(f"stage 07 shard {args.shard_id} complete ({len(rows)} users)", flush=True)


if __name__ == "__main__":
    main()
