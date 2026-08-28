#!/usr/bin/env python
"""Stage 06: aggregate full-validation predictions, freeze selections, train gate.

Selection uses the COMPLETE validation set, one configuration per predeclared
family, by the declared rule (default: max mean ndcg@10). The frozen selection
manifest is hashed; stage 07 refuses to run if it changes afterwards. The
user-adaptive gate (spec §25) is trained and frozen here, before any test
predictions exist.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic.gate import GATE_FEATURES, train_gate  # noqa: E402
from exp5_budgeted_cyclic.reports import (  # noqa: E402
    read_prediction_shards,
    summarize_predictions,
    validate_same_users,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p, needs_split=False)
    p.add_argument("--test-config", default=None, help="test_selected.yaml override")
    p.add_argument("--grid", default=None, help="validation_grid.yaml override")
    p.add_argument("--selection-metric", default="ndcg@10")
    p.add_argument("--allow-partial-users", action="store_true",
                   help="smoke/regression runs only: relax full-manifest coverage")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    tc_path = Path(args.test_config) if args.test_config else (
        Path(__file__).parent / "configs" / "test_selected.yaml")
    tsel = yaml.safe_load(tc_path.read_text())

    # hash -> full canonical config (so stage 07 can reconstruct exactly)
    from exp5_budgeted_cyclic.policies import build_validation_grid
    grid_path = Path(args.grid) if args.grid else (
        Path(__file__).parent / "configs" / "validation_grid.yaml")
    grid_cfg = yaml.safe_load(grid_path.read_text())
    grid_cfg.setdefault("transforms", cfg.get("transforms", []))
    grid_cfg.setdefault("hybrid", cfg.get("hybrid_grid", {}))
    hash_to_canonical = {
        c.canonical_hash(): c.canonical() for c in build_validation_grid(grid_cfg)}

    users = manifest.read_split_manifest(root, "validation")
    df = read_prediction_shards(root, "validation")
    validate_same_users(df, [int(i) for i in users["dataset_index"]],
                        allow_partial=args.allow_partial_users)
    summary = summarize_predictions(
        df, num_items=int(cfg["expected"]["num_items"]),
        base_fields={"run_id": cfg["run_id"], "git_commit": common.git_commit(),
                     "split": "validation", "seed": None,
                     "cache_mode": "offline_cache"})
    val_csv = root / "summary" / "validation_metrics.csv"
    val_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(val_csv, index=False)
    artifact_hash = hashlib.sha256(val_csv.read_bytes()).hexdigest()
    print(f"validation_metrics.csv: {len(summary)} configs "
          f"(sha256 {artifact_hash[:12]}...)", flush=True)

    metric = args.selection_metric
    selections = {}
    for fam in tsel["primary_families"]:
        sub = summary[(summary["policy_family"] == fam) & summary[metric].notna()]
        if not len(sub):
            print(f"  WARNING: no validation rows for primary family {fam}", flush=True)
            continue
        best = sub.loc[sub[metric].idxmax()]
        selections[fam] = {
            "family": fam,
            "config_name": str(best["config_name"]),
            "canonical_config_hash": str(best["canonical_config_hash"]),
            "canonical_config": hash_to_canonical.get(str(best["canonical_config_hash"])),
            "reward_transform": str(best["reward_transform"]),
            "target": str(best["target"]),
            "validation_metric_name": metric,
            "validation_metric": float(best[metric]),
            "selection_rule": f"max mean {metric} over the complete validation set",
        }
        print(f"  selected {fam}: {best['config_name']} "
              f"({metric}={best[metric]:.4f})", flush=True)
    if args.dry_run:
        return

    digest = manifest.write_selection_manifest(root, {
        "selections": selections,
        "selection_metric": metric,
        "validation_artifact_hash": artifact_hash,
        "code_git_commit": common.git_commit(),
        "n_validation_users": int(users["dataset_index"].nunique()),
        "primary_families": tsel["primary_families"],
    })
    print(f"selection manifest frozen: sha256 {digest[:16]}...", flush=True)

    # ---- gate training (validation-only; frozen before test) ----------------
    gate_cfg = cfg.get("gate", {})
    cand_fams = gate_cfg.get("candidate_families", [])
    cand = []
    for fam in cand_fams:
        if fam in selections:
            cand.append(selections[fam])
        elif fam == "reference/append_only_cyclic_parameterization":
            cand.append({"family": fam,
                         "config_name": "append_only_cyclic_parameterization",
                         "canonical_config_hash": None})
    if len(cand) >= 2:
        feats = pd.concat(
            [pd.read_parquet(s) for s in
             sorted((root / "predictions" / "validation" / "gate_features").glob(
                 "shard_*.parquet"))], ignore_index=True)
        per_user = {}
        for k, c in enumerate(cand):
            sub = df[(df["policy_family"] == c["family"])]
            if c.get("canonical_config_hash"):
                sub = sub[sub["canonical_config_hash"] == c["canonical_config_hash"]]
            per_user[k] = sub.set_index("dataset_index")[gate_cfg.get(
                "label_metric", "ndcg@10")]
        mat = pd.DataFrame(per_user).dropna()
        labels = mat.to_numpy().argmax(axis=1)
        feats = feats.set_index("dataset_index").loc[mat.index].reset_index()
        art = train_gate(
            feats, labels,
            candidate_policies=[c["config_name"] for c in cand],
            label_metric=gate_cfg.get("label_metric", "ndcg@10"),
            out_dir=root / "manifest" / "gate", seed=cfg["seed"])
        if art is not None:
            counts = np.bincount(labels, minlength=len(cand))
            print(f"  gate label distribution: "
                  f"{dict(zip([c['config_name'] for c in cand], counts.tolist()))}",
                  flush=True)
    else:
        print("gate: fewer than 2 candidate policies resolved; skipped", flush=True)
    print("stage 06 complete", flush=True)


if __name__ == "__main__":
    main()
