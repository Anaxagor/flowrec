#!/usr/bin/env python
"""Stage 09: aggregate everything, statistics with Holm correction, report.

Produces the summary CSVs (validation/test metrics, paired bootstrap with
Holm-adjusted p-values over the DECLARED primary comparisons, compute
counters, serving metrics, per-level and user-slice diagnostics), the plots,
and report/report.{md,json} answering the spec §31 questions under the §15
language constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic.bootstrap import (  # noqa: E402
    average_over_seeds,
    holm_adjust,
    paired_bootstrap,
)
from exp5_budgeted_cyclic.reports import (  # noqa: E402
    build_report,
    make_plots,
    read_prediction_shards,
    summarize_predictions,
    validate_same_users,
    assert_unique_prediction_sets,
)


def _try(fn, label):
    try:
        return fn()
    except FileNotFoundError as exc:
        print(f"  [skip] {label}: {exc}", flush=True)
        return None


def load_serving(root: Path) -> pd.DataFrame | None:
    shards = sorted((root / "predictions" / "serving").glob("*/shard_*.parquet"))
    if not shards:
        return None
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def load_compute(root: Path) -> pd.DataFrame | None:
    frames = []
    for split in ("validation", "test"):
        for s in sorted((root / "predictions" / split / "compute").glob("shard_*.parquet")):
            frames.append(pd.read_parquet(s))
    for kind in ("sasrec", "tiger"):
        for split in ("validation", "test"):
            d = root / "cache" / kind / split
            for meta_path in sorted(d.glob("shard_*.meta.json")):
                meta = json.loads(meta_path.read_text())
                row = {"policy_family": f"cache_build/{kind}", "split": split,
                       "config_name": kind, "canonical_config_hash": "-",
                       "reward_transform": "none",
                       "shard_id": meta.get("shard_id")}
                row.update(meta.get("counters", {}))
                frames.append(pd.DataFrame([row]))
    return pd.concat(frames, ignore_index=True) if frames else None


def load_flow_diagnostics(root: Path) -> pd.DataFrame | None:
    frames = []
    for split in ("validation", "test"):
        for s in sorted((root / "cache" / "flow" / split).glob("shard_*.parquet")):
            frames.append(pd.read_parquet(s))
    return pd.concat(frames, ignore_index=True) if frames else None


def test_bootstrap_table(
    test_df: pd.DataFrame, primaries: list[str], *, n_boot: int, seed: int,
) -> pd.DataFrame:
    ok = test_df[test_df["status"] == "ok"]
    base = ok[ok["policy_family"] == "reference/append_only_cyclic_parameterization"]
    if not len(base):
        raise RuntimeError("append-only reference missing from test predictions")
    base_idx = base.set_index("dataset_index")
    rows = []
    for (fam, name, h), sub in ok.groupby(
            ["policy_family", "config_name", "canonical_config_hash"]):
        if fam == "reference/append_only_cyclic_parameterization":
            continue
        joined = sub.set_index("dataset_index").join(
            base_idx[["ndcg@10", "recall@10", "mrr"]], rsuffix="_base").dropna(
            subset=["ndcg@10", "ndcg@10_base"])
        if len(joined) < 5:
            continue
        for metric in ("ndcg@10", "recall@10"):
            deltas = (joined[metric] - joined[f"{metric}_base"]).to_numpy()
            row = {"policy_family": fam, "config_name": name,
                   "canonical_config_hash": h, "metric": metric,
                   "is_primary": fam in primaries}
            row.update(paired_bootstrap(deltas, n_boot=n_boot, seed=seed))
            rows.append(row)
    boot = pd.DataFrame(rows)
    if len(boot):
        prim = boot[(boot["is_primary"]) & (boot["metric"] == "ndcg@10")]
        adj = holm_adjust({r["config_name"]: r["p_value_two_sided"]
                           for _, r in prim.iterrows()})
        boot["holm_adjusted_p"] = boot.apply(
            lambda r: adj.get(r["config_name"]) if (r["is_primary"]
                                                    and r["metric"] == "ndcg@10")
            else np.nan, axis=1)
    return boot


def user_slice_table(flow: pd.DataFrame | None, test_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Diagnostic slices by the user's dominant infinite-oracle cost level
    (dominant_oracle_cost_level_L*; user subsets, NOT policies — spec §13)."""
    if flow is None or test_df is None:
        return None
    lvl_cols = [c for c in flow.columns if c.startswith("E_del_inf_L")]
    if not lvl_cols:
        return None
    f = flow[flow["split"] == "test"]
    if not len(f):
        f = flow
    ref = f.groupby("dataset_index")[lvl_cols].mean()
    dom = ref.to_numpy().argmax(axis=1) + 1
    slice_of = pd.Series(
        [f"dominant_oracle_cost_level_L{d}" for d in dom], index=ref.index,
        name="user_slice")
    ok = test_df[test_df["status"] == "ok"].join(slice_of, on="dataset_index")
    rows = []
    for (fam, name, sl), sub in ok.groupby(
            ["policy_family", "config_name", "user_slice"]):
        rows.append({
            "policy_family": fam, "config_name": name, "user_slice": sl,
            "n_users": int(sub["dataset_index"].nunique()),
            "ndcg@10": float(sub["ndcg@10"].mean()),
            "recall@10": float(sub["recall@10"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p, needs_split=False)
    p.add_argument("--test-config", default=None)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--allow-partial-users", action="store_true",
                   help="smoke/regression runs only: relax full-manifest coverage")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    sdir = root / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    num_items = int(cfg["expected"]["num_items"])
    base_fields = {"run_id": cfg["run_id"], "git_commit": common.git_commit(),
                   "seed": None, "cache_mode": "offline_cache"}
    tc_path = Path(args.test_config) if args.test_config else (
        Path(__file__).parent / "configs" / "test_selected.yaml")
    primaries = yaml.safe_load(tc_path.read_text())["primary_families"]

    summaries = {}
    raw = {}
    for split in ("validation", "test"):
        df = _try(lambda s=split: read_prediction_shards(root, s), f"{split} predictions")
        if df is None:
            continue
        users = manifest.read_split_manifest(root, split)
        validate_same_users(
            df[~df["policy_family"].isin(["gated"])],
            [int(i) for i in users["dataset_index"]],
            allow_partial=args.allow_partial_users)
        if split == "test":
            assert_unique_prediction_sets(df)
            # test predictions must have been generated under the CURRENT
            # frozen selection (spec 14.1)
            if "selection_manifest_sha256" in df.columns:
                _, cur_hash = manifest.read_selection_manifest(root, verify=True)
                stale = set(df["selection_manifest_sha256"].dropna().unique()) - {cur_hash}
                if stale:
                    raise RuntimeError(
                        "test predictions were generated under a DIFFERENT "
                        f"selection manifest ({stale}) than the current frozen "
                        f"one ({cur_hash}); delete predictions/test and re-run "
                        "stage 07")
        raw[split] = df
        summaries[split] = summarize_predictions(
            df, num_items=num_items, base_fields={**base_fields, "split": split})
        summaries[split].to_csv(sdir / f"{split}_metrics.csv", index=False)
        print(f"  wrote {split}_metrics.csv ({len(summaries[split])} configs)",
              flush=True)

    boot = None
    if "test" in raw:
        boot = test_bootstrap_table(raw["test"], primaries,
                                    n_boot=args.n_boot, seed=cfg["seed"])
        boot.to_csv(sdir / "test_bootstrap.csv", index=False)
        print(f"  wrote test_bootstrap.csv ({len(boot)} comparisons; Holm over "
              f"{int(boot['is_primary'].sum() // 2) if len(boot) else 0} primaries)",
              flush=True)

    comp = load_compute(root)
    if comp is not None:
        comp.to_csv(sdir / "compute_metrics.csv", index=False)
    serving = load_serving(root)
    serving_summary = None
    if serving is not None:
        num_cols = [c for c in serving.select_dtypes(include=[np.number]).columns
                    if c not in ("dataset_index", "seed")]
        per_user = serving.groupby(
            ["policy_family", "config_name", "user"], as_index=False)[num_cols].mean()
        serving_summary = per_user.groupby(
            ["policy_family", "config_name"], as_index=False)[num_cols].mean()
        serving_summary["seed_aggregation"] = "per-user mean over seeds, then user mean"
        serving_summary.to_csv(sdir / "serving_metrics.csv", index=False)
    flow = load_flow_diagnostics(root)
    if flow is not None:
        num_cols = flow.select_dtypes(include=[np.number]).columns.tolist()
        flow.groupby(["split", "target"], as_index=False)[num_cols].mean().to_csv(
            sdir / "oracle_regression.csv", index=False)
    slices = user_slice_table(flow, raw.get("test"))
    if slices is not None:
        slices.to_csv(sdir / "user_slice_metrics.csv", index=False)

    lvl_rows = []
    for split, summ in summaries.items():
        sub = summ[summ["policy_family"].isin(
            ["level_budget_rollback", "suffix_only_rollback",
             "truncated_infinite_oracle"])]
        if len(sub):
            lvl_rows.append(sub.assign(split=split))
    if lvl_rows:
        pd.concat(lvl_rows).to_csv(sdir / "by_level.csv", index=False)

    for split, summ in summaries.items():
        make_plots(summ, root / "plots", split=split)

    protocol = None
    proto_path = sdir / "protocol_regression.json"
    if proto_path.exists():
        protocol = json.loads(proto_path.read_text())
    selection = None
    try:
        selection, _ = manifest.read_selection_manifest(root, verify=False)
    except FileNotFoundError:
        pass
    oracle_cost = flow[["dataset_index", "E_del_inf"]].dropna() if flow is not None \
        and "E_del_inf" in flow else None
    build_report(
        out_dir=root, run_id=cfg["run_id"],
        validation=summaries.get("validation"), test=summaries.get("test"),
        bootstrap=boot, serving=serving_summary, protocol=protocol,
        selection=selection, compute=comp, oracle_cost=oracle_cost)
    print("stage 09 complete", flush=True)


if __name__ == "__main__":
    main()
