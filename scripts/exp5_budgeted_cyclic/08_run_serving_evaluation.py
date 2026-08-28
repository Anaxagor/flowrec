#!/usr/bin/env python
"""Stage 08: serving-style sampled evaluation at fair cost (spec §26).

Multi-seed Monte Carlo candidate generation for the serving-feasible methods,
with genuine transition counters (online regime) and an optional end-to-end
latency benchmark that never mixes cached arithmetic with model latency.
Configured by configs/serving.yaml; runs on the frozen test selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5_budgeted_cyclic import common, manifest  # noqa: E402
from exp5_budgeted_cyclic import finite_budget_dp as dp  # noqa: E402
from exp5_budgeted_cyclic.compute_counters import ComputeCounters, LatencyBenchmark  # noqa: E402
from exp5_budgeted_cyclic.policies import PolicyConfig, user_context_from_caches  # noqa: E402

KS = (5, 10, 20)


def serving_row(terminal_rows, uc, m_values, raw_scores) -> dict:
    _, first_idx = np.unique(terminal_rows, return_index=True)
    uniq = terminal_rows[np.sort(first_idx)]
    out = {"n_unique_candidates": int(len(uniq)),
           "duplicate_rate": float(1.0 - len(uniq) / max(len(terminal_rows), 1))}
    for k in KS:
        firstk = uniq[:k]
        hit = uc.target_row in set(int(x) for x in firstk)
        rank = int(np.flatnonzero(firstk == uc.target_row)[0]) + 1 if hit else None
        out[f"pure_recall@{k}"] = float(hit)
        out[f"pure_ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
    for m in m_values:
        cand = uniq[:m]
        out[f"candidate_recall@{m}"] = float(uc.target_row in set(int(x) for x in cand))
        order = np.argsort(-raw_scores[cand], kind="stable")
        rr = cand[order]
        pos = np.flatnonzero(rr == uc.target_row)
        rank = int(pos[0]) + 1 if len(pos) else None
        for k in KS:
            hit = rank is not None and rank <= k
            out[f"rerank_M{m}_recall@{k}"] = float(bool(hit))
            out[f"rerank_M{m}_ndcg@{k}"] = float(1.0 / np.log2(rank + 1)) if hit else 0.0
    return out


def build_serving_policies(serving_cfg, selection, ctx_builder):
    """Yield (family, config_name, sampler_factory(uctx) -> sampler|None)."""
    sels = selection.get("selections") or {}

    def from_selected(fam):
        can = (sels.get(fam) or {}).get("canonical_config")
        return PolicyConfig.from_canonical(can) if can else None

    plans = []
    for name in serving_cfg.get("methods", []):
        if name == "tiger_stochastic":
            for tau in serving_cfg.get("taus", [1.0]):
                plans.append((("reference/tiger_stochastic", f"serving_tau{tau:g}"),
                              {"kind": "temperature", "tau": float(tau)}))
        elif name == "append_only":
            plans.append((("reference/append_only_cyclic_parameterization",
                           "serving_append_only"), {"kind": "append"}))
        elif name in ("suffix_only_rollback", "truncated_infinite_oracle",
                      "level_budget_rollback", "finite_budget_controller"):
            c = from_selected(name)
            if c is None:
                print(f"  serving: no selected config for {name}; skipped", flush=True)
                continue
            plans.append(((name, f"serving_{c.name}"), {"kind": "rollback", "config": c}))
        elif name == "no_repeat_heuristic":
            c = from_selected("suffix_only_rollback") or from_selected(
                "truncated_infinite_oracle")
            if c is None:
                continue
            plans.append((("no_repeat_heuristic", f"serving_no_repeat_{c.name}"),
                          {"kind": "rollback", "config": c, "no_repeat": True}))
        elif name == "local_prefix_guidance":
            c = from_selected("local_guidance")
            if c is None:
                continue
            plans.append((("local_guidance", f"serving_{c.name}"),
                          {"kind": "guidance", "config": c}))
        else:
            print(f"  serving: unknown method '{name}' skipped", flush=True)
    return plans


def sampler_for(plan, uctx, controllers):
    trie = uctx.trie
    zeros = np.zeros(trie.num_nodes)
    if plan["kind"] == "append":
        return dp.BudgetedCyclicSampler(trie, uctx.edge_prob, zeros, budget=0), 0
    if plan["kind"] == "temperature":
        p_tau = dp.temperature_edge_prob(trie, uctx.edge_logprob, plan["tau"])
        return dp.BudgetedCyclicSampler(trie, p_tau, zeros, budget=0), 0
    c: PolicyConfig = plan["config"]
    tgt = uctx.get_target(c.target_name)
    if tgt.d_dp is None:
        return None, 0
    if plan["kind"] == "guidance":
        q_lin = np.exp(tgt.log_q)
        p_a = dp.geometric_guidance_edge_prob(
            trie, uctx.edge_prob, q_lin, float(c.param("alpha")))
        return dp.BudgetedCyclicSampler(trie, p_a, zeros, budget=0), 0
    d = tgt.d_dp
    fam = c.family
    if fam == "suffix_only_rollback":
        d = dp.level_masked_rollback(trie, d, {trie.sid_len})
        return dp.BudgetedCyclicSampler(
            trie, uctx.edge_prob, d, budget=int(c.param("budget")),
            no_immediate_repeat=bool(plan.get("no_repeat"))), int(c.param("budget"))
    if fam == "level_budget_rollback" and c.param("mode") == "alloc":
        return dp.BudgetedCyclicSampler(
            trie, uctx.edge_prob, d, level_budgets=c.param("allocation")), \
            sum(c.param("allocation"))
    if fam == "finite_budget_controller":
        art = controllers.get((c.target_name, int(c.param("budget"))))
        if art is None:
            return None, 0
        feats = dp.controller_features(trie, tgt.d_dp, tgt.oracle.log_C, uctx.entropy)
        d_rows = art.rollback_rows(trie, feats)
        return dp.BudgetedCyclicSampler(
            trie, uctx.edge_prob, d_rows, budget=int(c.param("budget"))), \
            int(c.param("budget"))
    budget = int(c.param("budget"))
    if fam == "level_budget_rollback" and c.param("mode") == "mask":
        from exp5_budgeted_cyclic.policies import _mask_levels
        d = dp.level_masked_rollback(trie, d, _mask_levels(str(c.param("levels")),
                                                           trie.sid_len))
    return dp.BudgetedCyclicSampler(
        trie, uctx.edge_prob, d, budget=budget,
        no_immediate_repeat=bool(plan.get("no_repeat"))), budget


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(p)
    p.add_argument("--serving-config", default=None)
    p.add_argument("--grid", default=None)
    p.add_argument("--benchmark", action="store_true",
                   help="measure genuine end-to-end model latency on a few users")
    args = p.parse_args()
    cfg = common.load_config(args)
    root = common.run_root(cfg, args)
    sc_path = Path(args.serving_config) if args.serving_config else (
        Path(__file__).parent / "configs" / "serving.yaml")
    serving_cfg = yaml.safe_load(sc_path.read_text())
    selection, sel_hash = manifest.read_selection_manifest(root, verify=True)

    users = manifest.read_split_manifest(root, args.split)
    rows = common.shard_rows(len(users), args)
    plans = build_serving_policies(serving_cfg, selection, None)
    seeds = serving_cfg.get("seeds", [42, 43, 44, 45, 46])
    n_traj_grid = serving_cfg.get("n_trajectories", [20, 50])
    m_values = serving_cfg.get("candidate_set_sizes", [20, 50])
    print(f"{len(plans)} serving methods x {len(n_traj_grid)} budgets x "
          f"{len(seeds)} seeds x {len(rows)} users", flush=True)
    if args.dry_run:
        return

    spec = importlib.util.spec_from_file_location(
        "exp5_budgeted_cyclic.stage05b",
        Path(__file__).parent / "05_run_validation_grid.py")
    stage05 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage05
    spec.loader.exec_module(stage05)
    grid = stage05.load_grid_cfg(args, cfg)
    controllers = stage05.load_controllers(root, grid)

    assets = common.load_experiment_assets(
        args, cfg, need_models=args.benchmark)
    trie = assets.trie
    tiger_reader = common.ShardCacheReader(root / "cache" / "tiger" / args.split)
    sasrec_reader = common.ShardCacheReader(root / "cache" / "sasrec" / args.split)
    idx = [int(i) for i in users["dataset_index"].iloc[rows]]
    tiger_reader.validate_complete(idx)
    sasrec_reader.validate_complete(idx)

    out_rows: list[dict] = []
    t0 = time.time()
    for pos, (_, urow) in enumerate(users.iloc[rows].iterrows()):
        uctx, _ = user_context_from_caches(
            trie, urow, tiger_reader, sasrec_reader, assets.item_to_row)
        for (fam, name), plan in plans:
            sampler, budget = sampler_for(plan, uctx, controllers)
            if sampler is None:
                continue
            for n_traj in n_traj_grid:
                for seed in seeds:
                    ts = time.time()
                    out = sampler.sample(int(n_traj), seed=int(seed))
                    dt = time.time() - ts
                    row = {
                        "user": uctx.raw_user_id, "dataset_index": uctx.dataset_index,
                        "policy_family": fam, "config_name": f"{name}_n{n_traj}",
                        "canonical_config_hash": f"{name}_n{n_traj}",
                        "reward_transform": (plan.get("config").transform or "none"
                                             ) if plan.get("config") else "none",
                        "seed": int(seed), "n_trajectories": int(n_traj),
                        "budget": budget, "cache_mode": "offline_cache",
                        "selection_manifest_sha256": sel_hash,
                        "expected_deletes": float(out.n_deletes.mean()),
                        "mean_steps": float(out.n_steps.mean()),
                        "same_token_reselect": float(out.n_same_token_reselect)
                        / max(out.n_revisions, 1),
                        "revision_precision": (float(out.n_revisions_improved)
                                               / max(out.n_revisions, 1)
                                               if out.n_revisions_improved >= 0
                                               else np.nan),
                        "sampled_trajectory_transitions": int(out.n_steps.sum()),
                        "num_delete_actions": int(out.n_deletes.sum()),
                        "num_add_actions": int(out.n_steps.sum() - out.n_deletes.sum()
                                               - len(out.terminal_row)),
                        "num_stop_actions": int(len(out.terminal_row)),
                        "online_tiger_calls_if_uncached": int(
                            out.n_steps.sum() - out.n_deletes.sum()),
                        "cached_policy_runtime_s": dt,
                    }
                    row.update(serving_row(out.terminal_row, uctx, m_values,
                                           uctx.raw_scores))
                    out_rows.append(row)
        if (pos + 1) % 50 == 0:
            print(f"  serving user {pos + 1}/{len(rows)} "
                  f"({(pos + 1) / (time.time() - t0):.1f}/s)", flush=True)

    df = pd.DataFrame(out_rows)
    common.atomic_write_parquet(
        df, root / "predictions" / "serving" / args.split
        / f"{common.shard_name(args.shard_id)}.parquet")

    if args.benchmark:
        run_benchmark(args, cfg, root, assets, users.iloc[rows].head(4))
    print(f"stage 08 shard {args.shard_id} complete ({len(df)} rows)", flush=True)


def run_benchmark(args, cfg, root, assets, sample_users) -> None:
    """Genuine end-to-end latency: real model forward passes, CUDA-synced."""
    bench = LatencyBenchmark(device=args.device)
    results = []
    for _, urow in sample_users.iterrows():
        batch = common.single_user_batch(assets.dm, args.split,
                                         int(urow["dataset_index"]))
        moved = {k: (v.to(assets.device) if hasattr(v, "to") else v)
                 for k, v in batch.items()}
        results.append(bench.run(
            lambda: assets.tiger.generate(moved, num_beams=20,
                                          num_return_sequences=20,
                                          constrain_valid_sequences=True),
            label="tiger_beam20_generate_end_to_end", batch_size=1,
            caches_enabled=False))
        results.append(bench.run(
            lambda: assets.sasrec.score_items(moved),
            label="sasrec_full_catalog_end_to_end", batch_size=1,
            caches_enabled=False))
    out = root / "summary" / "latency_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"  wrote {out} (end_to_end_runtime; cached arithmetic reported "
          "separately as cached_policy_runtime_s)", flush=True)


if __name__ == "__main__":
    main()
