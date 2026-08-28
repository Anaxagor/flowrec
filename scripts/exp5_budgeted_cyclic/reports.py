"""Aggregation plots and the automated report (spec §15, §31).

Language discipline enforced here (spec §15): every result is labelled as one
of validation-selected primary result / sensitivity analysis / post-hoc
diagnostic / oracle upper bound / serving-feasible method;
``bootstrap_prob_improvement`` is never called a p-value; cached arithmetic is
never called end-to-end latency; candidate-limited methods never show catalog
ranks; and divergence-vs-NDCG divergences list ALL candidate explanations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMILY_COLORS = {
    "reference/tiger_exact_full_catalog": "#444444",
    "reference/append_only_cyclic_parameterization": "#000000",
    "reference/tiger_beam": "#9467bd",
    "reference/tiger_stochastic": "#17becf",
    "reference/sasrec_full_catalog": "#7f7f7f",
    "best_of_n": "#2ca02c",
    "beam_rerank": "#e377c2",
    "local_guidance": "#1f77b4",
    "global_hybrid_target": "#0b5394",
    "truncated_infinite_oracle": "#d62728",
    "scaled_truncated_infinite_oracle": "#ff7f0e",
    "suffix_only_rollback": "#8c564b",
    "level_budget_rollback": "#bcbd22",
    "finite_budget_controller": "#a83279",
    "no_repeat_heuristic": "#5b8c5a",
    "gated": "#00a0a0",
}


def _save(fig: plt.Figure, plots_dir: Path, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / name, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  plot {name}", flush=True)


def make_plots(summary: pd.DataFrame, plots_dir: Path, *, split: str) -> None:
    plt.rcParams.update({"figure.figsize": (7.5, 4.5), "axes.grid": True,
                         "grid.alpha": 0.3})
    full = summary[summary["eval_kind"] == "full_catalog"]

    for metric, fname in (("ndcg@10", f"{split}_ndcg10_vs_expected_steps.png"),
                          ("recall@10", f"{split}_recall10_vs_expected_steps.png")):
        sub = full.dropna(subset=["expected_steps", metric])
        if not len(sub):
            continue
        fig, ax = plt.subplots()
        for fam, fsub in sub.groupby("policy_family"):
            fsub = fsub.sort_values("expected_steps")
            ax.plot(fsub["expected_steps"], fsub[metric], "o-", ms=4, lw=1,
                    color=FAMILY_COLORS.get(fam, "k"), label=fam, alpha=0.85)
        ax.set_xlabel("expected trajectory length E[n]  (online serving cost proxy)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=6)
        ax.set_title(f"{metric} vs expected steps ({split}); reward-model cost "
                     "reported separately in compute_metrics.csv")
        _save(fig, plots_dir, fname)

    to = full[full["policy_family"].isin(
        ["truncated_infinite_oracle", "suffix_only_rollback",
         "finite_budget_controller"])].copy()
    if len(to) and "budget" in to.columns:
        fig, ax = plt.subplots()
        for (fam, t), fsub in to.groupby(["policy_family", "reward_transform"]):
            fsub = fsub.dropna(subset=["budget"]).sort_values("budget")
            if len(fsub):
                ax.plot(fsub["budget"], fsub["ndcg@10"], "o-", ms=4,
                        label=f"{fam.split('/')[-1]}:{t}", alpha=0.8)
        ax.set_xlabel("rollback budget B")
        ax.set_ylabel("ndcg@10")
        ax.set_xscale("symlog", linthresh=1)
        ax.legend(fontsize=6)
        _save(fig, plots_dir, f"{split}_quality_vs_budget.png")

    lvl_cols = [c for c in full.columns if c.startswith("deletes_L")]
    if lvl_cols and len(to):
        fig, ax = plt.subplots()
        top = to.sort_values("ndcg@10", ascending=False).head(8)
        top.set_index("config_name")[lvl_cols].plot.bar(ax=ax)
        ax.set_ylabel("exact E[DELETE] by level")
        ax.tick_params(axis="x", rotation=75, labelsize=6)
        _save(fig, plots_dir, f"{split}_deletes_by_level.png")


def build_report(
    *, out_dir: Path, run_id: str, validation: pd.DataFrame | None,
    test: pd.DataFrame | None, bootstrap: pd.DataFrame | None,
    serving: pd.DataFrame | None, protocol: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None, compute: pd.DataFrame | None,
    oracle_cost: pd.DataFrame | None,
) -> None:
    md: list[str] = [f"# exp5 budgeted cyclic — Beauty 2014 (run {run_id})", ""]
    payload: dict[str, Any] = {"run_id": run_id, "questions": {}}

    def q(num: int, title: str, lines: list[str]) -> None:
        md.append(f"## Q{num}. {title}")
        md.extend(lines or ["(not available in this run)"])
        md.append("")
        payload["questions"][f"q{num}"] = lines

    # ---- 1. protocol agreement -------------------------------------------
    lines = []
    if protocol:
        lines.append(
            f"Protocol regression on {protocol.get('n_users')} fixed users: "
            f"repo-evaluator vs trie-beam top-{protocol.get('beam')} agreement "
            f"= {protocol.get('topk_set_agreement'):.3f}; "
            f"hit agreement = {protocol.get('hit_agreement'):.3f}; "
            f"generate-constraint-fallback users = {protocol.get('n_generate_invalid')}.")
        if protocol.get("disagreement_note"):
            lines.append(f"Documented disagreement causes: {protocol['disagreement_note']}")
    q(1, "Do exact full-catalog and constrained beam evaluation agree?", lines)

    def _row(df: pd.DataFrame, fam: str) -> pd.Series | None:
        sub = df[df["policy_family"] == fam]
        return sub.iloc[0] if len(sub) else None

    lines = []
    for split_name, df in (("validation", validation), ("test", test)):
        if df is None:
            continue
        for fam in ("reference/tiger_exact_full_catalog", "reference/sasrec_full_catalog"):
            r = _row(df, fam)
            if r is not None:
                lines.append(
                    f"{split_name} {fam}: NDCG@10 {r['ndcg@10']:.4f}, "
                    f"Recall@10 {r['recall@10']:.4f}, MRR {r['mrr']:.4f} "
                    f"(n={int(r['n_users'])}).")
    q(2, "Full-split TIGER and SASRec reference metrics", lines)

    def _fam_lines(df, fams, label):
        out = []
        if df is None:
            return out
        base = _row(df, "reference/append_only_cyclic_parameterization")
        if base is None:
            return out
        for fam in fams:
            sub = df[df["policy_family"] == fam].sort_values("ndcg@10", ascending=False)
            if len(sub):
                r = sub.iloc[0]
                out.append(
                    f"{label}: best {fam} = {r['config_name']} NDCG@10 "
                    f"{r['ndcg@10']:.4f} vs append-only {base['ndcg@10']:.4f} "
                    f"(delta {r['ndcg@10'] - base['ndcg@10']:+.4f}).")
        return out

    q(3, "Does suffix-only rollback improve NDCG/Recall on the complete data?",
      _fam_lines(validation, ["suffix_only_rollback"], "validation")
      + _fam_lines(test, ["suffix_only_rollback"], "test"))
    q(4, "Does level-reserved budgeting beat hard global truncation?",
      _fam_lines(validation, ["level_budget_rollback", "truncated_infinite_oracle"],
                 "validation"))
    q(5, "Does the budget-aware controller beat the truncated infinite oracle?",
      _fam_lines(validation, ["finite_budget_controller", "truncated_infinite_oracle"],
                 "validation"))
    q(6, "Does rollback toward a global hybrid target beat pure-SASRec rollback?",
      _hybrid_lines(validation))
    q(7, "Local prefix guidance vs global hybrid terminal guidance",
      _fam_lines(validation, ["local_guidance", "global_hybrid_target"], "validation"))

    lines = []
    if bootstrap is not None and len(bootstrap):
        nd = bootstrap[bootstrap["metric"] == "ndcg@10"].sort_values("mean_delta",
                                                                     ascending=False)
        for _, r in nd.iterrows():
            adj = r.get("holm_adjusted_p")
            sig = ("SIGNIFICANT after Holm" if pd.notna(adj) and adj < 0.05 else
                   "not significant after Holm" if pd.notna(adj) else "nominal only")
            lines.append(
                f"{r['policy_family']} {r['config_name']}: delta NDCG@10 "
                f"{r['mean_delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                f"two-sided bootstrap p = {r['p_value_two_sided']:.4f}, "
                f"Holm-adjusted p = {adj if pd.notna(adj) else float('nan'):.4f} -> {sig}. "
                f"(bootstrap_prob_improvement = {r['bootstrap_prob_improvement']:.3f}; "
                "this probability is not a p-value.)")
    q(8, "Does any cyclic method beat simpler baselines at matched end-to-end cost? "
         "/ Q13. Are primary test improvements significant after multiplicity "
         "correction?", lines)

    lines = []
    if serving is not None and len(serving):
        cols = [c for c in ("candidate_recall@20", "candidate_recall@50") if c in serving]
        top = serving.sort_values(cols[0], ascending=False).head(8) if cols else serving.head(0)
        for _, r in top.iterrows():
            lines.append(
                f"{r['policy_family']} {r['config_name']} (serving-feasible): "
                + ", ".join(f"{c}={r[c]:.4f}" for c in cols)
                + f", mean deletes {r.get('expected_deletes', float('nan')):.2f}.")
    q(9, "Serving-style candidate quality (and the gate, where trained)", lines)

    lines = []
    if oracle_cost is not None and len(oracle_cost):
        c = oracle_cost["E_del_inf"].dropna()
        if len(c):
            top1 = float(c.max() / max(c.sum(), 1e-300))
            top10 = float(c.nlargest(10).sum() / max(c.sum(), 1e-300))
            lines.append(
                f"Infinite-oracle cost across users: median {c.median():.1f}, "
                f"p90 {c.quantile(0.9):.1f}, p95 {c.quantile(0.95):.1f}, "
                f"p99 {c.quantile(0.99):.1f}, max {c.max():.1f}; top-1 user share "
                f"{top1:.2%}, top-10 share {top10:.2%}. (oracle upper bound / "
                "diagnostic, not a serving method)")
    q(10, "How heavy-tailed is oracle cost across users?", lines)

    lines = []
    if validation is not None:
        lvl_cols = [c for c in validation.columns if c.startswith("deletes_L")]
        to = validation[validation["policy_family"] == "truncated_infinite_oracle"]
        if len(to) and lvl_cols:
            r = to.sort_values("expected_deletes", ascending=False).iloc[0]
            lines.append("Largest-budget oracle spend by level: " + ", ".join(
                f"{c}={r[c]:.3f}" for c in lvl_cols))
        rsel = [c for c in validation.columns if c.startswith("reselect_L")]
        if len(to) and rsel:
            r = to.sort_values("expected_deletes", ascending=False).iloc[0]
            lines.append("Same-token reselection by level (wasted compute): "
                         + ", ".join(f"{c}={r[c]:.3f}" for c in rsel))
    q(11, "At which SID levels is budget actually spent? / Q12. How much compute "
          "is wasted by same-token reselection?", lines)

    lines = []
    if bootstrap is not None and len(bootstrap):
        sig = bootstrap[(bootstrap["metric"] == "ndcg@10")
                        & (bootstrap.get("holm_adjusted_p", pd.Series(dtype=float)) < 0.05)]
        cyc = sig[sig["policy_family"].isin(
            ["truncated_infinite_oracle", "suffix_only_rollback",
             "level_budget_rollback", "finite_budget_controller"])]
        if len(cyc):
            lines.append(
                "A cyclic method shows a Holm-corrected significant improvement; "
                "proceeding to a learned rollback head is justified IF the gain "
                "also survives the matched-compute comparison in "
                "compute_metrics.csv.")
        else:
            lines.append(
                "No cyclic method shows a Holm-corrected significant improvement "
                "over append-only TIGER; per the interpretation constraints, do "
                "NOT proceed to a neural rollback head on this evidence.")
    q(14, "Is there sufficient evidence to proceed to a neural rollback head?", lines)

    md += [
        "## Interpretation constraints (verbatim policy)",
        "- Divergence improving while NDCG does not has multiple candidate causes: "
        "distributional-vs-ranking objective mismatch, finite-budget policy "
        "misspecification, hard-truncation bias, reward-model mismatch, numerical "
        "error, serving candidate limitations. No single-cause claims.",
        "- The infinite global-flow oracle and per-user free finite-budget "
        "optimization are oracle upper bounds / diagnostics, never serving methods.",
        "- Cached-arithmetic timings are reported as cached_policy_runtime; only "
        "benchmark rows measure end_to_end_runtime.",
        "",
    ]
    if selection:
        md.append("## Frozen validation selection")
        md.append(f"selection manifest sha256: `{selection.get('_selection_manifest_sha256')}`")
        for fam, cfg in (selection.get("selections") or {}).items():
            md.append(f"- {fam}: {cfg.get('config_name')} "
                      f"(validation {cfg.get('validation_metric_name')} = "
                      f"{cfg.get('validation_metric'):.4f}) [validation-selected primary]")
        md.append("")

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.md").write_text("\n".join(md) + "\n")
    (report_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {report_dir / 'report.md'}", flush=True)


IDENTITY_COLS = ["policy_family", "config_name", "canonical_config_hash",
                 "reward_transform", "target", "eval_kind"]


def read_prediction_shards(root: Path, split: str) -> pd.DataFrame:
    base = root / "predictions" / split
    frames = []
    for shard in sorted(base.glob("*/*/shard_*.parquet")):
        frames.append(pd.read_parquet(shard))
    if not frames:
        raise FileNotFoundError(f"no prediction shards under {base}")
    return pd.concat(frames, ignore_index=True)


def validate_same_users(
    df: pd.DataFrame, expected_indices: list[int], *, allow_partial: bool = False,
) -> None:
    """No policy-specific filtering may change the evaluated user set (spec §4).

    ``allow_partial`` (smoke/regression runs only) relaxes completeness against
    the full manifest but still enforces that every configuration covers the
    SAME user set.
    """
    expected = set(int(i) for i in expected_indices)
    if allow_partial:
        expected = set(int(i) for i in df["dataset_index"].unique())
    problems = []
    for (fam, name), sub in df.groupby(["policy_family", "config_name"]):
        got = set(int(i) for i in sub["dataset_index"])
        if got != expected:
            problems.append(
                f"{fam}/{name}: {len(got)} users (missing {len(expected - got)}, "
                f"extra {len(got - expected)})")
        if sub["dataset_index"].duplicated().any():
            problems.append(f"{fam}/{name}: duplicate users")
    if problems:
        raise AssertionError("user-universe violations: " + "; ".join(problems))


DECLARED_IDENTICAL_FAMILIES = [
    # spec §6: the B=0 cyclic parameterization MUST equal the exact ranking
    frozenset({"reference/tiger_exact_full_catalog",
               "reference/append_only_cyclic_parameterization"}),
    # tau = 1.0 decoder temperature is the identity transform
    frozenset({"reference/tiger_stochastic",
               "reference/append_only_cyclic_parameterization"}),
    frozenset({"reference/tiger_stochastic",
               "reference/tiger_exact_full_catalog"}),
]


def assert_unique_prediction_sets(df: pd.DataFrame) -> None:
    """Spec §7 guard: two configs with identical per-user predictions must not
    both enter a test table (duplicate transform-independent rows). Declared
    identity pairs (spec §6) are exempt."""
    import hashlib
    seen: dict[str, str] = {}
    for (fam, name, h), sub in df.groupby(
            ["policy_family", "config_name", "canonical_config_hash"]):
        sig_src = sub.sort_values("dataset_index")[
            ["dataset_index", "top20"]].to_csv(index=False)
        sig = hashlib.sha256(sig_src.encode()).hexdigest()
        if sig in seen and seen[sig].split("|")[2] != h:
            prev_fam = seen[sig].split("|")[0]
            if frozenset({fam, prev_fam}) in DECLARED_IDENTICAL_FAMILIES:
                continue
            raise AssertionError(
                f"duplicate per-user predictions: {fam}/{name} duplicates "
                f"{seen[sig]} — transform-independent method stored twice?")
        seen[sig] = f"{fam}|{name}|{h}"


def summarize_predictions(
    df: pd.DataFrame, *, num_items: int, base_fields: dict | None = None,
    coverage_boot: int = 500, seed: int = 42,
) -> pd.DataFrame:
    from .bootstrap import coverage_bootstrap
    from .metrics import parse_top20

    rows = []
    metric_cols = [c for c in df.columns
                   if c not in IDENTITY_COLS + ["user", "dataset_index", "top20",
                                                "status", "seed"]
                   and pd.api.types.is_numeric_dtype(df[c])]
    for key, sub in df.groupby(IDENTITY_COLS, dropna=False):
        ok = sub[sub["status"] == "ok"] if "status" in sub else sub
        if not len(ok):
            continue
        row = dict(zip(IDENTITY_COLS, key))
        row.update(base_fields or {})
        row["n_users"] = int(ok["dataset_index"].nunique())
        row["n_failed"] = int(len(sub) - len(ok))
        for c in metric_cols:
            row[c] = float(ok[c].mean()) if ok[c].notna().any() else np.nan
        lists = [parse_top20(s) for s in ok["top20"].dropna()]
        if lists:
            for k in (10, 20):
                cov = coverage_bootstrap(lists, num_items, k=k, n_boot=coverage_boot,
                                         seed=seed)
                row[f"coverage@{k}"] = cov["coverage"]
                row[f"coverage@{k}_ci_lo"] = cov["ci_lo"]
                row[f"coverage@{k}_ci_hi"] = cov["ci_hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def _hybrid_lines(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    to = df[df["policy_family"] == "truncated_infinite_oracle"]
    if not len(to):
        return []
    out = []
    sas = to[to["target"].str.startswith("exp_") | to["target"].str.startswith("softplus")]
    hyb = to[to["target"].str.startswith("hybrid")]
    if len(sas):
        r = sas.sort_values("ndcg@10", ascending=False).iloc[0]
        out.append(f"best rollback toward pure SASRec target: {r['config_name']} "
                   f"NDCG@10 {r['ndcg@10']:.4f}")
    if len(hyb):
        r = hyb.sort_values("ndcg@10", ascending=False).iloc[0]
        out.append(f"best rollback toward a global hybrid target: {r['config_name']} "
                   f"NDCG@10 {r['ndcg@10']:.4f}")
    return out
