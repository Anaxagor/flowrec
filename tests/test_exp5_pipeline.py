"""Unit tests for the exp5_budgeted_cyclic cluster pipeline (spec §29).

Complements tests/test_budgeted_cyclic.py (which covers the exact DPs and
samplers in flowrec.trajectories.budgeted). Everything here runs on synthetic
tries — no model checkpoints or datasets required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_budgeted_cyclic import make_trie, random_edge_prob  # noqa: E402

from flowrec.trajectories.nonacyclic import (  # noqa: E402
    build_global_flow_oracle,
    exact_terminal_distribution,
    reward_conditionals,
    reward_mass,
    tv_distance,
)

from exp5_budgeted_cyclic import common  # noqa: E402
from exp5_budgeted_cyclic import finite_budget_dp as dp  # noqa: E402
from exp5_budgeted_cyclic.bootstrap import (  # noqa: E402
    average_over_seeds,
    coverage_bootstrap,
    holm_adjust,
    paired_bootstrap,
)
from exp5_budgeted_cyclic.metrics import (  # noqa: E402
    candidate_prediction,
    full_catalog_prediction,
)
from exp5_budgeted_cyclic.oracle import build_log_flow_oracle, rollback_for_dp  # noqa: E402
from exp5_budgeted_cyclic.policies import (  # noqa: E402
    PolicyConfig,
    UserPolicyContext,
    assert_no_duplicate_configs,
    build_validation_grid,
    evaluate_policy,
)
from exp5_budgeted_cyclic.reports import (  # noqa: E402
    assert_unique_prediction_sets,
    validate_same_users,
)
from exp5_budgeted_cyclic.rewards import (  # noqa: E402
    hybrid_log_target,
    log_conditionals,
    log_prefix_masses,
    log_rewards,
    parse_transform,
    segment_logsumexp,
)


@pytest.fixture(scope="module")
def setup():
    trie = make_trie()
    p = random_edge_prob(trie)
    rng = np.random.default_rng(7)
    raw = rng.normal(0.0, 3.0, size=trie.num_items)
    return trie, p, np.log(p), raw


# --------------------------------------------------------------------------- #
# Log-space rewards / masses (spec 11.1)
# --------------------------------------------------------------------------- #
def test_log_rewards_match_linear_reference(setup):
    trie, p, logp, raw = setup
    lr = log_rewards(raw, parse_transform("exp_T1"))
    lin = np.exp(raw - raw.max())
    mass = reward_mass(trie, lin)
    lm = log_prefix_masses(trie, lr.log_r)
    assert np.abs(lm - np.log(mass)).max() < 1e-12
    lq = log_conditionals(trie, lm)
    assert np.abs(lq - np.log(reward_conditionals(trie, mass))).max() < 1e-12
    # conditionals normalize (log-space)
    internal = trie.internal_node_ids
    sums = segment_logsumexp(lq, trie.child_start[internal],
                             trie.child_count[internal])
    assert np.abs(sums).max() < 1e-12


def test_sharp_transform_no_artificial_zeros(setup):
    trie, _, _, raw = setup
    lr = log_rewards(raw * 40, parse_transform("exp_T0.05"))
    assert np.isfinite(lr.log_r).all()          # no -inf from underflow
    assert lr.stats["num_underflow_prevented"] > 0
    assert lr.stats["reward_floor"] is None     # no silent flooring
    lm = log_prefix_masses(trie, lr.log_r)
    assert np.isfinite(lm).all()
    lq = log_conditionals(trie, lm)
    internal = trie.internal_node_ids
    sums = segment_logsumexp(lq, trie.child_start[internal],
                             trie.child_count[internal])
    assert np.abs(sums).max() < 1e-9


def test_explicit_floor_is_counted(setup):
    _, _, _, raw = setup
    spec = parse_transform("exp_T0.25")
    floored = log_rewards(raw, type(spec)(spec.name, spec.kind, spec.temperature,
                                          log_floor=-30.0))
    assert floored.stats["reward_floor"] == -30.0
    assert floored.stats["num_floor_clamped"] == int(
        ((raw - raw.max()) / 0.25 < -30.0).sum())


def test_monotone_transform_preserves_raw_ranking(setup):
    _, _, _, raw = setup
    for name in ("exp_T0.5", "exp_T1", "softplus_T1"):
        lr = log_rewards(raw, parse_transform(name))
        assert (np.argsort(-lr.log_r, kind="stable")
                == np.argsort(-raw, kind="stable")).all()


def test_hybrid_target_endpoints(setup):
    trie, p, _, raw = setup
    p0 = exact_terminal_distribution(trie, p)
    h0 = hybrid_log_target(np.log(p0), raw, alpha=0.0, temperature=1.0)
    assert tv_distance(h0.p_target, p0) < 1e-12
    h1 = hybrid_log_target(np.log(p0), raw, alpha=1.0, temperature=1.0)
    b = log_rewards(raw, parse_transform("exp_T1"))
    assert tv_distance(h1.p_target, b.p_target) < 1e-12


# --------------------------------------------------------------------------- #
# Log-space oracle (spec 11.2/11.3)
# --------------------------------------------------------------------------- #
def test_log_oracle_matches_linear_reference(setup):
    trie, p, logp, raw = setup
    lr = log_rewards(raw, parse_transform("exp_T1"))
    lin = np.exp(lr.log_r)
    mass = reward_mass(trie, lin)
    o_lin = build_global_flow_oracle(trie, p, reward_conditionals(trie, mass), mass)
    lm = log_prefix_masses(trie, lr.log_r)
    o_log = build_log_flow_oracle(trie, logp, log_conditionals(trie, lm), lm)
    assert np.abs(o_log.d_star - o_lin.d_star).max() < 1e-9
    assert o_log.E_del_inf == pytest.approx(float(o_lin.kappa.sum() / mass[0]), rel=1e-9)
    assert o_log.diagnostics["n_support_violation"] == 0
    assert o_log.diagnostics["max_relative_error_on_edges_with_target_mass_above_threshold"] < 1e-9


def test_log_oracle_sharp_regime_classification(setup):
    trie, _, logp, raw = setup
    lr = log_rewards(raw * 40, parse_transform("exp_T0.05"))
    lm = log_prefix_masses(trie, lr.log_r)
    o = build_log_flow_oracle(trie, logp, log_conditionals(trie, lm), lm)
    assert np.isfinite(o.d_star).all() and (o.d_star >= 0).all() and (o.d_star <= 1).all()
    assert o.diagnostics["n_saturated_rollback"] > 0    # mathematical saturation
    d, info = rollback_for_dp(o)
    assert (d < 1.0).all()
    assert info["n_impl_clipped"] >= o.diagnostics["n_saturated_rollback"] - 1
    assert info["clip_value"] == 1.0 - 1e-12            # explicit, recorded clipping
    # scale-aware identity errors stay tiny even here
    assert o.diagnostics["mass_weighted_relative_error"] < 1e-9


# --------------------------------------------------------------------------- #
# Candidate-vs-full-catalog rank discipline (spec 8)
# --------------------------------------------------------------------------- #
def test_candidate_prediction_never_reports_catalog_rank():
    ranked = np.array([5, 3, 9])
    hit = candidate_prediction(ranked, 3, candidate_set_size=3)
    assert hit["eval_kind"] == "candidate"
    assert hit["candidate_hit"] is True
    assert hit["rank_within_candidates"] == 2
    assert hit["censored_rank"] == 2
    assert hit["full_catalog_rank"] is None
    miss = candidate_prediction(ranked, 777, candidate_set_size=3)
    assert miss["candidate_hit"] is False
    assert miss["rank_within_candidates"] is None
    assert miss["censored_rank"] == 4                  # M + 1
    assert miss["full_catalog_rank"] is None
    full = full_catalog_prediction(np.array([0.1, 0.9, 0.5]), 2)
    assert full["full_catalog_rank"] == 2 and full["candidate_hit"] is None


# --------------------------------------------------------------------------- #
# Policy taxonomy / dedup (spec 6, 7, 12)
# --------------------------------------------------------------------------- #
def test_canonical_hash_dedup_and_grid():
    grid = {
        "transforms": ["exp_T1"], "beams": [10], "rerank_beams": [20],
        "taus": [0.7], "best_of_n": [2], "budgets": [0, 1],
        "lambda_grid": [1.0], "scaled_budgets": [2], "suffix_budgets": [1],
        "level_masks": ["L4"], "level_mask_budgets": [2],
        "level_allocations": {1: [[1, 0, 0, 0]], 2: "all_compositions"},
        "alpha_local": [0.5], "hybrid": {"alpha": [0.5], "T": [1.0]},
        "hybrid_rollback_budgets": [1],
        "controller": {"budget": 4, "transforms": ["exp_T1"]},
    }
    configs = build_validation_grid(grid)
    assert_no_duplicate_configs(configs)
    # budget 0 never enters as truncated oracle (it IS the append-only reference)
    assert not any(c.family == "truncated_infinite_oracle" and c.param("budget") == 0
                   for c in configs)
    # all compositions of 2 over 4 levels
    allocs = [c.param("allocation") for c in configs
              if c.family == "level_budget_rollback" and c.param("mode") == "alloc"]
    assert len([a for a in allocs if sum(a) == 2]) == 10
    # transform-independent methods carry transform=None (stored once)
    beam = [c for c in configs if c.family == "reference/tiger_beam"]
    assert len(beam) == 1 and beam[0].transform is None
    with pytest.raises(ValueError):
        assert_no_duplicate_configs(configs + [beam[0]])
    # renamed policies exist; old names do not
    fams = {c.family for c in configs}
    assert "truncated_infinite_oracle" in fams and "local_guidance" in fams
    assert "truncated_oracle" not in fams and "direct_guidance" not in fams
    # round-trip through the canonical dict (selection manifest -> stage 07)
    for c in configs:
        rt = PolicyConfig.from_canonical(json.loads(json.dumps(c.canonical())))
        assert rt.canonical_hash() == c.canonical_hash()


def _user_ctx(setup, target_row=0):
    trie, p, logp, raw = setup
    return UserPolicyContext(
        trie, dataset_index=0, raw_user_id="u0", target_row=target_row,
        edge_logprob=logp, raw_scores=raw)


def test_append_only_parameterization_equals_exact(setup):
    u = _user_ctx(setup)
    exact = evaluate_policy(PolicyConfig("reference/tiger_exact_full_catalog"), u)
    b0 = evaluate_policy(
        PolicyConfig("reference/append_only_cyclic_parameterization"), u)
    assert b0["check_b0_equals_exact"] is True
    assert b0["full_catalog_rank"] == exact["full_catalog_rank"]
    assert b0["top20"] == exact["top20"]


def test_alpha1_local_guidance_equals_sasrec_ranking(setup):
    trie, _, _, raw = setup
    target_row = int(np.argsort(-raw)[3])
    u = _user_ctx(setup, target_row=target_row)
    g = evaluate_policy(PolicyConfig(
        "local_guidance", (("alpha", 1.0),), transform="exp_T1", target="sasrec"), u)
    s = evaluate_policy(PolicyConfig("reference/sasrec_full_catalog"), u)
    assert g["full_catalog_rank"] == s["full_catalog_rank"] == 4
    assert g["tv_to_target"] < 1e-9


def test_level_alloc_policy_matches_masked_scalar(setup):
    trie, p, _, raw = setup
    u = _user_ctx(setup)
    alloc = evaluate_policy(PolicyConfig(
        "level_budget_rollback",
        (("mode", "alloc"), ("allocation", (0, 0, 0, 2))),
        transform="exp_T1", target="sasrec"), u)
    suffix = evaluate_policy(PolicyConfig(
        "suffix_only_rollback", (("budget", 2),),
        transform="exp_T1", target="sasrec"), u)
    assert alloc["top20"] == suffix["top20"]
    assert alloc["expected_deletes"] == pytest.approx(suffix["expected_deletes"], rel=1e-9)


def test_no_repeat_is_serving_only(setup):
    u = _user_ctx(setup)
    with pytest.raises(ValueError):
        evaluate_policy(PolicyConfig("no_repeat_heuristic"), u)


# --------------------------------------------------------------------------- #
# Statistics (spec 14)
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_and_terminology():
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.05, 0.1, size=400)
    out = paired_bootstrap(deltas, n_boot=2000, seed=1)
    assert out["ci_lo"] < out["mean_delta"] < out["ci_hi"]
    assert 0 <= out["p_value_two_sided"] <= 1
    assert "bootstrap_prob_improvement" in out and "p_improved" not in out
    null = paired_bootstrap(rng.normal(0, 0.1, size=400), n_boot=2000, seed=1)
    assert null["p_value_two_sided"] > 0.05


def test_holm_adjustment_monotone_and_bounded():
    adj = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.5)
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_seed_averaging_before_bootstrap():
    df = pd.DataFrame({
        "user": ["a", "a", "b", "b"], "seed": [1, 2, 1, 2],
        "ndcg@10": [0.0, 1.0, 0.5, 0.5]})
    out = average_over_seeds(df, metric_cols=["ndcg@10"])
    assert len(out) == 2
    assert out.set_index("user").loc["a", "ndcg@10"] == pytest.approx(0.5)


def test_coverage_bootstrap_is_set_statistic():
    lists = [np.array([0, 1]), np.array([1, 2]), np.array([2, 3])]
    out = coverage_bootstrap(lists, 10, k=2, n_boot=500, seed=0)
    assert out["coverage"] == pytest.approx(0.4)
    assert out["ci_lo"] <= out["coverage"] <= out["ci_hi"]


# --------------------------------------------------------------------------- #
# Table guards (spec 4, 7)
# --------------------------------------------------------------------------- #
def _pred_frame(fam, name, users, top="1,2"):
    return pd.DataFrame({
        "policy_family": fam, "config_name": name,
        "canonical_config_hash": name, "dataset_index": users,
        "top20": top})


def test_validate_same_users_guard():
    df = pd.concat([_pred_frame("a", "a1", [1, 2, 3]),
                    _pred_frame("b", "b1", [1, 2])])
    with pytest.raises(AssertionError):
        validate_same_users(df, [1, 2, 3])
    validate_same_users(pd.concat([_pred_frame("a", "a1", [1, 2]),
                                   _pred_frame("b", "b1", [1, 2])]), [1, 2])


def test_duplicate_prediction_guard():
    df = pd.concat([_pred_frame("local_guidance", "g_expT1", [1, 2]),
                    _pred_frame("local_guidance", "g_expT2_duplicate", [1, 2])])
    with pytest.raises(AssertionError):
        assert_unique_prediction_sets(df)
    # declared identity pair is exempt
    ok = pd.concat([
        _pred_frame("reference/tiger_exact_full_catalog", "exact", [1, 2]),
        _pred_frame("reference/append_only_cyclic_parameterization", "b0", [1, 2])])
    assert_unique_prediction_sets(ok)


def test_json_safe_booleans():
    out = common.json_safe({"ok": np.bool_(True), "x": np.float64(1.5),
                            "n": np.int64(3), "arr": np.array([True, False])})
    assert out["ok"] is True and isinstance(out["ok"], bool)
    assert out["arr"] == [True, False]
    assert json.loads(json.dumps(out))["ok"] is True


def test_shard_rows_partition():
    class A:
        shard_id, num_shards, user_start, user_end = 0, 4, None, None
    all_rows = []
    for s in range(4):
        a = A()
        a.shard_id = s
        rows = common.shard_rows(103, a)
        all_rows.append(rows)
    cat = np.concatenate(all_rows)
    assert len(cat) == 103 and len(np.unique(cat)) == 103
    assert (np.sort(cat) == np.arange(103)).all()


# --------------------------------------------------------------------------- #
# Controller (spec 23)
# --------------------------------------------------------------------------- #
def test_controller_fit_and_rollback_rows(setup):
    trie, p, logp, raw = setup
    lr = log_rewards(raw, parse_transform("exp_T1"))
    lm = log_prefix_masses(trie, lr.log_r)
    o = build_log_flow_oracle(trie, logp, log_conditionals(trie, lm), lm)
    d_dp, _ = rollback_for_dp(o)
    feats = dp.controller_features(trie, d_dp, o.log_C, np.zeros(trie.num_nodes))
    users = [{"edge_prob": p, "features": feats, "p_target": lr.p_target,
              "dataset_index": 0}]
    art = dp.fit_budget_controller(trie, users, budget=2, target_name="exp_T1",
                                   steps=25, lr=0.1, seed=0)
    assert art.history[-1]["mean_js"] <= art.history[0]["mean_js"] + 1e-9
    rows = art.rollback_rows(trie, feats)
    assert rows.shape == (3, trie.num_nodes)
    assert (rows[0] == 0).all() and (rows[:, 0] == 0).all()
    res = dp.exact_budgeted_terminal_distribution(trie, p, rows, 2)
    assert res.normalization_error < 1e-9
