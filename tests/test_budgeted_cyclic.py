"""Tests for the budgeted cyclic sampling machinery (flowrec.trajectories.budgeted).

All tests run on small synthetic tries so every quantity has an independent
exact reference: the append-only product distribution for B = 0, the
infinite-horizon cyclic distribution (beta recursion) for B -> inf, and Monte
Carlo agreement for the samplers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from flowrec.evaluation.ranking import (
    aggregate_rank_metrics,
    coverage_at_k,
    metrics_from_rank,
    rank_of_target,
    topk_rows,
)
from flowrec.trajectories.nonacyclic import (
    SemanticIDTrie,
    build_global_flow_oracle,
    exact_cyclic_terminal_distribution,
    exact_terminal_distribution,
    reward_conditionals,
    reward_mass,
    tv_distance,
)
from flowrec.trajectories.budgeted import (
    BudgetedCyclicSampler,
    best_of_n_exact_distribution,
    exact_budgeted_terminal_distribution,
    exact_level_budgeted_terminal_distribution,
    geometric_guidance_edge_prob,
    level_masked_rollback,
    optimize_budgeted_rollback,
    scaled_rollback_prob,
    temperature_edge_prob,
    trie_beam_search,
)
from flowrec.trajectories.nonacyclic import node_reach_probability


def make_trie(seed: int = 0, num_items: int = 60, sid_len: int = 4, codebook: int = 4):
    rng = np.random.default_rng(seed)
    sids = set()
    while len(sids) < num_items:
        sids.add(tuple(rng.integers(0, codebook, size=sid_len)))
    return SemanticIDTrie(np.asarray(sorted(sids), dtype=np.int64))


def full_grid_trie(sid_len: int = 3, codebook: int = 3):
    """Every internal node has exactly `codebook` children (no single-child nodes)."""
    sids = np.asarray(list(itertools.product(range(codebook), repeat=sid_len)), dtype=np.int64)
    return SemanticIDTrie(sids)


def random_edge_prob(trie: SemanticIDTrie, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.zeros(trie.num_nodes, dtype=np.float64)
    p[0] = 1.0
    for nd in trie.internal_node_ids:
        cs, cc = trie.child_start[nd], trie.child_count[nd]
        p[cs : cs + cc] = rng.dirichlet(np.ones(cc) * 2.0)
    return p


@pytest.fixture(scope="module")
def setup():
    trie = make_trie()
    p = random_edge_prob(trie)
    rng = np.random.default_rng(7)
    rewards = np.exp(rng.normal(0.0, 1.5, size=trie.num_items))
    mass = reward_mass(trie, rewards)
    q = reward_conditionals(trie, mass)
    oracle = build_global_flow_oracle(trie, p, q, mass)
    p_r = rewards / rewards.sum()
    return trie, p, rewards, mass, q, oracle, p_r


def test_budget_zero_matches_append_only(setup):
    trie, p, _, _, _, oracle, _ = setup
    res = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, budget=0)
    ref = exact_terminal_distribution(trie, p)
    assert tv_distance(res.terminal, ref) < 1e-12
    assert res.expected_deletes == 0.0
    assert res.p_any_delete == 0.0
    assert res.expected_steps == pytest.approx(trie.sid_len + 1)


def test_budget_accounting_identities(setup):
    trie, p, _, _, _, oracle, _ = setup
    prev_del = -1.0
    for budget in (0, 1, 2, 4, 8, 16):
        res = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, budget=budget)
        assert res.normalization_error < 1e-9
        assert res.delete_count_pmf.sum() == pytest.approx(1.0, abs=1e-9)
        # E[N_DELETE] is monotone in the budget for a truncated fixed policy.
        assert res.expected_deletes >= prev_del - 1e-12
        prev_del = res.expected_deletes
        # n_tau = (sid_len + 1) + 2 N_DELETE and ADD = sid_len + N_DELETE, in expectation.
        assert res.expected_add_steps == pytest.approx(
            trie.sid_len + res.expected_deletes, rel=1e-9, abs=1e-9
        )
        assert res.expected_steps == pytest.approx(
            (trie.sid_len + 1) + 2 * res.expected_deletes, rel=1e-9, abs=1e-9
        )
        assert 0.0 <= res.p_any_delete <= 1.0
        assert res.p_full_budget == pytest.approx(res.delete_count_pmf[budget], abs=1e-12)


def test_large_budget_converges_to_infinite_horizon_oracle(setup):
    trie, p, _, mass, _, oracle, p_r = setup
    inf_dist = exact_cyclic_terminal_distribution(trie, p, oracle.d_star)
    assert tv_distance(inf_dist, p_r) < 1e-9  # oracle regression: reproduces R/Z
    tvs = []
    for budget in (0, 2, 8, 32, 128, 512):
        res = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, budget=budget)
        tvs.append(tv_distance(res.terminal, p_r))
    assert all(a >= b - 1e-12 for a, b in zip(tvs, tvs[1:]))
    assert tvs[-1] < 1e-9
    # the truncated policy's cost also converges to the oracle's exact E[N_DELETE] = sum(kappa)/Z
    inf_deletes = float(oracle.kappa.sum() / mass[0])
    assert res.expected_deletes == pytest.approx(inf_deletes, rel=1e-9)


def test_sampler_matches_exact_dp(setup):
    trie, p, _, _, _, oracle, _ = setup
    budget = 3
    res = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, budget=budget)
    sampler = BudgetedCyclicSampler(trie, p, oracle.d_star, budget=budget)
    out = sampler.sample(200_000, seed=123)
    assert (out.terminal_row >= 0).all()
    assert out.n_deletes.max() <= budget
    assert out.n_steps.max() <= (trie.sid_len + 1) + 2 * budget
    assert (out.n_steps == (trie.sid_len + 1) + 2 * out.n_deletes).all()
    emp = np.bincount(out.terminal_row, minlength=trie.num_items) / len(out.terminal_row)
    assert tv_distance(emp, res.terminal) < 0.02
    assert out.n_deletes.mean() == pytest.approx(res.expected_deletes, rel=0.05, abs=0.01)
    emp_pmf = np.bincount(out.n_deletes, minlength=budget + 1) / len(out.n_deletes)
    assert np.abs(emp_pmf - res.delete_count_pmf).max() < 0.01


def test_sampler_seed_reproducible(setup):
    trie, p, _, _, _, oracle, _ = setup
    sampler = BudgetedCyclicSampler(trie, p, oracle.d_star, budget=2)
    a = sampler.sample(5_000, seed=9)
    b = sampler.sample(5_000, seed=9)
    assert (a.terminal_row == b.terminal_row).all()
    assert (a.n_deletes == b.n_deletes).all()


def test_no_immediate_repeat_masking():
    trie = full_grid_trie()
    p = random_edge_prob(trie, seed=3)
    rng = np.random.default_rng(11)
    rewards = np.exp(rng.normal(0.0, 2.0, size=trie.num_items))
    mass = reward_mass(trie, rewards)
    q = reward_conditionals(trie, mass)
    oracle = build_global_flow_oracle(trie, p, q, mass)
    sampler = BudgetedCyclicSampler(
        trie, p, oracle.d_star, budget=4, no_immediate_repeat=True, node_value=mass
    )
    out = sampler.sample(50_000, seed=5)
    assert out.n_mask_fallbacks == 0  # full grid: masks never empty the support
    assert out.n_same_token_reselect == 0
    assert out.n_revisions > 0
    assert 0 <= out.n_revisions_improved <= out.n_revisions
    # The unmasked sampler on the same policy does reselect sometimes.
    base = BudgetedCyclicSampler(trie, p, oracle.d_star, budget=4)
    out2 = base.sample(50_000, seed=5)
    assert out2.n_same_token_reselect > 0
    assert out2.n_revisions_improved == -1  # no node_value supplied


def test_geometric_guidance_endpoints(setup):
    trie, p, _, _, q, _, p_r = setup
    p0 = geometric_guidance_edge_prob(trie, p, q, alpha=0.0)
    assert np.abs(p0 - p).max() < 1e-12
    p1 = geometric_guidance_edge_prob(trie, p, q, alpha=1.0)
    dist = exact_terminal_distribution(trie, p1)
    assert tv_distance(dist, p_r) < 1e-9
    for alpha in (0.25, 0.5, 0.75):
        pa = geometric_guidance_edge_prob(trie, p, q, alpha=alpha)
        sums = np.add.reduceat(pa, trie.child_start[trie.internal_node_ids])
        assert np.abs(sums - 1.0).max() < 1e-12


def test_temperature_edge_prob(setup):
    trie, p, _, _, _, _, _ = setup
    logp = np.log(p)
    p1 = temperature_edge_prob(trie, logp, 1.0)
    assert np.abs(p1 - p).max() < 1e-12
    p_hot = temperature_edge_prob(trie, logp, 4.0)
    p_cold = temperature_edge_prob(trie, logp, 0.25)
    ent = lambda x: -(x[1:] * np.log(np.maximum(x[1:], 1e-300))).sum()
    assert ent(p_hot) > ent(p1) > ent(p_cold)


def test_scaled_rollback(setup):
    _, _, _, _, _, oracle, _ = setup
    d = oracle.d_star
    assert np.abs(scaled_rollback_prob(d, 0.0) - d).max() < 1e-12
    weaker = scaled_rollback_prob(d, 2.0)
    assert (weaker <= d + 1e-15).all()
    assert (scaled_rollback_prob(d, 60.0) < 1e-12).all()
    stronger = scaled_rollback_prob(d, -2.0)
    assert (stronger >= d - 1e-15).all()
    assert (stronger < 1.0).all()
    assert weaker[0] == 0.0 and stronger[0] == 0.0


def test_level_masked_rollback(setup):
    trie, p, _, _, _, oracle, _ = setup
    all_levels = level_masked_rollback(trie, oracle.d_star, set(range(1, trie.sid_len + 1)))
    assert np.abs(all_levels - oracle.d_star).max() < 1e-15
    none = level_masked_rollback(trie, oracle.d_star, set())
    res = exact_budgeted_terminal_distribution(trie, p, none, budget=8)
    ref = exact_terminal_distribution(trie, p)
    assert tv_distance(res.terminal, ref) < 1e-12
    only3 = level_masked_rollback(trie, oracle.d_star, {3})
    assert (only3[trie.level != 3] == 0).all()
    assert np.abs(only3[trie.level == 3] - oracle.d_star[trie.level == 3]).max() < 1e-15


def test_best_of_n_exact(setup):
    trie, p, rewards, _, _, _, _ = setup
    p_append = exact_terminal_distribution(trie, p)
    b1 = best_of_n_exact_distribution(p_append, rewards, 1)
    assert tv_distance(b1, p_append) < 1e-12
    n = 5
    exact = best_of_n_exact_distribution(p_append, rewards, n)
    rng = np.random.default_rng(17)
    draws = rng.choice(trie.num_items, size=(100_000, n), p=p_append)
    picked = draws[np.arange(len(draws)), np.argmax(rewards[draws], axis=1)]
    emp = np.bincount(picked, minlength=trie.num_items) / len(picked)
    assert tv_distance(emp, exact) < 0.02
    # more candidates concentrate more mass on high-reward items
    e10 = best_of_n_exact_distribution(p_append, rewards, 10)
    assert (e10 * rewards).sum() > (exact * rewards).sum() > (p_append * rewards).sum()


def test_trie_beam_search(setup):
    trie, p, _, _, _, _, _ = setup
    logp = np.log(p)
    p_exact = exact_terminal_distribution(trie, p)
    rows, scores = trie_beam_search(trie, logp, beam_size=trie.num_items)
    assert len(rows) == trie.num_items
    assert np.abs(np.exp(scores) - p_exact[rows]).max() < 1e-12
    assert (np.diff(scores) <= 1e-12).all()
    rows5, scores5 = trie_beam_search(trie, logp, beam_size=5)
    # narrow beams are heuristic, but returned scores must still be the true
    # append-only log-probabilities of the returned items, sorted descending
    assert len(rows5) == 5
    assert np.abs(np.exp(scores5) - p_exact[rows5]).max() < 1e-12
    assert (np.diff(scores5) <= 1e-12).all()


def test_optimize_budgeted_rollback_improves_js(setup):
    trie, p, _, _, _, oracle, p_r = setup
    budget = 2
    base = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, budget=budget)
    out = optimize_budgeted_rollback(
        trie, p, p_r, budget=budget, init_rollback=np.clip(oracle.d_star, 1e-6, 1 - 1e-6),
        expected_deletes_cap=float(base.expected_deletes), steps=60, lr=0.1,
    )
    assert out.history[-1]["js"] <= out.history[0]["js"] + 1e-9
    res = exact_budgeted_terminal_distribution(trie, p, out.rollback_prob, budget=budget)
    assert res.normalization_error < 1e-9


def test_level_budget_dp_single_level_equals_masked_scalar(setup):
    trie, p, _, _, _, oracle, _ = setup
    for lvl in range(1, trie.sid_len + 1):
        budget = 2
        alloc = tuple(budget if k == lvl else 0 for k in range(1, trie.sid_len + 1))
        lv = exact_level_budgeted_terminal_distribution(trie, p, oracle.d_star, alloc)
        masked = level_masked_rollback(trie, oracle.d_star, {lvl})
        sc = exact_budgeted_terminal_distribution(trie, p, masked, budget)
        assert tv_distance(lv.terminal, sc.terminal) < 1e-12
        assert lv.expected_deletes == pytest.approx(sc.expected_deletes, rel=1e-9, abs=1e-12)
        # quota only spent at the allowed level
        other = [k for k in range(1, trie.sid_len + 1) if k != lvl]
        assert all(lv.expected_deletes_by_level[k] == 0.0 for k in other)


def test_level_budget_dp_accounting(setup):
    trie, p, _, _, _, oracle, _ = setup
    alloc = (1, 1, 1, 1)
    res = exact_level_budgeted_terminal_distribution(trie, p, oracle.d_star, alloc)
    assert res.normalization_error < 1e-9
    for k in range(1, trie.sid_len + 1):
        assert res.expected_deletes_by_level[k] <= alloc[k - 1] + 1e-12
    assert res.expected_steps == pytest.approx(
        (trie.sid_len + 1) + 2 * res.expected_deletes, rel=1e-9)
    assert 0.0 <= res.p_full_budget <= 1.0
    # zero allocation reproduces append-only exactly
    res0 = exact_level_budgeted_terminal_distribution(trie, p, oracle.d_star, (0, 0, 0, 0))
    assert tv_distance(res0.terminal, exact_terminal_distribution(trie, p)) < 1e-12


def test_level_budget_sampler_respects_quotas(setup):
    trie, p, _, _, _, oracle, _ = setup
    alloc = (1, 0, 0, 0)
    sampler = BudgetedCyclicSampler(trie, p, oracle.d_star, level_budgets=alloc)
    out = sampler.sample(50_000, seed=3)
    assert out.n_deletes.max() <= 1
    assert all(out.delete_level_counts[k] == 0 for k in (2, 3, 4))
    res = exact_level_budgeted_terminal_distribution(trie, p, oracle.d_star, alloc)
    emp = np.bincount(out.terminal_row, minlength=trie.num_items) / len(out.terminal_row)
    assert tv_distance(emp, res.terminal) < 0.02
    assert out.n_deletes.mean() == pytest.approx(res.expected_deletes, rel=0.1, abs=0.01)


def test_budget_dependent_rollback_rows(setup):
    trie, p, _, _, _, oracle, _ = setup
    budget = 3
    tiled = np.tile(oracle.d_star, (budget + 1, 1))
    a = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, budget)
    b = exact_budgeted_terminal_distribution(trie, p, tiled, budget)
    assert tv_distance(a.terminal, b.terminal) < 1e-15
    # a genuinely budget-dependent policy: weaker rollback as budget depletes
    rows = np.stack([oracle.d_star * (bb / budget) for bb in range(budget + 1)])
    rows[0] = 0.0
    res = exact_budgeted_terminal_distribution(trie, p, rows, budget)
    assert res.normalization_error < 1e-9
    assert res.expected_deletes < a.expected_deletes  # weaker policy deletes less
    sampler = BudgetedCyclicSampler(trie, p, rows, budget=budget)
    out = sampler.sample(100_000, seed=11)
    emp = np.bincount(out.terminal_row, minlength=trie.num_items) / len(out.terminal_row)
    assert tv_distance(emp, res.terminal) < 0.02
    assert out.n_deletes.mean() == pytest.approx(res.expected_deletes, rel=0.05, abs=0.01)


def test_state_visitation_mass(setup):
    trie, p, _, _, _, oracle, _ = setup
    res0 = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, 0)
    reach = node_reach_probability(trie, p)
    assert np.abs(res0.state_visitation_mass - reach).max() < 1e-12
    res = exact_budgeted_terminal_distribution(trie, p, oracle.d_star, 4)
    # every ADD/DELETE transition arrives at exactly one trie state, plus the
    # initial root visit; STOP leaves the trie: sum(visits) == E[n_tau]
    assert res.state_visitation_mass.sum() == pytest.approx(res.expected_steps, rel=1e-9)
    assert res.state_visitation_mass[0] >= 1.0
    lv = exact_level_budgeted_terminal_distribution(trie, p, oracle.d_star, (1, 1, 1, 1))
    assert lv.state_visitation_mass.sum() == pytest.approx(lv.expected_steps, rel=1e-9)


def test_ranking_metrics():
    scores = np.array([0.5, 0.9, 0.1, 0.9])
    assert rank_of_target(scores, 1) == 1  # tie with index 3 does not hurt
    assert rank_of_target(scores, 0) == 3
    m = metrics_from_rank(3, ks=(5, 10))
    assert m["recall@5"] == 1.0 and m["ndcg@5"] == pytest.approx(0.5)
    agg = aggregate_rank_metrics(np.array([1, 3, 100]), ks=(5,))
    assert agg["recall@5"] == pytest.approx(2 / 3)
    assert coverage_at_k([np.array([0, 1]), np.array([1, 2])], 10) == pytest.approx(0.3)
    top = topk_rows(scores, 2)
    assert set(top) == {1, 3}
