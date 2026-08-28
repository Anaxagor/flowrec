"""Budgeted (finite-rollback) cyclic GFlowNet sampling on the TIGER SID trie.

Extends :mod:`flowrec.trajectories.nonacyclic` with an explicit hard rollback
budget: the augmented state is ``(s, b)`` where ``b`` is the number of DELETE
actions still available. For ``b > 0`` the policy is the truncated version of
any per-node rollback policy ``d(c, s)`` (e.g. the global flow oracle's
``d_star``):

    P_F((parent(s), b-1) | (s, b)) = d(c, s)
    P_F((v, b)           | (s, b)) = (1 - d(c, s)) p_theta(v | c, s)

and at a complete item state ``x``: STOP with ``1 - d(c, x)``, DELETE with
``d(c, x)``. When the budget is exhausted DELETE is disabled:

    P_F((v, 0) | (s, 0)) = p_theta(v | c, s),   P_F(s_f | (x, 0)) = 1.

Because DELETE strictly decreases ``b`` and ADD strictly increases depth at a
fixed ``b``, the augmented graph is acyclic under the order (decreasing budget,
increasing depth), so the exact terminal distribution ``P_T^(B)(x | c)`` is a
finite dynamic program (:func:`exact_budgeted_terminal_distribution`).

The module also provides the budgeted Monte Carlo sampler (with an optional
no-immediate-repeat mask), the scaled-oracle rollback transform, level-masked
rollback, direct reward-guided decoding, decoder-temperature transforms, exact
Best-of-N + reward selection, a deterministic trie beam search, and an optional
differentiable budget-optimal rollback optimizer (phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .nonacyclic import SemanticIDTrie

__all__ = [
    "BudgetedTerminalResult",
    "LevelBudgetedTerminalResult",
    "exact_budgeted_terminal_distribution",
    "exact_level_budgeted_terminal_distribution",
    "scaled_rollback_prob",
    "level_masked_rollback",
    "BudgetedSampleResult",
    "BudgetedCyclicSampler",
    "geometric_guidance_edge_prob",
    "temperature_edge_prob",
    "best_of_n_exact_distribution",
    "trie_beam_search",
    "optimize_budgeted_rollback",
]


# ---------------------------------------------------------------------------
# Exact finite-budget dynamic program
# ---------------------------------------------------------------------------


@dataclass
class BudgetedTerminalResult:
    """Exact quantities of the budgeted cyclic policy for one user context.

    terminal:            [num_items] exact P_T^(B)(x | c) over catalog rows.
    delete_count_pmf:    [B+1] exact P(N_DELETE = k); trajectories absorbed in
                         budget slice ``b`` performed exactly ``B - b`` deletes.
    expected_deletes:    E[N_DELETE].
    expected_deletes_by_level: [sid_len+1] E[# DELETEs whose deleted state has
                         that SID level] (level 0 unused: the root never pops).
    p_any_delete:        P(N_DELETE >= 1).
    p_full_budget:       P(N_DELETE = B).
    expected_add_steps:  E[# ADD transitions] (= sid_len + E[N_DELETE]).
    expected_steps:      E[n_tau] = ADD + DELETE + STOP transitions
                         (= sid_len + 1 + 2 E[N_DELETE]).
    expected_decoder_evals: expected TIGER next-token evaluations, one per ADD
                         proposal at an internal state (= expected_add_steps).
    expected_same_token_reselect: E[# DELETE events immediately followed by an
                         ADD re-selecting the deleted token].
    expected_same_token_reselect_by_level: [sid_len+1] the same, grouped by the
                         SID level of the deleted state.
    normalization_error: |sum_x P_T^(B)(x) - 1| before renormalization.
    state_visitation_mass: [num_nodes] expected visits to each trie state,
                         summed over budget slices.
    """

    terminal: np.ndarray
    delete_count_pmf: np.ndarray
    expected_deletes: float
    expected_deletes_by_level: np.ndarray
    p_any_delete: float
    p_full_budget: float
    expected_add_steps: float
    expected_steps: float
    expected_decoder_evals: float
    expected_same_token_reselect: float
    expected_same_token_reselect_by_level: np.ndarray
    normalization_error: float
    state_visitation_mass: np.ndarray | None = None


def _check_policy_inputs(
    trie: SemanticIDTrie, edge_prob: np.ndarray, rollback_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the proposal and a 1-D or 2-D rollback array.

    ``rollback_prob`` may be [num_nodes] (budget-independent) or
    [budget+1, num_nodes] with row ``b`` = d(c, s, b) used while ``b`` DELETEs
    remain (row 0 is ignored: DELETE is disabled at b = 0).
    """
    p = np.asarray(edge_prob, dtype=np.float64)
    d = np.asarray(rollback_prob, dtype=np.float64)
    if p.shape != (trie.num_nodes,):
        raise ValueError("edge_prob must have shape (num_nodes,)")
    if d.ndim not in (1, 2) or d.shape[-1] != trie.num_nodes:
        raise ValueError("rollback_prob must be [num_nodes] or [budget+1, num_nodes]")
    if not (d[..., 0] == 0.0).all():
        raise ValueError("d(root) must be 0: the empty prefix has no parent")
    if not ((d >= 0) & (d < 1)).all():
        raise ValueError("rollback probabilities must be in [0, 1)")
    return p, d


def _rollback_rows(
    trie: SemanticIDTrie, d: np.ndarray, budget: int
) -> np.ndarray:
    """[budget+1, num_nodes] rollback table; row 0 forced to zero (b = 0)."""
    n = trie.num_nodes
    if d.ndim == 1:
        out = np.tile(d, (budget + 1, 1))
    else:
        if d.shape[0] != budget + 1:
            raise ValueError(
                f"2-D rollback_prob must have budget+1={budget + 1} rows, got {d.shape[0]}")
        out = d.copy()
    out[0] = 0.0
    return out


def exact_budgeted_terminal_distribution(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    rollback_prob: np.ndarray,
    budget: int,
    *,
    atol: float = 1e-9,
) -> BudgetedTerminalResult:
    """Exact terminal distribution of the hard-budget cyclic policy by DP.

    Probability mass is propagated over budget slices ``b = budget .. 0``;
    within a slice mass only flows downward (ADD), and DELETE mass moves to the
    parent state in slice ``b - 1``. ``budget = 0`` exactly reproduces
    append-only constrained sampling.
    """
    if budget < 0:
        raise ValueError("budget must be >= 0")
    p, d_raw = _check_policy_inputs(trie, edge_prob, rollback_prob)
    d_rows = _rollback_rows(trie, d_raw, budget)

    n = trie.num_nodes
    L = trie.sid_len
    term = trie.terminal_node_ids
    parent = trie.parent

    terminal = np.zeros(trie.num_items, dtype=np.float64)
    delete_count_pmf = np.zeros(budget + 1, dtype=np.float64)
    deletes_by_level = np.zeros(L + 1, dtype=np.float64)
    expected_deletes = 0.0
    expected_add_steps = 0.0
    same_reselect_by_level = np.zeros(L + 1, dtype=np.float64)
    visitation = np.zeros(n, dtype=np.float64)

    mass_in = np.zeros(n, dtype=np.float64)
    mass_in[0] = 1.0

    for b in range(budget, -1, -1):
        d_eff = d_rows[b]
        reach = mass_in
        # Top-down within the slice: parents are final before their children.
        for depth in range(1, L + 1):
            ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
            par = parent[ids]
            reach[ids] += reach[par] * (1.0 - d_eff[par]) * p[ids]
        visitation += reach
        keep = reach * (1.0 - d_eff)
        delete_out = reach * d_eff

        absorbed = keep[term]
        terminal[trie.item_row[term]] += absorbed
        delete_count_pmf[budget - b] = float(absorbed.sum())

        expected_add_steps += float(
            keep[trie.internal_node_ids].sum()
        )
        if b > 0:
            expected_deletes += float(delete_out[1:].sum())
            np.add.at(deletes_by_level, trie.level[1:], delete_out[1:])
            # After deleting v -> s at slice b-1, the immediate same-token
            # re-selection happens with prob (1 - d_eff(s, b-1)) p(v).
            d_next = d_rows[b - 1]
            v = np.arange(1, n)
            np.add.at(
                same_reselect_by_level,
                trie.level[v],
                delete_out[v] * (1.0 - d_next[parent[v]]) * p[v],
            )
            next_in = np.zeros(n, dtype=np.float64)
            np.add.at(next_in, parent[1:], delete_out[1:])
            mass_in = next_in
        else:
            if float(delete_out.sum()) != 0.0:
                raise AssertionError("DELETE mass produced at exhausted budget")

    total = float(terminal.sum())
    norm_err = abs(total - 1.0)
    if norm_err > atol:
        raise AssertionError(
            f"budgeted terminal distribution sums to {total} (err {norm_err:.3e})"
        )
    terminal = terminal / total

    pmf_deletes = float(np.sum(np.arange(budget + 1) * delete_count_pmf))
    if abs(pmf_deletes - expected_deletes) > 1e-6 * max(1.0, expected_deletes):
        raise AssertionError(
            f"delete-count accounting mismatch: pmf {pmf_deletes} vs flow {expected_deletes}"
        )

    return BudgetedTerminalResult(
        terminal=terminal,
        delete_count_pmf=delete_count_pmf / total,
        expected_deletes=expected_deletes,
        expected_deletes_by_level=deletes_by_level,
        p_any_delete=float(1.0 - delete_count_pmf[0] / total),
        p_full_budget=float(delete_count_pmf[budget] / total),
        expected_add_steps=expected_add_steps,
        expected_steps=expected_add_steps + expected_deletes + 1.0,
        expected_decoder_evals=expected_add_steps,
        expected_same_token_reselect=float(same_reselect_by_level.sum()),
        expected_same_token_reselect_by_level=same_reselect_by_level,
        normalization_error=norm_err,
        state_visitation_mass=visitation,
    )


# ---------------------------------------------------------------------------
# Level-reserved rollback budgets (augmented state (s, b_1..b_L))
# ---------------------------------------------------------------------------


@dataclass
class LevelBudgetedTerminalResult:
    """Exact quantities of the level-reserved budget policy for one context.

    Same semantics as :class:`BudgetedTerminalResult` but the remaining budget
    is a per-SID-level quota vector: deleting a state of depth ``l`` decrements
    only quota ``b_l``. ``p_full_budget`` is the probability of absorbing with
    EVERY quota exhausted.
    """

    terminal: np.ndarray
    level_budgets: tuple[int, ...]
    expected_deletes: float
    expected_deletes_by_level: np.ndarray
    p_any_delete: float
    p_full_budget: float
    expected_add_steps: float
    expected_steps: float
    expected_decoder_evals: float
    expected_same_token_reselect: float
    expected_same_token_reselect_by_level: np.ndarray
    normalization_error: float
    state_visitation_mass: np.ndarray


def exact_level_budgeted_terminal_distribution(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    rollback_prob: np.ndarray,
    level_budgets: tuple[int, ...] | list[int],
    *,
    atol: float = 1e-9,
) -> LevelBudgetedTerminalResult:
    """Exact DP over the augmented state (s, b_1..b_L) with per-level quotas.

    DELETE of a depth-``l`` state is allowed only while ``b_l > 0`` and
    decrements only ``b_l``; ADD keeps the quota vector and increases depth, so
    quota vectors ordered by decreasing total form a topological order.
    """
    from itertools import product as _product

    p, d = _check_policy_inputs(trie, edge_prob, rollback_prob)
    if d.ndim != 1:
        raise ValueError("level-budget DP takes a budget-independent 1-D rollback_prob")
    alloc = tuple(int(x) for x in level_budgets)
    L = trie.sid_len
    if len(alloc) != L or any(a < 0 for a in alloc):
        raise ValueError(f"level_budgets must be {L} non-negative quotas")

    n = trie.num_nodes
    term = trie.terminal_node_ids
    parent = trie.parent
    level = trie.level
    zero_vec = (0,) * L

    vectors = sorted(_product(*[range(a + 1) for a in alloc]), key=lambda v: (-sum(v), v))
    slices: dict[tuple[int, ...], np.ndarray] = {}
    start = np.zeros(n, dtype=np.float64)
    start[0] = 1.0
    slices[alloc] = start

    terminal = np.zeros(trie.num_items, dtype=np.float64)
    deletes_by_level = np.zeros(L + 1, dtype=np.float64)
    same_reselect_by_level = np.zeros(L + 1, dtype=np.float64)
    expected_add_steps = 0.0
    visitation = np.zeros(n, dtype=np.float64)
    absorbed_full_budget = 0.0
    absorbed_no_delete = 0.0

    for vec in vectors:
        reach = slices.pop(vec, None)
        if reach is None:
            continue
        allowed = np.zeros(n, dtype=bool)
        for lvl in range(1, L + 1):
            if vec[lvl - 1] > 0:
                allowed |= level == lvl
        d_eff = np.where(allowed, d, 0.0)
        for depth in range(1, L + 1):
            ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
            par = parent[ids]
            reach[ids] += reach[par] * (1.0 - d_eff[par]) * p[ids]
        visitation += reach
        keep = reach * (1.0 - d_eff)
        delete_out = reach * d_eff

        absorbed = float(keep[term].sum())
        terminal[trie.item_row[term]] += keep[term]
        if vec == alloc:
            absorbed_no_delete = absorbed
        if vec == zero_vec:
            absorbed_full_budget += absorbed
        expected_add_steps += float(keep[trie.internal_node_ids].sum())

        for lvl in range(1, L + 1):
            if vec[lvl - 1] == 0:
                continue
            ids = np.flatnonzero(level == lvl)
            m = delete_out[ids]
            total = float(m.sum())
            if total == 0.0:
                continue
            deletes_by_level[lvl] += total
            tgt = tuple(vec[k] - 1 if k == lvl - 1 else vec[k] for k in range(L))
            # same-token reselection: next ADD at the parent under the target
            # quota vector (parent deletable only if its own level quota > 0)
            par = parent[ids]
            if lvl - 1 >= 1 and tgt[lvl - 2] > 0:
                d_par = d[par]
            else:
                d_par = np.zeros(len(ids), dtype=np.float64)
            same_reselect_by_level[lvl] += float(np.sum(m * (1.0 - d_par) * p[ids]))
            arr = slices.get(tgt)
            if arr is None:
                arr = np.zeros(n, dtype=np.float64)
                slices[tgt] = arr
            np.add.at(arr, par, m)

    if slices:
        raise AssertionError(f"unprocessed budget slices remain: {sorted(slices)}")
    total = float(terminal.sum())
    norm_err = abs(total - 1.0)
    if norm_err > atol:
        raise AssertionError(
            f"level-budgeted terminal distribution sums to {total} (err {norm_err:.3e})")
    terminal = terminal / total
    expected_deletes = float(deletes_by_level.sum())

    return LevelBudgetedTerminalResult(
        terminal=terminal,
        level_budgets=alloc,
        expected_deletes=expected_deletes,
        expected_deletes_by_level=deletes_by_level,
        p_any_delete=float(1.0 - absorbed_no_delete / total),
        p_full_budget=float(absorbed_full_budget / total),
        expected_add_steps=expected_add_steps,
        expected_steps=expected_add_steps + expected_deletes + 1.0,
        expected_decoder_evals=expected_add_steps,
        expected_same_token_reselect=float(same_reselect_by_level.sum()),
        expected_same_token_reselect_by_level=same_reselect_by_level,
        normalization_error=norm_err,
        state_visitation_mass=visitation,
    )


# ---------------------------------------------------------------------------
# Rollback transforms
# ---------------------------------------------------------------------------


def scaled_rollback_prob(d_star: np.ndarray, lam: float) -> np.ndarray:
    """d_lambda(c, s) = sigmoid(logit(d_star(c, s)) - lambda); keeps d=0 fixed."""
    d_star = np.asarray(d_star, dtype=np.float64)
    if not ((d_star >= 0) & (d_star < 1)).all():
        raise ValueError("d_star must be in [0, 1)")
    out = np.zeros_like(d_star)
    nz = d_star > 0
    logit = np.log(d_star[nz]) - np.log1p(-d_star[nz])
    out[nz] = 1.0 / (1.0 + np.exp(-(logit - lam)))
    return out


def level_masked_rollback(
    trie: SemanticIDTrie, rollback_prob: np.ndarray, allowed_levels: set[int] | list[int]
) -> np.ndarray:
    """Zero the rollback probability at every SID level not in ``allowed_levels``.

    Levels are the depth of the state being popped (1..sid_len); the collision
    suffix is level ``sid_len``.
    """
    allowed = set(int(x) for x in allowed_levels)
    bad = allowed - set(range(1, trie.sid_len + 1))
    if bad:
        raise ValueError(f"invalid SID levels {sorted(bad)}")
    out = np.asarray(rollback_prob, dtype=np.float64).copy()
    mask = ~np.isin(trie.level, list(allowed))
    out[mask] = 0.0
    out[0] = 0.0
    return out


# ---------------------------------------------------------------------------
# Budgeted Monte Carlo sampler (with optional no-immediate-repeat mask)
# ---------------------------------------------------------------------------


@dataclass
class BudgetedSampleResult:
    """Vectorized budgeted rollout batch (every trajectory terminates).

    terminal_row:  [n_traj] catalog row of the accepted item.
    n_steps:       [n_traj] transitions (ADD + DELETE + STOP).
    n_deletes:     [n_traj] DELETE transitions (<= budget, asserted).
    delete_level_counts: [sid_len+1] total DELETEs per deleted-state level.
    n_same_token_reselect: total DELETE events whose immediate next action
                   re-added the deleted token.
    n_revisions:   DELETE events whose immediate next action was an ADD at the
                   same parent (same or different token).
    n_revisions_improved: revisions whose new child has a strictly larger
                   ``node_value`` (e.g. reward mass) than the deleted child;
                   -1 when the sampler was built without ``node_value``.
    n_mask_fallbacks: no-repeat ADDs where the mask left no valid child and the
                   sampler fell back to the unmasked proposal.
    """

    terminal_row: np.ndarray
    n_steps: np.ndarray
    n_deletes: np.ndarray
    delete_level_counts: np.ndarray
    n_same_token_reselect: int
    n_revisions: int
    n_revisions_improved: int
    n_mask_fallbacks: int


class BudgetedCyclicSampler:
    """Monte Carlo sampler for the hard-budget cyclic policy.

    With ``no_immediate_repeat=True``, after a DELETE of child ``v`` from
    parent ``s`` the very next ADD from ``s`` samples from
    ``p_retry(w) \\propto p_theta(w | c, s) [w != v]`` (one retry only). If the
    mask empties the support (single-child parent) the sampler falls back to
    the unmasked proposal and counts the event.
    """

    def __init__(
        self,
        trie: SemanticIDTrie,
        edge_prob: np.ndarray,
        rollback_prob: np.ndarray,
        *,
        budget: int | None = None,
        level_budgets: tuple[int, ...] | list[int] | None = None,
        no_immediate_repeat: bool = False,
        node_value: np.ndarray | None = None,
    ):
        p, d = _check_policy_inputs(trie, edge_prob, rollback_prob)
        if (budget is None) == (level_budgets is None):
            raise ValueError("pass exactly one of budget / level_budgets")
        if level_budgets is not None:
            if d.ndim != 1:
                raise ValueError("level_budgets requires a 1-D rollback_prob")
            self.level_budgets = tuple(int(x) for x in level_budgets)
            if len(self.level_budgets) != trie.sid_len or any(a < 0 for a in self.level_budgets):
                raise ValueError(f"level_budgets must be {trie.sid_len} non-negative quotas")
            budget = sum(self.level_budgets)
        else:
            self.level_budgets = None
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self.trie = trie
        self.edge_prob = p
        self.budget = int(budget)
        # d_rows[b] = rollback probabilities while b DELETEs remain (row 0 = 0)
        self.d_rows = _rollback_rows(trie, d, self.budget)
        self.no_immediate_repeat = bool(no_immediate_repeat)
        # optional per-node value (e.g. reward mass) for revision-precision stats
        self.node_value = None if node_value is None else np.asarray(node_value, dtype=np.float64)
        self._edge_cdf = np.cumsum(p)

    def _sample_children(
        self, nodes: np.ndarray, u: np.ndarray, forbidden: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Sample one child per node; ``forbidden`` is a child node id or -1."""
        trie = self.trie
        cdf = self._edge_cdf
        cs = trie.child_start[nodes]
        cc = trie.child_count[nodes]
        lo = cdf[cs - 1]  # cs >= 1 always: the root is nobody's child
        total = cdf[cs + cc - 1] - lo

        has_forb = forbidden >= 0
        n_fallback = 0
        if has_forb.any():
            fb = np.maximum(forbidden, 1)
            p_forb = np.where(has_forb, self.edge_prob[fb], 0.0)
            forb_lo = np.where(has_forb, cdf[fb - 1] - lo, np.inf)
            avail = total - p_forb
            # Degenerate mask (single valid child == forbidden): fall back.
            deg = has_forb & (avail <= total * 1e-12)
            n_fallback = int(deg.sum())
            p_forb = np.where(deg, 0.0, p_forb)
            forb_lo = np.where(deg, np.inf, forb_lo)
            avail = np.where(deg, total, avail)
            t = u * avail
            t = t + np.where(t >= forb_lo, p_forb, 0.0)
        else:
            t = u * total
        target = lo + np.minimum(t, np.nextafter(total, 0))
        idx = np.searchsorted(cdf, target, side="right")
        idx = np.clip(idx, cs, cs + cc - 1)
        # fp-boundary guard: never return the forbidden child of a live mask.
        bad = has_forb & (idx == forbidden) & (self.trie.child_count[nodes] > 1)
        if bad.any():
            up_ok = idx[bad] + 1 < (cs + cc)[bad]
            idx[bad] = np.where(up_ok, idx[bad] + 1, idx[bad] - 1)
        return idx, n_fallback

    def sample(self, n_traj: int, *, seed: int) -> BudgetedSampleResult:
        trie, B = self.trie, self.budget
        rng = np.random.default_rng(seed)

        state = np.zeros(n_traj, dtype=np.int64)
        budget_left = np.full(n_traj, B, dtype=np.int64)
        if self.level_budgets is not None:
            quota = np.tile(np.asarray(self.level_budgets, dtype=np.int64), (n_traj, 1))
        else:
            quota = None
        active = np.ones(n_traj, dtype=bool)
        n_steps = np.zeros(n_traj, dtype=np.int64)
        n_deletes = np.zeros(n_traj, dtype=np.int64)
        terminal_row = np.full(n_traj, -1, dtype=np.int64)
        delete_level_counts = np.zeros(trie.sid_len + 1, dtype=np.int64)

        # Pending mask/bookkeeping after a DELETE: the parent we popped to and
        # the deleted child node; consumed by the next action of the trajectory.
        pend_parent = np.full(n_traj, -1, dtype=np.int64)
        pend_child = np.full(n_traj, -1, dtype=np.int64)
        n_same = 0
        n_rev = 0
        n_rev_improved = 0
        n_fallback = 0

        max_steps = (trie.sid_len + 1) + 2 * B
        for _ in range(max_steps):
            if not active.any():
                break
            ids = np.flatnonzero(active)
            cur = state[ids]
            is_terminal = trie.level[cur] == trie.sid_len
            if quota is not None:
                lvl = trie.level[cur]
                can_delete = (lvl >= 1) & (
                    quota[ids, np.maximum(lvl, 1) - 1] > 0
                )
            else:
                can_delete = budget_left[ids] > 0
            d_cur = self.d_rows[np.minimum(budget_left[ids], B), cur]
            do_delete = (rng.random(len(ids)) < d_cur) & can_delete

            stop_mask = is_terminal & ~do_delete
            if stop_mask.any():
                sel = ids[stop_mask]
                terminal_row[sel] = trie.item_row[state[sel]]
                active[sel] = False

            if do_delete.any():
                sel = ids[do_delete]
                child = state[sel]
                np.add.at(delete_level_counts, trie.level[child], 1)
                if quota is not None:
                    quota[sel, trie.level[child] - 1] -= 1
                state[sel] = trie.parent[child]
                budget_left[sel] -= 1
                n_deletes[sel] += 1
                # a DELETE consumes any pending mask and opens a new one
                pend_parent[sel] = state[sel]
                pend_child[sel] = child

            add_mask = ~is_terminal & ~do_delete
            if add_mask.any():
                sel = ids[add_mask]
                pending = (pend_child[sel] >= 0) & (state[sel] == pend_parent[sel])
                forb = np.where(
                    pending & self.no_immediate_repeat, pend_child[sel], -1
                )
                child, fb = self._sample_children(
                    state[sel], rng.random(len(sel)), forb
                )
                n_fallback += fb
                if pending.any():
                    n_rev += int(pending.sum())
                    n_same += int(
                        (trie.token[child[pending]] == trie.token[pend_child[sel][pending]]).sum()
                    )
                    if self.node_value is not None:
                        n_rev_improved += int(
                            (self.node_value[child[pending]]
                             > self.node_value[pend_child[sel][pending]]).sum()
                        )
                pend_parent[sel] = -1
                pend_child[sel] = -1
                state[sel] = child

            n_steps[ids] += 1

        if active.any():
            raise AssertionError("budgeted trajectories must terminate within 5 + 2B steps")
        if int(n_deletes.max(initial=0)) > B:
            raise AssertionError("N_DELETE exceeded the hard budget")
        if int(n_steps.max(initial=0)) > max_steps:
            raise AssertionError("trajectory length exceeded 5 + 2B")

        return BudgetedSampleResult(
            terminal_row=terminal_row,
            n_steps=n_steps,
            n_deletes=n_deletes,
            delete_level_counts=delete_level_counts,
            n_same_token_reselect=n_same,
            n_revisions=n_rev,
            n_revisions_improved=(n_rev_improved if self.node_value is not None else -1),
            n_mask_fallbacks=n_fallback,
        )


# ---------------------------------------------------------------------------
# Non-cyclic baselines on the same trie tables
# ---------------------------------------------------------------------------


def geometric_guidance_edge_prob(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    target_q: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Direct reward-guided proposal P_alpha(v | c, s) ∝ p^(1-alpha) q_R^alpha.

    ``alpha = 0`` returns ``edge_prob``; ``alpha = 1`` returns ``target_q``
    (the exact append-only reward-conditional sampler).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    p = np.asarray(edge_prob, dtype=np.float64)
    q = np.asarray(target_q, dtype=np.float64)
    with np.errstate(divide="ignore"):
        logw = (1.0 - alpha) * np.log(np.where(p > 0, p, 1.0)) + alpha * np.log(
            np.where(q > 0, q, 1.0)
        )
    w = np.exp(logw)
    w[(p <= 0) & (alpha < 1.0)] = 0.0
    w[(q <= 0) & (alpha > 0.0)] = 0.0
    out = np.zeros_like(w)
    out[0] = 1.0
    internal = trie.internal_node_ids
    sums = np.add.reduceat(w, trie.child_start[internal])
    if not (sums > 0).all():
        raise ValueError("guided proposal has empty support at some internal node")
    # normalize each child block by its parent's sum
    denom = np.empty_like(w)
    for depth in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        denom[ids] = sums[trie.parent[ids]]
    out[1:] = w[1:] / denom[1:]
    return out


def temperature_edge_prob(
    trie: SemanticIDTrie, log_edge_prob: np.ndarray, temperature: float
) -> np.ndarray:
    """Constrained decoder-temperature proposal: p_tau ∝ exp(log p / tau).

    Equivalent to applying the temperature to the child logits before the
    constrained softmax (constant logit shifts cancel).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(log_edge_prob, dtype=np.float64) / float(temperature)
    internal = trie.internal_node_ids
    starts = trie.child_start[internal]
    zmax = np.maximum.reduceat(z, starts)
    shifted = np.empty_like(z)
    shifted[0] = 0.0
    for depth in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        shifted[ids] = z[ids] - zmax[trie.parent[ids]]
    w = np.exp(shifted)
    w[0] = 1.0
    sums = np.add.reduceat(w, starts)
    out = np.zeros_like(w)
    out[0] = 1.0
    for depth in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        out[ids] = w[ids] / sums[trie.parent[ids]]
    return out


def best_of_n_exact_distribution(
    p_append: np.ndarray, rewards: np.ndarray, n: int
) -> np.ndarray:
    """Exact distribution of the max-reward item among n iid append-only draws.

    Sort items by reward descending; the selected item has the highest reward
    among the n draws, so P(selected in tie-group g) = C(>=g)^n - C(>g)^n with
    C the cumulative draw probability. Within an exact-tie group mass is split
    proportionally to the draw probability (exact under a uniform random
    tie-break among tied drawn items when they have equal draw probability;
    a stated approximation otherwise — ties are measure-zero for continuous
    SASRec scores).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    p = np.asarray(p_append, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64)
    if p.shape != r.shape:
        raise ValueError("p_append and rewards must have the same shape")
    order = np.argsort(-r, kind="stable")
    p_sorted = p[order]
    r_sorted = r[order]
    # tie-group boundaries
    new_group = np.ones(len(r), dtype=bool)
    new_group[1:] = r_sorted[1:] != r_sorted[:-1]
    group_id = np.cumsum(new_group) - 1
    group_p = np.bincount(group_id, weights=p_sorted)
    c_incl = np.cumsum(group_p)
    c_excl = c_incl - group_p
    # max-reward draw lands in group g iff no draw hits a strictly better group
    # but at least one hits g: (1 - C_<g)^n - (1 - C_<=g)^n.
    group_mass = np.power(1.0 - c_excl, n) - np.power(1.0 - c_incl, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        within = np.where(group_p[group_id] > 0, p_sorted / group_p[group_id], 0.0)
    out_sorted = group_mass[group_id] * within
    out = np.zeros_like(p)
    out[order] = out_sorted
    total = out.sum()
    # 1 - total is the probability that all n draws fall on zero-probability
    # items — impossible; total deviates from 1 only by fp error.
    if not np.isclose(total, 1.0, atol=1e-8):
        raise AssertionError(f"best-of-n distribution sums to {total}")
    return out / total


def trie_beam_search(
    trie: SemanticIDTrie, log_edge_prob: np.ndarray, beam_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic constrained beam search over the SID trie.

    Returns (catalog rows, cumulative log-probabilities), sorted by descending
    log-probability, at most ``beam_size`` items. Ties are broken by node id
    (deterministic).
    """
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    logp = np.asarray(log_edge_prob, dtype=np.float64)
    nodes = np.array([0], dtype=np.int64)
    scores = np.array([0.0], dtype=np.float64)
    for _ in range(trie.sid_len):
        counts = trie.child_count[nodes]
        parents = np.repeat(np.arange(len(nodes)), counts)
        offsets = np.concatenate([np.arange(c) for c in counts]) if len(nodes) else np.empty(0, np.int64)
        children = trie.child_start[nodes][parents] + offsets
        cand_scores = scores[parents] + logp[children]
        if len(children) > beam_size:
            keep = np.argpartition(-cand_scores, beam_size - 1)[:beam_size]
        else:
            keep = np.arange(len(children))
        order = keep[np.lexsort((children[keep], -cand_scores[keep]))]
        nodes = children[order]
        scores = cand_scores[order]
    return trie.item_row[nodes], scores


# ---------------------------------------------------------------------------
# Optional phase 2: differentiable budget-optimal rollback (validation only)
# ---------------------------------------------------------------------------


@dataclass
class BudgetOptimalResult:
    rollback_prob: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)


def optimize_budgeted_rollback(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    target_p: np.ndarray,
    *,
    budget: int,
    expected_deletes_cap: float | None = None,
    init_rollback: np.ndarray | None = None,
    steps: int = 300,
    lr: float = 0.05,
    penalty: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
) -> BudgetOptimalResult:
    """Minimize D_JS(P_T^d, p_R) over per-node rollback logits under the exact
    finite-budget DP, with a soft penalty on E[N_DELETE] when
    ``expected_deletes_cap`` is given. Validation-only oracle upper bound for
    the frozen-TIGER + rollback parameterization (spec §9.3).
    """
    import torch

    torch.manual_seed(seed)
    dev = torch.device(device)
    p = torch.as_tensor(np.asarray(edge_prob, dtype=np.float64), device=dev)
    pr = torch.as_tensor(np.asarray(target_p, dtype=np.float64), device=dev)
    pr = pr / pr.sum()
    parent = torch.as_tensor(trie.parent, device=dev)
    item_row = torch.as_tensor(trie.item_row[trie.terminal_node_ids], device=dev)
    term_lo = int(trie.level_offsets[trie.sid_len])
    level_slices = [
        (int(trie.level_offsets[k]), int(trie.level_offsets[k + 1]))
        for k in range(trie.sid_len + 1)
    ]

    if init_rollback is not None:
        d0 = np.clip(np.asarray(init_rollback, dtype=np.float64), 1e-6, 1 - 1e-6)
        theta0 = np.log(d0) - np.log1p(-d0)
    else:
        theta0 = np.full(trie.num_nodes, -2.0, dtype=np.float64)
    theta = torch.tensor(theta0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)

    def dp(d: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        terminal = torch.zeros(trie.num_items, dtype=torch.float64, device=dev)
        exp_del = torch.zeros((), dtype=torch.float64, device=dev)
        mass_in = torch.zeros(trie.num_nodes, dtype=torch.float64, device=dev)
        mass_in = mass_in.index_put((torch.tensor([0], device=dev),),
                                    torch.tensor([1.0], dtype=torch.float64, device=dev))
        for b in range(budget, -1, -1):
            d_eff = d if b > 0 else torch.zeros_like(d)
            reach = mass_in
            for lo, hi in level_slices[1:]:
                ids = torch.arange(lo, hi, device=dev)
                par = parent[ids]
                add = reach[par] * (1.0 - d_eff[par]) * p[lo:hi]
                reach = reach.index_add(0, ids, add)
            keep = reach * (1.0 - d_eff)
            terminal = terminal.index_add(0, item_row, keep[term_lo:])
            if b > 0:
                delete_out = reach * d_eff
                exp_del = exp_del + delete_out[1:].sum()
                nxt = torch.zeros_like(mass_in)
                nxt = nxt.index_add(0, parent[1:], delete_out[1:])
                mass_in = nxt
        return terminal, exp_del

    history: list[dict[str, float]] = []
    for step in range(steps):
        opt.zero_grad()
        d = torch.sigmoid(theta)
        d = d * (1.0 - 1e-9)
        d = torch.cat([torch.zeros(1, dtype=torch.float64, device=dev), d[1:]])
        terminal, exp_del = dp(d)
        m = 0.5 * (terminal + pr)
        eps = 1e-30
        js = 0.5 * (terminal * (torch.log(terminal + eps) - torch.log(m + eps))).sum() + \
             0.5 * (pr * (torch.log(pr + eps) - torch.log(m + eps))).sum()
        loss = js
        if expected_deletes_cap is not None:
            loss = loss + penalty * torch.relu(exp_del - expected_deletes_cap)
        loss.backward()
        opt.step()
        history.append(
            {"step": step, "js": float(js), "expected_deletes": float(exp_del),
             "loss": float(loss)}
        )
    with torch.no_grad():
        d = torch.sigmoid(theta) * (1.0 - 1e-9)
        d[0] = 0.0
    return BudgetOptimalResult(rollback_prob=d.cpu().numpy(), history=history)
