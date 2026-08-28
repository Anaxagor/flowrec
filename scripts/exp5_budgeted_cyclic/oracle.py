"""Log-space global flow oracle (spec §11.2) with scale-aware diagnostics.

Construction (per user context and target):

    log C(s)   = max_v [ log q(v|s) - log p(v|s) ]        (internal nodes)
    slack(v)   = log C(parent(v)) + log p(v) - log q(v)   (>= 0 by definition)
    kappa(v)   = M(v) * expm1(slack(v))                   (reverse DELETE flow)
    d*(v)      = kappa / state_flow:
                   leaf:      1 - exp(-slack)
                   internal:  expm1(slack) / (C(v) + expm1(slack))

The slack-based form makes the net-flow identity F(s->v) - kappa(v) = M(v)
exact BY CONSTRUCTION; the residual diagnostics quantify the remaining
conservation identity G(s) = M(s) + sum_children kappa on scale-aware
yardsticks (spec §11.3): |delta log| IS the relative error of linear flows, and
absolute errors are reported in units of Z.

The oracle distinguishes four numerical regimes instead of silently clipping:
true support violations, floating-point underflow of cached proposal
log-probabilities, mathematically saturated rollback (leaf slack so large that
1 - d* underflows float64), and explicit implementation clipping applied only
when preparing d for the DP/sampler (which require d < 1 strictly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flowrec.trajectories.nonacyclic import SemanticIDTrie

from .rewards import segment_logsumexp

LOG_P_UNDERFLOW = -87.0        # float32 softmax cannot represent below ~exp(-87)
SATURATION_EPS = 1e-15         # 1 - d* below this = mathematically saturated
MASS_THRESHOLD = 1e-12         # spec 11.3 edge-mass threshold (documented)
DENOM_FLOOR = 1e-300           # spec 11.3 denominator floor (documented)


@dataclass
class LogFlowOracle:
    log_C: np.ndarray          # [num_nodes]; internal nodes; 0 elsewhere
    slack: np.ndarray          # [num_nodes]; per non-root node; 0 at root
    d_star: np.ndarray         # [num_nodes] float64 in [0, 1]; may equal 1.0 (saturated)
    log_mass: np.ndarray
    log_z: float
    E_del_inf: float
    E_n_inf: float
    E_del_inf_by_level: np.ndarray   # [sid_len+1]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_log_flow_oracle(
    trie: SemanticIDTrie,
    log_edge_prob: np.ndarray,
    log_q: np.ndarray,
    log_mass: np.ndarray,
) -> LogFlowOracle:
    n = trie.num_nodes
    log_p = np.asarray(log_edge_prob, dtype=np.float64)
    log_q = np.asarray(log_q, dtype=np.float64)
    log_mass = np.asarray(log_mass, dtype=np.float64)
    internal = trie.internal_node_ids
    parent = trie.parent
    log_z = float(log_mass[0])

    # normalization guard: sum_children p == 1 in log space
    logsum_p = segment_logsumexp(log_p, trie.child_start[internal],
                                 trie.child_count[internal])
    max_p_norm_dev = float(np.abs(logsum_p).max())
    if max_p_norm_dev > 1e-9:
        raise ValueError(
            f"edge log-probs not normalized per node (max |logsumexp| "
            f"{max_p_norm_dev:.3e}); renormalize in float64 first")

    # support / underflow classification
    with np.errstate(invalid="ignore"):
        n_support_violation = int(np.sum(np.isneginf(log_p[1:]) & np.isfinite(log_q[1:])))
        n_p_underflow = int(np.sum((log_p[1:] < LOG_P_UNDERFLOW) & np.isfinite(log_q[1:])))
    if n_support_violation:
        raise ValueError(
            f"{n_support_violation} true support violations: target mass on "
            "edges with exactly zero proposal probability")

    ratio = np.zeros(n, dtype=np.float64)
    ratio[1:] = log_q[1:] - log_p[1:]
    log_C = np.zeros(n, dtype=np.float64)
    log_C[internal] = np.maximum.reduceat(ratio, trie.child_start[internal])
    min_log_C = float(log_C[internal].min())  # must be >= 0 (C >= 1)

    slack = np.zeros(n, dtype=np.float64)
    slack[1:] = log_C[parent[1:]] + log_p[1:] - log_q[1:]
    n_negative_slack = int((slack[1:] < 0).sum())
    max_negative_slack = float(-slack[1:].min()) if n_negative_slack else 0.0
    slack[slack < 0] = 0.0   # fp noise at the maximizing child; magnitude recorded

    # d* from slack (stable in every regime)
    d_star = np.zeros(n, dtype=np.float64)
    term = trie.terminal_node_ids
    d_star[term] = -np.expm1(-slack[term])          # 1 - exp(-slack)
    is_int = np.zeros(n, dtype=bool)
    is_int[internal] = True
    is_int[0] = False
    with np.errstate(divide="ignore", over="ignore"):
        em1 = np.expm1(np.minimum(slack, 700.0))
        pos = is_int & (slack > 0)
        log_em1 = np.zeros(n)
        log_em1[pos] = np.where(slack[pos] > 700.0, slack[pos], np.log(em1[pos]))
        d_star[pos] = np.exp(
            log_em1[pos] - np.logaddexp(log_C[pos], log_em1[pos]))
    d_star[0] = 0.0
    n_saturated = int(np.sum((1.0 - d_star) < SATURATION_EPS))

    # exact infinite-horizon costs in log space
    nz = slack[1:] > 0
    v_ids = np.arange(1, n)[nz]
    if len(v_ids):
        log_kappa = log_mass[v_ids] + np.where(
            slack[v_ids] > 700.0, slack[v_ids],
            np.log(np.expm1(np.minimum(slack[v_ids], 700.0))))
        m = float(log_kappa.max())
        E_del = float(np.exp(m - log_z) * np.exp(log_kappa - m).sum())
        by_level = np.zeros(trie.sid_len + 1)
        np.add.at(by_level, trie.level[v_ids], np.exp(log_kappa - log_z))
    else:
        E_del, by_level = 0.0, np.zeros(trie.sid_len + 1)
        log_kappa = np.empty(0)

    # state flow: internal M*(C + expm1(slack)); leaf M*exp(slack); root M*C
    log_flow = np.empty(n, dtype=np.float64)
    with np.errstate(divide="ignore"):
        le_full = np.where(slack > 0,
                           np.where(slack > 700.0, slack,
                                    np.log(np.maximum(np.expm1(np.minimum(slack, 700.0)), 1e-300))),
                           -np.inf)
    log_flow[is_int] = log_mass[is_int] + np.logaddexp(log_C[is_int], le_full[is_int])
    log_flow[term] = log_mass[term] + slack[term]
    log_flow[0] = log_z + log_C[0]
    mf = float(log_flow.max())
    E_n = float(np.exp(mf - log_z) * np.exp(log_flow - mf).sum())

    # conservation identity G(s) = M(s) + sum_children kappa (scale-aware, 11.3)
    lhs = log_mass[internal] + log_C[internal]
    log_kappa_full = np.full(n, -np.inf)
    if len(v_ids):
        log_kappa_full[v_ids] = log_kappa
    kappa_children = segment_logsumexp(
        log_kappa_full, trie.child_start[internal], trie.child_count[internal])
    rhs = np.logaddexp(log_mass[internal], kappa_children)
    dlog = np.abs(lhs - rhs)
    node_mass = np.exp(np.minimum(log_mass[internal] - log_z, 0.0))
    heavy = node_mass > MASS_THRESHOLD
    lin_err = np.exp(np.maximum(lhs, rhs) - log_z) * -np.expm1(-dlog)
    diagnostics = {
        "n_support_violation": n_support_violation,
        "n_proposal_logp_underflow": n_p_underflow,
        "n_saturated_rollback": n_saturated,
        "n_negative_slack_clipped": n_negative_slack,
        "max_negative_slack": max_negative_slack,
        "min_log_C": min_log_C,
        "max_p_norm_deviation": max_p_norm_dev,
        "max_abs_error": float(lin_err.max()),
        "max_relative_error_with_denominator_floor": float(
            (lin_err / np.maximum(node_mass, DENOM_FLOOR)).max()),
        "max_relative_error_on_edges_with_target_mass_above_threshold": float(
            dlog[heavy].max()) if heavy.any() else 0.0,
        "mass_weighted_relative_error": float(
            (node_mass * dlog).sum() / max(node_mass.sum(), DENOM_FLOOR)),
        "mass_threshold": MASS_THRESHOLD,
        "denominator_floor": DENOM_FLOOR,
    }
    return LogFlowOracle(
        log_C=log_C, slack=slack, d_star=d_star, log_mass=log_mass, log_z=log_z,
        E_del_inf=E_del, E_n_inf=E_n + 0.0, E_del_inf_by_level=by_level,
        diagnostics=diagnostics)


def rollback_for_dp(
    oracle: LogFlowOracle, *, clip: float = 1.0 - 1e-12
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare d for the DP/sampler (which require d < 1 strictly).

    Saturated states (d* == 1 within float64) are clipped EXPLICITLY here; the
    count and the maximum probability displaced are recorded, never silent.
    """
    d = oracle.d_star.copy()
    over = d > clip
    n_clipped = int(over.sum())
    max_effect = float((d[over] - clip).max()) if n_clipped else 0.0
    d[over] = clip
    return d, {"n_impl_clipped": n_clipped, "max_clip_probability_effect": max_effect,
               "clip_value": clip}
