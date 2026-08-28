"""Paired user-level bootstrap, Holm correction, coverage uncertainty (spec §14).

Terminology is exact: ``bootstrap_prob_improvement`` is the bootstrap
probability that the mean paired delta is positive — it is NOT a p-value. The
two-sided bootstrap p-value and its Holm-adjusted version are reported
separately, and Holm is computed only over the predeclared primary family.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def paired_bootstrap(
    deltas: np.ndarray, *, n_boot: int = 10_000, seed: int = 42,
) -> dict[str, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if n < 2:
        raise ValueError("paired bootstrap needs at least 2 users")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    prob_pos = float((means > 0).mean())
    prob_neg = float((means < 0).mean())
    # two-sided bootstrap p-value with add-one smoothing (never exactly 0)
    p_two = min(1.0, 2.0 * min(
        (np.sum(means <= 0) + 1) / (n_boot + 1),
        (np.sum(means >= 0) + 1) / (n_boot + 1)))
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "bootstrap_prob_improvement": prob_pos,
        "bootstrap_prob_degradation": prob_neg,
        "p_value_two_sided": float(p_two),
        "n_users": int(n),
        "n_boot": int(n_boot),
    }


def holm_adjust(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over the DECLARED primary comparisons only."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)  # enforce monotonicity
        out[name] = running
    return out


def average_over_seeds(
    df: pd.DataFrame, *, user_col: str = "user", seed_col: str = "seed",
    metric_cols: Iterable[str],
) -> pd.DataFrame:
    """Average per-user metrics across Monte Carlo seeds BEFORE the user-level
    bootstrap (documented procedure, spec §14.3)."""
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(user_col, as_index=False)[metric_cols].mean()


def coverage_bootstrap(
    per_user_topk: list[np.ndarray], num_items: int, *, k: int,
    n_boot: int = 2_000, seed: int = 42,
) -> dict[str, float]:
    """Coverage is a catalog-level set statistic, not a per-user mean: its
    uncertainty comes from resampling USERS and recomputing coverage."""
    lists = [np.asarray(t[:k], dtype=np.int64) for t in per_user_topk]
    n = len(lists)
    point = len(np.unique(np.concatenate(lists))) / num_items if n else 0.0
    if n < 2:
        return {"coverage": float(point), "ci_lo": float(point), "ci_hi": float(point)}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.integers(0, n, size=n)
        stats[b] = len(np.unique(np.concatenate([lists[i] for i in take]))) / num_items
    return {
        "coverage": float(point),
        "ci_lo": float(np.percentile(stats, 2.5)),
        "ci_hi": float(np.percentile(stats, 97.5)),
    }
