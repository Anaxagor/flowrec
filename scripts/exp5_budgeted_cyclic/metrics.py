"""Per-user prediction rows and metric helpers (spec §8, §28).

Two row kinds, never mixed (spec §8):
  - ``full_catalog``: exact catalog-wide ranking -> exact ``full_catalog_rank``.
  - ``candidate``: candidate-limited methods report ``candidate_hit``,
    ``rank_within_candidates`` (nullable), and ``censored_rank = M + 1`` when
    the target is absent; ``full_catalog_rank`` stays NA.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flowrec.evaluation.ranking import rank_of_target, topk_rows

KS = (5, 10, 20)


def _rank_metrics(rank: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in KS:
        hit = rank <= k
        out[f"recall@{k}"] = float(hit)
        out[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 1.0)) if hit else 0.0
    out["mrr"] = 1.0 / rank
    return out


def full_catalog_prediction(scores: np.ndarray, target_row: int) -> dict[str, Any]:
    rank = rank_of_target(scores, target_row)
    row: dict[str, Any] = {"eval_kind": "full_catalog"}
    row.update(_rank_metrics(rank))
    row["full_catalog_rank"] = int(rank)
    row["candidate_set_size"] = None
    row["candidate_hit"] = None
    row["rank_within_candidates"] = None
    row["censored_rank"] = None
    row["top20"] = ",".join(str(int(r)) for r in topk_rows(scores, 20))
    return row


def candidate_prediction(
    ranked_rows: np.ndarray, target_row: int, *, candidate_set_size: int,
) -> dict[str, Any]:
    pos = np.flatnonzero(np.asarray(ranked_rows) == target_row)
    hit = bool(len(pos))
    rank = int(pos[0]) + 1 if hit else None
    censored = rank if hit else int(candidate_set_size) + 1
    row: dict[str, Any] = {"eval_kind": "candidate"}
    row.update(_rank_metrics(censored))
    row["full_catalog_rank"] = None      # never fabricate a catalog-scale rank
    row["candidate_set_size"] = int(candidate_set_size)
    row["candidate_hit"] = hit
    row["rank_within_candidates"] = rank
    row["censored_rank"] = int(censored)
    row["top20"] = ",".join(str(int(r)) for r in np.asarray(ranked_rows)[:20])
    return row


def alignment_metrics(
    dist: np.ndarray, log_p_target: np.ndarray, raw_scores: np.ndarray,
) -> dict[str, float]:
    """TV / JS / KL(dist || target) plus expected raw score and target log
    weight. KL is stable because the target log-probabilities are finite
    everywhere (log-space construction)."""
    p_t = np.exp(log_p_target)
    tv = 0.5 * float(np.abs(dist - p_t).sum())
    m = 0.5 * (dist + p_t)
    nz_d = dist > 0
    nz_t = p_t > 0
    js = 0.5 * float(np.sum(dist[nz_d] * (np.log(dist[nz_d]) - np.log(m[nz_d])))) \
        + 0.5 * float(np.sum(p_t[nz_t] * (log_p_target[nz_t] - np.log(m[nz_t]))))
    kl = float(np.sum(dist[nz_d] * (np.log(dist[nz_d]) - log_p_target[nz_d])))
    return {
        "tv_to_target": tv, "js_to_target": js, "kl_to_target": kl,
        "expected_raw_score": float((dist * raw_scores).sum()),
        "expected_target_logw": float((dist * log_p_target).sum()),
    }


def dp_behavior_fields(res: Any, sid_len: int) -> dict[str, Any]:
    out = {
        "expected_steps": float(res.expected_steps),
        "expected_deletes": float(res.expected_deletes),
        "p_any_delete": float(res.p_any_delete),
        "p_full_budget": float(res.p_full_budget),
        "same_token_reselect": float(res.expected_same_token_reselect),
        "online_expected_tiger_calls": float(res.expected_add_steps),
    }
    for lvl in range(1, sid_len + 1):
        out[f"deletes_L{lvl}"] = float(res.expected_deletes_by_level[lvl])
        out[f"reselect_L{lvl}"] = float(res.expected_same_token_reselect_by_level[lvl])
    return out


def static_behavior_fields(sid_len: int, *, steps: float | None = None) -> dict[str, Any]:
    out = {
        "expected_steps": float(steps if steps is not None else sid_len + 1),
        "expected_deletes": 0.0, "p_any_delete": 0.0, "p_full_budget": 0.0,
        "same_token_reselect": 0.0,
        "online_expected_tiger_calls": float(sid_len),
    }
    for lvl in range(1, sid_len + 1):
        out[f"deletes_L{lvl}"] = 0.0
        out[f"reselect_L{lvl}"] = 0.0
    return out


def gini_concentration(counts: np.ndarray) -> float:
    """Gini coefficient of the recommended-item count distribution."""
    c = np.sort(np.asarray(counts, dtype=np.float64))
    if c.sum() == 0:
        return 0.0
    n = len(c)
    cum = np.cumsum(c)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def parse_top20(s: str) -> np.ndarray:
    if not isinstance(s, str) or not s:
        return np.empty(0, dtype=np.int64)
    return np.asarray([int(x) for x in s.split(",")], dtype=np.int64)
