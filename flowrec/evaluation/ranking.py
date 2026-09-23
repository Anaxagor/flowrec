"""Shared ranking metrics for exact-distribution and Top-K evaluations.

Rank convention: competition ranking, ``rank = 1 + #{j : score_j > score_target}``
(ties do not hurt the target). All metrics are per-user; aggregate by averaging
over users. Coverage is a set metric over the union of users' Top-K lists.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rank_of_target",
    "ranks_of_targets",
    "metrics_from_rank",
    "aggregate_rank_metrics",
    "coverage_at_k",
    "topk_rows",
]


def rank_of_target(scores: np.ndarray, target_index: int) -> int:
    """Competition rank of ``target_index`` under descending ``scores``."""
    scores = np.asarray(scores)
    return int(np.sum(scores > scores[target_index])) + 1


def ranks_of_targets(scores: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """Row-wise :func:`rank_of_target` for a [n_users, n_items] score matrix."""
    scores = np.asarray(scores)
    tgt = scores[np.arange(scores.shape[0]), target_indices]
    return (scores > tgt[:, None]).sum(axis=1) + 1


def metrics_from_rank(rank: int, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    """Recall@K (hit), NDCG@K and MRR for a single held-out target."""
    out: dict[str, float] = {}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        out[f"recall@{k}"] = hit
        out[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1.0)) if rank <= k else 0.0
    out["mrr"] = 1.0 / rank
    out["target_rank"] = float(rank)
    return out


def aggregate_rank_metrics(
    ranks: np.ndarray, ks: tuple[int, ...] = (5, 10, 20)
) -> dict[str, float]:
    """Mean Recall@K / NDCG@K / MRR over users from per-user target ranks."""
    ranks = np.asarray(ranks, dtype=np.float64)
    out: dict[str, float] = {"n_users": int(len(ranks))}
    for k in ks:
        hits = ranks <= k
        out[f"recall@{k}"] = float(hits.mean())
        out[f"ndcg@{k}"] = float(np.where(hits, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
    out["mrr"] = float((1.0 / ranks).mean())
    out["median_target_rank"] = float(np.median(ranks))
    return out


def topk_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores, descending, deterministic tie-break by index."""
    scores = np.asarray(scores)
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


def coverage_at_k(per_user_topk: list[np.ndarray], num_items: int) -> float:
    """|union of users' Top-K lists| / num_items."""
    if not per_user_topk:
        return 0.0
    return float(len(np.unique(np.concatenate(per_user_topk))) / num_items)
