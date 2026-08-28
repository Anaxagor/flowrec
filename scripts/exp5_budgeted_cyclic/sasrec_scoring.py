"""SASRec full-catalog score cache (spec §18.1): raw scores, not rewards.

Shard arrays: ``raw_scores`` [n_users, num_items] float32 in SEMANTIC-ROW
order (== trie catalog-row order); the exact item-order hash is stored in the
shard meta and verified on read.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch

from .compute_counters import ComputeCounters


def item_order_hash(row_to_item: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(row_to_item, dtype=np.int64).tobytes())
    return h.hexdigest()


@torch.no_grad()
def score_user_batch(
    sasrec: Any, batch: dict[str, Any], row_to_item: np.ndarray, device: Any,
    counters: ComputeCounters | None = None,
) -> np.ndarray:
    """Full-catalog raw scores for a user batch -> [B, num_items] float32
    (semantic-row order)."""
    t0 = time.time()
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    raw = sasrec.score_items(moved).float().cpu().numpy()  # [B, item_vocab] atomic ids
    out = raw[:, row_to_item].astype(np.float32)
    if counters is not None:
        b = out.shape[0]
        counters.add(
            sasrec_user_forwards=b,
            sasrec_full_catalog_passes=b,
            sasrec_item_scores=b * out.shape[1],
            wall_time_precompute_s=time.time() - t0,
        )
    return out


def validate_batch_vs_single(
    sasrec: Any, dm: Any, split: str, indices: list[int], row_to_item: np.ndarray,
    device: Any, *, atol: float = 5e-2,
) -> float:
    """Spec §29: batched scoring must agree with single-user scoring.

    On GPU, batched vs single forward passes select different cuBLAS kernels,
    so raw logits (magnitude ~10-20) legitimately differ by a few 1e-3
    (observed ~1.3-2.1e-3 on the cluster). The guard targets item-ordering /
    id-space bugs, which produce errors of order the score RANGE — hence an
    absolute tolerance of 0.05, not machine precision.
    """
    from .common import single_user_batch, user_batches

    _, batch = next(user_batches(dm, split, indices, len(indices)))
    batched = score_user_batch(sasrec, batch, row_to_item, device)
    worst = 0.0
    for k, idx in enumerate(indices):
        single = score_user_batch(
            sasrec, single_user_batch(dm, split, idx), row_to_item, device)[0]
        worst = max(worst, float(np.abs(single - batched[k]).max()))
    if worst > atol:
        raise AssertionError(
            f"batched vs single SASRec scoring differs by {worst} > {atol}: "
            "likely an item-ordering or id-space bug, not kernel noise")
    return worst
