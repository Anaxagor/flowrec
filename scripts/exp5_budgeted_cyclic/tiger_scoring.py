"""TIGER prefix-score cache and exact/beam/derived evaluators (spec §5, §18.2).

Cache layout per shard (float32 storage, float64 aggregation on read):
  edge_logprob    [n_users, num_nodes]     constrained log p(token | prefix),
                                           indexed by CHILD node id, root = 0
  raw_valid_mass  [n_users, num_internal]  unconstrained softmax mass on the
                                           valid children of each internal node

The CSR trie layout (trie.py:csr_arrays) plus these flat vectors reconstruct:
exact item log-probability (path sum), valid-child conditionals at every
prefix, raw valid-token mass, entropy and top-token margin, and the exact
full-catalog ranking.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from flowrec.trajectories.nonacyclic import (
    SemanticIDTrie,
    compute_tiger_edge_distributions,
)

from .compute_counters import ComputeCounters
from .rewards import segment_logsumexp


def compute_user_edge_logprobs(
    tiger: Any, batch: dict[str, Any], trie: SemanticIDTrie, *, batch_size: int = 512,
    counters: ComputeCounters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One frozen-TIGER prefix sweep -> (edge_logprob[num_nodes] float32,
    raw_valid_mass[num_internal] float32)."""
    t0 = time.time()
    tables = compute_tiger_edge_distributions(
        tiger, batch, trie, batch_size=batch_size, skip_single_child=True)
    n_internal = int(trie.level_offsets[trie.sid_len])
    if counters is not None:
        n_eval = int(tables.evaluated.sum())
        counters.add(
            tiger_encoder_forwards=1,
            tiger_prefix_state_evals=n_eval,
            tiger_decoder_forward_batches=int(np.ceil(n_eval / batch_size)),
            tiger_token_logit_evals=float(trie.child_count[tables.evaluated].sum()),
            wall_time_precompute_s=time.time() - t0,
        )
    return (tables.log_edge_prob.astype(np.float32),
            tables.raw_valid_mass[:n_internal].astype(np.float32))


def renormalized_edge_logprob(trie: SemanticIDTrie, edge_logprob_f32: np.ndarray) -> np.ndarray:
    """float32 cache -> float64 log-probs exactly normalized per node."""
    lp = np.asarray(edge_logprob_f32, dtype=np.float64)
    internal = trie.internal_node_ids
    norm = segment_logsumexp(lp, trie.child_start[internal], trie.child_count[internal])
    if np.abs(norm).max() > 1e-3:
        raise AssertionError(
            f"cached edge log-probs off by {np.abs(norm).max():.2e}; corrupt cache?")
    lp[1:] = lp[1:] - np.repeat(norm, trie.child_count[internal])
    check = segment_logsumexp(lp, trie.child_start[internal], trie.child_count[internal])
    assert np.abs(check).max() < 1e-12
    lp[0] = 0.0
    return lp


def exact_item_logprobs(trie: SemanticIDTrie, edge_logprob: np.ndarray) -> np.ndarray:
    """log p_TIGER(x | c) = sum of constrained token log-probs along the SID.

    The analytical full-catalog reference ranking (``tiger_exact_full_catalog``),
    not a serving beam-search result.
    """
    log_reach = np.zeros(trie.num_nodes, dtype=np.float64)
    for depth in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        log_reach[ids] = log_reach[trie.parent[ids]] + edge_logprob[ids]
    out = np.empty(trie.num_items, dtype=np.float64)
    term = trie.terminal_node_ids
    out[trie.item_row[term]] = log_reach[term]
    return out


def node_entropy_and_margin(
    trie: SemanticIDTrie, edge_logprob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-internal-node predictive entropy (nats) and top1-top2 logprob margin."""
    internal = trie.internal_node_ids
    starts, counts = trie.child_start[internal], trie.child_count[internal]
    lp = edge_logprob
    p = np.exp(lp)
    ent_terms = np.where(np.isfinite(lp), -p * lp, 0.0)
    entropy = np.add.reduceat(ent_terms[starts[0]:], starts - starts[0])
    m1 = np.maximum.reduceat(lp[starts[0]:], starts - starts[0])
    masked = lp.copy()
    top_idx = np.repeat(m1, counts) == lp[starts[0]: starts[0] + counts.sum()]
    seg = masked[starts[0]: starts[0] + counts.sum()]
    seg[top_idx] = -np.inf
    m2 = np.maximum.reduceat(seg, starts - starts[0])
    margin = np.where(np.isfinite(m2), m1 - m2, 0.0)  # ties/singletons -> 0
    ent_full = np.zeros(trie.num_nodes)
    ent_full[internal] = entropy
    mar_full = np.zeros(trie.num_nodes)
    mar_full[internal] = margin
    return ent_full, mar_full


def validate_cache_against_model(
    tiger: Any, batch: dict[str, Any], trie: SemanticIDTrie,
    edge_logprob: np.ndarray, *, rows: np.ndarray, atol: float = 2e-2,
) -> dict[str, float]:
    """Spec §18.3: cached path sums must equal the model's own valid-action
    item log-probabilities for randomly chosen items.

    The tolerance covers float32 cache storage plus GPU kernel/reduction-order
    nondeterminism between the sweep's decoder batching and
    ``score_items_valid_actions`` (a few 1e-3 per token logit); a real
    convention bug (offsets, id spaces, ordering) shows up as errors of order
    the log-prob range itself.
    """
    import torch

    device = next(tiger.parameters()).device
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    log_items = exact_item_logprobs(trie, edge_logprob)
    cand = torch.from_numpy(trie.semantic_ids[rows]).unsqueeze(0).to(device)
    with torch.no_grad():
        model_scores = tiger.score_items_valid_actions(
            moved, candidate_token_ids=cand).float().cpu().numpy()[0]
    err = float(np.abs(model_scores - log_items[rows]).max())
    if err > atol:
        raise AssertionError(
            f"cache/model disagreement {err:.4f} > {atol}: cached path sums do "
            "not reproduce score_items_valid_actions")
    return {"max_abs_item_logprob_err": err, "n_items_checked": int(len(rows))}
