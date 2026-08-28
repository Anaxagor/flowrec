"""SID trie validation and CSR export (spec §2 invariants, §18 cache layout)."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from flowrec.trajectories.nonacyclic import SemanticIDTrie


def trie_metadata(trie: SemanticIDTrie) -> dict[str, Any]:
    return {
        "num_items": int(trie.num_items),
        "sid_len": int(trie.sid_len),
        "num_nodes": int(trie.num_nodes),
        "num_edges": int(trie.num_nodes - 1),
        "nodes_by_depth": [int(x) for x in trie.num_nodes_per_level],
        "num_internal_nodes": int(trie.level_offsets[trie.sid_len]),
        "num_multi_child_internal": int(
            (trie.child_count[trie.internal_node_ids] > 1).sum()),
    }


def validate_trie(
    trie: SemanticIDTrie, expected: Mapping[str, Any], *, codebook_size: int = 256,
    offset_by_one: bool = True,
) -> dict[str, Any]:
    """Assert the spec §2 invariants; returns the metadata on success."""
    meta = trie_metadata(trie)
    problems = []
    if expected.get("num_items") is not None and meta["num_items"] != expected["num_items"]:
        problems.append(f"num_items {meta['num_items']} != {expected['num_items']}")
    if expected.get("sid_len") is not None and meta["sid_len"] != expected["sid_len"]:
        problems.append(f"sid_len {meta['sid_len']} != {expected['sid_len']}")
    if expected.get("trie_nodes") is not None and meta["num_nodes"] != expected["trie_nodes"]:
        problems.append(f"trie_nodes {meta['num_nodes']} != {expected['trie_nodes']}")
    want_depth = expected.get("nodes_by_depth")
    if want_depth is not None and meta["nodes_by_depth"] != list(want_depth):
        problems.append(f"nodes_by_depth {meta['nodes_by_depth']} != {list(want_depth)}")

    # unique SIDs after collision suffix, valid root-to-leaf paths
    sids = trie.semantic_ids
    if len(np.unique(sids, axis=0)) != trie.num_items:
        problems.append("item->SID mapping is not unique after the collision suffix")
    for row in np.random.default_rng(0).integers(0, trie.num_items, size=32):
        node = trie.node_of_prefix(sids[row])  # raises on an invalid path
        assert trie.item_row[node] == row

    # level-offset token convention: level l tokens in [1 + 256*l, 256*(l+1)]
    base = 1 if offset_by_one else 0
    for lvl in range(1, trie.sid_len + 1):
        tok = trie.token[trie.level == lvl]
        lo, hi = base + codebook_size * (lvl - 1), base + codebook_size * lvl - 1
        if tok.min() < lo or tok.max() > hi:
            problems.append(
                f"level-{lvl} tokens [{tok.min()}, {tok.max()}] outside [{lo}, {hi}]")

    if problems:
        raise AssertionError("trie invariant violations: " + "; ".join(problems))
    return meta


def csr_arrays(trie: SemanticIDTrie) -> dict[str, np.ndarray]:
    """CSR-style edge layout (spec §18.2). Edges are indexed by CHILD node id
    (1..num_nodes-1); children of a state are contiguous, so
    ``state_child_offsets`` = ``child_start`` and per-user quantities are flat
    vectors indexed by child node id."""
    n = trie.num_nodes
    child = np.arange(1, n, dtype=np.int64)
    return {
        "parent_state_id": trie.parent[1:].astype(np.int64),
        "child_state_id": child,
        "child_token": trie.token[1:].astype(np.int64),
        "edge_depth": trie.level[1:].astype(np.int64),
        "state_child_offsets": trie.child_start.astype(np.int64),
        "state_child_counts": trie.child_count.astype(np.int64),
        "item_row_of_node": trie.item_row.astype(np.int64),
        "level_offsets": trie.level_offsets.astype(np.int64),
    }


def trie_hash(trie: SemanticIDTrie) -> str:
    """Stable identity of the trie/catalog ordering (guards cache reuse)."""
    h = hashlib.sha256()
    h.update(trie.semantic_ids.tobytes())
    h.update(trie.sorted_item_rows.tobytes())
    return h.hexdigest()
