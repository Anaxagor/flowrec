"""Non-acyclic (cyclic) GFlowNet rollout machinery over the TIGER semantic-ID trie.

Implements the controlled rollback experiment: a frozen TIGER model provides the
append (ADD) proposal distribution p_theta(a | c, s) over valid SID continuations,
a frozen SASRec model provides per-item rewards R(c, x) > 0, and a per-state
rollback probability d(c, s) in [0, 1) turns the append-only decoder into a
cyclic policy

    P_F(parent(s) | c, s) = d(c, s)
    P_F(s + a      | c, s) = (1 - d(c, s)) * p_theta(a | c, s)

with, at a complete item state x, P_F(s_f | c, x) = 1 - d(c, x) (STOP) and
P_F(parent(x) | c, x) = d(c, x).

All heavy quantities are flat numpy arrays indexed by trie node so that Monte
Carlo rollouts are vectorized across trajectories. Every non-root trie node has
a unique parent edge, so branch-specific quantities (edge probability
p_theta(a | c, s), reward conditional q_R(a | c, s), rollback probability
d(c, s + a)) are stored per *node*.

The module is model-agnostic: TIGER only enters through the per-edge proposal
probabilities computed by ``compute_tiger_edge_distributions``. A learned
rollback head (second phase, detailed-balance training) can reuse the same
``SemanticIDTrie`` / ``CyclicTrajectorySampler`` by supplying its own
``rollback_prob`` array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------


class SemanticIDTrie:
    """Static trie over all valid item semantic IDs.

    Nodes are indexed 0..num_nodes-1 with the root at index 0, sorted by
    (level, lexicographic prefix). Children of any node are contiguous.

    Because collision suffixes make every complete SID exactly ``sid_len``
    tokens long, terminal (complete-item) nodes are precisely the level-L
    nodes, and no internal node is itself an item.

    Attributes (all numpy arrays of length num_nodes unless stated):
        level:        depth of the node's prefix (root = 0).
        parent:       parent node index (root = -1).
        token:        vocabulary token on the edge parent -> node (root = -1).
        child_start:  index of the first child node (children are contiguous).
        child_count:  number of valid children (0 for terminal nodes).
        item_lo/hi:   range of descendant terminal items in ``sorted_item_rows``.
        item_row:     catalog row (into the [num_items, sid_len] table) for
                      terminal nodes, -1 otherwise.
        sorted_item_rows: [num_items] catalog rows in trie (lexicographic) order.
    """

    def __init__(self, semantic_ids: np.ndarray | torch.Tensor):
        if isinstance(semantic_ids, torch.Tensor):
            semantic_ids = semantic_ids.detach().cpu().numpy()
        semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
        if semantic_ids.ndim != 2 or semantic_ids.size == 0:
            raise ValueError("semantic_ids must be a non-empty [num_items, sid_len] array")
        num_items, sid_len = semantic_ids.shape
        if len(np.unique(semantic_ids, axis=0)) != num_items:
            raise ValueError("semantic IDs must be unique per item (collision suffix missing?)")

        self.num_items = num_items
        self.sid_len = sid_len
        self.semantic_ids = semantic_ids

        # Lexicographic order puts every prefix's descendants in one contiguous run.
        order = np.lexsort(tuple(semantic_ids[:, d] for d in range(sid_len - 1, -1, -1)))
        sorted_sids = semantic_ids[order]
        self.sorted_item_rows = order.astype(np.int64)

        # Per level: boundaries where the d-prefix changes -> one node per run.
        level_nodes: list[dict[str, np.ndarray]] = []
        for d in range(1, sid_len + 1):
            prefix = sorted_sids[:, :d]
            change = np.ones(num_items, dtype=bool)
            change[1:] = (prefix[1:] != prefix[:-1]).any(axis=1)
            starts = np.flatnonzero(change)
            ends = np.append(starts[1:], num_items)
            level_nodes.append({"starts": starts, "ends": ends})

        counts = [1] + [len(ln["starts"]) for ln in level_nodes]
        self.num_nodes = int(sum(counts))
        self.level_offsets = np.cumsum([0] + counts)  # node-id range per level
        self.num_nodes_per_level = np.asarray(counts)

        level = np.zeros(self.num_nodes, dtype=np.int8)
        parent = np.full(self.num_nodes, -1, dtype=np.int64)
        token = np.full(self.num_nodes, -1, dtype=np.int64)
        child_start = np.zeros(self.num_nodes, dtype=np.int64)
        child_count = np.zeros(self.num_nodes, dtype=np.int64)
        item_lo = np.zeros(self.num_nodes, dtype=np.int64)
        item_hi = np.zeros(self.num_nodes, dtype=np.int64)
        item_row = np.full(self.num_nodes, -1, dtype=np.int64)

        item_lo[0], item_hi[0] = 0, num_items
        for d in range(1, sid_len + 1):
            off = self.level_offsets[d]
            starts, ends = level_nodes[d - 1]["starts"], level_nodes[d - 1]["ends"]
            n = len(starts)
            ids = np.arange(off, off + n)
            level[ids] = d
            token[ids] = sorted_sids[starts, d - 1]
            item_lo[ids], item_hi[ids] = starts, ends
            if d == 1:
                parent[ids] = 0
            else:
                # A node's parent is the level-(d-1) node whose item range contains it.
                p_starts = level_nodes[d - 2]["starts"]
                parent[ids] = self.level_offsets[d - 1] + (
                    np.searchsorted(p_starts, starts, side="right") - 1
                )
            if d == sid_len:
                item_row[ids] = order[starts]

        # Children are contiguous because nodes are sorted lexicographically per level.
        for d in range(0, sid_len):
            ids = np.arange(self.level_offsets[d], self.level_offsets[d + 1])
            child_ids = np.arange(self.level_offsets[d + 1], self.level_offsets[d + 2])
            cnt = np.bincount(parent[child_ids] - self.level_offsets[d], minlength=len(ids))
            child_count[ids] = cnt
            child_start[ids] = self.level_offsets[d + 1] + np.cumsum(cnt) - cnt

        self.level = level
        self.parent = parent
        self.token = token
        self.child_start = child_start
        self.child_count = child_count
        self.item_lo = item_lo
        self.item_hi = item_hi
        self.item_row = item_row

        # Terminal nodes in catalog-row order (row -> node id) for fast lookups.
        term_ids = np.arange(self.level_offsets[sid_len], self.level_offsets[sid_len + 1])
        self.row_to_terminal_node = np.empty(num_items, dtype=np.int64)
        self.row_to_terminal_node[item_row[term_ids]] = term_ids

        assert (child_count[term_ids] == 0).all()
        assert int(child_count[: self.level_offsets[sid_len]].min()) >= 1

    # -- convenience -------------------------------------------------------

    def children_of(self, node: int) -> np.ndarray:
        s, c = self.child_start[node], self.child_count[node]
        return np.arange(s, s + c)

    def node_of_prefix(self, tokens: list[int] | np.ndarray) -> int:
        """Walk the trie from the root; raises KeyError on an invalid prefix."""
        node = 0
        for t in tokens:
            ch = self.children_of(node)
            hit = ch[self.token[ch] == int(t)]
            if len(hit) == 0:
                raise KeyError(f"invalid prefix {list(tokens)} at token {t}")
            node = int(hit[0])
        return node

    def prefix_of_node(self, node: int) -> list[int]:
        out: list[int] = []
        while node != 0:
            out.append(int(self.token[node]))
            node = int(self.parent[node])
        return out[::-1]

    @property
    def internal_node_ids(self) -> np.ndarray:
        return np.arange(0, self.level_offsets[self.sid_len])

    @property
    def terminal_node_ids(self) -> np.ndarray:
        return np.arange(self.level_offsets[self.sid_len], self.num_nodes)


# ---------------------------------------------------------------------------
# Exact reward mass and reward conditionals
# ---------------------------------------------------------------------------


def reward_mass(trie: SemanticIDTrie, item_rewards: np.ndarray) -> np.ndarray:
    """M_R(c, s) = sum of R(c, x) over descendant items of s, for every node.

    Computed bottom-up level by level (children of consecutive same-level nodes
    tile the next level, so each level is one reduceat over positive terms).
    Unlike a global prefix-sum, this has no cancellation, so subtree masses stay
    exact even under extreme reward dynamic range (e.g. Boltzmann targets whose
    tail rewards are ~1e-30 of Z). Root equals Z(c); leaves equal R(c, x).
    """
    r = np.asarray(item_rewards, dtype=np.float64)
    if r.shape != (trie.num_items,):
        raise ValueError(f"item_rewards must have shape ({trie.num_items},)")
    if not (r > 0).all():
        raise ValueError("rewards must be strictly positive for a GFlowNet target")
    mass = np.zeros(trie.num_nodes, dtype=np.float64)
    term = trie.terminal_node_ids
    mass[term] = r[trie.item_row[term]]
    for depth in range(trie.sid_len - 1, -1, -1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        lo, hi = trie.level_offsets[depth + 1], trie.level_offsets[depth + 2]
        mass[ids] = np.add.reduceat(mass[lo:hi], trie.child_start[ids] - lo)
    return mass


def reward_conditionals(trie: SemanticIDTrie, mass: np.ndarray) -> np.ndarray:
    """q_R(a | c, s) for every edge, stored at the child node: M(child)/M(parent)."""
    q = np.zeros(trie.num_nodes, dtype=np.float64)
    q[1:] = mass[1:] / mass[trie.parent[1:]]
    q[0] = 1.0
    return q


def check_reward_conditionals(trie: SemanticIDTrie, target_q: np.ndarray, atol: float = 1e-9) -> float:
    """Verify sum_a q_R(a | c, s) == 1 at every internal node; returns max |err|."""
    sums = np.add.reduceat(target_q, trie.child_start[trie.internal_node_ids])
    # reduceat with contiguous children: child_start of internal nodes indexes the
    # flat node array, and children of consecutive nodes are consecutive.
    err = float(np.abs(sums - 1.0).max())
    if err > atol:
        raise AssertionError(f"reward conditionals do not sum to 1 (max err {err:.3e})")
    return err


# ---------------------------------------------------------------------------
# Frozen TIGER per-edge proposal distribution
# ---------------------------------------------------------------------------


@dataclass
class TigerPrefixTables:
    """Frozen-TIGER quantities on the trie for one user context.

    edge_prob:  [num_nodes] p_theta(token(v) | c, parent(v)) restricted and
                renormalized over the valid children of parent(v) (constrained
                decoding); root = 1.
    log_edge_prob: log of the above (log-space, from log_softmax).
    entropy:    [num_nodes] predictive entropy of the constrained next-token
                distribution at each *internal* node (nats); 0 elsewhere.
    norm_entropy: entropy / log(child_count), 0 where child_count <= 1.
    raw_valid_mass: [num_nodes] unconstrained softmax mass TIGER puts on the
                valid children of each evaluated internal node (diagnostic).
    evaluated:  bool mask of internal nodes whose logits were actually queried
                (single-child nodes may be skipped: their constrained
                distribution is deterministic).
    """

    edge_prob: np.ndarray
    log_edge_prob: np.ndarray
    entropy: np.ndarray
    norm_entropy: np.ndarray
    raw_valid_mass: np.ndarray
    evaluated: np.ndarray


@torch.no_grad()
def compute_tiger_edge_distributions(
    model: Any,
    batch: dict[str, torch.Tensor],
    trie: SemanticIDTrie,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    skip_single_child: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> TigerPrefixTables:
    """Query frozen TIGER next-token logits at every internal trie node.

    ``batch`` must be a single-user TIGER batch (user_ids of size 1) with
    history tokens in the same pre-offset vocabulary as the trie tokens. The
    encoder runs once; the decoder is teacher-forced on each prefix, batched
    per level so decoder lengths stay uniform.
    """
    model_device = next(model.parameters()).device
    single = {
        k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }
    if int(single["user_ids"].size(0)) != 1:
        raise ValueError("expected a single-user batch")

    encoder_hidden, attention_mask = model._history_encoder_outputs(single)

    n = trie.num_nodes
    edge_prob = np.ones(n, dtype=np.float64)
    log_edge_prob = np.zeros(n, dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)
    norm_entropy = np.zeros(n, dtype=np.float64)
    raw_valid_mass = np.zeros(n, dtype=np.float64)
    evaluated = np.zeros(n, dtype=bool)

    internal = trie.internal_node_ids
    todo = internal if not skip_single_child else internal[trie.child_count[internal] > 1]
    # Deterministic edges (single child): constrained prob 1, log-prob 0 — already
    # the array defaults. Their parent entropy stays 0.

    start_tok = int(model.decoder_start_token_id)
    total = len(todo)
    done = 0
    for depth in range(0, trie.sid_len):
        ids = todo[trie.level[todo] == depth]
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            b = len(chunk)
            # decoder_input_ids = [start, z_1..z_depth] for each prefix.
            dec = np.empty((b, depth + 1), dtype=np.int64)
            dec[:, 0] = start_tok
            node = chunk.copy()
            for pos in range(depth, 0, -1):
                dec[:, pos] = trie.token[node]
                node = trie.parent[node]
            dec_t = torch.from_numpy(dec).to(model_device)
            out = model.model(
                encoder_outputs=(encoder_hidden.expand(b, -1, -1),),
                attention_mask=attention_mask.expand(b, -1),
                decoder_input_ids=dec_t,
            )
            logits = out.logits[:, -1, :].float()  # [b, vocab]

            # Gather each node's valid children tokens into a padded matrix.
            counts = trie.child_count[chunk]
            max_c = int(counts.max())
            child_tok = np.zeros((b, max_c), dtype=np.int64)
            child_mask = np.zeros((b, max_c), dtype=bool)
            for j, nd in enumerate(chunk):
                cs, cc = trie.child_start[nd], trie.child_count[nd]
                child_tok[j, :cc] = trie.token[cs : cs + cc]
                child_mask[j, :cc] = True
            tok_t = torch.from_numpy(child_tok).to(logits.device)
            mask_t = torch.from_numpy(child_mask).to(logits.device)

            child_logits = logits.gather(1, tok_t).masked_fill(~mask_t, -torch.inf)
            log_p = child_logits - torch.logsumexp(child_logits, dim=1, keepdim=True)
            p = log_p.exp().masked_fill(~mask_t, 0.0)
            ent = -(p * log_p.masked_fill(~mask_t, 0.0)).sum(dim=1)

            # Unconstrained mass on valid children (support diagnostic).
            full_logsumexp = torch.logsumexp(logits, dim=1, keepdim=True)
            raw_mass = (child_logits - full_logsumexp).exp().masked_fill(~mask_t, 0.0).sum(dim=1)

            p_np = p.cpu().numpy()
            logp_np = log_p.cpu().numpy()
            for j, nd in enumerate(chunk):
                cs, cc = trie.child_start[nd], trie.child_count[nd]
                edge_prob[cs : cs + cc] = p_np[j, :cc]
                log_edge_prob[cs : cs + cc] = logp_np[j, :cc]
            entropy[chunk] = ent.cpu().numpy()
            cc = counts.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_entropy[chunk] = np.where(cc > 1, entropy[chunk] / np.log(cc), 0.0)
            raw_valid_mass[chunk] = raw_mass.cpu().numpy()
            evaluated[chunk] = True
            done += b
            if progress is not None:
                progress(done, total)

    return TigerPrefixTables(
        edge_prob=edge_prob,
        log_edge_prob=log_edge_prob,
        entropy=entropy,
        norm_entropy=norm_entropy,
        raw_valid_mass=raw_valid_mass,
        evaluated=evaluated,
    )


# ---------------------------------------------------------------------------
# Distribution mismatch (per internal node)
# ---------------------------------------------------------------------------


def mismatch_tables(
    trie: SemanticIDTrie,
    tables: TigerPrefixTables,
    target_q: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-internal-node D_KL(q_R || p_theta), D_KL(p_theta || q_R) and TV distance."""
    internal = trie.internal_node_ids
    kl = np.zeros(len(internal), dtype=np.float64)
    kl_pq = np.zeros(len(internal), dtype=np.float64)
    tv = np.zeros(len(internal), dtype=np.float64)
    for k, nd in enumerate(internal):
        cs, cc = trie.child_start[nd], trie.child_count[nd]
        if cc <= 1:
            continue
        q = target_q[cs : cs + cc]
        p = tables.edge_prob[cs : cs + cc]
        logp = tables.log_edge_prob[cs : cs + cc]
        nz = q > 0
        kl[k] = float(np.sum(q[nz] * (np.log(q[nz]) - logp[nz])))
        pz = p > 0
        # KL(p || q): expected log over-representation of TIGER's own choices.
        kl_pq[k] = float(np.sum(p[pz] * (logp[pz] - np.log(np.maximum(q[pz], 1e-300)))))
        tv[k] = 0.5 * float(np.abs(q - p).sum())
    return {"node": internal, "kl_q_p": kl, "kl_p_q": kl_pq, "tv_q_p": tv}


# ---------------------------------------------------------------------------
# Rollback policies: per-node d(c, s) in [0, 1)
# ---------------------------------------------------------------------------


@dataclass
class OracleDiagnostics:
    """Support / clipping report for the oracle rejection construction."""

    n_support_mismatch_edges: int = 0
    n_clipped_accept: int = 0
    max_accept_before_clip: float = 1.0
    parent_C: np.ndarray | None = None  # per-node max_a q/p at internal nodes


def rollback_append_only(trie: SemanticIDTrie) -> np.ndarray:
    """A. d(c, s) = 0 everywhere: standard constrained stochastic TIGER."""
    return np.zeros(trie.num_nodes, dtype=np.float64)


def rollback_uniform_level(trie: SemanticIDTrie, d_by_level: dict[int, float] | float) -> np.ndarray:
    """B. d(c, s) = d_level for every node with |s| = level (level >= 1)."""
    d = np.zeros(trie.num_nodes, dtype=np.float64)
    for lvl in range(1, trie.sid_len + 1):
        val = d_by_level if isinstance(d_by_level, (int, float)) else d_by_level.get(lvl, 0.0)
        if not (0.0 <= float(val) < 1.0):
            raise ValueError(f"rollback prob at level {lvl} must be in [0,1), got {val}")
        d[trie.level == lvl] = float(val)
    d[0] = 0.0
    return d


def rollback_random_branch(
    trie: SemanticIDTrie,
    matched: np.ndarray,
    seed: int,
) -> np.ndarray:
    """C. Branch-specific but reward/uncertainty-blind rollback.

    Permutes the matched policy's d values *within every (level) stratum*, so
    the marginal distribution of rollback probabilities per SID level is
    matched exactly while any branch-specific information is destroyed.
    """
    rng = np.random.default_rng(seed)
    d = np.zeros(trie.num_nodes, dtype=np.float64)
    for lvl in range(1, trie.sid_len + 1):
        ids = np.flatnonzero(trie.level == lvl)
        d[ids] = rng.permutation(matched[ids])
    d[0] = 0.0
    return d


def rollback_entropy(
    trie: SemanticIDTrie,
    tables: TigerPrefixTables,
    *,
    d_min: float,
    d_max: float,
) -> np.ndarray:
    """D. d(c, s_l) = d_min + (d_max - d_min) * Hnorm_theta(c, s_{l-1}).

    The rollback probability of a state depends on TIGER's uncertainty at the
    *parent* decision that produced it — uncertainty before the selected token.
    """
    if not (0.0 <= d_min <= d_max < 1.0):
        raise ValueError("need 0 <= d_min <= d_max < 1")
    d = np.zeros(trie.num_nodes, dtype=np.float64)
    d[1:] = d_min + (d_max - d_min) * tables.norm_entropy[trie.parent[1:]]
    d[0] = 0.0
    return d


def rollback_oracle_reward(
    trie: SemanticIDTrie,
    tables: TigerPrefixTables,
    target_q: np.ndarray,
    *,
    eps: float = 1e-12,
    max_d: float = 1.0 - 1e-9,
) -> tuple[np.ndarray, OracleDiagnostics]:
    """E. d_R(c, s+a) = 1 - A_R(a | c, s) with A_R = q_R / (C * p_theta).

    C(c, s) = max over valid children with p_theta > eps of q_R / p_theta.
    Under repeated propose-from-p / accept-with-A rejection, accepted children
    are distributed exactly as q_R. Support mismatches (q_R > 0 where
    p_theta <= eps) are counted and reported, not silently clipped.
    """
    diag = OracleDiagnostics(parent_C=np.ones(trie.num_nodes, dtype=np.float64))
    d = np.zeros(trie.num_nodes, dtype=np.float64)
    accept = np.ones(trie.num_nodes, dtype=np.float64)
    internal = trie.internal_node_ids
    for nd in internal:
        cs, cc = trie.child_start[nd], trie.child_count[nd]
        if cc == 1:
            continue  # deterministic edge: q = p = 1, accept = 1, d = 0
        q = target_q[cs : cs + cc]
        p = tables.edge_prob[cs : cs + cc]
        ok = p > eps
        diag.n_support_mismatch_edges += int(np.sum(~ok & (q > 0)))
        ratio = np.where(ok, q / np.maximum(p, eps), 0.0)
        C = float(ratio.max())
        diag.parent_C[nd] = C
        a = np.where(ok, q / (C * np.maximum(p, eps)), 1.0)  # unreachable edges: keep d=0
        over = a > 1.0 + 1e-9
        if over.any():
            diag.n_clipped_accept += int(over.sum())
            diag.max_accept_before_clip = max(diag.max_accept_before_clip, float(a.max()))
        accept[cs : cs + cc] = np.clip(a, 1.0 - max_d, 1.0)
    d[1:] = 1.0 - accept[1:]
    return d, diag


# ---------------------------------------------------------------------------
# Global flow oracle (globally flow-consistent cyclic policy)
# ---------------------------------------------------------------------------


@dataclass
class GlobalFlowOracle:
    """Exact edge/state flows realizing the reward target under frozen TIGER ratios.

    Construction (per user context; all arrays indexed by trie node):

        C(s)   = max_v q_R(v|s) / p_theta(v|s)          (internal nodes; >= 1)
        G(s)   = M(s) C(s)                               total downward flow
        F(s->v) = G(s) p_theta(v|s)                      ADD edge flow (at child v)
        kappa(v) = M(p(v)) [C(p(v)) p_theta(v) - q_R(v)] reverse DELETE flow (at child v)
        F(s)   = G(s) + kappa(s)          internal      state flow
        F(x)   = R(x) + kappa(x)          item leaf
        d*(s)  = kappa(s) / F(s)                         rollback probability (root: 0)

    Conditional on ADD the policy equals p_theta exactly, so the oracle only
    chooses how often each state returns to its parent. G(s) = M(s)C(s) is the
    minimum downward flow keeping every kappa >= 0 under frozen TIGER ratios
    (at the maximizing child, kappa = 0).
    """

    C: np.ndarray            # per node; internal: max_v q/p, elsewhere 1
    G: np.ndarray            # per node: M * C (downward flow; leaves unused)
    add_flow: np.ndarray     # per non-root node v: F(p(v) -> v); root: 0
    kappa: np.ndarray        # per non-root node v: F(v -> p(v)); root: 0
    state_flow: np.ndarray   # per node: F(c, s)
    d_star: np.ndarray       # per node: P*_F(parent | s); root: 0
    diagnostics: dict = field(default_factory=dict)


def build_global_flow_oracle(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    target_q: np.ndarray,
    mass: np.ndarray,
    *,
    support_eps: float = 0.0,
) -> GlobalFlowOracle:
    """Construct the global flow oracle and unit-test every flow identity.

    Raises on true support violations (q_R > 0 where p_theta <= support_eps):
    the target cannot be realized while preserving frozen TIGER ADD ratios.
    """
    p = np.asarray(edge_prob, dtype=np.float64)
    q = np.asarray(target_q, dtype=np.float64)
    M = np.asarray(mass, dtype=np.float64)
    n = trie.num_nodes
    internal = trie.internal_node_ids
    par = trie.parent

    bad = (q > 0) & (p <= support_eps)
    bad[0] = False
    n_support_violations = int(bad.sum())
    if n_support_violations:
        raise ValueError(
            f"{n_support_violations} support violations: q_R > 0 where p_theta <= "
            f"{support_eps}; the oracle cannot realize the target for this user."
        )

    # The flow algebra assumes sum_children p = 1 exactly; float32 softmax leaves
    # ~1e-6 deviations, so require the caller to renormalize in float64 first.
    sums_p = np.add.reduceat(p, trie.child_start[internal])
    max_p_sum_dev = float(np.abs(sums_p - 1.0).max())
    if max_p_sum_dev > 1e-9:
        raise ValueError(
            f"sum of edge_prob over children deviates from 1 by {max_p_sum_dev:.3e}; "
            "renormalize the constrained proposal per node (float64) before building the oracle."
        )

    ratio = np.zeros(n, dtype=np.float64)
    ratio[1:] = q[1:] / p[1:]
    C = np.ones(n, dtype=np.float64)
    # children of consecutive internal nodes tile the node array (incl. across
    # level boundaries), so a single reduceat covers all internal nodes.
    C[internal] = np.maximum.reduceat(ratio, trie.child_start[internal])
    C_min = float(C[internal].min())

    G = M * C
    add_flow = np.zeros(n, dtype=np.float64)
    kappa = np.zeros(n, dtype=np.float64)
    add_flow[1:] = G[par[1:]] * p[1:]
    # Cancellation-free circulation: kappa := F(s->v) - M(v), making the
    # fundamental net-flow identity exact by construction. The analytic form
    # M(s)[C p - q] is identical algebraically but subtracts two large C p
    # terms; its agreement is checked below on the ADD-flow scale.
    kappa[1:] = add_flow[1:] - M[1:]
    kappa_formula = np.zeros(n, dtype=np.float64)
    kappa_formula[1:] = M[par[1:]] * (C[par[1:]] * p[1:] - q[1:])
    kappa_formula_disagreement = float(np.max(
        np.abs(kappa[1:] - kappa_formula[1:]) / np.maximum(add_flow[1:], 1e-300)))
    neg_mask = kappa < 0
    max_neg_kappa = float(-kappa[neg_mask].min()) if neg_mask.any() else 0.0
    kappa[neg_mask] = 0.0  # clip fp-noise negatives (magnitude reported)

    state_flow = np.empty(n, dtype=np.float64)
    state_flow[internal] = G[internal] + kappa[internal]
    term = trie.terminal_node_ids
    state_flow[term] = M[term] + kappa[term]
    d_star = np.zeros(n, dtype=np.float64)
    d_star[1:] = kappa[1:] / state_flow[1:]

    # --- identity unit tests (vectorized over the whole trie) ----------------
    # net flow across every parent-child cut: F(s->v) - F(v->s) = M(v)
    net_err = float(np.max(np.abs(add_flow[1:] - kappa[1:] - M[1:]) / np.maximum(M[1:], 1e-300)))
    # state-flow conservation <=> G(s) = M(s) + sum_children kappa(v)
    sum_kappa_children = np.add.reduceat(kappa, trie.child_start[internal])
    cons_err = float(np.max(np.abs(G[internal] - M[internal] - sum_kappa_children)
                            / np.maximum(G[internal], 1e-300)))
    # minimum-circulation property: per state, the maximizing child has kappa ~ 0
    min_kappa_rel = np.minimum.reduceat(
        np.where(np.arange(n) > 0, kappa / np.maximum(add_flow, 1e-300), np.inf),
        trie.child_start[internal],
    )
    max_min_kappa_rel = float(min_kappa_rel.max())
    # G p_theta(v) >= M(v) for every child (equality at the maximizing child)
    slack = add_flow[1:] - M[1:]
    min_slack_rel = float(np.min(slack / np.maximum(M[1:], 1e-300)))

    diagnostics = {
        "n_support_violations": n_support_violations,
        "C_min": C_min,
        "max_p_sum_deviation": max_p_sum_dev,
        "max_negative_kappa_before_clip": max_neg_kappa,
        "max_kappa_formula_disagreement": kappa_formula_disagreement,
        "max_rel_err_net_flow": net_err,
        "max_rel_err_conservation": cons_err,
        "max_over_states_min_child_kappa_rel": max_min_kappa_rel,
        "min_rel_slack_G_p_minus_M": min_slack_rel,
    }
    return GlobalFlowOracle(
        C=C, G=G, add_flow=add_flow, kappa=kappa,
        state_flow=state_flow, d_star=d_star, diagnostics=diagnostics,
    )


def absorbing_chain_expected_visits(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    rollback_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent absorbing-Markov-chain solution for any cyclic policy.

    Solves (I - P^T) mu = e_root with sparse LU, where P is the transient
    transition matrix of the policy (DELETE prob d(s) to the parent, ADD prob
    (1-d(s)) p_theta(v|s) to children; leaves absorb via STOP with 1-d(x)).

    Returns (mu, terminal): mu[s] = expected visits to state s from one source
    injection (theory: mu = F(c,s)/Z(c)); terminal = exact absorption
    distribution over catalog rows (mu[x] * (1 - d(x)) at leaves).
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n = trie.num_nodes
    d = np.asarray(rollback_prob, dtype=np.float64)
    p = np.asarray(edge_prob, dtype=np.float64)
    par = trie.parent
    v = np.arange(1, n)
    # P^T entries: (to, from). DELETE: v -> parent(v) with d[v]; ADD: parent(v) -> v.
    rows = np.concatenate([par[v], v])
    cols = np.concatenate([v, par[v]])
    data = np.concatenate([d[v], (1.0 - d[par[v]]) * p[v]])
    A = sparse.eye(n, format="csc") - sparse.csc_matrix((data, (rows, cols)), shape=(n, n))
    e = np.zeros(n)
    e[0] = 1.0
    mu = spsolve(A, e)
    term = trie.terminal_node_ids
    terminal = np.zeros(trie.num_items, dtype=np.float64)
    terminal[trie.item_row[term]] = mu[term] * (1.0 - d[term])
    return mu, terminal


# ---------------------------------------------------------------------------
# Cyclic trajectory sampler (vectorized Monte Carlo)
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryBatchResult:
    """Aggregated result of a vectorized cyclic rollout batch.

    terminal_row:   [n_traj] catalog row of the accepted item, -1 if the
                    trajectory hit max_trajectory_steps (non-terminated).
    n_steps:        [n_traj] number of sampled transitions (ADD+DELETE+STOP).
    n_deletes:      [n_traj] number of DELETE transitions.
    delete_events:  structured record of every DELETE: trajectory id, SID level
                    the delete happened at (level of the deleted state), the
                    deleted token, parent node, deleted child node, plus the
                    follow-up ADD (next action taken at that same parent):
                    reselected token / new child node, or -1 when the next
                    action was another DELETE (or the trajectory was truncated).
    trajectories:   optional list of full node-id paths for the first
                    ``record_trajectories`` trajectories (root included).
    """

    terminal_row: np.ndarray
    n_steps: np.ndarray
    n_deletes: np.ndarray
    delete_events: dict[str, np.ndarray]
    trajectories: list[list[int]] | None = None
    delete_level_counts: np.ndarray | None = None  # [sid_len+1] total DELETEs per level


class CyclicTrajectorySampler:
    """Vectorized sampler for the cyclic forward policy on the SID trie.

    Parameters
    ----------
    trie:          SemanticIDTrie.
    edge_prob:     [num_nodes] constrained TIGER edge probabilities (at child).
    rollback_prob: [num_nodes] d(c, s) per state; d[root] must be 0.
    """

    def __init__(
        self,
        trie: SemanticIDTrie,
        edge_prob: np.ndarray,
        rollback_prob: np.ndarray,
    ):
        if rollback_prob[0] != 0.0:
            raise ValueError("d(root) must be 0: the empty prefix has no parent")
        if not ((rollback_prob >= 0) & (rollback_prob < 1)).all():
            raise ValueError("rollback probabilities must be in [0, 1)")
        self.trie = trie
        self.d = rollback_prob
        # Per-internal-node CDF over children for O(log c) sampling via searchsorted
        # on the flat (contiguous-children) probability array.
        self.edge_prob = edge_prob
        self._edge_cdf = np.cumsum(edge_prob)

    def _sample_children(self, nodes: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Sample one child per node from the constrained TIGER distribution."""
        cs = self.trie.child_start[nodes]
        cc = self.trie.child_count[nodes]
        # cs >= 1 always (the root is node 0 and is nobody's child), so cs - 1 is safe.
        lo = self._edge_cdf[cs - 1]
        total = self._edge_cdf[cs + cc - 1] - lo
        # Guard against fp drift: clip the uniform draw inside the node's mass.
        target = lo + np.minimum(u * total, np.nextafter(total, 0))
        idx = np.searchsorted(self._edge_cdf, target, side="right")
        return np.clip(idx, cs, cs + cc - 1)

    def sample(
        self,
        n_traj: int,
        *,
        seed: int,
        max_trajectory_steps: int = 200,
        record_trajectories: int = 0,
        record_delete_events: bool = True,
    ) -> TrajectoryBatchResult:
        trie, d = self.trie, self.d
        rng = np.random.default_rng(seed)

        state = np.zeros(n_traj, dtype=np.int64)  # all start at the root
        active = np.ones(n_traj, dtype=bool)
        n_steps = np.zeros(n_traj, dtype=np.int64)
        n_deletes = np.zeros(n_traj, dtype=np.int64)
        terminal_row = np.full(n_traj, -1, dtype=np.int64)

        # Pending revision bookkeeping: after t deletes s+a -> s we wait for the
        # next action taken by t; if it is an ADD from the same s we record the
        # reselected token, otherwise the follow-up fields stay -1.
        pending_parent = np.full(n_traj, -1, dtype=np.int64)
        pending_event = np.full(n_traj, -1, dtype=np.int64)

        ev_traj: list[np.ndarray] = []
        ev_level: list[np.ndarray] = []
        ev_parent: list[np.ndarray] = []
        ev_deleted_child: list[np.ndarray] = []
        ev_new_child: list[np.ndarray] = []  # filled later; -1 = no immediate ADD
        n_events = 0
        delete_level_counts = np.zeros(trie.sid_len + 1, dtype=np.int64)

        keep = min(record_trajectories, n_traj)
        recorded: list[list[int]] | None = [[0] for _ in range(keep)] if keep else None

        new_child_flat: list[np.ndarray] = []

        for _ in range(max_trajectory_steps):
            if not active.any():
                break
            ids = np.flatnonzero(active)
            cur = state[ids]
            u_gate = rng.random(len(ids))
            is_terminal = trie.level[cur] == trie.sid_len
            do_delete = u_gate < d[cur]

            # --- STOP at complete item states -------------------------------
            stop_mask = is_terminal & ~do_delete
            if stop_mask.any():
                sel = ids[stop_mask]
                terminal_row[sel] = trie.item_row[state[sel]]
                active[sel] = False

            # --- DELETE ------------------------------------------------------
            del_mask = do_delete
            if del_mask.any():
                sel = ids[del_mask]
                child = state[sel]
                par = trie.parent[child]
                np.add.at(delete_level_counts, trie.level[child], 1)
                if record_delete_events:
                    # a DELETE both closes any pending revision (as "no ADD
                    # followed") and opens a new pending one at its own parent.
                    ev_traj.append(sel)
                    ev_level.append(trie.level[child].astype(np.int64))
                    ev_parent.append(par)
                    ev_deleted_child.append(child)
                    pending_parent[sel] = par
                    pending_event[sel] = n_events + np.arange(len(sel))
                    n_events += len(sel)
                state[sel] = par
                n_deletes[sel] += 1

            # --- ADD ---------------------------------------------------------
            add_mask = ~is_terminal & ~do_delete
            if add_mask.any():
                sel = ids[add_mask]
                child = self._sample_children(state[sel], rng.random(len(sel)))
                # close pending revisions whose next action is this ADD
                pend = pending_event[sel] >= 0
                if pend.any():
                    same = state[sel][pend] == pending_parent[sel][pend]
                    evs = pending_event[sel][pend][same]
                    new_child_flat.append(np.stack([evs, child[pend][same]], axis=1))
                pending_event[sel] = -1
                pending_parent[sel] = -1
                state[sel] = child

            n_steps[ids] += 1
            if recorded is not None:
                for t in range(keep):
                    if active[t]:
                        recorded[t].append(int(state[t]))

        delete_events: dict[str, np.ndarray]
        if n_events:
            ev_new = np.full(n_events, -1, dtype=np.int64)
            if new_child_flat:
                pairs = np.concatenate(new_child_flat, axis=0)
                ev_new[pairs[:, 0]] = pairs[:, 1]
            deleted_child = np.concatenate(ev_deleted_child)
            delete_events = {
                "traj": np.concatenate(ev_traj),
                "level": np.concatenate(ev_level),
                "parent": np.concatenate(ev_parent),
                "deleted_child": deleted_child,
                "deleted_token": trie.token[deleted_child],
                "new_child": ev_new,
                "new_token": np.where(ev_new >= 0, trie.token[np.maximum(ev_new, 0)], -1),
            }
        else:
            delete_events = {
                k: np.empty(0, dtype=np.int64)
                for k in ("traj", "level", "parent", "deleted_child", "deleted_token", "new_child", "new_token")
            }

        return TrajectoryBatchResult(
            terminal_row=terminal_row,
            n_steps=n_steps,
            n_deletes=n_deletes,
            delete_events=delete_events,
            trajectories=recorded,
            delete_level_counts=delete_level_counts,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def node_reach_probability(trie: SemanticIDTrie, edge_prob: np.ndarray) -> np.ndarray:
    """Append-only visit probability of every node: product of edge probs to it."""
    reach = np.ones(trie.num_nodes, dtype=np.float64)
    for d in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[d], trie.level_offsets[d + 1])
        reach[ids] = reach[trie.parent[ids]] * edge_prob[ids]
    return reach


def exact_terminal_distribution(trie: SemanticIDTrie, edge_prob: np.ndarray) -> np.ndarray:
    """Exact terminal item distribution of append-only (d=0) constrained sampling.

    P(x | c) = prod of constrained TIGER edge probabilities along the SID of x,
    returned over catalog rows. This is what stochastic constrained TIGER
    sampling converges to, and the d=0 reference for the sampler (Check 4).
    """
    reach = node_reach_probability(trie, edge_prob)
    p = np.zeros(trie.num_items, dtype=np.float64)
    term = trie.terminal_node_ids
    p[trie.item_row[term]] = reach[term]
    total = p.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise AssertionError(f"exact terminal distribution sums to {total}, expected 1")
    return p / total


def exact_cyclic_terminal_distribution(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    rollback_prob: np.ndarray,
) -> np.ndarray:
    """Exact terminal item distribution of the cyclic policy (no step limit).

    Let beta(v) = P(the walk eventually STOPs inside subtree(v) | it just
    arrived at v). At a terminal node beta = 1 - d (STOP now, else pop). At an
    internal node, one sojourn either pops (prob d) or descends into child a
    with prob (1-d) p(a); a child excursion that fails returns the walk to v,
    where the same sojourn distribution applies again:

        beta(v) = (1-d) B / (1 - (1-d)(1-B)),   B = sum_a p(a) beta(a).

    Given absorption below v, the committed child is a with probability
    p(a) beta(a) / B, so the exact terminal distribution is the product of
    these committed-child conditionals down the trie. Since d < 1 everywhere
    and d(root) = 0, the walk absorbs with probability 1 (beta(root) = 1);
    the finite max_trajectory_steps truncation of the sampler is the only
    difference from this infinite-horizon quantity.
    """
    d = rollback_prob
    beta = np.zeros(trie.num_nodes, dtype=np.float64)
    term = trie.terminal_node_ids
    beta[term] = 1.0 - d[term]
    for depth in range(trie.sid_len - 1, -1, -1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        # children of level-`depth` nodes tile exactly the level-(depth+1) slice;
        # restrict the reduction to that slice so the last node's segment ends there.
        lo, hi = trie.level_offsets[depth + 1], trie.level_offsets[depth + 2]
        contrib = edge_prob[lo:hi] * beta[lo:hi]
        B = np.add.reduceat(contrib, trie.child_start[ids] - lo)
        beta[ids] = (1.0 - d[ids]) * B / (1.0 - (1.0 - d[ids]) * (1.0 - B))
    if not np.isclose(beta[0], 1.0, atol=1e-9):
        raise AssertionError(f"root absorption probability is {beta[0]}, expected 1")
    # committed-child conditional q_hat(a | v) = p(a) beta(a) / B(v), taken down the trie.
    # (Indexing over all internal nodes at once is safe: consecutive internal nodes'
    # child ranges tile the node array, including across level boundaries.)
    B_parent = np.add.reduceat(edge_prob * beta, trie.child_start[trie.internal_node_ids])
    commit = np.ones(trie.num_nodes, dtype=np.float64)
    for depth in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        parents = trie.parent[ids]
        commit[ids] = commit[parents] * edge_prob[ids] * beta[ids] / B_parent[parents]
    p = np.zeros(trie.num_items, dtype=np.float64)
    p[trie.item_row[term]] = commit[term]
    total = p.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise AssertionError(f"exact cyclic terminal distribution sums to {total}")
    return p / total


def terminal_distribution(result: TrajectoryBatchResult, num_items: int) -> np.ndarray:
    """Empirical P_terminal(x | c) over catalog rows from terminated trajectories."""
    rows = result.terminal_row[result.terminal_row >= 0]
    counts = np.bincount(rows, minlength=num_items).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("no trajectory terminated; cannot form a terminal distribution")
    return counts / counts.sum()


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray, *, smooth: float = 1e-12) -> float:
    """D_KL(p || q) with additive smoothing on q (renormalized)."""
    qs = (q + smooth) / (q + smooth).sum()
    nz = p > 0
    return float(np.sum(p[nz] * (np.log(p[nz]) - np.log(qs[nz]))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, smooth=0.0) + 0.5 * kl_divergence(q, m, smooth=0.0)


def revision_metrics(
    trie: SemanticIDTrie,
    mass: np.ndarray,
    events: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Same-token reselection + revision precision / normalized reward-mass gain.

    Only DELETE events whose immediately following action was an ADD at the
    same parent count as *revisions* (new_child >= 0).
    """
    n = len(events["traj"])
    out: dict[str, Any] = {"n_delete_events": int(n)}
    if n == 0:
        out.update(
            n_revisions=0, same_token_reselect_prob=np.nan,
            revision_precision=np.nan, mean_delta_mass=np.nan,
        )
        return out
    rev = events["new_child"] >= 0
    out["n_revisions"] = int(rev.sum())
    if rev.any():
        same = events["new_token"][rev] == events["deleted_token"][rev]
        out["same_token_reselect_prob"] = float(same.mean())
        m_new = mass[events["new_child"][rev]]
        m_old = mass[events["deleted_child"][rev]]
        m_par = mass[events["parent"][rev]]
        out["revision_precision"] = float((m_new > m_old).mean())
        out["mean_delta_mass"] = float(((m_new - m_old) / m_par).mean())
        out["revision_levels"] = events["level"][rev]
        out["revision_improved"] = m_new > m_old
        out["revision_same_token"] = same
    else:
        out.update(same_token_reselect_prob=np.nan, revision_precision=np.nan, mean_delta_mass=np.nan)
    return out


# ---------------------------------------------------------------------------
# Local (single-state) rejection simulators for the analytical checks
# ---------------------------------------------------------------------------


def simulate_state_acceptance(
    p_children: np.ndarray,
    d_children: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Empirical accepted-child distribution of propose-p / reject-with-d loops.

    Models the local behavior of the cyclic policy at one decision point when
    every rejection returns to the same parent: propose a ~ p, accept with
    prob 1 - d(a), else repeat. Returns the empirical accepted distribution.
    """
    rng = np.random.default_rng(seed)
    accepted = np.zeros(len(p_children), dtype=np.int64)
    remaining = n_samples
    while remaining > 0:
        take = min(remaining, 1_000_000)
        a = rng.choice(len(p_children), size=take, p=p_children / p_children.sum())
        keep = rng.random(take) >= d_children[a]
        acc = a[keep]
        accepted += np.bincount(acc, minlength=len(p_children))
        remaining -= len(acc)
    return accepted / accepted.sum()
