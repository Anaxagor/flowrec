"""Log-space reward transforms, prefix masses, and hybrid targets (spec §11, §24).

All reward and mass arithmetic is carried in log space: no artificial zeros
from underflow, no silent flooring. A floor may be configured explicitly; when
it is, every clamp is counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flowrec.trajectories.nonacyclic import SemanticIDTrie

LOG_FLOAT64_TINY = -745.0  # exp() underflows to 0 below this


@dataclass(frozen=True)
class RewardTransformSpec:
    """name examples: exp_T0.25 (Boltzmann), softplus_T1 (legacy default)."""

    name: str
    kind: str            # 'boltzmann' | 'softplus'
    temperature: float
    log_floor: float | None = None   # explicit optional floor in log space


def parse_transform(name: str, log_floor: float | None = None) -> RewardTransformSpec:
    if name.startswith("exp_T"):
        return RewardTransformSpec(name, "boltzmann", float(name[5:]), log_floor)
    if name.startswith("softplus_T"):
        return RewardTransformSpec(name, "softplus", float(name[10:]), log_floor)
    raise ValueError(f"unknown reward transform '{name}'")


def _log_softplus(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    hi = x >= 30.0
    lo = x <= -30.0
    mid = ~(hi | lo)
    out[hi] = np.log(x[hi])
    out[lo] = x[lo]                     # softplus(x) ~ e^x
    out[mid] = np.log(np.log1p(np.exp(x[mid])))
    return out


@dataclass
class LogRewardResult:
    log_r: np.ndarray        # [num_items] float64, un-normalized log reward
    log_z: float             # logsumexp(log_r)
    stats: dict[str, Any]

    @property
    def log_p_target(self) -> np.ndarray:
        return self.log_r - self.log_z

    @property
    def p_target(self) -> np.ndarray:
        """Linear target; entries below exp(-745) underflow to 0.0, which is
        harmless for TV/JS/ranking metrics (documented, not an error)."""
        return np.exp(self.log_p_target)


def log_rewards(raw_scores: np.ndarray, spec: RewardTransformSpec) -> LogRewardResult:
    raw = np.asarray(raw_scores, dtype=np.float64)
    shift = float(raw.max())  # numerically stable user-specific b(c)
    if spec.kind == "boltzmann":
        log_r = (raw - shift) / spec.temperature
    elif spec.kind == "softplus":
        log_r = _log_softplus(raw / spec.temperature)
    else:
        raise ValueError(spec.kind)

    n_underflow_prevented = int((log_r < LOG_FLOAT64_TINY).sum())
    n_floor = 0
    if spec.log_floor is not None:
        n_floor = int((log_r < spec.log_floor).sum())
        log_r = np.maximum(log_r, spec.log_floor)

    m = float(log_r.max())
    log_z = m + float(np.log(np.exp(log_r - m).sum()))
    log_p = log_r - log_z
    # entropy via linear probs (tail below underflow contributes ~0)
    p = np.exp(log_p)
    nz = p > 0
    entropy = float(-(p[nz] * log_p[nz]).sum())
    stats = {
        "raw_score_min": float(raw.min()), "raw_score_max": float(raw.max()),
        "log_reward_min": float(log_r.min()), "log_reward_max": float(log_r.max()),
        "reward_floor": spec.log_floor,
        "num_floor_clamped": n_floor,
        "num_underflow_prevented": n_underflow_prevented,
        "target_entropy": entropy,
        "effective_support": float(np.exp(entropy)),
        "shift_b": shift,
    }
    return LogRewardResult(log_r=log_r, log_z=log_z, stats=stats)


# --------------------------------------------------------------------------- #
# Log-space prefix masses over the trie
# --------------------------------------------------------------------------- #
def segment_logsumexp(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """logsumexp over contiguous segments; -inf-safe.

    Segments must tile ``values[starts[0] : starts[0] + counts.sum()]``
    contiguously (true for trie children and per-level slices).
    """
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    base = int(starts[0]) if len(starts) else 0
    seg = np.asarray(values[base: base + int(counts.sum())], dtype=np.float64)
    rel = starts - base
    m = np.maximum.reduceat(seg, rel)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    shifted = np.exp(seg - np.repeat(safe_m, counts))
    sums = np.add.reduceat(shifted, rel)
    with np.errstate(divide="ignore"):
        out = safe_m + np.log(sums)
    out[~np.isfinite(m)] = -np.inf
    return out


def log_prefix_masses(trie: SemanticIDTrie, log_r: np.ndarray) -> np.ndarray:
    """log M(s) for every node: bottom-up per-level segment logsumexp.

    Exact analogue of ``reward_mass`` without any underflow: the log root mass
    equals log Z and leaf masses equal the item log rewards.
    """
    log_r = np.asarray(log_r, dtype=np.float64)
    if log_r.shape != (trie.num_items,):
        raise ValueError(f"log_r must have shape ({trie.num_items},)")
    log_mass = np.full(trie.num_nodes, -np.inf, dtype=np.float64)
    term = trie.terminal_node_ids
    log_mass[term] = log_r[trie.item_row[term]]
    for depth in range(trie.sid_len - 1, -1, -1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        lo = trie.level_offsets[depth + 1]
        log_mass[ids] = segment_logsumexp(
            log_mass[lo: trie.level_offsets[depth + 2]],
            trie.child_start[ids] - lo, trie.child_count[ids])
    return log_mass


def log_conditionals(trie: SemanticIDTrie, log_mass: np.ndarray) -> np.ndarray:
    """log q_R(v | parent) stored at the child node; root = 0."""
    log_q = np.zeros(trie.num_nodes, dtype=np.float64)
    log_q[1:] = log_mass[1:] - log_mass[trie.parent[1:]]
    return log_q


def check_log_conditionals(trie: SemanticIDTrie, log_q: np.ndarray, atol: float = 1e-9) -> float:
    internal = trie.internal_node_ids
    sums = segment_logsumexp(log_q, trie.child_start[internal], trie.child_count[internal])
    err = float(np.abs(sums).max())
    if err > atol:
        raise AssertionError(f"log conditionals do not normalize (max |logsum| {err:.3e})")
    return err


# --------------------------------------------------------------------------- #
# Global hybrid terminal target (spec §24.2)
# --------------------------------------------------------------------------- #
def hybrid_log_target(
    log_p0: np.ndarray, raw_scores: np.ndarray, *, alpha: float, temperature: float,
) -> LogRewardResult:
    """log w = (1-alpha) log p0(x|c) + alpha (r(c,x) - b(c)) / T, normalized.

    ``log_p0`` is the exact constrained TIGER terminal log-probability over the
    catalog. alpha=0 reproduces p0; alpha=1 reproduces the Boltzmann target.
    """
    log_p0 = np.asarray(log_p0, dtype=np.float64)
    raw = np.asarray(raw_scores, dtype=np.float64)
    log_w = (1.0 - alpha) * log_p0 + alpha * (raw - raw.max()) / temperature
    m = float(log_w.max())
    log_z = m + float(np.log(np.exp(log_w - m).sum()))
    log_p = log_w - log_z
    p = np.exp(log_p)
    nz = p > 0
    entropy = float(-(p[nz] * log_p[nz]).sum())
    stats = {
        "alpha": alpha, "temperature": temperature,
        "log_reward_min": float(log_w.min()), "log_reward_max": float(log_w.max()),
        "reward_floor": None, "num_floor_clamped": 0,
        "num_underflow_prevented": int((log_w - log_z < LOG_FLOAT64_TINY).sum()),
        "target_entropy": entropy, "effective_support": float(np.exp(entropy)),
    }
    return LogRewardResult(log_r=log_w, log_z=log_z, stats=stats)


def hybrid_target_name(alpha: float, temperature: float) -> str:
    return f"hybrid_a{alpha:g}_T{temperature:g}"
