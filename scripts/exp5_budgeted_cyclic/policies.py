"""Policy taxonomy, canonical configuration hashing, and per-user evaluation.

Families follow the corrected taxonomy (spec §6, §12):

  reference/tiger_exact_full_catalog       analytical full-catalog ranking
  reference/append_only_cyclic_parameterization   B=0 cyclic DP (must equal it)
  reference/tiger_beam                     constrained trie beam search
  reference/tiger_stochastic               EXACT distribution of the decoder-
                                           temperature sampler (serving stage
                                           samples it stochastically)
  reference/sasrec_full_catalog            external reward-model reference
  best_of_n / beam_rerank                  simple test-time baselines
  local_guidance                           local_prefix_guidance (spec 24.1)
  global_hybrid_target                     exact global hybrid ranking (24.2)
  truncated_infinite_oracle                infinite-horizon oracle + hard cap
  scaled_truncated_infinite_oracle         sigma(logit d_inf - lambda) + cap
  suffix_only_rollback                     rollback at the collision level only
  level_budget_rollback                    level masks and level-reserved quotas
  finite_budget_controller                 shared budget-aware d_eta (23.1)
  no_repeat_heuristic                      serving-only Monte Carlo policy

The canonical configuration key contains ONLY parameters that affect the
method (spec §7): transform-independent methods carry ``transform=None`` and
are stored exactly once.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from flowrec.trajectories.nonacyclic import SemanticIDTrie, exact_terminal_distribution

from . import finite_budget_dp as dp
from .compute_counters import ComputeCounters
from .metrics import (
    alignment_metrics,
    candidate_prediction,
    dp_behavior_fields,
    full_catalog_prediction,
    static_behavior_fields,
)
from .oracle import build_log_flow_oracle, rollback_for_dp
from .rewards import (
    hybrid_log_target,
    log_conditionals,
    log_prefix_masses,
    log_rewards,
    parse_transform,
)
from .tiger_scoring import exact_item_logprobs, node_entropy_and_margin

REFERENCE_FAMILIES = (
    "reference/tiger_exact_full_catalog",
    "reference/append_only_cyclic_parameterization",
    "reference/tiger_beam",
    "reference/tiger_stochastic",
    "reference/sasrec_full_catalog",
)
PRIMARY_REFERENCE = "reference/append_only_cyclic_parameterization"


@dataclass(frozen=True)
class PolicyConfig:
    family: str
    params: tuple[tuple[str, Any], ...] = ()
    transform: str | None = None      # reward transform, None if independent
    target: str | None = None         # 'sasrec' | 'hybrid_a{a}_T{T}' | None

    def param(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)

    @property
    def target_name(self) -> str | None:
        """Concrete target-context key: transform name or hybrid name."""
        if self.target == "sasrec":
            return self.transform
        return self.target

    def canonical(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "params": {k: v for k, v in sorted(self.params)},
            "transform": self.transform,
            "target": self.target,
        }

    def canonical_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    @classmethod
    def from_canonical(cls, canonical: Mapping[str, Any]) -> "PolicyConfig":
        params = []
        for k, v in sorted(dict(canonical.get("params", {})).items()):
            if isinstance(v, list):
                v = tuple(int(x) for x in v)
            params.append((k, v))
        return cls(family=canonical["family"], params=tuple(params),
                   transform=canonical.get("transform"),
                   target=canonical.get("target"))

    @property
    def name(self) -> str:
        bits = []
        for k, v in sorted(self.params):
            if isinstance(v, (list, tuple)):
                bits.append(f"{k}_" + "_".join(str(x) for x in v))
            else:
                bits.append(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}")
        if self.target_name:
            bits.append(self.target_name)
        tail = "__".join(bits)
        base = self.family.split("/")[-1]
        return f"{base}__{tail}" if tail else base

    @property
    def family_dir(self) -> str:
        return self.family.replace("/", "__")


def assert_no_duplicate_configs(configs: list[PolicyConfig]) -> None:
    """Duplicated canonical configurations must never enter a table (spec §7)."""
    seen: dict[str, PolicyConfig] = {}
    for c in configs:
        h = c.canonical_hash()
        if h in seen:
            raise ValueError(
                f"duplicate canonical config: {c.canonical()} == {seen[h].canonical()}")
        seen[h] = c


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #
def build_reference_grid(grid: Mapping[str, Any]) -> list[PolicyConfig]:
    out = [
        PolicyConfig("reference/tiger_exact_full_catalog"),
        PolicyConfig("reference/append_only_cyclic_parameterization"),
        PolicyConfig("reference/sasrec_full_catalog"),
    ]
    for k in grid.get("beams", [10, 20, 50]):
        out.append(PolicyConfig("reference/tiger_beam", (("beam", int(k)),)))
    for tau in grid.get("taus", [0.7, 1.0, 1.5]):
        out.append(PolicyConfig("reference/tiger_stochastic", (("tau", float(tau)),)))
    for n in grid.get("best_of_n", [2, 4, 8]):
        out.append(PolicyConfig("best_of_n", (("n", int(n)),)))
    for k in grid.get("rerank_beams", [20, 50]):
        out.append(PolicyConfig("beam_rerank", (("beam", int(k)),)))
    return out


def build_validation_grid(grid: Mapping[str, Any]) -> list[PolicyConfig]:
    transforms = list(grid.get("transforms", []))
    out = build_reference_grid(grid)

    for t in transforms:
        for a in grid.get("alpha_local", []):
            out.append(PolicyConfig(
                "local_guidance", (("alpha", float(a)),), transform=t, target="sasrec"))
        for b in grid.get("budgets", []):
            if b == 0:
                continue  # B=0 is the append-only reference, stored once
            out.append(PolicyConfig(
                "truncated_infinite_oracle", (("budget", int(b)),),
                transform=t, target="sasrec"))
        for lam in grid.get("lambda_grid", []):
            for b in grid.get("scaled_budgets", []):
                out.append(PolicyConfig(
                    "scaled_truncated_infinite_oracle",
                    (("budget", int(b)), ("lam", float(lam))),
                    transform=t, target="sasrec"))
        for b in grid.get("suffix_budgets", []):
            out.append(PolicyConfig(
                "suffix_only_rollback", (("budget", int(b)),),
                transform=t, target="sasrec"))
        for mask in grid.get("level_masks", []):
            for b in grid.get("level_mask_budgets", grid.get("suffix_budgets", [])):
                out.append(PolicyConfig(
                    "level_budget_rollback",
                    (("mode", "mask"), ("levels", str(mask)), ("budget", int(b))),
                    transform=t, target="sasrec"))
        for alloc in _allocations(grid.get("level_allocations", {})):
            out.append(PolicyConfig(
                "level_budget_rollback",
                (("mode", "alloc"), ("allocation", tuple(alloc))),
                transform=t, target="sasrec"))

    hyb = grid.get("hybrid", {})
    for a in hyb.get("alpha", []):
        for T in hyb.get("T", []):
            tname = f"hybrid_a{a:g}_T{T:g}"
            out.append(PolicyConfig(
                "global_hybrid_target", (("alpha", float(a)), ("T", float(T))),
                target=tname))
            for b in grid.get("hybrid_rollback_budgets", []):
                out.append(PolicyConfig(
                    "truncated_infinite_oracle", (("budget", int(b)),), target=tname))

    ctrl = grid.get("controller")
    if ctrl:
        for t in ctrl.get("transforms", transforms[:1]):
            out.append(PolicyConfig(
                "finite_budget_controller", (("budget", int(ctrl["budget"])),),
                transform=t, target="sasrec"))
    assert_no_duplicate_configs(out)
    return out


def _allocations(spec: Mapping[str, Any]) -> list[tuple[int, ...]]:
    from itertools import product

    out: list[tuple[int, ...]] = []
    for total, allocs in spec.items():
        if allocs == "all_compositions":
            for v in product(range(int(total) + 1), repeat=4):
                if sum(v) == int(total):
                    out.append(tuple(v))
        else:
            for v in allocs:
                if sum(v) != int(total):
                    raise ValueError(f"allocation {v} does not sum to {total}")
                out.append(tuple(int(x) for x in v))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Per-user context (lazy target construction with its own counter row)
# --------------------------------------------------------------------------- #
@dataclass
class TargetContext:
    name: str
    log_reward: Any
    log_mass: np.ndarray
    log_q: np.ndarray
    oracle: Any | None
    oracle_error: str | None
    d_dp: np.ndarray | None
    clip_info: dict[str, Any] = field(default_factory=dict)

    @property
    def p_target(self) -> np.ndarray:
        return self.log_reward.p_target

    @property
    def log_p_target(self) -> np.ndarray:
        return self.log_reward.log_p_target


class UserPolicyContext:
    def __init__(
        self, trie: SemanticIDTrie, *, dataset_index: int, raw_user_id: str,
        target_row: int, edge_logprob: np.ndarray, raw_scores: np.ndarray,
        target_counters: ComputeCounters | None = None,
    ):
        self.trie = trie
        self.dataset_index = int(dataset_index)
        self.raw_user_id = str(raw_user_id)
        self.target_row = int(target_row)
        self.edge_logprob = np.asarray(edge_logprob, dtype=np.float64)
        self.raw_scores = np.asarray(raw_scores, dtype=np.float64)
        self.target_counters = target_counters
        self._edge_prob: np.ndarray | None = None
        self._log_p0: np.ndarray | None = None
        self._entropy: np.ndarray | None = None
        self._targets: dict[str, TargetContext] = {}

    @property
    def edge_prob(self) -> np.ndarray:
        if self._edge_prob is None:
            self._edge_prob = np.exp(self.edge_logprob)
            self._edge_prob[0] = 1.0
        return self._edge_prob

    @property
    def log_p0(self) -> np.ndarray:
        if self._log_p0 is None:
            self._log_p0 = exact_item_logprobs(self.trie, self.edge_logprob)
        return self._log_p0

    @property
    def p0(self) -> np.ndarray:
        p = np.exp(self.log_p0)
        return p / p.sum()

    @property
    def entropy(self) -> np.ndarray:
        if self._entropy is None:
            self._entropy, self._margin = node_entropy_and_margin(
                self.trie, self.edge_logprob)
        return self._entropy

    def get_target(self, name: str) -> TargetContext:
        if name in self._targets:
            return self._targets[name]
        t0 = time.time()
        if name.startswith("hybrid_a"):
            body = name[len("hybrid_a"):]
            a_str, t_str = body.split("_T")
            lr = hybrid_log_target(
                self.log_p0, self.raw_scores,
                alpha=float(a_str), temperature=float(t_str))
        else:
            lr = log_rewards(self.raw_scores, parse_transform(name))
        log_mass = log_prefix_masses(self.trie, lr.log_r)
        log_q = log_conditionals(self.trie, log_mass)
        oracle, err, d_dp, clip_info = None, None, None, {}
        try:
            oracle = build_log_flow_oracle(self.trie, self.edge_logprob, log_q, log_mass)
            d_dp, clip_info = rollback_for_dp(oracle)
        except (ValueError, AssertionError) as exc:
            err = str(exc)
        ctx = TargetContext(name=name, log_reward=lr, log_mass=log_mass, log_q=log_q,
                            oracle=oracle, oracle_error=err, d_dp=d_dp,
                            clip_info=clip_info)
        self._targets[name] = ctx
        if self.target_counters is not None:
            self.target_counters.add(
                prefix_mass_nodes_processed=2 * self.trie.num_nodes,
                sasrec_full_catalog_passes=0,  # cached; accounted at cache build
                wall_time_policy_eval_s=time.time() - t0,
            )
        return ctx


# --------------------------------------------------------------------------- #
# Evaluation dispatch
# --------------------------------------------------------------------------- #
def evaluate_policy(
    cfg: PolicyConfig,
    u: UserPolicyContext,
    *,
    controllers: Mapping[tuple[str, int], Any] | None = None,
    counters: ComputeCounters | None = None,
) -> dict[str, Any]:
    trie = u.trie
    L = trie.sid_len
    t0 = time.time()
    row: dict[str, Any] = {
        "user": u.raw_user_id, "dataset_index": u.dataset_index,
        "policy_family": cfg.family, "config_name": cfg.name,
        "canonical_config_hash": cfg.canonical_hash(),
        "reward_transform": cfg.transform or "none",
        "target": cfg.target_name or "none",
        "requires_full_catalog_reward": cfg.target_name is not None,
        "status": "ok",
        # grid parameters as typed columns (plots/selection sensitivity)
        "budget": (float(cfg.param("budget"))
                   if cfg.param("budget") is not None else np.nan),
        "alpha": (float(cfg.param("alpha"))
                  if cfg.param("alpha") is not None else np.nan),
        "tau": float(cfg.param("tau")) if cfg.param("tau") is not None else np.nan,
        "lam": float(cfg.param("lam")) if cfg.param("lam") is not None else np.nan,
        "beam": float(cfg.param("beam")) if cfg.param("beam") is not None else np.nan,
        "n_samples": float(cfg.param("n")) if cfg.param("n") is not None else np.nan,
        "allocation": (",".join(str(x) for x in cfg.param("allocation"))
                       if cfg.param("allocation") is not None else None),
    }
    fam = cfg.family
    try:
        if fam == "reference/tiger_exact_full_catalog":
            row.update(full_catalog_prediction(u.log_p0, u.target_row))
            row.update(static_behavior_fields(L))
            _count(counters, tiger_item_trajectory_scores=trie.num_items)

        elif fam == "reference/append_only_cyclic_parameterization":
            res = dp.exact_budgeted_terminal_distribution(
                trie, u.edge_prob, np.zeros(trie.num_nodes), 0)
            row.update(full_catalog_prediction(res.terminal, u.target_row))
            row.update(dp_behavior_fields(res, L))
            # must be numerically identical to the exact TIGER ranking (spec 6)
            tv = 0.5 * float(np.abs(res.terminal - u.p0).sum())
            row["check_b0_equals_exact"] = bool(tv < 1e-12)
            if tv >= 1e-12:
                row["status"] = f"b0_mismatch_tv_{tv:.2e}"
            _count(counters, prefix_mass_nodes_processed=trie.num_nodes)

        elif fam == "reference/tiger_beam":
            k = int(cfg.param("beam"))
            rows_arr, _scores = dp.trie_beam_search(trie, u.edge_logprob, k)
            row.update(candidate_prediction(rows_arr, u.target_row, candidate_set_size=k))
            row.update(static_behavior_fields(L))
            row["online_expected_tiger_calls"] = float(L * k)
            _count(counters, tiger_prefix_state_evals=min(k, trie.num_nodes) * L)

        elif fam == "reference/tiger_stochastic":
            tau = float(cfg.param("tau"))
            p_tau = dp.temperature_edge_prob(trie, u.edge_logprob, tau)
            dist = exact_terminal_distribution(trie, p_tau)
            row.update(full_catalog_prediction(dist, u.target_row))
            row.update(static_behavior_fields(L))
            _count(counters, prefix_mass_nodes_processed=trie.num_nodes)

        elif fam == "reference/sasrec_full_catalog":
            row.update(full_catalog_prediction(u.raw_scores, u.target_row))
            row.update(static_behavior_fields(L, steps=0.0))
            row["online_expected_tiger_calls"] = 0.0
            row["requires_full_catalog_reward"] = True
            _count(counters, sasrec_item_scores=trie.num_items)

        elif fam == "best_of_n":
            n = int(cfg.param("n"))
            dist = dp.best_of_n_exact_distribution(u.p0, u.raw_scores, n)
            row.update(full_catalog_prediction(dist, u.target_row))
            row.update(static_behavior_fields(L, steps=float(n * (L + 1))))
            row["online_expected_tiger_calls"] = float(n * L)
            row["requires_full_catalog_reward"] = True
            _count(counters, prefix_mass_nodes_processed=trie.num_items)

        elif fam == "beam_rerank":
            k = int(cfg.param("beam"))
            rows_arr, _ = dp.trie_beam_search(trie, u.edge_logprob, k)
            order = np.argsort(-u.raw_scores[rows_arr], kind="stable")
            row.update(candidate_prediction(
                rows_arr[order], u.target_row, candidate_set_size=k))
            row.update(static_behavior_fields(L))
            row["online_expected_tiger_calls"] = float(L * k)
            row["requires_full_catalog_reward"] = True
            _count(counters, tiger_prefix_state_evals=min(k, trie.num_nodes) * L,
                   sasrec_item_scores=k)

        elif fam == "local_guidance":
            tgt = u.get_target(cfg.target_name)
            alpha = float(cfg.param("alpha"))
            q_lin = np.exp(tgt.log_q)
            p_a = dp.geometric_guidance_edge_prob(trie, u.edge_prob, q_lin, alpha)
            dist = exact_terminal_distribution(trie, p_a)
            row.update(full_catalog_prediction(dist, u.target_row))
            row.update(alignment_metrics(dist, tgt.log_p_target, u.raw_scores))
            row.update(static_behavior_fields(L))
            _count(counters, prefix_mass_nodes_processed=2 * trie.num_nodes)

        elif fam == "global_hybrid_target":
            tgt = u.get_target(cfg.target_name)
            row.update(full_catalog_prediction(tgt.log_p_target, u.target_row))
            row.update(alignment_metrics(tgt.p_target, tgt.log_p_target, u.raw_scores))
            row.update(static_behavior_fields(L, steps=0.0))
            row["online_expected_tiger_calls"] = float(trie.num_items * L)
            _count(counters, tiger_item_trajectory_scores=trie.num_items)

        elif fam in ("truncated_infinite_oracle", "suffix_only_rollback",
                     "scaled_truncated_infinite_oracle", "level_budget_rollback",
                     "finite_budget_controller"):
            tgt = u.get_target(cfg.target_name)
            if tgt.d_dp is None:
                row["status"] = f"oracle_failed: {tgt.oracle_error}"
                return row
            row.update(_evaluate_rollback(cfg, u, tgt, controllers, counters))
        else:
            raise ValueError(f"family '{fam}' has no exact evaluator "
                             "(no_repeat_heuristic is serving-only)")
    finally:
        if counters is not None:
            counters.add(wall_time_policy_eval_s=time.time() - t0)
            counters.n_users += 1
    return row


def _evaluate_rollback(cfg, u, tgt, controllers, counters) -> dict[str, Any]:
    trie = u.trie
    L = trie.sid_len
    fam = cfg.family
    d = tgt.d_dp
    if fam == "scaled_truncated_infinite_oracle":
        d = dp.scaled_rollback_prob(d, float(cfg.param("lam")))
    elif fam == "suffix_only_rollback":
        d = dp.level_masked_rollback(trie, d, {L})
    elif fam == "level_budget_rollback" and cfg.param("mode") == "mask":
        d = dp.level_masked_rollback(trie, d, _mask_levels(str(cfg.param("levels")), L))

    if fam == "level_budget_rollback" and cfg.param("mode") == "alloc":
        res = dp.exact_level_budgeted_terminal_distribution(
            trie, u.edge_prob, d, cfg.param("allocation"))
        n_slices = int(np.prod([a + 1 for a in cfg.param("allocation")]))
    elif fam == "finite_budget_controller":
        budget = int(cfg.param("budget"))
        art = (controllers or {}).get((cfg.target_name, budget))
        if art is None:
            return {"status": "controller_artifact_missing"}
        feats = dp.controller_features(
            trie, tgt.d_dp, tgt.oracle.log_C, u.entropy)
        d_rows = art.rollback_rows(trie, feats)
        res = dp.exact_budgeted_terminal_distribution(trie, u.edge_prob, d_rows, budget)
        n_slices = budget + 1
    else:
        budget = int(cfg.param("budget"))
        res = dp.exact_budgeted_terminal_distribution(trie, u.edge_prob, d, budget)
        n_slices = budget + 1

    out = full_catalog_prediction(res.terminal, u.target_row)
    out.update(alignment_metrics(res.terminal, tgt.log_p_target, u.raw_scores))
    out.update(dp_behavior_fields(res, L))
    out["n_d_star_impl_clipped"] = int(tgt.clip_info.get("n_impl_clipped", 0))
    _count(counters, prefix_mass_nodes_processed=n_slices * trie.num_nodes)
    return out


def user_context_from_caches(
    trie: SemanticIDTrie, user_row: Mapping[str, Any],
    tiger_reader: Any, sasrec_reader: Any, item_to_row: np.ndarray,
    *, target_counters: ComputeCounters | None = None,
) -> tuple[UserPolicyContext, np.ndarray]:
    """Build a UserPolicyContext from the stage-02/03 caches.

    Returns (context, raw_valid_mass[num_internal]) — the latter feeds the
    gate features. Edge log-probs are renormalized to float64 on load.
    """
    from .tiger_scoring import renormalized_edge_logprob

    ds_idx = int(user_row["dataset_index"])
    tz, trow = tiger_reader.arrays_for(ds_idx)
    sz, srow = sasrec_reader.arrays_for(ds_idx)
    edge_lp = renormalized_edge_logprob(trie, tz["edge_logprob"][trow])
    raw_scores = sz["raw_scores"][srow].astype(np.float64)
    target_item = int(user_row["target_item_id"])
    target_row = int(item_to_row[target_item])
    if target_row < 0:
        raise AssertionError(
            f"user {user_row['raw_user_id']} marked eligible but target item "
            f"{target_item} has no semantic row")
    ctx = UserPolicyContext(
        trie,
        dataset_index=ds_idx,
        raw_user_id=str(user_row["raw_user_id"]),
        target_row=target_row,
        edge_logprob=edge_lp,
        raw_scores=raw_scores,
        target_counters=target_counters,
    )
    ctx.target_item_id = target_item
    return ctx, np.asarray(tz["raw_valid_mass"][trow], dtype=np.float64)


def _mask_levels(spec: str, L: int) -> set[int]:
    if spec == "all":
        return set(range(1, L + 1))
    if spec == "semantic_L1_L3":
        return set(range(1, L))
    if spec.startswith("L") and spec[1:].isdigit():
        return {int(spec[1:])}
    raise ValueError(f"unknown level mask '{spec}'")


def _count(counters: ComputeCounters | None, **kw: float) -> None:
    if counters is not None:
        counters.add(**kw)
