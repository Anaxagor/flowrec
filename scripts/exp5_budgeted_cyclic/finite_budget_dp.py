"""Finite-budget DP wrappers and the budget-aware controller (spec §21-23).

The exact DPs live in ``flowrec.trajectories.budgeted`` (unit-tested there):
scalar hard budget with optionally budget-dependent d(c, s, b), and the
level-reserved quota DP over (s, b_1..b_L). This module adds the shared
low-dimensional budget-aware controller

    d_eta(c, s, b) = sigmoid( eta[b, l] . f(c, s) ),   l = depth(s),
    f = [1, logit d_inf, log C(parent(s)), log C(s), H_theta(c, s)]

fitted on VALIDATION users only by differentiating through the exact DP
(objective: JS to the target + lambda_steps * E[n_tau]), plus the per-user
free-logit oracle upper bound d(c, s, b) for a bounded diagnostic subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flowrec.trajectories.budgeted import (  # noqa: F401  (re-exports)
    BudgetedCyclicSampler,
    BudgetedTerminalResult,
    LevelBudgetedTerminalResult,
    best_of_n_exact_distribution,
    exact_budgeted_terminal_distribution,
    exact_level_budgeted_terminal_distribution,
    geometric_guidance_edge_prob,
    level_masked_rollback,
    scaled_rollback_prob,
    temperature_edge_prob,
    trie_beam_search,
)
from flowrec.trajectories.nonacyclic import SemanticIDTrie

CONTROLLER_FEATURES = ["bias", "logit_d_inf", "log_C_parent", "log_C_self", "entropy"]


def controller_features(
    trie: SemanticIDTrie, d_inf: np.ndarray, log_C: np.ndarray, entropy: np.ndarray,
) -> np.ndarray:
    """[num_nodes, 5] float64 feature matrix; inference-available under the
    offline-cache regime (d_inf and log_C need the full-catalog reward)."""
    n = trie.num_nodes
    f = np.zeros((n, len(CONTROLLER_FEATURES)), dtype=np.float64)
    f[:, 0] = 1.0
    d = np.clip(d_inf, 1e-9, 1 - 1e-9)
    f[:, 1] = np.clip(np.log(d) - np.log1p(-d), -30.0, 30.0)
    f[:, 2] = np.clip(log_C[np.maximum(trie.parent, 0)], -30.0, 700.0)
    f[:, 3] = np.clip(log_C, -30.0, 700.0)
    f[:, 4] = entropy
    return f


@dataclass
class ControllerArtifact:
    eta: np.ndarray            # [budget+1, sid_len+1, n_features]
    budget: int
    target_name: str
    fit_dataset_indices: list[int]
    history: list[dict[str, float]]

    def rollback_rows(self, trie: SemanticIDTrie, features: np.ndarray) -> np.ndarray:
        """[budget+1, num_nodes] d(c, s, b); row 0 = 0, root column = 0."""
        rows = np.zeros((self.budget + 1, trie.num_nodes), dtype=np.float64)
        lvl = trie.level.astype(np.int64)
        for b in range(1, self.budget + 1):
            z = np.einsum("nf,nf->n", features, self.eta[b, lvl])
            rows[b] = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        rows[:, 0] = 0.0
        rows = np.minimum(rows, 1.0 - 1e-9)
        return rows

    def save(self, path: Any) -> None:
        from .common import atomic_write_npz
        atomic_write_npz(
            path, eta=self.eta, budget=np.asarray([self.budget]),
            fit_dataset_indices=np.asarray(self.fit_dataset_indices, dtype=np.int64),
            target_name=np.frombuffer(self.target_name.encode(), dtype=np.uint8))

    @classmethod
    def load(cls, path: Any) -> "ControllerArtifact":
        z = np.load(path)
        return cls(
            eta=z["eta"].astype(np.float64), budget=int(z["budget"][0]),
            target_name=bytes(z["target_name"]).decode(),
            fit_dataset_indices=[int(x) for x in z["fit_dataset_indices"]],
            history=[])


def _torch_dp(torch, trie_idx, p, d_rows, budget, num_items):
    """Differentiable scalar-budget DP; mirrors the numpy implementation."""
    level_slices, parent, item_row, term_lo = trie_idx
    n = parent.shape[0]
    terminal = torch.zeros(num_items, dtype=torch.float64, device=p.device)
    exp_del = torch.zeros((), dtype=torch.float64, device=p.device)
    exp_add = torch.zeros((), dtype=torch.float64, device=p.device)
    mass_in = torch.zeros(n, dtype=torch.float64, device=p.device)
    mass_in = mass_in.index_put(
        (torch.tensor([0], device=p.device),),
        torch.tensor([1.0], dtype=torch.float64, device=p.device))
    zero = torch.zeros(n, dtype=torch.float64, device=p.device)
    for b in range(budget, -1, -1):
        d_eff = d_rows[b] if b > 0 else zero
        reach = mass_in
        for lo, hi in level_slices:
            ids = torch.arange(lo, hi, device=p.device)
            par = parent[ids]
            reach = reach.index_add(0, ids, reach[par] * (1.0 - d_eff[par]) * p[lo:hi])
        keep = reach * (1.0 - d_eff)
        terminal = terminal.index_add(0, item_row, keep[term_lo:])
        exp_add = exp_add + keep[:term_lo].sum()
        if b > 0:
            delete_out = reach * d_eff
            exp_del = exp_del + delete_out[1:].sum()
            nxt = torch.zeros(n, dtype=torch.float64, device=p.device)
            nxt = nxt.index_add(0, parent[1:], delete_out[1:])
            mass_in = nxt
    return terminal, exp_del, exp_add


def _trie_idx(torch, trie: SemanticIDTrie, device):
    level_slices = [
        (int(trie.level_offsets[k]), int(trie.level_offsets[k + 1]))
        for k in range(1, trie.sid_len + 1)
    ]
    parent = torch.as_tensor(trie.parent, device=device)
    item_row = torch.as_tensor(trie.item_row[trie.terminal_node_ids], device=device)
    term_lo = int(trie.level_offsets[trie.sid_len])
    return level_slices, parent, item_row, term_lo


def fit_budget_controller(
    trie: SemanticIDTrie,
    users: list[dict[str, np.ndarray]],
    *,
    budget: int,
    target_name: str,
    steps: int = 200,
    lr: float = 0.05,
    lambda_steps: float = 0.0,
    seed: int = 42,
    device: str = "cpu",
    init_scale: float = 0.0,
) -> ControllerArtifact:
    """Fit eta on validation users only (spec §23.1).

    ``users``: dicts with 'edge_prob' [n] linear, 'features' [n, 5],
    'p_target' [num_items] linear, 'dataset_index'.
    """
    import torch

    torch.manual_seed(seed)
    dev = torch.device(device)
    L = trie.sid_len
    n_feat = len(CONTROLLER_FEATURES)
    eta = torch.full((budget + 1, L + 1, n_feat), 0.0, dtype=torch.float64,
                     device=dev, requires_grad=True)
    with torch.no_grad():
        eta[:, :, 0] = -2.0 + init_scale
    opt = torch.optim.Adam([eta], lr=lr)
    trie_idx = _trie_idx(torch, trie, dev)
    lvl = torch.as_tensor(trie.level.astype(np.int64), device=dev)

    tensors = [
        {
            "p": torch.as_tensor(u["edge_prob"], dtype=torch.float64, device=dev),
            "f": torch.as_tensor(u["features"], dtype=torch.float64, device=dev),
            "t": torch.as_tensor(u["p_target"], dtype=torch.float64, device=dev),
        }
        for u in users
    ]
    history: list[dict[str, float]] = []
    eps = 1e-30
    for step in range(steps):
        opt.zero_grad()
        total_js = torch.zeros((), dtype=torch.float64, device=dev)
        total_steps = torch.zeros((), dtype=torch.float64, device=dev)
        for u in tensors:
            d_rows = torch.sigmoid(
                torch.einsum("bnf,nf->bn", eta[:, lvl, :], u["f"]))
            d_rows = d_rows.clamp(max=1.0 - 1e-9)
            d_rows = torch.cat(
                [torch.zeros_like(d_rows[:, :1]), d_rows[:, 1:]], dim=1)
            terminal, exp_del, exp_add = _torch_dp(
                torch, trie_idx, u["p"], d_rows, budget, trie.num_items)
            t = u["t"]
            m = 0.5 * (terminal + t)
            js = 0.5 * (terminal * (torch.log(terminal + eps) - torch.log(m + eps))).sum() \
                + 0.5 * (t * (torch.log(t + eps) - torch.log(m + eps))).sum()
            total_js = total_js + js
            total_steps = total_steps + exp_add + 2.0 * exp_del + 1.0
        loss = (total_js + lambda_steps * total_steps) / max(len(tensors), 1)
        loss.backward()
        opt.step()
        history.append({"step": step, "mean_js": float(total_js) / max(len(tensors), 1),
                        "mean_steps": float(total_steps) / max(len(tensors), 1),
                        "loss": float(loss)})
    return ControllerArtifact(
        eta=eta.detach().cpu().numpy(), budget=budget, target_name=target_name,
        fit_dataset_indices=[int(u["dataset_index"]) for u in users],
        history=history)


def fit_free_rollback_logits(
    trie: SemanticIDTrie,
    edge_prob: np.ndarray,
    p_target: np.ndarray,
    *,
    budget: int,
    expected_deletes_cap: float | None = None,
    init_rollback: np.ndarray | None = None,
    steps: int = 200,
    lr: float = 0.05,
    penalty: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Per-user free d(c, s, b) oracle upper bound (spec §23.2; diagnostic
    subset only — never a serving method)."""
    import torch

    torch.manual_seed(seed)
    dev = torch.device(device)
    n = trie.num_nodes
    if init_rollback is not None:
        d0 = np.clip(np.asarray(init_rollback, dtype=np.float64), 1e-6, 1 - 1e-6)
        theta0 = np.tile(np.log(d0) - np.log1p(-d0), (budget + 1, 1))
    else:
        theta0 = np.full((budget + 1, n), -2.0)
    theta = torch.tensor(theta0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    trie_idx = _trie_idx(torch, trie, dev)
    p = torch.as_tensor(np.asarray(edge_prob, dtype=np.float64), device=dev)
    t = torch.as_tensor(np.asarray(p_target, dtype=np.float64), device=dev)
    t = t / t.sum()
    eps = 1e-30
    history = []
    for step in range(steps):
        opt.zero_grad()
        d_rows = torch.sigmoid(theta).clamp(max=1.0 - 1e-9)
        d_rows = torch.cat([torch.zeros_like(d_rows[:, :1]), d_rows[:, 1:]], dim=1)
        terminal, exp_del, _ = _torch_dp(torch, trie_idx, p, d_rows, budget, trie.num_items)
        m = 0.5 * (terminal + t)
        js = 0.5 * (terminal * (torch.log(terminal + eps) - torch.log(m + eps))).sum() \
            + 0.5 * (t * (torch.log(t + eps) - torch.log(m + eps))).sum()
        loss = js
        if expected_deletes_cap is not None:
            loss = loss + penalty * torch.relu(exp_del - expected_deletes_cap)
        loss.backward()
        opt.step()
        history.append({"step": step, "js": float(js), "expected_deletes": float(exp_del)})
    with torch.no_grad():
        d_rows = torch.sigmoid(theta).clamp(max=1.0 - 1e-9)
        d_rows[:, 0] = 0.0
        d_rows[0] = 0.0
    return d_rows.cpu().numpy(), history
