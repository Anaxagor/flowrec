"""Analytical circulation and sampling on the bidirectional SID prefix trie."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from flowrec.ft_methods.gflow.exact_flows import CatalogPrefixFlowIndex


APPEND_ACTION = 0
DELETE_ACTION = 1
STOP_ACTION = 2
INVALID_ACTION = -1


@dataclass(slots=True)
class CyclicSemanticTrajectoryBatch:
    """Variable-length trajectories in a bidirectional semantic-prefix trie."""

    state_token_ids: torch.Tensor
    state_depths: torch.Tensor
    action_types: torch.Tensor
    step_mask: torch.Tensor
    terminal_item_row_ids: torch.Tensor
    terminal_token_ids: torch.Tensor
    terminal_rewards: torch.Tensor
    state_base_log_flows: torch.Tensor
    state_log_flows: torch.Tensor
    backward_logprobs: torch.Tensor
    root_backward_logprobs: torch.Tensor
    is_positive: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.terminal_token_ids.size(0))

    @property
    def num_candidates(self) -> int:
        return int(self.terminal_token_ids.size(1))

    @property
    def trajectory_length(self) -> int:
        return int(self.action_types.size(-1))

    @property
    def candidate_token_ids(self) -> torch.Tensor:
        return self.terminal_token_ids

    @property
    def candidate_token_mask(self) -> torch.Tensor:
        return torch.ones_like(self.terminal_token_ids, dtype=torch.bool)


def analytical_delete_probabilities(
    cycle_flow_ratio: float,
    *,
    delete_probability_cap: float | None = None,
) -> tuple[float, float]:
    """Return analytical deletion probabilities, validating an optional safety cap."""
    kappa = float(cycle_flow_ratio)
    if kappa <= 0.0:
        raise ValueError("cycle_flow_ratio must be positive.")
    internal = kappa / (1.0 + 2.0 * kappa)
    leaf = kappa / (1.0 + kappa)
    if delete_probability_cap is not None:
        cap = float(delete_probability_cap)
        if not 0.0 < cap < 0.5:
            raise ValueError("delete_probability_cap must lie in (0, 0.5).")
        if max(internal, leaf) >= cap:
            raise ValueError(
                "delete_probability_cap is below the analytical deletion probability. "
                "Increase the cap or reduce cycle_flow_ratio."
            )
    return internal, leaf


def analytical_state_log_flows(
    base_log_flows: torch.Tensor,
    depths: torch.Tensor,
    *,
    sid_len: int,
    cycle_flow_ratio: float,
) -> torch.Tensor:
    """Convert acyclic subtree flows G into exact cyclic expected-visit flows F."""
    if base_log_flows.shape != depths.shape:
        raise ValueError("base_log_flows and depths must have identical shapes.")
    kappa = float(cycle_flow_ratio)
    if kappa <= 0.0:
        raise ValueError("cycle_flow_ratio must be positive.")
    root_scale = math.log1p(kappa)
    internal_scale = math.log1p(2.0 * kappa)
    scale = torch.where(
        depths.eq(0),
        base_log_flows.new_full((), root_scale),
        base_log_flows.new_full((), internal_scale),
    )
    scale = torch.where(
        depths.eq(int(sid_len)),
        base_log_flows.new_full((), root_scale),
        scale,
    )
    return base_log_flows + scale


@torch.no_grad()
def sample_analytical_cyclic_trajectories(
    *,
    flow_index: CatalogPrefixFlowIndex,
    dense_rewards: torch.Tensor,
    item_tokens: torch.Tensor,
    num_trajectories: int,
    cycle_flow_ratio: float,
    max_steps: int,
    reward_epsilon: float,
    delete_probability_cap: float | None = None,
    generator: torch.Generator | None = None,
) -> CyclicSemanticTrajectoryBatch:
    """Sample exact off-policy trajectories from an analytical cyclic flow."""
    if dense_rewards.ndim != 2:
        raise ValueError("dense_rewards must have shape [B, num_items].")
    if item_tokens.ndim != 2:
        raise ValueError("item_tokens must have shape [num_items, sid_len].")
    if int(dense_rewards.size(1)) != int(item_tokens.size(0)):
        raise ValueError("Reward catalogue size must match item_tokens.")
    if int(item_tokens.size(1)) != int(flow_index.sid_len):
        raise ValueError("item_tokens SID length must match flow_index.")
    if not flow_index.has_unique_terminal_ids:
        raise ValueError("Cyclic semantic-ID training requires one unique terminal SID per item.")
    if int(num_trajectories) <= 0:
        raise ValueError("num_trajectories must be positive.")
    if int(max_steps) <= int(flow_index.sid_len):
        raise ValueError("max_steps must exceed the acyclic SID trajectory length.")

    kappa = float(cycle_flow_ratio)
    internal_delete_prob, leaf_delete_prob = analytical_delete_probabilities(
        kappa, delete_probability_cap=delete_probability_cap
    )
    device = dense_rewards.device
    dense_rewards = dense_rewards.float().clamp_min(float(reward_epsilon))
    item_tokens = item_tokens.to(device=device).long()
    bsz = int(dense_rewards.size(0))
    num_paths = int(num_trajectories)
    sid_len = int(item_tokens.size(1))
    horizon = int(max_steps)
    csum = flow_index.sorted_reward_cumsum(dense_rewards)

    states = torch.zeros((bsz, num_paths, horizon + 1, sid_len), device=device, dtype=torch.long)
    depths = torch.zeros((bsz, num_paths, horizon + 1), device=device, dtype=torch.long)
    actions = torch.full((bsz, num_paths, horizon), INVALID_ACTION, device=device, dtype=torch.int8)
    step_mask = torch.zeros((bsz, num_paths, horizon), device=device, dtype=torch.bool)
    base_log_flows = torch.zeros((bsz, num_paths, horizon + 1), device=device, dtype=torch.float32)
    state_log_flows = torch.zeros_like(base_log_flows)
    active = torch.ones((bsz, num_paths), device=device, dtype=torch.bool)
    leaf_rows = torch.full((bsz, num_paths), -1, device=device, dtype=torch.long)
    terminal_rows = torch.full_like(leaf_rows, -1)
    terminal_rewards = torch.zeros((bsz, num_paths), device=device, dtype=torch.float32)

    for step in range(horizon):
        current_tokens = states[:, :, step, :]
        current_depths = depths[:, :, step]
        lo, hi = flow_index.prefix_ranges(current_tokens, current_depths)
        base_flows = flow_index.range_sums(csum, lo, hi).clamp_min(float(reward_epsilon))
        current_base_log = base_flows.log().float()
        current_state_log = analytical_state_log_flows(
            current_base_log,
            current_depths,
            sid_len=sid_len,
            cycle_flow_ratio=kappa,
        )
        base_log_flows[:, :, step] = torch.where(active, current_base_log, base_log_flows[:, :, step])
        state_log_flows[:, :, step] = torch.where(active, current_state_log, state_log_flows[:, :, step])
        step_mask[:, :, step] = active

        random_values = torch.rand(active.shape, device=device, generator=generator)
        at_leaf = active & current_depths.eq(sid_len)
        at_internal = active & current_depths.gt(0) & current_depths.lt(sid_len)
        delete = (at_internal & random_values.lt(internal_delete_prob)) | (
            at_leaf & random_values.lt(leaf_delete_prob)
        )
        stop = at_leaf & ~delete
        append = active & ~delete & ~stop

        actions[:, :, step] = torch.where(append, torch.full_like(actions[:, :, step], APPEND_ACTION), actions[:, :, step])
        actions[:, :, step] = torch.where(delete, torch.full_like(actions[:, :, step], DELETE_ACTION), actions[:, :, step])
        actions[:, :, step] = torch.where(stop, torch.full_like(actions[:, :, step], STOP_ACTION), actions[:, :, step])

        next_tokens = current_tokens.clone()
        next_depths = current_depths.clone()
        sampled_rows = flow_index.sample_item_rows_from_ranges(csum, lo, hi, generator=generator, validate=False)
        flat_append = append.reshape(-1).nonzero(as_tuple=False).flatten()
        flat_next_tokens = next_tokens.reshape(-1, sid_len)
        flat_next_depths = next_depths.reshape(-1)
        flat_current_depths = current_depths.reshape(-1)
        flat_sampled_rows = sampled_rows.reshape(-1)
        append_depths = flat_current_depths.index_select(0, flat_append)
        append_rows = flat_sampled_rows.index_select(0, flat_append)
        append_tokens = item_tokens[append_rows, append_depths]
        flat_next_tokens[flat_append, append_depths] = append_tokens
        flat_next_depths[flat_append] = append_depths + 1
        reached_leaf = append_depths.add(1).eq(sid_len)
        leaf_rows.reshape(-1)[flat_append[reached_leaf]] = append_rows[reached_leaf]

        flat_delete = delete.reshape(-1).nonzero(as_tuple=False).flatten()
        deleted_depths = flat_next_depths.index_select(0, flat_delete) - 1
        flat_next_tokens[flat_delete, deleted_depths] = 0
        flat_next_depths[flat_delete] = deleted_depths
        leaf_rows.reshape(-1)[flat_delete] = -1

        terminal_rows = torch.where(stop, leaf_rows, terminal_rows)
        stopped_rewards = dense_rewards.gather(1, leaf_rows.clamp_min(0))
        terminal_rewards = torch.where(stop, stopped_rewards, terminal_rewards)
        terminal_log_rewards = terminal_rewards.clamp_min(float(reward_epsilon)).log()
        base_log_flows[:, :, step + 1] = torch.where(stop, terminal_log_rewards, base_log_flows[:, :, step + 1])
        state_log_flows[:, :, step + 1] = torch.where(stop, terminal_log_rewards, state_log_flows[:, :, step + 1])

        states[:, :, step + 1, :] = next_tokens
        depths[:, :, step + 1] = next_depths
        active = active & ~stop
        if not bool(active.any()):
            break

    if bool(active.any()):
        raise RuntimeError(
            "Analytical cyclic sampler exceeded max_steps={} for {} trajectories; "
            "increase max_steps or reduce cycle_flow_ratio.".format(horizon, int(active.sum()))
        )

    used_steps = int(step_mask.long().sum(dim=-1).max().item())
    states = states[:, :, : used_steps + 1, :]
    depths = depths[:, :, : used_steps + 1]
    actions = actions[:, :, :used_steps]
    step_mask = step_mask[:, :, :used_steps]
    base_log_flows = base_log_flows[:, :, : used_steps + 1]
    state_log_flows = state_log_flows[:, :, : used_steps + 1]

    current_base = base_log_flows[:, :, :-1]
    next_base = base_log_flows[:, :, 1:]
    next_state_flow = state_log_flows[:, :, 1:]
    backward_logprobs = torch.zeros_like(current_base)
    append_mask = step_mask & actions.eq(APPEND_ACTION)
    delete_mask = step_mask & actions.eq(DELETE_ACTION)
    backward_logprobs = torch.where(
        append_mask, next_base + math.log1p(kappa) - next_state_flow, backward_logprobs
    )
    backward_logprobs = torch.where(
        delete_mask, current_base + math.log(kappa) - next_state_flow, backward_logprobs
    )
    root_backward_logprobs = torch.full(
        terminal_rewards.shape, -math.log1p(kappa), device=device, dtype=torch.float32
    )
    terminal_tokens = item_tokens[terminal_rows]
    return CyclicSemanticTrajectoryBatch(
        state_token_ids=states,
        state_depths=depths,
        action_types=actions,
        step_mask=step_mask,
        terminal_item_row_ids=terminal_rows,
        terminal_token_ids=terminal_tokens,
        terminal_rewards=terminal_rewards,
        state_base_log_flows=base_log_flows,
        state_log_flows=state_log_flows,
        backward_logprobs=backward_logprobs,
        root_backward_logprobs=root_backward_logprobs,
        is_positive=torch.zeros_like(terminal_rows, dtype=torch.bool),
        extra={
            "sampler": "analytical_cyclic_circulation",
            "cycle_flow_ratio": kappa,
            "delete_probability_cap": delete_probability_cap,
            "expected_deletions": kappa * float(sid_len),
            "expected_actions": (1.0 + 2.0 * kappa) * float(sid_len) + 1.0,
        },
    )
