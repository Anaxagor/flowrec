from __future__ import annotations

import torch


def detailed_balance_loss(
    token_logprobs: torch.Tensor,
    parent_log_flows: torch.Tensor,
    child_log_flows: torch.Tensor,
    rewards: torch.Tensor,
    *,
    candidate_token_mask: torch.Tensor | None = None,
    candidate_weights: torch.Tensor | None = None,
    reward_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Detailed balance with caller-provided analytical prefix flows.

    The caller provides logF(parent prefix) and logF(child prefix). Terminal
    child states should already be clamped/tied to log reward. For our SID tree
    each non-root state has one parent, so log P_B(parent | child) is zero.
    """

    if token_logprobs.ndim != 3:
        raise ValueError("token_logprobs must have shape [B, N, L].")
    if parent_log_flows.shape != token_logprobs.shape:
        raise ValueError("parent_log_flows must match token_logprobs shape.")
    if child_log_flows.shape != token_logprobs.shape:
        raise ValueError("child_log_flows must match token_logprobs shape.")
    if rewards.shape != token_logprobs.shape[:2]:
        raise ValueError(
            "rewards must have shape [B, N]. "
            f"Got {tuple(rewards.shape)} for token_logprobs {tuple(token_logprobs.shape)}."
        )

    device = token_logprobs.device
    dtype = token_logprobs.dtype
    rewards = rewards.to(device=device, dtype=dtype).clamp_min(float(reward_epsilon))
    parent_log_flows = parent_log_flows.to(device=device, dtype=dtype)
    child_log_flows = child_log_flows.to(device=device, dtype=dtype)
    if candidate_token_mask is None:
        active_steps = torch.ones_like(token_logprobs, dtype=torch.bool, device=device)
    else:
        active_steps = candidate_token_mask.to(device=device).bool()
        if active_steps.shape != token_logprobs.shape:
            raise ValueError(
                "candidate_token_mask must have shape [B, N, L]. "
                f"Got {tuple(active_steps.shape)} for token_logprobs {tuple(token_logprobs.shape)}."
            )

    residual = parent_log_flows + token_logprobs - child_log_flows
    losses = residual.square()

    if candidate_weights is None:
        weights = rewards / rewards.sum(dim=1, keepdim=True).clamp_min(float(reward_epsilon))
    else:
        weights = candidate_weights.to(device=device, dtype=dtype).clamp_min(0.0)
        if weights.shape != rewards.shape:
            raise ValueError(
                "candidate_weights must have shape [B, N]. "
                f"Got {tuple(weights.shape)} for rewards {tuple(rewards.shape)}."
            )
    step_weights = weights.unsqueeze(-1) * active_steps.to(dtype)
    weight_sums = step_weights.sum(dim=(1, 2))
    per_context_loss = (losses * step_weights).sum(dim=(1, 2)) / weight_sums.clamp_min(1e-12)
    active_contexts = weight_sums.gt(0.0)
    loss = per_context_loss[active_contexts].mean() if active_contexts.any() else losses.new_zeros(())

    active_residual = residual[active_steps]
    if active_residual.numel() == 0:
        active_residual = residual.new_zeros((1,))
    active_logprobs = token_logprobs.detach()[active_steps]
    active_parent = parent_log_flows.detach()[active_steps]
    active_child = child_log_flows.detach()[active_steps]

    metrics = {
        "db_loss": loss.detach(),
        "exact_db_active": torch.tensor(1.0, device=device, dtype=torch.float32),
        "db_residual_mean": active_residual.detach().mean(),
        "db_residual_std": active_residual.detach().std(unbiased=False),
        "db_active_steps_mean": active_steps.to(dtype).sum(dim=(1, 2)).detach().mean(),
        "db_weight_sum_mean": weight_sums.detach().mean(),
        "db_parent_log_flow_mean": (
            active_parent.mean() if active_parent.numel() else parent_log_flows.detach().mean()
        ),
        "db_child_log_flow_mean": (active_child.mean() if active_child.numel() else child_log_flows.detach().mean()),
        "token_logprob_mean": (active_logprobs.mean() if active_logprobs.numel() else token_logprobs.detach().mean()),
        "reward_mean": rewards.detach().mean(),
        "reward_min": rewards.detach().amin(),
        "reward_max": rewards.detach().amax(),
    }
    return loss, metrics


def _prefix_flows(
    token_ids: torch.Tensor,
    rewards: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _, seq_len = token_ids.shape
    parent_flows = rewards.new_zeros(token_ids.shape)
    child_flows = rewards.new_zeros(token_ids.shape)
    for batch_idx in range(bsz):
        active_reward = rewards[batch_idx]
        for step in range(seq_len):
            active = token_mask[batch_idx, :, step] & active_reward.gt(0.0)
            if not active.any():
                continue
            if step == 0:
                parent_flows[batch_idx, active, step] = active_reward[active].sum()
            else:
                parent_flows[batch_idx, :, step] = _flows_for_prefix_length(
                    token_ids[batch_idx], active_reward, active, step
                )
            child_flows[batch_idx, :, step] = _flows_for_prefix_length(
                token_ids[batch_idx], active_reward, active, step + 1
            )
    return parent_flows, child_flows


def _flows_for_prefix_length(
    token_ids: torch.Tensor,
    rewards: torch.Tensor,
    active: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    out = rewards.new_zeros(rewards.shape)
    active_idx = active.nonzero(as_tuple=False).flatten()
    if active_idx.numel() == 0:
        return out
    prefixes = token_ids.index_select(0, active_idx)[:, :prefix_len]
    _, inverse = torch.unique(prefixes, dim=0, return_inverse=True)
    group_flows = rewards.new_zeros(int(inverse.max().item()) + 1)
    group_flows.scatter_add_(0, inverse, rewards.index_select(0, active_idx))
    out.index_copy_(0, active_idx, group_flows.index_select(0, inverse))
    return out
