"""TB, DB, and SubTB objectives for finite trajectories with repeated states."""

from __future__ import annotations

import torch


def _validate_transition_tensors(
    forward_logprobs: torch.Tensor,
    backward_logprobs: torch.Tensor,
    step_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if forward_logprobs.ndim != 3:
        raise ValueError("Cyclic transition tensors must have shape [B, N, T].")
    if backward_logprobs.shape != forward_logprobs.shape:
        raise ValueError("backward_logprobs must match forward_logprobs.")
    if step_mask.shape != forward_logprobs.shape:
        raise ValueError("step_mask must match forward_logprobs.")
    return (
        forward_logprobs,
        backward_logprobs.to(forward_logprobs.device, forward_logprobs.dtype),
        step_mask.to(forward_logprobs.device).bool(),
    )


def cyclic_trajectory_balance_loss(
    forward_logprobs: torch.Tensor,
    backward_logprobs: torch.Tensor,
    step_mask: torch.Tensor,
    rewards: torch.Tensor,
    log_z: torch.Tensor,
    root_backward_logprobs: torch.Tensor,
    *,
    reward_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trajectory balance for source-to-terminal paths with repeated states."""
    forward_logprobs, backward_logprobs, step_mask = _validate_transition_tensors(
        forward_logprobs, backward_logprobs, step_mask
    )
    if rewards.shape != forward_logprobs.shape[:2]:
        raise ValueError("rewards must have shape [B, N].")
    if root_backward_logprobs.shape != rewards.shape:
        raise ValueError("root_backward_logprobs must have shape [B, N].")
    if log_z.ndim == 1:
        log_z = log_z.unsqueeze(1)
    if log_z.ndim != 0 and log_z.shape not in {(int(rewards.size(0)), 1), rewards.shape}:
        raise ValueError("log_z must be scalar, [B], [B, 1], or [B, N].")

    mask = step_mask.to(forward_logprobs.dtype)
    forward_sum = (forward_logprobs * mask).sum(dim=-1)
    backward_sum = (backward_logprobs * mask).sum(dim=-1)
    backward_sum = backward_sum + root_backward_logprobs.to(forward_sum.device, forward_sum.dtype)
    log_rewards = rewards.to(forward_sum.device, forward_sum.dtype).clamp_min(float(reward_epsilon)).log()
    residual = log_z.to(forward_sum.device, forward_sum.dtype) + forward_sum
    residual = residual - backward_sum - log_rewards
    loss = residual.square().mean()
    return loss, {
        "tb_loss": loss.detach(),
        "tb_residual_mean": residual.detach().mean(),
        "tb_residual_std": residual.detach().std(unbiased=False),
        "sequence_logprob_mean": forward_sum.detach().mean(),
        "backward_logprob_mean": backward_sum.detach().mean(),
        "reward_mean": rewards.detach().mean(),
        "reward_min": rewards.detach().amin(),
        "reward_max": rewards.detach().amax(),
    }


def cyclic_detailed_balance_loss(
    forward_logprobs: torch.Tensor,
    backward_logprobs: torch.Tensor,
    state_log_flows: torch.Tensor,
    step_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Detailed balance on every sampled append, delete, and stop edge."""
    forward_logprobs, backward_logprobs, step_mask = _validate_transition_tensors(
        forward_logprobs, backward_logprobs, step_mask
    )
    expected = forward_logprobs.shape[:2] + (int(forward_logprobs.size(-1)) + 1,)
    if state_log_flows.shape != expected:
        raise ValueError("state_log_flows must have shape [B, N, T + 1].")
    state_log_flows = state_log_flows.to(forward_logprobs.device, forward_logprobs.dtype)
    residual = state_log_flows[..., :-1] + forward_logprobs - state_log_flows[..., 1:] - backward_logprobs
    active = residual[step_mask]
    if active.numel() == 0:
        raise ValueError("Cyclic DB received no active transitions.")
    loss = active.square().mean()
    return loss, {
        "db_loss": loss.detach(),
        "db_residual_mean": active.detach().mean(),
        "db_residual_std": active.detach().std(unbiased=False),
        "db_active_steps_mean": step_mask.sum(dim=-1).float().mean(),
        "db_parent_log_flow_mean": state_log_flows[..., :-1][step_mask].detach().mean(),
        "db_child_log_flow_mean": state_log_flows[..., 1:][step_mask].detach().mean(),
        "token_logprob_mean": forward_logprobs[step_mask].detach().mean(),
    }


def cyclic_subtrajectory_balance_loss(
    forward_logprobs: torch.Tensor,
    backward_logprobs: torch.Tensor,
    state_log_flows: torch.Tensor,
    step_mask: torch.Tensor,
    *,
    subtb_lambda: float = 1.0,
    max_subtrajectory_length: int | None = None,
    estimator: str = "exact",
    num_mc_samples: int = 32,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SubTB over finite segments of cyclic trajectories."""
    forward_logprobs, backward_logprobs, step_mask = _validate_transition_tensors(
        forward_logprobs, backward_logprobs, step_mask
    )
    expected = forward_logprobs.shape[:2] + (int(forward_logprobs.size(-1)) + 1,)
    if state_log_flows.shape != expected:
        raise ValueError("state_log_flows must have shape [B, N, T + 1].")
    if float(subtb_lambda) <= 0.0:
        raise ValueError("subtb_lambda must be positive.")
    if estimator not in {"exact", "mc"}:
        raise ValueError("Cyclic SubTB estimator must be 'exact' or 'mc'.")

    dtype = forward_logprobs.dtype
    state_log_flows = state_log_flows.to(forward_logprobs.device, dtype)
    delta = (forward_logprobs - backward_logprobs).masked_fill(~step_mask, 0.0)
    cumulative = torch.cat([delta.new_zeros(delta.shape[:2] + (1,)), delta.cumsum(dim=-1)], dim=-1)
    lengths = step_mask.long().sum(dim=-1)
    max_len = int(lengths.max().item())
    if max_subtrajectory_length is not None:
        max_len = min(max_len, int(max_subtrajectory_length))
    if max_len < 1:
        raise ValueError("Cyclic SubTB received no active segments.")

    if estimator == "mc":
        loss, residual, active_segments = _cyclic_subtb_mc(
            cumulative,
            state_log_flows,
            lengths,
            subtb_lambda=float(subtb_lambda),
            max_length=max_len,
            num_samples=int(num_mc_samples),
        )
    else:
        loss, residual, active_segments = _cyclic_subtb_exact(
            cumulative,
            state_log_flows,
            lengths,
            subtb_lambda=float(subtb_lambda),
            max_length=max_len,
        )

    active_states = torch.arange(int(state_log_flows.size(-1)), device=lengths.device).view(1, 1, -1) <= lengths.unsqueeze(-1)
    return loss, {
        "subtb_loss": loss.detach(),
        "subtb_lambda": torch.tensor(float(subtb_lambda), device=loss.device, dtype=dtype),
        "subtb_residual_mean": residual.detach().mean(),
        "subtb_residual_std": residual.detach().std(unbiased=False),
        "subtb_active_segments_mean": active_segments.detach(),
        "subtb_root_log_flow_mean": state_log_flows[..., 0].detach().mean(),
        "subtb_terminal_log_flow_mean": state_log_flows.gather(-1, lengths.unsqueeze(-1)).detach().mean(),
        "subtb_log_flow_mean": state_log_flows[active_states].detach().mean(),
        "token_logprob_mean": forward_logprobs[step_mask].detach().mean(),
        "subtb_mc_active": torch.tensor(float(estimator == "mc"), device=loss.device, dtype=dtype),
    }


def _cyclic_subtb_exact(
    cumulative: torch.Tensor,
    state_log_flows: torch.Tensor,
    lengths: torch.Tensor,
    *,
    subtb_lambda: float,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_sums = cumulative.new_zeros(lengths.shape)
    weight_sums = cumulative.new_zeros(lengths.shape)
    residual_values: list[torch.Tensor] = []
    segment_count = cumulative.new_zeros(())
    max_steps = int(lengths.max().item())
    for segment_length in range(1, max_length + 1):
        weight = float(subtb_lambda) ** segment_length
        for start in range(0, max_steps - segment_length + 1):
            end = start + segment_length
            active = lengths.ge(end)
            if not bool(active.any()):
                continue
            residual = state_log_flows[..., start] + cumulative[..., end] - cumulative[..., start] - state_log_flows[..., end]
            loss_sums = loss_sums + residual.square() * active.to(residual.dtype) * weight
            weight_sums = weight_sums + active.to(residual.dtype) * weight
            residual_values.append(residual[active])
            segment_count = segment_count + active.sum()
    active_trajectories = weight_sums.gt(0.0)
    loss = (loss_sums[active_trajectories] / weight_sums[active_trajectories].clamp_min(1e-12)).mean()
    residual = torch.cat(residual_values) if residual_values else cumulative.new_zeros((1,))
    segments_per_trajectory = segment_count / active_trajectories.sum().clamp_min(1)
    return loss, residual, segments_per_trajectory


def _cyclic_subtb_mc(
    cumulative: torch.Tensor,
    state_log_flows: torch.Tensor,
    lengths: torch.Tensor,
    *,
    subtb_lambda: float,
    max_length: int,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_mc_samples must be positive.")
    flat_lengths = lengths.reshape(-1)
    active = flat_lengths.gt(0)
    if not bool(active.any()):
        raise ValueError("Cyclic SubTB MC received no active trajectories.")
    active_lengths = flat_lengths[active]
    possible_lengths = torch.arange(1, max_length + 1, device=lengths.device).view(1, -1)
    valid = possible_lengths <= active_lengths.unsqueeze(1)
    starts_per_length = (active_lengths.unsqueeze(1) - possible_lengths + 1).clamp_min(0)
    length_weights = starts_per_length.to(cumulative.dtype) * (cumulative.new_tensor(float(subtb_lambda)) ** possible_lengths)
    length_weights = length_weights.masked_fill(~valid, 0.0)
    sampled_length_indices = torch.multinomial(length_weights, num_samples=num_samples, replacement=True)
    sampled_lengths = sampled_length_indices + 1
    num_starts = active_lengths.unsqueeze(1) - sampled_lengths + 1
    starts = torch.floor(
        torch.rand(sampled_lengths.shape, device=lengths.device, dtype=cumulative.dtype) * num_starts.to(cumulative.dtype)
    ).long()
    ends = starts + sampled_lengths

    flat_cumulative = cumulative.reshape(-1, cumulative.size(-1))[active]
    flat_flows = state_log_flows.reshape(-1, state_log_flows.size(-1))[active]
    residual = flat_flows.gather(1, starts) + flat_cumulative.gather(1, ends) - flat_cumulative.gather(1, starts) - flat_flows.gather(1, ends)
    return residual.square().mean(), residual.reshape(-1), cumulative.new_tensor(float(num_samples))
