from __future__ import annotations

import torch


def subtrajectory_balance_loss(
    token_logprobs: torch.Tensor,
    prefix_log_flows: torch.Tensor,
    rewards: torch.Tensor,
    *,
    candidate_token_mask: torch.Tensor | None = None,
    candidate_weights: torch.Tensor | None = None,
    reward_epsilon: float = 1e-6,
    subtb_lambda: float = 1.0,
    max_subtrajectory_length: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SubTB over semantic-ID prefix subtrajectories.

    ``prefix_log_flows`` must contain log F(history, prefix) for every prefix
    length [0..L]. Terminal prefix flows are expected to be tied to log reward
    by the caller. In our left-to-right SID environment every non-root prefix
    has exactly one parent, so backward log-probabilities are zero.
    """

    if token_logprobs.ndim != 3:
        raise ValueError("token_logprobs must have shape [B, N, L].")
    expected_flow_shape = token_logprobs.shape[:2] + (token_logprobs.size(-1) + 1,)
    if prefix_log_flows.shape != expected_flow_shape:
        raise ValueError(
            "prefix_log_flows must have shape [B, N, L + 1]. "
            f"Got {tuple(prefix_log_flows.shape)} for token_logprobs {tuple(token_logprobs.shape)}."
        )
    if rewards.shape != token_logprobs.shape[:2]:
        raise ValueError(
            "rewards must have shape [B, N]. "
            f"Got {tuple(rewards.shape)} for token_logprobs {tuple(token_logprobs.shape)}."
        )
    if float(subtb_lambda) <= 0.0:
        raise ValueError("subtb_lambda must be positive.")

    device = token_logprobs.device
    dtype = token_logprobs.dtype
    rewards = rewards.to(device=device, dtype=dtype).clamp_min(float(reward_epsilon))
    log_flows = prefix_log_flows.to(device=device, dtype=dtype)
    if candidate_token_mask is None:
        token_mask = torch.ones_like(token_logprobs, dtype=torch.bool, device=device)
    else:
        token_mask = candidate_token_mask.to(device=device).bool()
        if token_mask.shape != token_logprobs.shape:
            raise ValueError(
                "candidate_token_mask must have shape [B, N, L]. "
                f"Got {tuple(token_mask.shape)} for token_logprobs {tuple(token_logprobs.shape)}."
            )

    if candidate_weights is None:
        weights = torch.ones_like(rewards)
    else:
        weights = candidate_weights.to(device=device, dtype=dtype).clamp_min(0.0)
        if weights.shape != rewards.shape:
            raise ValueError(
                "candidate_weights must have shape [B, N]. "
                f"Got {tuple(weights.shape)} for rewards {tuple(rewards.shape)}."
            )

    seq_len = int(token_logprobs.size(-1))
    max_len = seq_len if max_subtrajectory_length is None else min(int(max_subtrajectory_length), seq_len)
    if max_len < 1:
        raise ValueError("max_subtrajectory_length must be at least 1.")

    token_logprobs = token_logprobs.masked_fill(~token_mask, 0.0)
    cumulative_logprobs = torch.cat(
        [
            token_logprobs.new_zeros(token_logprobs.shape[:2] + (1,)),
            token_logprobs.cumsum(dim=-1),
        ],
        dim=-1,
    )

    loss_sums = token_logprobs.new_zeros(rewards.shape[0])
    weight_sums = token_logprobs.new_zeros(rewards.shape[0])
    residual_values: list[torch.Tensor] = []
    length_loss_sums: dict[int, torch.Tensor] = {}
    length_weight_sums: dict[int, torch.Tensor] = {}
    active_segments = token_logprobs.new_zeros(())

    for start in range(seq_len):
        upper = min(seq_len, start + max_len)
        for end in range(start + 1, upper + 1):
            length = end - start
            segment_active = token_mask[..., start:end].all(dim=-1)
            if not bool(segment_active.any()):
                continue
            length_weight = float(subtb_lambda) ** length
            segment_logprob = cumulative_logprobs[..., end] - cumulative_logprobs[..., start]
            residual = log_flows[..., start] + segment_logprob - log_flows[..., end]
            segment_weights = weights * segment_active.to(dtype) * length_weight
            segment_losses = residual.square() * segment_weights
            loss_sums = loss_sums + segment_losses.sum(dim=1)
            weight_sums = weight_sums + segment_weights.sum(dim=1)
            active_segments = active_segments + segment_active.to(dtype).sum()
            residual_values.append(residual.detach()[segment_active])
            length_loss_sums[length] = length_loss_sums.get(length, token_logprobs.new_zeros(())) + segment_losses.sum()
            length_weight_sums[length] = (
                length_weight_sums.get(length, token_logprobs.new_zeros(())) + segment_weights.sum()
            )

    active_contexts = weight_sums.gt(0.0)
    if active_contexts.any():
        per_context_loss = loss_sums[active_contexts] / weight_sums[active_contexts].clamp_min(1e-12)
        loss = per_context_loss.mean()
    else:
        loss = token_logprobs.new_zeros(())

    if residual_values:
        active_residual = torch.cat([x.reshape(-1) for x in residual_values if x.numel() > 0])
    else:
        active_residual = token_logprobs.new_zeros((1,))
    if active_residual.numel() == 0:
        active_residual = token_logprobs.new_zeros((1,))

    prefix_active = torch.ones(log_flows.shape, dtype=torch.bool, device=device)
    for prefix_len in range(1, seq_len + 1):
        prefix_active[..., prefix_len] = token_mask[..., :prefix_len].all(dim=-1)
    active_log_flows = log_flows.detach()[prefix_active]
    active_logprobs = token_logprobs.detach()[token_mask]
    metrics = {
        "subtb_loss": loss.detach(),
        "subtb_lambda": torch.tensor(float(subtb_lambda), device=device, dtype=dtype),
        "exact_subtb_active": torch.tensor(1.0, device=device, dtype=torch.float32),
        "subtb_residual_mean": active_residual.detach().mean(),
        "subtb_residual_std": active_residual.detach().std(unbiased=False),
        "subtb_active_segments_mean": (active_segments / max(int(rewards.size(0)), 1)).detach(),
        "subtb_weight_sum_mean": weight_sums.detach().mean(),
        "subtb_root_log_flow_mean": log_flows[..., 0].detach().mean(),
        "subtb_terminal_log_flow_mean": rewards.detach().log().mean(),
        "subtb_log_flow_mean": (active_log_flows.mean() if active_log_flows.numel() else log_flows.detach().mean()),
        "subtb_log_flow_std": (
            active_log_flows.std(unbiased=False) if active_log_flows.numel() else log_flows.detach().std(unbiased=False)
        ),
        "token_logprob_mean": (active_logprobs.mean() if active_logprobs.numel() else token_logprobs.detach().mean()),
        "reward_mean": rewards.detach().mean(),
        "reward_min": rewards.detach().amin(),
        "reward_max": rewards.detach().amax(),
    }
    for length, numerator in sorted(length_loss_sums.items()):
        denom = length_weight_sums[length].clamp_min(1e-12)
        metrics[f"subtb_len{length}_loss"] = (numerator / denom).detach()
    return loss, metrics
