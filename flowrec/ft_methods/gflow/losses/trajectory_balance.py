from __future__ import annotations

import torch


def trajectory_balance_loss(
    sequence_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    log_z: torch.Tensor,
    *,
    reward_epsilon: float = 1e-6,
    candidate_mask: torch.Tensor | None = None,
    candidate_weights: torch.Tensor | None = None,
    normalize_candidate_weights: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the trajectory-balance loss used by GFlowGR.

    With the paper's GR specialization, backward probabilities are one, so
    each trajectory residual is:

        logZ + sum_t log P_GR(token_t | user, prefix_t) - log R(terminal)
    """

    if sequence_logprobs.shape != rewards.shape:
        raise ValueError(
            "sequence_logprobs and rewards must have the same shape. "
            f"Got {tuple(sequence_logprobs.shape)} and {tuple(rewards.shape)}."
        )
    if sequence_logprobs.ndim != 2:
        raise ValueError("sequence_logprobs and rewards must have shape [B, N].")

    rewards = rewards.to(sequence_logprobs.device).to(sequence_logprobs.dtype)
    log_z = _broadcast_log_z(
        log_z.to(sequence_logprobs.device).to(sequence_logprobs.dtype),
        target_shape=sequence_logprobs.shape,
    )
    log_rewards = rewards.clamp_min(float(reward_epsilon)).log()
    residual = log_z + sequence_logprobs - log_rewards
    losses = residual.square()

    mask = None
    if candidate_mask is not None:
        mask = candidate_mask.to(sequence_logprobs.device).bool()
        if mask.shape != losses.shape:
            raise ValueError(
                "candidate_mask must have shape [B, N]. " f"Got {tuple(mask.shape)} for losses {tuple(losses.shape)}."
            )

    weight_sums: torch.Tensor | None = None
    if candidate_weights is not None:
        weights = candidate_weights.to(sequence_logprobs.device).to(sequence_logprobs.dtype)
        if weights.shape != losses.shape:
            raise ValueError(
                "candidate_weights must have shape [B, N]. "
                f"Got {tuple(weights.shape)} for losses {tuple(losses.shape)}."
            )
        weights = weights.clamp_min(0.0)
        if mask is not None:
            weights = weights.masked_fill(~mask, 0.0)

        # Normalize per context so each user-prefix contributes one loss value.
        # The TB residual target is unchanged; weights only decide which
        # candidate residuals dominate the finite-support approximation.
        weight_sums = weights.sum(dim=1)
        weighted_losses = (losses * weights).sum(dim=1)
        if normalize_candidate_weights:
            denom = weight_sums.clamp_min(1e-12).to(losses.dtype)
            weighted_losses = weighted_losses / denom
        loss = weighted_losses.mean()
        active_residual = residual[weights > 0]
    elif mask is not None:
        denom = mask.sum().clamp_min(1).to(losses.dtype)
        loss = losses.masked_fill(~mask, 0.0).sum() / denom
        active_residual = residual[mask]
    else:
        loss = losses.mean()
        active_residual = residual.reshape(-1)

    if active_residual.numel() == 0:
        active_residual = residual.new_zeros((1,))

    metrics = {
        "tb_loss": loss.detach(),
        "tb_residual_mean": active_residual.detach().mean(),
        "tb_residual_std": active_residual.detach().std(unbiased=False),
        "sequence_logprob_mean": sequence_logprobs.detach().mean(),
        "reward_mean": rewards.detach().mean(),
        "reward_min": rewards.detach().amin(),
        "reward_max": rewards.detach().amax(),
    }
    if weight_sums is not None:
        metrics.update(
            {
                "tb_weight_sum_mean": weight_sums.detach().mean(),
            }
        )
    return loss, metrics


def _broadcast_log_z(log_z: torch.Tensor, *, target_shape: torch.Size) -> torch.Tensor:
    bsz, n_candidates = int(target_shape[0]), int(target_shape[1])
    if log_z.ndim == 0:
        return log_z
    if log_z.ndim == 1 and log_z.size(0) == bsz:
        return log_z.unsqueeze(1)
    if log_z.ndim == 2 and log_z.shape == (bsz, 1):
        return log_z
    if log_z.ndim == 2 and log_z.shape == (bsz, n_candidates):
        return log_z
    raise ValueError(
        "log_z must be scalar, [B], [B, 1], or [B, N]. "
        f"Got {tuple(log_z.shape)} for target shape {tuple(target_shape)}."
    )
