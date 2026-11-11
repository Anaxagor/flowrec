"""Distill exact soft Q-values: Q(s, a) = log F(sa) on a catalog tree."""

import torch


def q_distill_loss(
    online_q: torch.Tensor,
    prefix_log_flows: torch.Tensor,
    candidate_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if online_q.ndim != 3 or candidate_token_mask.shape != online_q.shape:
        raise ValueError("Q-values and mask must have matching [B, N, L] shapes.")
    if prefix_log_flows.shape != (*online_q.shape[:2], online_q.size(-1) + 1):
        raise ValueError("Prefix log flows must have shape [B, N, L+1], including the root.")
    if online_q.numel() == 0 or not bool(candidate_token_mask.bool().all()):
        raise ValueError("Q distillation requires complete catalog SID paths.")
    targets = prefix_log_flows[..., 1:].detach().to(online_q.device).float()
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Q distillation targets must be finite log flows.")
    residual = online_q.float() - targets
    return residual.square().mean(), {
        "q_distill_abs_error_mean": residual.detach().abs().mean(),
        "q_distill_target_mean": targets.mean(),
    }
