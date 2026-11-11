"""Reward-conditioned cross entropy over valid catalog children."""

import torch


def prefix_ce_loss(
    logits: torch.Tensor,
    child_log_flows: torch.Tensor,
    valid_children: torch.Tensor,
    *,
    candidate_token_mask: torch.Tensor | None = None,
    candidate_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean over sampled path nodes per context, then over active contexts.

    Candidate multiplicities and optional reward weighting match DB/SubTB.
    Zero-mass nodes are skipped; targets and weights carry no gradients.
    """
    if logits.ndim != 4 or child_log_flows.shape != valid_children.shape:
        raise ValueError("Expected logits and child targets with shape [B, N, L, V].")
    if logits.shape[:3] != child_log_flows.shape[:3] or logits.size(-1) < child_log_flows.size(-1):
        raise ValueError("Logits must cover all catalog child token IDs.")
    logits = logits[..., :child_log_flows.size(-1)].float()
    valid = valid_children.to(device=logits.device).bool()
    teacher = child_log_flows.detach().to(device=logits.device, dtype=torch.float32)
    if bool((torch.isnan(teacher) | torch.isposinf(teacher)).any()):
        raise ValueError("Child log flows must be finite or negative infinity.")
    teacher = teacher.masked_fill(~valid, -torch.inf)
    active = torch.isfinite(teacher).any(-1)
    if candidate_token_mask is not None:
        if candidate_token_mask.shape != active.shape:
            raise ValueError("candidate_token_mask must have shape [B, N, L].")
        active = active & candidate_token_mask.to(logits.device).bool()
    target = teacher.masked_fill(~active.unsqueeze(-1), 0).softmax(-1)
    target = target.masked_fill(~active.unsqueeze(-1), 0)
    student = logits.masked_fill(~valid, -torch.inf)
    student = student.masked_fill(~valid.any(-1, keepdim=True), 0).log_softmax(-1)
    student = student.masked_fill(~valid, 0)
    ce = -(target * student).sum(-1)
    entropy = -(target * target.clamp_min(torch.finfo(target.dtype).tiny).log()).sum(-1)
    weights = torch.ones_like(ce)
    if candidate_weights is not None:
        if candidate_weights.shape != ce.shape[:2]:
            raise ValueError("candidate_weights must have shape [B, N].")
        candidate_weights = candidate_weights.detach().to(logits.device).float()
        if not bool((torch.isfinite(candidate_weights) & candidate_weights.ge(0)).all()):
            raise ValueError("Candidate weights must be finite and nonnegative.")
        weights = weights * candidate_weights.unsqueeze(-1)
    weights = weights * active

    def average(values, node_weights):
        mass = node_weights.sum(dim=(1, 2))
        per_context = (values * node_weights).sum(dim=(1, 2)) / mass.clamp_min(1e-12)
        return per_context.sum() / mass.gt(0).sum().clamp_min(1)

    loss = average(ce, weights)
    metrics = {"prefix_ce_loss": loss.detach(),
               "prefix_ce_kl": average(ce - entropy, weights).detach(),
               "prefix_ce_target_entropy": average(entropy, weights).detach(),
               "prefix_ce_active_nodes_mean": active.float().sum((1, 2)).mean()}
    for level in range(ce.size(-1)):
        for name, values in (("ce", ce), ("kl", ce - entropy), ("target_entropy", entropy)):
            metrics[f"prefix_ce_{name}_level_{level}"] = average(
                values[..., level:level+1], weights[..., level:level+1]
            ).detach()
    return loss, metrics
