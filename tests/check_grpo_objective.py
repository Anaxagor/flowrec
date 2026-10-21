#!/usr/bin/env python3
"""Isolated CPU checks of the GRPO formulas audited on 2026-10-19.

Source: researcher/flowrec, commit 08b7957056a31994069f3b49d9500bef2dc51952,
flowrec/ft_methods/mini_one_rec/method.py, compute_loss and rule/rank rewards.
This does NOT run the repository trainer or load a recommendation checkpoint.
Requires PyTorch. Run: python check_grpo_objective.py
"""
from __future__ import annotations
import json
import math
import torch


def advantages(rewards: torch.Tensor) -> torch.Tensor:
    return (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(
        -1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def rule_rank(positive: torch.Tensor, rank_weight: float = 1.0) -> torch.Tensor:
    rank = torch.arange(1, positive.shape[-1] + 1, dtype=torch.float64)
    penalty = (1.0 / torch.log(rank + 1.0)) * (1.0 - positive)
    denominator = penalty.sum(-1, keepdim=True)
    rank_term = torch.where(denominator > 0, -penalty / denominator.clamp_min(1e-12), 0.0)
    return positive + rank_weight * rank_term


def run_checks() -> dict:
    torch.manual_seed(42)
    # Two independent prompts, sixteen completions, four tokens per item.
    logits = torch.randn(2, 16, 4, 8, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 8, (2, 16, 4))
    logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    old_logp = logp.detach()
    ref_logp = logp.detach()
    positive = torch.zeros(2, 16, dtype=torch.float64)
    positive[0, 3] = 1
    positive[1, 10] = 1
    adv = advantages(rule_rank(positive)).unsqueeze(-1)
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(0.9, 1.1) * adv)
    pg_loss = -surrogate.mean()
    direct_loss = -(adv * logp).mean()
    grad_pg = torch.autograd.grad(pg_loss, logits, retain_graph=True)[0]
    grad_direct = torch.autograd.grad(direct_loss, logits, retain_graph=True)[0]
    assert abs(pg_loss.item()) < 1e-12
    assert grad_pg.norm().item() > 0
    assert torch.allclose(grad_pg, grad_direct, atol=1e-12, rtol=1e-12)

    # At identical policy/reference weights, the sampled K3 term and its
    # first derivative vanish; changing beta cannot affect the initial gradient.
    d = ref_logp - logp
    kl = (d.exp() - d - 1.0).mean()
    grad_kl = torch.autograd.grad(kl, logits, retain_graph=True)[0]
    beta_differences = {}
    for beta in (0.0, 0.1, 1.0):
        g = torch.autograd.grad(pg_loss + beta * kl, logits, retain_graph=True)[0]
        beta_differences[str(beta)] = (g - grad_pg).abs().max().item()
        assert torch.allclose(g, grad_pg, atol=1e-12, rtol=1e-12)
    assert grad_kl.abs().max().item() == 0

    # No-hit groups retain nonzero, rank-driven advantages. Rescaling a
    # positive rank coefficient is cancelled by within-group standardization.
    no_hit = torch.zeros(1, 16, dtype=torch.float64)
    a1 = advantages(rule_rank(no_hit, 1.0))
    a_small = advantages(rule_rank(no_hit, 0.05))
    a_zero = advantages(rule_rank(no_hit, 0.0))
    assert torch.allclose(a1, a_small, atol=1e-12, rtol=1e-12)
    assert a1[0, 0] < 0 and a1[0, -1] > 0
    assert a_zero.abs().max().item() == 0

    # One forced positive and fifteen equally rewarded negatives.
    p = torch.zeros(1, 16, dtype=torch.float64)
    p[0, -1] = 1
    a_exact = advantages(rule_rank(p, 0.0))
    assert abs(a_exact[0, -1].item() - math.sqrt(15)) < 1e-12
    assert abs(a_exact[0, 0].item() + 1 / math.sqrt(15)) < 1e-12

    # Two length-2 paths. First-token probabilities are .6 and .4. At
    # the next step, each path has one valid child, with unconstrained
    # probabilities .01 and .9 respectively. Mask-only beam scores and
    # valid-action-normalized scores rank these paths differently.
    raw_path_probability = [0.6 * 0.01, 0.4 * 0.9]
    valid_normalized_path_probability = [0.6, 0.4]
    assert raw_path_probability[0] < raw_path_probability[1]
    assert valid_normalized_path_probability[0] > valid_normalized_path_probability[1]
    return {
        "scope": "isolated formulas; no checkpoint, GPU training or repository integration test",
        "torch_version": torch.__version__,
        "zero_loss_nonzero_gradient": {
            "pg_loss": pg_loss.item(), "pg_gradient_norm": grad_pg.norm().item(),
            "max_error_vs_advantage_weighted_logp_gradient": (grad_pg - grad_direct).abs().max().item(),
            "clip_fraction": ((ratio < .9) | (ratio > 1.1)).double().mean().item(),
        },
        "identical_reference": {
            "kl": kl.item(), "kl_gradient_norm": grad_kl.norm().item(),
            "gradient_max_difference_by_beta": beta_differences,
        },
        "all_negative_rank_group": {
            "advantages_rank_1_to_16": a1.squeeze(0).tolist(),
            "positive_advantage_count": int((a1 > 0).sum()),
            "difference_when_rank_weight_changes_1_to_0_05": (a1-a_small).abs().max().item(),
            "max_advantage_when_rank_weight_zero": a_zero.abs().max().item(),
        },
        "exact_reward_one_positive": {
            "positive_advantage": a_exact[0, -1].item(), "negative_advantage": a_exact[0, 0].item(),
        },
        "normalization_ranking_counterexample": {
            "full_vocabulary_mask_only_path_probabilities": raw_path_probability,
            "valid_action_normalized_path_probabilities": valid_normalized_path_probability,
        },
        "all_checks_passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, ensure_ascii=False))
