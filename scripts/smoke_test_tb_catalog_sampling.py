#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss
from flowrec.ft_methods.gflow.method import GFlowMethod


class FakeCache:
    def __init__(self, rewards: torch.Tensor, power: float) -> None:
        self.rewards = rewards
        self.power = power
        self.powered = rewards.float().clamp_min(1e-6).pow(power).clamp_min(1e-6)

    def select_full_catalog(self, indices, device):
        selected = self.rewards.index_select(0, indices.cpu().long()).to(device)
        sums = self.powered.index_select(0, indices.cpu().long()).sum(dim=1).to(device)
        return selected, sums

    def transform_candidate_rewards(self, rewards):
        return rewards.float().clamp_min(1e-6).pow(self.power).clamp_min(1e-6)


def build_method(strategy: str, head: int, tail: int) -> GFlowMethod:
    return GFlowMethod(
        GFlowConfig(
            objective="tb",
            tb_logz_source="reward_cache",
            tb_loss_weighting="reward",
            tb_catalog_sampling=strategy,
            tb_head_sample_size=head,
            tb_tail_sample_size=tail,
            tb_candidate_weights_normalize=False,
        )
    )


def test_uniform_exclusion() -> None:
    torch.manual_seed(7)
    excluded = torch.tensor([[1, 3, 7], [0, 5, 9]])
    sampled = GFlowMethod._sample_uniform_excluding(excluded, 5000, 10)
    assert sampled.shape == (2, 5000)
    assert sampled.min().item() >= 0 and sampled.max().item() < 10
    for row in range(2):
        assert not torch.isin(sampled[row], excluded[row]).any()


def test_hybrid_and_uniform_weights() -> None:
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    cache = FakeCache(raw, power=0.5)
    indices = torch.tensor([0])
    head_rows = torch.tensor([[5, 4]])
    head_rewards = cache.transform_candidate_rewards(raw[:, [5, 4]])

    torch.manual_seed(11)
    hybrid = build_method("hybrid", 2, 2)
    rows, rewards, full_sum, weights, head_mass, name = hybrid._sample_full_catalog_tb(
        cache, indices, head_rows, head_rewards, torch.device("cpu")
    )
    assert rows.shape == rewards.shape == weights.shape == (1, 4)
    assert name == "reward_cache_hybrid_2_2"
    assert not torch.isin(rows[:, 2:], head_rows).any()
    expected_mass = head_rewards.sum(dim=1) / full_sum
    torch.testing.assert_close(head_mass, expected_mass)
    torch.testing.assert_close(weights[:, :2].sum(dim=1), expected_mass)

    torch.manual_seed(13)
    uniform = build_method("uniform", 0, 4)
    rows, rewards, full_sum, weights, _, name = uniform._sample_full_catalog_tb(
        cache, indices, head_rows, head_rewards, torch.device("cpu")
    )
    assert name == "reward_cache_full_uniform"
    expected = rewards * raw.size(1) / full_sum.unsqueeze(1) / 4.0
    torch.testing.assert_close(weights, expected)


def test_unnormalized_importance_loss() -> None:
    sequence_logprobs = torch.log(torch.tensor([[0.2, 0.3]]))
    rewards = torch.tensor([[0.4, 0.6]])
    weights = torch.tensor([[0.25, 0.50]])
    log_z = torch.tensor([1.0])
    residual = log_z.unsqueeze(1) + sequence_logprobs - rewards.log()
    expected = (residual.square() * weights).sum(dim=1).mean()
    loss, _ = trajectory_balance_loss(
        sequence_logprobs,
        rewards,
        log_z,
        candidate_weights=weights,
        normalize_candidate_weights=False,
    )
    torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    test_uniform_exclusion()
    test_hybrid_and_uniform_weights()
    test_unnormalized_importance_loss()
    print("tb catalogue sampling smoke tests passed")
