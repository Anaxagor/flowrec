from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


RewardTransform = Literal["identity", "log1p", "binary_positive"]


@dataclass(frozen=True, slots=True)
class RatingRewardConfig:
    """
    Positive trajectories use batch[reward_key].  Random negatives use a fixed
    negative_reward because they do not have observed ratings.
    """

    reward_key: str = "target_rating"
    reward_epsilon: float = 1e-6
    negative_reward: float = 1e-3
    transform: RewardTransform = "identity"


class RatingRewardModel(nn.Module):
    """Assign rewards to positive and negative semantic-ID trajectories.

    For a candidate set where candidate 0 is the observed target:
        R(positive) = transformed target_rating
        R(negative) = negative_reward

    All rewards are clamped to reward_epsilon to make log R well-defined.
    """

    def __init__(self, cfg: RatingRewardConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RatingRewardConfig()

    def forward(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        if self.cfg.reward_key not in batch:
            raise KeyError(f"Batch is missing reward key: {self.cfg.reward_key}")

        positive_reward = batch[self.cfg.reward_key]
        if not torch.is_tensor(positive_reward):
            positive_reward = torch.as_tensor(positive_reward)
        positive_reward = positive_reward.to(trajectories.candidate_token_ids.device).float()
        positive_reward = self._transform(positive_reward)
        positive_reward = positive_reward.clamp_min(float(self.cfg.reward_epsilon))

        rewards = torch.full(
            trajectories.is_positive.shape,
            fill_value=float(self.cfg.negative_reward),
            device=trajectories.candidate_token_ids.device,
            dtype=torch.float32,
        )
        rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
        rewards = torch.where(trajectories.is_positive, positive_reward.unsqueeze(1), rewards)
        return rewards

    def _transform(self, reward: torch.Tensor) -> torch.Tensor:
        if self.cfg.transform == "identity":
            return reward
        if self.cfg.transform == "log1p":
            return torch.log1p(reward.clamp_min(0.0))
        if self.cfg.transform == "binary_positive":
            return (reward > 0).float()
        raise ValueError(f"Unknown reward transform: {self.cfg.transform}")
