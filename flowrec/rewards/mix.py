from __future__ import annotations

import torch
import torch.nn as nn

from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class MultiplicativeRewardModel(nn.Module):
    """
    Product-of-components reward:

        R = prod_i R_i ** w_i

    Intended composition: a dense collaborative base reward (e.g.
    SASRecRewardModel) multiplied by one or more logged-evidence modulators
    that stay ~1.0 where they carry no information (e.g. AmazonLogRewardModel
    with reward_mode="item_centered"). Unobserved candidates then keep the
    base reward untouched, while logged interactions re-weight it.

    Weights act as exponents, i.e. weighted additive terms in log-reward
    space, which is what TB-style losses consume. Unlike MixedRewardModel,
    no candidate-set normalization is applied, so the result stays a fixed
    terminal reward as long as every component is one.
    """

    def __init__(
        self,
        components: list[nn.Module],
        *,
        weights: list[float] | None = None,
        reward_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("MultiplicativeRewardModel requires at least one component.")
        self.components = nn.ModuleList(components)
        self.weights = [float(w) for w in (weights or [1.0] * len(components))]
        if len(self.weights) != len(self.components):
            raise ValueError("weights must match the number of reward components.")
        self.reward_epsilon = float(reward_epsilon)

    def forward(self, batch: Batch, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        combined: torch.Tensor | None = None
        for model, weight in zip(self.components, self.weights):
            rewards = model(batch, trajectories).float()
            if rewards.ndim != 2:
                raise ValueError(
                    f"Reward components must return [B, N], got {tuple(rewards.shape)}."
                )
            rewards = rewards.clamp_min(self.reward_epsilon)
            term = rewards if weight == 1.0 else rewards.pow(weight)
            if combined is None:
                combined = term
            elif combined.shape != term.shape:
                raise ValueError(
                    "Reward components disagree on candidate shape: "
                    f"{tuple(combined.shape)} vs {tuple(term.shape)}."
                )
            else:
                combined = combined * term
        assert combined is not None
        return combined.clamp_min(self.reward_epsilon)
