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

class MixedRewardModel(nn.Module):
    """Weighted-sum reward composition:

        R = sum_i w_i * normalize_i(R_i)

    TB-safety caveat: any batch/row normalizer makes the reward depend on the
    candidate set, which is inconsistent with a precomputed full-catalog logZ
    (``uses_candidate_set_normalization`` flags this). With
    ``normalizers=["none", ...]`` the sum of pointwise components stays a fixed
    terminal reward. Note the analysis in
    examples/amazon_reward_analysis.ipynb (section 5.1) found batch_max
    normalization lets the smallest-scale component dominate the ranking;
    prefer MultiplicativeRewardModel for base-times-modulator compositions.
    """

    def __init__(
        self,
        components: list[nn.Module],
        *,
        weights: list[float] | None = None,
        normalizers: list[str] | None = None,
        reward_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("MixedRewardModel requires at least one component.")
        self.components = nn.ModuleList(components)
        self.weights = [float(w) for w in (weights or [1.0] * len(components))]
        self.normalizers = [str(n) for n in (normalizers or ["batch_max"] * len(components))]
        self.reward_epsilon = float(reward_epsilon)
        if len(self.weights) != len(self.components) or len(self.normalizers) != len(self.components):
            raise ValueError("weights and normalizers must match the number of reward components.")

    @property
    def uses_candidate_set_normalization(self) -> bool:
        return any(str(mode).lower() not in {"none", "identity"} for mode in self.normalizers)

    def forward(self, batch: Batch, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        mixed: torch.Tensor | None = None
        for model, weight, normalizer in zip(self.components, self.weights, self.normalizers):
            rewards = model(batch, trajectories).float()
            if rewards.ndim != 2:
                raise ValueError(f"Reward components must return [B, N], got {tuple(rewards.shape)}.")
            term = self._normalize(rewards, normalizer) * weight
            mixed = term if mixed is None else mixed + term
        if mixed is None:
            raise RuntimeError("MixedRewardModel has no reward components.")
        return mixed.clamp_min(self.reward_epsilon)

    def _normalize(self, rewards: torch.Tensor, mode: str) -> torch.Tensor:
        mode = mode.lower()
        eps = self.reward_epsilon
        rewards = rewards.clamp_min(eps)
        if mode in {"none", "identity"}:
            return rewards
        if mode in {"batch_max", "max"}:
            denom = rewards.detach().amax().clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"batch_mean", "mean"}:
            denom = rewards.detach().mean().clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"batch_minmax", "minmax"}:
            min_reward = rewards.detach().amin()
            denom = (rewards.detach().amax() - min_reward).clamp_min(eps)
            scaled = (rewards - min_reward) / denom
            if torch.isclose(denom, rewards.new_tensor(eps)):
                scaled = torch.ones_like(rewards)
            return scaled.clamp_min(eps)
        if mode in {"row_max", "per_row_max"}:
            denom = rewards.detach().amax(dim=1, keepdim=True).clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"row_mean", "per_row_mean"}:
            denom = rewards.detach().mean(dim=1, keepdim=True).clamp_min(eps)
            return (rewards / denom).clamp_min(eps)
        if mode in {"row_minmax", "per_row_minmax"}:
            min_reward = rewards.detach().amin(dim=1, keepdim=True)
            denom = (rewards.detach().amax(dim=1, keepdim=True) - min_reward).clamp_min(eps)
            scaled = (rewards - min_reward) / denom
            scaled = torch.where(denom <= eps, torch.ones_like(scaled), scaled)
            return scaled.clamp_min(eps)
        raise ValueError(
            f"Unknown reward normalizer {mode!r}. "
            "Expected none, batch_max, batch_mean, batch_minmax, row_max, row_mean, or row_minmax."
        )
