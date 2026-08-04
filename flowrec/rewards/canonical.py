from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import torch
import torch.nn.functional as F


_COMPONENTS = ("sasrec", "rating", "similarity")


@dataclass(frozen=True, slots=True)
class CanonicalRewardSpec:
    reward_type: Literal["sasrec", "mix"]
    sasrec_adapter: Literal["softplus"] = "softplus"
    component_normalization: Literal["row_max"] = "row_max"
    mix_weights: tuple[tuple[str, float], ...] = (
        ("sasrec", 1.0),
        ("rating", 1.0),
        ("similarity", 1.0),
    )
    reward_epsilon: float = 1e-6

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CanonicalRewardSpec":
        reward_type = data.get("reward_type", "sasrec")
        adapter = data.get("sasrec_adapter", "softplus")
        normalizer = data.get("component_normalization", "row_max")
        if reward_type not in {"sasrec", "mix"}:
            raise ValueError(f"unknown canonical reward type: {reward_type}")
        if adapter != "softplus":
            raise ValueError(f"unknown SASRec adapter: {adapter}")
        if normalizer != "row_max":
            raise ValueError(f"unknown component normalizer: {normalizer}")

        epsilon = float(data.get("reward_epsilon", 1e-6))
        _validate_epsilon(epsilon)
        raw_weights = data.get("mix_weights")
        if raw_weights is None:
            weights = dict(cls.__dataclass_fields__["mix_weights"].default)
        elif isinstance(raw_weights, Mapping):
            unknown = set(raw_weights) - set(_COMPONENTS)
            if unknown:
                raise ValueError(f"unknown Mix weight keys: {sorted(unknown)}")
            weights = {name: raw_weights.get(name, 0.0) for name in _COMPONENTS}
        else:
            raise ValueError("mix_weights must be a mapping")

        canonical_weights = tuple((name, float(weights[name])) for name in _COMPONENTS)
        if any(not math.isfinite(value) or value < 0 for _, value in canonical_weights):
            raise ValueError("Mix weights must be finite and non-negative")
        if not math.isfinite(sum(value for _, value in canonical_weights)) or not any(
            value > 0 for _, value in canonical_weights
        ):
            raise ValueError("Mix weights must have a finite positive total")
        return cls(
            reward_type=reward_type,
            sasrec_adapter=adapter,
            component_normalization=normalizer,
            mix_weights=canonical_weights,
            reward_epsilon=epsilon,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "reward_pipeline": "canonical_v1",
            "reward_type": self.reward_type,
            "sasrec_adapter": self.sasrec_adapter,
            "component_normalization": self.component_normalization,
            "mix_weights": dict(self.mix_weights),
            "reward_epsilon": self.reward_epsilon,
        }


def sasrec_softplus_rewards(logits: torch.Tensor, epsilon: float) -> torch.Tensor:
    _validate_epsilon(epsilon)
    return F.softplus(logits.float()).clamp_min(epsilon)


def row_max_normalize(rewards: torch.Tensor, epsilon: float) -> torch.Tensor:
    _validate_epsilon(epsilon)
    positive = rewards.float().clamp_min(epsilon)
    return positive / positive.amax(dim=1, keepdim=True).clamp_min(epsilon)


def weighted_average_rewards(
    components: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    epsilon: float,
) -> torch.Tensor:
    _validate_epsilon(epsilon)
    selected: list[tuple[torch.Tensor, float]] = []
    for name, raw_weight in weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Mix weight {name} must be finite and non-negative")
        if weight == 0:
            continue
        if name not in components:
            raise ValueError(f"missing non-zero Mix component: {name}")
        selected.append((components[name].float(), weight))
    if not selected:
        raise ValueError("Mix weights must have a positive total")
    shape = selected[0][0].shape
    if any(component.shape != shape for component, _ in selected[1:]):
        raise ValueError("Mix components must have identical tensor shapes")
    total = sum(weight for _, weight in selected)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Mix weights must have a finite positive total")
    mixed = sum(
        (component * (weight / total) for component, weight in selected),
        torch.zeros_like(selected[0][0]),
    )
    assert_unit_interval("weighted average rewards", mixed)
    return mixed


def apply_reward_power(rewards: torch.Tensor, alpha: float, epsilon: float) -> torch.Tensor:
    _validate_epsilon(epsilon)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("reward power alpha must be finite and positive")
    return rewards.float().clamp_min(epsilon).pow(alpha).clamp_min(epsilon)


def assert_unit_interval(name: str, rewards: torch.Tensor, tolerance: float = 1e-6) -> None:
    if not torch.isfinite(rewards).all():
        raise ValueError(f"{name} must contain only finite values")
    if torch.any(rewards < -tolerance) or torch.any(rewards > 1 + tolerance):
        raise ValueError(f"{name} must be in [0, 1] within tolerance {tolerance}")


def _validate_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("reward epsilon must be finite and positive")
