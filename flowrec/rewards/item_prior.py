"""Item-rating priors from the GFlow Amazon reward ablation.

This ports ``ItemPriorRatingReward`` and ``RewardFactory._build_rating`` from
``scripts/exp4-reward-ablations/rewards_ablation_amazon.py``. Its statistics use
the FULL supplied raw reviews frame, including reviews outside the training
split and items outside the model vocabulary. It is a fixed per-item prior,
not a timestamp-filtered interaction reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch
import torch.nn as nn

from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class ItemPriorRatingRewardConfig:
    shrinkage: float = 10.0
    beta: float = 1.0
    clip_min: float = 0.5
    clip_max: float = 2.0
    reward_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("shrinkage", "beta", "clip_min", "clip_max", "reward_epsilon"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"ItemPriorRatingRewardConfig.{name} must be finite.")
        if self.shrinkage < 0:
            raise ValueError("ItemPriorRatingRewardConfig.shrinkage must be >= 0.")
        if not 0 < self.clip_min <= self.clip_max:
            raise ValueError("ItemPriorRatingRewardConfig requires 0 < clip_min <= clip_max.")
        if self.reward_epsilon <= 0:
            raise ValueError("ItemPriorRatingRewardConfig.reward_epsilon must be positive.")


def build_item_rating_prior(
    frame: pd.DataFrame,
    item2id: Mapping[str, int],
    item_vocab_size: int,
    *,
    cfg: ItemPriorRatingRewardConfig | None = None,
    item_key: str = "asin",
    rating_key: str = "rating",
) -> torch.Tensor:
    """Build the exp4 multiplier table from a full raw pandas reviews frame.

    ``mu_i = (sum_i + shrinkage * global_mean) / (count_i + shrinkage)``;
    ``R_i = clip(exp(beta * (mu_i - global_mean) / residual_std))``.
    The residual scale is the sample standard deviation over items, matching
    the original pandas implementation. With only one item or no residual
    variation, the prior is neutral. Missing/nonfinite ratings are discarded;
    items without usable ratings and padding ID 0 retain multiplier 1.
    """
    import numpy as np
    import pandas as pd

    cfg = cfg or ItemPriorRatingRewardConfig()
    if item_vocab_size < 1:
        raise ValueError("item_vocab_size must include at least the padding item.")
    missing = {item_key, rating_key} - set(frame.columns)
    if missing:
        raise KeyError(f"Item-rating prior requires review columns: {sorted(missing)}")

    ratings = pd.to_numeric(frame[rating_key], errors="coerce").astype(float)
    valid = np.isfinite(ratings.to_numpy()) & frame[item_key].notna().to_numpy()
    values = frame.loc[valid, [item_key]].copy()
    values[rating_key] = ratings.loc[valid].to_numpy()
    if values.empty:
        raise ValueError("Item-rating prior requires at least one finite raw review rating.")

    grouped = values.groupby(item_key)[rating_key].agg(["sum", "count"])
    global_mean = float(values[rating_key].mean())
    shrunk = (grouped["sum"] + float(cfg.shrinkage) * global_mean) / (
        grouped["count"] + float(cfg.shrinkage)
    )
    residuals = (shrunk - global_mean).to_numpy(dtype=float)
    scale = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        multipliers = np.ones_like(residuals)
    else:
        # Clipping in log space is equivalent and avoids overflow in exp.
        log_multiplier = float(cfg.beta) * (residuals / scale)
        log_multiplier = np.clip(log_multiplier, math.log(cfg.clip_min), math.log(cfg.clip_max))
        multipliers = np.exp(log_multiplier)

    table = torch.ones(int(item_vocab_size), dtype=torch.float32)
    for raw_item, multiplier in zip(grouped.index, multipliers):
        item_id = item2id.get(str(raw_item))
        if item_id is not None and 0 < int(item_id) < table.numel():
            table[int(item_id)] = float(multiplier)
    return table.clamp_min(float(cfg.reward_epsilon))


class ItemPriorRatingReward(nn.Module):
    """Fixed per-item multipliers; missing/invalid candidates stay neutral."""

    def __init__(
        self,
        item_multiplier: torch.Tensor,
        cfg: ItemPriorRatingRewardConfig | None = None,
        *,
        item_id_by_semantic_row: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ItemPriorRatingRewardConfig()
        self.reward_epsilon = float(self.cfg.reward_epsilon)
        table = item_multiplier.detach().float()
        if table.ndim != 1 or table.numel() == 0:
            raise ValueError("item_multiplier must be a nonempty [item_vocab_size] tensor.")
        if not bool(torch.isfinite(table).all()) or not bool(table.gt(0).all()):
            raise ValueError("item_multiplier must contain only finite positive rewards.")
        self.register_buffer("item_multiplier", table.clone())
        mapping = (
            item_id_by_semantic_row.detach().long()
            if item_id_by_semantic_row is not None
            else torch.empty(0, dtype=torch.long)
        )
        if mapping.ndim != 1:
            raise ValueError("item_id_by_semantic_row must be a one-dimensional tensor.")
        self.register_buffer("item_id_by_semantic_row", mapping.clone())

    def _candidate_item_ids(self, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        device = trajectories.candidate_token_ids.device
        candidate_ids = (trajectories.extra or {}).get("candidate_item_ids")
        if torch.is_tensor(candidate_ids):
            ids = candidate_ids.to(device).long()
        else:
            rows = trajectories.candidate_item_row_ids
            if rows is None or self.item_id_by_semantic_row.numel() == 0:
                raise ValueError(
                    "ItemPriorRatingReward needs candidate_item_ids in trajectories.extra "
                    "or candidate_item_row_ids + item_id_by_semantic_row."
                )
            rows = rows.to(device).long()
            mapping = self.item_id_by_semantic_row.to(device)
            valid_rows = rows.ge(0) & rows.lt(mapping.numel())
            mapped = mapping[rows.clamp(0, mapping.numel() - 1)]
            ids = torch.where(valid_rows, mapped, torch.full_like(mapped, -1))
        expected = trajectories.candidate_token_ids.shape[:2]
        if ids.shape != expected:
            raise ValueError(f"candidate item IDs must have shape {tuple(expected)}, got {tuple(ids.shape)}.")
        return ids

    def forward(self, batch: Batch, trajectories: SemanticTrajectoryBatch) -> torch.Tensor:
        ids = self._candidate_item_ids(trajectories)
        table = self.item_multiplier.to(ids.device)
        valid = ids.gt(0) & ids.lt(table.numel())
        rewards = table[ids.clamp(0, table.numel() - 1)]
        return torch.where(valid, rewards, torch.ones_like(rewards)).clamp_min(self.reward_epsilon)
