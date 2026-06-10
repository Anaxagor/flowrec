from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


RatingMode = Literal[
    "raw",
    "normalized",
    "centered_positive",
    "binary_positive",
]

RewardTransform = Literal[
    "identity",
    "log1p",
    "sqrt",
]

HistoryAggregation = Literal[
    "mean",
    "max",
    "sum",
]


@dataclass(frozen=True, slots=True)
class AmazonLogRewardConfig:
    """
    Amazon-style logged-interaction reward for semantic-ID trajectories.

    Reward semantics:

        1. Current positive target:
              reward = rating / verified_purchase based target reward.

        2. Candidate item appears in the user's history:
              reward = aggregate over all previous rating entries for that item.

        3. Candidate item is unobserved for the user:
              reward = unobserved_reward.

    This avoids treating historical positive items as uniformly bad random
    negatives.

    Required batch fields for full behavior:
        - history_item_ids: [B, H]
        - history_ratings: [B, H]
        - target_ratings: [B]

    Optional batch fields:
        - history_verified_purchase: [B, H]
        - target_verified_purchase: [B]
        - history_mask: [B, H]
    """

    # Candidate ID resolution.
    candidate_item_key: str = "candidate_item_ids"
    pad_item_id: int = 0

    # Target fields.
    target_rating_key: str = "target_ratings"
    target_verified_key: str = "target_verified_purchase"

    # History fields.
    history_item_key: str = "history_item_ids"
    history_rating_key: str = "history_ratings"
    history_verified_key: str = "history_verified_purchase"
    history_mask_key: str = "history_mask"

    # Reward floors.
    reward_epsilon: float = 1e-6
    unobserved_reward: float = 1e-3

    # Rating processing.
    rating_mode: RatingMode = "normalized"
    rating_min: float = 1.0
    rating_max: float = 5.0
    rating_neutral: float = 3.0
    positive_threshold: float = 4.0

    # Verified-purchase processing.
    verified_bonus: float = 0.25
    unverified_multiplier: float = 1.0

    # Logged reward transform.
    transform: RewardTransform = "log1p"
    log_scale: float = 1.0

    # How to combine repeated ratings for the same user-item pair in history.
    history_aggregation: HistoryAggregation = "mean"

    # Global scaling.
    interaction_weight: float = 1.0

    # Strictness.
    require_target_rating: bool = True
    require_history_features: bool = True
    allow_missing_candidate_item_ids: bool = False


class AmazonLogRewardModel(nn.Module):
    """
    GFlowGR-style logged reward model for Amazon review data.

    For each candidate item c and user u:

        - if c is the current target:
              use target rating and optional target verified_purchase;

        - else if c appears in the user's history:
              use the average/max/sum of all logged rating entries for c
              in that user's history, optionally boosted by verified_purchase;

        - else:
              assign a small unobserved reward.

    The model expects candidate atomic item IDs in either:

        trajectories.extra["candidate_item_ids"]

    or, if candidates only provide semantic table rows:

        trajectories.candidate_item_row_ids
        + item_id_by_semantic_row

    This is important for RandomSemanticIDCandidateSampler, whose negatives
    are sampled as semantic-table rows.
    """

    def __init__(
        self,
        cfg: AmazonLogRewardConfig | None = None,
        *,
        item_id_by_semantic_row: torch.Tensor | None = None,
        item_id_to_semantic_row: torch.Tensor | None = None,
        num_semantic_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or AmazonLogRewardConfig()

        if item_id_by_semantic_row is not None:
            row_to_item = item_id_by_semantic_row.long()
        elif item_id_to_semantic_row is not None:
            row_to_item = self._invert_item_id_to_semantic_row(
                item_id_to_semantic_row.long(),
                num_semantic_rows=num_semantic_rows,
            )
        else:
            row_to_item = torch.empty(0, dtype=torch.long)

        self.register_buffer("item_id_by_semantic_row", row_to_item, persistent=True)

    def forward(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        device = trajectories.candidate_token_ids.device
        is_positive = trajectories.is_positive.to(device).bool()

        if is_positive.ndim != 2:
            raise ValueError(
                "trajectories.is_positive must have shape [B, N]. "
                f"Got {tuple(is_positive.shape)}."
            )

        bsz, n_candidates = is_positive.shape

        candidate_item_ids = self._resolve_candidate_item_ids(
            batch=batch,
            trajectories=trajectories,
            device=device,
            expected_shape=(bsz, n_candidates),
        )

        invalid_candidates = self._invalid_candidate_mask(candidate_item_ids)

        rewards = torch.full(
            (bsz, n_candidates),
            fill_value=float(self.cfg.unobserved_reward),
            device=device,
            dtype=torch.float32,
        )

        # 1. Assign rewards for candidates that appeared in the user's history.
        history_reward, has_history_reward = self._history_rewards(
            batch=batch,
            candidate_item_ids=candidate_item_ids,
            invalid_candidates=invalid_candidates,
            device=device,
            bsz=bsz,
        )
        rewards = torch.where(has_history_reward, history_reward, rewards)

        # 2. Override current target candidates with target reward.
        target_reward = self._target_rewards(
            batch=batch,
            device=device,
            bsz=bsz,
        )
        if target_reward is not None:
            rewards = torch.where(is_positive, target_reward.unsqueeze(1), rewards)

        # 3. Invalid non-positive candidates should not receive accidental reward.
        rewards = torch.where(
            invalid_candidates & ~is_positive,
            torch.full_like(rewards, float(self.cfg.unobserved_reward)),
            rewards,
        )

        rewards = rewards * float(self.cfg.interaction_weight)
        rewards = rewards.clamp_min(float(self.cfg.reward_epsilon))
        return rewards

    def _target_rewards(
        self,
        *,
        batch: Batch,
        device: torch.device,
        bsz: int,
    ) -> torch.Tensor | None:
        target_rating = self._get_optional_tensor(
            batch,
            preferred=self.cfg.target_rating_key,
            aliases=(
                "target_rating",
                "target_ratings",
                "rating",
                "ratings",
            ),
        )

        if target_rating is None:
            if self.cfg.require_target_rating:
                raise KeyError(
                    "AmazonLogRewardModel requires target ratings. "
                    f"Missing key {self.cfg.target_rating_key!r} and aliases."
                )
            return None

        target_rating = target_rating.to(device).float().reshape(bsz)

        target_verified = self._get_optional_tensor(
            batch,
            preferred=self.cfg.target_verified_key,
            aliases=(
                "target_verified_purchase",
                "target_verified_purchases",
                "target_verified",
                "verified_purchase",
                "verified_purchases",
            ),
        )
        if target_verified is not None:
            target_verified = target_verified.to(device).reshape(bsz)

        base = self._base_interaction_value(
            rating=target_rating,
            verified=target_verified,
        )
        return self._apply_transform(base)

    def _history_rewards(
        self,
        *,
        batch: Batch,
        candidate_item_ids: torch.Tensor,
        invalid_candidates: torch.Tensor,
        device: torch.device,
        bsz: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_item_ids = self._get_optional_tensor(
            batch,
            preferred=self.cfg.history_item_key,
            aliases=(
                "history_item_ids",
                "history_items",
            ),
        )
        history_ratings = self._get_optional_tensor(
            batch,
            preferred=self.cfg.history_rating_key,
            aliases=(
                "history_ratings",
                "history_rating",
            ),
        )

        if history_item_ids is None or history_ratings is None:
            if self.cfg.require_history_features:
                missing = []
                if history_item_ids is None:
                    missing.append(self.cfg.history_item_key)
                if history_ratings is None:
                    missing.append(self.cfg.history_rating_key)
                raise KeyError(
                    "AmazonLogRewardModel requires history item IDs and "
                    f"history ratings. Missing: {missing}."
                )

            empty_reward = torch.zeros_like(candidate_item_ids, dtype=torch.float32)
            empty_mask = torch.zeros_like(candidate_item_ids, dtype=torch.bool)
            return empty_reward, empty_mask

        history_item_ids = history_item_ids.to(device).long()
        history_ratings = history_ratings.to(device).float()

        if history_item_ids.ndim != 2:
            raise ValueError(
                "history_item_ids must have shape [B, H]. "
                f"Got {tuple(history_item_ids.shape)}."
            )
        if history_ratings.shape != history_item_ids.shape:
            raise ValueError(
                "history_ratings must have the same shape as history_item_ids. "
                f"Got history_ratings={tuple(history_ratings.shape)}, "
                f"history_item_ids={tuple(history_item_ids.shape)}."
            )
        if history_item_ids.size(0) != bsz:
            raise ValueError(
                f"history batch size {history_item_ids.size(0)} does not match "
                f"candidate batch size {bsz}."
            )

        history_verified = self._get_optional_tensor(
            batch,
            preferred=self.cfg.history_verified_key,
            aliases=(
                "history_verified_purchase",
                "history_verified_purchases",
                "history_verified",
            ),
        )
        if history_verified is not None:
            history_verified = history_verified.to(device)
            if history_verified.shape != history_item_ids.shape:
                raise ValueError(
                    "history_verified_purchase must have the same shape as "
                    "history_item_ids. "
                    f"Got {tuple(history_verified.shape)} and "
                    f"{tuple(history_item_ids.shape)}."
                )

        history_mask = self._history_mask(
            batch=batch,
            history_item_ids=history_item_ids,
            device=device,
        )

        base_history_value = self._base_interaction_value(
            rating=history_ratings,
            verified=history_verified,
        )

        # matches[b, n, h] is true when candidate n for user b is the same
        # item as history position h.
        matches = candidate_item_ids.unsqueeze(-1).eq(history_item_ids.unsqueeze(1))
        matches = matches & history_mask.unsqueeze(1)
        matches = matches & ~invalid_candidates.unsqueeze(-1)

        aggregated, has_observed = self._aggregate_history_values(
            matches=matches,
            history_values=base_history_value,
        )

        reward = self._apply_transform(aggregated)
        return reward, has_observed

    def _base_interaction_value(
        self,
        *,
        rating: torch.Tensor,
        verified: torch.Tensor | None,
    ) -> torch.Tensor:
        rating = rating.float()

        if self.cfg.rating_mode == "raw":
            base = rating.clamp_min(0.0)

        elif self.cfg.rating_mode == "normalized":
            denom = max(float(self.cfg.rating_max - self.cfg.rating_min), 1e-12)
            base = ((rating - float(self.cfg.rating_min)) / denom).clamp(0.0, 1.0)

        elif self.cfg.rating_mode == "centered_positive":
            denom = max(float(self.cfg.rating_max - self.cfg.rating_neutral), 1e-12)
            base = ((rating - float(self.cfg.rating_neutral)) / denom).clamp(0.0, 1.0)

        elif self.cfg.rating_mode == "binary_positive":
            base = (rating >= float(self.cfg.positive_threshold)).float()

        else:
            raise ValueError(f"Unknown rating_mode: {self.cfg.rating_mode}")

        if verified is not None:
            verified_float = verified.to(base.device).float()
            verified_bool = verified_float > 0
            multiplier = torch.where(
                verified_bool,
                torch.full_like(base, 1.0 + float(self.cfg.verified_bonus)),
                torch.full_like(base, float(self.cfg.unverified_multiplier)),
            )
            base = base * multiplier

        return base.clamp_min(0.0)

    def _apply_transform(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(0.0)

        if self.cfg.transform == "identity":
            return value

        if self.cfg.transform == "log1p":
            return torch.log1p(float(self.cfg.log_scale) * value)

        if self.cfg.transform == "sqrt":
            return torch.sqrt(value)

        raise ValueError(f"Unknown reward transform: {self.cfg.transform}")

    def _aggregate_history_values(
        self,
        *,
        matches: torch.Tensor,
        history_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # matches:        [B, N, H]
        # history_values: [B, H]
        matches_f = matches.to(history_values.dtype)
        counts = matches_f.sum(dim=-1)
        has_observed = counts > 0

        if self.cfg.history_aggregation == "mean":
            summed = (matches_f * history_values.unsqueeze(1)).sum(dim=-1)
            aggregated = summed / counts.clamp_min(1.0)

        elif self.cfg.history_aggregation == "sum":
            aggregated = (matches_f * history_values.unsqueeze(1)).sum(dim=-1)

        elif self.cfg.history_aggregation == "max":
            neg_inf = torch.full(
                matches.shape,
                -float("inf"),
                device=history_values.device,
                dtype=history_values.dtype,
            )
            values = torch.where(
                matches,
                history_values.unsqueeze(1).expand_as(matches_f),
                neg_inf,
            )
            aggregated = values.max(dim=-1).values
            aggregated = torch.where(
                has_observed,
                aggregated,
                torch.zeros_like(aggregated),
            )

        else:
            raise ValueError(
                f"Unknown history_aggregation: {self.cfg.history_aggregation}"
            )

        aggregated = torch.where(
            has_observed,
            aggregated,
            torch.zeros_like(aggregated),
        )
        return aggregated, has_observed

    def _history_mask(
        self,
        *,
        batch: Batch,
        history_item_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        mask = self._get_optional_tensor(
            batch,
            preferred=self.cfg.history_mask_key,
            aliases=(
                "history_mask",
                "history_item_mask",
            ),
        )
        if mask is not None:
            mask = mask.to(device).bool()
            if mask.shape != history_item_ids.shape:
                raise ValueError(
                    "history_mask must have the same shape as history_item_ids. "
                    f"Got {tuple(mask.shape)} and {tuple(history_item_ids.shape)}."
                )
            return mask

        return history_item_ids.ne(int(self.cfg.pad_item_id))

    def _resolve_candidate_item_ids(
        self,
        *,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
        device: torch.device,
        expected_shape: tuple[int, int],
    ) -> torch.Tensor:
        # Preferred path: samplers such as SVD/SASRec score-based samplers
        # should store atomic item IDs here.
        extra = trajectories.extra or {}
        candidate_item_ids = extra.get(self.cfg.candidate_item_key)

        if candidate_item_ids is not None:
            if not torch.is_tensor(candidate_item_ids):
                candidate_item_ids = torch.as_tensor(candidate_item_ids)
            candidate_item_ids = candidate_item_ids.to(device).long()
            return self._reshape_candidate_item_ids(
                candidate_item_ids,
                expected_shape=expected_shape,
            )

        # Fallback path: random semantic sampler returns semantic table rows.
        rows = trajectories.candidate_item_row_ids
        if rows is not None and self.item_id_by_semantic_row.numel() > 0:
            rows = rows.to(device).long()
            if rows.shape != expected_shape:
                raise ValueError(
                    "candidate_item_row_ids must have shape [B, N]. "
                    f"Expected {expected_shape}, got {tuple(rows.shape)}."
                )

            row_to_item = self.item_id_by_semantic_row.to(device)
            valid = (rows >= 0) & (rows < row_to_item.numel())
            safe_rows = rows.clamp_min(0).clamp_max(max(row_to_item.numel() - 1, 0))
            mapped = row_to_item[safe_rows]
            return torch.where(
                valid,
                mapped,
                torch.full_like(mapped, -1),
            )

        if self.cfg.allow_missing_candidate_item_ids:
            return torch.full(
                expected_shape,
                -1,
                device=device,
                dtype=torch.long,
            )

        raise KeyError(
            "AmazonLogRewardModel could not resolve candidate item IDs. "
            f"Expected trajectories.extra[{self.cfg.candidate_item_key!r}] "
            "or candidate_item_row_ids plus item_id_by_semantic_row."
        )

    @staticmethod
    def _reshape_candidate_item_ids(
        candidate_item_ids: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
    ) -> torch.Tensor:
        if candidate_item_ids.shape == expected_shape:
            return candidate_item_ids

        if candidate_item_ids.ndim == 1 and candidate_item_ids.numel() == expected_shape[0] * expected_shape[1]:
            return candidate_item_ids.reshape(expected_shape)

        raise ValueError(
            "candidate_item_ids must have shape [B, N]. "
            f"Expected {expected_shape}, got {tuple(candidate_item_ids.shape)}."
        )

    def _invalid_candidate_mask(self, candidate_item_ids: torch.Tensor) -> torch.Tensor:
        return (
            candidate_item_ids.lt(0)
            | candidate_item_ids.eq(int(self.cfg.pad_item_id))
        )

    @staticmethod
    def _get_optional_tensor(
        batch: Batch,
        *,
        preferred: str,
        aliases: tuple[str, ...] = (),
    ) -> torch.Tensor | None:
        keys = (preferred, *aliases)
        seen: set[str] = set()

        for key in keys:
            if not key or key in seen:
                continue
            seen.add(key)

            if key not in batch:
                continue

            value = batch[key]
            if torch.is_tensor(value):
                return value

            try:
                return torch.as_tensor(value)
            except Exception as exc:
                raise TypeError(f"Batch key {key!r} cannot be converted to a tensor.") from exc

        return None

    @staticmethod
    def _invert_item_id_to_semantic_row(
        item_id_to_semantic_row: torch.Tensor,
        *,
        num_semantic_rows: int | None,
    ) -> torch.Tensor:
        item_id_to_semantic_row = item_id_to_semantic_row.long()

        valid = item_id_to_semantic_row >= 0
        if not valid.any():
            size = int(num_semantic_rows or 0)
            return torch.full((size,), -1, dtype=torch.long)

        max_row = int(item_id_to_semantic_row[valid].max().item())
        size = int(num_semantic_rows) if num_semantic_rows is not None else max_row + 1
        size = max(size, max_row + 1)

        row_to_item = torch.full((size,), -1, dtype=torch.long)
        item_ids = valid.nonzero(as_tuple=True)[0]
        rows = item_id_to_semantic_row[valid]
        row_to_item[rows] = item_ids
        return row_to_item