"""Data-driven rewards over the InteractionLogStore, leakage-free by design.

Two models cover the "reward from logged data" role:

- :class:`PositiveRewardModel` — the base terminal reward. The positive
  target gets ``positive_reward``, every other candidate ``negative_reward``.
  It reads nothing but ``trajectories.is_positive``, so it cannot leak. Under
  the positive/auxiliary datamodule contract positives carry no payload
  (user / item / timestamp only), which makes this binary base the honest
  replacement for rating-keyed rewards.

- :class:`InteractionRewardModel` — a logged-evidence modulator around 1.0
  (the MultiplicativeRewardModel convention). For every sampled candidate
  item c of user u at target time t it consults:

    1. the auxiliary :class:`InteractionLogStore` — ratings, likes/dislikes,
       sub-threshold listens — via the ``*_before`` lookups, **strictly
       before t**, always. There is no configuration to disable the cutoff:
       an event at or after the target's own timestamp is not past
       information.
    2. optionally a positive-history store (see
       ``build_positive_history_store``) answering "did u positively
       interact with c before t". This only matters when the candidate
       sampler runs with ``exclude_history=False``; with exclusion on, no
       sampled negative can be in the user's positive history.

  The positive target itself takes ``positive_reward`` (``None`` keeps it
  neutral so a dense base reward carries it). Candidates without any logged
  evidence stay at ``unobserved_reward`` (1.0 = neutral).

Evidence precedence for a non-positive candidate, highest first:

    amazon:  rating before t (high/low around the threshold)
             > positive-history membership > unobserved
    yambda:  like/dislike final state before t (unlike/undislike revoke)
             > positive-history membership
             > sub-threshold listen evidence (graded by played ratio and
               boosted by how many times the track was listened before t)
             > unobserved
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from flowrec.data.interaction_store import POSITIVE_EVENT_TYPE, InteractionLogStore
from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

InteractionDataset = Literal["amazon", "yambda"]


@dataclass(frozen=True, slots=True)
class PositiveRewardConfig:
    positive_reward: float = 1.0
    negative_reward: float = 1e-3
    reward_epsilon: float = 1e-6


class PositiveRewardModel(nn.Module):
    """Binary base reward: positive target vs everything else."""

    def __init__(self, cfg: PositiveRewardConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PositiveRewardConfig()

    def forward(
        self,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
    ) -> torch.Tensor:
        is_positive = trajectories.is_positive.bool()
        rewards = torch.where(
            is_positive,
            torch.full_like(is_positive, float(self.cfg.positive_reward), dtype=torch.float32),
            torch.full_like(is_positive, float(self.cfg.negative_reward), dtype=torch.float32),
        )
        return rewards.clamp_min(float(self.cfg.reward_epsilon))


@dataclass(frozen=True, slots=True)
class InteractionRewardConfig:
    dataset: InteractionDataset = "amazon"

    # Candidate / batch plumbing.
    candidate_item_key: str = "candidate_item_ids"
    pad_item_id: int = 0
    user_key: str = "user_ids"
    target_timestamp_key: str = "target_timestamps"

    # Multiplicative-modulator values; 1.0 means "no logged evidence".
    unobserved_reward: float = 1.0
    # Reward for the positive target candidate. None -> unobserved_reward,
    # letting the base reward carry the positive.
    positive_reward: float | None = None
    # Candidate found in the user's positive history strictly before t
    # (reachable only when the sampler runs with exclude_history=False).
    positive_history_reward: float = 1.2
    # Safety clamp in multiplier space.
    reward_clip_min: float = 0.25
    reward_clip_max: float = 2.0

    # --- Amazon: auxiliary reviews before t.
    review_event_type: str = "review"
    rating_column: str = "rating"
    high_rating_threshold: float = 4.0
    high_rating_reward: float = 1.25
    low_rating_reward: float = 0.6

    # --- Yambda: multi-event feedback before t.
    like_reward: float = 1.3
    dislike_reward: float = 0.5
    unlike_reward: float = 0.85
    undislike_reward: float = 1.1
    listen_event_type: str = "listen"
    played_pct_column: str = "played_ratio_pct"
    # Listen evidence grades linearly from listen_reward_floor (0% played,
    # single listen) up to 1.0 (uninformative) at listen_pct_threshold.
    listen_pct_threshold: float = 50.0
    listen_reward_floor: float = 0.7
    # Every repeat listen before t adds this many effective played-ratio
    # points, so a track listened to many times drifts toward neutral even
    # when each listen was short. Capped at listen_pct_threshold.
    repeat_listen_bonus_pct: float = 10.0


class InteractionRewardModel(nn.Module):
    """Strictly-before-t logged-evidence reward over interaction stores.

    Candidate item IDs are taken from ``trajectories.extra["candidate_item_ids"]``
    when present, else ``trajectories.candidate_item_row_ids`` mapped through
    ``item_id_by_semantic_row``.
    """

    def __init__(
        self,
        store: InteractionLogStore,
        cfg: InteractionRewardConfig | None = None,
        *,
        positive_store: InteractionLogStore | None = None,
        item_id_by_semantic_row: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or InteractionRewardConfig()
        self.store = store
        self.positive_store = positive_store
        if len(store) == 0 and (positive_store is None or len(positive_store) == 0):
            warnings.warn(
                "InteractionRewardModel got an empty interaction store and no "
                "positive-history store — the reward will be a constant "
                f"{float(self.cfg.unobserved_reward)} for every candidate. "
                "Check positive_criterion: with None (or a criterion the data "
                "never violates) no auxiliary events exist.",
                stacklevel=2,
            )
        row_to_item = (
            item_id_by_semantic_row.long()
            if item_id_by_semantic_row is not None
            else torch.empty(0, dtype=torch.long)
        )
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
                f"trajectories.is_positive must have shape [B, N], got {tuple(is_positive.shape)}."
            )
        bsz, n_candidates = is_positive.shape

        candidate_item_ids = self._resolve_candidate_item_ids(
            batch=batch,
            trajectories=trajectories,
            device=device,
            expected_shape=(bsz, n_candidates),
        )
        invalid = (
            candidate_item_ids.lt(0)
            | candidate_item_ids.eq(int(self.cfg.pad_item_id))
        )

        for key in (self.cfg.user_key, self.cfg.target_timestamp_key):
            if key not in batch:
                raise KeyError(
                    f"InteractionRewardModel requires batch[{key!r}]; the "
                    "prefix-target collator provides user_ids and target_timestamps."
                )
        user_ids = batch[self.cfg.user_key].detach().cpu().numpy().astype(np.int64)
        timestamps = (
            batch[self.cfg.target_timestamp_key].detach().cpu().numpy().astype(np.int64)
        )

        flat_users = np.repeat(user_ids, n_candidates)
        flat_times = np.repeat(timestamps, n_candidates)
        flat_items = (
            candidate_item_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
        )
        # Out-of-range ids never match any stored pair, but keep keys sane.
        flat_items = np.clip(flat_items, 0, max(self.store.item_vocab_size - 1, 0))

        if self.cfg.dataset == "amazon":
            flat_rewards = self._amazon_rewards(flat_users, flat_items, flat_times)
        elif self.cfg.dataset == "yambda":
            flat_rewards = self._yambda_rewards(flat_users, flat_items, flat_times)
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset!r}")

        rewards = torch.as_tensor(
            flat_rewards.reshape(bsz, n_candidates), dtype=torch.float32, device=device
        )

        neutral = torch.full_like(rewards, float(self.cfg.unobserved_reward))
        rewards = torch.where(invalid, neutral, rewards)
        positive_fill = (
            neutral
            if self.cfg.positive_reward is None
            else torch.full_like(rewards, float(self.cfg.positive_reward))
        )
        rewards = torch.where(is_positive, positive_fill, rewards)
        return rewards.clamp(
            min=float(self.cfg.reward_clip_min), max=float(self.cfg.reward_clip_max)
        )

    # -- shared evidence -----------------------------------------------------

    def _positive_history_before(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        if self.positive_store is None or len(self.positive_store) == 0:
            return np.zeros(users.shape, dtype=bool)
        return self.positive_store.has_pair_before(
            POSITIVE_EVENT_TYPE, users, items, times, strict=True
        )

    # -- per-dataset evidence scoring ----------------------------------------

    def _amazon_rewards(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        rewards = np.full(users.shape, float(cfg.unobserved_reward), dtype=float)

        in_history = self._positive_history_before(users, items, times)
        rewards[in_history] = float(cfg.positive_history_reward)

        rating = self.store.payload_for_pairs_before(
            cfg.review_event_type,
            users,
            items,
            times,
            cfg.rating_column,
            agg="last",
            strict=True,
        )
        observed = ~np.isnan(rating)
        rewards[observed] = np.where(
            rating[observed] >= float(cfg.high_rating_threshold),
            float(cfg.high_rating_reward),
            float(cfg.low_rating_reward),
        )
        return rewards

    def _yambda_rewards(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg

        def last_ts(event_type: str) -> np.ndarray:
            return self.store.payload_for_pairs_before(
                event_type, users, items, times, "timestamp",
                agg="last", strict=True, default=-np.inf,
            )

        ts_like = last_ts("like")
        ts_unlike = last_ts("unlike")
        ts_dislike = last_ts("dislike")
        ts_undislike = last_ts("undislike")

        # Final feedback state before t: an unlike revokes an earlier like,
        # an undislike revokes an earlier dislike; when both like and dislike
        # states are active, the later event wins.
        like_state = np.isfinite(ts_like) & (ts_like > ts_unlike)
        dislike_state = np.isfinite(ts_dislike) & (ts_dislike > ts_undislike)
        both = like_state & dislike_state
        like_state = like_state & ~(both & (ts_dislike > ts_like))
        dislike_state = dislike_state & ~(both & (ts_like >= ts_dislike))

        # Revocation-only signals (state switched off before t).
        unlike_state = np.isfinite(ts_unlike) & ~like_state & ~dislike_state
        undislike_state = (
            np.isfinite(ts_undislike) & ~like_state & ~dislike_state & ~unlike_state
        )

        rewards = np.full(users.shape, float(cfg.unobserved_reward), dtype=float)

        # Graded soft negative from sub-threshold listens before t: the best
        # played ratio, plus a bonus per repeat listen — a track the user
        # kept coming back to is weaker negative evidence than a one-off
        # skip. Explicit feedback and history membership override it below.
        pct = self.store.payload_for_pairs_before(
            cfg.listen_event_type, users, items, times, cfg.played_pct_column,
            agg="max", strict=True,
        )
        count = self.store.count_pairs_before(
            cfg.listen_event_type, users, items, times, strict=True
        )
        listened = ~np.isnan(pct) & (count > 0)
        effective_pct = np.clip(
            np.nan_to_num(pct, nan=0.0)
            + float(cfg.repeat_listen_bonus_pct) * np.maximum(count - 1, 0),
            0.0,
            float(cfg.listen_pct_threshold),
        )
        graded = float(cfg.listen_reward_floor) + (
            effective_pct / float(cfg.listen_pct_threshold)
        ) * (1.0 - float(cfg.listen_reward_floor))
        rewards[listened] = graded[listened]

        in_history = self._positive_history_before(users, items, times)
        rewards[in_history] = float(cfg.positive_history_reward)

        rewards[undislike_state] = float(cfg.undislike_reward)
        rewards[unlike_state] = float(cfg.unlike_reward)
        rewards[dislike_state] = float(cfg.dislike_reward)
        rewards[like_state] = float(cfg.like_reward)
        return rewards

    # -- candidate resolution --------------------------------------------------

    def _resolve_candidate_item_ids(
        self,
        *,
        batch: Batch,
        trajectories: SemanticTrajectoryBatch,
        device: torch.device,
        expected_shape: tuple[int, int],
    ) -> torch.Tensor:
        extra = trajectories.extra or {}
        candidate_item_ids = extra.get(self.cfg.candidate_item_key)
        if candidate_item_ids is not None:
            if not torch.is_tensor(candidate_item_ids):
                candidate_item_ids = torch.as_tensor(candidate_item_ids)
            candidate_item_ids = candidate_item_ids.to(device).long()
            if candidate_item_ids.shape != expected_shape:
                if (
                    candidate_item_ids.ndim == 1
                    and candidate_item_ids.numel() == expected_shape[0] * expected_shape[1]
                ):
                    candidate_item_ids = candidate_item_ids.reshape(expected_shape)
                else:
                    raise ValueError(
                        "candidate_item_ids must have shape [B, N]. "
                        f"Expected {expected_shape}, got {tuple(candidate_item_ids.shape)}."
                    )
            return candidate_item_ids

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
            return torch.where(valid, mapped, torch.full_like(mapped, -1))

        raise KeyError(
            "InteractionRewardModel could not resolve candidate item IDs. "
            f"Expected trajectories.extra[{self.cfg.candidate_item_key!r}] "
            "or candidate_item_row_ids plus item_id_by_semantic_row."
        )
