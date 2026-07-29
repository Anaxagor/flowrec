"""Reward from the auxiliary interaction log, with a strict time cutoff.

For every sampled candidate item c of user u at target time t, this model
looks up u's auxiliary events for c **strictly before t** in the
InteractionLogStore and converts them into a multiplicative reward modulator
(the MultiplicativeRewardModel convention):

    - no evidence            -> 1.0  (base reward untouched)
    - evidence-positive logs -> > 1.0 (high unverified rating, like, ...)
    - evidence-negative logs -> < 1.0 (low rating, dislike, short listen, ...)

Never looking at or past t is the leakage rule: an event at the target's own
timestamp (or later) is not past information. The positive target candidate
itself is left at 1.0 by default — the base reward (SASRec / similarity)
carries it, and the log modulator only re-weights sampled candidates on
logged evidence.

Default magnitudes are chosen to be informative but bounded in log-space
(|log r| <= ~0.7), so one log hit re-weights, rather than dominates, the
dense base reward. Yambda values follow the multi-event analysis
(examples/yambda_reward_analysis.ipynb): dislikes are the strongest
rejection signal, likes the strongest approval, unlike/undislike are weak
revocation signals, and a sub-threshold listen is a graded soft negative in
the played-ratio percentage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from flowrec.data.interaction_store import InteractionLogStore
from flowrec.models.base import Batch
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

from .amazon_log import AmazonLogRewardModel

InteractionLogDataset = Literal["amazon", "yambda"]


@dataclass
class InteractionLogRewardConfig:
    dataset: InteractionLogDataset = "amazon"

    # Candidate / batch plumbing (same contract as AmazonLogRewardModel).
    candidate_item_key: str = "candidate_item_ids"
    pad_item_id: int = 0
    user_key: str = "user_ids"
    target_timestamp_key: str = "target_timestamps"

    # Time cutoff. strict=True excludes events at exactly the target time.
    strict_time_cutoff: bool = True

    # Multiplicative-modulator values; 1.0 means "no logged evidence".
    unobserved_reward: float = 1.0
    # Reward for the positive target candidate. None -> unobserved_reward,
    # letting the base reward carry the positive.
    positive_reward: float | None = None
    # Safety clamp in multiplier space.
    reward_clip_min: float = 0.25
    reward_clip_max: float = 2.0

    # --- Amazon: auxiliary reviews (e.g. unverified under the
    # verified_purchase criterion, or low-rated under a rating criterion).
    review_event_type: str = "review"
    rating_column: str = "rating"
    high_rating_threshold: float = 4.0
    high_rating_reward: float = 1.25
    low_rating_reward: float = 0.6

    # --- Yambda: multi-event feedback.
    like_reward: float = 1.3
    dislike_reward: float = 0.5
    unlike_reward: float = 0.85
    undislike_reward: float = 1.1
    listen_event_type: str = "listen"
    played_pct_column: str = "played_ratio_pct"
    # Sub-threshold listens grade linearly from listen_reward_floor at 0%
    # played up to 1.0 (uninformative) at listen_pct_threshold.
    listen_pct_threshold: float = 50.0
    listen_reward_floor: float = 0.7


class InteractionLogRewardModel(nn.Module):
    """Time-cutoff logged-evidence reward over an InteractionLogStore.

    Candidate item IDs are resolved exactly like AmazonLogRewardModel:
    ``trajectories.extra["candidate_item_ids"]`` when present, else
    ``trajectories.candidate_item_row_ids`` mapped through
    ``item_id_by_semantic_row``.
    """

    # Reuse the candidate-resolution contract instead of duplicating it.
    _resolve_candidate_item_ids = AmazonLogRewardModel._resolve_candidate_item_ids
    _reshape_candidate_item_ids = staticmethod(AmazonLogRewardModel._reshape_candidate_item_ids)
    _invalid_candidate_mask = AmazonLogRewardModel._invalid_candidate_mask

    def __init__(
        self,
        store: InteractionLogStore,
        cfg: InteractionLogRewardConfig | None = None,
        *,
        item_id_by_semantic_row: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or InteractionLogRewardConfig()
        self.store = store
        if len(store) == 0:
            warnings.warn(
                "InteractionLogRewardModel got an empty interaction store — the "
                "reward will be a constant "
                f"{float(self.cfg.unobserved_reward)} for every candidate. Check "
                "positive_criterion: with None (or a criterion the data never "
                "violates, e.g. verified_purchase == True on beauty_2014 where "
                "the field is always True) no auxiliary events exist.",
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
        invalid = self._invalid_candidate_mask(candidate_item_ids)

        for key in (self.cfg.user_key, self.cfg.target_timestamp_key):
            if key not in batch:
                raise KeyError(
                    f"InteractionLogRewardModel requires batch[{key!r}]; the "
                    "prefix-target collator provides user_ids and "
                    "target_timestamps."
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

    # -- per-dataset evidence scoring ---------------------------------------

    def _amazon_rewards(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        rating = self.store.payload_for_pairs_before(
            cfg.review_event_type,
            users,
            items,
            times,
            cfg.rating_column,
            agg="last",
            strict=cfg.strict_time_cutoff,
        )
        rewards = np.full(users.shape, float(cfg.unobserved_reward), dtype=float)
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
        strict = cfg.strict_time_cutoff

        def last_ts(event_type: str) -> np.ndarray:
            return self.store.payload_for_pairs_before(
                event_type, users, items, times, "timestamp",
                agg="last", strict=strict, default=-np.inf,
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

        # Graded soft negative from sub-threshold listens (strongest played
        # ratio seen before t); explicit feedback states override it below.
        pct = self.store.payload_for_pairs_before(
            cfg.listen_event_type, users, items, times, cfg.played_pct_column,
            agg="max", strict=strict,
        )
        listened = ~np.isnan(pct)
        graded = float(cfg.listen_reward_floor) + (
            np.clip(pct, 0.0, float(cfg.listen_pct_threshold))
            / float(cfg.listen_pct_threshold)
        ) * (1.0 - float(cfg.listen_reward_floor))
        rewards[listened] = graded[listened]

        rewards[undislike_state] = float(cfg.undislike_reward)
        rewards[unlike_state] = float(cfg.unlike_reward)
        rewards[dislike_state] = float(cfg.dislike_reward)
        rewards[like_state] = float(cfg.like_reward)
        return rewards
