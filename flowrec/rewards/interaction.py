"""Data-driven rewards over the InteractionLogStore, leakage-free by design.

Unified reward pipeline over logged interactions, for every dataset:

1. **Split.** ``partition_interactions(frame, positive_criterion)`` divides
   the raw log into the positive stream (payload stripped to
   user / item / timestamp) and the auxiliary stream (every payload column
   kept). The criterion string is the only dataset-specific part —
   ``"rating >= 4"`` for Amazon reviews, ``"event_type == 'listen' and
   played_ratio_pct >= 50"`` for Yambda.

2. **Binary logic** — :class:`PositiveRewardModel`, the base terminal
   reward: the positive target gets ``positive_reward``, every other
   candidate ``negative_reward``. It reads nothing but
   ``trajectories.is_positive``, so it is dataset-agnostic and cannot leak.

3. **Field-based logic** — :class:`InteractionRewardModel`, a
   logged-evidence modulator around 1.0 (the MultiplicativeRewardModel
   convention) whose evidence scoring is the per-dataset part: Amazon reads
   ratings, Yambda reads like/dislike state, played ratios, and repeat
   listens. For every candidate item c of user u at target time t it
   consults:

    1. the auxiliary :class:`InteractionLogStore` — ratings, likes/dislikes,
       sub-threshold listens — via the ``*_before`` lookups, **strictly
       before t**, always. There is no configuration to disable the cutoff:
       an event at or after the target's own timestamp is not past
       information.
    2. optionally a positive-history store (see
       ``build_positive_history_store``) answering "how often did u
       positively interact with c before t".

  **Positive candidates** follow ``positive_scoring``:

    - ``"evidence"`` (default): the positive target is scored from the same
      strictly-before-t evidence as any other candidate — a liked, replayed
      Yambda track or a repeat Amazon purchase earns more than a
      first-encounter positive. Its own event at t is excluded by the strict
      cutoff, so positives without prior evidence fall back to
      ``positive_reward`` (``None`` keeps them neutral so a dense base
      reward carries them).
    - ``"constant"``: every positive takes ``positive_reward`` flat.

  Candidates without any logged evidence stay at ``unobserved_reward``
  (1.0 = neutral).

Evidence precedence for a non-positive candidate, highest first:

    amazon:  rating before t (high/low around the threshold)
             > positive-history membership > unobserved
    yambda (yambda_scoring="engagement", default):
             the like/dislike/unlike/undislike final state before t picks a
             reward band; the position inside the band is set by an
             engagement score E in [0, 1] combining the best played ratio
             and a log-saturating repeat-listen count, both strictly before
             t. With no feedback state, listens alone span listened_band
             (soft negative for one skip, soft positive for repeated full
             plays); no evidence at all stays at unobserved_reward. Full
             (criterion-passing) listens are visible only through the
             positive-history store, which keeps no payload — each counts
             ``full_listen_pct`` toward completion.
    yambda (yambda_scoring="legacy"):
             like/dislike final state before t (unlike/undislike revoke)
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
    # How the positive target candidate is rewarded. "evidence" scores it
    # from the same strictly-before-t logged evidence as every other
    # candidate, falling back to positive_reward where none exists;
    # "constant" fills every positive with positive_reward flat.
    positive_scoring: Literal["evidence", "constant"] = "evidence"
    # Positive fallback / constant value. None -> unobserved_reward,
    # letting the base reward carry the positive.
    positive_reward: float | None = None
    # Candidate found in the user's positive history strictly before t.
    # Reachable with exclude_history=False, and also with exclusion on for
    # items older than the batch's max_history_len window — the sampler only
    # excludes the windowed history it sees.
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
    yambda_scoring: Literal["engagement", "legacy"] = "engagement"
    listen_event_type: str = "listen"
    played_pct_column: str = "played_ratio_pct"

    # --- Yambda "engagement" scoring. The feedback state picks a band
    # (base, span); the reward is base + span * E with the engagement score
    #   E = completion_weight * p + (1 - completion_weight) * r
    #   p = clip(best played ratio before t, 0, 100) / 100
    #   r = clip(log(n_listens_before_t) / log(repeat_saturation_listens), 0, 1)
    # Full listens live in the positive stream and reach the reward only via
    # the payload-free positive-history store: each counts full_listen_pct
    # toward p and adds to n. Saturation at 8 listens = the median listen
    # count of liked (user, track) pairs in the 50m multi_event logs.
    completion_weight: float = 0.6
    repeat_saturation_listens: float = 8.0
    full_listen_pct: float = 90.0
    like_band: tuple[float, float] = (1.2, 0.6)
    undislike_band: tuple[float, float] = (0.9, 0.3)
    listened_band: tuple[float, float] = (0.7, 0.6)
    unlike_band: tuple[float, float] = (0.6, 0.4)
    dislike_band: tuple[float, float] = (0.35, 0.35)

    # --- Yambda "legacy" scoring: flat per-state constants, listen evidence
    # only grades sub-threshold plays toward neutral.
    like_reward: float = 1.3
    dislike_reward: float = 0.5
    unlike_reward: float = 0.85
    undislike_reward: float = 1.1
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
            flat_rewards, flat_evidence = self._amazon_rewards(
                flat_users, flat_items, flat_times
            )
        elif self.cfg.dataset == "yambda":
            flat_rewards, flat_evidence = self._yambda_rewards(
                flat_users, flat_items, flat_times
            )
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset!r}")

        rewards = torch.as_tensor(
            flat_rewards.reshape(bsz, n_candidates), dtype=torch.float32, device=device
        )
        has_evidence = torch.as_tensor(
            flat_evidence.reshape(bsz, n_candidates), dtype=torch.bool, device=device
        )

        neutral = torch.full_like(rewards, float(self.cfg.unobserved_reward))
        rewards = torch.where(invalid, neutral, rewards)
        positive_fill = (
            neutral
            if self.cfg.positive_reward is None
            else torch.full_like(rewards, float(self.cfg.positive_reward))
        )
        if self.cfg.positive_scoring == "evidence":
            # Positives with strictly-before-t evidence keep their evidence
            # score; the rest (and invalid ids) take the fallback fill.
            fill_mask = is_positive & (~has_evidence | invalid)
        elif self.cfg.positive_scoring == "constant":
            fill_mask = is_positive
        else:
            raise ValueError(
                f"Unknown positive_scoring: {self.cfg.positive_scoring!r}; "
                "expected 'evidence' or 'constant'."
            )
        rewards = torch.where(fill_mask, positive_fill, rewards)
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
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return rewards, in_history | observed

    def _yambda_feedback_states(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mutually exclusive (like, dislike, unlike, undislike) state masks
        from events strictly before t."""

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
        return like_state, dislike_state, unlike_state, undislike_state

    def _yambda_engagement(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(engagement score E in [0, 1], listen count n), strictly before t.

        Completion uses the best played ratio among sub-criterion listens in
        the auxiliary store; a full listen (found in the payload-free
        positive-history store) floors completion at full_listen_pct.
        Repetition counts listens from both stores and saturates
        logarithmically at repeat_saturation_listens.
        """
        cfg = self.cfg
        pct = self.store.payload_for_pairs_before(
            cfg.listen_event_type, users, items, times, cfg.played_pct_column,
            agg="max", strict=True,
        )
        n_partial = self.store.count_pairs_before(
            cfg.listen_event_type, users, items, times, strict=True
        )
        if self.positive_store is not None and len(self.positive_store) > 0:
            n_full = self.positive_store.count_pairs_before(
                POSITIVE_EVENT_TYPE, users, items, times, strict=True
            )
        else:
            n_full = np.zeros(users.shape, dtype=np.int64)

        completion = np.clip(np.nan_to_num(pct, nan=0.0), 0.0, 100.0) / 100.0
        completion = np.where(
            n_full > 0,
            np.maximum(completion, float(cfg.full_listen_pct) / 100.0),
            completion,
        )
        n_listens = n_partial + n_full
        log_sat = np.log(max(float(cfg.repeat_saturation_listens), 1.0 + 1e-9))
        repetition = np.clip(
            np.log(np.maximum(n_listens, 1)) / log_sat, 0.0, 1.0
        )
        weight = float(cfg.completion_weight)
        engagement = weight * completion + (1.0 - weight) * repetition
        return engagement, n_listens

    def _yambda_rewards(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.yambda_scoring == "engagement":
            return self._yambda_rewards_engagement(users, items, times)
        if self.cfg.yambda_scoring == "legacy":
            return self._yambda_rewards_legacy(users, items, times)
        raise ValueError(
            f"Unknown yambda_scoring: {self.cfg.yambda_scoring!r}; "
            "expected 'engagement' or 'legacy'."
        )

    def _yambda_rewards_engagement(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        like_state, dislike_state, unlike_state, undislike_state = (
            self._yambda_feedback_states(users, items, times)
        )
        engagement, n_listens = self._yambda_engagement(users, items, times)

        rewards = np.full(users.shape, float(cfg.unobserved_reward), dtype=float)

        def apply_band(spec: tuple[float, float], mask: np.ndarray) -> None:
            base, span = float(spec[0]), float(spec[1])
            rewards[mask] = base + span * engagement[mask]

        any_state = like_state | dislike_state | unlike_state | undislike_state
        apply_band(cfg.listened_band, ~any_state & (n_listens > 0))
        apply_band(cfg.undislike_band, undislike_state)
        apply_band(cfg.unlike_band, unlike_state)
        apply_band(cfg.dislike_band, dislike_state)
        apply_band(cfg.like_band, like_state)
        return rewards, any_state | (n_listens > 0)

    def _yambda_rewards_legacy(
        self, users: np.ndarray, items: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        like_state, dislike_state, unlike_state, undislike_state = (
            self._yambda_feedback_states(users, items, times)
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
        has_evidence = (
            listened
            | in_history
            | like_state
            | dislike_state
            | unlike_state
            | undislike_state
        )
        return rewards, has_evidence

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
