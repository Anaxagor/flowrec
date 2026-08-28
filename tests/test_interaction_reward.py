"""Time-aware store lookups and the store-backed interaction rewards."""

import numpy as np
import pandas as pd
import pytest
import torch

from flowrec.data.interaction_store import (
    InteractionLogStore,
    build_interaction_store,
    build_positive_history_store,
)
from flowrec.rewards.interaction import (
    InteractionRewardConfig,
    InteractionRewardModel,
    PositiveRewardConfig,
    PositiveRewardModel,
)
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


def _store(events: list[dict], *, num_users: int = 4, vocab: int = 10) -> InteractionLogStore:
    frame = pd.DataFrame(events)
    return InteractionLogStore(frame, num_users=num_users, item_vocab_size=vocab)


class TestTimeAwareLookups:
    def test_strict_cutoff_excludes_event_at_t(self) -> None:
        store = _store(
            [
                {"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 100, "rating": 5.0},
            ]
        )
        assert store.has_pair_before("review", [0], [1], [101]).tolist() == [True]
        assert store.has_pair_before("review", [0], [1], [100]).tolist() == [False]
        assert store.has_pair_before("review", [0], [1], [100], strict=False).tolist() == [True]
        assert store.has_pair_before("review", [0], [1], [99]).tolist() == [False]

    def test_payload_before_uses_only_past_events(self) -> None:
        store = _store(
            [
                {"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 2.0},
                {"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 20, "rating": 5.0},
            ]
        )
        assert store.payload_for_pairs_before("review", [0], [1], [15], "rating")[0] == 2.0
        assert store.payload_for_pairs_before("review", [0], [1], [25], "rating")[0] == 5.0
        assert store.payload_for_pairs_before("review", [0], [1], [25], "rating", agg="max")[0] == 5.0
        assert np.isnan(store.payload_for_pairs_before("review", [0], [1], [5], "rating")[0])

    def test_count_pairs_before(self) -> None:
        store = _store(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10},
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 20},
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 30},
            ]
        )
        assert store.count_pairs_before("listen", [0], [1], [25]).tolist() == [2]
        assert store.count_pairs_before("listen", [0], [1], [30]).tolist() == [2]
        assert store.count_pairs_before("listen", [0], [1], [30], strict=False).tolist() == [3]
        assert store.count_pairs_before("listen", [0], [1], [5]).tolist() == [0]
        assert store.count_pairs_before("listen", [1], [1], [100]).tolist() == [0]

    def test_absent_pair_gets_default(self) -> None:
        store = _store(
            [{"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 2.0}]
        )
        out = store.payload_for_pairs_before("review", [1, 0], [1, 2], [100, 100], "rating", default=-1.0)
        assert out.tolist() == [-1.0, -1.0]


class TestPositiveHistoryStore:
    def test_membership_is_strictly_before_t(self) -> None:
        frame = pd.DataFrame(
            {"user_id": [0, 0], "item_id": [3, 4], "timestamp": [50, 200]}
        )
        store = build_positive_history_store(frame, num_users=2, item_vocab_size=10)
        assert store.has_pair_before("positive", [0], [3], [100]).tolist() == [True]
        assert store.has_pair_before("positive", [0], [3], [50]).tolist() == [False]
        assert store.has_pair_before("positive", [0], [4], [100]).tolist() == [False]

    def test_empty_frame(self) -> None:
        frame = pd.DataFrame({"user_id": [], "item_id": [], "timestamp": []})
        store = build_positive_history_store(frame, num_users=2, item_vocab_size=10)
        assert len(store) == 0


def _trajectories(candidate_item_ids: torch.Tensor, is_positive: torch.Tensor) -> SemanticTrajectoryBatch:
    b, n = candidate_item_ids.shape
    return SemanticTrajectoryBatch(
        candidate_token_ids=torch.zeros(b, n, 2, dtype=torch.long),
        candidate_token_mask=torch.ones(b, n, 2, dtype=torch.bool),
        is_positive=is_positive,
        extra={"candidate_item_ids": candidate_item_ids},
    )


def _batch(user_ids, timestamps) -> dict:
    return {
        "user_ids": torch.as_tensor(user_ids, dtype=torch.long),
        "target_timestamps": torch.as_tensor(timestamps, dtype=torch.long),
    }


class TestPositiveRewardModel:
    def test_binary_split(self) -> None:
        model = PositiveRewardModel(PositiveRewardConfig(positive_reward=1.0, negative_reward=0.01))
        traj = _trajectories(
            torch.tensor([[1, 2, 3]]), torch.tensor([[True, False, False]])
        )
        rewards = model({}, traj)
        assert rewards[0].tolist() == pytest.approx([1.0, 0.01, 0.01])


class TestAmazonReward:
    def _model(self, positive_store=None, **cfg) -> InteractionRewardModel:
        store = _store(
            [
                {"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 5.0},
                {"user_id": 0, "item_id": 2, "event_type": "review", "timestamp": 10, "rating": 2.0},
                {"user_id": 0, "item_id": 3, "event_type": "review", "timestamp": 500, "rating": 5.0},
                {"user_id": 1, "item_id": 1, "event_type": "review", "timestamp": 100, "rating": 4.0},
            ]
        )
        return InteractionRewardModel(
            store,
            InteractionRewardConfig(dataset="amazon", **cfg),
            positive_store=positive_store,
        )

    def test_rating_evidence_with_time_cutoff(self) -> None:
        model = self._model()
        cfg = model.cfg
        # user 0 at t=100: item1 high, item2 low, item3 in the future, item4 unseen
        candidates = torch.tensor([[1, 2, 3, 4]])
        is_positive = torch.zeros(1, 4, dtype=torch.bool)
        rewards = model(_batch([0], [100]), _trajectories(candidates, is_positive))
        assert rewards[0].tolist() == pytest.approx(
            [cfg.high_rating_reward, cfg.low_rating_reward, cfg.unobserved_reward, cfg.unobserved_reward]
        )

    def test_event_at_target_time_is_not_visible(self) -> None:
        model = self._model()
        candidates = torch.tensor([[1]])
        rewards = model(_batch([1], [100]), _trajectories(candidates, torch.zeros(1, 1, dtype=torch.bool)))
        assert rewards[0, 0].item() == pytest.approx(model.cfg.unobserved_reward)
        rewards = model(_batch([1], [101]), _trajectories(candidates, torch.zeros(1, 1, dtype=torch.bool)))
        assert rewards[0, 0].item() == pytest.approx(model.cfg.high_rating_reward)

    def test_invalid_candidates_stay_neutral(self) -> None:
        model = self._model()
        candidates = torch.tensor([[-1, 0]])
        rewards = model(
            _batch([0], [100]), _trajectories(candidates, torch.zeros(1, 2, dtype=torch.bool))
        )
        assert rewards[0].tolist() == pytest.approx([1.0, 1.0])

    def test_positive_scored_from_prior_evidence(self) -> None:
        model = self._model()
        cfg = model.cfg
        # Both positive at t=100: item2 has a prior low rating, item4 nothing.
        candidates = torch.tensor([[2, 4]])
        is_positive = torch.tensor([[True, True]])
        rewards = model(_batch([0], [100]), _trajectories(candidates, is_positive))
        assert rewards[0].tolist() == pytest.approx(
            [cfg.low_rating_reward, cfg.unobserved_reward]
        )

    def test_positive_repeat_purchase_earns_history_reward(self) -> None:
        positives = pd.DataFrame({"user_id": [0], "item_id": [5], "timestamp": [50]})
        model = self._model(
            positive_store=build_positive_history_store(positives, num_users=4, item_vocab_size=10)
        )
        rewards = model(
            _batch([0], [100]),
            _trajectories(torch.tensor([[5]]), torch.tensor([[True]])),
        )
        assert rewards[0, 0].item() == pytest.approx(model.cfg.positive_history_reward)

    def test_positive_own_review_at_t_is_invisible(self) -> None:
        # user1's only review of item1 is at t=100; as the positive target at
        # that moment it must fall back, not see its own outcome.
        model = self._model()
        traj = _trajectories(torch.tensor([[1]]), torch.tensor([[True]]))
        rewards = model(_batch([1], [100]), traj)
        assert rewards[0, 0].item() == pytest.approx(model.cfg.unobserved_reward)
        rewards = model(_batch([1], [101]), traj)
        assert rewards[0, 0].item() == pytest.approx(model.cfg.high_rating_reward)

    def test_positive_constant_mode_ignores_evidence(self) -> None:
        model = self._model(positive_scoring="constant", positive_reward=1.5)
        candidates = torch.tensor([[2, 4]])
        is_positive = torch.tensor([[True, True]])
        rewards = model(_batch([0], [100]), _trajectories(candidates, is_positive))
        assert rewards[0].tolist() == pytest.approx([1.5, 1.5])

    def test_positive_history_membership(self) -> None:
        positives = pd.DataFrame(
            {"user_id": [0, 0], "item_id": [5, 6], "timestamp": [50, 300]}
        )
        model = self._model(
            positive_store=build_positive_history_store(positives, num_users=4, item_vocab_size=10)
        )
        cfg = model.cfg
        # item5 in positive history before t; item6 only after t; item2 has
        # explicit low-rating store evidence which beats membership.
        candidates = torch.tensor([[5, 6, 2]])
        rewards = model(_batch([0], [100]), _trajectories(candidates, torch.zeros(1, 3, dtype=torch.bool)))
        assert rewards[0].tolist() == pytest.approx(
            [cfg.positive_history_reward, cfg.unobserved_reward, cfg.low_rating_reward]
        )

    def test_semantic_row_fallback_path(self) -> None:
        store = _store(
            [{"user_id": 0, "item_id": 7, "event_type": "review", "timestamp": 10, "rating": 1.0}]
        )
        model = InteractionRewardModel(
            store,
            InteractionRewardConfig(dataset="amazon"),
            item_id_by_semantic_row=torch.tensor([7, 8]),
        )
        trajectories = SemanticTrajectoryBatch(
            candidate_token_ids=torch.zeros(1, 2, 2, dtype=torch.long),
            candidate_token_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            is_positive=torch.zeros(1, 2, dtype=torch.bool),
            candidate_item_row_ids=torch.tensor([[0, 1]]),
        )
        rewards = model(_batch([0], [100]), trajectories)
        assert rewards[0].tolist() == pytest.approx([model.cfg.low_rating_reward, 1.0])


class TestYambdaLegacyReward:
    def _model(self, events: list[dict], positive_store=None) -> InteractionRewardModel:
        return InteractionRewardModel(
            _store(events),
            InteractionRewardConfig(dataset="yambda", yambda_scoring="legacy"),
            positive_store=positive_store,
        )

    def _score(self, model: InteractionRewardModel, user: int, item: int, t: int) -> float:
        rewards = model(
            _batch([user], [t]),
            _trajectories(torch.tensor([[item]]), torch.zeros(1, 1, dtype=torch.bool)),
        )
        return float(rewards[0, 0])

    def test_like_and_dislike_states(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 10, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 2, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None},
            ]
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(model.cfg.like_reward)
        assert self._score(model, 0, 2, 100) == pytest.approx(model.cfg.dislike_reward)
        assert self._score(model, 0, 1, 10) == pytest.approx(1.0)

    def test_revocation_states(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 10, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 1, "event_type": "unlike", "timestamp": 20, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 2, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 2, "event_type": "undislike", "timestamp": 20, "played_ratio_pct": None},
            ]
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(model.cfg.unlike_reward)
        assert self._score(model, 0, 2, 100) == pytest.approx(model.cfg.undislike_reward)
        assert self._score(model, 0, 1, 15) == pytest.approx(model.cfg.like_reward)

    def test_later_feedback_wins_when_both_states_active(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 10, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 1, "event_type": "dislike", "timestamp": 20, "played_ratio_pct": None},
            ]
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(model.cfg.dislike_reward)
        assert self._score(model, 0, 1, 15) == pytest.approx(model.cfg.like_reward)

    def test_sub_threshold_listen_grades_with_played_pct(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 25.0},
                {"user_id": 0, "item_id": 2, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 0.0},
            ]
        )
        cfg = model.cfg
        expected_mid = cfg.listen_reward_floor + 0.5 * (1.0 - cfg.listen_reward_floor)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected_mid)
        assert self._score(model, 0, 2, 100) == pytest.approx(cfg.listen_reward_floor)

    def test_repeat_listens_soften_the_negative(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 10.0},
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 20, "played_ratio_pct": 10.0},
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 30, "played_ratio_pct": 10.0},
                {"user_id": 0, "item_id": 2, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 10.0},
            ]
        )
        cfg = model.cfg
        # item1: max pct 10 + 2 repeat listens * 10 pct-points -> effective 30
        effective = (10.0 + 2 * cfg.repeat_listen_bonus_pct) / cfg.listen_pct_threshold
        expected = cfg.listen_reward_floor + effective * (1.0 - cfg.listen_reward_floor)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected)
        # A single listen at the same ratio stays lower.
        single = cfg.listen_reward_floor + (10.0 / cfg.listen_pct_threshold) * (
            1.0 - cfg.listen_reward_floor
        )
        assert self._score(model, 0, 2, 100) == pytest.approx(single)
        assert self._score(model, 0, 1, 100) > self._score(model, 0, 2, 100)

    def test_many_listens_cap_at_neutral(self) -> None:
        events = [
            {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10 + i, "played_ratio_pct": 20.0}
            for i in range(20)
        ]
        model = self._model(events)
        assert self._score(model, 0, 1, 1000) == pytest.approx(1.0)

    def test_explicit_feedback_overrides_listen_and_history(self) -> None:
        positives = pd.DataFrame({"user_id": [0], "item_id": [1], "timestamp": [5]})
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 5.0},
                {"user_id": 0, "item_id": 1, "event_type": "dislike", "timestamp": 20, "played_ratio_pct": None},
            ],
            positive_store=build_positive_history_store(positives, num_users=4, item_vocab_size=10),
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(model.cfg.dislike_reward)

    def test_history_membership_beats_listen_evidence(self) -> None:
        positives = pd.DataFrame({"user_id": [0], "item_id": [1], "timestamp": [5]})
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 5.0},
            ],
            positive_store=build_positive_history_store(positives, num_users=4, item_vocab_size=10),
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(model.cfg.positive_history_reward)

    def test_rewards_are_bounded(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None}]
        )
        score = self._score(model, 0, 1, 100)
        assert model.cfg.reward_clip_min <= score <= model.cfg.reward_clip_max


class TestYambdaEngagementReward:
    """State-anchored bands positioned by engagement (the default scoring)."""

    def _model(self, events: list[dict], positive_store=None, **cfg) -> InteractionRewardModel:
        if not events:
            frame = pd.DataFrame(
                {
                    "user_id": pd.Series(dtype="int64"),
                    "item_id": pd.Series(dtype="int64"),
                    "event_type": pd.Series(dtype="object"),
                    "timestamp": pd.Series(dtype="int64"),
                }
            )
            store = InteractionLogStore(frame, num_users=4, item_vocab_size=10)
        else:
            store = _store(events)
        return InteractionRewardModel(
            store,
            InteractionRewardConfig(dataset="yambda", **cfg),
            positive_store=positive_store,
        )

    def _score(
        self, model: InteractionRewardModel, user: int, item: int, t: int, *, positive: bool = False
    ) -> float:
        is_positive = torch.tensor([[positive]])
        rewards = model(
            _batch([user], [t]),
            _trajectories(torch.tensor([[item]]), is_positive),
        )
        return float(rewards[0, 0])

    @staticmethod
    def _expected(cfg: InteractionRewardConfig, band: tuple[float, float], p: float, n: int) -> float:
        r = 0.0
        if n > 1:
            r = min(1.0, np.log(n) / np.log(cfg.repeat_saturation_listens))
        e = cfg.completion_weight * p + (1.0 - cfg.completion_weight) * r
        return band[0] + band[1] * e

    @staticmethod
    def _positive_store(rows: list[tuple[int, int, int]]):
        frame = pd.DataFrame(
            {
                "user_id": [r[0] for r in rows],
                "item_id": [r[1] for r in rows],
                "timestamp": [r[2] for r in rows],
            }
        )
        return build_positive_history_store(frame, num_users=4, item_vocab_size=10)

    def test_no_evidence_is_neutral(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 20.0}]
        )
        assert self._score(model, 0, 2, 100) == pytest.approx(1.0)
        assert self._score(model, 1, 1, 100) == pytest.approx(1.0)

    def test_single_skip_is_soft_negative(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 10.0}]
        )
        cfg = model.cfg
        expected = self._expected(cfg, cfg.listened_band, 0.1, 1)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected)
        assert self._score(model, 0, 1, 100) < 1.0

    def test_single_full_listen_is_soft_positive(self) -> None:
        model = self._model([], positive_store=self._positive_store([(0, 1, 10)]))
        cfg = model.cfg
        expected = self._expected(cfg, cfg.listened_band, cfg.full_listen_pct / 100.0, 1)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected)
        assert self._score(model, 0, 1, 100) > 1.0

    def test_repeat_listens_saturate(self) -> None:
        cfg = InteractionRewardConfig(dataset="yambda")
        sat = int(cfg.repeat_saturation_listens)
        store_8 = self._positive_store([(0, 1, 10 + i) for i in range(sat)])
        store_20 = self._positive_store([(0, 1, 10 + i) for i in range(20)])
        model_8 = self._model([], positive_store=store_8)
        model_20 = self._model([], positive_store=store_20)
        expected = self._expected(cfg, cfg.listened_band, cfg.full_listen_pct / 100.0, sat)
        assert self._score(model_8, 0, 1, 100) == pytest.approx(expected)
        assert self._score(model_20, 0, 1, 100) == pytest.approx(expected)

    def test_like_band_scales_with_engagement(self) -> None:
        like = {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 10, "played_ratio_pct": None}
        bare = self._model([like])
        engaged = self._model(
            [like], positive_store=self._positive_store([(0, 1, 20 + i) for i in range(8)])
        )
        cfg = bare.cfg
        assert self._score(bare, 0, 1, 100) == pytest.approx(cfg.like_band[0])
        expected = self._expected(cfg, cfg.like_band, cfg.full_listen_pct / 100.0, 8)
        assert self._score(engaged, 0, 1, 100) == pytest.approx(expected)
        assert self._score(engaged, 0, 1, 100) > self._score(bare, 0, 1, 100)

    def test_dislike_stays_negative_even_when_engaged(self) -> None:
        dislike = {"user_id": 0, "item_id": 1, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None}
        engaged = self._model(
            [dislike], positive_store=self._positive_store([(0, 1, 20 + i) for i in range(8)])
        )
        cfg = engaged.cfg
        expected = self._expected(cfg, cfg.dislike_band, cfg.full_listen_pct / 100.0, 8)
        assert self._score(engaged, 0, 1, 100) == pytest.approx(expected)
        assert self._score(engaged, 0, 1, 100) < 1.0

    def test_revocation_bands(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 10, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 1, "event_type": "unlike", "timestamp": 20, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 2, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 2, "event_type": "undislike", "timestamp": 20, "played_ratio_pct": None},
            ]
        )
        cfg = model.cfg
        assert self._score(model, 0, 1, 100) == pytest.approx(cfg.unlike_band[0])
        assert self._score(model, 0, 2, 100) == pytest.approx(cfg.undislike_band[0])
        # Before the revocation the original state still holds.
        assert self._score(model, 0, 1, 15) == pytest.approx(cfg.like_band[0])
        assert self._score(model, 0, 2, 15) == pytest.approx(cfg.dislike_band[0])

    def test_overplayed_ratio_is_clipped(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 250.0}]
        )
        cfg = model.cfg
        expected = self._expected(cfg, cfg.listened_band, 1.0, 1)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected)

    def test_aux_pct_above_full_listen_credit_wins(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 95.0}],
            positive_store=self._positive_store([(0, 1, 20)]),
        )
        cfg = model.cfg
        expected = self._expected(cfg, cfg.listened_band, 0.95, 2)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected)

    def test_band_ordering_under_equal_engagement(self) -> None:
        events = []
        for item, event_type in [(1, "like"), (2, "dislike")]:
            events.append(
                {"user_id": 0, "item_id": item, "event_type": event_type, "timestamp": 10, "played_ratio_pct": None}
            )
        positives = self._positive_store(
            [(0, item, 20 + i) for item in (1, 2, 3) for i in range(4)]
        )
        model = self._model(events, positive_store=positives)
        liked = self._score(model, 0, 1, 100)
        disliked = self._score(model, 0, 2, 100)
        listened = self._score(model, 0, 3, 100)
        unseen = self._score(model, 0, 4, 100)
        assert liked > listened > unseen > disliked

    # --- leakage guarantees: nothing at or after t may move the reward.

    def test_listen_at_t_is_invisible(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 100, "played_ratio_pct": 10.0}]
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(1.0)
        assert self._score(model, 0, 1, 101) < 1.0

    def test_feedback_at_t_is_invisible(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 100, "played_ratio_pct": None},
                {"user_id": 0, "item_id": 2, "event_type": "dislike", "timestamp": 100, "played_ratio_pct": None},
            ]
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(1.0)
        assert self._score(model, 0, 2, 100) == pytest.approx(1.0)
        assert self._score(model, 0, 1, 101) == pytest.approx(model.cfg.like_band[0])

    def test_full_listen_at_t_is_invisible(self) -> None:
        model = self._model([], positive_store=self._positive_store([(0, 1, 100)]))
        assert self._score(model, 0, 1, 100) == pytest.approx(1.0)
        assert self._score(model, 0, 1, 101) > 1.0

    def test_engagement_counts_only_past_listens(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 10.0},
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 20, "played_ratio_pct": 10.0},
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 30, "played_ratio_pct": 40.0},
            ]
        )
        cfg = model.cfg
        # At t=25 only the first two listens (max pct 10) exist.
        expected = self._expected(cfg, cfg.listened_band, 0.1, 2)
        assert self._score(model, 0, 1, 25) == pytest.approx(expected)
        # At t=100 all three (max pct 40) count.
        expected_all = self._expected(cfg, cfg.listened_band, 0.4, 3)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected_all)

    # --- positive-target scoring from the same evidence.

    def test_positive_target_scored_from_past_engagement(self) -> None:
        like = {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 50, "played_ratio_pct": None}
        model = self._model(
            [like], positive_store=self._positive_store([(0, 1, 100 + i) for i in range(8)])
        )
        cfg = model.cfg
        expected = self._expected(cfg, cfg.like_band, cfg.full_listen_pct / 100.0, 8)
        assert self._score(model, 0, 1, 1000, positive=True) == pytest.approx(expected)
        # Evidence scoring is symmetric: the same item scores identically
        # whether it is the target or a sampled candidate.
        assert self._score(model, 0, 1, 1000, positive=False) == pytest.approx(expected)

    def test_first_encounter_positive_falls_back(self) -> None:
        # The positive store holds only the target event itself (at t) — the
        # strict cutoff hides it, so the positive falls back to neutral.
        model = self._model([], positive_store=self._positive_store([(0, 1, 100)]))
        assert self._score(model, 0, 1, 100, positive=True) == pytest.approx(1.0)
        # A later target of the same pair sees it as one past full listen.
        cfg = model.cfg
        expected = self._expected(cfg, cfg.listened_band, cfg.full_listen_pct / 100.0, 1)
        assert self._score(model, 0, 1, 101, positive=True) == pytest.approx(expected)

    def test_positive_fallback_uses_positive_reward(self) -> None:
        # Store evidence exists only for another user's pair; the positive
        # target itself has none and takes the configured fallback.
        model = self._model(
            [], positive_store=self._positive_store([(1, 2, 10)]), positive_reward=1.4
        )
        assert self._score(model, 0, 1, 100, positive=True) == pytest.approx(1.4)

    def test_positive_constant_mode_ignores_evidence(self) -> None:
        like = {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 50, "played_ratio_pct": None}
        model = self._model(
            [like],
            positive_store=self._positive_store([(0, 1, 100 + i) for i in range(8)]),
            positive_scoring="constant",
            positive_reward=1.4,
        )
        assert self._score(model, 0, 1, 1000, positive=True) == pytest.approx(1.4)

    def test_rewards_stay_inside_clip_range(self) -> None:
        like = {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 10, "played_ratio_pct": None}
        dislike = {"user_id": 0, "item_id": 2, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None}
        positives = self._positive_store([(0, 1, 20 + i) for i in range(30)])
        model = self._model([like, dislike], positive_store=positives)
        cfg = model.cfg
        for item in (1, 2):
            score = self._score(model, 0, item, 1000)
            assert cfg.reward_clip_min <= score <= cfg.reward_clip_max


def test_missing_batch_keys_raise() -> None:
    store = _store([{"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 5.0}])
    model = InteractionRewardModel(store, InteractionRewardConfig(dataset="amazon"))
    with pytest.raises(KeyError, match="target_timestamps"):
        model(
            {"user_ids": torch.tensor([0])},
            _trajectories(torch.tensor([[1]]), torch.zeros(1, 1, dtype=torch.bool)),
        )


class TestEmptyStore:
    def _empty_store(self) -> InteractionLogStore:
        return build_interaction_store(
            pd.DataFrame(), user2id={}, item2id={}, num_users=5, item_vocab_size=10
        )

    def test_reward_is_neutral_and_warns(self) -> None:
        store = self._empty_store()
        with pytest.warns(UserWarning, match="empty interaction store"):
            amazon = InteractionRewardModel(store, InteractionRewardConfig(dataset="amazon"))
        with pytest.warns(UserWarning, match="empty interaction store"):
            yambda = InteractionRewardModel(store, InteractionRewardConfig(dataset="yambda"))

        batch = _batch([0, 1], [100, 200])
        traj = _trajectories(
            torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.zeros(2, 3, dtype=torch.bool)
        )
        for model in (amazon, yambda):
            rewards = model(batch, traj)
            assert torch.all(rewards == 1.0), rewards

    def test_no_warning_with_positive_store(self) -> None:
        import warnings as _warnings

        positives = pd.DataFrame({"user_id": [0], "item_id": [1], "timestamp": [5]})
        positive_store = build_positive_history_store(positives, num_users=5, item_vocab_size=10)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            InteractionRewardModel(
                self._empty_store(),
                InteractionRewardConfig(dataset="amazon"),
                positive_store=positive_store,
            )
