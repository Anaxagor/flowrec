"""Time-aware store lookups and the interaction-log reward modulator."""

import numpy as np
import pandas as pd
import pytest
import torch

from flowrec.data.interaction_store import InteractionLogStore
from flowrec.rewards.interaction_log import (
    InteractionLogRewardConfig,
    InteractionLogRewardModel,
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
        # Between the events only the first is visible; after both, "last" is 5.
        assert store.payload_for_pairs_before("review", [0], [1], [15], "rating")[0] == 2.0
        assert store.payload_for_pairs_before("review", [0], [1], [25], "rating")[0] == 5.0
        assert store.payload_for_pairs_before("review", [0], [1], [25], "rating", agg="max")[0] == 5.0
        assert store.payload_for_pairs_before("review", [0], [1], [15], "rating", agg="max")[0] == 2.0
        assert np.isnan(store.payload_for_pairs_before("review", [0], [1], [5], "rating")[0])

    def test_absent_pair_gets_default(self) -> None:
        store = _store(
            [{"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 2.0}]
        )
        out = store.payload_for_pairs_before("review", [1, 0], [1, 2], [100, 100], "rating", default=-1.0)
        assert out.tolist() == [-1.0, -1.0]


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


class TestAmazonReward:
    def _model(self) -> InteractionLogRewardModel:
        store = _store(
            [
                {"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 5.0},
                {"user_id": 0, "item_id": 2, "event_type": "review", "timestamp": 10, "rating": 2.0},
                {"user_id": 0, "item_id": 3, "event_type": "review", "timestamp": 500, "rating": 5.0},
                {"user_id": 1, "item_id": 1, "event_type": "review", "timestamp": 100, "rating": 4.0},
            ]
        )
        return InteractionLogRewardModel(store, InteractionLogRewardConfig(dataset="amazon"))

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

    def test_positive_and_invalid_candidates_stay_neutral(self) -> None:
        model = self._model()
        candidates = torch.tensor([[2, -1, 0]])
        is_positive = torch.tensor([[True, False, False]])
        rewards = model(_batch([0], [100]), _trajectories(candidates, is_positive))
        # item2 has low-rating evidence, but it is the positive target -> neutral;
        # -1 and PAD are invalid -> neutral.
        assert rewards[0].tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_semantic_row_fallback_path(self) -> None:
        store = _store(
            [{"user_id": 0, "item_id": 7, "event_type": "review", "timestamp": 10, "rating": 1.0}]
        )
        model = InteractionLogRewardModel(
            store,
            InteractionLogRewardConfig(dataset="amazon"),
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


class TestYambdaReward:
    def _model(self, events: list[dict]) -> InteractionLogRewardModel:
        return InteractionLogRewardModel(
            _store(events), InteractionLogRewardConfig(dataset="yambda")
        )

    def _score(self, model: InteractionLogRewardModel, user: int, item: int, t: int) -> float:
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
        # Before the events happened: neutral.
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
        # Before the revocation, the original state is still active.
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
        # 25% of a 50% threshold -> halfway between the floor and neutral.
        expected_mid = cfg.listen_reward_floor + 0.5 * (1.0 - cfg.listen_reward_floor)
        assert self._score(model, 0, 1, 100) == pytest.approx(expected_mid)
        assert self._score(model, 0, 2, 100) == pytest.approx(cfg.listen_reward_floor)

    def test_explicit_feedback_overrides_listen_evidence(self) -> None:
        model = self._model(
            [
                {"user_id": 0, "item_id": 1, "event_type": "listen", "timestamp": 10, "played_ratio_pct": 5.0},
                {"user_id": 0, "item_id": 1, "event_type": "like", "timestamp": 20, "played_ratio_pct": None},
            ]
        )
        assert self._score(model, 0, 1, 100) == pytest.approx(model.cfg.like_reward)

    def test_rewards_are_bounded(self) -> None:
        model = self._model(
            [{"user_id": 0, "item_id": 1, "event_type": "dislike", "timestamp": 10, "played_ratio_pct": None}]
        )
        score = self._score(model, 0, 1, 100)
        assert model.cfg.reward_clip_min <= score <= model.cfg.reward_clip_max


def test_missing_batch_keys_raise() -> None:
    store = _store([{"user_id": 0, "item_id": 1, "event_type": "review", "timestamp": 10, "rating": 5.0}])
    model = InteractionLogRewardModel(store, InteractionLogRewardConfig(dataset="amazon"))
    with pytest.raises(KeyError, match="target_timestamps"):
        model(
            {"user_ids": torch.tensor([0])},
            _trajectories(torch.tensor([[1]]), torch.zeros(1, 1, dtype=torch.bool)),
        )


class TestEmptyStore:
    def _empty_store(self) -> InteractionLogStore:
        from flowrec.data.interaction_store import build_interaction_store

        return build_interaction_store(
            pd.DataFrame(), user2id={}, item2id={}, num_users=5, item_vocab_size=10
        )

    def test_lookups_degrade_to_defaults(self) -> None:
        store = self._empty_store()
        assert store.has_pair_before("review", [0], [1], [100]).tolist() == [False]
        assert np.isnan(store.payload_for_pairs_before("review", [0], [1], [100], "rating")[0])
        assert store.payload_for_pairs("review", [0], [1], "rating", default=-1.0)[0] == -1.0

    def test_reward_is_neutral_and_warns(self) -> None:
        store = self._empty_store()
        with pytest.warns(UserWarning, match="empty interaction store"):
            amazon = InteractionLogRewardModel(store, InteractionLogRewardConfig(dataset="amazon"))
        with pytest.warns(UserWarning, match="empty interaction store"):
            yambda = InteractionLogRewardModel(store, InteractionLogRewardConfig(dataset="yambda"))

        batch = _batch([0, 1], [100, 200])
        traj = _trajectories(
            torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.zeros(2, 3, dtype=torch.bool)
        )
        for model in (amazon, yambda):
            rewards = model(batch, traj)
            assert torch.all(rewards == 1.0), rewards
