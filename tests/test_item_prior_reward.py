"""GFlow's full-raw-review item prior is reusable on generated Mini groups."""

import math

import pandas as pd
import pytest
import torch

from flowrec.rewards.item_prior import (
    ItemPriorRatingReward,
    ItemPriorRatingRewardConfig,
    build_item_rating_prior,
)
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


def _trajectories(rows, *, item_ids=None):
    rows = torch.tensor(rows)
    b, n = rows.shape
    return SemanticTrajectoryBatch(
        candidate_token_ids=torch.zeros(b, n, 2, dtype=torch.long),
        candidate_token_mask=torch.ones(b, n, 2, dtype=torch.bool),
        is_positive=torch.zeros(b, n, dtype=torch.bool),
        candidate_item_row_ids=rows,
        extra={} if item_ids is None else {"candidate_item_ids": torch.tensor(item_ids)},
    )


def test_prior_matches_gflow_formula_using_all_raw_items():
    # The unmapped item participates in global mean and sample residual std.
    frame = pd.DataFrame({"asin": ["a", "a", "b", "outside"], "rating": [5, 3, 1, 2]})
    cfg = ItemPriorRatingRewardConfig(beta=0.25)
    table = build_item_rating_prior(frame, {"a": 2, "b": 1}, 4, cfg=cfg)
    global_mean = 11 / 4
    residuals = torch.tensor([
        (8 + 10 * global_mean) / 12 - global_mean,
        (1 + 10 * global_mean) / 11 - global_mean,
        (2 + 10 * global_mean) / 11 - global_mean,
    ], dtype=torch.float64)
    expected = (0.25 * residuals / residuals.std()).exp().clamp(0.5, 2)
    assert table.tolist() == pytest.approx([1, expected[1].item(), expected[0].item(), 1])


@pytest.mark.parametrize("items,ratings", [
    (["a", "a"], [1, 5]),
    (["a", "b", "a"], [3, 3, 3]),
])
def test_single_item_and_constant_ratings_are_neutral(items, ratings):
    frame = pd.DataFrame({"asin": items, "rating": ratings})
    table = build_item_rating_prior(frame, {"a": 1, "b": 2}, 3)
    assert torch.equal(table, torch.ones(3))


def test_nonfinite_and_non_numeric_reviews_do_not_poison_prior():
    good = pd.DataFrame({"asin": ["a", "b"], "rating": [1, 5]})
    bad = pd.DataFrame({"asin": ["a", "b", "c", None], "rating": [float("inf"), "bad", None, 1]})
    actual = build_item_rating_prior(pd.concat([good, bad], ignore_index=True), {"a": 1, "b": 2}, 4)
    expected = build_item_rating_prior(good, {"a": 1, "b": 2}, 4)
    assert torch.equal(actual, expected)


def test_no_usable_ratings_raises_clear_error():
    with pytest.raises(ValueError, match="finite raw review"):
        build_item_rating_prior(pd.DataFrame({"asin": ["a"], "rating": [math.nan]}), {"a": 1}, 2)


def test_large_beta_clips_without_overflow():
    frame = pd.DataFrame({"asin": ["a", "b"], "rating": [1, 5]})
    table = build_item_rating_prior(frame, {"a": 1, "b": 2}, 3, cfg=ItemPriorRatingRewardConfig(beta=1e6))
    assert table.tolist() == pytest.approx([1, 0.5, 2])


def test_semantic_row_mapping_rejects_both_out_of_range_directions():
    model = ItemPriorRatingReward(
        torch.tensor([1, 0.6, 1.8]), item_id_by_semantic_row=torch.tensor([2, 1, -1, 7])
    )
    rewards = model({}, _trajectories([[0, 1, -1, 4, 2, 3]]))
    assert rewards.tolist()[0] == pytest.approx([1.8, 0.6, 1, 1, 1, 1])


def test_explicit_atomic_ids_take_precedence_and_invalid_ids_are_neutral():
    model = ItemPriorRatingReward(
        torch.tensor([8, 0.6, 1.8]), item_id_by_semantic_row=torch.tensor([2, 1])
    )
    rewards = model({}, _trajectories([[0, 1, 0, 1, 0]], item_ids=[[1, 2, 0, -1, 3]]))
    assert rewards.tolist()[0] == pytest.approx([0.6, 1.8, 1, 1, 1])


def test_reward_shape_and_mapping_are_validated():
    model = ItemPriorRatingReward(torch.ones(3))
    with pytest.raises(ValueError, match="candidate_item_ids"):
        model({}, _trajectories([[0, 1]]))
    with pytest.raises(ValueError, match="shape"):
        model({}, _trajectories([[0, 1]], item_ids=[[1]]))


def test_prior_is_fixed_and_independent_of_candidate_group():
    table = torch.tensor([1, 0.6, 1.8], requires_grad=True)
    model = ItemPriorRatingReward(table)
    pair = model({}, _trajectories([[0, 0]], item_ids=[[1, 2]]))
    singleton = model({}, _trajectories([[0]], item_ids=[[1]]))
    assert not pair.requires_grad
    assert pair[0, 0] == singleton[0, 0]


@pytest.mark.parametrize("kwargs", [
    {"shrinkage": -1}, {"beta": math.nan}, {"clip_min": 0},
    {"clip_min": 3}, {"reward_epsilon": math.inf},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ItemPriorRatingRewardConfig(**kwargs)
