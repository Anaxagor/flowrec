"""MiniOneRec reward selection uses the same reward models as GFlow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig
from flowrec.ft_methods.mini_one_rec.method import MiniOneRecMethod
from flowrec.rewards.interaction import PositiveRewardModel
from flowrec.rewards.mix import MixedRewardModel, MultiplicativeRewardModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


def _trajectories() -> SemanticTrajectoryBatch:
    return SemanticTrajectoryBatch(
        candidate_token_ids=torch.tensor([[[1, 1], [2, 2], [3, 3]]]),
        candidate_token_mask=torch.ones((1, 3, 2), dtype=torch.bool),
        is_positive=torch.tensor([[True, False, False]]),
        candidate_item_row_ids=torch.tensor([[0, 1, 2]]),
    )


class FixedReward(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, batch, trajectories):
        return self.values


def test_binary_and_powered_external_rewards_replace_rule_rank():
    trajectories = _trajectories()
    method = MiniOneRecMethod(
        MiniOneRecConfig(reward_type="binary", reward_power=0.5),
        reward_model=PositiveRewardModel(),
    )
    rewards, parts = method._compute_rewards({}, trajectories)
    assert torch.allclose(rewards, torch.tensor([[1.0, 0.001**0.5, 0.001**0.5]]))
    assert parts["rank_reward_mean"].item() == 0.0


def test_external_reward_is_frozen_and_combined_pointwise():
    trajectories = _trajectories()
    first = FixedReward([[2.0, 4.0, 8.0]])
    second = FixedReward([[1.0, 0.5, 0.25]])
    model = MultiplicativeRewardModel([first, second])
    method = MiniOneRecMethod(MiniOneRecConfig(reward_type="multiply"), reward_model=model)
    method.train()
    rewards, _ = method._compute_rewards({}, trajectories)
    assert torch.allclose(rewards, torch.tensor([[2.0, 2.0, 2.0]]))
    assert not rewards.requires_grad
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not model.training

    mixed = MixedRewardModel(
        [FixedReward([[2.0, 4.0, 8.0]]), FixedReward([[1.0, 0.5, 0.25]])],
        normalizers=["none", "none"],
    )
    mix_method = MiniOneRecMethod(MiniOneRecConfig(reward_type="mix"), reward_model=mixed)
    mixed_rewards, _ = mix_method._compute_rewards({}, trajectories)
    assert torch.allclose(mixed_rewards, torch.tensor([[3.0, 4.5, 8.25]]))


def test_invalid_external_rewards_fail_before_advantage_normalization():
    method = MiniOneRecMethod(
        MiniOneRecConfig(reward_type="sasrec"),
        reward_model=FixedReward([[1.0, float("nan"), 2.0]]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        method._compute_rewards({}, _trajectories())


def test_reward_selection_cannot_silently_ignore_a_model():
    with pytest.raises(ValueError, match="reward_model was provided"):
        MiniOneRecMethod(MiniOneRecConfig(), reward_model=FixedReward([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="requires a reward_model"):
        MiniOneRecMethod(MiniOneRecConfig(reward_type="interaction"))
    with pytest.raises(ValueError, match="reward_power"):
        MiniOneRecConfig(reward_type="rule_rank", reward_power=0.25)


def test_launch_configs_select_their_named_reward(monkeypatch):
    pytest.importorskip("polars")
    from scripts import run_tiger

    selected = []

    def fake_reward_builder(name, **kwargs):
        selected.append(name)
        return FixedReward([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(run_tiger, "build_reward_model", fake_reward_builder)
    monkeypatch.setattr(run_tiger, "build_sasrec_teacher_model", lambda *a, **kw: object())
    monkeypatch.setattr(run_tiger, "build_svd_scoring_model", lambda **kw: object())
    model = SimpleNamespace(config_dict=lambda: {"num_beams": 20, "num_return_sequences": 20})
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "MiniOneRec"
    paths = sorted(config_dir.glob("beauty_*.json"))
    assert len(paths) >= 8
    for path in paths:
        config = run_tiger.load_json(path)
        method = run_tiger.build_method(
            config, model, data=SimpleNamespace(), train_loader=None, config_path=path
        )
        assert isinstance(method, MiniOneRecMethod)
        assert method.cfg.reward_type == path.stem.removeprefix("beauty_").replace(
            "sim_embed", "similarity"
        ).replace("sim_hier", "similarity")
        if method.cfg.reward_type != "rule_rank":
            assert selected[-1] == method.cfg.reward_type


def test_runner_rejects_conflicting_reward_type_and_model(monkeypatch):
    pytest.importorskip("polars")
    from scripts import run_tiger

    model = SimpleNamespace(config_dict=lambda: {"num_beams": 20, "num_return_sequences": 20})
    config = {
        "model": {"top_k": 20},
        "method": {"name": "mini-one-rec", "reward_type": "rule_rank", "reward_model": "sasrec"},
    }
    with pytest.raises(ValueError, match="different rewards"):
        run_tiger.build_method(
            config, model, data=SimpleNamespace(), train_loader=None, config_path=Path("dummy.json")
        )


def test_shared_builder_combines_binary_and_item_prior(tmp_path):
    pytest.importorskip("polars")
    import pandas as pd

    from scripts import run_tiger

    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {"asin": ["A", "A", "B", "B", "C"], "rating": [5, 5, 1, 1, 3]}
    ).to_parquet(reviews_path)
    data = SimpleNamespace(
        reviews_path=reviews_path,
        cfg=SimpleNamespace(item_key="asin"),
        item2id={"A": 1, "B": 2, "C": 3},
        item_vocab_size=4,
        semantic_ids={"A": [1, 1], "B": [2, 2], "C": [3, 3]},
    )
    cfg = {
        "normalize": "none",
        "components": [
            {"name": "binary", "weight": 1, "reward": {"negative_reward": 0.1}},
            {"name": "item_prior", "weight": 1},
        ],
    }
    model = run_tiger.build_reward_model(
        "mix", data=data, train_loader=None, config_path=tmp_path / "config.json",
        seed=42, reward_cfg=cfg, svd_cfg={}, svd_model=None, sasrec_model=None,
        reward_epsilon=1e-6, allow_default_rating=False,
    )
    rewards = model({}, _trajectories())
    assert rewards.shape == (1, 3)
    assert rewards[0, 0] > rewards[0, 2] > rewards[0, 1]
