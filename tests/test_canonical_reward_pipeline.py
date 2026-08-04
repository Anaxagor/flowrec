import torch
import pytest

from flowrec.rewards.canonical import (
    CanonicalRewardSpec,
    apply_reward_power,
    row_max_normalize,
    sasrec_softplus_rewards,
    weighted_average_rewards,
)
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache


def test_canonical_sasrec_is_positive_and_bounded():
    logits = torch.tensor([[-4.0, 0.0, 3.0], [1.0, -2.0, 0.5]])
    rewards = row_max_normalize(sasrec_softplus_rewards(logits, 1e-6), 1e-6)
    assert torch.all(rewards > 0)
    assert torch.all(rewards <= 1)
    torch.testing.assert_close(rewards.amax(dim=1), torch.ones(2))


def test_reward_power_controls_pairwise_ratio():
    base = torch.tensor([[0.25, 1.0]])
    assert apply_reward_power(base, 0.5, 1e-6)[0, 1] / apply_reward_power(base, 0.5, 1e-6)[0, 0] < 4
    assert apply_reward_power(base, 4.0, 1e-6)[0, 1] / apply_reward_power(base, 4.0, 1e-6)[0, 0] > 4


def test_equal_weight_mix_is_convex_average():
    components = {
        "sasrec": torch.tensor([[1.0, 0.0]]),
        "rating": torch.tensor([[0.0, 1.0]]),
        "similarity": torch.tensor([[0.5, 0.5]]),
    }
    actual = weighted_average_rewards(components, {name: 1.0 for name in components}, 1e-6)
    torch.testing.assert_close(actual, torch.tensor([[0.5, 0.5]]))


def test_spec_canonicalizes_weights_and_manifest():
    spec = CanonicalRewardSpec.from_mapping(
        {"reward_type": "mix", "mix_weights": {"similarity": 2, "sasrec": 1, "rating": 3}}
    )
    assert spec.mix_weights == (("sasrec", 1.0), ("rating", 3.0), ("similarity", 2.0))
    assert spec.to_manifest()["reward_pipeline"] == "canonical_v1"


def test_invalid_canonical_settings_fail():
    for mapping in (
        {"reward_type": "other"},
        {"sasrec_adapter": "sigmoid"},
        {"component_normalization": "minmax"},
        {"mix_weights": {"sasrec": -1}},
        {"mix_weights": {"sasrec": 0, "rating": 0, "similarity": 0}},
    ):
        try:
            CanonicalRewardSpec.from_mapping(mapping)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid spec to fail: {mapping}")


def test_training_rejects_cache_with_different_canonical_spec(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text('{"canonical_reward": null}')
    cfg = GFlowConfig.from_dict({"reward_pipeline": "canonical_v1", "reward_type": "sasrec"})
    with pytest.raises(ValueError, match="does not match"):
        _PrecomputedRewardCache(
            cache_dir,
            top_m=1,
            probability="top",
            expected_canonical_reward=cfg.canonical_reward_manifest(),
        )


if __name__ == "__main__":
    test_canonical_sasrec_is_positive_and_bounded()
    test_reward_power_controls_pairwise_ratio()
    test_equal_weight_mix_is_convex_average()
    test_spec_canonicalizes_weights_and_manifest()
    test_invalid_canonical_settings_fail()
    print("canonical reward tests passed")
