import json

import pytest
import torch

from flowrec.candidates.sampler import CandidateSamplerConfig
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.gflow.losses.detailed_balance import detailed_balance_loss
from flowrec.ft_methods.gflow.losses.subtrajectory_balance import subtrajectory_balance_loss
from flowrec.ft_methods.gflow.losses.trajectory_balance import trajectory_balance_loss
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod


def test_gflow_config_accepts_generic_method_name():
    cfg = GFlowConfig.from_dict({"name": "gflow", "objective": "db"})

    assert cfg.objective == "db"


def test_full_catalog_cache_mmaps_shards_and_looks_up_unsorted_rows(tmp_path, monkeypatch):
    rewards = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    load_calls = []
    original_load = torch.load

    def recording_load(*args, **kwargs):
        load_calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    for shard_idx in range(2):
        shard_rewards = rewards[shard_idx : shard_idx + 1]
        torch.save(
            {
                "example_start": shard_idx,
                "num_examples": 1,
                "target_semantic_row_ids": torch.tensor([2 + shard_idx]),
                "full_rewards": shard_rewards,
                "reward_sum_power_2": shard_rewards.square().sum(dim=1, keepdim=True),
            },
            tmp_path / f"shard_{shard_idx:05d}.pt",
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "num_examples": 2,
                "stored_power_sums": [2.0],
                "shards": [
                    {"path": "shard_00000.pt", "example_start": 0, "num_examples": 1},
                    {"path": "shard_00001.pt", "example_start": 1, "num_examples": 1},
                ],
            }
        )
    )

    cache = _PrecomputedRewardCache(tmp_path, power=2.0)
    candidate_rows = torch.tensor([[-1, 0], [-1, 4]])
    selected, reward_sum, full = cache.select_candidates(
        torch.tensor([1, 0]), candidate_rows, device=torch.device("cpu"), include_full=True
    )

    assert torch.equal(selected, torch.tensor([[25.0, 4.0], [9.0, 25.0]]))
    assert torch.equal(reward_sum, torch.tensor([90.0, 55.0]))
    assert torch.equal(full, rewards[[1, 0]].square())
    assert not hasattr(cache, "full_rewards")
    assert load_calls and all(call.get("mmap") is True for call in load_calls)


def test_balance_losses_are_uniform_by_default():
    rewards = torch.tensor([[1.0, 0.01, 0.01, 0.01]])
    residuals = torch.tensor([[0.0, 2.0, 2.0, 2.0]])
    expected = torch.tensor(3.0)

    tb_loss, _ = trajectory_balance_loss(
        sequence_logprobs=rewards.log() + residuals,
        rewards=rewards,
        log_z=torch.zeros(1),
    )
    token_logprobs = residuals.unsqueeze(-1)
    zeros = torch.zeros_like(token_logprobs)
    db_loss, _ = detailed_balance_loss(token_logprobs, zeros, zeros, rewards)
    prefix_flows = torch.zeros((1, 4, 2))
    subtb_loss, _ = subtrajectory_balance_loss(token_logprobs, prefix_flows, rewards)

    assert torch.allclose(tb_loss, expected)
    assert torch.allclose(db_loss, expected)
    assert torch.allclose(subtb_loss, expected)


def test_gflow_weighting_defaults_to_uniform():
    assert GFlowConfig(reward_cache_path="unused").tb_loss_weighting == "uniform"


def test_removed_top_m_config_is_not_accepted():
    with pytest.raises(TypeError):
        GFlowConfig.from_dict({"reward_cache_top_m": 64})


@pytest.mark.parametrize(
    "key",
    [
        "tb_logz_source",
        "logz_init",
        "logz_context",
        "logz_vocab_size",
        "logz_vector_size",
        "logz_embedding_dim",
        "logz_hidden_dim",
        "logz_dropout",
        "reward_cache_transform",
        "sampling_temperature",
        "metric_prefix",
    ],
)
def test_live_reward_and_learned_logz_settings_are_removed(key):
    with pytest.raises(TypeError):
        GFlowConfig.from_dict({key: "legacy"})


def test_gflow_config_does_not_own_sft_or_evaluation_settings():
    removed = (
        "reward_transform",
        "compute_train_metrics",
        "compute_eval_metrics",
        "compute_token_metrics",
        "evaluate_generation",
        "generation_topk",
        "generation_num_beams",
        "generation_num_return_sequences",
        "generation_constrain_valid_sequences",
        "generation_exclude_seen_items",
    )

    cfg = GFlowConfig()

    assert not any(hasattr(cfg, name) for name in removed)


def test_external_curriculum_sampler_still_receives_training_step():
    class Sampler:
        def __init__(self):
            self.step = None

        def sample(self, batch, *, model, step):
            self.step = step
            return "curriculum-trajectories"

    sampler = Sampler()
    method = GFlowMethod(
        GFlowConfig(reward_cache_path="unused"),
        candidate_sampler=sampler,
        sft_method=SFTMethod(SFTConfig()),
    )

    result = method._sample_trajectories(object(), {}, step=17)

    assert result == "curriculum-trajectories"
    assert sampler.step == 17


def test_gflow_method_requires_an_explicit_candidate_sampler():
    with pytest.raises(TypeError, match="candidate_sampler"):
        GFlowMethod(GFlowConfig(reward_cache_path="unused"))


def test_gflow_method_requires_an_explicit_sft_method():
    class Sampler:
        pass

    with pytest.raises(TypeError, match="sft_method"):
        GFlowMethod(
            GFlowConfig(reward_cache_path="unused"),
            candidate_sampler=Sampler(),
        )


def test_gflow_config_owns_candidate_sampling_settings():
    cfg = GFlowConfig.from_dict(
        {
            "candidate_sampler": "sasrec",
            "num_negative_samples": 3,
            "num_random_negatives": 1,
            "easy_negative_steps": 100,
            "candidate_pool_size": 1024,
            "candidate_sampling_temperature": 0.5,
        }
    )

    assert cfg.candidate_sampler == "sasrec"
    assert cfg.num_negative_samples == 3
    assert cfg.num_random_negatives == 1
    assert cfg.easy_negative_steps == 100
    assert cfg.candidate_pool_size == 1024
    assert cfg.candidate_sampling_temperature == 0.5


@pytest.mark.parametrize(
    "key",
    [
        "include_positive",
        "avoid_positive",
        "semantic_ids_are_fixed_length",
        "max_resample_attempts",
        "exclude_history_in_sampling",
        "sasrec_oversample_factor",
        "keep_scorer_eval",
        "pad_token_id",
        "reward_key",
        "reward_epsilon",
        "negative_reward",
    ],
)
def test_fixed_gflow_sampling_invariants_are_not_configurable(key):
    with pytest.raises(TypeError):
        GFlowConfig.from_dict({key: True})


def test_sampler_config_only_exposes_real_choices():
    cfg = CandidateSamplerConfig(mode="random", num_negatives=3, include_positive=False)

    # Invariants of the unified sampler that must stay non-configurable: the
    # positive target is always excluded from negatives, sampling never
    # retries, and sampler naming is fixed.
    fixed_fields = {
        "avoid_positive",
        "max_resample_attempts",
        "sampler_name",
        "keep_scorer_eval",
        "candidate_pool_size",
        "pool_strategy",
    }
    assert not any(hasattr(cfg, name) for name in fixed_fields)
    # Real choices the unified sampler does expose.
    assert cfg.exclude_history is not None
    assert cfg.mode in ("random", "top_n", "curriculum")
