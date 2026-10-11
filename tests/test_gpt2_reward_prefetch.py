"""The GPT-2 worker pipeline must preserve data, rewards and GFlow gradients."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon as g  # noqa: E402

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.data.collators import SequenceBatchCollator
from flowrec.data.datasets import SequenceDataset
from flowrec.ft_methods.gflow.factorized_cache import (
    CACHE_KIND,
    FACTORS_FILENAME,
    ComposedFactorizedRewardCache,
    FactorizedRewardCache,
)
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.rewards.sasrec import SASRecRewardConfig
from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel


REWARD_CASES = [
    pytest.param(("sasrec",), 0.25, id="sasrec_pow0p25"),
    pytest.param(("interaction",), 1.0, id="interaction_pow1"),
    pytest.param(("similarity",), 4.0, id="similarity_pow4"),
    pytest.param(("sasrec", "interaction"), 16.0, id="sasrec_interaction_pow16"),
    pytest.param(("sasrec", "similarity"), 1.0, id="sasrec_similarity_pow1"),
    pytest.param(("interaction", "similarity"), 0.25, id="interaction_similarity_pow0p25"),
    pytest.param(("sasrec", "interaction", "similarity"), 4.0, id="all_pow4"),
]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture()
def problem(tmp_path):
    torch.manual_seed(42)
    catalog = torch.tensor([[1 + i // 8, 4 + i // 4 % 2, 8 + i % 4] for i in range(16)])
    mapping = {f"item_{i:02d}": row.tolist() for i, row in enumerate(catalog)}
    id2item = ["<PAD>", *sorted(mapping)]
    targets = torch.tensor([1, 4, 11, 15, 6])
    records = []
    for index, target in enumerate(targets.tolist()):
        # Vary history length to exercise left padding as well as truncation.
        history = list(range(1, index + 2))
        records.append(dict(
            raw_user_id=f"user-{index}", user_id=index,
            history_item_ids=history, history_raw_items=[id2item[item] for item in history],
            history_ratings=[float(item) for item in history],
            history_verified_purchase=[bool(item % 2) for item in history],
            target_item_id=target + 1, target_raw_item=id2item[target + 1],
            target_rating=float(index), target_verified_purchase=None,
            target_timestamp=100 + index, target_position=len(history), split="train",
        ))
    dataset = SequenceDataset(pd.DataFrame(records), "train")
    collator = SequenceBatchCollator(
        max_history_len=3, history_padding="left", semantic_ids=mapping,
        sid_len=3, include_raw=False,
    )
    dm = SimpleNamespace(
        train_dataset=dataset, id2item=id2item, _make_collator=lambda: collator,
        cfg=SimpleNamespace(tensorized_batches=True, num_workers=0,
                            pin_memory=False, persistent_workers=False),
    )
    # Reward factors follow SUBSET order, including repeated source rows.
    subset = [3, 0, 3, 1, 2]
    rows = torch.tensor(subset)
    sasrec_cfg = SASRecRewardConfig(transform="sigmoid", temperature=1.0)
    sim_cfg = SimilarityRewardConfig(mode="hierarchical", token_offset=0, level_stride=4)
    similarity = SimilarityRewardModel(sim_cfg, codebooks=torch.randn(3, 4, 3))
    evidence = [([1, 7], [1.3, 0.6]), ([], []), ([0, 4, 11], [0.8, 1.2, 1.1]),
                ([15], [0.4]), ([3, 6], [0.9, 1.25])]
    selected = [evidence[row] for row in subset]
    sizes = torch.tensor([len(columns) for columns, _ in selected])
    factors = {
        "sasrec": {
            "hidden": torch.randn(5, 4).half()[rows], "item_table": torch.randn(16, 4),
            "row_valid": torch.ones(16, dtype=torch.bool),
        },
        "interaction": {
            "indptr": torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0))),
            "cols": torch.tensor([column for columns, _ in selected for column in columns]),
            "vals": torch.tensor([value for _, values in selected for value in values]),
        },
        "similarity": {
            "target_token_ids": catalog[targets[rows]], "item_tokens": catalog,
            "code_dist": similarity.code_dist,
        },
    }
    paths = {}
    for name, values in factors.items():
        path = tmp_path / name
        path.mkdir()
        torch.save({**values, "target_semantic_row_ids": targets[rows]}, path / FACTORS_FILENAME)
        config = asdict(sasrec_cfg) if name == "sasrec" else asdict(sim_cfg) if name == "similarity" else {}
        (path / "manifest.json").write_text(json.dumps({
            "kind": CACHE_KIND, "factors_path": FACTORS_FILENAME,
            "num_examples": len(subset), "num_catalog_items": len(catalog),
            "reward_epsilon": 1e-6, "fill_value": 1.0,
            "component": {"name": name, "config": config,
                          "mode": "hierarchical" if name == "similarity" else None},
        }))
        paths[name] = path
    model = GPT2RecModel(
        codebook_size=4, sem_id_len=3, semantic_id_mapping=mapping,
        embedding_dim=8, num_heads=2, num_layers=1, dim_feedforward=16,
        max_position_embeddings=16, dropout=0.1,
    ).train()
    teacher = SASRec(SASRecConfig(
        item_vocab_size=17, max_history_len=3, d_model=8, nhead=2, num_layers=1,
        dropout=0.0, pos_drop=0.0,
    )).eval()
    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(mode="curriculum", num_negatives=3, curriculum_steps=10),
        item_semantic_ids=catalog, item_id_to_semantic_row=torch.tensor([-1, *range(16)]),
        scorer=teacher,
    )
    return SimpleNamespace(dm=dm, subset=subset, records=records, collator=collator,
                           paths=paths, model=model, sampler=sampler)


def _cache(problem, components, power):
    return ComposedFactorizedRewardCache(
        [(name, FactorizedRewardCache(problem.paths[name]), 1.0) for name in components],
        composition="multiply", power=power,
    )


def _loader(problem, cache=None, *, workers=0):
    return g.gflow_ablation.indexed_train_loader(
        problem.dm, problem.subset, shuffle=True, batch_size=3, seed=44,
        num_workers=workers, reward_cache=cache, tensorized=True, prefetch_factor=2,
    )


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
@pytest.mark.parametrize("components,power", REWARD_CASES)
def test_gpt2_prefetched_reward_factors_preserve_sampling_losses_and_gradients(
    problem, objective, components, power, monkeypatch,
):
    cache = _cache(problem, components, power)
    synchronous = next(iter(_loader(problem)))
    prepared = next(iter(_loader(problem, cache)))
    for key, expected in synchronous.items():
        torch.testing.assert_close(prepared[key], expected, rtol=0, atol=0, equal_nan=True)
    method = g.make_method(
        g.GFlowConfig(objective=objective, reward_cache_path="injected", gflow_weight=0.04),
        problem.sampler,
        g.SFTConfig(compute_train_metrics=False, compute_eval_metrics=False), cache,
    )

    def evaluate(batch):
        # Reset BOTH stochastic curriculum sampling and GPT-2 training dropout.
        torch.manual_seed(739)
        result = method.compute_loss(problem.model, batch, stage="train", step=3)
        gradients = torch.autograd.grad(result.loss, tuple(problem.model.parameters()), allow_unused=True)
        return result, gradients, torch.random.get_rng_state()

    expected, expected_grads, expected_rng = evaluate(synchronous)
    received = []
    select_candidates = cache.select_candidates

    def check_prepared(*args, **kwargs):
        received.append(kwargs.get("prepared"))
        return select_candidates(*args, **kwargs)

    monkeypatch.setattr(cache, "select_candidates", check_prepared)
    actual, actual_grads, actual_rng = evaluate(prepared)
    assert len(received) == 1 and received[0] is prepared["_gflow_prepared_rewards"]
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
    for key in ("rewards", "sequence_logprobs", "token_logprobs"):
        if key in expected.extra:
            torch.testing.assert_close(actual.extra[key], expected.extra[key], rtol=1e-6, atol=1e-7)
    for key in ("candidate_token_ids", "candidate_token_mask", "candidate_item_row_ids", "is_positive"):
        torch.testing.assert_close(getattr(actual.extra["trajectories"], key),
                                   getattr(expected.extra["trajectories"], key), rtol=0, atol=0)
    for key in expected.metrics:
        torch.testing.assert_close(actual.metrics[key], expected.metrics[key], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(actual.loss)
    assert any(grad is not None and bool(grad.abs().sum() > 0) for grad in actual_grads)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert (actual_grad is None) == (expected_grad is None)
        if expected_grad is not None:
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("workers", [0, 2])
def test_gpt2_indexed_prefetch_preserves_shuffled_subset_and_full_reward_normalizer(problem, workers):
    cache = _cache(problem, ("sasrec", "interaction", "similarity"), 16.0)
    seen = []
    for actual in _loader(problem, cache, workers=workers):
        positions = actual["example_indices"].tolist()
        seen.extend(positions)
        expected = problem.collator([
            dict(problem.records[problem.subset[position]], example_index=position)
            for position in positions
        ])
        assert actual.keys() - expected.keys() == {"_gflow_prepared_rewards", "_gflow_reward_prepare_seconds"}
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)
        candidates = torch.tensor([[-1, 0, 4, 11]]).expand(len(positions), -1)
        kwargs = dict(include_full=True)
        synchronous = cache.select_candidates(actual["example_indices"], candidates, torch.device("cpu"), **kwargs)
        prefetched = cache.select_candidates(
            actual["example_indices"], candidates, torch.device("cpu"),
            prepared=actual["_gflow_prepared_rewards"], **kwargs,
        )
        for value, reference in zip(prefetched, synchronous):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert actual["_gflow_reward_prepare_seconds"] >= 0
    assert sorted(seen) == list(range(len(problem.subset)))
    assert seen != list(range(len(problem.subset)))
    assert problem.dm.train_dataset._records is None
