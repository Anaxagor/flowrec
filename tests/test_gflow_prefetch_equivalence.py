"""Reward-factor prefetch must preserve sampled TIGER losses and gradients."""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.data.collators import SequenceBatchCollator
from flowrec.data.datasets import SequenceDataset
from flowrec.data.indexed_batches import create_indexed_parts
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.factorized_cache import (
    CACHE_KIND,
    FACTORS_FILENAME,
    ComposedFactorizedRewardCache,
    FactorizedRewardCache,
)
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.ft_methods.gflow.prefetch import RewardPrefetchCollator
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.models.tiger import TigerModel
from flowrec.rewards.sasrec import SASRecRewardConfig
from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel
from flowrec.training.trainer import move_to_device


SELECTED_REWARDS = [
    pytest.param(("sasrec",), 4.0, id="sasrec_pow4"),
    pytest.param(("sasrec", "interaction"), 16.0, id="coverage_pow16"),
    pytest.param(("sasrec", "similarity"), 0.25, id="pareto_pow0p25"),
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
def tiny_problem(tmp_path):
    torch.manual_seed(42)
    # Shared prefixes, unique terminal sequences, and enough allowed items
    # for the real K=10 curriculum sampler after history/target exclusions.
    catalog = torch.tensor([[1 + i // 8, 4 + i // 4 % 2, 8 + i % 4] for i in range(16)])
    targets = torch.tensor([1, 4, 11, 15, 6])
    histories = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
    all_batches = {
        "user_ids": torch.arange(5),
        "history_item_ids": histories,
        "history_mask": torch.ones(5, 3, dtype=torch.bool),
        "history_token_ids": catalog[histories - 1],
        "history_token_mask": torch.ones(5, 3, 3, dtype=torch.bool),
        "target_item_ids": targets + 1,
        "target_token_ids": catalog[targets],
        "target_token_mask": torch.ones(5, 3, dtype=torch.bool),
    }
    # Repeated, out-of-order cache rows expose alignment errors that sorted
    # unique rows would hide.
    indices = torch.tensor([4, 1, 4, 2])
    batch = {key: value[indices] for key, value in all_batches.items()}
    batch["example_indices"] = indices

    sasrec_cfg = SASRecRewardConfig(transform="sigmoid", temperature=1.0)
    sim_cfg = SimilarityRewardConfig(mode="hierarchical", token_offset=0, level_stride=4)
    similarity = SimilarityRewardModel(sim_cfg, codebooks=torch.randn(3, 4, 3))
    factors = {
        "sasrec": {
            "hidden": torch.randn(5, 4).half(),
            "item_table": torch.randn(16, 4),
            "row_valid": torch.ones(16, dtype=torch.bool),
        },
        # Empty rows, unequal row lengths and non-neutral evidence values.
        "interaction": {
            "indptr": torch.tensor([0, 2, 2, 5, 6, 8]),
            "cols": torch.tensor([1, 7, 0, 4, 11, 15, 3, 6]),
            "vals": torch.tensor([1.3, 0.6, 0.8, 1.2, 1.1, 0.4, 0.9, 1.25]),
        },
        "similarity": {
            "target_token_ids": catalog[targets],
            "item_tokens": catalog,
            "code_dist": similarity.code_dist,
        },
    }
    paths = {}
    for name, values in factors.items():
        path = tmp_path / name
        path.mkdir()
        torch.save({**values, "target_semantic_row_ids": targets}, path / FACTORS_FILENAME)
        config = asdict(sasrec_cfg) if name == "sasrec" else asdict(sim_cfg) if name == "similarity" else {}
        (path / "manifest.json").write_text(json.dumps({
            "kind": CACHE_KIND,
            "factors_path": FACTORS_FILENAME,
            "num_examples": 5,
            "num_catalog_items": 16,
            "reward_epsilon": 1e-6,
            "fill_value": 1.0,
            "component": {"name": name, "mode": "hierarchical" if name == "similarity" else None,
                          "config": config},
        }))
        paths[name] = path

    model = TigerModel(
        codebook_size=4, sem_id_len=3, user_ids_count=0, use_user_token=False,
        semantic_id_mapping={f"item_{i:02d}": row.tolist() for i, row in enumerate(catalog)},
        embedding_dim=8, num_heads=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=16, d_kv=4, dropout=0.1,
    ).train()
    teacher = SASRec(SASRecConfig(
        item_vocab_size=17, max_history_len=3, d_model=8, nhead=2,
        num_layers=1, dropout=0.0, pos_drop=0.0,
    )).eval()
    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(mode="curriculum", num_negatives=10, curriculum_steps=10),
        item_semantic_ids=catalog,
        item_id_to_semantic_row=torch.tensor([-1, *range(16)]),
        scorer=teacher,
    )
    return batch, model, sampler, paths


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
@pytest.mark.parametrize("components,power", SELECTED_REWARDS)
def test_prepared_factors_preserve_tiger_loss_sampling_and_all_gradients(
    tiny_problem, objective, components, power, monkeypatch,
):
    batch, model, sampler, paths = tiny_problem
    cache = ComposedFactorizedRewardCache(
        [(name, FactorizedRewardCache(paths[name]), 1.0) for name in components],
        composition="multiply", power=power,
    )
    method = GFlowMethod(
        GFlowConfig(objective=objective, reward_cache_path="injected", gflow_weight=0.04),
        candidate_sampler=sampler,
        sft_method=SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False)),
    )
    method._reward_cache = cache
    params = tuple(model.parameters())

    def evaluate(input_batch):
        # Reset immediately before the forward: preserve BOTH stochastic
        # curriculum sampling and real training-mode TIGER dropout.
        torch.manual_seed(739)
        result = method.compute_loss(model, input_batch, stage="train", step=3)
        gradients = torch.autograd.grad(result.loss, params, allow_unused=True)
        return result, gradients, torch.random.get_rng_state()

    expected, expected_gradients, expected_rng = evaluate(batch)
    prepared_batch = {**batch, "_gflow_prepared_rewards": cache.prepare_rows(batch["example_indices"])}
    received = []
    select_candidates = cache.select_candidates

    def checked_select(*args, **kwargs):
        received.append(kwargs.get("prepared"))
        return select_candidates(*args, **kwargs)

    monkeypatch.setattr(cache, "select_candidates", checked_select)
    actual, actual_gradients, actual_rng = evaluate(prepared_batch)
    assert len(received) == 1 and received[0] is prepared_batch["_gflow_prepared_rewards"]
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
    for key in ("rewards", "sequence_logprobs", "token_logprobs"):
        if key in expected.extra:
            torch.testing.assert_close(actual.extra[key], expected.extra[key], rtol=1e-6, atol=1e-7)
    for key in ("candidate_token_ids", "candidate_token_mask", "candidate_item_row_ids", "is_positive"):
        torch.testing.assert_close(
            getattr(actual.extra["trajectories"], key), getattr(expected.extra["trajectories"], key),
            rtol=0, atol=0,
        )
    for key in expected.metrics:
        torch.testing.assert_close(actual.metrics[key], expected.metrics[key], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(actual.loss)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in actual_gradients)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        assert (actual_grad is None) == (expected_grad is None)
        if expected_grad is not None:
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


def _reorder_cache_rows(paths, order):
    """Write the reward-cache order used by a nonmonotonic subset."""
    order = torch.as_tensor(order)
    for name, path in paths.items():
        factors = torch.load(path / FACTORS_FILENAME, weights_only=True)
        factors["target_semantic_row_ids"] = factors["target_semantic_row_ids"][order]
        if name == "sasrec":
            factors["hidden"] = factors["hidden"][order]
        elif name == "similarity":
            factors["target_token_ids"] = factors["target_token_ids"][order]
        else:
            bounds = factors["indptr"]
            pieces = [slice(int(bounds[row]), int(bounds[row + 1])) for row in order]
            sizes = torch.tensor([piece.stop - piece.start for piece in pieces])
            factors["indptr"] = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
            for key in ("cols", "vals"):
                factors[key] = torch.cat([factors[key][piece] for piece in pieces])
        torch.save(factors, path / FACTORS_FILENAME)


@pytest.mark.parametrize("context", [None, "spawn", "fork"])
def test_prefetch_workers_preserve_subset_positions_and_reward_rows(tiny_problem, context):
    if context is not None and context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} is unavailable")
    _, model, _, paths = tiny_problem
    mapping = {name: model.item_semantic_ids[row].tolist()
               for name, row in model._semantic_item_to_row.items()}
    id2item = ["<PAD>", *sorted(mapping)]
    targets = torch.load(paths["sasrec"] / FACTORS_FILENAME, weights_only=True)["target_semantic_row_ids"]
    records = []
    for index, target in enumerate(targets.tolist()):
        history = list(range(index + 1, index + 4))
        records.append(dict(
            raw_user_id=f"user-{index}", user_id=index,
            history_item_ids=history, history_raw_items=[id2item[item] for item in history],
            history_ratings=[1., 2., 3.], history_verified_purchase=[True, False, None],
            target_item_id=target + 1, target_raw_item=id2item[target + 1],
            target_rating=float(index), target_verified_purchase=None,
            target_timestamp=100 + index, target_position=3, split="train",
        ))
    frame = pd.DataFrame(records)
    source = SequenceDataset(frame, "train")
    base_collator = SequenceBatchCollator(
        max_history_len=3, semantic_ids=mapping, sid_len=3, include_raw=False,
    )
    dm = SimpleNamespace(train_dataset=source, id2item=id2item,
                         cfg=SimpleNamespace(tensorized_batches=True))
    subset = [3, 0, 3, 1, 2]
    _reorder_cache_rows(paths, subset)
    dataset, collator = create_indexed_parts(dm, subset, collator=base_collator)
    cache = ComposedFactorizedRewardCache(
        [(name, FactorizedRewardCache(paths[name]), 1.0) for name in ("sasrec", "interaction")],
        composition="multiply", power=16.0,
    )
    kwargs = {} if context is None else dict(
        multiprocessing_context=context, prefetch_factor=2, timeout=30,
    )
    loader = DataLoader(
        dataset, collate_fn=RewardPrefetchCollator(collator, cache.make_row_reader()),
        batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(44),
        num_workers=0 if context is None else 2, **kwargs,
    )
    seen = []
    for actual in loader:
        positions = actual["example_indices"].tolist()
        seen.extend(positions)
        expected = base_collator([
            dict(records[subset[position]], example_index=position) for position in positions
        ])
        assert actual.keys() - expected.keys() == {"_gflow_prepared_rewards", "_gflow_reward_prepare_seconds"}
        for key, value in expected.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)
            else:
                assert actual[key] == value
        candidates = torch.tensor([[-1, 0, 4, 11]]).expand(len(positions), -1)
        reference = cache.select_candidates(actual["example_indices"], candidates, torch.device("cpu"), include_full=True)
        prefetched = cache.select_candidates(
            actual["example_indices"], candidates, torch.device("cpu"), include_full=True,
            prepared=actual["_gflow_prepared_rewards"],
        )
        for actual_value, expected_value in zip(prefetched, reference):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        assert actual["_gflow_reward_prepare_seconds"] >= 0
    assert sorted(seen) == list(range(len(subset)))
    assert seen != list(range(len(subset)))
    assert source._records is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pinned_prefetch_preserves_amp_subtb_gradients(tiny_problem):
    from torch.utils.data._utils.pin_memory import pin_memory

    batch, model, sampler, paths = tiny_problem
    cache = ComposedFactorizedRewardCache(
        [(name, FactorizedRewardCache(paths[name]), 1.0) for name in ("sasrec", "interaction")],
        composition="multiply", power=16.0,
    )
    # Exercise the actual pin-memory and recursive Trainer transfer paths.
    cpu_prepared = {**batch, "_gflow_prepared_rewards": cache.make_row_reader().prepare_rows(batch["example_indices"])}
    device = torch.device("cuda")
    prepared_batch = move_to_device(pin_memory(cpu_prepared), device)
    baseline_batch = move_to_device(batch, device)
    model.to(device)
    sampler.scorer.to(device)
    method = GFlowMethod(
        GFlowConfig(objective="subtb", reward_cache_path="injected", gflow_weight=0.04),
        candidate_sampler=sampler,
        sft_method=SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False)),
    )
    method._reward_cache = cache

    def evaluate(input_batch):
        torch.manual_seed(739)
        with torch.autocast("cuda", dtype=torch.float16):
            output = method.compute_loss(model, input_batch, stage="train", step=3)
        gradients = torch.autograd.grad(output.loss, tuple(model.parameters()), allow_unused=True)
        return output, gradients

    expected, expected_gradients = evaluate(baseline_batch)
    actual, actual_gradients = evaluate(prepared_batch)
    torch.testing.assert_close(actual.extra["rewards"], expected.extra["rewards"])
    torch.testing.assert_close(actual.extra["trajectories"].candidate_item_row_ids,
                               expected.extra["trajectories"].candidate_item_row_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual.loss)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        assert (actual_grad is None) == (expected_grad is None)
        if actual_grad is not None:
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
