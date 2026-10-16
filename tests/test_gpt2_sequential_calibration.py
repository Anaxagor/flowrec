"""Calibration keeps the old dropout/sampling protocol without two live graphs."""

import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.calibration import (
    calibrate_from_loss_pairs, calibrate_sequential_variants, variant_loss_pairs,
)
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "exp4-reward-ablations"
sys.path.insert(0, str(SCRIPT_DIR))
import gflow_finetune_gpt2_amazon as experiment  # noqa: E402


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
def test_sequential_calibration_matches_old_gpt2_dropout_and_sampling(objective):
    catalog = torch.tensor([[1, 4, 7], [1, 4, 8], [1, 5, 9], [2, 6, 10]])
    torch.manual_seed(42)
    model = GPT2RecModel(
        codebook_size=4, sem_id_len=3,
        semantic_id_mapping={str(i): row.tolist() for i, row in enumerate(catalog)},
        embedding_dim=8, num_heads=2, num_layers=1, dim_feedforward=16,
        max_position_embeddings=16, dropout=.2,
    ).train()
    history = torch.tensor([[[0, 0, 0], [1, 4, 7]], [[1, 5, 9], [2, 6, 10]]])
    batch = {
        "user_ids": torch.arange(2), "example_indices": torch.arange(2),
        "history_token_ids": history, "history_token_mask": history.ne(0),
        "target_token_ids": catalog[[0, 3]],
        "target_token_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    sample_calls = []

    def sample(*args, **kwargs):
        rows = torch.stack([torch.randperm(4), torch.randperm(4)])
        sample_calls.append(rows.clone())
        tokens = catalog[rows]
        return SemanticTrajectoryBatch(tokens, torch.ones_like(tokens).bool(), rows.eq(0), rows)

    rewards = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    cache = SimpleNamespace(select_candidates=lambda indices, rows, device, include_full=False: (
        rewards.gather(1, rows), rewards.sum(-1), rewards if include_full else None,
    ))
    sft = SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False))

    def variant(reference):
        method = experiment.GPT2GFlowMethod(
            GFlowConfig(objective=objective, sft_weight=float(reference),
                        gflow_weight=float(not reference), reward_cache_path="injected"),
            candidate_sampler=SimpleNamespace(sample=sample), sft_method=sft,
        )
        method._reward_cache = cache
        return method

    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.ones_like(p)
    legacy = calibrate_from_loss_pairs(
        variant_loss_pairs(model, variant(True), variant(False), [batch] * 2, seed=7),
        params, target_ratio=.1,
    )
    old_component_samples = sample_calls[1::2]
    sample_calls.clear()
    actual = calibrate_sequential_variants(
        model, sft, variant(False), [batch] * 2, params, seed=7, target_ratio=.1,
    )
    for name, values in actual["per_batch"].items():
        assert values == pytest.approx(legacy["per_batch"][name], rel=1e-6, abs=1e-7)
    assert actual["fixed_gflow_weight"] == pytest.approx(legacy["fixed_gflow_weight"], rel=1e-6)
    assert len(sample_calls) == 2  # Reference never samples unused candidates.
    for old, new in zip(old_component_samples, sample_calls):
        torch.testing.assert_close(old, new)
    for original, p in zip(before, params):
        torch.testing.assert_close(p, original)
        torch.testing.assert_close(p.grad, torch.ones_like(p))


def test_reference_activations_are_freed_before_component_forward():
    refs = []
    model = torch.nn.Linear(4, 3)

    def forward(model, batch, **kwargs):
        # Squaring saves this intermediate for backward. No next variant or
        # batch may start while a previous variant still owns it.
        assert all(ref() is None for ref in refs)
        activation = model(batch["x"]).sin()
        refs.append(weakref.ref(activation))
        return MethodOutput(loss=activation.square().mean())

    method = SimpleNamespace(compute_loss=forward)
    result = calibrate_sequential_variants(
        model, method, method, [{"x": torch.randn(2, 4)}] * 3,
        list(model.parameters()), target_ratio=1., seed=12,
    )
    assert result["num_batches"] == 3 and result["weight"] == pytest.approx(1.)
    assert len(refs) == 6 and all(ref() is None for ref in refs)


def test_sequential_calibration_reports_missing_batches():
    model = torch.nn.Linear(4, 3)
    method = SimpleNamespace(compute_loss=lambda model, batch, **kw:
                             MethodOutput(loss=model(batch["x"]).square().mean()))
    with pytest.raises(ValueError, match="yielded only 1"):
        calibrate_sequential_variants(
            model, method, method, [{"x": torch.randn(2, 4)}], list(model.parameters()),
            target_ratio=1., num_batches=2,
        )


def test_experiment_restores_model_mode_and_training_seed_on_calibration_error(monkeypatch):
    from flowrec.ft_methods import calibration

    model = torch.nn.Linear(4, 3).eval()
    args = SimpleNamespace(smoke=True, reward_prefetch=False, gflow_calibration_seed=1234,
                           loader_prefetch_factor=2, tensorized_train_loader=True,
                           gflow_calibration_eps=1e-12, seed=42)
    monkeypatch.setattr(experiment.gflow_ablation, "indexed_train_loader", lambda *a, **kw: [])

    def fail(*args, **kwargs):
        assert model.training
        assert isinstance(args[1], experiment.gflow_ablation.AliasedSFTMethod)
        raise torch.cuda.OutOfMemoryError("calibration OOM")

    monkeypatch.setattr(calibration, "calibrate_sequential_variants", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        experiment.calibrate_gflow_weight(
            model=model, base_gflow_cfg=GFlowConfig(reward_cache_path="injected"),
            sampler=object(), sft_cfg=SFTConfig(), cache=object(), dm=object(),
            subset_indices=[], device=torch.device("cpu"), args=args, tag="trial36",
            target_ratio=1., batch_size=128,
        )
    assert not model.training
    assert torch.initial_seed() == 42
