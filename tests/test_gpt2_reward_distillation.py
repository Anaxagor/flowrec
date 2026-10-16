"""Differentiable GPT2 prefix caching and CE distillation equivalence."""

import pytest
import torch

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY
from flowrec.ft_methods.calibration import calibrate_component_weight
from flowrec.ft_methods.reward_distillation import RewardDistillationMethod
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.gpt2 import GPT2RecModel


CATALOG = torch.tensor([[1, 4, 7], [1, 4, 8], [1, 5, 9], [2, 6, 10]])
REWARDS = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])


class Cache:
    def __init__(self, rewards=REWARDS):
        self.rewards = rewards

    def select_candidates(self, indices, rows, device, *, include_full, **kwargs):
        rewards = self.rewards[indices].to(device)
        return rewards.gather(1, rows), rewards.sum(1), rewards


def tiny_model(catalog=CATALOG, dropout=0.):
    torch.manual_seed(12)
    return GPT2RecModel(
        codebook_size=4, sem_id_len=3,
        semantic_id_mapping={str(i): row.tolist() for i, row in enumerate(catalog)},
        embedding_dim=8, num_heads=2, num_layers=2, dim_feedforward=16,
        max_position_embeddings=20, dropout=dropout,
    )


def batch():
    history = torch.tensor([[[1, 4, 7], [2, 6, 10]], [[0, 0, 0], [1, 5, 9]]])
    return {
        "user_ids": torch.arange(2), "example_indices": torch.arange(2),
        "history_token_ids": history, "history_token_mask": history.ne(0),
        "target_token_ids": CATALOG[[1, 3]],
        "target_token_mask": torch.ones(2, 3, dtype=torch.bool),
    }


def sft():
    return SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False))


def method(**kwargs):
    return RewardDistillationMethod(Cache(), CATALOG, sft(), **kwargs)


def parameter_grads(loss, model):
    values = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(model.parameters(), values)]


def test_paired_sampling_arms_calibrate_identical_lambda_and_restore_is_sampler():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ce_calibration_test", Path(__file__).resolve().parents[1]
        / "scripts/exp7-ce-distillation/ce_distill_tiger.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model = tiny_model(dropout=.1)
    direct = method(num_samples=3)
    corrected = method(sampling="curriculum_smooth_is", num_samples=3, curriculum_steps=10)
    args = dict(component="distill", target_ratio=.1, num_batches=2, seed=1234, device="cpu")
    direct_cal = runner.calibrate_distillation_weight(
        calibrate_component_weight, model, direct, [batch(), batch()], **args)
    corrected_cal = runner.calibrate_distillation_weight(
        calibrate_component_weight, model, corrected, [batch(), batch()], **args)
    assert corrected_cal == {**direct_cal, "sampling": "reward", "ce_target": "q"}
    assert corrected.cfg.sampling == "curriculum_smooth_is"


@pytest.mark.parametrize("chunk_size", [1, 3, 20])
def test_gpt2_exact_ce_and_all_parameter_gradients_match_full_enumeration(chunk_size):
    model = tiny_model()
    trainer = method(estimator="exact", catalog_chunk_size=chunk_size, distill_weight=.7)
    inputs = batch()
    widths = []
    hook = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: widths.append(kwargs["input_ids"].size(1)), with_kwargs=True,
    )
    result = trainer.compute_loss(model, inputs)
    actual_grads = parameter_grads(result.loss, model)
    hook.remove()
    # One SFT history+target forward, one differentiable history prefill.
    # Forward and checkpoint recomputation only decode <= 2 candidate tokens.
    assert widths.count(9) == 1 and widths.count(6) == 1
    assert all(width <= 2 for width in widths if width not in (6, 9))

    sft_loss = sft().compute_loss(model, inputs).loss
    all_tokens = CATALOG.unsqueeze(0).expand(2, -1, -1)
    token_lp = model.score_item_tokens_valid_actions(
        inputs, candidate_token_ids=all_tokens,
        candidate_token_mask=torch.ones_like(all_tokens, dtype=torch.bool),
    )
    q = REWARDS / REWARDS.sum(1, keepdim=True)
    expected_ce = -(q * token_lp.sum(-1)).sum(-1).mean()
    torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], expected_ce)
    expected_grads = parameter_grads(sft_loss + .7 * expected_ce, model)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-4)


@pytest.mark.parametrize("sampling,step", [
    ("reward", 0), ("curriculum", 0), ("curriculum", 10),
    ("curriculum_smooth", 0), ("curriculum_smooth", 5), ("curriculum_smooth", 10),
    ("curriculum_smooth_is", 0), ("curriculum_smooth_is", 5), ("curriculum_smooth_is", 10),
])
def test_gpt2_sampled_ce_and_all_gradients_match_uncached_selected_items(sampling, step):
    torch.manual_seed(99)
    model = tiny_model()
    trainer = method(sampling=sampling, curriculum_steps=10, num_samples=5)
    result = trainer.compute_loss(model, batch(), step=step)
    actual_grads = parameter_grads(result.extra[LOSS_COMPONENTS_KEY]["distill"], model)
    rows = result.extra["distill_sampled_item_rows"]
    tokens = CATALOG[rows]
    selected_lp = model.score_item_tokens_valid_actions(
        batch(), candidate_token_ids=tokens, candidate_token_mask=torch.ones_like(tokens, dtype=torch.bool),
    ).sum(-1)
    if sampling == "curriculum_smooth_is":
        q = REWARDS / REWARDS.sum(1, keepdim=True)
        high_mask = torch.tensor([[False, False, True, True], [True, True, False, False]])
        high, low = q * high_mask, q * ~high_mask
        alpha = step / 10
        proposal = .9 * ((1-alpha) * low / low.sum(1, keepdim=True)
                         + alpha * high / high.sum(1, keepdim=True)) + .1 * q
        reference = -(selected_lp * (q / proposal).gather(1, rows)).mean()
    else:
        reference = -selected_lp.mean()
    torch.testing.assert_close(result.extra[LOSS_COMPONENTS_KEY]["distill"], reference)
    expected_grads = parameter_grads(reference, model)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-4)


def test_differentiable_history_cache_preserves_gradients_masks_and_shared_state():
    model = tiny_model()
    inputs = batch()
    cached = model.cache_history_encoding_grad(inputs)
    key_values, prefix_mask, first_logits = cached["_gpt2_history_grad_state"]
    assert all(keys.requires_grad and values.requires_grad for keys, values in key_values)
    assert first_logits.requires_grad
    lengths = [keys.size(-2) for keys, _ in key_values]
    snapshots = [(keys.clone(), values.clone()) for keys, values in key_values]
    tokens = CATALOG[[0, 1, 3]].unsqueeze(0).expand(2, -1, -1).clone()
    mask = torch.ones_like(tokens, dtype=torch.bool)
    tokens[:, 1, 2] = -100
    mask[:, 1, 2] = False
    actual = model.score_item_tokens_valid_actions(cached, candidate_token_ids=tokens, candidate_token_mask=mask)
    expected = model.score_item_tokens_valid_actions(inputs, candidate_token_ids=tokens, candidate_token_mask=mask)
    torch.testing.assert_close(actual, expected)
    cache_grads = torch.autograd.grad(
        actual.sum(), tuple(tensor for layer in key_values for tensor in layer), retain_graph=True,
    )
    assert all(gradient.abs().sum() > 0 for gradient in cache_grads)
    actual_grads = parameter_grads(actual.sum(), model)
    expected_grads = parameter_grads(expected.sum(), model)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-4)
    for (keys, values), (old_keys, old_values), length in zip(key_values, snapshots, lengths):
        assert keys.size(-2) == values.size(-2) == length
        torch.testing.assert_close(keys, old_keys)
        torch.testing.assert_close(values, old_values)
    assert prefix_mask.tolist() == [[1] * 6, [0, 0, 0, 1, 1, 1]]


def test_existing_inference_history_cache_still_matches_full_scoring():
    model = tiny_model().eval()
    with torch.no_grad():
        cached = model.cache_history_encoding(batch())
        tokens = CATALOG.unsqueeze(0).expand(2, -1, -1)
        actual = model.candidate_token_logits(cached, candidate_token_ids=tokens)
        expected = model.candidate_token_logits(batch(), candidate_token_ids=tokens)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("estimator", ["sampled", "exact"])
def test_gpt2_distillation_dropout_and_checkpoint_backward(estimator):
    model = tiny_model(dropout=.1)
    trainer = method(estimator=estimator, catalog_chunk_size=1)
    result = trainer.compute_loss(model, batch())
    result.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model.transformer.wte.weight.grad[5].abs().sum() > 0


def test_singleton_catalog_exact_has_zero_distillation_loss():
    catalog = CATALOG[:1]
    model = tiny_model(catalog)
    inputs = {**batch(), "history_token_ids": catalog.unsqueeze(0).expand(2, -1, -1),
              "history_token_mask": torch.ones(2, 1, 3, dtype=torch.bool),
              "target_token_ids": catalog.expand(2, -1)}
    trainer = RewardDistillationMethod(Cache(torch.ones(2, 1)), catalog, sft(), estimator="exact")
    result = trainer.compute_loss(model, inputs)
    assert result.extra[LOSS_COMPONENTS_KEY]["distill"].item() == 0.
    result.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("sampling", ["reward", "curriculum", "curriculum_smooth"])
def test_gpt2_validation_skips_reward_cache_and_differentiable_prefill(monkeypatch, sampling):
    model = tiny_model()
    trainer = RewardDistillationMethod(object(), CATALOG, sft(), sampling=sampling, curriculum_steps=10)
    monkeypatch.setattr(model, "cache_history_encoding_grad", lambda *args: pytest.fail("unexpected prefill"))
    inputs = batch()
    del inputs["example_indices"]
    expected = sft().compute_loss(model, inputs, stage="val").loss
    actual = trainer.compute_loss(model, inputs, stage="val").loss
    torch.testing.assert_close(actual, expected)


def test_gpt2_exact_checkpoint_cpu_autocast_backward():
    model = tiny_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = method(estimator="exact", catalog_chunk_size=1).compute_loss(model, batch())
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gpt2_exact_component_calibration_backpropagates_shared_context():
    model = tiny_model()
    trainer = method(estimator="exact", catalog_chunk_size=1)
    result = calibrate_component_weight(
        model, trainer, [batch()], component="distill", target_ratio=.1, num_batches=1,
    )
    assert result["weight"] > 0
    assert result["effective_ratio_check"] == pytest.approx(.1)
