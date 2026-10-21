"""SDPO and GRPO use the GPT2 history cache only when the flag is on."""
from __future__ import annotations

import copy

import pytest
import torch

from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig
from flowrec.ft_methods.mini_one_rec.method import MiniOneRecMethod
from flowrec.ft_methods.preference_sdpo.config import SDPOConfig
from flowrec.ft_methods.preference_sdpo.method import SDPOMethod
from flowrec.models.base import RecModelOutput
from flowrec.models.gpt2 import GPT2RecModel

ATOL = 2e-6
RTOL = 3e-4
CATALOG = torch.tensor([[1, 4, 7], [1, 4, 8], [1, 5, 9], [2, 6, 10]])


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, batch):
        ids = batch["target_token_ids"]
        return RecModelOutput(logits=self.weight * torch.ones(ids.shape[0], ids.shape[1], 4))

    def sequence_logprobs(self, batch, *, target_token_ids, target_token_mask):
        return self.weight * target_token_ids[:, 0].float()

    def score_items(self, batch, *, candidate_token_ids, candidate_token_mask):
        return self.weight * candidate_token_ids[:, :, 0].float()


def _tiny_model(*, dropout: float = 0.0) -> GPT2RecModel:
    torch.manual_seed(17)
    return GPT2RecModel(
        codebook_size=4,
        sem_id_len=3,
        semantic_id_mapping={str(i): sid.tolist() for i, sid in enumerate(CATALOG)},
        embedding_dim=8,
        num_heads=2,
        num_layers=2,
        dim_feedforward=16,
        max_position_embeddings=32,
        dropout=dropout,
        attention_dropout=dropout,
        residual_dropout=dropout,
        embedding_dropout=dropout,
    ).train()


def _batch() -> dict[str, torch.Tensor]:
    history = torch.tensor([
        [[1, 4, 7], [2, 6, 10]],
        [[0, 0, 0], [1, 5, 9]],
    ])
    return {
        "user_ids": torch.arange(2),
        "history_token_ids": history,
        "history_token_mask": history.ne(0),
        "target_token_ids": CATALOG[[1, 3]],
        "target_token_mask": torch.ones((2, 3), dtype=torch.bool),
        "target_item_ids": torch.tensor([1, 3]),
    }


def _grads(loss: torch.Tensor, model: torch.nn.Module) -> list[torch.Tensor]:
    params = tuple(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(param) if grad is None else grad for param, grad in zip(params, grads)]


def _sdpo(model: GPT2RecModel, *, cache: bool) -> SDPOMethod:
    method = SDPOMethod(
        SDPOConfig(
            num_negatives=2,
            sft_weight=0.0,
            calibrate_sft_weight=False,
            evaluate_generation=False,
            compute_train_metrics=False,
            gpt2_shared_history_cache=cache,
        ),
        model=model,
    )
    negatives = CATALOG[[0, 2]].unsqueeze(0).expand(2, -1, -1).contiguous()
    mask = torch.ones_like(negatives, dtype=torch.bool)

    def _fixed(batch, *, model, num_negatives, step):
        del batch, model, num_negatives, step
        return negatives, mask

    method._sample_distinct_rejected = _fixed
    return method


def test_shared_history_cache_defaults_off_and_roundtrips():
    assert SDPOConfig().gpt2_shared_history_cache is False
    assert MiniOneRecConfig().gpt2_shared_history_cache is False
    sdpo = SDPOConfig.from_dict({"gpt2_shared_history_cache": True, "calibrate_sft_weight": False})
    grpo = MiniOneRecConfig.from_dict({"gpt2_shared_history_cache": True, "calibrate_sft_weight": False})
    assert sdpo.to_dict()["gpt2_shared_history_cache"] is True
    assert grpo.to_dict()["gpt2_shared_history_cache"] is True


def test_non_gpt2_sdpo_keeps_the_legacy_scorer_when_the_flag_is_on():
    policy = _ToyPolicy()
    method = SDPOMethod(
        SDPOConfig(
            num_negatives=1,
            sft_weight=0.0,
            calibrate_sft_weight=False,
            evaluate_generation=False,
            gpt2_shared_history_cache=True,
        ),
        model=policy,
    )
    batch = {
        "target_token_ids": torch.tensor([[3, 1]]),
        "target_token_mask": torch.ones((1, 2), dtype=torch.bool),
    }
    rejected = torch.tensor([[[4, 1]]])
    rejected_mask = torch.ones_like(rejected, dtype=torch.bool)
    method._sample_distinct_rejected = lambda *args, **kwargs: (rejected, rejected_mask)
    output = method.compute_loss(policy, batch, step=0)
    assert torch.isfinite(output.loss)


def test_sdpo_cached_scoring_matches_legacy_loss_and_gradients():
    model = _tiny_model()
    with torch.no_grad():
        model.model.lm_head.weight[1].add_(0.02)
    legacy = _sdpo(model, cache=False).compute_loss(model, _batch(), step=0)
    legacy_grads = _grads(legacy.loss, model)
    cached_method = _sdpo(model, cache=True)
    cached = cached_method.compute_loss(model, _batch(), step=0)
    cached_grads = _grads(cached.loss, model)
    torch.testing.assert_close(cached.loss, legacy.loss, atol=ATOL, rtol=RTOL)
    for actual, expected in zip(cached_grads, legacy_grads):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    assert all(not parameter.requires_grad for parameter in cached_method.reference_model.parameters())


def test_sdpo_cache_falls_back_while_dropout_is_active(caplog):
    model = _tiny_model(dropout=0.5).train()

    def _forbidden(batch):
        raise AssertionError("shared history cache must not run with active SDPO dropout")

    method = _sdpo(model, cache=True)
    # The frozen reference is already in eval(), so its own prefill is safe.
    # Only the training policy must refuse the shared cache.
    model.cache_history_encoding_grad = _forbidden
    with caplog.at_level("WARNING"):
        output = method.compute_loss(model, _batch(), step=0)
    assert torch.isfinite(output.loss)
    assert "dropout is active" in caplog.text


@pytest.mark.parametrize("valid_actions", [False, True])
def test_grpo_cached_token_logprobs_match_legacy_and_keep_reference_frozen(valid_actions: bool):
    model = _tiny_model().eval()
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    tokens = CATALOG.unsqueeze(0).expand(2, -1, -1).contiguous()
    mask = torch.ones_like(tokens, dtype=torch.bool)
    batch = _batch()

    def method(cache: bool) -> MiniOneRecMethod:
        return MiniOneRecMethod(
            MiniOneRecConfig(
                num_rollouts=4,
                sft_weight=0.0,
                calibrate_sft_weight=False,
                evaluate_generation=False,
                rollout_constrain_valid_sequences=valid_actions,
                gpt2_shared_history_cache=cache,
            )
        )

    legacy = method(False)._candidate_token_logprobs(model, batch, tokens, mask)
    cached = method(True)._candidate_token_logprobs(model, batch, tokens, mask)
    weights = torch.tensor([0.2, -0.1, 0.4, 0.7]).view(1, 4, 1)
    legacy_grads = _grads((legacy * weights).sum(), model)
    cached_grads = _grads((cached * weights).sum(), model)
    torch.testing.assert_close(cached, legacy, atol=ATOL, rtol=RTOL)
    for actual, expected in zip(cached_grads, legacy_grads):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)

    with torch.no_grad():
        method(True)._candidate_token_logprobs(reference, batch, tokens, mask)
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert any(grad.abs().sum().item() > 0 for grad in cached_grads)
