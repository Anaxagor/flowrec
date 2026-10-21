"""Candidate chunking keeps one global loss and recomputes in the forward mode."""
from __future__ import annotations

import copy

import pytest
import torch

from flowrec.ft_methods.gpt2_history_cache import run_candidate_chunks
from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig
from flowrec.ft_methods.mini_one_rec.method import MiniOneRecMethod
from flowrec.ft_methods.preference_sdpo.config import SDPOConfig
from flowrec.ft_methods.preference_sdpo.method import SDPOMethod
from flowrec.models.gpt2 import GPT2RecModel

ATOL = 2e-6
RTOL = 3e-4
CATALOG = torch.tensor([[1, 4, 7], [1, 4, 8], [1, 5, 9], [2, 6, 10]])


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


def _sdpo(
    model: GPT2RecModel,
    *,
    chunk_size: int | None,
    checkpoint: bool,
) -> SDPOMethod:
    method = SDPOMethod(
        SDPOConfig(
            num_negatives=2,
            sft_weight=0.0,
            calibrate_sft_weight=False,
            evaluate_generation=False,
            compute_train_metrics=False,
            gpt2_shared_history_cache=True,
            gpt2_candidate_chunk_size=chunk_size,
            gpt2_checkpoint_candidates=checkpoint,
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


def test_chunk_flags_default_off_and_roundtrip():
    assert SDPOConfig().gpt2_candidate_chunk_size is None
    assert SDPOConfig().gpt2_checkpoint_candidates is False
    assert MiniOneRecConfig().gpt2_candidate_chunk_size is None
    parsed = SDPOConfig.from_dict({
        "calibrate_sft_weight": False,
        "gpt2_candidate_chunk_size": 4,
        "gpt2_checkpoint_candidates": True,
    })
    assert parsed.to_dict()["gpt2_candidate_chunk_size"] == 4
    assert parsed.to_dict()["gpt2_checkpoint_candidates"] is True
    with pytest.raises(ValueError, match="chunk_size"):
        SDPOConfig(gpt2_candidate_chunk_size=0, calibrate_sft_weight=False)


@pytest.mark.parametrize(
    ("chunk_size", "checkpoint"),
    [(1, False), (2, False), (3, False), (1, True), (2, True)],
)
def test_sdpo_chunked_scores_match_one_global_loss(chunk_size: int, checkpoint: bool):
    model = _tiny_model()
    full = _sdpo(model, chunk_size=None, checkpoint=False).compute_loss(model, _batch(), step=0)
    full_grads = _grads(full.loss, model)
    chunked = _sdpo(model, chunk_size=chunk_size, checkpoint=checkpoint).compute_loss(
        model, _batch(), step=0
    )
    chunked_grads = _grads(chunked.loss, model)
    torch.testing.assert_close(chunked.loss, full.loss, atol=ATOL, rtol=RTOL)
    for actual, expected in zip(chunked_grads, full_grads):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_checkpoint_recompute_stays_in_eval_after_train_is_restored():
    class Probe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.25))
            self.seen: list[bool] = []

    probe = Probe().eval()
    ids = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
    mask = torch.ones(1, 3, 2, dtype=torch.bool)

    def score(_batch, chunk_ids, _mask):
        probe.seen.append(probe.training)
        return probe.scale * chunk_ids.sum(dim=-1, keepdim=True)

    scored = run_candidate_chunks(
        score,
        probe,
        ids,
        mask,
        {"unused": torch.tensor(1.0)},
        chunk_size=1,
        checkpoint_chunks=True,
    )
    probe.train()
    scored.sum().backward()
    assert probe.seen == [False, False, False, False, False, False]
    assert probe.training is True
    assert probe.scale.grad is not None


def test_grpo_checkpoint_grads_match_after_returning_to_train():
    dropout = 0.5
    plain_model = _tiny_model(dropout=dropout).eval()
    checkpoint_model = copy.deepcopy(plain_model).eval()
    tokens = CATALOG.unsqueeze(0).expand(2, -1, -1).contiguous()
    mask = torch.ones_like(tokens, dtype=torch.bool)
    batch = _batch()

    def method(checkpoint: bool) -> MiniOneRecMethod:
        return MiniOneRecMethod(
            MiniOneRecConfig(
                num_rollouts=4,
                sft_weight=0.0,
                calibrate_sft_weight=False,
                evaluate_generation=False,
                rollout_constrain_valid_sequences=True,
                gpt2_shared_history_cache=True,
                gpt2_candidate_chunk_size=1,
                gpt2_checkpoint_candidates=checkpoint,
            )
        )

    plain = method(False)._candidate_token_logprobs(plain_model, batch, tokens, mask)
    checked = method(True)._candidate_token_logprobs(checkpoint_model, batch, tokens, mask)
    plain_model.train()
    checkpoint_model.train()
    plain_grads = _grads(plain.sum(), plain_model)
    checked_grads = _grads(checked.sum(), checkpoint_model)
    torch.testing.assert_close(checked, plain, atol=ATOL, rtol=RTOL)
    for actual, expected in zip(checked_grads, plain_grads):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    assert plain_model.training and checkpoint_model.training
