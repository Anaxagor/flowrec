"""Catalog-normalized GPT-2 scores and their use by all GFlow objectives."""

from types import SimpleNamespace

import pytest
import torch

from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


CATALOG = torch.tensor([[1, 4, 7], [1, 4, 8], [1, 5, 9], [2, 6, 10]])


def tiny_model(*, catalog=True, dropout=0.0):
    torch.manual_seed(42)
    return GPT2RecModel(
        codebook_size=4, sem_id_len=3,
        semantic_id_mapping={str(i): row.tolist() for i, row in enumerate(CATALOG)} if catalog else None,
        embedding_dim=8, num_heads=2, num_layers=1, dim_feedforward=16,
        max_position_embeddings=16, dropout=dropout,
    )


def batch():
    history = torch.tensor([[[0, 0, 0], [1, 4, 7]], [[1, 5, 9], [2, 6, 10]]])
    return {
        "user_ids": torch.arange(2),
        "history_token_ids": history,
        "history_token_mask": history.ne(0),
        "target_token_ids": CATALOG[[0, 3]],
        "target_token_mask": torch.ones(2, 3, dtype=torch.bool),
        "example_indices": torch.arange(2),
    }


def dense_reference(catalog, logits, targets, mask):
    """Independent dense vocabulary mask, including every catalog prefix."""
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    for b in range(targets.size(0)):
        for n in range(targets.size(1)):
            for d in range(targets.size(2)):
                matches = catalog[:, :d].eq(targets[b, n, :d]).all(-1)
                allowed[b, n, d, catalog[matches, d].unique()] = True
    values = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_probs = values.masked_fill(~allowed, -torch.inf).log_softmax(-1)
    safe_targets = targets.masked_fill(~mask, 0)
    return log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, 0), allowed


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_catalog_scores_and_gradients_match_dense_mask(monkeypatch, dtype):
    model = tiny_model()
    targets = CATALOG.unsqueeze(0).expand(2, -1, -1)
    mask = torch.ones_like(targets, dtype=torch.bool)
    logits = torch.randn(2, 4, 3, model._semantic_vocab_size, dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    monkeypatch.setattr(model, "candidate_token_logits", lambda *a, **kw: logits)
    actual = model.score_item_tokens_valid_actions(batch(), candidate_token_ids=targets)
    expected, allowed = dense_reference(CATALOG, reference_logits, targets, mask)
    tolerance = 1e-6 if dtype == torch.float16 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-3, atol=tolerance)
    assert torch.count_nonzero(logits.grad[~allowed]) == 0
    # [1, 5] and [2, 6] each have just one child: no loss or gradient at that step.
    assert torch.count_nonzero(actual[:, 2:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, 2:, -1]) == 0


def test_catalog_distribution_masks_and_single_candidate_shapes():
    model = tiny_model().eval()
    targets = CATALOG.unsqueeze(0).expand(2, -1, -1).clone()
    full = model.score_items_valid_actions(batch(), candidate_token_ids=targets)
    torch.testing.assert_close(full.exp().sum(-1), torch.ones(2))
    assert (model.score_items(batch(), candidate_token_ids=targets).exp().sum(-1) < 0.1).all()
    single = model.score_items_valid_actions(batch(), candidate_token_ids=targets[:, 0])
    torch.testing.assert_close(single, full[:, 0])

    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[:, 1, -1] = False
    targets[:, 1, -1] = -100  # masked padding must not enter the vocabulary gather
    tokens = model.score_item_tokens_valid_actions(
        batch(), candidate_token_ids=targets, candidate_token_mask=mask,
    )
    assert torch.count_nonzero(tokens[:, 1, -1]) == 0
    expected, _ = dense_reference(
        CATALOG, model.candidate_token_logits(batch(), candidate_token_ids=targets, candidate_token_mask=mask),
        targets, mask,
    )
    torch.testing.assert_close(tokens, expected)
    prefix = model.score_items_valid_actions(batch(), candidate_token_ids=targets[:, 1, :2])
    torch.testing.assert_close(prefix, tokens[:, 1].sum(-1))


def test_loading_catalog_invalidates_normalization_index():
    model = tiny_model().eval()
    candidates = CATALOG.unsqueeze(0).expand(2, -1, -1)
    model.score_items_valid_actions(batch(), candidate_token_ids=candidates)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["item_semantic_ids"][1] = torch.tensor([1, 5, 8])
    model.load_state_dict(state)
    candidates = state["item_semantic_ids"].unsqueeze(0).expand(2, -1, -1)
    logits = model.candidate_token_logits(batch(), candidate_token_ids=candidates)
    expected, _ = dense_reference(state["item_semantic_ids"], logits, candidates, torch.ones_like(candidates).bool())
    actual = model.score_item_tokens_valid_actions(batch(), candidate_token_ids=candidates)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1).exp().sum(-1), torch.ones(2))


def test_no_catalog_retains_unconstrained_scoring_like_tiger():
    model = tiny_model(catalog=False).eval()
    candidates = CATALOG.unsqueeze(0).expand(2, -1, -1)
    actual = model.score_items_valid_actions(batch(), candidate_token_ids=candidates)
    expected = model.score_items(batch(), candidate_token_ids=candidates)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
def test_gflow_automatically_uses_catalog_scores_in_one_forward(monkeypatch, objective):
    from scripts.gflow_finetune_gpt2_beauty import GPT2GFlowMethod

    model = tiny_model(dropout=0.2).train()
    candidates = CATALOG.unsqueeze(0).expand(2, -1, -1)
    mask = torch.ones_like(candidates, dtype=torch.bool)
    rows = torch.arange(4).expand(2, -1)
    trajectories = SemanticTrajectoryBatch(candidates, mask, rows.eq(0), rows)
    rewards = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    sampler = SimpleNamespace(sample=lambda *a, **kw: trajectories)
    cache = SimpleNamespace(select_candidates=lambda *a, include_full=False, **kw: (
        rewards, rewards.sum(-1), rewards if include_full else None,
    ))
    method = GPT2GFlowMethod(
        GFlowConfig(objective=objective, sft_weight=0.0, reward_cache_path="injected"),
        candidate_sampler=sampler,
        sft_method=SFTMethod(SFTConfig(compute_train_metrics=False)),
    )
    method._reward_cache = cache
    calls = []
    forward_calls = []
    original = model.candidate_token_logits

    def capture(*args, **kwargs):
        logits = original(*args, **kwargs)
        calls.append(logits)
        return logits

    def forbid_unconstrained(*args, **kwargs):
        pytest.fail("GFlow must not fall back to unconstrained sequence scores")

    monkeypatch.setattr(model, "candidate_token_logits", capture)
    monkeypatch.setattr(model, "score_items", forbid_unconstrained)
    hook = model.model.register_forward_hook(lambda *a: forward_calls.append(1))
    try:
        result = method.compute_loss(model, batch())
    finally:
        hook.remove()
    assert len(calls) == 1
    assert len(forward_calls) == 2  # one SFT forward and one for every candidate transition
    expected, _ = dense_reference(CATALOG, calls[0], candidates, mask)
    torch.testing.assert_close(result.extra["sequence_logprobs"], expected.sum(-1))
    if objective != "tb":
        torch.testing.assert_close(result.extra["token_logprobs"], expected)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
