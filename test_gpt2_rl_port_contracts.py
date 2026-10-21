"""Acceptance contracts for the selective GPT2 port into flowrec/rl_ablations.

Place this file under <flowrec>/tests and run:
    python -m pytest tests/test_gpt2_rl_port_contracts.py -q

These tests are intended to fail on the unpatched RL snapshot. They were
syntax-checked, but NOT executed against flowrec during the audit: the audit
runtime has no Transformers installation. Real method-level integration,
production data audits, AMP and performance tests are still required.

No downloads, model assets or GPU are required by the tests themselves.
The normal flowrec development dependencies must be installed.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowrec.data.collators import DenseCausalCollator, SequenceBatchCollator
from flowrec.data.sequence_builder import (
    SequenceSplitConfig,
    build_sequence_splits,
    factorize_interaction_ids,
)
from flowrec.models.gpt2 import GPT2RecModel

ATOL = 2e-6
RTOL = 3e-4
CATALOG = torch.tensor([[1, 4, 7], [1, 4, 8], [1, 5, 9], [2, 6, 10]])


def _events() -> pd.DataFrame:
    raw = pd.DataFrame([
        {
            "raw_user_id": "u",
            "raw_item_id": item,
            "timestamp": 1000 + i,
            "rating": 5.0,
            "verified_purchase": True,
        }
        for i, item in enumerate("ABCDEFGHI")
    ])
    return factorize_interaction_ids(raw)[0]


def _splits(*, dense: bool, stride: int = 3):
    cfg = SequenceSplitConfig(
        split_strategy="leave_one_out",
        min_history_len=1,
        max_history_len=3,
        train_example_layout="dense_causal" if dense else "prefix_target",
        train_window_len=3,
        train_window_stride=stride,
        train_all_prefixes=True,
        train_target_policy="all",
    )
    events = _events()
    return events, build_sequence_splits(events, cfg=cfg)


def _history_pairs(frame: pd.DataFrame) -> list[tuple[tuple[str, ...], str]]:
    ordered = frame.sort_values(["user_id", "target_position"], kind="stable")
    return [
        (tuple(row.history_raw_items), str(row.target_raw_item))
        for row in ordered.itertuples(index=False)
    ]


@pytest.mark.parametrize("source", ["amazon", "yambda"])
def test_runner_really_propagates_train_prefix_from_dense(tmp_path: Path, source: str):
    from scripts.run_tiger import build_data_config

    data: dict[str, Any] = {
        "source": source,
        "semantic_id_path": str(tmp_path / "semantic_ids.json"),
    }
    if source == "amazon":
        data["reviews_path"] = str(tmp_path / "reviews.parquet")
    else:
        data.update(interactions_path=str(tmp_path / "interactions.parquet"), use_hf=False)
    config = {
        "seed": 42,
        "data": data,
        "dataset": {
            "max_history_len": 3,
            "max_sequence_length": 3,
            "train_example_layout": "prefix_target",
            "train_prefix_from_dense": True,
            "train_window_len": 3,
            "train_window_stride": 3,
            "train_all_prefixes": True,
            "train_target_policy": "all",
            "hf_recent_months": None,
        },
        "model": {"use_user_token": False},
        "dataloader": {"num_workers": 0, "persistent_workers": False},
    }
    result = build_data_config(config, config_path=tmp_path / "launch.json")
    assert "train_prefix_from_dense" in {field.name for field in fields(result)}, (
        "The effective dataconfig does not declare the requested flag."
    )
    assert result.train_prefix_from_dense is True, "Runner silently discarded the flag."


def test_dense_expansion_is_not_a_sliding_prefix():
    from flowrec.data.dense_prefix import prefixes_from_dense_windows

    events, dense = _splits(dense=True)
    _, rolling = _splits(dense=False)
    prefix = prefixes_from_dense_windows(dense.train, events)
    assert _history_pairs(prefix) == [
        (("A",), "B"), (("A", "B"), "C"), (("A", "B", "C"), "D"),
        (("D",), "E"), (("D", "E"), "F"), (("D", "E", "F"), "G"),
    ]
    assert _history_pairs(rolling.train) == [
        (("A",), "B"), (("A", "B"), "C"), (("A", "B", "C"), "D"),
        (("B", "C", "D"), "E"), (("C", "D", "E"), "F"),
        (("D", "E", "F"), "G"),
    ]
    assert prefix.target_raw_item.tolist() == list("BCDEFG")
    assert len(prefix) == len(rolling.train) == 6
    assert dense.val.target_raw_item.tolist() == ["H"]
    assert dense.test.target_raw_item.tolist() == ["I"]
    for history, target in _history_pairs(prefix):
        assert target not in ("H", "I")
        assert "H" not in history and "I" not in history


def test_overlap_mask_and_last_n_preserve_original_window_context():
    from flowrec.data.dense_prefix import prefixes_from_dense_windows

    for stride in (1, 2, 3):
        events, splits = _splits(dense=True, stride=stride)
        full = prefixes_from_dense_windows(splits.train, events)
        assert len(full) == sum(sum(mask) for mask in splits.train.label_mask) == 6
        assert sorted(full.target_position.tolist()) == list(range(1, 7))
        assert not full.duplicated(["user_id", "target_position"]).any()
        tail = prefixes_from_dense_windows(splits.train, events, last_n=2)
        pd.testing.assert_frame_equal(tail, full.tail(2).reset_index(drop=True))


def test_expanding_training_does_not_change_heldout_rows():
    from flowrec.data.dense_prefix import example_digest, expand_dense_training_split

    events, splits = _splits(dense=True)
    before = (example_digest(splits.val), example_digest(splits.test))
    val_copy, test_copy = splits.val.copy(deep=True), splits.test.copy(deep=True)
    audit = expand_dense_training_split(splits, events, last_n=None, catalog_items=9)
    assert before == (example_digest(splits.val), example_digest(splits.test))
    pd.testing.assert_frame_equal(val_copy, splits.val)
    pd.testing.assert_frame_equal(test_copy, splits.test)
    assert audit["prefix_train_rows"] == 6
    assert audit["validation"]["equal"] and audit["test"]["equal"]


def _tiny_model() -> GPT2RecModel:
    torch.manual_seed(17)
    model = GPT2RecModel(
        codebook_size=4,
        sem_id_len=3,
        semantic_id_mapping={str(i): sid.tolist() for i, sid in enumerate(CATALOG)},
        embedding_dim=8,
        num_heads=2,
        num_layers=2,
        dim_feedforward=16,
        max_position_embeddings=32,
        dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
    )
    return model.eval()


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
    }


def _cache(model: GPT2RecModel, batch: dict[str, Any]):
    prepare = getattr(model, "cache_history_encoding_grad", None)
    assert callable(prepare), "Port the graph-preserving cache API before enabling the fast path."
    return prepare(batch)


def _grads(loss: torch.Tensor, model: GPT2RecModel) -> list[torch.Tensor]:
    params = tuple(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def _assert_grads_equal(left: list[torch.Tensor], right: list[torch.Tensor]) -> None:
    assert len(left) == len(right)
    for actual, expected in zip(left, right):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_teacher_forcing_does_not_read_current_or_future_candidate_tokens():
    model = _tiny_model()
    tokens = CATALOG[[0, 1, 3]].unsqueeze(0).expand(2, -1, -1).clone()
    expected = model.candidate_token_logits(_batch(), candidate_token_ids=tokens)
    for depth in range(tokens.size(-1)):
        changed = tokens.clone()
        # Keep token values in the model vocabulary; this test concerns raw
        # causal logits, not catalog validity of the perturbed candidate.
        changed[:, 0, depth] = (changed[:, 0, depth] + 1) % 10 + 1
        actual = model.candidate_token_logits(_batch(), candidate_token_ids=changed)
        torch.testing.assert_close(
            actual[:, 0, :depth + 1], expected[:, 0, :depth + 1], atol=ATOL, rtol=RTOL,
        )


@pytest.mark.parametrize("normalization", ["semantic", "valid_actions"])
@pytest.mark.parametrize("masked", [False, True])
def test_cache_matches_full_scoring_and_all_parameter_gradients(normalization: str, masked: bool):
    model = _tiny_model()
    tokens = CATALOG[[0, 1, 3]].unsqueeze(0).expand(2, -1, -1).clone()
    mask = torch.ones_like(tokens, dtype=torch.bool)
    if masked:
        tokens[:, 1, -1] = -100
        mask[:, 1, -1] = False
    scorer = (model.score_item_tokens if normalization == "semantic"
              else model.score_item_tokens_valid_actions)
    cached = _cache(model, _batch())
    states, prefix_mask, first_logits = cached["_gpt2_history_grad_state"]
    snapshots = [(key.detach().clone(), value.detach().clone()) for key, value in states]
    assert first_logits.requires_grad
    assert all(key.requires_grad and value.requires_grad for key, value in states)
    assert prefix_mask.tolist() == [[1] * 6, [0, 0, 0, 1, 1, 1]]
    weights = torch.tensor([0.7, -0.3, 1.1]).view(1, 3, 1)

    actual = scorer(cached, candidate_token_ids=tokens, candidate_token_mask=mask)
    actual_grads = _grads((actual * weights).sum(), model)
    expected = scorer(_batch(), candidate_token_ids=tokens, candidate_token_mask=mask)
    expected_grads = _grads((expected * weights).sum(), model)
    torch.testing.assert_close(actual.detach(), expected.detach(), atol=ATOL, rtol=RTOL)
    _assert_grads_equal(actual_grads, expected_grads)
    assert all(torch.isfinite(grad).all() for grad in actual_grads)
    for (key, value), (old_key, old_value) in zip(states, snapshots):
        torch.testing.assert_close(key.detach(), old_key)
        torch.testing.assert_close(value.detach(), old_value)


def test_history_only_token_embedding_receives_gradient():
    model = _tiny_model()
    tokens = CATALOG[[0, 1, 3]].unsqueeze(0).expand(2, -1, -1)
    # Token 5 occurs in history, but in none of these candidate IDs.
    values = model.score_item_tokens(_cache(model, _batch()), candidate_token_ids=tokens)
    (-values.sum()).backward()
    grad = model.model.transformer.wte.weight.grad
    assert grad is not None and grad[5].abs().sum().item() > 0


def test_one_token_candidate_uses_prefill_logits():
    model = _tiny_model()
    tokens = CATALOG[[0, 3]].unsqueeze(0).expand(2, -1, -1)[:, :, :1]
    actual = model.candidate_token_logits(_cache(model, _batch()), candidate_token_ids=tokens)
    expected = model.candidate_token_logits(_batch(), candidate_token_ids=tokens)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def _preference_loss(logp: torch.Tensor, ref_logp: torch.Tensor, beta: float = 0.1):
    rewards = beta * (logp - ref_logp)
    return (torch.logsumexp(rewards, dim=1) - rewards[:, 0]).mean()


def test_sdpo_combined_cached_scoring_matches_legacy_sum_semantics():
    model = _tiny_model()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():
        # Non-identical policy and reference make the value comparison nontrivial.
        model.model.lm_head.weight[1].add_(0.02)
    inputs = _batch()
    chosen = inputs["target_token_ids"]
    negatives = CATALOG[[0, 2]].unsqueeze(0).expand(2, -1, -1)
    tokens = torch.cat((chosen[:, None], negatives), dim=1)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    with torch.no_grad():
        ref_context = _cache(reference, inputs)
        ref_lp = reference.score_item_tokens(
            ref_context, candidate_token_ids=tokens, candidate_token_mask=masks,
        ).sum(-1)
    actual_lp = model.score_item_tokens(
        _cache(model, inputs), candidate_token_ids=tokens, candidate_token_mask=masks,
    ).sum(-1)
    actual_loss = _preference_loss(actual_lp, ref_lp)
    actual_grads = _grads(actual_loss, model)

    expected_lp = torch.cat((
        model.sequence_logprobs(inputs, target_token_ids=chosen,
                                target_token_mask=masks[:, 0])[:, None],
        model.score_items(inputs, candidate_token_ids=negatives,
                          candidate_token_mask=masks[:, 1:]),
    ), dim=1)
    expected_loss = _preference_loss(expected_lp, ref_lp)
    expected_grads = _grads(expected_loss, model)
    torch.testing.assert_close(actual_lp.detach(), expected_lp.detach(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual_loss.detach(), expected_loss.detach(), atol=ATOL, rtol=RTOL)
    _assert_grads_equal(actual_grads, expected_grads)
    assert all(parameter.grad is None for parameter in reference.parameters())


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_nonreentrant_chunking_preserves_scores_and_gradients(chunk_size: int):
    model = _tiny_model()
    tokens = CATALOG.unsqueeze(0).expand(2, -1, -1)
    cached = _cache(model, _batch())

    def score(context, selected):
        return model.score_item_tokens(
            context, candidate_token_ids=selected,
            candidate_token_mask=torch.ones_like(selected, dtype=torch.bool),
        ).sum(-1)

    chunks = [
        checkpoint(score, cached, tokens[:, start:start + chunk_size], use_reentrant=False)
        for start in range(0, tokens.size(1), chunk_size)
    ]
    actual = torch.cat(chunks, dim=1)
    # A single global logsumexp, not a sum of per-chunk preference losses.
    actual_loss = (torch.logsumexp(actual, dim=1) - actual[:, 0]).mean()
    actual_grads = _grads(actual_loss, model)
    expected = score(_batch(), tokens)
    expected_loss = (torch.logsumexp(expected, dim=1) - expected[:, 0]).mean()
    expected_grads = _grads(expected_loss, model)
    torch.testing.assert_close(actual.detach(), expected.detach(), atol=ATOL, rtol=RTOL)
    _assert_grads_equal(actual_grads, expected_grads)


def test_dense_and_expanded_prefix_sft_loss_and_gradients_match():
    from flowrec.data.dense_prefix import prefixes_from_dense_windows

    torch.manual_seed(11)
    events, splits = _splits(dense=True)
    prefix = prefixes_from_dense_windows(splits.train, events)
    mapping = {item: [i + 1, 17 + i] for i, item in enumerate("ABCDEFGHI")}
    model = GPT2RecModel(
        codebook_size=16, sem_id_len=2, semantic_id_mapping=mapping,
        embedding_dim=8, num_heads=2, num_layers=2, dim_feedforward=16,
        max_position_embeddings=32, dropout=0.0,
    ).eval()
    dense_batch = DenseCausalCollator(
        max_seq_len=3, history_padding="right", semantic_ids=mapping,
        sid_len=2, include_raw=False,
    )(splits.train.to_dict("records"))
    prefix_batch = SequenceBatchCollator(
        max_history_len=3, history_padding="left", semantic_ids=mapping,
        sid_len=2, include_raw=False,
    )(prefix.to_dict("records"))

    def masked_token_ce(logits, labels, mask):
        safe = labels.long().masked_fill(~mask.bool(), 0)
        values = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 safe.reshape(-1), reduction="none").reshape_as(safe)
        return (values * mask).sum() / mask.sum()

    dense_loss = masked_token_ce(
        model(dense_batch).logits, dense_batch["label_token_ids"],
        dense_batch["token_label_mask"],
    )
    dense_grads = _grads(dense_loss, model)
    prefix_loss = masked_token_ce(
        model(prefix_batch).logits, prefix_batch["target_token_ids"],
        prefix_batch["target_token_mask"],
    )
    prefix_grads = _grads(prefix_loss, model)
    assert int(dense_batch["token_label_mask"].sum()) == 12
    assert int(prefix_batch["target_token_mask"].sum()) == 12
    torch.testing.assert_close(dense_loss.detach(), prefix_loss.detach(), atol=ATOL, rtol=RTOL)
    _assert_grads_equal(dense_grads, prefix_grads)
