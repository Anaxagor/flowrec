"""Tests for the scalable cross-entropy (SCE) item objective and its SFT wiring."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.ft_methods.sft.scalable_cross_entropy import (
    ScalableCrossEntropyConfig,
    ScalableCrossEntropyLoss,
)
from flowrec.models.sasrec import SASRec, SASRecConfig


def _exact_cross_entropy_without_pad(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Full softmax over the catalog with the padding column (id 0) removed."""
    logits = (hidden.reshape(-1, hidden.size(-1)) @ weight.T).clone()
    logits[:, 0] = -math.inf
    return F.cross_entropy(logits, targets.reshape(-1), ignore_index=-100)


def _exhaustive(n_examples: int, vocab: int) -> ScalableCrossEntropyConfig:
    """Every bucket keeps every supervised position and every non-pad item."""
    return ScalableCrossEntropyConfig(
        n_buckets=3, bucket_size_x=n_examples, bucket_size_y=vocab - 1
    )


def test_exhaustive_buckets_match_full_cross_entropy_and_gradients() -> None:
    torch.manual_seed(0)
    n, d, v = 300, 16, 50
    x = torch.randn(n, d, requires_grad=True)
    w = torch.randn(v, d, requires_grad=True)
    y = torch.randint(1, v, (n,))

    exact = _exact_cross_entropy_without_pad(x, y, w)
    sce = ScalableCrossEntropyLoss(_exhaustive(n, v))(x, y, w)
    torch.testing.assert_close(sce, exact, rtol=0, atol=1e-5)

    grads_exact = torch.autograd.grad(exact, [x, w])
    grads_sce = torch.autograd.grad(sce, [x, w])
    for g_exact, g_sce in zip(grads_exact, grads_sce):
        torch.testing.assert_close(g_sce, g_exact, rtol=1e-5, atol=1e-6)


def test_padded_and_masked_positions_are_excluded() -> None:
    torch.manual_seed(1)
    b, t, d, v = 4, 10, 16, 50
    hidden = torch.randn(b, t, d)
    w = torch.randn(v, d)
    labels = torch.randint(1, v, (b, t))
    labels[:, 7:] = 0  # padding targets
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, :7] = True
    mask[0, 3] = False  # a real item id that must not be supervised

    sce = ScalableCrossEntropyLoss(_exhaustive(b * t, v))(
        hidden, labels, w, target_mask=mask
    )
    exact = _exact_cross_entropy_without_pad(
        hidden, labels.masked_fill(~mask, -100), w
    )
    torch.testing.assert_close(sce, exact, rtol=0, atol=1e-5)


@pytest.mark.parametrize("mix_x", [False, True])
def test_bucketed_regime_is_finite_and_clamps_bucket_sizes(mix_x: bool) -> None:
    torch.manual_seed(2)
    n, d, v = 200, 8, 30
    x = torch.randn(n, d, requires_grad=True)
    w = torch.randn(v, d)
    y = torch.randint(1, v, (n,))

    # Bucket sizes larger than the batch/catalog clamp instead of failing.
    oversized = ScalableCrossEntropyConfig(
        n_buckets=4, bucket_size_x=10_000, bucket_size_y=10_000, mix_x=mix_x
    )
    loss = ScalableCrossEntropyLoss(oversized)(x, y, w)
    assert torch.isfinite(loss)
    (grad,) = torch.autograd.grad(loss, [x])
    assert torch.isfinite(grad).all()

    small = ScalableCrossEntropyConfig(
        n_buckets=4, bucket_size_x=16, bucket_size_y=8, mix_x=mix_x
    )
    loss_small = ScalableCrossEntropyLoss(small)(x, y, w)
    assert torch.isfinite(loss_small)
    (grad_small,) = torch.autograd.grad(loss_small, [x])
    # Only positions selected by at least one bucket receive gradient.
    assert int((grad_small.abs().sum(-1) == 0).sum()) > 0


def test_invalid_targets_raise() -> None:
    x = torch.randn(5, 4)
    w = torch.randn(9, 4)
    loss = ScalableCrossEntropyLoss(
        ScalableCrossEntropyConfig(n_buckets=1, bucket_size_x=5, bucket_size_y=4)
    )
    with pytest.raises(ValueError, match="no valid supervised targets"):
        loss(x, torch.zeros(5, dtype=torch.long), w)
    with pytest.raises(ValueError, match="outside the item embedding table"):
        loss(x, torch.tensor([1, 2, 3, 4, 9]), w)
    with pytest.raises(ValueError, match="align"):
        loss(x, torch.tensor([1, 2]), w)


def test_config_validation_and_round_trip() -> None:
    cfg = SFTConfig.from_dict(
        {
            "item_loss": "Scalable_Cross_Entropy",
            "scalable_cross_entropy": {
                "n_buckets": 7,
                "bucket_size_x": 3,
                "bucket_size_y": 5,
                "mix_x": True,
            },
        }
    )
    assert cfg.item_loss == "scalable_cross_entropy"
    assert cfg.scalable_cross_entropy == ScalableCrossEntropyConfig(7, 3, 5, mix_x=True)
    assert SFTConfig.from_dict(cfg.to_dict()) == cfg
    assert SFTConfig().item_loss == "cross_entropy"
    with pytest.raises(ValueError, match="item_loss"):
        SFTConfig.from_dict({"item_loss": "sce"})
    with pytest.raises(ValueError, match="n_buckets"):
        ScalableCrossEntropyConfig(n_buckets=0)
    with pytest.raises(ValueError, match="auto"):
        ScalableCrossEntropyConfig.from_dict({"n_buckets": "auto"})


def _dense_batch(v: int, b: int, t: int) -> dict[str, torch.Tensor]:
    inputs = torch.randint(1, v, (b, t))
    attention = torch.ones(b, t, dtype=torch.bool)
    labels = torch.randint(1, v, (b, t))
    label_mask = torch.ones(b, t, dtype=torch.bool)
    # Left padding on the first row, unsupervised real labels on the second.
    inputs[0, :4] = 0
    attention[0, :4] = False
    labels[0, :4] = 0
    label_mask[0, :4] = False
    label_mask[1, 5:8] = False
    return {
        "input_item_ids": inputs,
        "attention_mask": attention,
        "label_item_ids": labels,
        "label_mask": label_mask,
    }


def test_sft_method_uses_sce_only_for_dense_training_batches() -> None:
    torch.manual_seed(3)
    v, t, b, d = 40, 12, 5, 16
    model = SASRec(
        SASRecConfig(
            item_vocab_size=v,
            max_history_len=t,
            d_model=d,
            nhead=2,
            num_layers=1,
            dropout=0.0,
            pos_drop=0.0,
        )
    )
    model.eval()  # deterministic hidden states for the comparison
    batch = _dense_batch(v, b, t)
    method = SFTMethod(
        SFTConfig.from_dict(
            {
                "item_loss": "scalable_cross_entropy",
                "scalable_cross_entropy": {
                    "n_buckets": 2,
                    "bucket_size_x": b * t,
                    "bucket_size_y": v - 1,
                },
                "compute_train_metrics": False,
            }
        )
    )
    masked_labels = batch["label_item_ids"].masked_fill(~batch["label_mask"], -100)

    train_out = method.compute_loss(model, batch, stage="train")
    assert train_out.model_output.logits is None  # no [B, T, V] tensor was built
    hidden = model.dense_item_hidden_states(batch).detach()
    exact = _exact_cross_entropy_without_pad(
        hidden, masked_labels, model.item_embedding.weight.detach()
    )
    torch.testing.assert_close(train_out.loss.detach(), exact, rtol=0, atol=1e-5)
    train_out.loss.backward()
    assert torch.isfinite(model.item_embedding.weight.grad).all()

    # Validation/test on a dense batch fall back to the exact full-catalog loss.
    val_out = method.validation_step(model, batch)
    assert val_out.model_output.logits is not None
    exact_full = F.cross_entropy(
        val_out.model_output.logits.reshape(-1, v),
        masked_labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(val_out.loss, exact_full, rtol=0, atol=1e-6)

    # Prefix-target validation batches still produce ranking metrics.
    prefix = {
        "history_item_ids": torch.randint(1, v, (b, t)),
        "history_mask": torch.ones(b, t, dtype=torch.bool),
        "target_item_ids": torch.randint(1, v, (b,)),
    }
    prefix_out = method.validation_step(model, prefix)
    assert "sft/ndcg@10" in prefix_out.metrics
