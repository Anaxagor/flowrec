"""Tests for the dense-causal training layout.

Covers the boundary guarantees that matter most:
- dense supervision is identical to train_target_policy="all" prefix supervision;
- validation/test targets never leak into dense train rows;
- windowing supervises every eligible label exactly once;
- collator, SASRec dense forward, SFT loss, and post-split filtering agree on
  the input/label alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from flowrec.data.collators import DenseCausalCollator, SequenceBatchCollator
from flowrec.data.post_split_filtering import filter_splits_to_train_item_catalog
from flowrec.data.sequence_builder import (
    SequenceSplitConfig,
    build_sequence_splits,
    factorize_interaction_ids,
)
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig


def make_interactions(user_seqs: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    order = 0
    for user, items in user_seqs.items():
        for idx, item in enumerate(items):
            rows.append(
                {
                    "raw_user_id": user,
                    "raw_item_id": item,
                    "timestamp": 1_000 + idx,
                    "rating": 5.0,
                    "verified_purchase": True,
                    "order_idx": order,
                }
            )
            order += 1
    frame = pd.DataFrame(rows)
    frame, *_ = factorize_interaction_ids(frame)
    return frame


def unique_user_seqs(lengths: dict[str, int]) -> dict[str, list[str]]:
    """Every event gets a globally unique raw item, so raw-item membership
    identifies event positions without any index arithmetic."""
    return {user: [f"{user}_i{i}" for i in range(n)] for user, n in lengths.items()}


def dense_supervised_pairs(train: pd.DataFrame) -> set[tuple[str, str]]:
    pairs = set()
    for _, row in train.iterrows():
        for raw_item, valid in zip(row["label_raw_items"], row["label_mask"], strict=True):
            if valid:
                pairs.add((row["raw_user_id"], raw_item))
    return pairs


def prefix_supervised_pairs(train: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (row["raw_user_id"], row["target_raw_item"]) for _, row in train.iterrows()
    }


@pytest.mark.parametrize("min_history_len", [1, 3])
def test_loo_dense_matches_prefix_all_supervision(min_history_len: int) -> None:
    seqs = unique_user_seqs({"u1": 3, "u2": 4, "u3": 7, "u4": 12, "u5": 5})
    events = make_interactions(seqs)

    prefix_cfg = SequenceSplitConfig(
        min_history_len=min_history_len,
        train_target_policy="all",
        max_history_len=20,
    )
    dense_cfg = SequenceSplitConfig(
        min_history_len=min_history_len,
        train_example_layout="dense_causal",
        max_history_len=20,
    )

    prefix_splits = build_sequence_splits(events, cfg=prefix_cfg)
    dense_splits = build_sequence_splits(events, cfg=dense_cfg)

    assert dense_supervised_pairs(dense_splits.train) == prefix_supervised_pairs(
        prefix_splits.train
    )
    # Val/test are layout-independent.
    pd.testing.assert_frame_equal(prefix_splits.val, dense_splits.val)
    pd.testing.assert_frame_equal(prefix_splits.test, dense_splits.test)


def test_loo_dense_never_touches_holdout_events() -> None:
    seqs = unique_user_seqs({"u1": 5, "u2": 9, "u3": 30})
    events = make_interactions(seqs)
    cfg = SequenceSplitConfig(
        train_example_layout="dense_causal",
        max_history_len=6,
        train_window_len=6,
    )
    splits = build_sequence_splits(events, cfg=cfg)

    holdout_items = set()
    for user, items in seqs.items():
        holdout_items.update(items[-2:])  # val and test targets
    assert set(splits.val["target_raw_item"]) | set(splits.test["target_raw_item"]) <= holdout_items

    for _, row in splits.train.iterrows():
        observed = set(row["history_raw_items"]) | set(row["label_raw_items"])
        assert not observed & holdout_items, (
            f"holdout event leaked into dense train row for {row['raw_user_id']}"
        )


def test_global_time_dense_labels_before_cutoff() -> None:
    # Timestamps are 1000 + position; put the last 3 events of the long users
    # past the cutoffs.
    seqs = unique_user_seqs({"u1": 10, "u2": 8, "u3": 12})
    events = make_interactions(seqs)
    item_ts = dict(zip(events["raw_item_id"], events["timestamp"], strict=True))

    val_cutoff, test_cutoff = 1_007, 1_009
    prefix_cfg = SequenceSplitConfig(
        split_strategy="global_time",
        train_target_policy="all",
        global_val_timestamp=val_cutoff,
        global_test_timestamp=test_cutoff,
        max_history_len=20,
    )
    dense_cfg = SequenceSplitConfig(
        split_strategy="global_time",
        train_example_layout="dense_causal",
        global_val_timestamp=val_cutoff,
        global_test_timestamp=test_cutoff,
        max_history_len=20,
    )
    prefix_splits = build_sequence_splits(events, cfg=prefix_cfg)
    dense_splits = build_sequence_splits(events, cfg=dense_cfg)

    for _, row in dense_splits.train.iterrows():
        for raw_item in list(row["history_raw_items"]) + list(row["label_raw_items"]):
            assert item_ts[raw_item] < val_cutoff

    assert dense_supervised_pairs(dense_splits.train) == prefix_supervised_pairs(
        prefix_splits.train
    )


@pytest.mark.parametrize("window_len,stride", [(8, 8), (8, 3)])
def test_windowing_tiles_labels_exactly_once(window_len: int, stride: int) -> None:
    seqs = unique_user_seqs({"u1": 32})
    events = make_interactions(seqs)
    cfg = SequenceSplitConfig(
        train_example_layout="dense_causal",
        max_history_len=window_len,
        train_window_len=window_len,
        train_window_stride=stride,
    )
    splits = build_sequence_splits(events, cfg=cfg)

    labeled = []
    for _, row in splits.train.iterrows():
        assert len(row["history_item_ids"]) <= window_len
        assert (
            len(row["history_item_ids"])
            == len(row["label_item_ids"])
            == len(row["label_mask"])
        )
        labeled.extend(
            raw
            for raw, valid in zip(row["label_raw_items"], row["label_mask"], strict=True)
            if valid
        )

    # Train region is items 0..29 (val=30, test=31): every item 1..29 must be a
    # supervised label exactly once regardless of window overlap.
    expected = [f"u1_i{i}" for i in range(1, 30)]
    assert sorted(labeled) == sorted(expected)
    assert len(labeled) == len(set(labeled))


def test_dense_collator_alignment_and_padding() -> None:
    seqs = unique_user_seqs({"u1": 7, "u2": 12, "u3": 4})
    events = make_interactions(seqs)
    cfg = SequenceSplitConfig(train_example_layout="dense_causal", max_history_len=8)
    splits = build_sequence_splits(events, cfg=cfg)

    collator = DenseCausalCollator(max_seq_len=8, pad_item_id=0, include_raw=True)
    examples = splits.train.to_dict(orient="records")
    batch = collator(examples)

    bsz = len(examples)
    for key in ("input_item_ids", "attention_mask", "label_item_ids", "label_mask"):
        assert batch[key].shape == (bsz, 8)

    mask = batch["attention_mask"]
    lengths = mask.sum(dim=1)
    for row in range(bsz):
        n = int(lengths[row])
        # Right padding: real positions form a prefix.
        assert mask[row, :n].all() and not mask[row, n:].any()
        assert not batch["label_mask"][row, n:].any()
        # label[i] is the item at input position i+1.
        assert torch.equal(
            batch["label_item_ids"][row, : n - 1], batch["input_item_ids"][row, 1:n]
        )
        # The last real label is the row's target item.
        assert int(batch["label_item_ids"][row, n - 1]) == int(
            examples[row]["target_item_id"]
        )


def _make_eval_sasrec(vocab_size: int, max_len: int) -> SASRec:
    model = SASRec(
        SASRecConfig(
            item_vocab_size=vocab_size,
            max_history_len=max_len,
            d_model=16,
            nhead=2,
            num_layers=1,
            dropout=0.0,
            pos_drop=0.0,
        )
    )
    model.eval()
    return model


def test_sasrec_dense_forward_matches_prefix_scores() -> None:
    torch.manual_seed(0)
    vocab, max_len, bsz = 50, 10, 4
    model = _make_eval_sasrec(vocab, max_len)

    lengths = [3, 10, 6, 1]
    items = torch.zeros(bsz, max_len, dtype=torch.long)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    for row, n in enumerate(lengths):
        items[row, :n] = torch.randint(1, vocab, (n,))
        mask[row, :n] = True

    with torch.no_grad():
        dense_out = model({"input_item_ids": items, "attention_mask": mask})
        prefix_out = model({"history_item_ids": items, "history_mask": mask})

    assert dense_out.logits is not None and dense_out.logits.shape == (bsz, max_len, vocab)
    last_idx = torch.as_tensor([n - 1 for n in lengths])
    last_logits = dense_out.logits[torch.arange(bsz), last_idx]
    torch.testing.assert_close(last_logits, prefix_out.scores, rtol=1e-4, atol=1e-5)


def test_sft_dense_loss_matches_manual_masked_ce() -> None:
    torch.manual_seed(0)
    vocab, max_len, bsz = 30, 6, 3
    model = _make_eval_sasrec(vocab, max_len)
    method = SFTMethod(SFTConfig(compute_train_metrics=False))

    inputs = torch.randint(1, vocab, (bsz, max_len))
    attention = torch.ones(bsz, max_len, dtype=torch.bool)
    labels = torch.randint(1, vocab, (bsz, max_len))
    label_mask = torch.rand(bsz, max_len) > 0.4
    label_mask[:, -1] = True  # at least one valid label per row
    batch = {
        "user_ids": torch.arange(bsz),
        "input_item_ids": inputs,
        "attention_mask": attention,
        "label_item_ids": labels,
        "label_mask": label_mask,
    }

    out = method.compute_loss(model, batch, stage="train")
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.loss.requires_grad

    with torch.no_grad():
        logits = model(batch).logits
        target = labels.masked_fill(~label_mask, -100)
        manual = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), target.reshape(-1), ignore_index=-100
        )
    torch.testing.assert_close(out.loss.detach(), manual)


def test_post_split_filter_renumbers_dense_rows_without_drops() -> None:
    seqs = unique_user_seqs({"u1": 6, "u2": 9, "u3": 4})
    events = make_interactions(seqs)
    _, _, _, _, id2item = factorize_interaction_ids(
        pd.DataFrame(
            [
                {
                    "raw_user_id": u,
                    "raw_item_id": i,
                    "timestamp": 1_000 + k,
                    "rating": 5.0,
                    "verified_purchase": True,
                    "order_idx": 0,
                }
                for u, items in seqs.items()
                for k, i in enumerate(items)
            ]
        )
    )
    cfg = SequenceSplitConfig(train_example_layout="dense_causal", max_history_len=8)
    splits = build_sequence_splits(events, cfg=cfg)
    n_rows_before = len(splits.train)

    result = filter_splits_to_train_item_catalog(
        splits, old_id2item=id2item, train_item_source="observed"
    )
    train = result.splits.train
    assert len(train) == n_rows_before
    assert result.stats.dropped_train_examples == 0
    assert result.stats.removed_train_history_items == 0

    for _, row in train.iterrows():
        mapped_hist = [result.item2id[raw] for raw in row["history_raw_items"]]
        mapped_labels = [result.item2id[raw] for raw in row["label_raw_items"]]
        assert row["history_item_ids"] == mapped_hist
        assert row["label_item_ids"] == mapped_labels
        assert row["label_item_ids"][:-1] == row["history_item_ids"][1:]
        assert row["target_item_id"] == row["label_item_ids"][-1] or not row["label_mask"][-1]

    with pytest.raises(ValueError, match="observed"):
        filter_splits_to_train_item_catalog(
            splits, old_id2item=id2item, train_item_source="targets"
        )


def test_datamodule_dense_end_to_end(tmp_path) -> None:
    from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule

    rng = np.random.default_rng(0)
    rows = []
    for u in range(12):
        n = int(rng.integers(5, 15))
        for i in range(n):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "asin": f"item{int(rng.integers(0, 40))}",
                    "timestamp": 1_000 + i,
                    "rating": 5.0,
                    "verified_purchase": True,
                }
            )
    reviews_path = tmp_path / "reviews.parquet"
    pd.DataFrame(rows).to_parquet(reviews_path)

    # Dense layout demands right padding; the default "left" must fail fast.
    bad_cfg = AmazonReviewDataConfig(
        reviews_path=reviews_path,
        train_example_layout="dense_causal",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    with pytest.raises(ValueError, match="history_padding"):
        AmazonReviewDataModule(bad_cfg).setup()

    cfg = AmazonReviewDataConfig(
        reviews_path=reviews_path,
        min_user_interactions=5,
        min_item_interactions=0,
        train_example_layout="dense_causal",
        max_history_len=8,
        history_padding="right",
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    data = AmazonReviewDataModule(cfg)
    data.setup()

    train_batch = next(iter(data.train_dataloader()))
    for key in ("input_item_ids", "attention_mask", "label_item_ids", "label_mask"):
        assert key in train_batch
    assert train_batch["input_item_ids"].shape[1] == 8
    assert int(train_batch["label_mask"].sum()) > 0
    max_item = int(train_batch["input_item_ids"].max())
    assert max_item < data.item_vocab_size

    val_batch = next(iter(data.val_dataloader()))
    assert "history_item_ids" in val_batch and "target_item_ids" in val_batch
    assert "input_item_ids" not in val_batch

    summary = data.summary()
    assert summary["train_example_layout"] == "dense_causal"


def test_yambda_datamodule_dense_end_to_end(tmp_path) -> None:
    from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

    rng = np.random.default_rng(0)
    rows = []
    for u in range(10):
        n = int(rng.integers(6, 60))
        for i in range(n):
            rows.append(
                {
                    "uid": u,
                    "item_id": int(rng.integers(0, 50)),
                    "timestamp": 1_000 + i,
                    "played_ratio_pct": 100,
                }
            )
    interactions_path = tmp_path / "yambda.parquet"
    pd.DataFrame(rows).to_parquet(interactions_path)

    cfg = YambdaDataConfig(
        interactions_path=interactions_path,
        min_user_interactions=5,
        min_item_interactions=0,
        train_example_layout="dense_causal",
        max_history_len=16,
        train_window_stride=8,
        history_padding="right",
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        include_raw_in_batch=False,
    )
    data = YambdaDataModule(cfg)
    data.setup()

    train_batch = next(iter(data.train_dataloader()))
    for key in ("input_item_ids", "attention_mask", "label_item_ids", "label_mask"):
        assert key in train_batch
    assert train_batch["input_item_ids"].shape[1] == 16
    assert int(train_batch["label_mask"].sum()) > 0

    val_batch = next(iter(data.val_dataloader()))
    assert "history_item_ids" in val_batch and "input_item_ids" not in val_batch
    assert data.summary()["train_example_layout"] == "dense_causal"


def test_vk_lsvd_dense_guards_fire_before_data_load() -> None:
    from flowrec.data.vk_lsvd import VKLSVDDataConfig, VKLSVDDataModule

    # Left padding rejected first (config-only check, no files needed).
    with pytest.raises(ValueError, match="history_padding"):
        VKLSVDDataModule(
            VKLSVDDataConfig(train_example_layout="dense_causal")
        ).setup()

    # The 'provided' split strategy has no dense builder.
    with pytest.raises(NotImplementedError, match="provided"):
        VKLSVDDataModule(
            VKLSVDDataConfig(
                train_example_layout="dense_causal",
                history_padding="right",
                split_strategy="provided",
            )
        ).setup()


def test_popularity_fits_from_dense_batches() -> None:
    from flowrec.models.popularity import PopularityConfig, PopularityModel

    dense_batch = {
        "input_item_ids": torch.tensor([[1, 2, 3, 0], [2, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        "label_item_ids": torch.tensor([[2, 3, 4, 0], [2, 5, 0, 0]]),
        "label_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
    }
    model = PopularityModel(PopularityConfig(num_items=10)).fit_from_dataloader(
        [dense_batch]
    )
    counts = model.item_counts.cpu()
    # Valid labels: 2,3,4 and 2,5 -> item 2 twice, items 3/4/5 once, pads zero.
    assert counts[2] == 2 and counts[3] == 1 and counts[4] == 1 and counts[5] == 1
    assert counts[0] == 0

    with pytest.raises(ValueError, match="countable"):
        PopularityModel(PopularityConfig(num_items=10)).fit_from_dataloader(
            [{"user_ids": torch.arange(2)}]
        )
