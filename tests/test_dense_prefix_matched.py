"""The opt-in GPT-2 layout matches TIGER targets, histories, and catalog."""
from dataclasses import replace
import json

import pandas as pd
import pytest
import torch

from flowrec.data.collators import DenseCausalCollator, SequenceBatchCollator
from flowrec.data.sequence_builder import (
    SequenceSplitConfig,
    build_sequence_splits,
    factorize_interaction_ids,
)
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule


def _events(lengths=(3, 5, 53, 205, 270)):
    rows = [
        dict(raw_user_id=str(user), raw_item_id=f"{user}_{i}",
             timestamp=1000 + i, rating=float(i % 5),
             verified_purchase=bool(i % 2), order_idx=i)
        for user, length in enumerate(lengths) for i in range(length)
    ]
    return factorize_interaction_ids(pd.DataFrame(rows))[0]


@pytest.mark.parametrize("policy", ["last_n", "all"])
@pytest.mark.parametrize("min_history", [1, 3])
def test_selected_targets_and_full_contexts_match_prefix(policy, min_history):
    events = _events()
    prefix_cfg = SequenceSplitConfig(
        train_target_policy=policy, train_last_n=128,
        max_history_len=50, min_history_len=min_history,
    )
    prefix = build_sequence_splits(events, cfg=prefix_cfg)
    dense = build_sequence_splits(events, cfg=replace(
        prefix_cfg, train_example_layout="dense_causal", train_dense_match_prefix=True,
    ))
    # All metadata and all histories agree, including the earliest selected
    # target's full preceding context, which a simple loss-mask crop can lose.
    pd.testing.assert_frame_equal(dense.train[prefix.train.columns], prefix.train, check_dtype=False)
    pd.testing.assert_frame_equal(dense.val, prefix.val)
    pd.testing.assert_frame_equal(dense.test, prefix.test)
    for row in dense.train.itertuples():
        assert row.label_mask == [False] * (len(row.history_item_ids) - 1) + [True]
        assert row.label_item_ids == row.history_item_ids[1:] + [row.target_item_id]
        assert row.label_raw_items == row.history_raw_items[1:] + [row.target_raw_item]
    if policy == "last_n":
        assert dense.train.groupby("user_id").size().max() == 128
        assert len(dense.train.loc[dense.train.user_id.eq(4)].iloc[0].history_item_ids) == 50


def test_legacy_dense_still_supervises_all_targets_despite_last_n_policy():
    events = _events((270,))
    cfg = SequenceSplitConfig(train_example_layout="dense_causal", max_history_len=50)
    legacy_all = build_sequence_splits(events, cfg=cfg).train
    legacy_last = build_sequence_splits(events, cfg=replace(
        cfg, train_target_policy="last_n", train_last_n=128,
    )).train
    pd.testing.assert_frame_equal(legacy_all, legacy_last)
    assert sum(sum(mask) for mask in legacy_last.label_mask) == 267


@pytest.mark.parametrize("override,message", [
    ({"train_example_layout": "prefix_target"}, "train_example_layout"),
    ({"train_window_stride": 50}, "train_window_stride"),
    ({"train_window_len": 49}, "train_window_len"),
    ({"max_history_len": None}, "max_history_len"),
    ({"max_history_len": 1, "min_history_len": 2}, "max_history_len"),
    ({"train_target_policy": "sampled"}, "train_target_policy"),
    ({"train_target_policy": "all", "train_all_prefixes": False}, "train_target_policy"),
    ({"train_last_n": 0}, "train_last_n"),
])
def test_rejects_settings_that_break_prefix_equivalence(override, message):
    cfg = SequenceSplitConfig(
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        max_history_len=50, train_target_policy="last_n", train_last_n=128,
    )
    with pytest.raises(ValueError, match=message):
        build_sequence_splits(_events(), cfg=replace(cfg, **override))


def test_yambda_rejects_incompatible_dense_prefix_conversion_before_io():
    cfg = YambdaDataConfig(
        interactions_path="/nonexistent/events.parquet", train_example_layout="dense_causal",
        train_dense_match_prefix=True, train_prefix_from_dense=True,
    )
    with pytest.raises(ValueError, match="train_prefix_from_dense"):
        YambdaDataModule(cfg).setup()


def test_yambda_matches_catalog_semantic_ids_and_heldout_batches(tmp_path):
    rows = [dict(uid=user, item_id=i % 61, timestamp=1000 + i, played_ratio_pct=100)
            for user, length in enumerate([5, 53, 205, 270]) for i in range(length)]
    # 888 appears in an old discarded history and as a held-out target; it
    # must be removed just as for TIGER. 999 is a completely cold held-out item.
    rows[-270]["item_id"] = 888
    rows[-1]["item_id"] = 999
    rows[3]["item_id"] = 888
    path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(path)
    sid_path = tmp_path / "sids.json"
    sid_path.write_text(json.dumps({str(i): [i // 32, i % 32]
                                    for i in [*range(61), 888, 999]}))
    prefix_cfg = YambdaDataConfig(
        interactions_path=path, semantic_id_path=sid_path, hf_recent_months=None,
        min_user_interactions=3, min_item_interactions=0,
        train_target_policy="last_n", train_last_n=128, max_history_len=50,
        num_workers=0, persistent_workers=False, pin_memory=False,
        include_raw_in_batch=False, batch_size=4, eval_batch_size=64,
    )
    prefix = YambdaDataModule(prefix_cfg)
    dense = YambdaDataModule(replace(
        prefix_cfg, train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_window_len=50, train_window_stride=1,
    ))
    prefix.setup()
    dense.setup()
    assert dense.item2id == prefix.item2id
    assert dense.semantic_ids == prefix.semantic_ids
    assert "999" not in dense.item2id
    assert "888" not in dense.item2id
    pd.testing.assert_frame_equal(
        dense.train_dataset.examples[prefix.train_dataset.examples.columns],
        prefix.train_dataset.examples,
    )
    for name in ("val", "test"):
        pd.testing.assert_frame_equal(getattr(dense, f"{name}_dataset").examples,
                                      getattr(prefix, f"{name}_dataset").examples)
        for left, right in zip(getattr(prefix, f"{name}_dataloader")(),
                               getattr(dense, f"{name}_dataloader")(), strict=True):
            for key in left:
                if isinstance(left[key], torch.Tensor):
                    torch.testing.assert_close(left[key], right[key])
    batch = next(iter(dense.train_dataloader()))
    assert batch["label_mask"].sum(dim=1).tolist() == [1] * len(batch["user_ids"])
    assert batch["token_label_mask"].sum(dim=1).tolist() == [2] * len(batch["user_ids"])
    assert dense.summary()["train_dense_match_prefix"] is True


@pytest.mark.parametrize("padding", ["left", "right"])
def test_semantic_masks_supervise_only_the_complete_final_target(padding):
    events = _events((5, 205))
    splits = build_sequence_splits(events, cfg=SequenceSplitConfig(
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_target_policy="last_n", train_last_n=128, max_history_len=50,
    ))
    examples = splits.train.iloc[[0, 1, -1]].to_dict("records")
    sids = {raw: [i // 16 + 1, i % 16 + 1]
            for i, raw in enumerate(events.raw_item_id)}
    batch = DenseCausalCollator(max_seq_len=50, semantic_ids=sids, sid_len=2,
                               history_padding=padding)(examples)
    for i, example in enumerate(examples):
        active = batch["token_label_mask"][i]
        assert batch["label_token_ids"][i, active].tolist() == sids[example["target_raw_item"]]
        stream = [token for item in example["history_raw_items"] + [example["target_raw_item"]]
                  for token in sids[item]]
        assert batch["input_token_ids"][i, batch["token_attention_mask"][i]].tolist() == stream[:-1]


def test_matched_dense_and_prefix_have_same_gpt2_loss_and_gradients():
    pytest.importorskip("transformers")
    from flowrec.ft_methods.sft import SFTConfig, SFTMethod
    from flowrec.models.gpt2 import GPT2RecModel

    torch.manual_seed(42)
    events = _events((5, 205))
    cfg = SequenceSplitConfig(train_target_policy="last_n", train_last_n=128, max_history_len=50)
    prefix = build_sequence_splits(events, cfg=cfg).train.iloc[[0, 1, -1]]
    dense = build_sequence_splits(events, cfg=replace(
        cfg, train_example_layout="dense_causal", train_dense_match_prefix=True,
    )).train.iloc[[0, 1, -1]]
    sids = {raw: [i // 16 + 1, i % 16 + 1]
            for i, raw in enumerate(events.raw_item_id)}
    dense_batch = DenseCausalCollator(max_seq_len=50, semantic_ids=sids, sid_len=2,
                                     history_padding="right")(dense.to_dict("records"))
    prefix_batch = SequenceBatchCollator(max_history_len=50, semantic_ids=sids, sid_len=2,
                                         history_padding="left")(prefix.to_dict("records"))
    model = GPT2RecModel(codebook_size=16, sem_id_len=2, semantic_id_mapping=sids,
                         embedding_dim=8, num_heads=2, num_layers=2, dim_feedforward=16,
                         max_position_embeddings=104, dropout=0.0).eval()
    method = SFTMethod(SFTConfig(compute_train_metrics=False))
    dense_loss = method.compute_loss(model, dense_batch).loss
    dense_loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    prefix_loss = method.compute_loss(model, prefix_batch).loss
    prefix_loss.backward()
    torch.testing.assert_close(dense_loss, prefix_loss)
    for name, parameter in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], parameter.grad, rtol=2e-4, atol=2e-6)
