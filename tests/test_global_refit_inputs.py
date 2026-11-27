"""Check the refit window, the original test cohort and teacher IDs on deliberate cold events."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from flowrec.data.datasets import DatasetStats, SequenceDataset
from flowrec.data.dense_prefix import example_digest, expand_dense_training_split
from flowrec.data.post_split_filtering import filter_splits_to_train_item_catalog, remap_interactions_to_catalog
from flowrec.data.sequence_builder import SequenceSplitConfig, build_sequence_splits, factorize_interaction_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
from refit_inputs import load_native_refit, prepare_native_refit, rebuild_combined_training
from yambda_gflow_teacher import dense_teacher_ids


@pytest.mark.parametrize("dense,tail", [(False, False), (True, False), (False, True)])
def test_refit_rebuilds_every_eligible_training_label_and_preserves_test(dense, tail):
    # a's event at 11 is cold, first seen after teacher/student training. A new
    # user b is allowed to contribute once its pre-test history has a warm item.
    events = [
        ("a", "A", 1), ("a", "B", 2), ("a", "C", 3), ("a", "A", 9),
        ("a", "B", 10), ("a", "X", 11), ("a", "C", 12), ("a", "A", 13),
        ("a", "B", 20), ("a", "C", 21),
        ("b", "B", 10), ("b", "A", 11), ("b", "C", 20),
    ]
    raw = pd.DataFrame(events, columns=["raw_user_id", "raw_item_id", "timestamp"])
    raw["rating"], raw["verified_purchase"] = 1., True
    raw, users, _, _, items = factorize_interaction_ids(raw)
    native = SequenceSplitConfig(split_strategy="global_time", max_history_len=2, train_last_n=3,
        train_target_policy="last_n" if tail else "all", global_val_timestamp=10, global_test_timestamp=20,
        global_eval_target_mode="last_per_user", global_history_policy="rolling")
    reference = replace(native, train_example_layout="dense_causal" if dense else "prefix_target")
    frozen = filter_splits_to_train_item_catalog(build_sequence_splits(raw, cfg=reference), old_id2item=items)
    mapped = remap_interactions_to_catalog(raw, frozen.item2id)
    splits = frozen.splits
    if dense:
        expand_dense_training_split(splits, mapped, last_n=None, catalog_items=len(frozen.item2id))
    cfg = SimpleNamespace(**asdict(native), train_prefix_from_dense=dense)
    dm = SimpleNamespace(cfg=cfg, interactions=mapped, item2id=frozen.item2id,
        num_items=len(frozen.item2id), global_val_cutoff=10, global_test_cutoff=20,
        train_dataset=SequenceDataset(splits.train, "train"), val_dataset=SequenceDataset(splits.val, "val"),
        test_dataset=SequenceDataset(splits.test, "test"), stats=DatasetStats(2, 3, 0, 0, 0, len(mapped)))
    test_digest = example_digest(splits.test)
    teacher_ids, size = dense_teacher_ids(mapped, val_cutoff=10, student_vocab_size=len(frozen.id2item), num_users=2)
    _, new_size = dense_teacher_ids(mapped, val_cutoff=20, student_vocab_size=len(frozen.id2item), num_users=2)
    assert size == 4 and new_size == 5
    audit = rebuild_combined_training(dm, block_size=1)
    assert audit["test_sha256"] == test_digest == example_digest(dm.test_dataset.examples)
    assert audit["original_val_examples"] == 2 and len(dm.val_dataset) == 0
    assert dm.train_item_counts_by_raw_id == {"A": 4, "B": 3, "C": 2}
    # Both test events (timestamp >=20) and the validation-only item X are excluded.
    rows = dm.train_dataset.examples
    expected = {10, 13} if dense else ({12, 13} if tail else {10, 12, 13})
    a = rows[rows.user_id == users["a"]]
    assert set(a.loc[a.target_timestamp >= 10, "target_timestamp"]) == expected
    assert list(rows[rows.user_id == users["b"]].target_timestamp) == [11]
    assert max(rows.target_timestamp) == 13
    assert audit["pool_validation_window_targets"] == len(expected) + 1
    for row in rows.itertuples():
        assert 0 < row.target_item_id < size and all(0 < item < size for item in row.history_item_ids)
    # Windows are anchored at the last pre-test target. The dense label at
    # t=12 sees only X, while a rolling warm history also contains B.
    if not dense:
        assert a.loc[a.target_timestamp.eq(12)].iloc[0].history_item_ids == [frozen.item2id["B"]]
    assert a.loc[a.target_timestamp.eq(13)].iloc[0].history_item_ids == [frozen.item2id["C"]]
    assert list(teacher_ids) == list(range(size))
    assert dm.global_val_cutoff == dm.global_test_cutoff == 20


def test_sampled50_uses_last128_at_test_boundary_and_keeps_all_test_rows(tmp_path):
    import json
    import torch
    import ce_yambda_prepared_data as compact
    from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
    from gflow_finetune_yambda import store_reward_events
    from precompute_reward_cache import per_user_random_indices

    path = tmp_path / "events.parquet"
    rows = [dict(uid=user, item_id=pos % 17, timestamp=1000 + pos, played_ratio_pct=100, event_type="listen")
            for user in range(2) for pos in range(216)]
    rows.extend(dict(uid=0, item_id=1, timestamp=t, played_ratio_pct=1, event_type="listen")
                for t in [1140, 1199, 1200])
    pd.DataFrame(rows).to_parquet(path)
    sids = tmp_path / "ids.json"
    sids.write_text(json.dumps({str(i): [i % 8, 10 + i // 8] for i in range(17)}))
    cfg = YambdaDataConfig(interactions_path=path, semantic_id_path=sids, dataset_size="500m",
        hf_recent_months=None, min_user_interactions=3, split_strategy="global_time",
        global_val_timestamp=1150, global_test_timestamp=1200, global_eval_target_mode="last_per_user",
        train_target_policy="last_n", train_last_n=128, max_history_len=50,
        positive_criterion="played_ratio_pct >= 50", include_raw_in_batch=False,
        num_workers=0, persistent_workers=False, batch_size=2, pin_memory=False)
    dm = YambdaDataModule(cfg)
    dm.setup()
    original = next(iter(dm.test_dataloader()))
    audit = rebuild_combined_training(dm, block_size=50)
    assert audit["pool_train_examples"] == 256 and audit["pool_validation_window_targets"] == 100
    assert set(dm.train_dataset.examples.target_position) == set(range(72, 200))
    store_reward_events(dm, tmp_path)
    for name, end in [("positive", 1199), ("auxiliary", 1199)]:
        assert pd.read_parquet(tmp_path / f"{name}.parquet").timestamp.max() == end
    selected = per_user_random_indices(dm.train_dataset, 50, 42)
    compact.prepare(tmp_path / "prefixes", dm, selected, {"subset_seed": 42}, audit=audit)
    stored = compact.load(tmp_path / "prefixes", cfg, {"subset_seed": 42})
    assert stored.train_dataset.examples.groupby("user_id").size().tolist() == [50, 50]
    assert stored._ce_prepared_source_indices.tolist() == selected
    actual = next(iter(stored.test_dataloader()))
    for key, tensor in original.items():
        if torch.is_tensor(tensor):
            torch.testing.assert_close(actual[key], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["prefix", "packed_dense", "matched_dense"])
def test_native_refit_cache_preserves_labels_and_frozen_test(tmp_path, layout):
    import json
    import torch
    from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

    # ID 9 is first observed in validation. ID 8 is first observed on test.
    seq = [1, 2, 3, 1, 2, 3, 9, 2, 3, 1, 8, 2]
    rows = [dict(uid=u, timestamp=t, item_id=v, played_ratio_pct=100, event_type="listen")
            for u in (1, 2) for t, v in enumerate(seq, 1)]
    events = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(events)
    sid = tmp_path / "ids.json"
    sid.write_text(json.dumps({str(i): [i, 10 + i] for i in set(seq)}))
    matched = layout == "matched_dense"
    dense = layout != "prefix"
    cfg = YambdaDataConfig(interactions_path=events, semantic_id_path=sid, dataset_size="50m",
        hf_recent_months=None, min_user_interactions=1, min_item_interactions=1, split_strategy="global_time",
        global_val_timestamp=6, global_test_timestamp=11, global_eval_target_mode="last_per_user",
        train_target_policy="last_n" if matched else "all", train_last_n=3, max_history_len=4,
        train_example_layout="dense_causal" if dense else "prefix_target", train_dense_match_prefix=matched,
        train_window_len=4, train_window_stride=1 if matched else 2, include_raw_in_batch=False,
        positive_criterion="played_ratio_pct >= 50", num_workers=0, persistent_workers=False,
        batch_size=2, pin_memory=False)
    dm = YambdaDataModule(cfg)
    dm.setup()
    reference_test = next(iter(dm.test_dataloader()))
    original_catalog = dm.item2id.copy()
    expected_audit = rebuild_combined_training(dm, block_size=3, native_layout=True)

    prepare_native_refit(tmp_path / "cache", YambdaDataModule, cfg)
    cached, audit = load_native_refit(tmp_path / "cache", replace(cfg, batch_size=1))
    assert audit["test_sha256"] == expected_audit["test_sha256"]
    assert len(audit["tensors_sha256"]) == 64
    assert audit["pool_validation_window_targets"] == (6 if matched else 8)
    assert audit["max_target_timestamp"] == 10
    assert cached.item2id == original_catalog and "9" not in original_catalog
    assert cached.train_item_counts_by_raw_id == dm.train_item_counts_by_raw_id
    assert len(cached.train_dataset) == len(dm.train_dataset) and len(cached.val_dataset) == 0
    for i in range(len(cached.train_dataset)):
        expected, got = dm.train_dataset[i], cached.train_dataset[i]
        keys = ["history_item_ids", "history_raw_items", "target_item_id", "target_timestamp"]
        if dense:
            keys += ["label_mask", "label_item_ids", "label_raw_items"]
        assert all(expected[k] == got[k] for k in keys)
        assert all(got["history_item_ids"]) and got["target_item_id"] > 0
        if matched:
            assert got["label_mask"] == [False] * (len(got["history_item_ids"]) - 1) + [True]
        elif layout == "packed_dense":
            assert got["label_item_ids"] == got["history_item_ids"][1:] + [got["target_item_id"]]
    # Exercise both native collators, also on the reloaded, memory-mapped train rows.
    next(iter(cached.train_dataloader()))
    actual = next(iter(cached.test_dataloader()))
    for key, expected in reference_test.items():
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
