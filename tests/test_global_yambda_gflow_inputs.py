"""Regression checks for the global last-128 layouts and fine-tuning subsets."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from flowrec.data.post_split_filtering import filter_splits_to_train_item_catalog, remap_interactions_to_catalog
from flowrec.data.sequence_builder import build_sequence_splits, factorize_interaction_ids
from flowrec.data.yambda import YambdaDataConfig, _sequence_split_config_from_yambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
from yambda_gflow_inputs import _verify_dense_rows


def test_global_dense_replay_and_observed_remap_match_all_rows():
    rows = []
    for user in range(70):
        for position in range(190 if user == 0 else 6):
            timestamp = position if user == 0 else [0, 1, 2, 178, 183, 189][position]
            # Includes cold items in the held-out windows and repeated IDs.
            item = "cold" if timestamp in (178, 183) else str(position % 13)
            rows.append(dict(raw_user_id=str(user), raw_item_id=item, timestamp=timestamp,
                             order_idx=len(rows), rating=float(position % 5), verified_purchase=None))
    events, users, _, _, catalog = factorize_interaction_ids(pd.DataFrame(rows))
    cfg = YambdaDataConfig(split_strategy="global_time", global_val_timestamp=175, global_test_timestamp=185,
                           global_eval_target_mode="last_per_user", train_target_policy="last_n", train_last_n=128,
                           max_history_len=50, history_padding="right")
    prefix = build_sequence_splits(events, cfg=_sequence_split_config_from_yambda(cfg))
    reference = filter_splits_to_train_item_catalog(prefix, old_id2item=catalog)
    actual = filter_splits_to_train_item_catalog(prefix, old_id2item=catalog, normalized_prefix_train=True)
    pd.testing.assert_frame_equal(actual.splits.train, reference.splits.train)
    assert actual.stats == reference.stats
    dm = SimpleNamespace(cfg=cfg, num_users=len(users), item2id=actual.item2id, id2item=actual.id2item,
                         interactions=remap_interactions_to_catalog(events, actual.item2id))
    for split in ("train", "val", "test"):
        setattr(dm, split + "_dataset", SimpleNamespace(examples=getattr(actual.splits, split)))
    audit = _verify_dense_rows(dm)
    assert audit["rows_compared"]["train"] == 128 + 2 * 69
    assert audit["rows_compared"]["test"] == 70
    original = dm.train_dataset.examples.at[0, "history_item_ids"]
    dm.train_dataset.examples.at[0, "history_item_ids"] = original[1:]
    with pytest.raises(ValueError, match="dense/prefix mismatch"):
        _verify_dense_rows(dm)


def test_global_tiger_sampled50_preparation(tmp_path, monkeypatch):
    import gflow_finetune_yambda as runner
    import ce_yambda_prepared_data as compact
    import gflow_finetune_yambda500m as tiger
    import yambda_gflow_teacher
    from flowrec.data.yambda import YambdaDataModule
    from sasrec_config_runner import training_item_popularity

    # One user has more than 128 train targets, one fewer than 50.
    times = [range(154), [0, 1, 2, 3, 150, 153]]
    events = [dict(uid=u, item_id=t % 8, timestamp=t, event_type="listen", played_ratio_pct=99)
              for u, timestamps in enumerate(times) for t in timestamps]
    events.append({**events[0], "event_type": "like"})
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(events).to_parquet(events_path)
    cfg = YambdaDataConfig(interactions_path=events_path, min_user_interactions=1, hf_recent_months=None,
        positive_criterion="event_type == 'listen'", train_target_policy="last_n", train_last_n=128,
        split_strategy="global_time", global_val_timestamp=150, global_test_timestamp=152,
        global_eval_target_mode="last_per_user", drop_items_missing_semantic_id=False)
    cli = SimpleNamespace(model="tiger", dataset=runner.Y500, smoke=False, device="cpu", num_workers=0)
    args = runner.training_args(tiger, cli, 2024, tmp_path, tmp_path)
    assert (args.train_random_n, args.train_subset_seed, args.seed) == (50, 42, 2024)
    source = YambdaDataModule(cfg)
    source.setup()
    expected = tiger.finetune_indices(source.train_dataset, args)
    expected_frames = {s: getattr(source, s + "_dataset").examples for s in ("train", "val", "test")}
    assert len(expected_frames["train"]) == 131 and len(expected) == 53
    translation = tmp_path / "translation.json"
    monkeypatch.setattr(tiger.tiger_runner, "build_data_config", lambda *a, **k: cfg)
    monkeypatch.setattr(yambda_gflow_teacher, "prepare_teacher_translation",
                        lambda *args: translation.write_text("{}"))
    subset = dict(sampling="uniform_without_replacement", pool_last_n=128, **tiger.subset_spec(args))
    session = SimpleNamespace(base=tiger, args=args, cache_root=tmp_path, output_root=tmp_path,
        source=dict(dataset=cli.dataset, model=cli.model), source_identity={"finetune_subset": subset},
        config_path=tmp_path / "config.json", config={"method": {"sasrec": {"item_id_translation_path": str(translation)}}})
    profile = SimpleNamespace(teacher_checkpoint=None, verify_layout=lambda dm: dm.summary())
    runner.prepare(session, profile)
    prepared = compact.load(tmp_path / "prefixes", cfg, session.source_identity)
    indices = prepared._ce_prepared_source_indices
    assert indices.tolist() == expected == sorted(set(expected))
    assert sorted(expected_frames["train"].iloc[expected].groupby("user_id").size()) == [3, 50]
    for split, frame in expected_frames.items():
        if split == "train":
            frame = frame.iloc[expected]
        actual = getattr(prepared, split + "_dataset")
        assert [actual[i]["history_item_ids"] for i in range(len(actual))] == frame["history_item_ids"].tolist()
        assert actual.examples["target_position"].tolist() == frame["target_position"].tolist()
    assert prepared.item2id == source.item2id
    # ARP counts all 154 positive train events, including those outside last_128
    # and sampled50. The like and the events at/after the cutoff do not count.
    popularity = torch.zeros(source.item_vocab_size, dtype=torch.float64)
    for raw_item, count in enumerate([20, 20, 20, 20, 19, 19, 18, 18]):
        popularity[source.item2id[str(raw_item)]] = count
    assert prepared.interactions.empty
    torch.testing.assert_close(training_item_popularity(source), popularity)
    torch.testing.assert_close(training_item_popularity(prepared), popularity)
    assert len(pd.read_parquet(tmp_path / "positive.parquet")) == 154
    assert len(pd.read_parquet(tmp_path / "auxiliary.parquet")) == 1
    record = session.config["finetune_subset"]
    assert (record["num_source_examples"], record["num_selected_examples"]) == (131, 53)
    assert record["source_indices_sha256"] == hashlib.sha256(torch.tensor(expected).numpy().tobytes()).hexdigest()
    audit = json.loads((tmp_path / "preparation/layout_audit.json").read_text())
    assert audit["num_train_examples"] == 131 and audit["finetune_subset"] == record
