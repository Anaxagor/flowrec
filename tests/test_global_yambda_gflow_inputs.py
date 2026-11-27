"""Regression checks for the global last-128 layouts and fine-tuning subsets."""
from dataclasses import asdict
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
from yambda_gflow_inputs import GlobalYambdaProfile, _verify_dense_rows


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


@pytest.mark.parametrize("model", ["tiger", "gpt2"])
def test_global_sampled50_preparation(tmp_path, monkeypatch, model):
    import ce_finetune_yambda as ce
    import gflow_finetune_yambda as runner
    import ce_yambda_prepared_data as compact
    import gflow_finetune_gpt2_amazon as gpt2
    import gflow_finetune_yambda500m as tiger
    import yambda_gflow_inputs
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
    base = tiger if model == "tiger" else gpt2
    cli = SimpleNamespace(model=model, dataset=runner.Y500, smoke=False, device="cpu", num_workers=0)
    args = runner.training_args(base, cli, 2024, tmp_path, tmp_path)
    assert (args.train_random_n, args.train_subset_seed, args.seed) == (50, 42, 2024)
    source = YambdaDataModule(cfg)
    source.setup()
    expected = tiger.finetune_indices(source.train_dataset, args)
    expected_frames = {s: getattr(source, s + "_dataset").examples for s in ("train", "val", "test")}
    assert len(expected_frames["train"]) == 131 and len(expected) == 53
    teacher_trial = tmp_path / "teacher_trial.json"
    teacher_trial.write_text("{}")
    profile = GlobalYambdaProfile(
        model=model, dataset=cli.dataset, checkpoint=tmp_path / "student.pt",
        teacher_checkpoint=tmp_path / "teacher.pt", teacher_trial=teacher_trial,
        semantic_ids=tmp_path / "semantic_ids.json", codebook=tmp_path / "codebook.pt",
        model_config={}, data_config=asdict(cfg), recorded_summary=source.summary(),
        source_identity={"student": "fixture"},
    )
    if model == "gpt2":
        asset = yambda_gflow_inputs.GPT_INPUTS[cli.dataset][1]
        sidecar = tmp_path / "ce-mmap-v1" / "first" / asset.name
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text('{"trial":17}')
        profile.source_identity["inputs"] = {str(asset): str(sidecar)}
    monkeypatch.setattr(base.tiger_runner, "build_data_config", lambda *a, **k: cfg)
    monkeypatch.setattr(yambda_gflow_inputs, "stage_inputs", lambda *args: None)
    monkeypatch.setattr(yambda_gflow_inputs, "load_profile", lambda *args: profile)
    monkeypatch.setattr(yambda_gflow_teacher, "prepare_teacher_translation",
                        lambda dataset, dm, checkpoint, destination: destination.write_text("{}"))
    output = tmp_path / "results"
    prepare_args = ["prepare", "--model", model, "--dataset", cli.dataset, "--workspace", str(tmp_path)]
    first, second = (runner, ce) if model == "tiger" else (ce, runner)
    first.main([*prepare_args, "--output-root", str(output)])
    pair = Path(model) / cli.dataset / "sampled50"
    scratch = tmp_path / runner.EXPERIMENT / pair
    identity = json.loads((scratch / "prepared.json").read_text())
    prepared = compact.load(scratch / "prefixes", cfg, identity)
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
    assert len(pd.read_parquet(scratch / "positive.parquet")) == 154
    assert len(pd.read_parquet(scratch / "auxiliary.parquet")) == 1
    record = json.loads((output / pair / "data_reward_config.json").read_text())["finetune_subset"]
    assert (record["num_source_examples"], record["num_selected_examples"]) == (131, 53)
    assert record["source_indices_sha256"] == hashlib.sha256(torch.tensor(expected).numpy().tobytes()).hexdigest()
    audit = json.loads((output / pair / "preparation/layout_audit.json").read_text())
    assert audit["num_train_examples"] == 131 and audit["finetune_subset"] == record
    if model == "gpt2":
        assert audit["dense_prefix_equal"]
        assert audit["rows_compared"] == {s: len(frame) for s, frame in expected_frames.items()}

    # The other experiment can publish its own audit using the same prepared rows.
    monkeypatch.setattr(YambdaDataModule, "setup", lambda self: pytest.fail("Rebuilt an existing snapshot"))
    if model == "gpt2":
        # Recopying scripts to the cluster restages this small sidecar at a new mtime-keyed path.
        restaged = tmp_path / "ce-mmap-v1" / "second" / asset.name
        restaged.parent.mkdir(parents=True)
        restaged.write_bytes(sidecar.read_bytes())
        profile.source_identity["inputs"][str(asset)] = str(restaged)
    second_output = tmp_path / "other_results"
    second.main([*prepare_args, "--output-root", str(second_output)])
    assert json.loads((second_output / pair / "preparation/layout_audit.json").read_text()) == audit
    assert json.loads((second_output / pair / "data_reward_config.json").read_text())["finetune_subset"] == record
    if model == "gpt2":
        restaged.write_text('{"trial":18}')
        with pytest.raises(RuntimeError, match="preparation inputs changed.*checkpoint.json"):
            second.main([*prepare_args, "--output-root", str(second_output)])
