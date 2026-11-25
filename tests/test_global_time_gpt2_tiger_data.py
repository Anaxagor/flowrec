"""Check global Yambda training supervision and rolling evaluation against TIGER."""

from collections import Counter

import optuna
import pandas as pd
import pytest
import yaml

from flowrec.item_representation.base import SemanticIDStore
from scripts import global_time_tiger_tuning as tiger
from test_global_time_gpt2_tuning import tuning


DATASETS = ("yambda_50m_listens", "yambda_500m_listens")


def make_inputs(root, name):
    config = yaml.safe_load((tuning.CONFIG_DIR / "dataset_yambda.yaml").read_text())
    entry = config["datasets"][name]
    val, test = entry["global_val_timestamp"], entry["global_test_timestamp"]
    rows = []
    for user in range(1, 9):
        length = 145 if user == 1 else 8
        train = [(val - length + i, 1 + i % 13) for i in range(length)]
        # Item 98 is first observed in validation. Item 99 is an auxiliary
        # event below the positive threshold. Both have an embedding.
        held_out = [(val, 9), (val + 1, 98), (val + 2, 10), (val + 3, 11),
                    (test, 12), (test + 1, 13)]
        for timestamp, item in train + held_out + [(val - 170, 99)]:
            rows.append(dict(uid=user, item_id=item, timestamp=timestamp, event_type="listen",
                             played_ratio_pct=10 if item == 99 else 100))
    events_path = root / "events.parquet"
    pd.DataFrame(rows).to_parquet(events_path)
    embeddings = root / "embeddings.parquet"
    pd.DataFrame({"item_id": [*range(1, 14), 98, 99]}).to_parquet(embeddings)
    config.update(embeddings_path=str(embeddings), min_item_interactions=1)
    config["datasets"] = {name: {**entry, "interactions_path": str(events_path)}}
    config_dir = root / "configs"
    config_dir.mkdir()
    (config_dir / "dataset_yambda.yaml").write_text(yaml.safe_dump(config))
    sids = root / "sids"
    SemanticIDStore({str(i): [i, 256 + i, 512 + i, 768 + i] for i in range(1, 14)}).save_json(
        sids / name / "semantic_ids.json")
    return config_dir, sids


def gpt_context(name, *extra):
    args = tuning.parse_args(["--datasets", name, "--device", "cpu", "--num-workers", "0", *extra])
    return tuning.build_context(args, tuning.base.load_regime())


def supervised_tokens(data, *, dense):
    examples = Counter()
    for batch in data.train_dataloader():
        for row, user in enumerate(batch["user_ids"].tolist()):
            if dense:
                tokens = batch["input_token_ids"][row][batch["token_attention_mask"][row]].tolist()
                labels = batch["label_token_ids"][row][batch["token_label_mask"][row]].tolist()
                assert len(labels) == data.sid_len
                # The dense decoder sees the prefix and the first L-1 target
                # tokens under teacher forcing. Its L active labels are the SID.
                assert tokens[-(data.sid_len - 1):] == labels[:-1]
                history = tokens[:-(data.sid_len - 1)]
            else:
                history = batch["history_token_ids"][row][batch["history_token_mask"][row]].tolist()
                labels = batch["target_token_ids"][row][batch["target_token_mask"][row]].tolist()
            examples[user, tuple(history), tuple(labels)] += 1
    return examples


def validation_targets(trainer, data):
    cfg = trainer.cfg
    assert (cfg.val_sample_fraction, cfg.val_sample_seed, cfg.val_sample_resample) == (0.25, 0, False)
    sampled, _ = trainer._maybe_sample_validation_loader(data.val_dataloader(), stage="val")
    return [(user, target, timestamp) for batch in sampled
            for user, target, timestamp in zip(batch["user_ids"].tolist(),
                                               batch["target_item_ids"].tolist(),
                                               batch["target_timestamps"].tolist())]


@pytest.mark.parametrize("name", DATASETS)
def test_global_yambda_matches_tiger_targets_contexts_and_model_inputs(tmp_path, name):
    config_dir, sids = make_inputs(tmp_path, name)
    ctx = gpt_context(name, "--config-dir", str(config_dir), "--semantic-ids-dir", str(sids),
                      "--output-root", str(tmp_path / "gpt"))
    gpt = tuning.base.build_datamodule(name, ctx)
    tiger_args = tiger.build_parser("test TIGER").parse_args([
        "--config-dir", str(config_dir), "--semantic-ids-dir", str(sids), "--device", "cpu",
        "--num-workers", "0"])
    run = tiger.Run(name, tiger_args, search={})
    incumbent = run.incumbent
    reference = incumbent.group.data_module(incumbent.data_cfg)
    reference.setup()
    assert gpt.cfg.train_dense_match_prefix and gpt.cfg.train_window_stride == 1
    assert reference.cfg.train_example_layout == "prefix_target"
    assert gpt.cfg.max_history_len == reference.cfg.max_history_len == 50
    assert (gpt.global_val_cutoff, gpt.global_test_cutoff) == (
        reference.global_val_cutoff, reference.global_test_cutoff)
    assert gpt.item2id == reference.item2id and set(gpt.item2id) == {str(i) for i in range(1, 14)}
    assert gpt.semantic_ids == reference.semantic_ids
    for split in ("train", "val", "test"):
        dense = getattr(gpt, f"{split}_dataset").examples.sort_values(["user_id", "target_position"])
        prefix = getattr(reference, f"{split}_dataset").examples.sort_values(["user_id", "target_position"])
        pd.testing.assert_frame_equal(dense[prefix.columns].reset_index(drop=True),
                                      prefix.reset_index(drop=True), check_dtype=False)

    train = gpt.train_dataset.examples
    assert sorted(train.loc[train.raw_user_id == "1", "target_position"]) == list(range(17, 145))
    assert sorted(train.loc[train.raw_user_id == "2", "target_position"]) == list(range(1, 8))
    assert train.history_item_ids.map(len).max() == 50
    assert train.target_timestamp.max() < gpt.global_val_cutoff
    assert supervised_tokens(gpt, dense=True) == supervised_tokens(reference, dense=False)
    for data, expected_timestamp, target, suffix in (
            (gpt.val_dataset.examples, gpt.global_val_cutoff + 3, "11", ["9", "10"]),
            (gpt.test_dataset.examples, gpt.global_test_cutoff + 1, "13", ["9", "10", "11", "12"])):
        assert len(data) == 8 and data.raw_user_id.nunique() == 8
        assert set(data.target_timestamp) == {expected_timestamp}
        assert set(data.target_raw_item) == {target}
        assert all(history[-len(suffix):] == suffix for history in data.history_raw_items)

    hparams = dict(ctx.regime.baselines[name], embedding_dim=8, num_layers=1, num_heads=2)
    gpt_trainer = tuning.base.build_trainer(
        ctx=ctx, model=tuning.base.build_model(ctx, gpt, hparams, ctx.spec(name), name),
        method=tuning.base.build_method(ctx), hparams=hparams, spec=ctx.spec(name),
        checkpoint_dir=tmp_path / "gpt-trial", logger=None)
    small_model = dict(incumbent.config.model, embedding_dim=8, dim_feedforward=16,
                       num_heads=2, d_kv=4, num_encoder_layers=1, num_decoder_layers=1)
    tiger_trainer = tiger.Trainer.from_optimizer_config(
        model=tiger.tiger.build_model(small_model, reference), method=run.method(incumbent.config, final=False),
        cfg=run.trainer_config(incumbent.config, tmp_path / "tiger-trial", final=False), device="cpu")
    selected = validation_targets(gpt_trainer, gpt)
    assert len(selected) == 2 and selected == validation_targets(tiger_trainer, reference)


@pytest.mark.parametrize("name", DATASETS)
def test_global_yambda_keeps_full_history_separate_and_resumable(tmp_path, name):
    matched = gpt_context(name)
    legacy = gpt_context(name, "--yambda-regime", "full_history")
    assert matched.dataset_dir(name) != legacy.dataset_dir(name)
    assert matched.tuned_config_path(name) != legacy.tuned_config_path(name)
    description = tuning.base.describe_protocol(name, legacy)
    assert description["trainer"]["val_sample_fraction"] == 0.1
    assert description["study_name"] == f"gpt2_{name}_val_generation_ndcg10_exp2_global_time"
    assert tuning.base.describe_protocol(name, matched)["study_name"] != description["study_name"]
    old_data = description["data_config"]
    assert old_data["train_target_policy"] == "all"
    assert not old_data["train_dense_match_prefix"]
    assert old_data["train_window_stride"] is None

    custom_paths = ["--output-root", str(tmp_path / "same-output")]
    matched = gpt_context(name, *custom_paths)
    legacy = gpt_context(name, *custom_paths, "--yambda-regime", "full_history")
    legacy.dataset_dir(name).mkdir(parents=True)
    study = optuna.create_study()
    tuning.base.validate_study_protocol(study, name, legacy)
    trial = study.ask()
    study.tell(trial, 0.25)
    # The archived global-time v1 journal predates this default-false key.
    original_protocol = study.user_attrs["protocol"]
    assert original_protocol["data_config"].pop("train_dense_match_prefix") is False
    study.set_user_attr("protocol", original_protocol)
    tuning.base.validate_study_protocol(study, name, legacy)
    with pytest.raises(ValueError):
        tuning.base.validate_study_protocol(study, name, matched)
    # Optuna will open a *different*, initially empty study for the new name.
    # Its empty user_attrs must never authorize rewriting legacy run files.
    manifest = legacy.dataset_dir(name) / "study_config.json"
    saved = manifest.read_bytes()
    with pytest.raises(ValueError):
        tuning.base.validate_study_protocol(optuna.create_study(), name, matched)
    assert manifest.read_bytes() == saved
