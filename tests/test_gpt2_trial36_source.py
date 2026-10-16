"""The chosen base checkpoint and its data identities survive fine-tuning."""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "scripts/exp2-tuned-params",
                  ROOT / "scripts/exp4-reward-ablations"):
    sys.path.insert(0, str(directory))

import gflow_finetune_gpt2_amazon as g
import gpt2_matched_finetune as source
from flowrec.data.dense_prefix import dense_prefix_signature
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from tuning_gpt2 import audit_matched_data


@pytest.fixture
def trial(tmp_path, monkeypatch):
    events = tmp_path / "events.parquet"
    pd.DataFrame([{"uid": user, "item_id": i % 11, "timestamp": 1000 + i,
                   "played_ratio_pct": 100}
                  for user, size in enumerate((10, 190)) for i in range(size)]).to_parquet(events)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({str(i): [i + 1, i + 17] for i in range(11)}))
    cfg = YambdaDataConfig(
        interactions_path=events, semantic_id_path=semantic, hf_recent_months=None,
        min_user_interactions=3, min_item_interactions=0, drop_items_missing_embedding=False,
        offset_semantic_ids_by_one=False, user_hash_mod=None,
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_prefix_from_dense=False, train_window_len=50, train_window_stride=1,
        train_target_policy="last_n", train_last_n=128, train_all_prefixes=False,
        max_history_len=50, history_padding="right", num_workers=0, persistent_workers=False,
        include_raw_in_batch=False, batch_size=4, seed=42,
    )
    dense = YambdaDataModule(cfg)
    dense.setup()
    recorded = {"dataset": source.DATASET, "seed": 42,
                "model_config": {**source.EXPECTED_ARCHITECTURE, "codebook_size": 256,
                                 "max_position_embeddings": 205},
                "hparams": {"dropout": 0.10413119232313227},
                "data_config": asdict(cfg), "data_summary": dense.summary(),
                "resume_identity": {"data_audit": audit_matched_data(dense)}}
    paths = source.trial_paths(tmp_path)
    paths["resolved_config"].parent.mkdir(parents=True)
    paths["resolved_config"].write_text(json.dumps(recorded, default=str))
    paths["sidecar"].write_text(json.dumps({"model_config": recorded["model_config"]}))
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    return tmp_path, dense, recorded


def test_explicit_trial_selection_never_falls_back_to_promoted_model(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    args = g.parse_args(["--datasets", source.DATASET, "--yambda50-base", "trial36"])
    assert g.tuned_checkpoint_paths(source.DATASET, args)["checkpoint"] == (
        tmp_path / source.TRIAL_DIRECTORY / "best.pt")
    assert g.tuned_checkpoint_paths(source.DATASET)["checkpoint"].name == "best_model.pt"
    with pytest.raises(FileNotFoundError, match="no fallback"):
        source.load_recorded(tmp_path)
    with pytest.raises(SystemExit):
        g.parse_args(["--datasets", source.DATASET, "--yambda50-base", "trial36",
                      "--yambda50-all-targets", "--sft-only"])


def test_recorded_matched_data_roundtrip_equals_dense_targets_histories_and_catalog(trial):
    root, dense, recorded = trial
    args = g.parse_args(["--datasets", source.DATASET, "--yambda50-base", "trial36",
                        "--seed", "43", "--num-workers", "0"])
    assert not args.prefix_from_dense
    config = g.build_run_config(source.DATASET, args, 1)
    cfg = g.tiger_runner.build_data_config(config, config_path=root / "config.json")
    assert cfg.seed == 42 and args.seed == 43
    assert cfg.user_hash_mod is None and not cfg.offset_semantic_ids_by_one
    assert not cfg.drop_items_missing_embedding and not cfg.train_prefix_from_dense
    assert cfg.train_target_policy == "last_n" and cfg.train_last_n == 128
    dm = YambdaDataModule(cfg)
    dm.setup()
    audit = source.audit_prefix_data(dm, source.load_recorded(root))
    assert audit["all_fingerprints_equal"]
    assert dm.item2id == dense.item2id and dm.semantic_ids == dense.semantic_ids
    assert max(map(len, dm.train_dataset.examples.history_item_ids)) == 50
    for column in ("target_position", "history_item_ids", "target_item_id"):
        assert dm.train_dataset.examples[column].tolist() == dense.train_dataset.examples[column].tolist()
    pd.testing.assert_frame_equal(dm.val_dataset.examples, dense.val_dataset.examples)
    pd.testing.assert_frame_equal(dm.test_dataset.examples, dense.test_dataset.examples)
    signature = dense_prefix_signature(config["dataset"])
    assert signature["context"] == "matched_prefix"
    assert signature["data_audit"]["train_contexts_sha256"] == audit["train_contexts_sha256"]
    cache_config = g.cache_config(config, "interaction")
    assert dense_prefix_signature(cache_config["dataset"]) == signature
    dm.train_dataset.examples.at[0, "history_item_ids"] = [999]
    with pytest.raises(RuntimeError, match="train_contexts_sha256"):
        source.audit_prefix_data(dm, source.load_recorded(root))


def test_profile_uses_trial_dropout_and_checks_checkpoint_metadata(trial):
    root, _, recorded = trial
    args = SimpleNamespace(yambda50_base="trial36")
    profile, config = g.tuned_profile(source.DATASET, args)
    assert profile["model"]["dropout"] == recorded["hparams"]["dropout"]
    assert config["num_layers"] == 4 and config["num_heads"] == 2
    path = source.trial_paths(root)["sidecar"]
    path.write_text(json.dumps({"model_config": {**config, "num_layers": 5}}))
    with pytest.raises(RuntimeError, match="differs"):
        g.tuned_profile(source.DATASET, args)


def test_incomplete_or_wrong_trial_config_fails_closed(trial):
    root, _, recorded = trial
    path = source.trial_paths(root)["resolved_config"]
    recorded["data_config"]["train_last_n"] = 64
    path.write_text(json.dumps(recorded, default=str))
    with pytest.raises(ValueError, match="train_last_n"):
        source.load_recorded(root)
    recorded["data_config"]["train_last_n"] = 128
    del recorded["resume_identity"]["data_audit"]["train_contexts_sha256"]
    path.write_text(json.dumps(recorded, default=str))
    with pytest.raises(ValueError, match="full saved data audit"):
        source.load_recorded(root)


def test_sasrec_translation_verifies_raw_catalog_and_reorders_scores(tmp_path, monkeypatch):
    import hashlib
    from flowrec.models.sasrec import SASRec, SASRecConfig

    runner = g.tiger_runner
    model = SASRec(SASRecConfig(item_vocab_size=5, max_history_len=3, d_model=4,
                               nhead=1, num_layers=1, dropout=0.0))
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model_config": model.config_dict(), "model_state_dict": model.state_dict()}, checkpoint)
    dm = SimpleNamespace(item_vocab_size=3, item2id={"<PAD>": 0, "b": 1, "a": 2},
                         cfg=SimpleNamespace(max_history_len=3))
    mapping = {"target_catalog_sha256": hashlib.sha256(
        json.dumps(sorted(dm.item2id.items())).encode()).hexdigest(),
        "teacher_item_vocab_size": 5, "item_id_translation": [0, 4, 2]}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    spec = {"checkpoint_path": str(checkpoint), "item_id_translation_path": str(path)}
    loaded = runner.build_sasrec_teacher_model(dm, spec, config_path=tmp_path / "config.json").eval()
    batch = {"history_item_ids": torch.tensor([[0, 1, 2]]),
             "history_mask": torch.tensor([[False, True, True]])}
    expected_batch = {**batch, "history_item_ids": torch.tensor([[0, 4, 2]])}
    expected = loaded.scorer.score_items(expected_batch)
    actual = loaded.score_items(batch)
    torch.testing.assert_close(actual, expected[:, [0, 4, 2]])
    dm.item2id = {"<PAD>": 0, "a": 1, "b": 2}
    with pytest.raises(ValueError, match="raw-item translation"):
        runner.build_sasrec_teacher_model(dm, spec, config_path=tmp_path / "config.json")


def test_teacher_source_reconstruction_checks_counts_and_reuses_mapping(tmp_path, monkeypatch):
    from flowrec.data import yambda

    checkpoint = tmp_path / "teacher" / "best_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"source-checkpoint-identity")
    counts = {"name": source.DATASET, "num_items": 3, "num_users": 1, "item_vocab_size": 4}
    (checkpoint.parent / "training_summary.json").write_text(json.dumps({"runs": [{"dataset": counts}]}))
    monkeypatch.setitem(g.DATASETS, source.DATASET,
                        {**g.DATASETS[source.DATASET], "sasrec_checkpoint": str(checkpoint)})
    constructed = []

    class TeacherData:
        def __init__(self, cfg):
            constructed.append(cfg)
            self.item_vocab_size = 4
            self.item2id = {"<PAD>": 0, "a": 1, "extra": 2, "b": 3}

        def setup(self):
            pass

        def summary(self):
            return {key: value for key, value in counts.items() if key != "name"}

    monkeypatch.setattr(yambda, "YambdaDataModule", TeacherData)
    dm = SimpleNamespace(item_vocab_size=3, item2id={"<PAD>": 0, "b": 1, "a": 2})
    args = SimpleNamespace(cache_root=tmp_path / "cache")
    path = g.matched_teacher_translation(source.DATASET, dm, args)
    assert json.loads(path.read_text())["item_id_translation"] == [0, 3, 1]
    assert g.matched_teacher_translation(source.DATASET, dm, args) == path
    assert len(constructed) == 1
    assert constructed[0].train_example_layout == "dense_causal"
    assert constructed[0].semantic_id_path is None
    dm.item2id = {"<PAD>": 0, "missing": 1, "a": 2}
    with pytest.raises(RuntimeError, match="cannot score 1"):
        g.matched_teacher_translation(source.DATASET, dm, args)
