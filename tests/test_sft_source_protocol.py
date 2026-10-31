"""Source manifests validate selected weights and recorded protocols."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/exp8-from-sft/sft_source_protocol.py"
spec = importlib.util.spec_from_file_location("sft_source_protocol", MODULE_PATH)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)


@pytest.fixture
def source_factory(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "source_protocols.json"
    snapshot = {"sources": {}}
    monkeypatch.setattr(protocol, "SNAPSHOT_PATH", snapshot_path)

    def make(source_id="tiger_beauty", seed=42):
        definition = protocol.SOURCE_SPECS[source_id]
        paths = protocol.source_paths(source_id, seed, tmp_path)
        paths["checkpoint"].parent.mkdir(parents=True)
        paths["checkpoint"].write_bytes(b"fixture weights")
        recorded_checkpoint = "/cluster/flowrec/" + paths["checkpoint"].relative_to(tmp_path).as_posix()
        model = {"embedding_dim": 4, "num_heads": 1}
        trainer = {"gradient_accumulation_steps": 8, "max_epochs": 50, "val_sample_fraction": None}
        data = {"semantic_id_path": f"/cluster/flowrec/datasets/{definition.dataset}/sid.json"}
        result = {"dataset": definition.dataset, "seed": seed, "status": "completed",
                  "variant": "sft_finetune" if definition.model == "tiger" else "sft_only",
                  "best_checkpoint": recorded_checkpoint,
                  "validation": {"num_examples": 4}, "test": {"num_examples": 3}}
        if definition.model == "tiger":
            config = {"seed": seed, "method": {"name": "sft"}, "model": model,
                      "data": data, "dataset": {"train_target_policy": "last_n", "train_last_n": 128},
                      "training": trainer, "optimizer": {"lr": 3e-5}, "scheduler": None,
                      "dataloader": {"batch_size": 256, "train_batch_size": 256, "validation_batch_size": 64}}
            resolved = {"dataset": definition.dataset, "seed": seed, "trainer": trainer,
                        "num_train_examples": 30, "item_vocab_size": 6,
                        "finetune_subset": ({"per_user_random_n": 15, "seed": 42, "num_examples": 15}
                                            if definition.protocol == "finetuned_matched_r15" else "full train set")}
            auxiliary = {"resolved": resolved}
        else:
            amazon = definition.protocol == "dense_window_v1"
            positive = ("verified_purchase == True" if amazon else
                        "(event_type in ['listen']) and (event_type != 'listen' or played_ratio_pct >= 50.0)")
            summary = {**data, "positive_criterion": positive, "num_items": 5, "num_users": 2,
                       "num_train_examples": 30, "num_finetune_examples": 30,
                       "num_val_examples": 4, "num_test_examples": 3,
                       "finetune_per_user_last_n": None, "finetune_per_user_random_n": 0,
                       "finetune_subset_seed": 0, "finetune_per_user_pool_last_n": None if amazon else 128}
            if amazon:
                summary["training_context"] = {"context": "dense_window", "last_n": None,
                    "target_policy": "all", "version": 1, "window_len": 20, "window_stride": 20}
                record = {"data": data, "dataset": {"positive_criterion": positive,
                          "train_prefix_from_dense": True, "history_padding": "left"},
                          "model": {"use_user_token": True, "user_ids_count": 1}}
            else:
                record = {"dataset": definition.dataset, "data_config": {
                    **data, "train_dense_match_prefix": True, "train_example_layout": "dense_causal",
                    "train_target_policy": "last_n", "train_last_n": 128, "max_history_len": 50,
                    "train_window_stride": 1, "seed": 42, "user_hash_mod": None,
                    "allowed_event_types": ["listen"], "min_played_ratio_pct": 50.0,
                    "offset_semantic_ids_by_one": False},
                    "resume_identity": {"data_audit": {key: "a" * 64 for key in (
                        "catalog_sha256", "semantic_ids_sha256", "train_targets_sha256",
                        "train_contexts_sha256", "validation_sha256", "test_sha256")}}}
            config = {"dataset": definition.dataset, "seed": seed, "variant": "sft_only",
                      "model": model, "model_profile": {"model": {**model, "dropout": 0.27}},
                      "data": summary, "optimizer": {"lr": 3e-5, "weight_decay": 0.01},
                      "trainer": trainer, "batching": {"batch_size": 32, "gradient_accumulation": 64,
                                                       "effective_batch": 2048},
                      "cli_args": {"prefix_from_dense": amazon, "val_batch_size": 64,
                                   "test_batch_size": 64, "num_workers": 0}}
            auxiliary = {"input_record": record}
        sidecar = {"model_type": definition.model, "method_name": "sft", "model_config": model,
                   "trainer_config": trainer}
        for role, value in (("config", config), ("result", result), ("sidecar", sidecar)):
            paths[role].write_text(json.dumps(value))
        entry = snapshot["sources"].setdefault(source_id, {
            "model": definition.model, "dataset": definition.dataset, "protocol": definition.protocol, "seeds": {}})
        entry["seeds"][str(seed)] = {
            "source_config_sha256": hashlib.sha256(paths["config"].read_bytes()).hexdigest(),
            "records": {role: {"origin": "source.json", "original_sha256": "b" * 64,
                                "content_sha256": protocol._json_hash(value), "content": value}
                        for role, value in auxiliary.items()}}
        snapshot_path.write_text(json.dumps(snapshot))
        (paths["checkpoint"].parent / "provenance.json").write_text(json.dumps({
            "stage": "sft-finetuned", "split": "loo_split", "model": definition.model, "dataset": definition.dataset,
            "seed": seed, "checkpoint": paths["checkpoint"].relative_to(tmp_path).as_posix(),
            **({"regime": paths["checkpoint"].parent.name} if definition.model == "gpt2" and definition.dataset.startswith("yambda_") else {}),
            "sha256": hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest(),
            "metadata_sha256": {paths[role].name: hashlib.sha256(paths[role].read_bytes()).hexdigest()
                                for role in ("config", "result", "sidecar")}}))
        return paths, config, result, sidecar

    return make


def test_default_sources_are_the_nine_intended_available_families():
    ids = protocol.default_source_ids()
    assert len(ids) == len(set(ids)) == 9
    assert "tiger_beauty_2023" not in ids
    assert "tiger_yambda500_matched" not in ids
    assert "tiger_yambda500_matched_r15" in ids
    assert len(protocol.default_source_ids("gpt2")) == 5


def test_tiger_source_is_portable_and_uses_exact_seed(source_factory, tmp_path):
    paths, _, _, _ = source_factory(seed=43)
    assert not paths["resolved"].exists()
    manifest = protocol.resolve_source("tiger_beauty", 43, tmp_path)
    assert manifest["checkpoint"] == str(paths["checkpoint"])
    assert manifest["source_seed"] == 43
    assert manifest["data_config"]["data"]["semantic_id_path"] == str(tmp_path / "datasets/beauty/sid.json")
    assert manifest["batching"]["batch_size"] == 256
    assert manifest["fingerprint_kind"] == "checkpoint_and_metadata"
    assert protocol.available_source_seeds("tiger_beauty", tmp_path) == [43]
    alias = protocol.resolve_source("tiger_beauty", 42, tmp_path)
    assert alias["source_seed"] == 43
    assert alias["source_fingerprint"] == manifest["source_fingerprint"]


def test_primary_weights_are_required(source_factory, tmp_path):
    paths, _, _, _ = source_factory(seed=43)
    paths["checkpoint"].unlink()
    assert protocol.available_source_seeds("tiger_beauty", tmp_path) == []
    with pytest.raises(FileNotFoundError, match="missing exact SFT-only source assets"):
        protocol.resolve_source("tiger_beauty", 43, tmp_path)


def test_data_paths_relocate_with_project_root(tmp_path):
    relative = Path("datasets/beauty/semantic_ids.json")
    config = {"data": {"semantic_id_path": "/cluster/flowrec/" + relative.as_posix()}}
    relocated = protocol._relocate_config(config, tmp_path)
    assert relocated["data"]["semantic_id_path"] == str(tmp_path / relative)
    assert protocol.relocate_path(relative, tmp_path) == str(tmp_path / relative)


def test_tiger_r15_subset_comes_from_frozen_experiment_record(source_factory, tmp_path):
    source_factory("tiger_yambda500_matched_r15")
    manifest = protocol.resolve_source("tiger_yambda500_matched_r15", project_root=tmp_path)
    assert manifest["subset"] == {"train_last_n": 0, "train_random_n": 15,
                                  "train_subset_seed": 42, "train_pool_last_n": None}
    assert manifest["expected_counts"]["num_train_examples"] == 30
    assert manifest["expected_counts"]["num_finetune_examples"] == 15


@pytest.mark.parametrize("source_id", ["gpt2_beauty", "gpt2_yambda50_trial36"])
def test_gpt2_recovers_saved_dropout_and_context_without_current_defaults(source_factory, tmp_path, source_id):
    source_factory(source_id)
    manifest = protocol.resolve_source(source_id, project_root=tmp_path)
    assert manifest["model_profile"]["model"]["dropout"] == 0.27
    assert "dropout" not in manifest["model_config"]
    assert manifest["batching"]["batch_size"] == 32
    dataset = manifest["data_config"]["dataset"]
    assert dataset["train_prefix_from_dense"] is (source_id == "gpt2_beauty")
    if source_id == "gpt2_yambda50_trial36":
        assert dataset["offset_semantic_ids_by_one"] is False
        assert manifest["data_config"]["model"]["user_ids_count"] is None
        assert manifest["subset"]["train_pool_last_n"] == 128
        assert manifest["input_audit"]["train_contexts_sha256"] == "a" * 64


def test_beauty2023_saved_inputs_keep_parent_asin_and_source_window_context(source_factory, tmp_path):
    source_factory("gpt2_beauty_2023")
    snapshot = json.loads(protocol.SNAPSHOT_PATH.read_text())
    record = snapshot["sources"]["gpt2_beauty_2023"]["seeds"]["42"]["records"]["input_record"]
    record["content"] = {"data": {
        "reviews_path": "/cluster/flowrec/data/reviews.parquet",
        "metadata_path": "/cluster/flowrec/data/meta.parquet",
        "semantic_id_path": "/cluster/flowrec/datasets/beauty_2023/sid.json",
        "item_key": "parent_asin", "metadata_item_key": "parent_asin", "max_history_len": 20,
        "positive_criterion": "verified_purchase == True", "train_example_layout": "dense_causal",
        "history_padding": "right", "user_hash_mod": 1}}
    record["content_sha256"] = protocol._json_hash(record["content"])
    protocol.SNAPSHOT_PATH.write_text(json.dumps(snapshot))
    manifest = protocol.resolve_source("gpt2_beauty_2023", project_root=tmp_path)
    dataset = manifest["data_config"]["dataset"]
    assert dataset["item_key"] == dataset["metadata_item_key"] == "parent_asin"
    assert dataset["train_prefix_from_dense"] is True
    assert dataset["history_padding"] == "left"
    assert dataset["train_window_len"] is dataset["train_window_stride"] is None
    assert manifest["preprocessing_origin"] == "saved_pretraining_data_config+dense_window_v1"


@pytest.mark.parametrize("field,value", [("status", "failed"), ("dataset", "wrong"), ("seed", 44),
                                         ("variant", "gflow")])
def test_wrong_source_identity_is_rejected(source_factory, tmp_path, field, value):
    paths, _, result, _ = source_factory()
    result[field] = value
    paths["result"].write_text(json.dumps(result))
    with pytest.raises(ValueError):
        protocol.resolve_source("tiger_beauty", project_root=tmp_path)


def test_changed_source_config_cannot_reuse_frozen_preprocessing(source_factory, tmp_path):
    paths, config, _, _ = source_factory()
    config["dataset"]["train_last_n"] = 8
    paths["config"].write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source config changed"):
        protocol.resolve_source("tiger_beauty", project_root=tmp_path)


def test_checkpoint_replacement_changes_identity_but_inventory_skips_weight_hash(source_factory, tmp_path, monkeypatch):
    paths, _, _, _ = source_factory()
    before = protocol.resolve_source("tiger_beauty", project_root=tmp_path)
    paths["checkpoint"].write_bytes(b"different weights")
    with pytest.raises(ValueError, match="recorded source hash"):
        protocol.resolve_source("tiger_beauty", project_root=tmp_path)
    real_hash = protocol._sha256

    def metadata_hash(path):
        assert path != paths["checkpoint"], "inventory read checkpoint weights"
        return real_hash(path)

    monkeypatch.setattr(protocol, "_sha256", metadata_hash)
    cheap = protocol.resolve_source("tiger_beauty", project_root=tmp_path, hash_checkpoint=False)
    assert cheap["checkpoint_sha256"] is None
    assert cheap["fingerprint_kind"] == "metadata_only"


def test_frozen_input_corruption_is_rejected(source_factory, tmp_path):
    source_factory()
    snapshot = json.loads(protocol.SNAPSHOT_PATH.read_text())
    snapshot["sources"]["tiger_beauty"]["seeds"]["42"]["records"]["resolved"]["content"]["num_train_examples"] = 999
    protocol.SNAPSHOT_PATH.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="corrupt frozen"):
        protocol.resolve_source("tiger_beauty", project_root=tmp_path)


def test_module_import_and_inventory_need_only_standard_library():
    result = subprocess.run([sys.executable, "-S", "-c",
        "import sys; sys.path.insert(0,sys.argv[1]); import sft_source_protocol as p; "
        "print(len(p.default_source_ids())); assert 'torch' not in sys.modules", str(MODULE_PATH.parent)],
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "9"


@pytest.mark.parametrize('source_id', ['tiger_beauty', 'gpt2_yambda50_trial36'])
def test_curated_source_uses_only_current_files_and_frozen_records(source_factory, tmp_path, source_id, monkeypatch):
    paths, _, _, _ = source_factory(source_id, seed=42)
    before = protocol.resolve_source(source_id, 42, tmp_path)
    real_open, real_is_file = Path.open, Path.is_file
    def assert_current(path):
        assert path == protocol.SNAPSHOT_PATH or path.is_relative_to(paths['run_dir'])
    def guarded_open(path, *args, **kwargs):
        assert_current(path)
        return real_open(path, *args, **kwargs)
    def guarded_is_file(path):
        assert_current(path)
        return real_is_file(path)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(Path, 'is_file', guarded_is_file)
    after = protocol.resolve_source(source_id, 43, tmp_path)
    assert after['checkpoint'] == str(paths['checkpoint'])
    assert after['source_fingerprint'] == before['source_fingerprint']
    assert after['source_seed'] == 42


def test_seed_alias_uses_actual_primary_seed_without_changing_protocol(source_factory, tmp_path):
    paths42, _, _, _ = source_factory('tiger_yambda500_matched_r15', seed=42)
    canonical = paths42['checkpoint']
    assert protocol.source_paths('tiger_yambda500_matched_r15', 42, tmp_path)['checkpoint'] == canonical
    assert protocol.source_paths('tiger_yambda500_matched_r15', 43, tmp_path)['checkpoint'] == canonical
    actual = protocol.resolve_source('tiger_yambda500_matched_r15', 43, tmp_path)
    assert actual['source_seed'] == 42
    assert actual['source_fingerprint'] == protocol.resolve_source('tiger_yambda500_matched_r15', 42, tmp_path)['source_fingerprint']


def test_supplied_beauty2023_weights_do_not_fabricate_a_matched_source(tmp_path):
    directory = tmp_path / 'checkpoints/exp2/loo_split/sft-finetuned/tiger/beauty_2023'
    directory.mkdir(parents=True)
    (directory / 'best.pt').write_bytes(b'supplied external checkpoint')
    (directory / 'provenance.json').write_text(json.dumps({
        'stage': 'sft-finetuned', 'split': 'loo_split', 'model': 'tiger', 'dataset': 'beauty_2023', 'seed': 42,
        'checkpoint': 'checkpoints/exp2/loo_split/sft-finetuned/tiger/beauty_2023/best.pt', 'sha256': 'a' * 64}))
    assert protocol.available_source_seeds('tiger_beauty_2023', tmp_path) == [42]
    assert protocol.source_paths('tiger_beauty_2023', 43, tmp_path)['checkpoint'] == directory / 'best.pt'
    with pytest.raises(FileNotFoundError, match='missing exact SFT-only source assets'):
        protocol.resolve_source('tiger_beauty_2023', 42, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("dataset", "wrong"), ("seed", 44), ("split", "global_split"),
    ("checkpoint", "checkpoints/other/best.pt"), ("sha256", "not-a-sha256"),
])
def test_provenance_identity_mismatch_is_rejected(source_factory, tmp_path, field, value):
    paths, _, _, _ = source_factory()
    provenance_path = paths["checkpoint"].parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance[field] = value
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="primary provenance|source seed"):
        protocol.resolve_source("tiger_beauty", project_root=tmp_path)


def test_changed_result_metadata_is_rejected(source_factory, tmp_path):
    paths, _, result, _ = source_factory()
    result["best_checkpoint"] = "different/best.pt"
    paths["result"].write_text(json.dumps(result))
    with pytest.raises(ValueError, match="source result differs from its recorded metadata hash"):
        protocol.resolve_source("tiger_beauty", project_root=tmp_path)
