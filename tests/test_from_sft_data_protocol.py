"""Source preprocessing and supervised inputs remain fixed across exp8 arms."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "from_sft_data_runtime_tests", ROOT / "scripts/exp8-from-sft/from_sft_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def source(model="tiger", dataset="instruments"):
    return dict(source_id=f"{model}_{dataset}", model=model, dataset=dataset,
                protocol="dense_window_v1" if model == "gpt2" else "finetuned",
                data_config={"data": {"reviews_path": "/recorded/reviews.parquet",
                                      "semantic_id_path": "/recorded/semantic_ids.json"},
                             "dataset": {"history_padding": "left", "max_history_len": 20,
                                         "offset_semantic_ids_by_one": True,
                                         "train_example_layout": "prefix_target", "use_verified_only": True},
                             "model": {"use_user_token": False, "user_ids_count": 0},
                             "seed": 42, "dataloader": {"pin_memory": True}},
                expected_counts={"num_items": 3, "num_train_examples": 4,
                                 "num_finetune_examples": 2, "num_val_examples": 1,
                                 "num_test_examples": 1},
                subset={"train_random_n": 15, "train_subset_seed": 42},
                input_audit={"context": "dense_window", "window_len": 20},
                preprocessing_origin="source_run_config")


def args():
    return SimpleNamespace(batch_size=2, val_batch_size=1, num_workers=0, smoke=False)


def base():
    return SimpleNamespace(component_spec=lambda name, *args: {"name": name},
                           sasrec_method_cfg=lambda *args: {"checkpoint_path": "/teacher.pt"},
                           GFLOW_REWARD_EPSILON=1e-8, _smoke_examples=lambda args: 1)


@pytest.mark.parametrize("model", ["tiger", "gpt2"])
def test_config_preserves_input_and_catalog_settings_without_mutating_source(model):
    record = source(model)
    if model == "gpt2":
        record["data_config"]["dataset"].update(train_prefix_from_dense=True,
            train_window_len=20, train_window_stride=20, positive_criterion="verified_purchase == True")
    original = deepcopy(record)
    config = runtime.build_config(record, args(), base())
    assert record == original
    assert config["model"] == original["data_config"]["model"]
    assert config["seed"] == 42
    assert config["dataset"]["history_padding"] == "left"
    assert config["dataset"]["max_history_len"] == 20
    assert config["dataset"]["offset_semantic_ids_by_one"] is True
    assert config["dataloader"]["train_drop_last"] is False
    if model == "gpt2":
        assert config["dataset"] == original["data_config"]["dataset"]
        assert config["method"]["component_keys"] == ["sasrec", "interaction", "sim_hier"]
    else:
        assert config["dataset"]["use_verified_only"] is False
        assert config["dataset"]["positive_criterion"] == "verified_purchase == True"


def test_yambda_reexpresses_source_filter_without_changing_collator_or_catalog():
    record = source(dataset="yambda_50m_listens")
    record["data_config"]["dataset"] = dict(allowed_event_types=["listen"], min_played_ratio_pct=50.,
        max_history_len=50, history_padding="left", train_target_policy="last_n", train_last_n=128,
        offset_semantic_ids_by_one=False)
    config = runtime.build_config(record, args(), base())
    assert config["dataset"] == {**record["data_config"]["dataset"], "allowed_event_types": None,
        "min_played_ratio_pct": None, "positive_criterion": "event_type == 'listen' and played_ratio_pct >= 50"}
    assert config["model"] == {"use_user_token": False, "user_ids_count": 0}


@pytest.mark.parametrize("change", [{"allowed_event_types": ["like"]},
                                     {"min_played_ratio_pct": 80},
                                     {"positive_criterion": "timestamp > 123"}])
def test_yambda_unknown_positive_stream_fails_instead_of_overwriting_it(change):
    record = source(dataset="yambda_50m_listens")
    record["data_config"]["dataset"].update(allowed_event_types=["listen"], min_played_ratio_pct=50.)
    record["data_config"]["dataset"].update(change)
    with pytest.raises(ValueError, match="positive"):
        runtime.build_config(record, args(), base())


def session(tmp_path, record=None):
    record = source() if record is None else record
    b, a = base(), args()
    obj = runtime.SourceSession(record, SimpleNamespace(), b, a,
        runtime.build_config(record, a, b), tmp_path, tmp_path, tmp_path / "best.pt",
        {"source_id": record["source_id"]}, [])
    obj._original_indices = lambda dataset, args: [1, 3]
    return obj


def dm_fixture():
    rows = pd.DataFrame({"raw_user_id": ["u"], "user_id": [1], "history_item_ids": [[1, 2]],
                         "history_raw_items": [["a", "b"]], "target_item_id": [3],
                         "target_raw_item": ["c"], "target_timestamp": [100], "target_position": [2],
                         "target_rating": [5.], "target_verified_purchase": [True],
                         "history_ratings": [[4., 5.]], "history_verified_purchase": [[True, True]]})
    return SimpleNamespace(item2id={"a": 1, "b": 2, "c": 3}, user2id={"u": 1},
        semantic_ids={"a": [1], "b": [2], "c": [3]}, setup=lambda: None,
        **{name: SimpleNamespace(examples=rows.assign(split=split)) for name, split in (
            ("train_dataset", "train"), ("val_dataset", "val"), ("test_dataset", "test"))})


def attach_reference(obj, original):
    obj.base.tiger_runner = SimpleNamespace(build_data_config=lambda config, **kwargs: config,
                                            build_data_module=lambda config: original)


def test_tiger_comparison_ignores_only_annotations_not_consumed_by_model(tmp_path):
    obj, original, actual = session(tmp_path), dm_fixture(), dm_fixture()
    attach_reference(obj, original)
    for name in ("train_dataset", "val_dataset", "test_dataset"):
        frame = getattr(actual, name).examples
        frame["target_rating"] = 0.
        frame["target_verified_purchase"] = False
        frame["history_ratings"] = [[0., 0.]]
        frame["history_verified_purchase"] = [[False, False]]
    result = runtime.compare_tiger_source(obj, actual)
    assert "all supervised rows" in result["comparison"]
    assert set(result["auxiliary_columns_not_consumed_by_tiger"]) == set(runtime.AUXILIARY_COLUMNS)


@pytest.mark.parametrize("split,column,value", [("train_dataset", "history_item_ids", [2, 1]),
    ("val_dataset", "target_timestamp", 101), ("test_dataset", "target_item_id", 2)])
def test_tiger_comparison_rejects_supervised_input_or_target_drift(tmp_path, split, column, value):
    obj, original, actual = session(tmp_path), dm_fixture(), dm_fixture()
    attach_reference(obj, original)
    getattr(actual, split).examples.at[0, column] = value
    with pytest.raises(AssertionError):
        runtime.compare_tiger_source(obj, actual)


@pytest.mark.parametrize("section,key,value", [("dataset", "history_padding", "right"),
    ("dataset", "max_history_len", 10), ("dataset", "offset_semantic_ids_by_one", False),
    ("model", "use_user_token", True), ("data", "semantic_id_path", "/another.json")])
def test_equal_rows_do_not_hide_changed_tokenization_or_data_settings(tmp_path, section, key, value):
    obj, original = session(tmp_path), dm_fixture()
    attach_reference(obj, original)
    obj.config[section][key] = value
    with pytest.raises(RuntimeError, match="configuration changed"):
        runtime.compare_tiger_source(obj, deepcopy(original))


@pytest.mark.parametrize("mapping", ["item2id", "user2id", "semantic_ids"])
def test_tiger_comparison_rejects_changed_catalog_order_or_semantic_ids(tmp_path, mapping):
    obj, original, actual = session(tmp_path), dm_fixture(), dm_fixture()
    attach_reference(obj, original)
    setattr(actual, mapping, {})
    with pytest.raises(RuntimeError, match=mapping):
        runtime.compare_tiger_source(obj, actual)


def audit_dm(record):
    counts = {key: value for key, value in record["expected_counts"].items()
              if key != "num_finetune_examples"}
    return SimpleNamespace(train_dataset=list(range(4)),
        summary=lambda: {**counts, "training_context": deepcopy(record["input_audit"])},
        dense_prefix_audit={"validation": {"equal": True}, "test": {"equal": True}})


def test_smoke_audit_checks_full_source_subset_not_smoke_prefix(tmp_path):
    obj = session(tmp_path)
    obj.args.smoke = True
    dm = audit_dm(obj.source)
    assert obj.indices(dm) == [1]
    assert obj.audit_data(dm)["counts"]["num_finetune_examples"] == 2


def test_audit_rejects_changed_source_counts_and_dense_context(tmp_path):
    obj = session(tmp_path, source("gpt2"))
    dm = audit_dm(obj.source)
    assert obj.audit_data(dm)["dense_prefix_audit"] == dm.dense_prefix_audit
    dm.summary = lambda: {**obj.source["expected_counts"], "num_items": 4}
    with pytest.raises(RuntimeError, match="num_items=3"):
        obj.audit_data(dm)
    dm.summary = lambda: {**obj.source["expected_counts"], "training_context": {"window_len": 19}}
    with pytest.raises(RuntimeError, match="dense-window context changed"):
        obj.audit_data(dm)


def test_matched_source_audits_full_recorded_fingerprints(tmp_path):
    record = source("gpt2", "yambda_50m_listens")
    record["protocol"] = "matched_trial36"
    record["input_audit"] = {"train_contexts_sha256": "saved-history-fingerprint"}
    obj, calls = session(tmp_path, record), []
    def check(dm, recorded):
        calls.append(recorded)
        return {"all_fingerprints_equal": True}
    obj.base.matched_source = SimpleNamespace(audit_prefix_data=check)
    result = obj.audit_data(audit_dm(record))
    assert calls == [{"dataset": record["dataset"], "matched_data_audit": record["input_audit"]}]
    assert result["matched_prefix_audit"]["all_fingerprints_equal"] is True
