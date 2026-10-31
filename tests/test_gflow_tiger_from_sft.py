"""GFlow-after-SFT runs use the requested grid and the correct source weights."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts",
              PROJECT_ROOT / "scripts/exp2-tuned-params",
              PROJECT_ROOT / "scripts/exp4-reward-ablations"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gflow_finetune_tiger_from_sft as runner  # noqa: E402

base = runner.base


@pytest.fixture(autouse=True)
def restore_active_dataset():
    """The underlying legacy runner selects a dataset using module globals."""
    keys = {key for spec in base.DATASETS.values() for key in spec} | {"DATASET"}
    before = {key: getattr(base, key) for key in keys}
    specs = dict(base.DATASETS)
    yield
    base.DATASETS.clear()
    base.DATASETS.update(specs)
    for key, value in before.items():
        setattr(base, key, value)


@pytest.fixture
def make_source(tmp_path, monkeypatch):
    """Only metadata is needed; no dataset or torch checkpoint is loaded."""
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    # Full fingerprint/provenance validation is covered by test_sft_source_protocol;
    # these tests isolate this wrapper's model, catalog and data checks.
    monkeypatch.setattr(runner.sft_sources, "resolve_source", lambda *_a, **_k: {})

    def make(dataset="beauty", seed=42):
        path = runner.source_checkpoint(dataset, seed)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test checkpoint placeholder")
        canonical = base.load_json(base.abspath(base.DATASETS[dataset]["SFT_CONFIG"]))
        config = copy.deepcopy(canonical)
        config["method"] = {"name": "sft"}
        config["seed"] = seed
        config["optimizer"] = {"lr": 3e-5}
        config["scheduler"] = None
        config["data"]["semantic_id_path"] = "/another/host/flowrec/" + canonical["data"]["semantic_id_path"]
        for key, value in base.DATASETS[dataset]["DATA_OVERRIDES"].items():
            config["data"][key] = "/another/host/flowrec/" + value
        config_path = path.parent / "run_config.json"
        config_path.write_text(json.dumps(config))
        return SimpleNamespace(dataset=dataset, sft_seed=seed), path, config_path, config

    return make


@pytest.mark.parametrize("dataset", runner.DATASETS)
def test_requested_grid_is_a_complete_cartesian_product(dataset):
    args = runner.parse_args(["--dataset", dataset])
    grid = base.build_grid(args)
    actual = {(base.combo_key(cell["combo"]), cell["power"], cell["objective"], cell["rho"])
              for cell in grid}
    expected = set(itertools.product(
        ("sasrec", "interaction", "similarity", "sasrec_x_interaction_x_similarity"),
        (0.25, 1.0, 4.0, 16.0), ("tb", "db", "subtb"), (1.0, 0.1)))
    assert len(grid) == len(actual) == 96
    assert actual == expected
    # rho drives gradient calibration; it is not a constant loss multiplier.
    assert args.gflow_weight_mode == "fixed_grad_ratio"
    assert {cell["num_negatives"] for cell in grid} == {10}
    assert {cell["sampler"] for cell in grid} == {"teacher"}
    assert args.validate_at_start is True
    assert args.train_last_n == args.train_random_n == 0
    assert (args.batch_size, args.grad_accum) == (32, 64)


@pytest.mark.parametrize("dataset", runner.DATASETS)
def test_preprocessing_and_loaders_equal_existing_amazon_gflow(dataset):
    """Changing the checkpoint must not change how existing GFlow sees data."""
    with runner.sft_source(dataset, 42):
        new_args = runner.parse_args(["--dataset", dataset])
        new_config = base.build_run_config(new_args)
    existing_args = runner.matched.parse_args(["--dataset", dataset])
    existing_config = base.build_run_config(existing_args)
    for section in ("data", "dataset", "dataloader"):
        assert new_config[section] == existing_config[section], section
    assert new_args.train_last_n == existing_args.train_last_n == 0
    assert new_args.train_random_n == existing_args.train_random_n == 0


@pytest.mark.parametrize("dataset", runner.DATASETS)
def test_base_tiger_preprocessing_changes_only_verified_rule_representation(dataset):
    """The auxiliary reward store must not change supervised example rules."""
    spec = base.control.DATASETS[dataset]
    canonical = base.load_json(base.abspath(spec["sft_config"]))
    canonical["data"].update(spec["data_overrides"])
    for key, value in canonical["data"].items():
        if key.endswith("_path") and value:
            canonical["data"][key] = str(base.abspath(value))
    with runner.sft_source(dataset, 42):
        args = runner.parse_args(["--dataset", dataset])
        config = base.build_run_config(args)
    assert config["data"] == canonical["data"]
    assert config["model"] == canonical["model"]
    different = {key for key in set(config["dataset"]) | set(canonical["dataset"])
                 if config["dataset"].get(key) != canonical["dataset"].get(key)}
    assert different <= {"positive_criterion", "use_verified_only"}
    assert config["dataset"]["positive_criterion"] == "verified_purchase == True"
    assert config["dataset"]["use_verified_only"] is False
    assert config["dataset"]["train_target_policy"] == "all"
    assert config["dataset"]["train_all_prefixes"] is True


@pytest.mark.parametrize("dataset", runner.DATASETS)
def test_source_override_survives_repeated_base_parsing_and_restores_state(dataset):
    base.activate_dataset("yambda_500m_listens")
    old_spec = base.DATASETS[dataset]
    before = {key: getattr(base, key) for key in old_spec}
    before["DATASET"] = base.DATASET
    expected = str(runner.source_checkpoint(dataset, 43))
    with pytest.raises(RuntimeError, match="interrupt"):
        with runner.sft_source(dataset, 43):
            for _ in range(2):
                base.parse_args(["--dataset", dataset])
                assert base.TIGER_CHECKPOINT == expected
                assert base.DATASETS[dataset]["TIGER_CHECKPOINT"] == expected
                assert base.checkpoint_matches("/cluster/flowrec/" + Path(expected).relative_to(runner.PROJECT_ROOT).as_posix())
                assert not base.checkpoint_matches(old_spec["TIGER_CHECKPOINT"])
            raise RuntimeError("interrupt")
    assert base.DATASETS[dataset] is old_spec
    assert {key: getattr(base, key) for key in before} == before


def test_missing_source_never_falls_back_to_initial(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(base, "main", lambda argv: calls.append(argv))
    with pytest.raises(FileNotFoundError, match="no fallback to initial"):
        runner.main(["--dataset", "beauty", "--index", "0"])
    assert calls == []


@pytest.mark.parametrize("dataset", runner.DATASETS)
def test_valid_source_accepts_relocated_catalog_paths(make_source, dataset):
    options, checkpoint, config_path, _ = make_source(dataset, seed=43)
    with runner.sft_source(dataset, 43):
        source = runner.validate_source(options)
    assert source["checkpoint"] == str(checkpoint)
    assert source["run_config"] == str(config_path)
    assert source["sft_seed"] == 43
    assert source["source_optimizer"]["lr"] == 3e-5
    assert source["source_scheduler"] is None


@pytest.mark.parametrize("field,value,message", [
    ("method", {"name": "gflow"}, "not an SFT-only"),
    ("seed", 99, "source seed"),
    ("model", {}, "model differs"),
    ("dataset", {}, "dataset differs"),
    ("data", {"semantic_id_path": "datasets/another/table.json"}, "semantic ID table"),
])
def test_source_validation_rejects_incompatible_metadata(make_source, field, value, message):
    options, _, config_path, config = make_source()
    config[field] = value
    config_path.write_text(json.dumps(config))
    with runner.sft_source(options.dataset, options.sft_seed):
        with pytest.raises(ValueError, match=message):
            runner.validate_source(options)


@pytest.mark.parametrize("field", ["reviews_path", "metadata_path"])
def test_source_validation_rejects_different_raw_data(make_source, field):
    options, _, config_path, config = make_source()
    config["data"][field] = "data/raw/different_dataset.parquet"
    config_path.write_text(json.dumps(config))
    with runner.sft_source(options.dataset, options.sft_seed):
        with pytest.raises(ValueError, match=field + " differs"):
            runner.validate_source(options)


@pytest.mark.parametrize("flags,message", [
    (["--train-last-n", "8"], "all base TIGER train prefixes"),
    (["--train-random-n", "15"], "all base TIGER train prefixes"),
    (["--val-sample-fraction", "0.5"], "full base TIGER validation"),
    (["--max-train-batches", "2"], "only allowed with --smoke"),
    (["--max-eval-batches", "2"], "only allowed with --smoke"),
])
def test_protocol_rejects_partial_training_or_evaluation(flags, message):
    args = runner.parse_args(["--dataset", "beauty", *flags])
    with pytest.raises(ValueError, match=message):
        runner.check_full_data_protocol(args)


def test_protocol_accepts_full_data_and_isolated_smoke_batches():
    runner.check_full_data_protocol(runner.parse_args(["--dataset", "beauty"]))
    runner.check_full_data_protocol(runner.parse_args(
        ["--dataset", "beauty", "--smoke", "--max-train-batches", "2", "--max-eval-batches", "2"]))


def test_prepared_baseline_must_evaluate_sft_only_checkpoint(tmp_path):
    args = runner.parse_args(["--dataset", "beauty", "--index", "0",
                              "--output-root", str(tmp_path / "runs"),
                              "--cache-root", str(tmp_path / "factors")])
    # Entry zero needs only the SASRec factor cache. Exercise the real
    # manifest checks while keeping factors and data unnecessary here.
    manifest = base.component_cache_dir("sasrec", args) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"kind": base.CACHE_KIND,
                                    "component": {"name": "sasrec"},
                                    **base.subset_spec(args)}))
    path = args.output_root / "baseline/init_metrics.json"
    path.parent.mkdir(parents=True)
    initial = base.DATASETS["beauty"]["TIGER_CHECKPOINT"]
    with runner.sft_source("beauty", 42):
        for payload in (
            {"checkpoint": initial, "validation": {}, "test": {}},
            {"checkpoint": base.TIGER_CHECKPOINT, "validation": {}},
        ):
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="not a complete baseline"):
                runner.require_prepared(args)
        path.write_text(json.dumps({"checkpoint": base.TIGER_CHECKPOINT,
                                    "validation": {}, "test": {}}))
        with pytest.raises(FileNotFoundError, match="data_protocol.json"):
            runner.require_prepared(args)
        config = base.build_run_config(args)
        audit_path = args.output_root / "data_protocol.json"
        audit = {"catalog_and_supervised_examples_equal": True,
                 "data": config["data"], "dataset": config["dataset"]}
        audit_path.write_text(json.dumps(audit))
        runner.require_prepared(args)
        audit["dataset"]["train_target_policy"] = "last"
        audit_path.write_text(json.dumps(audit))
        with pytest.raises(ValueError, match="data protocol changed"):
            runner.require_prepared(args)


@pytest.fixture
def toy_data_audit(monkeypatch):
    frame = pd.DataFrame({
        "raw_user_id": ["u", "u"], "user_id": [1, 1],
        "history_raw_items": [["a"], ["a", "b"]], "history_item_ids": [[1], [1, 2]],
        "target_raw_item": ["b", "a"], "target_item_id": [2, 1],
        "target_timestamp": [10, 20], "target_position": [1, 2],
        "target_rating": [5.0, 4.0], "target_verified_purchase": [True, True],
        "history_ratings": [[4.0], [4.0, 5.0]], "history_verified_purchase": [[True], [True, True]],
    })
    original = SimpleNamespace(item2id={"a": 1, "b": 2}, user2id={"u": 1},
                               semantic_ids={"a": [1, 257], "b": [2, 258]}, setup=lambda: None)
    for attribute, split in (("train_dataset", "train"), ("val_dataset", "val"), ("test_dataset", "test")):
        setattr(original, attribute, SimpleNamespace(examples=frame.assign(split=split)))
    actual = copy.deepcopy(original)
    reference_configs = []

    def build_reference(config, *, config_path):
        reference_configs.append(config)
        return config

    monkeypatch.setattr(base.tiger_runner, "build_data_config", build_reference)
    monkeypatch.setattr(base.tiger_runner, "build_data_module", lambda config: original)
    return original, actual, reference_configs


def test_runtime_audit_ignores_only_unused_auxiliary_annotations(toy_data_audit, tmp_path):
    _, actual, references = toy_data_audit
    for attribute in ("train_dataset", "val_dataset", "test_dataset"):
        examples = getattr(actual, attribute).examples
        examples["target_rating"] = 0.0
        examples["target_verified_purchase"] = False
        examples["history_ratings"] = [[0.0], [0.0, 0.0]]
        examples["history_verified_purchase"] = [[False], [False, False]]
    with runner.sft_source("beauty", 42):
        config = base.build_run_config(runner.parse_args(["--dataset", "beauty"]))
        result = runner.verify_base_examples(config, tmp_path / "config.json", actual)
        canonical = base.load_json(base.abspath(base.SFT_CONFIG))
    assert result["catalog_and_supervised_examples_equal"] is True
    assert result["examples"] == {"train": 2, "validation": 2, "test": 2}
    assert references[0]["dataset"] == canonical["dataset"]


@pytest.mark.parametrize("field", ["item2id", "user2id", "semantic_ids",
                                  "train_dataset", "val_dataset", "test_dataset"])
def test_runtime_audit_rejects_catalog_or_supervised_example_drift(toy_data_audit, tmp_path, field):
    _, actual, _ = toy_data_audit
    if field.endswith("_dataset"):
        getattr(actual, field).examples.loc[0, "target_item_id"] = 999
    else:
        setattr(actual, field, {"wrong": 999})
    with runner.sft_source("beauty", 42):
        config = base.build_run_config(runner.parse_args(["--dataset", "beauty"]))
        with pytest.raises(ValueError, match="relative to base TIGER|examples differ from base TIGER"):
            runner.verify_base_examples(config, tmp_path / "config.json", actual)


def test_unavailable_source_seeds_keep_requested_labels_until_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    first = runner.parse_args(["--dataset", "beauty", "--sft-seed", "42", "--seed", "44"])
    second = runner.parse_args(["--dataset", "beauty", "--sft-seed", "43", "--seed", "44"])
    assert first.seed == second.seed == 44
    assert first.output_root != second.output_root
    assert first.metrics_csv_name != second.metrics_csv_name
    assert first.cache_root == second.cache_root
    assert first.control_init_baseline == Path(".")


def test_primary_seed_aliases_use_actual_seed_and_share_outputs(monkeypatch):
    monkeypatch.setattr(runner.sft_sources, "primary_source_seed", lambda *_args: 42)
    first = runner.parse_args(["--dataset", "beauty", "--sft-seed", "42", "--seed", "44"])
    second = runner.parse_args(["--dataset", "beauty", "--sft-seed", "43", "--seed", "44"])
    assert first.sft_seed == second.sft_seed == 42
    assert first.seed == second.seed == 44
    assert first.output_root == second.output_root
    assert first.metrics_csv_name == second.metrics_csv_name


def test_smoke_cannot_reuse_full_experiment_outputs_or_factors():
    full = runner.parse_args(["--dataset", "beauty"])
    smoke = runner.parse_args(["--dataset", "beauty", "--smoke"])
    assert full.output_root != smoke.output_root
    assert full.cache_root != smoke.cache_root
    assert full.metrics_csv_name != smoke.metrics_csv_name


def test_smoke_isolates_explicit_output_and_cache_overrides(tmp_path):
    overrides = ["--dataset", "beauty", "--output-root", str(tmp_path / "runs"),
                 "--cache-root", str(tmp_path / "factors"),
                 "--metrics-csv-name", "custom.csv"]
    full = runner.parse_args(overrides)
    smoke = runner.parse_args([*overrides, "--smoke"])
    assert smoke.output_root == full.output_root / "smoke"
    assert smoke.cache_root == full.cache_root / "smoke"
    assert smoke.metrics_csv_name == "custom_smoke.csv"


@pytest.mark.parametrize("status", [None, "failed", "completed"])
def test_wrapper_propagates_caught_training_failures(make_source, tmp_path, monkeypatch, status):
    _, checkpoint, _, _ = make_source()
    calls = []

    def fake_training(argv):
        # The base runner parses a second time before loading the weights.
        args = base.parse_args(argv)
        calls.append(base.TIGER_CHECKPOINT)
        if status is not None:
            entry = base.select_grid(args)[0]
            path = base.run_dir_for(base.abspath(args.output_root), entry, args.seed) / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": status}))

    monkeypatch.setattr(base, "main", fake_training)
    argv = ["--dataset", "beauty", "--index", "0", "--output-root", str(tmp_path / "runs")]
    if status == "completed":
        runner.main(argv)
    else:
        with pytest.raises(RuntimeError, match="GFlow cells did not complete"):
            runner.main(argv)
    assert calls == [str(checkpoint)]
    source = json.loads((tmp_path / "runs/sources/seed_42_index_0.json").read_text())
    assert source["checkpoint"] == str(checkpoint)
    assert source["gflow_seed"] == 42


def test_failed_forced_rerun_is_not_hidden_by_old_completed_result(make_source, tmp_path, monkeypatch):
    make_source()
    argv = ["--dataset", "beauty", "--index", "0", "--force",
            "--output-root", str(tmp_path / "runs")]
    args = runner.parse_args(argv)
    entry = base.select_grid(args)[0]
    result = base.run_dir_for(args.output_root, entry, args.seed) / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"status": "completed"}))
    caught = []

    def fail_training(**kwargs):
        raise RuntimeError("training failed after old completed result")

    def catch_failure_like_legacy_main(argv):
        try:
            base.run_single(entry=entry)
        except RuntimeError as exc:
            caught.append(str(exc))

    monkeypatch.setattr(base, "run_single", fail_training)
    monkeypatch.setattr(base, "main", catch_failure_like_legacy_main)
    with pytest.raises(RuntimeError, match="GFlow cells did not complete"):
        runner.main(argv)
    assert caught == ["training failed after old completed result"]
    assert base.run_single is fail_training
    assert json.loads(result.read_text())["status"] == "completed"


@pytest.mark.parametrize("smoke", [False, True])
def test_training_indices_respect_smoke_cache_and_restore_shared_helpers(make_source, tmp_path, monkeypatch, smoke):
    make_source()
    before_write = base.write_json

    def all_indices(dataset, args):
        return list(range(5000))

    observed = []

    def fake_training(argv):
        args = base.parse_args(argv)
        indices = base.finetune_indices(None, args)
        expected = base._smoke_examples(args) if smoke else 5000
        assert indices == list(range(expected))
        entry = base.select_grid(args)[0]
        output = base.abspath(args.output_root)
        base.write_json(base.run_dir_for(output, entry, args.seed) / "result.json", {"status": "completed"})
        base.write_json(output / "summary.json", {"runs": []})
        observed.append(output)

    monkeypatch.setattr(base, "finetune_indices", all_indices)
    monkeypatch.setattr(base, "main", fake_training)
    argv = ["--dataset", "beauty", "--index", "0", "--output-root", str(tmp_path / "runs")]
    runner.main([*argv, "--smoke"] if smoke else argv)
    assert base.finetune_indices is all_indices
    assert base.write_json is before_write
    assert len(observed) == 1
    assert (observed[0] / "summaries/seed_42_index_0.json").is_file()
    assert not (observed[0] / "summary.json").exists()
