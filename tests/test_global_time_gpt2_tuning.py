"""Global-time GPT-2 catalog parity and real search/resume/rerank coverage."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import optuna
import pytest
import torch
import yaml

from flowrec.item_representation.base import SemanticIDStore
from test_global_time_representation import _assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/global-time/tuning_gpt2.py"
SPEC = importlib.util.spec_from_file_location("global_gpt2_tuning", SCRIPT_PATH)
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)


def _inputs(root, name):
    family = "yambda" if name.startswith("yambda") else "amazon"
    config_dir = _assets(root, name, family)
    path = config_dir / tuning.base.TUNED_CONFIG_DATA_YAML[name]
    config = yaml.safe_load(path.read_text())
    config["gpt2"].update(batch_size=2, eval_batch_size=2, num_workers=0,
                          persistent_workers=False, pin_memory=False)
    if name == "yambda_500m_listens":
        config["datasets"][name]["dataset_size"] = "500m"
    path.write_text(yaml.safe_dump(config))
    sid_dir = root / "sids"
    # Only the final training catalog; cold item 3 must remain in the
    # pre-split stream so a cold last target is dropped, never substituted.
    SemanticIDStore({"1": [0, 256, 512, 768], "2": [1, 257, 513, 769]}).save_json(
        sid_dir / name / "semantic_ids.json"
    )
    return config_dir, sid_dir


def _context(root, config_dir, sid_dir):
    args = tuning.parse_args([
        "--config-dir", str(config_dir), "--semantic-ids-dir", str(sid_dir),
        "--output-root", str(root / "out"), "--device", "cpu", "--num-workers", "0",
    ])
    return tuning.build_context(args, tuning.base.load_regime())


def test_defaults_reuse_search_with_global_time_paths_and_all_five_datasets():
    args = tuning.parse_args([])
    regime = tuning.base.load_regime()
    ctx = tuning.build_context(args, regime)
    assert args.datasets == ["beauty", "instruments", "beauty_2023", "yambda_50m_listens", "yambda_500m_listens"]
    assert [ctx.n_trials(tuning.base.TUNING_SPECS[n]) for n in args.datasets] == [100, 100, 60, 40, 20]
    for path in (ctx.config_dir, ctx.config_out_dir, ctx.output_root):
        assert "global-time" in path.parts
    assert ctx.semantic_ids_dir == PROJECT_ROOT / "datasets/semantic_ids/exp2-tuned-params"
    assert ctx.regime == regime
    tuning.base.validate_specs(regime)
    for name in args.datasets:
        _, cfg = tuning.base.resolve_data_config(name, ctx)
        assert cfg.split_strategy == "global_time"
        assert cfg.semantic_id_path == ctx.semantic_ids_dir / name / "semantic_ids.json"
        assert cfg.train_example_layout == "dense_causal" and cfg.history_padding == "right"
        assert cfg.global_val_timestamp < cfg.global_test_timestamp
    assert tuning.base.study_name_for("beauty", split_strategy=ctx.required_split_strategy).endswith("_global_time")
    legacy = tuning.base.build_context(tuning.base.parse_args([]), regime)
    assert legacy.config_dir == tuning.base.CONFIG_DIR and legacy.required_split_strategy is None
    assert not tuning.base.study_name_for("beauty").endswith("_global_time")


@pytest.mark.parametrize("name", tuning.DATASETS)
@pytest.mark.parametrize("include_cold_sids", [False, True])
def test_all_five_catalogs_and_targets_match_sid_free_global_split(tmp_path, name, include_cold_sids):
    config_dir, sid_dir = _inputs(tmp_path, name)
    if include_cold_sids:
        # Reused exp2 dictionaries may include items first observed after
        # the cutoff. They must not enlarge the global-time training catalog.
        SemanticIDStore({"1": [0, 256, 512, 768], "2": [1, 257, 513, 769],
                         "3": [2, 258, 514, 770]}).save_json(sid_dir / name / "semantic_ids.json")
    ctx = _context(tmp_path, config_dir, sid_dir)
    data = tuning.base.build_datamodule(name, ctx)
    group, cfg = tuning.base.resolve_data_config(name, ctx)
    reference = group.data_module(replace(cfg, semantic_id_path=None))
    reference.setup()
    assert data.item2id == reference.item2id
    assert data.item2id.keys() == data.semantic_ids.keys() == {"1", "2"}
    assert data.post_split_filter_stats.num_items_removed == 1
    assert len(data.val_dataset) == len(data.test_dataset) == 1
    for split in ("train", "val", "test"):
        assert getattr(data, f"{split}_dataset").examples.equals(getattr(reference, f"{split}_dataset").examples)
    method = tuning.base.build_method(ctx)
    assert method.cfg.generation_num_beams == 20 and method.cfg.generation_topk == (10,)
    assert not method.cfg.generation_exclude_seen_items


@pytest.mark.parametrize("changes", [
    {"split_strategy": "leave_one_out"},
    {"global_val_timestamp": None},
    {"global_test_timestamp": 1},
    {"global_history_policy": "train_only"},
    {"global_eval_target_mode": "all"},
    {"filter_cold_items_after_split": False},
    {"cold_item_train_source": "targets"},
    {"drop_items_missing_semantic_id": True},
])
def test_incompatible_split_settings_are_rejected(changes):
    ctx = tuning.build_context(tuning.parse_args([]), tuning.base.load_regime())
    _, cfg = tuning.base.resolve_data_config("beauty", ctx)
    with pytest.raises(ValueError, match="requires"):
        tuning.base.validate_data_config("beauty", replace(cfg, **changes), ctx)


def test_incomplete_training_sids_fail_without_changing_the_catalog(tmp_path):
    config_dir, sid_dir = _inputs(tmp_path, "beauty")
    SemanticIDStore({"1": [0, 256, 512, 768]}).save_json(sid_dir / "beauty/semantic_ids.json")
    with pytest.raises(ValueError, match="semantic IDs miss 1 training items"):
        tuning.base.build_datamodule("beauty", _context(tmp_path, config_dir, sid_dir))


def test_protocol_rejects_changed_cutoffs_and_training_settings_on_resume(tmp_path):
    config_dir, sid_dir = _inputs(tmp_path, "beauty")
    ctx = _context(tmp_path, config_dir, sid_dir)
    ctx.dataset_dir("beauty").mkdir(parents=True)
    study = optuna.create_study()
    tuning.base.validate_study_protocol(study, "beauty", ctx)
    tuning.base.validate_study_protocol(study, "beauty", replace(ctx, num_workers=2, n_trials_override=200))
    with pytest.raises(ValueError, match="different split/training protocol"):
        tuning.base.validate_study_protocol(study, "beauty", replace(ctx, debug_max_batches=1))
    path = config_dir / "dataset_amazon.yaml"
    config = yaml.safe_load(path.read_text())
    config["datasets"]["beauty"]["global_val_timestamp"] = 9
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="different split/training protocol"):
        tuning.base.validate_study_protocol(study, "beauty", ctx)


def test_missing_sid_launch_can_resume_with_corrected_path_before_trials_start(tmp_path):
    config_dir, sid_dir = _inputs(tmp_path, "beauty")
    ctx = _context(tmp_path, config_dir, sid_dir)
    missing_ctx = replace(ctx, semantic_ids_dir=tmp_path / "missing-global-time-sids")
    with pytest.raises(FileNotFoundError, match="missing input files"):
        tuning.base.tune_one_dataset(missing_ctx, "beauty")

    study = optuna.load_study(
        study_name=tuning.base.study_name_for("beauty", split_strategy="global_time"),
        storage=optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
            str(ctx.dataset_dir("beauty") / "optuna_journal.log")
        )),
    )
    assert len(study.trials) == 1 and study.trials[0].state == optuna.trial.TrialState.WAITING
    tuning.base.validate_study_protocol(study, "beauty", ctx)
    assert study.user_attrs["protocol"]["data_config"]["semantic_id_path"] == str(sid_dir / "beauty/semantic_ids.json")
    assert not tuning.base.ensure_incumbent_enqueued(study, ctx.regime.baselines["beauty"])
    trial = study.ask()
    with pytest.raises(ValueError, match="different split/training protocol"):
        tuning.base.validate_study_protocol(study, "beauty", missing_ctx)
    study.tell(trial, 0.5)
    with pytest.raises(ValueError, match="different split/training protocol"):
        tuning.base.validate_study_protocol(study, "beauty", missing_ctx)


@pytest.mark.parametrize("name", ["beauty", "yambda_500m_listens"])
def test_cli_trains_resumes_and_reranks_on_global_time(tmp_path, name):
    config_dir, sid_dir = _inputs(tmp_path, name)
    output = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPT_PATH), "--datasets", name,
           "--config-dir", str(config_dir), "--semantic-ids-dir", str(sid_dir),
           "--output-root", str(output), "--n-trials", "1", "--max-epochs", "1",
           "--debug-max-batches", "1", "--num-workers", "0", "--device", "cpu", "--no-progress-bar"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(*extra):
        result = subprocess.run(cmd + list(extra), cwd=PROJECT_ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
        return result.stdout

    run()
    dataset_dir = output / name
    best = json.loads((dataset_dir / "best_trial.json").read_text())
    assert best["study_name"].endswith("_global_time")
    assert best["best_params"] == tuning.base.load_regime().baselines[name]
    assert torch.load(dataset_dir / "best_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    trials = json.loads((dataset_dir / "all_trials.json").read_text())
    assert len(trials) == 1 and trials[0]["state"] == str(optuna.trial.TrialState.COMPLETE)
    config_path = output / "configs" / f"gpt2_model_{name}.yaml"
    config_text = config_path.read_text()
    assert "Validation split: global_time" in config_text and str(config_dir) in config_text
    profile = tuning.base.run_all.model_profile(yaml.safe_load(config_text), name)
    assert profile["optimizer"]["lr"] == best["best_params"]["lr"]

    log = dataset_dir / tuning.base.EPOCH_LOG_FILENAME
    previous_log = log.read_bytes()
    assert "Study already holds its trial budget" in run()
    assert log.read_bytes() == previous_log
    assert json.loads((dataset_dir / "best_trial.json").read_text()) == best

    if name == "beauty":
        run("--rerank", "--rerank-top-k", "1")
        rerank = json.loads((dataset_dir / "rerank_summary.json").read_text())
        assert [r["seed"] for r in rerank["winner"]["seed_results"]] == [42, 43, 44]
        final = json.loads((dataset_dir / "best_trial.json").read_text())
        assert final["study_name"] == best["study_name"] and "re-rank" in final["selection"]
        assert "Validation split: global_time" in config_path.read_text()
