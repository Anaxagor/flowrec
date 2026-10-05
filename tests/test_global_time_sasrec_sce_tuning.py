"""SCE tuning retains its training protocol on the global-time Yambda split."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from flowrec.ft_methods.sft.config import SFTConfig
from test_global_time_representation import _assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/global-time/tuning_sasrec_sce.py"
SPEC = importlib.util.spec_from_file_location("global_sasrec_sce_tuning", SCRIPT_PATH)
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)
DATASET = "yambda_500m_listens"


def _config(rt):
    group, config = tuning.base.find_group(DATASET, config_dir=rt.config_dir)
    return group.build_data_config(DATASET, config, seed=tuning.base.SEED)


def _smoke_config(tmp_path):
    config_dir = _assets(tmp_path, DATASET, "yambda")
    path = config_dir / "dataset_yambda.yaml"
    config = yaml.safe_load(path.read_text())
    config["datasets"][DATASET]["dataset_size"] = "500m"
    path.write_text(yaml.safe_dump(config))
    return config_dir


def test_defaults_change_only_split_and_paths():
    args = tuning.parse_args([])
    rt = tuning.runtime_from_args(args)
    legacy = tuning.base.runtime_from_args(tuning.base.parse_args([]))
    assert args.datasets == [DATASET] and args.spec is None and rt.n_trials is None
    assert tuning.base.TUNING_SPECS[DATASET].n_trials == 30
    assert replace(rt, config_dir=legacy.config_dir, checkpoint_root=legacy.checkpoint_root,
                   artifacts_dir=legacy.artifacts_dir, tuned_config_dir=legacy.tuned_config_dir,
                   required_split_strategy=None) == legacy
    for path in (rt.config_dir, rt.checkpoint_root, rt.artifacts_dir, rt.tuned_config_dir):
        assert "global-time" in path.parts
    cfg = _config(rt)
    tuning.base.validate_data_config(DATASET, cfg, rt)
    assert cfg.dataset_size == "500m"
    assert (cfg.global_val_timestamp, cfg.global_test_timestamp) == (25001165, 25510950)
    old_cfg = _config(legacy)
    assert replace(cfg, split_strategy=old_cfg.split_strategy,
                   global_val_timestamp=old_cfg.global_val_timestamp,
                   global_test_timestamp=old_cfg.global_test_timestamp,
                   global_history_policy=old_cfg.global_history_policy,
                   global_eval_target_mode=old_cfg.global_eval_target_mode,
                   drop_items_missing_semantic_id=old_cfg.drop_items_missing_semantic_id) == old_cfg
    assert tuning.base.study_name_for(DATASET, split_strategy=rt.required_split_strategy) != tuning.base.study_name_for(DATASET)


@pytest.mark.parametrize("changes", [
    {"split_strategy": "leave_one_out"},
    {"global_val_timestamp": None},
    {"global_test_timestamp": 1},
    {"global_history_policy": "train_only"},
    {"global_eval_target_mode": "all"},
    {"filter_cold_items_after_split": False},
    {"cold_item_train_source": "targets"},
    {"semantic_id_path": Path("semantic_ids.json")},
])
def test_incompatible_data_configs_are_rejected(changes):
    rt = tuning.runtime_from_args(tuning.parse_args([]))
    with pytest.raises(ValueError, match="requires|semantic IDs"):
        tuning.base.validate_data_config(DATASET, replace(_config(rt), **changes), rt)


def test_split_drops_cold_last_targets_without_substituting_earlier_warm_items(tmp_path):
    rt = tuning.runtime_from_args(tuning.parse_args([
        "--config-dir", str(_smoke_config(tmp_path)), "--device", "cpu", "--num-workers", "0",
    ]))
    data = tuning.base.build_datamodule(DATASET, rt)
    assert data.item2id.keys() == {"1", "2"}
    assert data.post_split_filter_stats.num_items_removed == 1
    assert len(data.val_dataset) == len(data.test_dataset) == 1
    method = tuning.base.build_method(data, tuning.base.resolve_sce_config(
        tuning.base.SCE_BASELINE_PARAMS, bucket_base=tuning.base.sce_bucket_base(data),
    ))
    assert method.cfg.item_loss == "scalable_cross_entropy"
    assert method.cfg.item_topk == (10,)


def test_cli_search_resume_and_rerank_use_one_global_time_study(tmp_path):
    config_dir = _smoke_config(tmp_path)
    checkpoint_root, artifacts, tuned_dir = (tmp_path / name for name in ("checkpoints", "artifacts", "tuned"))
    cmd = [sys.executable, str(SCRIPT_PATH),
           "--config-dir", str(config_dir), "--checkpoint-root", str(checkpoint_root),
           "--artifacts-dir", str(artifacts), "--tuned-config-dir", str(tuned_dir),
           "--worker-id", "0", "--n-trials", "1", "--max-epochs", "1",
           "--train-batches-per-epoch", "1", "--max-eval-batches", "1",
           "--num-workers", "0", "--device", "cpu", "--no-progress-bar"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(*extra):
        result = subprocess.run(cmd + list(extra), cwd=PROJECT_ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        return result.stdout

    run()
    dataset_dir = checkpoint_root / DATASET
    best_path = dataset_dir / "best_trial.json"
    best = json.loads(best_path.read_text())
    assert best["study_name"].endswith("_global_time")
    assert best["trials_completed"] == best["trials_planned"] == 1
    assert best["item_loss"] == "scalable_cross_entropy"
    assert best["best_params"] == {**tuning.base.tuning.EXP1_BASELINE_PARAMS, **tuning.base.SCE_BASELINE_PARAMS}
    checkpoint = dataset_dir / "best_model.pt"
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    config_path = tuned_dir / f"sasrec_sce_model_{DATASET}.yaml"
    config_text = config_path.read_text()
    assert "Validation split: global_time" in config_text
    assert str(config_dir / "dataset_yambda.yaml") in config_text
    assert str(checkpoint) in config_text
    sft = SFTConfig.from_dict(yaml.safe_load(config_text)["sft"])
    assert sft.scalable_cross_entropy.to_dict() == best["scalable_cross_entropy"]

    log_path = dataset_dir / tuning.base.EPOCH_LOG_FILENAME
    epoch_log = log_path.read_bytes()
    assert "Resuming study: 1/1 finished trials." in run()
    assert log_path.read_bytes() == epoch_log
    assert json.loads(best_path.read_text()) == best

    run("--rerank", "--rerank-top-k", "1")
    rerank = json.loads((dataset_dir / "rerank_summary.json").read_text())
    assert rerank["study_name"] == best["study_name"]
    assert rerank["rerank_seeds"] == list(tuning.base.RERANK_SEEDS)
    assert len(rerank["rerank_ranking"][0]["seed_results"]) == 3
    final = json.loads(best_path.read_text())
    assert "re-rank" in final["selection"]
    assert json.loads((artifacts / f"sasrec_sce_{DATASET}_best_trial.json").read_text()) == final
    assert checkpoint.is_file()
