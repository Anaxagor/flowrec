"""Global-time full-CE tuning: catalog, shared budget, recovery and outputs."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import sys
from dataclasses import asdict, replace
from pathlib import Path

import optuna
import pytest
import torch
import yaml
from optuna.trial import TrialState

from test_global_time_representation import _assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "global_sasrec_tuning", PROJECT_ROOT / "scripts/global-time/tuning_sasrec.py"
)
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)


def _runtime(root, config_dir, **kwargs):
    return tuning.Runtime(
        config_dir=config_dir, checkpoint_root=root / "checkpoints",
        artifacts_dir=root / "artifacts", tuned_config_dir=root / "tuned",
        device="cpu", use_amp=False, progress_bar=False, **kwargs,
    )


def _study(rt, *, worker_id=0):
    _, cfg = tuning.resolve_data_config("beauty", rt)
    return tuning.open_study("beauty", rt, worker_id=worker_id, data_cfg=cfg)


def test_defaults_are_four_full_ce_studies_of_100_trials():
    args = tuning.parse_args([])
    assert args.datasets == ["beauty", "instruments", "beauty_2023", "yambda_50m_listens"]
    assert args.n_trials == 100
    for name, spec in tuning.TUNING_SPECS.items():
        assert spec.n_trials == 100
        assert asdict(replace(spec, n_trials=tuning.base.TUNING_SPECS[name].n_trials)) == asdict(tuning.base.TUNING_SPECS[name])
    for path in (args.config_dir, args.checkpoint_root, args.artifacts_dir, args.tuned_config_dir):
        assert "global-time" in path.parts


@pytest.mark.parametrize("name", tuning.DATASETS)
def test_every_dataset_removes_cold_targets_after_global_split(tmp_path, name):
    family = "yambda" if name.startswith("yambda") else "amazon"
    config_dir = _assets(tmp_path, name, family)
    rt = _runtime(tmp_path, config_dir)
    group, cfg = tuning.resolve_data_config(name, rt)
    data = tuning.build_datamodule(group, cfg)
    assert data.item2id.keys() == {"1", "2"}
    assert data.post_split_filter_stats.enabled
    assert data.post_split_filter_stats.num_items_removed == 1
    assert len(data.val_dataset) == len(data.test_dataset) == 1
    assert cfg.semantic_id_path is None
    assert tuning.base.build_method(data).cfg.item_loss == "cross_entropy"


@pytest.mark.parametrize("changes", [{"filter_cold_items_after_split": False},
                                     {"split_strategy": "leave_one_out"},
                                     {"cold_item_train_source": "targets"}])
def test_incompatible_split_settings_are_rejected(tmp_path, changes):
    config_dir = _assets(tmp_path, "beauty")
    path = config_dir / "dataset_amazon.yaml"
    config = yaml.safe_load(path.read_text())
    config.update(changes)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="requires global_time"):
        tuning.resolve_data_config("beauty", _runtime(tmp_path, config_dir))


def test_array_maps_both_dataset_and_trial_worker_dimensions():
    assignments = [tuning.array_assignment(i, 8) for i in range(32)]
    assert assignments == [(name, worker) for name in tuning.DATASETS for worker in range(8)]
    args = tuning.parse_args(["--array-task-id", "23", "--workers-per-dataset", "8"])
    assert args.datasets == ["beauty_2023"] and args.worker_id == 7
    with pytest.raises(ValueError):
        tuning.array_assignment(32, 8)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "scripts/slurm/tuning_sasrec_global_time_array.sbatch").is_file(),
    reason="Slurm scripts are gitignored",
)
def test_array_launcher_resources():
    launcher = (PROJECT_ROOT / "scripts/slurm/tuning_sasrec_global_time_array.sbatch").read_text()
    directives = [line for line in launcher.splitlines() if line.startswith("#SBATCH")]
    assert all(not any(key in line for key in ("--constraint", "--partition", "--nodelist", "--exclude", "--gres"))
               for line in directives)


def test_interrupted_trial_reuses_slot_and_preserves_baseline(tmp_path):
    rt = _runtime(tmp_path, _assets(tmp_path, "beauty"), n_trials=1)
    study = _study(rt, worker_id=7)  # Worker 0 is not required to enqueue baseline.
    spec = tuning.effective_spec("beauty", rt)
    dataset_dir = rt.checkpoint_root / "beauty"
    lease = tuning.claim_trial(study, dataset_dir, spec, worker_id=7)
    assert lease.trial.params == tuning.base.EXP1_BASELINE_PARAMS
    assert tuning.claim_trial(study, dataset_dir, spec, worker_id=8) is None
    lease.close()  # Simulate process exit without a tell().
    recovered = tuning.claim_trial(study, dataset_dir, spec, worker_id=8)
    assert recovered.trial.number == 0
    assert recovered.trial.params == tuning.base.EXP1_BASELINE_PARAMS
    assert recovered.trial.user_attrs["restarts"] == 1
    study.tell(recovered.trial, state=TrialState.PRUNED)
    recovered.close()
    assert tuning.claim_trial(study, dataset_dir, spec, worker_id=9) is None
    assert len(study.trials) == 1


def test_resume_rejects_different_cutoff_but_allows_different_gpu_precision(tmp_path):
    rt = _runtime(tmp_path, _assets(tmp_path, "beauty"), n_trials=2)
    _study(rt)
    _study(replace(rt, amp_dtype="float16"), worker_id=1)
    _, cfg = tuning.resolve_data_config("beauty", rt)
    with pytest.raises(ValueError, match="different split/training protocol"):
        tuning.open_study("beauty", rt, worker_id=1,
                          data_cfg=replace(cfg, global_val_timestamp=9))


def _fake_train_once(**kwargs):
    number = kwargs["trial"].number
    path = kwargs["trial_dir"] / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"checkpoint of trial {number}")
    return {"value": number / 100, "best_epoch": 0, "epochs_completed": 1, "checkpoint": str(path)}


def _parallel_worker(root, config_dir, barrier, worker_id):
    root, config_dir = Path(root), Path(config_dir)
    rt = _runtime(root, config_dir, n_trials=12)
    study = _study(rt, worker_id=worker_id)
    tuning.train_once = _fake_train_once
    barrier.wait(timeout=30)
    while (lease := tuning.claim_trial(study, rt.checkpoint_root / "beauty",
                                      tuning.effective_spec("beauty", rt), worker_id=worker_id)) is not None:
        tuning.run_trial(lease, study=study, dataset_name="beauty", data=None, rt=rt)


def test_parallel_workers_share_exact_budget_and_publish_consistent_best(tmp_path):
    config_dir = _assets(tmp_path, "beauty")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(4)
    processes = [ctx.Process(target=_parallel_worker, args=(str(tmp_path), str(config_dir), barrier, i)) for i in range(4)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    rt = _runtime(tmp_path, config_dir, n_trials=12)
    study = _study(rt)
    assert len(study.trials) == 12
    assert all(t.state == TrialState.COMPLETE for t in study.trials)
    assert len({t.user_attrs["worker_id"] for t in study.trials}) > 1
    dataset_dir = rt.checkpoint_root / "beauty"
    best = json.loads((dataset_dir / "best_trial.json").read_text())
    assert best["best_trial_number"] == 11 and best["trials_completed"] == 12
    assert (dataset_dir / "best_model.pt").read_text() == "checkpoint of trial 11"
    assert json.loads((rt.artifacts_dir / "sasrec_beauty_best_trial.json").read_text()) == best
    assert len(json.loads((dataset_dir / "all_trials.json").read_text())) == 12


@pytest.mark.parametrize("error,state", [(RuntimeError("CUDA out of memory"), TrialState.PRUNED),
                                        (FloatingPointError("non-finite loss"), TrialState.PRUNED),
                                        (ValueError("unexpected bug"), TrialState.FAIL)])
def test_unsuccessful_trials_are_recorded_without_a_best_model(tmp_path, monkeypatch, error, state):
    rt = _runtime(tmp_path, _assets(tmp_path, "beauty"), n_trials=1)
    study = _study(rt)
    dataset_dir = rt.checkpoint_root / "beauty"
    lease = tuning.claim_trial(study, dataset_dir, tuning.effective_spec("beauty", rt), worker_id=0)
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(tuning, "train_once", fail)
    if state == TrialState.FAIL:
        with pytest.raises(ValueError, match="unexpected bug"):
            tuning.run_trial(lease, study=study, dataset_name="beauty", data=None, rt=rt)
    else:
        tuning.run_trial(lease, study=study, dataset_name="beauty", data=None, rt=rt)
    assert study.trials[0].state == state
    assert not (dataset_dir / "best_model.pt").exists()
    summary = json.loads((dataset_dir / "tuning_summary.json").read_text())
    assert summary["trial_states"][state.name] == 1


def test_cpu_training_writes_loadable_checkpoint_and_resumes_without_retraining(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    rt = _runtime(tmp_path, _assets(tmp_path, "beauty"), n_trials=2, max_epochs=1,
                  num_workers=0, train_batches_per_epoch=1, max_eval_batches=1)
    summary = tuning.tune_one_dataset("beauty", rt)
    assert summary["trials_completed"] == 2
    checkpoint = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["model_state_dict"]
    config = yaml.safe_load(Path(summary["tuned_config"]).read_text())
    assert config["sft"]["item_loss"] == "cross_entropy"
    data_summary = json.loads((rt.checkpoint_root / "beauty/data_summary.json").read_text())
    assert data_summary["post_split_filter"]["num_items_removed"] == 1
    def unexpected_setup(*args, **kwargs):
        raise AssertionError("Finished study must not rebuild data or train")
    monkeypatch.setattr(tuning, "build_datamodule", unexpected_setup)
    assert tuning.tune_one_dataset("beauty", rt, worker_id=3) == summary
