"""Global-time TIGER selection, final budget, and interruption regressions."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from flowrec.item_representation.base import SemanticIDStore
from scripts import global_time_tiger_tuning as shared
from test_global_time_representation import _assets


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, shared.PROJECT_ROOT / "scripts/global-time" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


small = load_script("tuning_tiger")
large = load_script("tuning_tiger_yambda_500m")


def test_defaults_partition_datasets_and_keep_full_trainer_schedule(tmp_path):
    args = small.parse_args([])
    assert args.datasets == ["beauty", "instruments", "beauty_2023", "yambda_50m_listens"]
    with pytest.raises(SystemExit):
        small.parse_args(["--datasets", shared.LARGE_DATASET])
    for name in (*args.datasets, shared.LARGE_DATASET):
        run = shared.Run(name, args, search={})
        cfg = run.incumbent.config
        shared.validate_data_config(run.incumbent.data_cfg)
        full = run.trainer_config(cfg, tmp_path, final=True)
        assert full.max_epochs == 500 and full.early_stopping_patience == 10
        assert full.monitor == "val/sft/ndcg@1" and full.val_sample_fraction is None
        assert full.max_grad_norm == 2 and full.gradient_accumulation_steps == 1
        assert run.method(cfg, final=True).cfg.generation_topk == (1,)
        assert run.seeds == [42, 43, 44]
    args = large.parse_args([])
    inc = shared.load_incumbent(shared.LARGE_DATASET, args)
    plan, spec = large.build_plan(args, inc)
    assert len(plan) == 10 and plan[0].name == "baseline"
    run = shared.Run(shared.LARGE_DATASET, args, search=spec)
    short = run.trainer_config(inc.config, tmp_path, final=False)
    assert short.max_epochs == 8 and short.early_stopping_patience is None
    assert short.monitor == shared.MONITOR and short.val_sample_fraction == 0.25
    assert not short.val_sample_resample


@pytest.mark.parametrize("changes", [
    {"split_strategy": "leave_one_out"}, {"global_val_timestamp": None},
    {"global_val_timestamp": 30000000}, {"drop_items_missing_semantic_id": True},
    {"filter_cold_items_after_split": False}, {"cold_item_train_source": "targets"},
])
def test_rejects_incompatible_split(changes):
    cfg = shared.load_incumbent(shared.LARGE_DATASET, large.parse_args([])).data_cfg
    with pytest.raises(ValueError):
        shared.validate_data_config(replace(cfg, **changes))


def test_protocol_and_final_selection_guards(tmp_path):
    args = large.parse_args(["--checkpoint-root", str(tmp_path)])
    run = shared.Run(shared.LARGE_DATASET, args, search={"grid": [1, 2]})
    with shared.dataset_lock(run.root):
        run.check_protocol()
        with pytest.raises(RuntimeError, match="Another TIGER"):
            with shared.dataset_lock(run.root):
                pass
    run.search = {"grid": [1, 3]}
    with pytest.raises(ValueError, match="different split/training/search"):
        run.check_protocol()
    with pytest.raises(RuntimeError, match="entire search"):
        run.train_best({"search_complete": False, "winner": {"overrides": {}}})


@pytest.mark.parametrize("name", ["yambda_50m_listens", shared.LARGE_DATASET])
def test_legacy_protocol_accepts_disabled_dense_option_and_keeps_backup(tmp_path, name):
    args = large.parse_args(["--checkpoint-root", str(tmp_path)])
    run = shared.Run(name, args, search={"grid": [1, 2]})
    current = run.protocol()
    legacy = json.loads(json.dumps(current))
    assert legacy["data"].pop("train_dense_match_prefix") is False
    path = run.root / "study_config.json"
    shared.atomic_json(path, legacy)
    with shared.dataset_lock(run.root):
        run.check_protocol()
        backup = run.root / "study_config.before_compatibility_update.json"
        assert json.loads(backup.read_text()) == legacy
        assert json.loads(path.read_text()) == current
        previous = backup.read_bytes()
        run.check_protocol()
        assert backup.read_bytes() == previous
    # Normalization must not mutate the raw payload used for provenance.
    assert "train_dense_match_prefix" not in legacy["data"]


@pytest.mark.parametrize("field,value", [
    ("data.train_dense_match_prefix", True),
    ("data.global_val_timestamp", 25001166),
    ("data.train_last_n", 64),
    ("data.semantic_id_path", "/different/semantic_ids.json"),
    ("data.unknown_future_option", False),
    ("trainer.search.max_epochs", 7),
    ("trainer.final.amp_dtype", "bfloat16"),
    ("model.embedding_dim", 256),
    ("search.grid", [1, 3]),
])
def test_legacy_protocol_still_rejects_real_changes_without_rewriting(tmp_path, field, value):
    args = large.parse_args(["--checkpoint-root", str(tmp_path)])
    run = shared.Run(shared.LARGE_DATASET, args, search={"grid": [1, 2]})
    saved = run.protocol()
    saved["data"].pop("train_dense_match_prefix")
    parent = saved
    keys = field.split(".")
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value
    path = run.root / "study_config.json"
    shared.atomic_json(path, saved)
    original = path.read_bytes()
    with pytest.raises(ValueError) as error:
        run.check_protocol()
    assert field in str(error.value)
    assert "saved=" in str(error.value) and "current=" in str(error.value)
    assert "Restore the original settings" in str(error.value)
    assert path.read_bytes() == original
    assert not (run.root / "study_config.before_compatibility_update.json").exists()


def smoke_args(root, name):
    family = "yambda" if name.startswith("yambda") else "amazon"
    config_dir = _assets(root, name, family)
    config_name = ("dataset_yambda.yaml" if family == "yambda" else
                   "dataset_amazon_2023.yaml" if name == "beauty_2023" else "dataset_amazon.yaml")
    path = config_dir / config_name
    cfg = yaml.safe_load(path.read_text())
    cfg["tiger"].update(batch_size=2, eval_batch_size=2, test_eval_batch_size=2,
                         num_workers=0, persistent_workers=False, pin_memory=False)
    path.write_text(yaml.safe_dump(cfg))
    sid_dir = root / "sids"
    SemanticIDStore({"1": [0, 256, 512, 768], "2": [1, 257, 513, 769]}).save_json(
        sid_dir / name / "semantic_ids.json")
    model = shared.runner.load_yaml(shared.EXP1_CONFIG_DIR / "tiger_model.yaml")
    model.update(embedding_dim=16, num_heads=2, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=32, d_kv=8)
    model_path = root / "model.yaml"
    model_path.write_text(yaml.safe_dump(model))
    return ["--config-dir", str(config_dir), "--semantic-ids-dir", str(sid_dir),
            "--model-config", str(model_path), "--checkpoint-root", str(root / "checkpoints"),
            "--artifacts-dir", str(root / "artifacts"), "--tuned-config-dir", str(root / "tuned"),
            "--device", "cpu", "--num-workers", "0", "--seeds", "42",
            "--epochs-per-trial", "1", "--max-epochs", "2",
            "--train-batches-per-epoch", "1", "--max-eval-batches", "1", "--time-budget-hours", "-1"]


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_grid_then_fresh_full_run_and_completed_resume(tmp_path, monkeypatch):
    argv = smoke_args(tmp_path, shared.LARGE_DATASET) + ["--params", "lr", "--grid", "lr=0.003"]
    large.main(argv)
    root = tmp_path / "checkpoints" / shared.LARGE_DATASET
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["search_complete"] and len(summary["trials"]) == 2
    assert all(t["epochs_completed"] == 1 for t in summary["trials"])
    assert all("test_metrics" not in t for t in summary["trials"])
    final = json.loads((root / "seed_42/result.json").read_text())
    assert final["epochs_completed"] == 2 and final["epochs_planned"] == 2
    assert final["monitor"] == "val/sft/ndcg@1" and "test_metrics" in final
    assert final["hparams"] == summary["winner"]["hparams"]
    resolved = json.loads((root / "seed_42/resolved_config.json").read_text())
    assert resolved["trainer"]["val_sample_fraction"] is None
    assert resolved["trainer"]["early_stopping_patience"] == 10
    assert resolved["method"]["generation_num_beams"] == 1
    assert yaml.safe_load((tmp_path / "tuned" / f"tiger_model_{shared.LARGE_DATASET}.yaml").read_text())["lr"] == final["hparams"]["lr"]
    # A code update adding the disabled dense-data option must not invalidate
    # the earlier grid results or reload the expensive 500M datamodule.
    protocol_path = root / "study_config.json"
    legacy = json.loads(protocol_path.read_text())
    legacy["data"].pop("train_dense_match_prefix")
    shared.atomic_json(protocol_path, legacy)
    existing_results = {path: path.read_bytes() for path in root.rglob("result.json")}
    def no_data(self):
        pytest.fail("Completed run must not reload the data")
    monkeypatch.setattr(shared.Run, "get_data", no_data)
    large.main(argv)
    assert json.loads((root / "study_config.before_compatibility_update.json").read_text()) == legacy
    assert all(path.read_bytes() == contents for path, contents in existing_results.items())


def test_real_optuna_interruption_resumes_same_trial(tmp_path, monkeypatch):
    name = "yambda_50m_listens"
    argv = smoke_args(tmp_path, name) + ["--datasets", name, "--n-trials", "1", "--stage", "tune"]
    original = shared.Run.train
    interrupted = False
    def interrupt_once(self, *args, **kwargs):
        nonlocal interrupted
        result = original(self, *args, **kwargs)
        if not interrupted:
            interrupted = True
            raise shared.TimeBudgetExceeded("Synthetic interruption before Optuna tell")
        return result
    monkeypatch.setattr(shared.Run, "train", interrupt_once)
    small.main(argv)
    root = tmp_path / "checkpoints" / name
    first = json.loads((root / "tuning_summary.json").read_text())
    assert not first["search_complete"] and first["trials"][0]["status"] == "running"
    small.main(argv)
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["search_complete"] and len(summary["trials"]) == 1
    assert summary["winner"]["trial"] == "trial_0000"
    assert not (root / "seed_42").exists()
    small.main(argv[:-2] + ["--stage", "train-best"])
    assert (root / "seed_42/result.json").is_file()


def test_grid_nan_keeps_completed_trials_continues_and_does_not_retry(tmp_path, monkeypatch):
    argv = smoke_args(tmp_path, shared.LARGE_DATASET) + [
        "--params", "lr", "--grid", "lr=0.003,0.001", "--epochs-per-trial", "2",
    ]
    original = shared.Trainer.train_one_epoch
    failures = []

    def unstable_trial(self, *args, **kwargs):
        if (self.cfg.monitor == shared.MONITOR and kwargs["epoch"] == 1
                and self.optimizer.param_groups[0]["lr"] == 0.003):
            failures.append(self.state.global_step)
            raise FloatingPointError("Non-finite loss at step 29672: nan")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(shared.Trainer, "train_one_epoch", unstable_trial)
    large.main(argv)
    root = tmp_path / "checkpoints" / shared.LARGE_DATASET
    failed_dir = root / "trials/lr-0.003"
    failed = json.loads((failed_dir / "result.json").read_text())
    assert failed["status"] == "non_finite_loss" and failed["best_value"] is None
    assert "29672" in failed["error"] and failed["hparams"]["lr"] == 0.003
    assert (failed_dir / "last.pt").exists() and (failed_dir / "best.pt").exists()
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["search_complete"] and len(summary["trials"]) == 3
    assert summary["winner"]["trial"] != "lr-0.003"
    assert json.loads((root / "trials/lr-0.001/result.json").read_text())["status"] == "complete"
    assert (root / "seed_42/result.json").exists()
    preserved = {p: p.read_bytes() for p in root.rglob("result.json")}

    def no_data(self):
        pytest.fail("Completed and numerically failed trials must not reload data on resume")

    monkeypatch.setattr(shared.Run, "get_data", no_data)
    large.main(argv)
    assert len(failures) == 1
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize("name", shared.DATASETS)
def test_optuna_nan_prunes_trial_and_continues(tmp_path, monkeypatch, name):
    argv = smoke_args(tmp_path, name) + ["--datasets", name, "--n-trials", "2", "--stage", "tune"]
    original = shared.Trainer.train_one_epoch
    calls = 0

    def first_trial_nan(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("Non-finite loss at step 0: nan")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(shared.Trainer, "train_one_epoch", first_trial_nan)
    small.main(argv)
    root = tmp_path / "checkpoints" / name
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["search_complete"] and len(summary["trials"]) == 2
    failed, completed = summary["trials"]
    assert failed["status"] == "pruned" and failed["pruned_reason"] == "non_finite_loss"
    assert "Non-finite loss" in failed["error"]
    assert completed["status"] == "complete" and summary["winner"]["trial"] == "trial_0001"
    calls_before_resume = calls
    small.main(argv)
    assert calls == calls_before_resume


def test_optuna_resumed_trial_nan_preserves_earlier_results(tmp_path, monkeypatch):
    name = "beauty"
    argv = smoke_args(tmp_path, name) + [
        "--datasets", name, "--n-trials", "3", "--stage", "tune", "--epochs-per-trial", "2",
    ]
    original = shared.Trainer.train_one_epoch
    failed_attempts = 0

    class OldWorkerCrash(BaseException):
        """Reproduce the old worker exiting without telling Optuna a final state."""

    def unstable_trial(self, *args, **kwargs):
        nonlocal failed_attempts
        if Path(self.cfg.checkpoint_dir).name == "trial_0001" and kwargs["epoch"] == 1:
            failed_attempts += 1
            if failed_attempts == 1:
                raise OldWorkerCrash()
            raise FloatingPointError("Non-finite loss at step 1242: nan")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(shared.Trainer, "train_one_epoch", unstable_trial)
    with pytest.raises(OldWorkerCrash):
        small.main(argv)
    root = tmp_path / "checkpoints" / name
    interrupted = root / "trials/trial_0001"
    assert (interrupted / "last.pt").exists()
    assert not (interrupted / "result.json").exists()
    completed_files = {p: p.read_bytes() for p in (root / "trials/trial_0000").iterdir()}

    # Reload the same journal and checkpoint under the updated handler.
    small.main(argv)
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["search_complete"] and len(summary["trials"]) == 3
    assert [t["status"] for t in summary["trials"]] == ["complete", "pruned", "complete"]
    assert summary["trials"][1]["pruned_reason"] == "non_finite_loss"
    assert "1242" in summary["trials"][1]["error"]
    assert summary["winner"]["trial"] != "trial_0001"
    assert all(p.read_bytes() == value for p, value in completed_files.items())
    # The interrupted trial resumes at epoch two instead of repeating epoch one.
    records = [json.loads(line) for line in (interrupted / "epoch_log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in records] == [0]
    assert failed_attempts == 2
    small.main(argv)
    assert failed_attempts == 2


def test_all_grid_points_nan_prevents_final_training(tmp_path, monkeypatch):
    argv = smoke_args(tmp_path, shared.LARGE_DATASET) + ["--params", "lr", "--grid", "lr=0.003"]

    def always_nan(self, *args, **kwargs):
        raise FloatingPointError("Non-finite loss at step 0: nan")

    monkeypatch.setattr(shared.Trainer, "train_one_epoch", always_nan)
    with pytest.raises(RuntimeError, match="no successful trial"):
        large.main(argv)
    root = tmp_path / "checkpoints" / shared.LARGE_DATASET
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["search_complete"] and summary["winner"] is None
    assert all(t["status"] == "non_finite_loss" for t in summary["trials"])
    assert not (root / "seed_42").exists()
    assert not (tmp_path / "tuned" / f"tiger_model_{shared.LARGE_DATASET}.yaml").exists()


def test_final_nan_still_stops_training(tmp_path, monkeypatch):
    args = large.parse_args(smoke_args(tmp_path, shared.LARGE_DATASET))
    run = shared.Run(shared.LARGE_DATASET, args, search={})

    def always_nan(self, *args, **kwargs):
        raise FloatingPointError("Non-finite loss at step 0: nan")

    monkeypatch.setattr(shared.Trainer, "train_one_epoch", always_nan)
    with pytest.raises(FloatingPointError):
        run.train_best({"search_complete": True, "winner": {"trial": "baseline", "overrides": {}}})
    assert not (run.root / "seed_42/result.json").exists()


def test_grid_unexpected_error_is_not_discarded(tmp_path, monkeypatch):
    argv = smoke_args(tmp_path, shared.LARGE_DATASET) + ["--params", "lr"]

    def broken(self, *args, **kwargs):
        raise RuntimeError("Unexpected tensor shape")

    monkeypatch.setattr(shared.Trainer, "train_one_epoch", broken)
    with pytest.raises(RuntimeError, match="Unexpected tensor shape"):
        large.main(argv)
    assert not (tmp_path / "checkpoints" / shared.LARGE_DATASET / "trials/baseline/result.json").exists()


@pytest.mark.parametrize("early_stopped", [False, True])
def test_resume_preserves_completed_epochs_and_early_stop(tmp_path, early_stopped):
    argv = smoke_args(tmp_path, shared.LARGE_DATASET)
    args = large.parse_args(argv)
    run = shared.Run(shared.LARGE_DATASET, args, search={})
    cfg = run.incumbent.config
    directory = run.root / "seed_42"
    # Interrupt after checkpointing epoch one, before a completion marker.
    def stop(payload):
        raise shared.TimeBudgetExceeded("Synthetic walltime")
    with pytest.raises(shared.TimeBudgetExceeded):
        run.train(cfg, directory, final=True, report_epoch=stop)
    assert (directory / "last.pt").exists() and not (directory / "result.json").exists()
    if early_stopped:
        checkpoint = torch.load(directory / "last.pt", weights_only=False)
        checkpoint["trainer_state"]["should_stop"] = True
        torch.save(checkpoint, directory / "last.pt")
    run.train(cfg, directory, final=True)
    records = [json.loads(line) for line in (directory / "epoch_log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in records] == ([0] if early_stopped else [0, 1])
    result = json.loads((directory / "result.json").read_text())
    assert result["epochs_completed"] == (1 if early_stopped else 2)


@pytest.mark.skipif(
    not all((shared.PROJECT_ROOT / "scripts/slurm" / name).is_file() for name in (
        "tuning_tiger_yambda_500m_global_time.sbatch", "tuning_tiger_global_time_array.sbatch"
    )), reason="Slurm scripts are gitignored",
)
def test_slurm_500m_restricted_to_e_f_only():
    root = shared.PROJECT_ROOT / "scripts/slurm"
    large_text = (root / "tuning_tiger_yambda_500m_global_time.sbatch").read_text()
    assert '#SBATCH --constraint="type_e|type_f"' in large_text
    assert "type_h" not in large_text
    small_text = (root / "tuning_tiger_global_time_array.sbatch").read_text()
    assert "#SBATCH --array=0-3" in small_text
    assert "#SBATCH --constraint" not in small_text
