"""Recovery preserves completed epochs/optimizer/best state and never reopens a failed trial."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import optuna
import pytest
import torch
from torch import nn

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.training.trainer import TrainerConfig
from scripts.gpt2_tuning_resume import (
    AtomicTrialTrainer, inspect_resume_checkpoint, restore_trial_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpt2_resume_test", ROOT / "scripts/exp2-tuned-params/tuning_gpt2.py")
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)

PARAMS = dict(embedding_dim=8, num_heads=2, num_layers=1, ff_mult=2,
              dropout=0.0, lr=0.01, weight_decay=0.01)
IDENTITY = dict(version=1, dataset="tiny", seed=42, hparams=PARAMS,
                protocol_sha256="protocol", data_audit={"train_contexts_sha256": "history"})


class ToyModel(FlowRecModel):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return RecModelOutput()


class ToyMethod(FlowRecMethod):
    def __init__(self, score=0.5):
        super().__init__()
        self.score = score

    def compute_loss(self, model, batch, *, stage="train", step=None):
        if stage == "train":
            return MethodOutput(loss=(model.weight - 1).square().sum())
        return MethodOutput(loss=torch.zeros(()), metrics={"score": self.score})


def trainer(folder, *, max_epochs=3, cadence=1, patience=10, score=0.5, logger=None):
    cfg = TrainerConfig(max_epochs=max_epochs, checkpoint_dir=folder,
                        validate_every_epochs=cadence, monitor="val/score", monitor_mode="max",
                        early_stopping_patience=patience, progress_bar=False, log_every_steps=0)
    return AtomicTrialTrainer.from_optimizer_config(model=ToyModel(), method=ToyMethod(score),
        cfg=cfg, device="cpu", lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"], logger=logger)


def manifest(value, identity=IDENTITY):
    folder = Path(value.cfg.checkpoint_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "resolved_config.json").write_text(json.dumps(dict(
        resume_identity=identity, model_config=value.model.config_dict(), trainer=asdict(value.cfg),
    ), default=str))


def one_completed_epoch(folder, *, identity=IDENTITY, cadence=1, patience=10):
    value = trainer(folder, cadence=cadence, patience=patience)
    manifest(value, identity)
    # Crash at epoch_end: last.pt exists, while the total budget remains 3.
    def interrupt(payload):
        if payload.get("type") == "epoch_end":
            raise KeyboardInterrupt
    value.logger = interrupt
    with pytest.raises(KeyboardInterrupt):
        value.fit([{"x": torch.zeros(2, 1)}], [{"x": torch.zeros(2, 1)}])
    return value


def test_resume_skips_completed_epoch_restores_optimizer_and_keeps_prior_best(tmp_path):
    source = one_completed_epoch(tmp_path / "source")
    recovery = inspect_resume_checkpoint(tmp_path / "source", IDENTITY)
    resumed = trainer(tmp_path / "retry", score=0.4)
    logger = SimpleNamespace(epoch=0)
    assert not restore_trial_checkpoint(resumed, recovery, logger=logger)
    assert resumed.state.epoch == 1 and logger.epoch == 1
    assert resumed.state.optimizer_step == source.state.optimizer_step == 1
    torch.testing.assert_close(resumed.model.weight, source.model.weight)
    for state in resumed.optimizer.state.values():
        assert int(state["step"]) == 1
    result = resumed.fit([{"x": torch.zeros(2, 1)}], [{"x": torch.zeros(2, 1)}])
    assert result.state.global_step == 3  # Three total epochs, not four.
    assert result.state.epoch == 2
    assert result.state.best_epoch == 0 and result.state.best_metric == 0.5
    assert result.state.bad_epochs == 2
    preserved = torch.load(tmp_path / "retry/best.pt", weights_only=False)
    torch.testing.assert_close(preserved["model_state_dict"]["weight"], source.model.weight)


@pytest.mark.parametrize("reason", ["epochs", "patience", "steps"])
def test_finished_checkpoint_does_not_restart_training(tmp_path, reason):
    one_completed_epoch(tmp_path / "source")
    last = tmp_path / "source/last.pt"
    payload = torch.load(last, weights_only=False)
    if reason == "patience":
        payload["trainer_state"]["should_stop"] = True
        torch.save(payload, last)
    recovery = inspect_resume_checkpoint(tmp_path / "source", IDENTITY)
    resumed = trainer(tmp_path / "retry", max_epochs=1 if reason == "epochs" else 3)
    if reason == "steps":
        resumed.cfg = replace(resumed.cfg, max_steps=1)
    assert restore_trial_checkpoint(resumed, recovery)
    assert resumed.state.epoch == 0  # Reports one completed epoch on the no-fit path.


def test_checkpoint_before_first_validation_is_resumable(tmp_path):
    one_completed_epoch(tmp_path / "source", cadence=5)
    recovery = inspect_resume_checkpoint(tmp_path / "source", IDENTITY)
    assert recovery.best_path is None
    resumed = trainer(tmp_path / "retry", max_epochs=6, cadence=5)
    reporter = SimpleNamespace(epoch=100)
    assert not restore_trial_checkpoint(resumed, recovery, logger=reporter)
    assert reporter.epoch == 0 and resumed.state.best_metric is None


def test_best_snapshot_matches_last_when_killed_between_best_and_last(tmp_path):
    value = one_completed_epoch(tmp_path / "source")
    value.state.epoch = value.state.best_epoch = 1
    value.state.best_metric = 0.8
    value.save_checkpoint("best.pt")  # No following last.pt save.
    recovery = inspect_resume_checkpoint(tmp_path / "source", IDENTITY)
    assert recovery.best_path.name == "resume_best_epoch_0.pt"
    assert torch.load(recovery.best_path, weights_only=False)["trainer_state"]["best_metric"] == 0.5


def test_atomic_save_keeps_previous_checkpoint_on_write_failure(tmp_path, monkeypatch):
    value = one_completed_epoch(tmp_path / "source")
    previous = (tmp_path / "source/last.pt").read_bytes()
    def fail(payload, path):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="interrupted"):
        value.save_checkpoint("last.pt")
    assert (tmp_path / "source/last.pt").read_bytes() == previous
    assert not list((tmp_path / "source").glob("*.tmp"))


@pytest.mark.parametrize("changed", ["seed", "hparams", "data_audit", "protocol_sha256"])
def test_recovery_rejects_changed_identity(tmp_path, changed):
    one_completed_epoch(tmp_path / "source")
    identity = dict(IDENTITY, **{changed: "different"})
    with pytest.raises(ValueError, match="protocol, data, parameters or seed"):
        inspect_resume_checkpoint(tmp_path / "source", identity)


def setup_study(tmp_path):
    name = "yambda_50m_listens"
    ctx = tuning.build_context(tuning.parse_args([
        "--match-tiger", "--datasets", name, "--device", "cpu", "--num-workers", "0",
        "--output-root", str(tmp_path),
    ]), tuning.load_regime()).for_dataset(name)
    folder = ctx.dataset_dir(name)
    folder.mkdir(parents=True)
    (folder / "study_config.json").write_text(json.dumps({"fingerprint_sha256": "protocol"}))
    (folder / "data_audit.json").write_text(json.dumps({"train_contexts_sha256": "history"}))
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(tmp_path / "journal.log"))
    )
    study = optuna.create_study(direction="maximize", storage=storage)
    study.enqueue_trial(PARAMS)
    source = study.ask()
    for key, value in PARAMS.items():
        source.suggest_categorical(key, [value])
    identity = tuning.matched_resume_identity(ctx, name, PARAMS, 42)
    one_completed_epoch(folder / "trials/trial_0", identity=identity)
    return ctx, name, study


def test_failed_source_is_immutable_and_retry_queue_is_idempotent(tmp_path):
    ctx, name, study = setup_study(tmp_path)
    last = ctx.dataset_dir(name) / "trials/trial_0/last.pt"
    original = last.read_bytes()
    assert tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name) == ([0], [0])
    trials = study.get_trials()
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert trials[1].state == optuna.trial.TrialState.WAITING
    assert trials[1].system_attrs["fixed_params"] == PARAMS
    assert trials[1].user_attrs["resume_source_trial"] == 0
    assert tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name) == ([], [])
    assert len(study.trials) == 2 and last.read_bytes() == original


def test_retry_interrupted_before_first_epoch_follows_original_checkpoint(tmp_path):
    ctx, name, study = setup_study(tmp_path)
    tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name)
    study.ask()  # Retry died before suggesting all parameters or saving an epoch.
    assert tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name) == ([1], [1])
    source = tuning.matched_resume_source(study, ctx=ctx, dataset_name=name,
                                         source_number=1, hparams=PARAMS)
    assert source.last_path.parent.name == "trial_0"
    assert study.trials[-1].user_attrs["resume_source_trial"] == 1


def test_recovery_finishes_durable_intent_after_failure_before_enqueue(tmp_path, monkeypatch):
    ctx, name, study = setup_study(tmp_path)
    original = study.enqueue_trial
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(study, "enqueue_trial", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    monkeypatch.setattr(study, "enqueue_trial", original)
    assert tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name) == ([], [0])


def test_recovery_rejects_changed_data_before_marking_source_failed(tmp_path):
    ctx, name, study = setup_study(tmp_path)
    (ctx.dataset_dir(name) / "data_audit.json").write_text(json.dumps({"train_contexts_sha256": "changed"}))
    with pytest.raises(ValueError, match="protocol, data"):
        tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name)
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING
    assert len(study.trials) == 1


def test_missing_checkpoint_fails_without_queueing_unverified_continuation(tmp_path):
    ctx, name, study = setup_study(tmp_path)
    (ctx.dataset_dir(name) / "trials/trial_0/last.pt").unlink()
    assert tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name) == ([0], [])
    assert len(study.trials) == 1


def test_recovery_fails_closed_if_lock_fails(tmp_path, monkeypatch):
    ctx, name, study = setup_study(tmp_path)
    def fail(*args):
        raise OSError("locking unavailable")
    monkeypatch.setattr(tuning.fcntl, "flock", fail)
    with pytest.raises(OSError, match="locking unavailable"):
        tuning.recover_stale_matched_trials(study, ctx=ctx, dataset_name=name)
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING


def test_train_once_continuation_returns_prior_best_and_original_total_budget(tmp_path, monkeypatch):
    from flowrec.data.yambda import YambdaDataConfig
    ctx, name, study = setup_study(tmp_path)
    source = tuning.matched_resume_source(study, ctx=ctx, dataset_name=name,
                                         source_number=0, hparams=PARAMS)
    source_bytes = source.last_path.read_bytes()
    monkeypatch.setattr(tuning, "build_model", lambda *args: ToyModel())
    monkeypatch.setattr(tuning, "build_method", lambda *args: ToyMethod(0.4))
    def make_trainer(**kwargs):
        return AtomicTrialTrainer.from_optimizer_config(
            model=kwargs["model"], method=kwargs["method"], device="cpu",
            cfg=TrainerConfig(max_epochs=3, checkpoint_dir=kwargs["checkpoint_dir"],
                              monitor="val/score", monitor_mode="max", early_stopping_patience=10,
                              progress_bar=False, log_every_steps=0),
            lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"],
        )
    monkeypatch.setattr(tuning, "build_trainer", make_trainer)
    batches = [{"x": torch.zeros(2, 1)}]
    dm = SimpleNamespace(cfg=YambdaDataConfig(), summary=lambda: {},
                         train_dataloader=lambda: batches, val_dataloader=lambda: batches)
    output = tmp_path / "continued"
    result = tuning.train_once(ctx=ctx, dm=dm, dataset_name=name, hparams=PARAMS,
                               spec=ctx.spec(name), checkpoint_dir=output, seed=42,
                               resume_from=source)
    assert result["epochs_completed"] == 3 and result["best_epoch"] == 0
    assert result["value"] == 0.5
    assert result["checkpoint"] == output / "best.pt"
    last = torch.load(output / "last.pt", weights_only=False)
    assert last["trainer_state"]["global_step"] == 3
    assert source.last_path.read_bytes() == source_bytes
