"""run_validation_at_start makes the untouched init weights the reference.

Fine-tuning protocol (configs/exp2-tuned-params/finetune_*.json): the init
checkpoint is scored before the first epoch, seeds best_metric / best.pt
(best_epoch = -1) and is the reference for early-stopping patience, so a run
that never beats its init stops after `patience` validations and keeps the
init as its best checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.training.trainer import Trainer, TrainerConfig


class _ToyModel(FlowRecModel):
    model_type = "toy"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, batch):  # pragma: no cover - not used by the toy method
        return RecModelOutput()


class _ScriptedMethod(FlowRecMethod):
    """Training moves the weight; validation returns a scripted sequence of
    monitor values (one per validation call), independent of the weight."""

    method_name = "scripted"

    def __init__(self, val_values: list[float]) -> None:
        super().__init__()
        self.val_values = list(val_values)
        self.val_calls = 0

    def compute_loss(self, model, batch, *, stage="train", step=None):
        if stage == "train":
            loss = (model.weight - 1.0).pow(2).sum()
            return MethodOutput(loss=loss)
        value = self.val_values[min(self.val_calls, len(self.val_values) - 1)]
        self.val_calls += 1
        return MethodOutput(loss=torch.zeros(()), metrics={"ndcg@5": torch.tensor(float(value))})


def _loader(n: int = 2):
    return [{"x": torch.zeros(4, 1)} for _ in range(n)]


def _trainer(tmp_path: Path, method: _ScriptedMethod, **overrides) -> Trainer:
    cfg = TrainerConfig(
        max_epochs=overrides.pop("max_epochs", 100),
        validate_every_epochs=overrides.pop("validate_every_epochs", 5),
        run_validation_at_start=overrides.pop("run_validation_at_start", True),
        monitor="val/ndcg@5",
        monitor_mode="max",
        early_stopping_patience=overrides.pop("early_stopping_patience", 3),
        checkpoint_dir=tmp_path / "checkpoints",
        progress_bar=False,
        log_every_steps=10_000,
        **overrides,
    )
    return Trainer.from_optimizer_config(model=_ToyModel(), method=method, cfg=cfg, device="cpu", lr=0.1)


def test_input_timing_reports_first_batch_without_changing_optimizer_updates():
    events = []
    trainers = []
    for log_every in (0, 2):
        cfg = TrainerConfig(
            max_epochs=1, validate_every_epochs=None, gradient_accumulation_steps=2,
            progress_bar=False, log_every_steps=log_every, save_best=False, save_last=False,
        )
        trainer = Trainer.from_optimizer_config(
            model=_ToyModel(), method=_ScriptedMethod([]), cfg=cfg,
            device="cpu", lr=0.1, logger=events.append if log_every else None,
        )
        batches = [dict(batch, _gflow_reward_prepare_seconds=0.125) for batch in _loader(3)]
        trainer.fit(batches)
        trainers.append(trainer)
    assert torch.equal(trainers[0].model.weight, trainers[1].model.weight)
    assert trainers[0].state.optimizer_step == trainers[1].state.optimizer_step == 2
    steps = [event for event in events if event["type"] == "train_step"]
    assert [event["global_step"] for event in steps] == [1, 2]
    for event in steps:
        assert 0 <= event["train/data_wait_sec"] <= event["train/step_wall_sec"]
        assert event["train/examples_per_sec"] > 0
        assert event["train/reward_prepare_sec"] == pytest.approx(0.125)


class TestValidationAtStart:
    def test_init_seeds_best_and_saves_best_pt(self, tmp_path: Path) -> None:
        # init 0.5, then every validation worse than that
        method = _ScriptedMethod([0.5, 0.4, 0.3, 0.2])
        trainer = _trainer(tmp_path, method, early_stopping_patience=3)
        fit = trainer.fit(_loader(), _loader(1))

        assert fit.state.best_metric == pytest.approx(0.5)
        assert fit.state.best_epoch == -1
        assert fit.best_checkpoint_path is not None
        assert Path(fit.best_checkpoint_path).name == "best.pt"
        payload = torch.load(fit.best_checkpoint_path, map_location="cpu")
        # best.pt holds the untouched init weights (zeros), not the trained ones
        assert float(payload["model_state_dict"]["weight"]) == 0.0
        assert float(trainer.model.weight) != 0.0
        # history: the start record first, tagged as the initial weights
        assert fit.state.history[0]["epoch"] == -1
        assert fit.state.history[0]["initial_weights"] is True

    def test_non_improving_run_stops_after_patience_validations(self, tmp_path: Path) -> None:
        method = _ScriptedMethod([0.5, 0.4])
        trainer = _trainer(tmp_path, method, validate_every_epochs=5, early_stopping_patience=10)
        fit = trainer.fit(_loader(), _loader(1))
        # 10 non-improving validations at epochs 4, 9, ..., 49 -> stop after epoch 49
        assert fit.state.should_stop is True
        assert fit.state.epoch + 1 == 50
        assert method.val_calls == 11  # init + 10 validations
        assert fit.state.best_epoch == -1
        assert fit.state.best_metric == pytest.approx(0.5)

    def test_improving_epoch_replaces_init(self, tmp_path: Path) -> None:
        # init 0.5, first validation 0.6 (epoch 4), then worse
        method = _ScriptedMethod([0.5, 0.6, 0.55])
        trainer = _trainer(tmp_path, method, validate_every_epochs=5, early_stopping_patience=2)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.best_metric == pytest.approx(0.6)
        assert fit.state.best_epoch == 4
        payload = torch.load(fit.best_checkpoint_path, map_location="cpu")
        assert float(payload["model_state_dict"]["weight"]) != 0.0
        # epoch 4 best, then validations at 9 and 14 fail -> stop after epoch 14
        assert fit.state.epoch + 1 == 15

    def test_disabled_keeps_the_old_rule(self, tmp_path: Path) -> None:
        method = _ScriptedMethod([0.4, 0.3])
        trainer = _trainer(tmp_path, method, run_validation_at_start=False,
                           validate_every_epochs=5, early_stopping_patience=2)
        fit = trainer.fit(_loader(), _loader(1))
        assert method.val_calls == 3
        assert fit.state.best_epoch == 4  # the first post-training validation
        assert fit.state.best_metric == pytest.approx(0.4)
        assert fit.state.history[0]["epoch"] == 0

    def test_resume_skips_the_start_validation(self, tmp_path: Path) -> None:
        method = _ScriptedMethod([0.5, 0.4])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1,
                           early_stopping_patience=None, max_epochs=2)
        fit = trainer.fit(_loader(), _loader(1))
        assert method.val_calls == 3  # init + 2 epochs
        assert fit.last_checkpoint_path is not None

        resumed_method = _ScriptedMethod([0.9])  # would win if it ran at start
        resumed = _trainer(tmp_path / "resume", resumed_method, validate_every_epochs=1,
                           early_stopping_patience=None, max_epochs=3)
        resumed.load_checkpoint(fit.last_checkpoint_path)
        assert resumed.state.epoch > 0 or resumed.state.global_step > 0
        fit2 = resumed.fit(_loader(), _loader(1))
        # no start record was added: the restored state is continued as-is
        assert all(rec["epoch"] >= 0 for rec in fit2.state.history if rec not in fit.state.history)
        assert fit2.state.best_metric == pytest.approx(0.9)  # improved by a real epoch, best_epoch >= 0
        assert fit2.state.best_epoch >= 0

    def test_scheduler_is_not_stepped_at_start(self, tmp_path: Path) -> None:
        method = _ScriptedMethod([0.5, 0.4])
        model = _ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        cfg = TrainerConfig(
            max_epochs=1, validate_every_epochs=1, run_validation_at_start=True,
            monitor="val/ndcg@5", monitor_mode="max", scheduler_step_on="epoch",
            checkpoint_dir=tmp_path / "checkpoints", progress_bar=False,
        )
        trainer = Trainer(model=model, method=method, optimizer=optimizer, cfg=cfg,
                          device="cpu", scheduler=scheduler)
        trainer.fit(_loader(), _loader(1))
        # exactly one scheduler step (the epoch's), none for the start validation
        assert scheduler.last_epoch == 1


class TestEarlyStoppingWarmup:
    """early_stopping_min_epochs: the first epochs never count towards
    patience (a run may not be judged before e.g. its curriculum has ended)."""

    def test_patience_starts_counting_after_the_warmup(self, tmp_path: Path) -> None:
        # init 0.5, nothing ever beats it; validate every epoch, patience 5
        method = _ScriptedMethod([0.5, 0.4])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1, early_stopping_patience=5,
                           early_stopping_min_epochs=10)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.should_stop is True
        assert fit.state.epoch + 1 == 15  # 10 warm-up epochs + 5 counted validations
        assert fit.state.best_epoch == -1 and fit.state.best_metric == pytest.approx(0.5)

    def test_without_warmup_the_same_run_stops_after_patience(self, tmp_path: Path) -> None:
        method = _ScriptedMethod([0.5, 0.4])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1, early_stopping_patience=5)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.epoch + 1 == 5

    def test_best_checkpoint_is_still_tracked_during_the_warmup(self, tmp_path: Path) -> None:
        # epoch index 2 sets the best inside the warm-up; nothing later beats it
        method = _ScriptedMethod([0.5, 0.4, 0.4, 0.7, 0.4])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1, early_stopping_patience=5,
                           early_stopping_min_epochs=10)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.best_epoch == 2 and fit.state.best_metric == pytest.approx(0.7)
        assert fit.state.epoch + 1 == 15


class TestPatienceRule:
    """Patience, validated every epoch: fine-tuning stops once `patience`
    consecutive epochs fail to surpass the CURRENT best (the running maximum,
    seeded by the init checkpoint's score) - not a fixed reference."""

    def test_stops_when_no_epoch_surpasses_the_current_best_within_patience(self, tmp_path: Path) -> None:
        # init 0.50 | epoch 0: 0.40 | epoch 1: 0.60 = new best | then five epochs below 0.60
        method = _ScriptedMethod([0.50, 0.40, 0.60, 0.55, 0.58, 0.59, 0.57, 0.56, 0.99])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1, early_stopping_patience=5)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.best_epoch == 1 and fit.state.best_metric == pytest.approx(0.60)
        # 0.55-0.59 all beat the INIT (0.50) but not the current best: they count
        assert fit.state.epoch + 1 == 7 and fit.state.should_stop is True

    def test_every_new_best_resets_the_counter(self, tmp_path: Path) -> None:
        # four epochs without a new best, then one: the count starts again from there
        method = _ScriptedMethod([0.50, 0.4, 0.4, 0.4, 0.4, 0.60, 0.5, 0.5, 0.5, 0.5, 0.5, 0.99])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1, early_stopping_patience=5)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.best_epoch == 4
        assert fit.state.epoch + 1 == 10

    def test_equalling_the_best_is_not_surpassing_it(self, tmp_path: Path) -> None:
        method = _ScriptedMethod([0.50, 0.50])
        trainer = _trainer(tmp_path, method, validate_every_epochs=1, early_stopping_patience=5)
        fit = trainer.fit(_loader(), _loader(1))
        assert fit.state.best_epoch == -1 and fit.state.epoch + 1 == 5
