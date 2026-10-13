"""Resuming an optimizer-boundary checkpoint reproduces stochastic training."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.training import TrainerConfig


SPEC = importlib.util.spec_from_file_location(
    "exp7_resumable_trainer", Path(__file__).resolve().parents[1]
    / "scripts/exp7-ce-distillation/resumable_trainer.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EpochSeededTrainer = MODULE.EpochSeededTrainer


class Rows(Dataset):
    def __len__(self):
        return 23

    def __getitem__(self, index):
        return {"x": torch.tensor([index / 23., (index % 5) / 5.]), "row": index}


class Model(FlowRecModel):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(2, 7), torch.nn.Dropout(.4), torch.nn.Linear(7, 1))

    def forward(self, batch):
        return RecModelOutput(scores=self.network(batch["x"]))

    def config_dict(self):
        return {}


class Loss(FlowRecMethod):
    def __init__(self):
        super().__init__()
        self.seen = []

    def compute_loss(self, model, batch, *, stage="train", step=None):
        if stage == "train":
            self.seen.append((step, tuple(batch["row"].tolist())))
        noise = .5 + random.random() + float(np.random.random())
        loss = (model(batch).scores.square().mean() + model(batch).scores.mean() * .1) * noise
        return MethodOutput(loss=loss)


class Interrupted(RuntimeError):
    pass


class InterruptingTrainer(EpochSeededTrainer):
    interrupt_at = None

    def save_checkpoint(self, filename):
        result = super().save_checkpoint(filename)
        if self._in_epoch and self.state.optimizer_step == self.interrupt_at:
            raise Interrupted("simulated walltime kill immediately after atomic save")
        return result


def make_trainer(folder, initial, *, trainer_type=EpochSeededTrainer, max_batches=None, **config):
    model = Model()
    model.load_state_dict(initial)
    trainer = trainer_type.from_optimizer_config(
        model=model, method=Loss(), device="cpu", seed=37, checkpoint_every_updates=1,
        cfg=TrainerConfig(checkpoint_dir=folder, max_epochs=3, gradient_accumulation_steps=3,
                          max_train_batches=max_batches, log_every_steps=0, **config),
        lr=.01,
    )
    return trainer


def loader(workers=0):
    return DataLoader(Rows(), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(1),
                      num_workers=workers, persistent_workers=workers > 0)


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("max_batches,interrupt_at", [(None, 2), (10, 2), (10, 4), (10, 6)])
def test_mid_epoch_resume_matches_parameters_optimizer_steps_and_rng(tmp_path, max_batches, interrupt_at):
    torch.manual_seed(9)
    initial = copy.deepcopy(Model().state_dict())
    full = make_trainer(tmp_path / "full", initial, max_batches=max_batches)
    full.fit(loader())
    full_rng = MODULE.capture_rng()
    interrupted = make_trainer(tmp_path / "resume", initial, trainer_type=InterruptingTrainer, max_batches=max_batches)
    interrupted.interrupt_at = interrupt_at
    with pytest.raises(Interrupted):
        interrupted.fit(loader())
    checkpoint = tmp_path / "resume/last.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["epoch_progress"]["epoch_complete"] is False
    assert saved["epoch_progress"]["completed_batches"] > 0
    assert all(parameter.grad is None for parameter in interrupted.model.parameters())
    resumed = make_trainer(tmp_path / "resume", initial, max_batches=max_batches)
    payload = resumed.load_checkpoint(checkpoint)
    resumed.restore_progress(payload)
    resumed.fit(loader())
    assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())
    assert_nested_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    assert (resumed.state.global_step, resumed.state.optimizer_step) == (full.state.global_step, full.state.optimizer_step)
    assert interrupted.method.seen + resumed.method.seen == full.method.seen
    assert_nested_equal(MODULE.capture_rng(), full_rng)
    final = torch.load(checkpoint, weights_only=False)
    assert final["epoch_progress"]["epoch_complete"] is True
    assert final["epoch_progress"]["completed_batches"] == (max_batches or len(loader()))


def test_epoch_shuffle_independent_of_worker_seed_rng(tmp_path):
    # Emulate the one-time generator draw when a worker pool is initialized.
    # Keeping this in-process also runs on hosts without multiprocessing SHM.
    class WorkerSeedLoader(DataLoader):
        started = False

        def __iter__(self):
            if not self.started:
                torch.empty((), dtype=torch.int64).random_(generator=self.generator)
                self.started = True
            return super().__iter__()

    def worker_loader():
        return WorkerSeedLoader(Rows(), batch_size=2, shuffle=True, generator=torch.Generator())

    torch.manual_seed(11)
    initial = copy.deepcopy(Model().state_dict())
    full = make_trainer(tmp_path / "full", initial, max_batches=7)
    full.fit(worker_loader())
    interrupted = make_trainer(tmp_path / "resume", initial, trainer_type=InterruptingTrainer, max_batches=7)
    interrupted.interrupt_at = 5  # second epoch, after one worker-pool reuse
    with pytest.raises(Interrupted):
        interrupted.fit(worker_loader())
    resumed = make_trainer(tmp_path / "resume", initial, max_batches=7)
    resumed.restore_progress(resumed.load_checkpoint(tmp_path / "resume/last.pt"))
    resumed.fit(worker_loader())
    assert interrupted.method.seen + resumed.method.seen == full.method.seen
    assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())


@pytest.mark.parametrize("stopped", [False, True])
def test_completed_or_early_stopped_resume_performs_no_updates(tmp_path, stopped):
    initial = Model().state_dict()
    trainer = make_trainer(tmp_path, initial, max_batches=2)
    trainer.fit(loader())
    trainer.state.should_stop = stopped
    if stopped:
        trainer.state.epoch = 0
    trainer.save_checkpoint("last.pt")
    resumed = make_trainer(tmp_path, initial, max_batches=2)
    resumed.restore_progress(resumed.load_checkpoint(tmp_path / "last.pt"))
    before = copy.deepcopy(resumed.model.state_dict())
    step = resumed.state.global_step
    result = resumed.fit(loader())
    assert resumed.state.global_step == step
    assert resumed.method.seen == []
    assert result.state.epoch == (0 if stopped else 2)
    assert_nested_equal(resumed.model.state_dict(), before)


def test_failed_atomic_save_retains_previous_checkpoint(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, Model().state_dict())
    trainer.save_checkpoint("last.pt")
    original = (tmp_path / "last.pt").read_bytes()

    def interrupted_save(payload, stream):
        stream.write(b"incomplete")
        raise OSError("simulated write failure")

    monkeypatch.setattr(MODULE.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="write failure"):
        trainer.save_checkpoint("last.pt")
    assert (tmp_path / "last.pt").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_changed_unfinished_epoch_limit_is_rejected(tmp_path):
    initial = Model().state_dict()
    trainer = make_trainer(tmp_path, initial, trainer_type=InterruptingTrainer)
    trainer.interrupt_at = 1
    with pytest.raises(Interrupted):
        trainer.fit(loader())
    resumed = make_trainer(tmp_path, initial, max_batches=5)
    resumed.restore_progress(resumed.load_checkpoint(tmp_path / "last.pt"))
    with pytest.raises(ValueError, match="length or max_train_batches"):
        resumed.fit(loader())


def test_epoch_boundary_resume_advances_once(tmp_path):
    class AtEpochEnd(EpochSeededTrainer):
        def save_checkpoint(self, filename):
            result = super().save_checkpoint(filename)
            if str(filename) == "last.pt" and self._epoch_complete and self.state.epoch == 0:
                raise Interrupted("after completed epoch")
            return result

    initial = Model().state_dict()
    full = make_trainer(tmp_path / "full", initial, max_batches=5)
    full.fit(loader())
    interrupted = make_trainer(tmp_path / "resume", initial, max_batches=5, trainer_type=AtEpochEnd)
    with pytest.raises(Interrupted):
        interrupted.fit(loader())
    resumed = make_trainer(tmp_path / "resume", initial, max_batches=5)
    payload = resumed.load_checkpoint(tmp_path / "resume/last.pt")
    assert payload["epoch_progress"]["epoch_complete"] is True
    resumed.restore_progress(payload)
    assert resumed.state.epoch == 1
    resumed.fit(loader())
    assert interrupted.method.seen + resumed.method.seen == full.method.seen
    assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())


def test_mid_epoch_resume_preserves_batch_scheduler(tmp_path):
    initial = Model().state_dict()

    def scheduled(folder, trainer_type=EpochSeededTrainer):
        trainer = make_trainer(folder, initial, trainer_type=trainer_type, scheduler_step_on="batch", max_batches=7)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=2, gamma=.8)
        return trainer

    full = scheduled(tmp_path / "full")
    full.fit(loader())
    interrupted = scheduled(tmp_path / "resume", InterruptingTrainer)
    interrupted.interrupt_at = 2
    with pytest.raises(Interrupted):
        interrupted.fit(loader())
    resumed = scheduled(tmp_path / "resume")
    resumed.restore_progress(resumed.load_checkpoint(tmp_path / "resume/last.pt"))
    resumed.fit(loader())
    assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())
    assert_nested_equal(resumed.scheduler.state_dict(), full.scheduler.state_dict())
    assert_nested_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
