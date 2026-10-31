"""An exp8 stage takes source weights, never its optimization progress."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "scripts/exp2-tuned-params",
                  ROOT / "scripts/exp8-from-sft"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import stage_training as stage
from resumable_trainer import EpochSeededTrainer
from sft_finetune_control import load_tiger_checkpoint
from gflow_finetune_gpt2_beauty import load_gpt2_checkpoint
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.calibration import calibrate_component_weight
from flowrec.training import TrainerConfig


class TinyModel(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))


@pytest.fixture(params=[load_tiger_checkpoint, load_gpt2_checkpoint], ids=["tiger", "gpt2"])
def source(request, tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1)
    for _ in range(3):
        optimizer.zero_grad()
        (model.weight - 2).square().backward()
        optimizer.step()
        scheduler.step()
    payload = {"model_state_dict": deepcopy(model.state_dict()),
               "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(),
               "trainer_state": {"epoch": 9, "global_step": 99, "optimizer_step": 33},
               "method_state_dict": {"old_auxiliary_parameter": torch.tensor(999.)}}
    assert payload["optimizer_state_dict"]["state"]
    assert payload["scheduler_state_dict"]["last_epoch"] == 3
    assert optimizer.param_groups[0]["lr"] != 3e-5
    checkpoint = tmp_path / "source.pt"
    torch.save(payload, checkpoint)
    return request.param, checkpoint, payload


@pytest.mark.parametrize("arm", ["sft", "gflow", "ce"])
def test_source_weights_start_with_fresh_adamw_and_no_inherited_schedule(source, arm):
    load, checkpoint, source_payload = source
    model = TinyModel(-10.)
    load(model, checkpoint)
    torch.testing.assert_close(model.weight, source_payload["model_state_dict"]["weight"], rtol=0, atol=0)
    trainer_type = (EpochSeededTrainer if arm == "ce" else
                    stage.trainer_class(SimpleNamespace(seed=42, checkpoint_every_updates=100)))
    # The real exp8 factories receive no scheduler because the selected SFT
    # sources used constant LR. A decayed schedule in a payload must not leak.
    trainer = trainer_type.from_optimizer_config(
        model=model, method=torch.nn.Identity(),
        cfg=TrainerConfig(use_amp=False), device="cpu", lr=3e-5, weight_decay=.01)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert not trainer.optimizer.state
    assert {group["lr"] for group in trainer.optimizer.param_groups} == {3e-5}
    assert trainer.scheduler is None and trainer.cfg.scheduler_step_on == "none"
    assert (trainer.state.epoch, trainer.state.global_step, trainer.state.optimizer_step) == (0, 0, 0)
    # Check the first actual update too: stale Adam moments or a stale step
    # counter would produce a different result from a fresh reference AdamW.
    reference = TinyModel(float(model.weight.detach()))
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-5, weight_decay=.01)
    for candidate, optimizer in ((model, trainer.optimizer), (reference, expected_optimizer)):
        (candidate.weight + 4).square().backward()
        optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    assert all(float(state["step"]) == 1 for state in trainer.optimizer.state.values())


@dataclass
class SamplingConfig:
    sampling: str


class CalibrationMethod(torch.nn.Module):
    """Two different gradients make source-weight changes observable."""
    def __init__(self, sampling):
        super().__init__()
        self.cfg = SamplingConfig(sampling)

    def compute_loss(self, model, _batch, **_kwargs):
        reference = (model.weight - 4).square()
        auxiliary = (model.weight + 1).square()
        return MethodOutput(reference + auxiliary,
                            extra={"loss_components": {"sft": reference, "distill": auxiliary}})


@pytest.mark.parametrize("sampling", ["reward", "curriculum_smooth_is"])
def test_new_checkpoint_calibration_uses_its_gradients_and_requested_rho(source, sampling, tmp_path):
    load, original_checkpoint, _ = source
    changed = TinyModel(2.)
    changed_checkpoint = tmp_path / "new_sft_weights.pt"
    torch.save({"model_state_dict": changed.state_dict()}, changed_checkpoint)
    method = CalibrationMethod(sampling)

    def calibrate(checkpoint, rho):
        model = TinyModel(-10.)
        load(model, checkpoint)
        before = model.weight.detach().clone()
        result = stage.ce.tiger.calibrate_distillation_weight(
            calibrate_component_weight, model, method, [{}], component="distill",
            target_ratio=rho, num_batches=1, seed=1234, device="cpu")
        assert method.cfg.sampling == sampling
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        assert model.weight.grad is None
        assert result["effective_ratio_check"] == pytest.approx(rho)
        return result["weight"]

    old_lambda = calibrate(original_checkpoint, .1)
    new_lambda = calibrate(changed_checkpoint, .1)
    assert new_lambda != pytest.approx(old_lambda)
    assert calibrate(changed_checkpoint, 1.) == pytest.approx(10 * new_lambda)
