"""Sequential two-variant calibration matches the held-graph weight."""
from __future__ import annotations

import torch
from torch import nn

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY, MethodOutput
from flowrec.ft_methods.calibration import (
    calibrate_component_weight,
    calibrate_from_loss_pairs,
    calibrate_sequential_variants,
    variant_loss_pairs,
)


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.5, -0.25]))


class _ScaledLoss(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def compute_loss(self, model, batch, *, stage, step):
        del stage, step
        noise = torch.randn(())
        loss = self.scale * ((model.w * batch["x"]).sum() + noise)
        return MethodOutput(loss=loss)


class _SharedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def compute_loss(self, model, batch, *, stage, step):
        del stage, step
        self.calls += 1
        reference = (model.w * batch["x"]).sum()
        component = (model.w * batch["x"]).square().sum()
        return MethodOutput(
            loss=reference + component,
            extra={LOSS_COMPONENTS_KEY: {"sft": reference, "aux": component}},
        )


def test_sequential_variants_match_held_graphs_and_leave_grad_empty():
    model = _Linear()
    batches = [{"x": torch.tensor([1.0, 2.0])}, {"x": torch.tensor([-1.0, 0.5])}]
    params = list(model.parameters())
    held = calibrate_from_loss_pairs(
        variant_loss_pairs(
            model,
            _ScaledLoss(1.0),
            _ScaledLoss(3.0),
            batches,
            seed=11,
        ),
        params,
        target_ratio=1.0,
        reference_name="sft",
        component_name="aux",
    )
    sequential = calibrate_sequential_variants(
        model,
        _ScaledLoss(1.0),
        _ScaledLoss(3.0),
        batches,
        params,
        target_ratio=1.0,
        seed=11,
        reference_name="sft",
        component_name="aux",
    )
    assert sequential["weight"] == held["weight"]
    assert sequential["per_batch"]["ratios"] == held["per_batch"]["ratios"]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_shared_forward_calibration_still_uses_one_sample():
    model = _Linear()
    method = _SharedForward()
    result = calibrate_component_weight(
        model,
        method,
        [{"x": torch.tensor([1.0, 2.0])}, {"x": torch.tensor([0.5, -1.0])}],
        component="aux",
        reference="sft",
        num_batches=2,
        seed=3,
    )
    assert method.calls == 2
    assert result["num_batches"] == 2
    assert result["weight"] > 0
    assert all(parameter.grad is None for parameter in model.parameters())
