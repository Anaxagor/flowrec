from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn

from flowrec.models.base import Batch, FlowRecModel, RecModelOutput


MetricValue = torch.Tensor | float | int


@dataclass(slots=True)
class MethodOutput:
    """Output returned by a FlowRec fine-tuning method.

    Fine-tuning methods own objectives. They receive a model and a batch, call
    model capabilities such as ``forward``/``token_logprobs``/``score_items``,
    compute a scalar objective, and return this common object.
    """

    loss: torch.Tensor
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    model_output: RecModelOutput | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def detached_metrics(self) -> dict[str, float]:
        """Return scalar metrics as Python floats when possible."""
        out: dict[str, float] = {}
        for key, value in self.metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                out[key] = float(value.detach().cpu().item())
            else:
                out[key] = float(value)
        return out


class FlowRecMethod(nn.Module, ABC):
    """Base class for all FlowRec fine-tuning methods.

    Methods, not models, own the loss. This inherits from ``nn.Module`` so
    future methods can own trainable parameters such as GFlowNet ``logZ``, flow
    heads, value heads, or learned reward adapters.
    """

    method_name: str = "base"

    def __init__(self) -> None:
        super().__init__()

    def build_batch(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> Batch:
        """Optionally transform a datamodule batch into a method-specific batch."""
        return batch

    @abstractmethod
    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        """Compute this method's objective for a model and batch."""

    def training_step(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None = None,
    ) -> MethodOutput:
        self.train(True)
        return self.compute_loss(model, batch, stage="train", step=step)

    @torch.no_grad()
    def validation_step(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None = None,
    ) -> MethodOutput:
        was_model_training = model.training
        was_method_training = self.training
        model.eval()
        self.eval()
        try:
            return self.compute_loss(model, batch, stage="val", step=step)
        finally:
            model.train(was_model_training)
            self.train(was_method_training)

    @torch.no_grad()
    def test_step(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None = None,
    ) -> MethodOutput:
        was_model_training = model.training
        was_method_training = self.training
        model.eval()
        self.eval()
        try:
            return self.compute_loss(model, batch, stage="test", step=step)
        finally:
            model.train(was_model_training)
            self.train(was_method_training)


def prefix_metrics(
    metrics: Mapping[str, MetricValue],
    *,
    prefix: str,
    skip_existing_prefix: bool = True,
) -> dict[str, MetricValue]:
    """Add a logging prefix to metric names."""

    out: dict[str, MetricValue] = {}
    normalized_prefix = prefix.rstrip("/")
    for key, value in metrics.items():
        if skip_existing_prefix and key.startswith(normalized_prefix + "/"):
            out[key] = value
        else:
            out[f"{normalized_prefix}/{key}"] = value
    return out
