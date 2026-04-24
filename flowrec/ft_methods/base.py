from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from flowrec.models.base import Batch, FlowRecModel, RecModelOutput


MetricValue = torch.Tensor | float | int


@dataclass(slots=True)
class MethodOutput:
    """Output returned by a FlowRec fine-tuning method.

    Fine-tuning methods own the training objective, not the backbone.  A method
    receives a model and a batch, computes the objective for that method, and
    returns this common object.

    Attributes:
        loss:
            Scalar loss used for ``backward()``.
        metrics:
            Scalars useful for logging.  Tensor metrics are expected to be
            detached unless the caller explicitly needs gradients.
        model_output:
            Optional raw model output from the underlying ``FlowRecModel``.
        extra:
            Optional method-specific artifacts, for example sampled candidates
            or per-token losses.
    """

    loss: torch.Tensor
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    model_output: RecModelOutput | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def detached_metrics(self) -> dict[str, float]:
        """Return metrics as plain Python floats when possible."""

        out: dict[str, float] = {}
        for key, value in self.metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                out[key] = float(value.detach().cpu().item())
            else:
                out[key] = float(value)
        return out


class FlowRecMethod(ABC):
    """Base class for all FlowRec fine-tuning methods.

    The method abstraction is deliberately lightweight.  It is not an
    ``nn.Module`` because most methods do not own trainable parameters; they
    define how to use a ``FlowRecModel``.  Future methods may still hold state
    such as a reference model, replay buffer, or adaptive coefficient.
    """

    method_name: str = "base"

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
        return self.compute_loss(model, batch, stage="train", step=step)

    @torch.no_grad()
    def validation_step(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None = None,
    ) -> MethodOutput:
        was_training = model.training
        model.eval()
        try:
            return self.compute_loss(model, batch, stage="val", step=step)
        finally:
            model.train(was_training)

    @torch.no_grad()
    def test_step(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        step: int | None = None,
    ) -> MethodOutput:
        was_training = model.training
        model.eval()
        try:
            return self.compute_loss(model, batch, stage="test", step=step)
        finally:
            model.train(was_training)


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
