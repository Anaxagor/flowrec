from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from flowrec.methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.methods.registry import METHOD_REGISTRY
from flowrec.methods.sft.batch_builder import SFTBatchBuilder
from flowrec.methods.sft.config import SFTConfig
from flowrec.methods.sft.metrics import (
    generation_recall_metrics,
    item_ranking_metrics,
    token_prediction_metrics,
)
from flowrec.models.base import Batch, FlowRecModel, RecModelOutput


@METHOD_REGISTRY.register("sft")
@dataclass(slots=True)
class SFTMethod(FlowRecMethod):
    """Supervised fine-tuning method.

    This method is intentionally model-agnostic.  It uses the model's default
    supervised loss:

    - ``TigerModel``: next semantic-token prediction loss.
    - ``SASRecModel``: next atomic-item prediction loss.

    The method adds:
      1. consistent loss scaling,
      2. consistent metric logging,
      3. optional generation metrics for TIGER-like models.
    """

    cfg: SFTConfig = field(default_factory=SFTConfig)
    batch_builder: SFTBatchBuilder = field(default_factory=SFTBatchBuilder)

    method_name: str = "sft"

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        method_batch = self.batch_builder.build(batch, stage=stage, step=step)

        model_output = model(method_batch)  # type: ignore[misc]
        if not isinstance(model_output, RecModelOutput):
            raise TypeError(
                f"{model.__class__.__name__}.forward must return RecModelOutput, "
                f"got {type(model_output)!r}."
            )
        if model_output.loss is None:
            # Fall back to model.compute_loss in case a model computes the loss
            # outside forward.  This may run a second forward pass, so standard
            # FlowRec models should prefer output.loss.
            base_loss = model.compute_loss(method_batch)
        else:
            base_loss = model_output.loss

        loss = base_loss * float(self.cfg.loss_weight)

        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "base_loss": base_loss.detach(),
        }

        if self.cfg.include_model_metrics and model_output.metrics:
            for key, value in model_output.metrics.items():
                metrics[f"model_{key}"] = value.detach() if torch.is_tensor(value) else value

        if self._should_compute_metrics(stage):
            metrics.update(self._compute_generic_metrics(model, method_batch, model_output, stage=stage))

        return MethodOutput(
            loss=loss,
            metrics=prefix_metrics(metrics, prefix=self.cfg.metric_prefix),
            model_output=model_output,
            extra={"stage": stage, "step": step, "method_batch": method_batch},
        )

    def _should_compute_metrics(self, stage: str) -> bool:
        if stage == "train":
            return self.cfg.compute_train_metrics
        return self.cfg.compute_eval_metrics

    def _compute_generic_metrics(
        self,
        model: FlowRecModel,
        batch: Batch,
        model_output: RecModelOutput,
        *,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}

        if (
            self.cfg.compute_item_metrics
            and model_output.scores is not None
            and "target_item_ids" in batch
            and torch.is_tensor(batch["target_item_ids"])
        ):
            scores = model_output.scores
            if scores.ndim == 2:
                metrics.update(
                    item_ranking_metrics(
                        scores=scores,
                        target_item_ids=batch["target_item_ids"],
                        ks=self.cfg.item_topk,
                    )
                )

        if (
            self.cfg.compute_token_metrics
            and model_output.logits is not None
            and "target_token_ids" in batch
            and torch.is_tensor(batch["target_token_ids"])
        ):
            logits = model_output.logits
            # Token logits for TIGER are [B, L, V].  SASRec uses [B, V] item
            # logits/scores, which are handled above.
            if logits.ndim == 3:
                metrics.update(
                    token_prediction_metrics(
                        logits=logits,
                        target_token_ids=batch["target_token_ids"],
                        target_token_mask=batch.get("target_token_mask")
                        if torch.is_tensor(batch.get("target_token_mask"))
                        else None,
                    )
                )

        if self.cfg.evaluate_generation and stage != "train":
            metrics.update(self._compute_generation_metrics(model, batch))

        return metrics

    @torch.no_grad()
    def _compute_generation_metrics(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        if "target_token_ids" not in batch or not torch.is_tensor(batch["target_token_ids"]):
            return {}

        try:
            generated = model.generate(
                batch,
                num_beams=self.cfg.generation_num_beams,
                num_return_sequences=self.cfg.generation_num_return_sequences,
                constrained=self.cfg.generation_constrained,
            )
        except NotImplementedError:
            return {}

        return generation_recall_metrics(
            generated_token_ids=generated,
            target_token_ids=batch["target_token_ids"],
            target_token_mask=batch.get("target_token_mask")
            if torch.is_tensor(batch.get("target_token_mask"))
            else None,
            ks=self.cfg.generation_topk,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SFTMethod":
        return cls(cfg=SFTConfig.from_dict(data))
