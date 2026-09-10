from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from flowrec.ft_methods.base import FlowRecMethod, MethodOutput, prefix_metrics
from flowrec.ft_methods.sft.batch_builder import SFTBatchBuilder
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.scalable_cross_entropy import ScalableCrossEntropyLoss
from flowrec.ft_methods.sft.metrics import (
    generation_recall_metrics,
    item_ranking_metrics,
    token_prediction_metrics,
)
from flowrec.models.base import Batch, FlowRecModel, RecModelOutput


class SFTMethod(FlowRecMethod):
    """Supervised fine-tuning method.

    The model only returns outputs. This method owns the supervised loss:

    - if the batch carries ``label_item_ids`` (dense-causal layout) and
      ``model_output.logits`` has shape [B, T, num_items], compute per-position
      next-item CE masked by ``label_mask``;
    - if ``model_output.logits`` has shape [B, L, V], compute semantic-token CE;
    - if ``model_output.scores`` or 2-D ``logits`` has shape [B, num_items],
      compute next-item CE.
    """

    method_name = "sft"

    def __init__(
        self,
        cfg: SFTConfig | None = None,
        batch_builder: SFTBatchBuilder | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SFTConfig()
        self.batch_builder = batch_builder or SFTBatchBuilder()
        self.scalable_cross_entropy = ScalableCrossEntropyLoss(
            self.cfg.scalable_cross_entropy,
            pad_item_id=self.cfg.pad_item_id,
        )

    def compute_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        method_batch = self.batch_builder.build(batch, stage=stage, step=step)

        if self._uses_scalable_dense_item_loss(model, method_batch, stage=stage):
            model_output, base_loss = self._compute_scalable_dense_item_loss(
                model,
                method_batch,
            )
        else:
            model_output = model(method_batch)  # type: ignore[misc]
            if not isinstance(model_output, RecModelOutput):
                raise TypeError(
                    f"{model.__class__.__name__}.forward must return RecModelOutput, "
                    f"got {type(model_output)!r}."
                )

            base_loss = self._compute_supervised_loss(
                method_batch,
                model_output,
                dense_supervision=model.dense_supervision,
            )
        loss = base_loss * float(self.cfg.loss_weight)

        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "base_loss": base_loss.detach(),
        }

        if self.cfg.include_model_metrics and model_output.metrics:
            for key, value in model_output.metrics.items():
                metrics[f"model_{key}"] = value.detach() if torch.is_tensor(value) else value

        extra: dict[str, Any] = {"stage": stage, "step": step, "method_batch": method_batch}

        if self._should_compute_metrics(stage):
            metrics.update(
                self._compute_generic_metrics(
                    model,
                    method_batch,
                    model_output,
                    stage=stage,
                    extra=extra,
                )
            )

        return MethodOutput(
            loss=loss,
            metrics=prefix_metrics(metrics, prefix=self.cfg.metric_prefix),
            model_output=model_output,
            extra=extra,
        )

    def _uses_scalable_dense_item_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        stage: str,
    ) -> bool:
        # SCE is a stochastic training-time approximation of the full item
        # softmax. Validation/test keep the exact full-catalog loss so reported
        # losses stay comparable across item_loss settings and metrics that
        # need logits are never silently skipped.
        return (
            stage == "train"
            and self.cfg.item_loss == "scalable_cross_entropy"
            and model.dense_supervision == "items"
            and torch.is_tensor(batch.get("input_item_ids"))
            and torch.is_tensor(batch.get("label_item_ids"))
        )

    def _compute_scalable_dense_item_loss(
        self,
        model: FlowRecModel,
        batch: Batch,
    ) -> tuple[RecModelOutput, torch.Tensor]:
        encode = getattr(model, "dense_item_hidden_states", None)
        output_weight = getattr(model, "item_output_embedding_weight", None)
        if not callable(encode) or not callable(output_weight):
            raise TypeError(
                "scalable_cross_entropy for dense item SFT requires the model to expose "
                "dense_item_hidden_states(batch) and item_output_embedding_weight()."
            )

        hidden_states = encode(batch)
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            raise ValueError(
                "dense_item_hidden_states(batch) must return [B, T, D] tensor."
            )
        item_weight = output_weight()
        if not torch.is_tensor(item_weight) or item_weight.ndim != 2:
            raise ValueError(
                "item_output_embedding_weight() must return [V, D] tensor."
            )

        labels = batch["label_item_ids"]
        label_mask = (
            batch.get("label_mask")
            if torch.is_tensor(batch.get("label_mask"))
            else None
        )
        loss = self.scalable_cross_entropy(
            hidden_states,
            labels,
            item_weight,
            target_mask=label_mask,
        )
        model_output = RecModelOutput(
            extra={"dense_item_hidden_states": hidden_states}
        )
        return model_output, loss

    def _compute_supervised_loss(
        self,
        batch: Batch,
        model_output: RecModelOutput,
        *,
        dense_supervision: str,
    ) -> torch.Tensor:
        """Compute SFT loss from model outputs and batch targets."""

        if (
            model_output.logits is not None
            and model_output.logits.ndim == 3
            and torch.is_tensor(batch.get("label_item_ids"))
        ):
            if dense_supervision == "tokens":
                label_token_ids = batch.get("label_token_ids")
                if not torch.is_tensor(label_token_ids):
                    raise KeyError(
                        "Dense token supervision requires "
                        "batch['label_token_ids']."
                    )

                return self._token_cross_entropy(
                    logits=model_output.logits,
                    target_token_ids=label_token_ids,
                    target_token_mask=(
                        batch.get("token_label_mask")
                        if torch.is_tensor(batch.get("token_label_mask"))
                        else None
                    ),
                )

            if dense_supervision != "items":
                raise ValueError(
                    "Unknown dense supervision mode "
                    f"{dense_supervision!r}. Expected 'items' or 'tokens'."
                )

            return self._token_cross_entropy(
                logits=model_output.logits,
                target_token_ids=batch["label_item_ids"],
                target_token_mask=(
                    batch.get("label_mask")
                    if torch.is_tensor(batch.get("label_mask"))
                    else None
                ),
            )

        if model_output.logits is not None and model_output.logits.ndim == 3:
            target_token_ids = batch.get("target_token_ids")
            if not torch.is_tensor(target_token_ids):
                raise KeyError("Token SFT requires batch['target_token_ids'].")

            return self._token_cross_entropy(
                logits=model_output.logits,
                target_token_ids=target_token_ids,
                target_token_mask=(
                    batch.get("target_token_mask")
                    if torch.is_tensor(batch.get("target_token_mask"))
                    else None
                ),
            )

        scores = model_output.scores
        if (
            scores is None
            and model_output.logits is not None
            and model_output.logits.ndim == 2
        ):
            scores = model_output.logits

        if scores is not None and scores.ndim == 2:
            target_item_ids = batch.get("target_item_ids")
            if not torch.is_tensor(target_item_ids):
                raise KeyError("Item SFT requires batch['target_item_ids'].")

            return self._item_cross_entropy(
                scores=scores,
                target_item_ids=target_item_ids,
            )

        raise ValueError(
            "Could not infer SFT loss from model output. Expected either "
            "dense logits [B, T, V] with model-declared supervision, "
            "token logits [B, L, V] with target_token_ids, or item scores "
            "[B, num_items] with target_item_ids."
        )

    def _token_cross_entropy(
        self,
        *,
        logits: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("Token logits must have shape [B, L, V].")
        target = target_token_ids.to(logits.device).long()
        if target.ndim != 2 or target.shape != logits.shape[:2]:
            raise ValueError(
                "target_token_ids must have shape [B, L] matching logits[:2]. "
                f"Got logits={tuple(logits.shape)}, target={tuple(target.shape)}."
            )
        if target_token_mask is not None:
            mask = target_token_mask.to(logits.device).bool()
            if mask.shape != target.shape:
                raise ValueError("target_token_mask must have same shape as target_token_ids.")
            target = target.masked_fill(~mask, self.cfg.pad_token_label_id)

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.cfg.pad_token_label_id,
        )

    def _item_cross_entropy(
        self,
        *,
        scores: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("Item scores must have shape [B, num_items].")
        target = target_item_ids.to(scores.device).long()
        if target.ndim != 1 or target.size(0) != scores.size(0):
            raise ValueError("target_item_ids must have shape [B].")
        return F.cross_entropy(scores, target, ignore_index=self.cfg.pad_item_id)

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
        extra: dict[str, Any],
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
                        pad_item_id=self.cfg.pad_item_id,
                    )
                )

        if (
            self.cfg.compute_token_metrics
            and model_output.logits is not None
            and "target_token_ids" in batch
            and torch.is_tensor(batch["target_token_ids"])
        ):
            logits = model_output.logits
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
            metrics.update(self._compute_generation_metrics(model, batch, extra=extra))

        return metrics

    @torch.no_grad()
    def _compute_generation_metrics(
        self,
        model: FlowRecModel,
        batch: Batch,
        *,
        extra: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if "target_token_ids" not in batch or not torch.is_tensor(batch["target_token_ids"]):
            return {}

        try:
            generated = model.generate(
                batch,
                num_beams=self.cfg.generation_num_beams,
                num_return_sequences=self.cfg.generation_num_return_sequences,
                constrain_valid_sequences=self.cfg.generation_constrain_valid_sequences,
                exclude_seen_items=self.cfg.generation_exclude_seen_items,
            )
        except NotImplementedError:
            return {}

        extra["generated_token_ids"] = generated.detach()
        extra["generation_topk"] = tuple(int(k) for k in self.cfg.generation_topk)
        extra["generation_metric_prefix"] = self.cfg.metric_prefix

        metrics = generation_recall_metrics(
            generated_token_ids=generated,
            target_token_ids=batch["target_token_ids"],
            target_token_mask=batch.get("target_token_mask")
            if torch.is_tensor(batch.get("target_token_mask"))
            else None,
            ks=self.cfg.generation_topk,
        )
        for key, value in list(metrics.items()):
            if key.startswith("generation_"):
                metrics.setdefault(key.removeprefix("generation_"), value)
        return metrics

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SFTMethod":
        return cls(cfg=SFTConfig.from_dict(data))
