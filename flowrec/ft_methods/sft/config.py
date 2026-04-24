from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Configuration for supervised fine-tuning.

    SFT is the default training objective for both types of FlowRec backbone:

    - TIGER-style models: next semantic-token prediction.
    - SASRec-style models: next atomic-item prediction.

    The method delegates the core supervised loss to ``model.forward(batch)`` or
    ``model.compute_loss(batch)``.  This keeps model-specific loss details inside
    the model while the method handles scaling, logging, and generic metrics.
    """

    name: str = "sft"
    loss_weight: float = 1.0

    # Logging prefix.  Final metrics look like ``sft/loss`` or
    # ``sft/token_accuracy``.
    metric_prefix: str = "sft"

    # Whether to compute generic metrics.  Training metrics are cheap and useful
    # but can be disabled for very large runs.
    compute_train_metrics: bool = True
    compute_eval_metrics: bool = True

    # Item-ranking metrics are computed when the model output contains
    # ``scores`` with shape [B, num_items] and the batch contains target_item_ids.
    item_topk: tuple[int, ...] = (5, 10, 20)

    # Token metrics are computed when the model output contains token logits
    # with shape [B, L, vocab_size] and the batch contains target_token_ids.
    compute_token_metrics: bool = True
    compute_item_metrics: bool = True

    # Optional generation metrics for TIGER-like models.  This calls
    # model.generate(), so it should usually be used only for validation/test.
    evaluate_generation: bool = False
    generation_topk: tuple[int, ...] = (5, 10)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrained: bool | None = None

    # Whether to include model-provided metrics from RecModelOutput.metrics.
    include_model_metrics: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SFTConfig":
        if data is None:
            return cls()

        normalized = dict(data)

        # YAML often loads lists; the dataclass stores tuples for immutability.
        for key in ("item_topk", "generation_topk"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = tuple(int(x) for x in normalized[key])

        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loss_weight": self.loss_weight,
            "metric_prefix": self.metric_prefix,
            "compute_train_metrics": self.compute_train_metrics,
            "compute_eval_metrics": self.compute_eval_metrics,
            "item_topk": list(self.item_topk),
            "compute_token_metrics": self.compute_token_metrics,
            "compute_item_metrics": self.compute_item_metrics,
            "evaluate_generation": self.evaluate_generation,
            "generation_topk": list(self.generation_topk),
            "generation_num_beams": self.generation_num_beams,
            "generation_num_return_sequences": self.generation_num_return_sequences,
            "generation_constrained": self.generation_constrained,
            "include_model_metrics": self.include_model_metrics,
        }
