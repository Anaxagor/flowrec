from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Configuration for supervised fine-tuning.

    SFT owns the supervised objective:

    - TIGER-style outputs: token cross-entropy over semantic-ID tokens.
    - SASRec-style outputs: item cross-entropy over atomic item IDs.

    Models return logits/scores; this method converts them into losses.
    """

    name: str = "sft"
    loss_weight: float = 1.0
    metric_prefix: str = "sft"

    pad_token_label_id: int = -100
    pad_item_id: int = 0

    compute_train_metrics: bool = True
    compute_eval_metrics: bool = True

    item_topk: tuple[int, ...] = (5, 10, 20)
    compute_token_metrics: bool = True
    compute_item_metrics: bool = True

    evaluate_generation: bool = False
    generation_topk: tuple[int, ...] = (5, 10)
    generation_num_beams: int | None = None
    generation_num_return_sequences: int | None = None
    generation_constrained: bool | None = None

    include_model_metrics: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SFTConfig":
        if data is None:
            return cls()
        normalized = dict(data)
        for key in ("item_topk", "generation_topk"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = tuple(int(x) for x in normalized[key])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loss_weight": self.loss_weight,
            "metric_prefix": self.metric_prefix,
            "pad_token_label_id": self.pad_token_label_id,
            "pad_item_id": self.pad_item_id,
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
