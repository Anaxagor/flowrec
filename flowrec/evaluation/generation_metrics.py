"""Generation ranking extras: HitRate aliases and ARP@K.

SFT/DPO/SDPO already generate top-k semantic IDs on val/test. This adapter
adds HitRate@K (alias of single-target Recall@K) and ARP@K from train-only
item popularity without changing the inner method's loss.
"""

from __future__ import annotations

import logging
from typing import Mapping

import torch

from flowrec.data.datamodule import AmazonReviewDataModule
from flowrec.data.yambda import YambdaDataModule
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput

LOGGER = logging.getLogger("flowrec.evaluation.generation_metrics")

__all__ = [
    "GenerationMetricAdapter",
    "build_semantic_popularity_lookup",
    "generation_metric_ks",
    "unwrap_method",
]


class GenerationMetricAdapter(FlowRecMethod):
    """Add HitRate aliases and ARP without changing the generic Trainer."""

    method_name = "generation_metric_adapter"

    def __init__(
        self,
        base_method: FlowRecMethod,
        *,
        popularity_by_semantic_id: Mapping[tuple[int, ...], float],
        ks: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.base_method = base_method
        self._popularity_by_semantic_id = dict(popularity_by_semantic_id)
        self._ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))

    @property
    def cfg(self):
        return self.base_method.cfg

    def compute_loss(
        self,
        model,
        batch,
        *,
        stage: str = "train",
        step: int | None = None,
    ) -> MethodOutput:
        out = self.base_method.compute_loss(model, batch, stage=stage, step=step)
        if stage == "train":
            return out

        generated = out.extra.get("generated_token_ids")
        if not torch.is_tensor(generated):
            return out
        if generated.ndim == 2:
            generated = generated.unsqueeze(1)
        if generated.ndim != 3:
            return out

        prefix = out.extra.get("generation_metric_prefix")
        if not isinstance(prefix, str) or not prefix:
            prefix = "sft"
        prefix = prefix.rstrip("/")

        max_n = int(generated.size(1))
        for k in self._ks:
            kk = int(k)
            if kk <= 0 or kk > max_n:
                continue

            recall_key = f"{prefix}/recall@{k}"
            generation_recall_key = f"{prefix}/generation_recall@{k}"
            if recall_key in out.metrics:
                out.metrics[f"{prefix}/hitrate@{k}"] = out.metrics[recall_key]
            elif generation_recall_key in out.metrics:
                out.metrics[f"{prefix}/hitrate@{k}"] = out.metrics[generation_recall_key]

            out.metrics[f"{prefix}/arp@{k}"] = self._arp(generated[:, :kk, :])

        return out

    def _arp(self, generated: torch.Tensor) -> torch.Tensor:
        rows = generated.detach().cpu().long().reshape(-1, generated.size(-1)).tolist()
        if not rows:
            return generated.new_zeros((), dtype=torch.float32)
        values = [
            float(self._popularity_by_semantic_id.get(tuple(int(x) for x in row), 0.0))
            for row in rows
        ]
        return torch.tensor(
            sum(values) / max(len(values), 1),
            dtype=torch.float32,
            device=generated.device,
        )


def unwrap_method(method: FlowRecMethod) -> FlowRecMethod:
    current = method
    while isinstance(current, GenerationMetricAdapter):
        current = current.base_method
    return current


def generation_metric_ks(config: Mapping[str, object]) -> tuple[int, ...]:
    training = dict(config.get("training", {}) or {})
    ks: list[int] = []
    for key in ("generation_topk", "test_generation_topk"):
        raw = training.get(key)
        if raw is None:
            continue
        ks.extend(int(k) for k in raw)
    if not ks:
        ks = [5, 10, 20]
    return tuple(sorted({k for k in ks if k > 0}))


def build_semantic_popularity_lookup(
    data: AmazonReviewDataModule | YambdaDataModule,
) -> dict[tuple[int, ...], float]:
    """Map semantic-ID tuples to train-only interaction popularity counts."""
    if data.semantic_ids is None:
        return {}
    interactions = getattr(data, "interactions", None)
    if interactions is None or len(interactions) == 0:
        LOGGER.warning("Cannot compute ARP popularity: datamodule has no interactions frame.")
        return {tuple(tokens): 0.0 for tokens in data.semantic_ids.values()}

    frame = interactions
    split_strategy = str(getattr(data.cfg, "split_strategy", "leave_one_out"))
    if split_strategy == "leave_one_out":
        if "position" not in frame.columns or "user_id" not in frame.columns:
            raise ValueError("ARP for leave_one_out requires interactions columns user_id and position.")
        last_position = frame.groupby("user_id")["position"].transform("max")
        train_frame = frame.loc[frame["position"] <= (last_position - 2)]
    elif split_strategy == "global_time":
        cutoff = getattr(data, "global_val_cutoff", None)
        if cutoff is None:
            cutoff = getattr(data.cfg, "global_val_timestamp", None)
        if cutoff is None:
            raise ValueError("ARP for global_time requires a validation cutoff.")
        train_frame = frame.loc[frame["timestamp"] < int(cutoff)]
    else:
        raise ValueError(f"Unsupported split_strategy for ARP: {split_strategy!r}")

    counts = train_frame["raw_item_id"].astype(str).value_counts().to_dict()
    return {
        tuple(int(token) for token in tokens): float(counts.get(str(raw_item), 0))
        for raw_item, tokens in data.semantic_ids.items()
    }
