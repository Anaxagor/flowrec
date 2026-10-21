"""ARP / HitRate extras on generated semantic IDs."""

from types import SimpleNamespace

import pandas as pd
import torch

from flowrec.evaluation.generation_metrics import (
    GenerationMetricAdapter,
    build_semantic_popularity_lookup,
    generation_metric_ks,
    unwrap_method,
)
from flowrec.ft_methods.base import MethodOutput


class _Inner(torch.nn.Module):
    method_name = "inner"

    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(name="inner")

    def compute_loss(self, model, batch, *, stage="train", step=None):
        generated = torch.tensor([[[1, 1], [2, 2], [3, 3]]], dtype=torch.long)
        return MethodOutput(
            loss=torch.zeros(()),
            metrics={"sft/recall@2": torch.tensor(1.0), "sft/generation_recall@3": torch.tensor(0.5)},
            extra={"generated_token_ids": generated, "generation_metric_prefix": "sft"},
        )


def test_adapter_adds_hitrate_and_arp_on_eval():
    popularity = {(1, 1): 10.0, (2, 2): 20.0, (3, 3): 30.0}
    method = GenerationMetricAdapter(_Inner(), popularity_by_semantic_id=popularity, ks=(2, 3))
    out = method.compute_loss(None, {}, stage="val")
    assert float(out.metrics["sft/hitrate@2"]) == 1.0
    assert float(out.metrics["sft/hitrate@3"]) == 0.5
    assert float(out.metrics["sft/arp@2"]) == 15.0
    assert float(out.metrics["sft/arp@3"]) == 20.0


def test_adapter_skips_k_larger_than_generated_n():
    popularity = {(1, 1): 10.0, (2, 2): 20.0, (3, 3): 30.0}
    method = GenerationMetricAdapter(_Inner(), popularity_by_semantic_id=popularity, ks=(2, 20))
    out = method.compute_loss(None, {}, stage="val")
    assert "sft/arp@2" in out.metrics
    assert "sft/arp@20" not in out.metrics


def test_adapter_skips_arp_on_train():
    method = GenerationMetricAdapter(_Inner(), popularity_by_semantic_id={(1, 1): 1.0}, ks=(2,))
    out = method.compute_loss(None, {}, stage="train")
    assert "sft/arp@2" not in out.metrics


def test_leave_one_out_popularity_excludes_last_two_positions():
    data = SimpleNamespace(
        semantic_ids={"a": (1, 1), "b": (2, 2)},
        cfg=SimpleNamespace(split_strategy="leave_one_out"),
        interactions=pd.DataFrame(
            {
                "user_id": [0, 0, 0, 1, 1, 1],
                "position": [0, 1, 2, 0, 1, 2],
                "raw_item_id": ["a", "a", "b", "b", "b", "a"],
            }
        ),
    )
    lookup = build_semantic_popularity_lookup(data)
    assert lookup[(1, 1)] == 1.0
    assert lookup[(2, 2)] == 1.0


def test_generation_metric_ks_unions_val_and_test():
    ks = generation_metric_ks(
        {"training": {"generation_topk": [10], "test_generation_topk": [5, 10, 20]}}
    )
    assert ks == (5, 10, 20)


def test_unwrap_adapter():
    inner = _Inner()
    wrapped = GenerationMetricAdapter(inner, popularity_by_semantic_id={}, ks=(5,))
    assert unwrap_method(wrapped) is inner
    assert unwrap_method(inner) is inner
