from __future__ import annotations

import math
from typing import Any

import torch


def _to_float_metric(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def token_prediction_metrics(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    target_token_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute token-level metrics for autoregressive semantic-ID prediction.

    Args:
        logits:
            Tensor with shape ``[B, L, V]``.
        target_token_ids:
            Tensor with shape ``[B, L]``.
        target_token_mask:
            Optional boolean tensor with shape ``[B, L]``.

    Returns:
        Metrics containing token accuracy and exact sequence match.
    """

    if logits.ndim != 3:
        raise ValueError("token logits must have shape [B, L, V]")
    if target_token_ids.ndim != 2:
        raise ValueError("target_token_ids must have shape [B, L]")
    if logits.shape[:2] != target_token_ids.shape:
        raise ValueError(
            "logits and target_token_ids disagree on batch/sequence dimensions: "
            f"{tuple(logits.shape)} vs {tuple(target_token_ids.shape)}"
        )

    target_token_ids = target_token_ids.to(logits.device).long()
    pred = logits.argmax(dim=-1)

    if target_token_mask is None:
        mask = torch.ones_like(target_token_ids, dtype=torch.bool, device=logits.device)
    else:
        mask = target_token_mask.to(logits.device).bool()

    valid_count = mask.sum().clamp_min(1)
    correct = (pred == target_token_ids) & mask
    token_acc = correct.sum().float() / valid_count.float()

    # Exact match ignores padded token positions.
    per_sequence_correct = (correct | ~mask).all(dim=-1)
    sequence_has_target = mask.any(dim=-1)
    exact_match = per_sequence_correct[sequence_has_target].float().mean()
    if torch.isnan(exact_match):
        exact_match = torch.zeros((), dtype=torch.float32, device=logits.device)

    return {
        "token_accuracy": _to_float_metric(token_acc),
        "sequence_exact_match": _to_float_metric(exact_match),
    }


def item_ranking_metrics(
    scores: torch.Tensor,
    target_item_ids: torch.Tensor,
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    pad_item_id: int = 0,
    exclude_invalid_targets: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute Recall@K and NDCG@K from full item scores.

    Args:
        scores:
            Tensor with shape ``[B, num_items]``.
        target_item_ids:
            Tensor with shape ``[B]``.
        ks:
            K values for ranking metrics.
        pad_item_id:
            Padding item ID. Targets equal to this value are ignored by default.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [B, num_items]")
    target_item_ids = target_item_ids.to(scores.device).long()
    if target_item_ids.ndim != 1 or target_item_ids.size(0) != scores.size(0):
        raise ValueError("target_item_ids must have shape [B]")

    valid = torch.ones_like(target_item_ids, dtype=torch.bool, device=scores.device)
    if exclude_invalid_targets:
        valid &= target_item_ids != pad_item_id
        valid &= target_item_ids >= 0
        valid &= target_item_ids < scores.size(1)

    if valid.sum() == 0:
        return {
            f"recall@{k}": torch.zeros((), device=scores.device)
            for k in ks
        } | {
            f"ndcg@{k}": torch.zeros((), device=scores.device)
            for k in ks
        }

    scores_valid = scores[valid]
    targets_valid = target_item_ids[valid]
    max_k = min(max(ks), scores_valid.size(1))
    topk = torch.topk(scores_valid, k=max_k, dim=-1).indices

    metrics: dict[str, torch.Tensor] = {}
    for k in ks:
        kk = min(int(k), max_k)
        topk_k = topk[:, :kk]
        matches = topk_k.eq(targets_valid.unsqueeze(1))
        hit = matches.any(dim=1)

        recall = hit.float().mean()

        # Rank is 0-based.  Only meaningful for hit rows.
        rank_positions = matches.float().argmax(dim=1)
        discounts = torch.zeros_like(recall).expand_as(hit.float()).clone()
        if hit.any():
            hit_ranks = rank_positions[hit].float()
            discounts_hit = 1.0 / torch.log2(hit_ranks + 2.0)
            discounts = torch.zeros_like(hit.float())
            discounts[hit] = discounts_hit
        ndcg = discounts.mean()

        metrics[f"recall@{k}"] = _to_float_metric(recall)
        metrics[f"ndcg@{k}"] = _to_float_metric(ndcg)

    return metrics


def generation_recall_metrics(
    generated_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    target_token_mask: torch.Tensor | None = None,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, torch.Tensor]:
    """Compute semantic-ID generation Recall@K and NDCG@K.

    Args:
        generated_token_ids:
            Tensor with shape ``[B, N, L]`` or ``[B, L]``.
        target_token_ids:
            Tensor with shape ``[B, L]``.
        target_token_mask:
            Optional boolean tensor with shape ``[B, L]``.
    """

    if generated_token_ids.ndim == 2:
        generated_token_ids = generated_token_ids.unsqueeze(1)
    if generated_token_ids.ndim != 3:
        raise ValueError("generated_token_ids must have shape [B, N, L] or [B, L]")
    if target_token_ids.ndim != 2:
        raise ValueError("target_token_ids must have shape [B, L]")
    if generated_token_ids.size(0) != target_token_ids.size(0):
        raise ValueError("generated and target batches must have the same size")

    generated_token_ids = generated_token_ids.to(target_token_ids.device).long()
    target_token_ids = target_token_ids.long()

    if target_token_mask is None:
        mask = torch.ones_like(target_token_ids, dtype=torch.bool)
    else:
        mask = target_token_mask.to(target_token_ids.device).bool()

    eq = generated_token_ids.eq(target_token_ids.unsqueeze(1))
    eq = eq | ~mask.unsqueeze(1)
    matches = eq.all(dim=-1)  # [B, N]

    metrics: dict[str, torch.Tensor] = {}
    max_n = generated_token_ids.size(1)
    for k in ks:
        kk = min(int(k), max_n)
        matches_k = matches[:, :kk]
        hit = matches_k.any(dim=1)
        recall = hit.float().mean()

        rank_positions = matches_k.float().argmax(dim=1)
        discounts = torch.zeros_like(hit.float())
        if hit.any():
            hit_ranks = rank_positions[hit].float()
            discounts[hit] = 1.0 / torch.log2(hit_ranks + 2.0)
        ndcg = discounts.mean()

        metrics[f"generation_recall@{k}"] = _to_float_metric(recall)
        metrics[f"generation_ndcg@{k}"] = _to_float_metric(ndcg)

    return metrics


def merge_metric_dicts(*metric_dicts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for metrics in metric_dicts:
        merged.update(metrics)
    return merged
