from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import numpy as np
import torch

from flowrec.models.base import Batch, FlowRecModel, RecModelOutput



PopularityScoreTransform = Literal["count", "log1p", "sqrt", "log_prob"]


@dataclass(frozen=True, slots=True)
class PopularityConfig:
    """Configuration for the non-parametric most-popular baseline.

    Args:
        num_items:
            Number of item IDs including the padding item. In FlowRec this is
            usually ``dm.vocab_item_size``.
        pad_item_id:
            Padding item ID. FlowRec reserves item ID 0 for padding.
        score_transform:
            Monotonic transform applied to item counts before ranking. Ranking
            is identical for ``count``, ``log1p`` and ``sqrt``; ``log_prob`` is
            useful if scores are consumed as logits/probability-like values.
        smoothing:
            Additive smoothing used by ``log_prob``.
        mask_pad_item:
            If true, padding item is assigned ``-inf`` when scoring all items.
        exclude_history_in_forward:
            If true, ``forward``/``score_items`` masks items already present in
            ``batch['history_item_ids']``. Default is false because repeated
            next-item interactions may be valid in some datasets.
        exclude_history_in_generate:
            If true, ``generate`` masks already-seen history items before top-k.
    """

    num_items: int
    pad_item_id: int = 0
    score_transform: PopularityScoreTransform = "log1p"
    smoothing: float = 1.0
    mask_pad_item: bool = True
    exclude_history_in_forward: bool = False
    exclude_history_in_generate: bool = True
    topk_k: int = 10



class PopularityModel(FlowRecModel):
    """Most-popular item recommendation baseline.

    This model has no trainable neural parameters. It estimates item popularity
    from training interactions or training targets and then returns the same
    popularity scores for every user, optionally masking padding/history items.

    It follows the FlowRec model API so it can be evaluated with the same metric
    code as SASRec-style item-scoring models:

        output = model(batch)
        output.scores  # [B, num_items]

    Fit options:
        - ``fit_from_counts`` for precomputed counts.
        - ``fit_from_store`` for SequenceStore + optional TargetIndex.
        - ``fit_from_dataloader`` for ordinary FlowRec train dataloaders.
    """

    model_type = "popularity"

    def __init__(self, cfg: PopularityConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = PopularityConfig(**cfg)
        if cfg.num_items <= 1:
            raise ValueError("num_items must include padding plus at least one real item")
        if cfg.pad_item_id < 0 or cfg.pad_item_id >= cfg.num_items:
            raise ValueError("pad_item_id must be inside [0, num_items)")
        if cfg.smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.cfg = cfg

        self.register_buffer("item_counts", torch.zeros(cfg.num_items, dtype=torch.float32))
        self.register_buffer("item_scores", torch.zeros(cfg.num_items, dtype=torch.float32))
        self.register_buffer("_is_fitted", torch.tensor(False, dtype=torch.bool))

    def fit_from_counts(self, counts: torch.Tensor | np.ndarray | Iterable[float]) -> "PopularityModel":
        """Fit the baseline from item occurrence counts."""
        counts_tensor = torch.as_tensor(counts, dtype=torch.float32)
        if counts_tensor.ndim != 1:
            raise ValueError("counts must be a 1-D tensor/array")
        if counts_tensor.numel() != self.cfg.num_items:
            raise ValueError(
                f"counts length must equal num_items={self.cfg.num_items}, "
                f"got {counts_tensor.numel()}"
            )
        counts_tensor = counts_tensor.clamp_min(0)
        self.item_counts.copy_(counts_tensor.to(self.item_counts.device))
        self.item_scores.copy_(self._transform_counts(self.item_counts))
        self._is_fitted.fill_(True)
        return self

    def fit_from_store(
        self,
        store: Any,
        target_index: Any | None = None,
        *,
        count_targets_only: bool = True,
    ) -> "PopularityModel":
        """Fit from FlowRec SequenceStore and optional training TargetIndex.

        Args:
            store:
                SequenceStore-like object with ``item_ids`` and optionally
                ``user_offsets``.
            target_index:
                TargetIndex-like object. If provided and ``count_targets_only``
                is true, popularity is counted over training target positions.
            count_targets_only:
                If false or target_index is missing, counts all interactions in
                ``store.item_ids``.
        """
        if target_index is not None and count_targets_only:
            user_offsets = np.asarray(store.user_offsets, dtype=np.int64)
            user_indices = np.asarray(target_index.user_indices, dtype=np.int64)
            target_positions = np.asarray(target_index.target_positions, dtype=np.int64)
            abs_positions = user_offsets[user_indices] + target_positions
            item_ids = np.asarray(store.item_ids, dtype=np.int64)[abs_positions]
        else:
            item_ids = np.asarray(store.item_ids, dtype=np.int64)
        counts = np.bincount(
            item_ids[(item_ids >= 0) & (item_ids < self.cfg.num_items)],
            minlength=self.cfg.num_items,
        ).astype(np.float32, copy=False)
        return self.fit_from_counts(counts)

    def fit_from_dataloader(
        self,
        dataloader: Any,
        *,
        count_targets: bool = True,
        count_history: bool = False,
    ) -> "PopularityModel":
        """Fit from a FlowRec dataloader.

        By default this counts ``batch['target_item_ids']``. If
        ``count_history`` is true, valid history items are also counted.
        Dense-causal train batches are supported too: per-position labels
        (``label_item_ids`` under ``label_mask``) are the targets, and
        ``input_item_ids`` under ``attention_mask`` is the history.
        """
        counts = torch.zeros(self.cfg.num_items, dtype=torch.float32)
        saw_countable_batch = False
        for batch in dataloader:
            if count_targets:
                target = _get_first_tensor(batch, ("target_item_ids", "target_item_id"))
                if target is not None:
                    saw_countable_batch = True
                    self._accumulate_counts(counts, target)
                else:
                    labels = _get_first_tensor(batch, ("label_item_ids",))
                    if labels is not None:
                        label_mask = _get_first_tensor(batch, ("label_mask",))
                        if label_mask is not None:
                            labels = labels[label_mask.to(torch.bool)]
                        saw_countable_batch = True
                        self._accumulate_counts(counts, labels)
            if count_history:
                history = _get_first_tensor(batch, ("history_item_ids", "input_item_ids"))
                if history is not None:
                    history_mask = _get_first_tensor(batch, ("history_mask", "attention_mask"))
                    if history_mask is not None:
                        history = history[history_mask.to(torch.bool)]
                    saw_countable_batch = True
                    self._accumulate_counts(counts, history)
        if not saw_countable_batch:
            raise ValueError(
                "fit_from_dataloader saw no batches with countable item keys "
                "(target_item_ids / label_item_ids / history_item_ids / "
                "input_item_ids); refusing to fit an all-zero popularity model."
            )
        return self.fit_from_counts(counts)

    def _accumulate_counts(self, counts: torch.Tensor, item_ids: torch.Tensor) -> None:
        flat = item_ids.detach().cpu().long().reshape(-1)
        valid = (flat >= 0) & (flat < self.cfg.num_items)
        if self.cfg.mask_pad_item:
            valid &= flat != self.cfg.pad_item_id
        flat = flat[valid]
        if flat.numel() == 0:
            return
        bincount = torch.bincount(flat, minlength=self.cfg.num_items).to(counts.dtype)
        counts.add_(bincount)

    def _transform_counts(self, counts: torch.Tensor) -> torch.Tensor:
        if self.cfg.score_transform == "count":
            scores = counts.clone()
        elif self.cfg.score_transform == "log1p":
            scores = torch.log1p(counts)
        elif self.cfg.score_transform == "sqrt":
            scores = torch.sqrt(counts)
        elif self.cfg.score_transform == "log_prob":
            smoothed = counts + float(self.cfg.smoothing)
            scores = torch.log(smoothed / smoothed.sum().clamp_min(1e-12))
        else:
            raise ValueError(f"Unknown score_transform: {self.cfg.score_transform}")
        return scores

    def _check_fitted(self) -> None:
        if not bool(self._is_fitted.item()):
            raise RuntimeError(
                "PopularityModel is not fitted. Call fit_from_counts(), "
                "fit_from_store(), or fit_from_dataloader() before scoring."
            )

    def forward(self, batch: Batch) -> RecModelOutput:
        scores = self.score_items(batch)
        return RecModelOutput(logits=scores, scores=scores)

    def score_items(
        self,
        batch: Batch,
        candidate_item_ids: torch.Tensor | None = None,
        candidate_token_ids: torch.Tensor | None = None,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return popularity scores for all items or selected candidates."""
        if candidate_token_ids is not None:
            raise NotImplementedError("PopularityModel scores atomic item IDs, not semantic tokens.")
        self._check_fitted()
        batch_size = _infer_batch_size(batch)
        scores = self.item_scores.to(self.device).unsqueeze(0).expand(batch_size, -1)
        scores = self._apply_common_masks(scores, batch, exclude_history=self.cfg.exclude_history_in_forward)

        if candidate_item_ids is None:
            return scores

        candidate_item_ids = candidate_item_ids.to(scores.device).long()
        if candidate_item_ids.ndim == 1:
            if candidate_item_ids.numel() == batch_size:
                return scores.gather(dim=1, index=candidate_item_ids[:, None]).squeeze(1)
            return scores[:, candidate_item_ids]
        if candidate_item_ids.ndim == 2:
            if candidate_item_ids.size(0) != batch_size:
                raise ValueError("candidate_item_ids with shape [B, N] must match batch size")
            return scores.gather(dim=1, index=candidate_item_ids)
        raise ValueError("candidate_item_ids must have shape [N], [B], or [B, N]")

    @torch.no_grad()
    def generate(
        self,
        batch: Batch,
        *,
        num_beams: int | None = None,
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        constrain_valid_sequences: bool = True,
        exclude_seen_items: bool = False,
    ) -> torch.Tensor:
        """Return top-k popular item IDs, optionally excluding history items."""
        self._check_fitted()
        k = int(num_return_sequences or self.cfg.topk_k)
        scores = self.item_scores.to(self.device).unsqueeze(0).expand(_infer_batch_size(batch), -1)
        scores = self._apply_common_masks(scores, batch, exclude_history=self.cfg.exclude_history_in_generate)
        k = min(k, scores.size(1))
        return torch.topk(scores, k=k, dim=1).indices

    def _apply_common_masks(
        self,
        scores: torch.Tensor,
        batch: Batch,
        *,
        exclude_history: bool,
    ) -> torch.Tensor:
        needs_clone = bool(self.cfg.mask_pad_item or exclude_history)
        if needs_clone:
            scores = scores.clone()
        if self.cfg.mask_pad_item:
            scores[:, self.cfg.pad_item_id] = -torch.inf
        if exclude_history and "history_item_ids" in batch:
            history = batch["history_item_ids"].to(scores.device).long()
            mask = batch.get("history_mask")
            if torch.is_tensor(mask):
                mask = mask.to(scores.device).bool()
            else:
                mask = history != self.cfg.pad_item_id
            safe_history = history.masked_fill(~mask, self.cfg.pad_item_id)
            scores.scatter_(
                dim=1,
                index=safe_history.clamp_min(0).clamp_max(scores.size(1) - 1),
                value=-torch.inf,
            )
        return scores

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.cfg)


# Backward-compatible aliases.
MostPopularConfig = PopularityConfig
MostPopularModel = PopularityModel


def popularity_from_dataloader(
    dataloader: Any,
    *,
    num_items: int,
    pad_item_id: int = 0,
    score_transform: PopularityScoreTransform = "log1p",
    count_targets: bool = True,
    count_history: bool = False,
) -> PopularityModel:
    model = PopularityModel(
        PopularityConfig(
            num_items=num_items,
            pad_item_id=pad_item_id,
            score_transform=score_transform,
        )
    )
    return model.fit_from_dataloader(
        dataloader,
        count_targets=count_targets,
        count_history=count_history,
    )


def _infer_batch_size(batch: Batch) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.size(0))
    raise ValueError("Cannot infer batch size from batch")


def _get_first_tensor(batch: Batch, keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    return None
