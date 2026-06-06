from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


Batch = Mapping[str, Any]


@dataclass(slots=True)
class RecModelOutput:
    """Common output object for every FlowRec model.

    Models are responsible for running the backbone and returning model outputs.
    They do not own training objectives. Losses such as SFT, GFlowGR-TB/DB,
    DPO, GRPO, etc. are computed in ``flowrec.methods``.

    Typical fields:
      - TIGER-style models return ``logits`` with shape [B, L, V].
      - SASRec-style models return ``scores`` with shape [B, num_items].
      - ``predictions`` stores generated semantic IDs or top-k item IDs.
    """

    logits: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    token_logprobs: torch.Tensor | None = None
    sequence_logprobs: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class FlowRecModel(nn.Module, ABC):
    """Single base class for all FlowRec models.

    The model API exposes capabilities, not objectives. Concrete models may
    support any subset of ``forward``, ``score_items``, ``token_logprobs``,
    ``sequence_logprobs``, and ``generate``. Fine-tuning methods decide how to
    turn these outputs into losses.
    """

    model_type: str = "base"

    @abstractmethod
    def forward(self, batch: Batch) -> RecModelOutput:
        """Run the model on a FlowRec batch and return raw outputs."""
        raise NotImplementedError

    def token_logprobs(
        self,
        batch: Batch,
        *,
        target_token_ids: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-token log-probabilities for semantic-ID trajectories."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose token_logprobs()."
        )

    def sequence_logprobs(
        self,
        batch: Batch,
        *,
        target_token_ids: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return full semantic-ID trajectory log-probabilities."""
        token_lp = self.token_logprobs(
            batch,
            target_token_ids=target_token_ids,
            target_token_mask=target_token_mask,
        )

        if target_token_mask is None:
            target_token_mask = batch.get("target_token_mask")

        if target_token_mask is None:
            return token_lp.sum(dim=-1)

        return (token_lp * target_token_mask.to(token_lp.device).to(token_lp.dtype)).sum(dim=-1)

    @torch.no_grad()
    def generate(
        self,
        batch: Batch,
        *,
        num_beams: int | None = None,
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        constrain_valid_sequences: bool = False,
        exclude_seen_items: bool = False,
    ) -> torch.Tensor:
        """Generate recommendations.

        TIGER returns semantic-ID token sequences of shape [B, K, L].
        SASRec returns top-k item IDs of shape [B, K].
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement generate().")

    def score_items(
        self,
        batch: Batch,
        candidate_item_ids: torch.Tensor | None = None,
        candidate_token_ids: torch.Tensor | None = None,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score items or candidate semantic-ID trajectories."""
        if candidate_token_ids is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement item-ID scoring."
            )

        if candidate_token_ids.ndim == 2:
            return self.sequence_logprobs(
                batch,
                target_token_ids=candidate_token_ids,
                target_token_mask=candidate_token_mask,
            )

        if candidate_token_ids.ndim != 3:
            raise ValueError("candidate_token_ids must have shape [B, L] or [B, N, L]")

        bsz, n_candidates, sid_len = candidate_token_ids.shape
        flat_candidates = candidate_token_ids.reshape(bsz * n_candidates, sid_len)

        flat_mask = None
        if candidate_token_mask is not None:
            flat_mask = candidate_token_mask.reshape(bsz * n_candidates, sid_len)

        expanded_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == (bsz,):
                expanded_batch[key] = value.repeat_interleave(n_candidates, dim=0)
            else:
                expanded_batch[key] = value

        scores = self.sequence_logprobs(
            expanded_batch,
            target_token_ids=flat_candidates,
            target_token_mask=flat_mask,
        )
        return scores.reshape(bsz, n_candidates)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    def config_dict(self) -> dict[str, Any]:
        """Return a serializable model config. Concrete models may override."""
        return {}

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save a portable model-only checkpoint."""
        payload = {
            "model_type": self.model_type,
            "model_class": self.__class__.__name__,
            "config": self.config_dict(),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, Path(path))

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self.load_state_dict(state_dict, strict=strict)
        return payload if isinstance(payload, dict) else {"state_dict": state_dict}


def require_batch_keys(batch: Batch, keys: list[str]) -> None:
    missing = [key for key in keys if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing required keys: {missing}")


def infer_semantic_vocab_size_from_mapping(
    semantic_ids: Mapping[str, list[int]],
    *,
    pad_token_id: int | None = None,
) -> int:
    """Infer semantic-token vocabulary size from a semantic-ID mapping."""
    max_id = -1
    for tokens in semantic_ids.values():
        if tokens:
            max_id = max(max_id, max(int(t) for t in tokens))
    if pad_token_id is not None:
        max_id = max(max_id, int(pad_token_id))
    if max_id < 0:
        raise ValueError("Cannot infer semantic vocab size from empty semantic IDs.")
    return max_id + 1
