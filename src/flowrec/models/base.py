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

    FlowRec uses one base class for all recommenders. Some models generate
    semantic-ID tokens, such as TIGER. Others score item IDs, such as SASRec.
    This output object therefore exposes common fields while allowing
    model-specific values to live in ``extra``.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    token_logprobs: torch.Tensor | None = None
    sequence_logprobs: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class FlowRecModel(nn.Module, ABC):
    """Single base class for all FlowRec models.

    The base class defines checkpointing and optional capabilities used by
    fine-tuning methods:

    - ``compute_loss`` for supervised training.
    - ``score_items`` for item-scoring or candidate-scoring models.
    - ``token_logprobs`` and ``sequence_logprobs`` for generative models.
    - ``generate`` for recommendation generation.

    Concrete models implement only the capabilities they support. For example,
    SASRec implements item scoring. TIGER implements token log-probabilities and
    semantic-ID generation.
    """

    model_type: str = "base"

    @abstractmethod
    def forward(self, batch: Batch) -> RecModelOutput:
        """Run the model on a FlowRec batch."""
        raise NotImplementedError

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        """Return the model's default supervised loss.

        SFT-style methods should call this method rather than assuming whether
        a model predicts item IDs or semantic-ID tokens.
        """
        output = self.forward(batch)
        if output.loss is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.forward() did not return a loss. "
                "Override compute_loss() for this model."
            )
        return output.loss

    def compute_sft_loss(self, batch: Batch) -> torch.Tensor:
        """Alias kept for readability in SFT-oriented method code."""
        return self.compute_loss(batch)

    def token_logprobs(
        self,
        batch: Batch,
        *,
        target_token_ids: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-token log-probabilities for semantic-ID trajectories.

        Shape:
            [batch_size, target_length]

        Generative models should override this method.
        """
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
        """Return full semantic-ID trajectory log-probabilities.

        The default implementation sums ``token_logprobs``. This is what
        GFlowGR-TB, DPO, and candidate reranking need for generative models.
        """
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
        constrained: bool | None = None,
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
        """Score items or candidate semantic-ID trajectories.

        SASRec-style models should support ``candidate_item_ids``.
        TIGER-style models can rely on the default ``candidate_token_ids`` path,
        which scores semantic-ID trajectories by sequence log-likelihood.
        """
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
        """Save a portable checkpoint.

        We intentionally save ``state_dict`` rather than pickling the full model.
        """
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
