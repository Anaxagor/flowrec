"""Optional differentiable history cache for GPT2 candidate scoring.

The cache is an optimization of the existing teacher-forced scorer. It is off
unless a method config sets ``gpt2_shared_history_cache``. Non-GPT2 models and
an SDPO policy that is training with dropout keep the legacy path.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from flowrec.models.base import Batch, FlowRecModel

LOGGER = logging.getLogger("flowrec.ft_methods.gpt2_history_cache")

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    LOGGER.warning(message)


def has_grad_history_cache(model: FlowRecModel) -> bool:
    return callable(getattr(model, "cache_history_encoding_grad", None))


def policy_dropout_rate(model: FlowRecModel) -> float:
    config = getattr(model, "config", None)
    if config is None:
        return 0.0
    return max(
        float(getattr(config, "resid_pdrop", 0.0) or 0.0),
        float(getattr(config, "embd_pdrop", 0.0) or 0.0),
        float(getattr(config, "attn_pdrop", 0.0) or 0.0),
    )


def shared_history_cache_enabled(
    model: FlowRecModel,
    *,
    enabled: bool,
    allow_active_dropout: bool,
) -> bool:
    """Return whether this call should prefill one shared history.

    ``allow_active_dropout`` is for callers that already run the scorer in
    ``eval()`` (GRPO). SDPO policy scoring must pass False: a shared prefill
    would give every candidate the same dropout mask.
    """
    if not enabled:
        return False
    if not has_grad_history_cache(model):
        _warn_once(
            "missing-cache",
            "gpt2_shared_history_cache is set, but this model has no differentiable "
            "history cache. Using the legacy candidate scorer.",
        )
        return False
    if (
        not allow_active_dropout
        and model.training
        and policy_dropout_rate(model) > 0.0
    ):
        _warn_once(
            "sdpo-dropout",
            "gpt2_shared_history_cache skipped for SDPO policy scoring because "
            "dropout is active. Using the legacy candidate scorer.",
        )
        return False
    return True


class _ScoringMode:
    """Reapply the mode captured at the scoring forward during recomputation."""

    def __init__(
        self,
        flags: Sequence[tuple[torch.nn.Module, bool]],
        autocast_state: tuple[str, torch.dtype, bool],
    ) -> None:
        self._flags = flags
        self._device_type, self._dtype, self._enabled = autocast_state
        self._previous: list[tuple[torch.nn.Module, bool]] | None = None
        self._autocast: torch.autocast | None = None

    def __enter__(self) -> "_ScoringMode":
        self._previous = [(module, module.training) for module, _ in self._flags]
        for module, training in self._flags:
            module.train(training)
        self._autocast = torch.autocast(
            device_type=self._device_type,
            dtype=self._dtype,
            enabled=self._enabled,
        )
        self._autocast.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._autocast is not None:
            self._autocast.__exit__(exc_type, exc, tb)
        if self._previous is not None:
            for module, training in self._previous:
                module.train(training)
        return False


def _capture_scoring_mode(
    model: FlowRecModel,
    sample: torch.Tensor,
) -> tuple[list[tuple[torch.nn.Module, bool]], tuple[str, torch.dtype, bool]]:
    flags = [(module, module.training) for module in model.modules()]
    device_type = sample.device.type
    if device_type not in {"cpu", "cuda"}:
        device_type = "cpu"
    dtype = torch.get_autocast_dtype(device_type)
    enabled = torch.is_autocast_enabled(device_type)
    return flags, (device_type, dtype, enabled)


def run_candidate_chunks(
    score_fn: Callable[..., torch.Tensor],
    model: FlowRecModel,
    candidate_token_ids: torch.Tensor,
    candidate_token_mask: torch.Tensor,
    *bound,
    chunk_size: int | None,
    checkpoint_chunks: bool,
) -> torch.Tensor:
    """Score candidate groups and concatenate them on the candidate axis.

    ``score_fn(*bound, ids, mask)`` must return one tensor per chunk. Chunking
    does not reduce those tensors: the caller keeps the original reduction,
    including one global logsumexp over the concatenated scores. Checkpoint
    recomputation restores the module train/eval flags and autocast state
    captured here, so a later ``model.train()`` does not turn dropout back on
    inside the backward recompute.
    """
    num_candidates = int(candidate_token_ids.size(1))
    if num_candidates < 1:
        raise ValueError("candidate_token_ids must contain at least one candidate.")
    size = num_candidates if chunk_size is None else int(chunk_size)
    if size < 1:
        raise ValueError("gpt2_candidate_chunk_size must be >= 1 when set.")
    if size >= num_candidates and not checkpoint_chunks:
        return score_fn(*bound, candidate_token_ids, candidate_token_mask)

    flags, autocast_state = _capture_scoring_mode(model, candidate_token_ids)

    def context_fn() -> tuple[Any, _ScoringMode]:
        # The forward already runs in the captured mode. Only recomputation,
        # which happens after GRPO has restored train(), must re-enter it.
        return nullcontext(), _ScoringMode(flags, autocast_state)

    parts: list[torch.Tensor] = []
    for start in range(0, num_candidates, size):
        chunk_ids = candidate_token_ids[:, start : start + size]
        chunk_mask = candidate_token_mask[:, start : start + size]
        args = (*bound, chunk_ids, chunk_mask)
        if checkpoint_chunks and torch.is_grad_enabled():
            parts.append(
                checkpoint(
                    score_fn,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=context_fn,
                )
            )
        else:
            parts.append(score_fn(*args))
    return torch.cat(parts, dim=1)


def score_item_token_logprobs(
    model: FlowRecModel,
    batch: Batch,
    candidate_token_ids: torch.Tensor,
    candidate_token_mask: torch.Tensor,
    *,
    valid_actions: bool,
    chunk_size: int | None = None,
    checkpoint_chunks: bool = False,
) -> torch.Tensor:
    """Score candidates from one history prefill of ``model``.

    The returned tensor has shape ``[B, N, L]``. Semantic-vocabulary scoring
    and catalog-valid-action scoring stay separate: the caller chooses the
    same convention as the legacy path. The history prefill stays outside
    candidate chunks, so every chunk reads the same K/V tensors.
    """
    context = model.cache_history_encoding_grad(batch)
    if valid_actions:
        scorer = getattr(model, "score_item_tokens_valid_actions", None)
        if not callable(scorer):
            raise RuntimeError(
                "Valid-action GPT2 scoring was requested, but the model does not implement it."
            )
    else:
        scorer = model.score_item_tokens

    def score(
        cached: Batch,
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return scorer(cached, candidate_token_ids=ids, candidate_token_mask=mask)

    return run_candidate_chunks(
        score,
        model,
        candidate_token_ids,
        candidate_token_mask,
        context,
        chunk_size=chunk_size,
        checkpoint_chunks=checkpoint_chunks,
    )
