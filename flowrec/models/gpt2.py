from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Mapping

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList

from .base import Batch, FlowRecModel, RecModelOutput
from .catalog import _CatalogActionIndex


log = logging.getLogger(__name__)


class SemanticTokenLogitsProcessor(LogitsProcessor):
    """Disallow user and special tokens during semantic-ID generation."""

    def __init__(self, semantic_vocab_size: int) -> None:
        self.semantic_vocab_size = int(semantic_vocab_size)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        del input_ids
        scores[:, self.semantic_vocab_size :] = -torch.inf
        return scores


class GPT2ValidSequenceLogitsProcessor(LogitsProcessor):
    """Restrict each generated prefix to semantic IDs present in the catalog.

    The prefix trie is prepared once by :class:`GPT2RecModel`; unlike the old
    implementation, this processor never scans the full item catalog at every
    decoding step.
    """

    def __init__(
        self,
        *,
        allowed_next_tokens: tuple[Mapping[tuple[int, ...], tuple[int, ...]], ...],
        prompt_length: int,
        semantic_vocab_size: int,
    ) -> None:
        self.allowed_next_tokens = allowed_next_tokens
        self.prompt_length = int(prompt_length)
        self.semantic_vocab_size = int(semantic_vocab_size)
        self.sid_len = len(allowed_next_tokens)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        step = input_ids.size(1) - self.prompt_length
        if step < 0 or step >= self.sid_len:
            return scores

        generated = input_ids[:, self.prompt_length :]
        constrained = scores.new_full(scores.shape, -torch.inf)

        # Group equal prefixes to avoid repeating the same dictionary lookup.
        rows_by_prefix: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for row, prefix in enumerate(generated.tolist()):
            rows_by_prefix[tuple(int(token) for token in prefix)].append(row)

        step_mapping = self.allowed_next_tokens[step]
        for prefix, rows in rows_by_prefix.items():
            allowed = step_mapping.get(prefix)
            if not allowed:
                raise RuntimeError(
                    "Generation reached a semantic-ID prefix absent from the catalog: "
                    f"step={step}, prefix={prefix}."
                )
            row_index = torch.as_tensor(rows, dtype=torch.long, device=scores.device)
            token_index = torch.as_tensor(allowed, dtype=torch.long, device=scores.device)
            constrained[row_index[:, None], token_index[None, :]] = scores[
                row_index[:, None], token_index[None, :]
            ]

        # Explicitly retain only semantic classes even if the trie is malformed.
        constrained[:, self.semantic_vocab_size :] = -torch.inf
        return constrained


class GPT2RecModel(FlowRecModel):
    """Decoder-only recommender that generates semantic item IDs.

    The model accepts both FlowRec layouts:

    * ``prefix_target`` for one target item per history;
    * ``dense_causal`` for a continuous item-token stream with a masked loss.

    Validation uses compact left-padded prompts with explicit ``position_ids``.
    This is essential for decoder-only models: target token 0 must be predicted
    from the final real prompt token, not from a right-padding token.
    """

    model_type = "gpt2"
    supported_train_layouts = ("prefix_target", "dense_causal")
    dense_supervision = "tokens"

    def __init__(
        self,
        *,
        codebook_size: int,
        sem_id_len: int,
        semantic_id_mapping: dict[str, list[int]] | None = None,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        attention_dropout: float | None = None,
        residual_dropout: float | None = None,
        embedding_dropout: float | None = None,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 512,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        fast_greedy_generation: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads: "
                f"{embedding_dim} % {num_heads} != 0"
            )
        if sem_id_len <= 0:
            raise ValueError("sem_id_len must be positive")

        self._codebook_size = int(codebook_size)
        self._sem_id_len = int(sem_id_len)
        self._num_beams = int(num_beams)
        self._num_return_sequences = int(num_return_sequences)
        self._fast_greedy_generation = bool(fast_greedy_generation)

        semantic_ids_tensor, sorted_items = self._prepare_semantic_catalog(
            semantic_id_mapping
        )
        self._semantic_item_to_row = {
            item_id: row for row, item_id in enumerate(sorted_items)
        }
        self.register_buffer("item_semantic_ids", semantic_ids_tensor)

        if semantic_ids_tensor.numel() > 0:
            semantic_vocab_size = int(semantic_ids_tensor.max().item()) + 1
        else:
            semantic_vocab_size = self._codebook_size * self._sem_id_len
        self._semantic_vocab_size = int(semantic_vocab_size)

        # Keep special tokens outside the semantic-token range.  The model no
        # longer has user-specific input tokens, so the vocabulary contains only
        # semantic tokens plus BOS/EOS/PAD.
        self._unified_vocab_size = self._semantic_vocab_size + 3
        self.bos_token_id = self._unified_vocab_size - 3
        self.eos_token_id = self._unified_vocab_size - 2
        self.pad_token_id = self._unified_vocab_size - 1

        # Backward compatibility with existing GPT-2 YAMLs that may still pass
        # user_ids_count.  It must not leak into GPT2Config or affect the model.
        kwargs.pop("user_ids_count", None)

        config = GPT2Config(
            vocab_size=self._unified_vocab_size,
            n_positions=int(max_position_embeddings),
            n_ctx=int(max_position_embeddings),
            n_embd=int(embedding_dim),
            n_layer=int(num_layers),
            n_head=int(num_heads),
            n_inner=int(dim_feedforward),
            resid_pdrop=float(residual_dropout if residual_dropout is not None else dropout),
            embd_pdrop=float(embedding_dropout if embedding_dropout is not None else dropout),
            attn_pdrop=float(attention_dropout if attention_dropout is not None else dropout),
            layer_norm_epsilon=float(layer_norm_eps),
            initializer_range=float(initializer_range),
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=False,
            tie_word_embeddings=False,
            **kwargs,
        )
        self.config = config
        self.model = GPT2LMHeadModel(config)
        self._allowed_next_tokens = self._build_prefix_trie(
            semantic_ids_tensor
        )
        self._catalog_action_index: _CatalogActionIndex | None = None
        self._batch_range_validated = False

    def _prepare_semantic_catalog(
        self,
        semantic_id_mapping: dict[str, list[int]] | None,
    ) -> tuple[torch.Tensor, list[str]]:
        if not semantic_id_mapping:
            return torch.empty(0, self._sem_id_len, dtype=torch.long), []

        sorted_items = sorted(str(item) for item in semantic_id_mapping)
        semantic_ids: list[list[int]] = []
        for item_id in sorted_items:
            tokens = [int(token) for token in semantic_id_mapping[item_id]]
            if len(tokens) != self._sem_id_len:
                raise ValueError(
                    f"Semantic ID for item {item_id!r} has length {len(tokens)}, "
                    f"expected {self._sem_id_len}."
                )
            if any(token < 0 for token in tokens):
                raise ValueError(f"Semantic ID for item {item_id!r} contains a negative token.")
            semantic_ids.append(tokens)

        tensor = torch.as_tensor(semantic_ids, dtype=torch.long)
        unique_count = int(torch.unique(tensor, dim=0).size(0))
        if unique_count != tensor.size(0):
            raise ValueError(
                "Semantic IDs must be unique after collision resolution: "
                f"{unique_count} unique IDs for {tensor.size(0)} items."
            )
        return tensor, sorted_items

    @staticmethod
    def _build_prefix_trie(
        semantic_ids: torch.Tensor,
    ) -> tuple[dict[tuple[int, ...], tuple[int, ...]], ...]:
        if semantic_ids.numel() == 0:
            return tuple()

        rows = semantic_ids.cpu().tolist()
        sid_len = int(semantic_ids.size(1))
        result: list[dict[tuple[int, ...], tuple[int, ...]]] = []
        for step in range(sid_len):
            mapping: dict[tuple[int, ...], set[int]] = defaultdict(set)
            for row in rows:
                mapping[tuple(int(x) for x in row[:step])].add(int(row[step]))
            result.append(
                {prefix: tuple(sorted(tokens)) for prefix, tokens in mapping.items()}
            )
        return tuple(result)

    def _get_allowed_next_tokens(
		self,
	) -> tuple[dict[tuple[int, ...], tuple[int, ...]], ...]:
        if self._allowed_next_tokens is None:
            self._allowed_next_tokens = self._build_prefix_trie(
                self.item_semantic_ids
            )
        return self._allowed_next_tokens

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

        self._allowed_next_tokens = None
        self._catalog_action_index = None
        self._batch_range_validated = False

        if self.item_semantic_ids.numel() > 0:
            max_token = int(self.item_semantic_ids.max().item())
            if max_token >= self._semantic_vocab_size:
                raise ValueError(
                    "Loaded item_semantic_ids are incompatible with the "
                    "GPT-2 semantic vocabulary: "
                    f"max token={max_token}, "
                    f"semantic_vocab_size={self._semantic_vocab_size}."
                )

        return result

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    def _prefix(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a compact left-padded history-only prompt.

        Left padding makes the final physical prompt position a real token for
        every row. Explicit position IDs keep real-token positions identical to
        an unpadded sequence.
        """
        history_token_ids = batch["history_token_ids"]
        history_token_mask = batch["history_token_mask"]
        batch_size = history_token_ids.size(0)

        history_ids = history_token_ids.long().view(batch_size, -1)
        history_mask = history_token_mask.bool().view(batch_size, -1)

        lengths = history_mask.sum(dim=1).long()
        if torch.any(lengths == 0):
            raise ValueError(
                "GPT2RecModel requires at least one history semantic token."
            )

        width = int(lengths.max().item())
        input_ids = torch.full(
            (batch_size, width),
            self.pad_token_id,
            dtype=torch.long,
            device=history_ids.device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row in range(batch_size):
            valid_history = history_ids[row][history_mask[row]]
            input_ids[row, -valid_history.numel() :] = valid_history
            attention_mask[row, -valid_history.numel() :] = 1

        return input_ids, attention_mask

    def _scoring_prefix(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a history-only context for explicit trajectory scoring.

        Prefix-target batches already expose ``history_token_ids`` directly.
        Dense-causal batches expose the one-token-shifted semantic stream in
        ``input_token_ids``. That stream ends with the first ``sid_len - 1``
        tokens of the final dense label item, so those tokens must be removed
        before scoring an independent candidate trajectory; otherwise candidate
        scoring would leak a prefix of the target item into the context.
        """
        if "history_token_ids" in batch:
            return self._prefix(batch)

        if "input_token_ids" not in batch:
            raise KeyError(
                "GPT2 trajectory scoring requires either history_token_ids or "
                "input_token_ids in the batch."
            )

        tokens = batch["input_token_ids"].long()
        token_mask = batch.get(
            "token_attention_mask",
            tokens.ne(self.pad_token_id),
        ).bool()

        trim = self._sem_id_len - 1
        context_rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for row in range(tokens.size(0)):
            valid_tokens = tokens[row][token_mask[row]]
            if trim:
                if valid_tokens.numel() < trim:
                    raise ValueError(
                        "Dense semantic stream is too short to reconstruct the "
                        "history context for trajectory scoring."
                    )
                valid_tokens = valid_tokens[:-trim]

            if valid_tokens.numel() == 0:
                raise ValueError(
                    "GPT2RecModel requires a non-empty history for trajectory scoring."
                )

            context_rows.append(valid_tokens)
            lengths.append(int(valid_tokens.numel()))

        width = max(lengths)
        input_ids = torch.full(
            (tokens.size(0), width),
            self.pad_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row, history_tokens in enumerate(context_rows):
            input_ids[row, -history_tokens.numel() :] = history_tokens
            attention_mask[row, -history_tokens.numel() :] = 1

        return input_ids, attention_mask

    def validate_batch_token_range(self, batch: Batch) -> None:
        for key in ("history_token_ids", "target_token_ids", "input_token_ids", "label_token_ids"):
            value = batch.get(key)
            if not torch.is_tensor(value) or value.numel() == 0:
                continue
            mask_key = {
                "history_token_ids": "history_token_mask",
                "target_token_ids": "target_token_mask",
                "input_token_ids": "token_attention_mask",
                "label_token_ids": "token_label_mask",
            }[key]
            mask = batch.get(mask_key)
            valid = value[mask.bool()] if torch.is_tensor(mask) else value.reshape(-1)
            if valid.numel() == 0:
                continue
            min_token = int(valid.min().item())
            max_token = int(valid.max().item())
            if min_token < 0 or max_token >= self._semantic_vocab_size:
                raise ValueError(
                    f"{key} contains semantic tokens outside [0, "
                    f"{self._semantic_vocab_size - 1}]: min={min_token}, max={max_token}."
                )

    def forward(self, batch: Batch) -> RecModelOutput:
        if not self._batch_range_validated:
            self.validate_batch_token_range(batch)
            self._batch_range_validated = True

        if "input_token_ids" in batch:
            return self._forward_dense(batch)
        return self._forward_prefix_target(batch)

    def _forward_dense(self, batch: Batch) -> RecModelOutput:
        tokens = batch["input_token_ids"].long()

        token_mask = batch.get(
            "token_attention_mask",
            tokens.ne(self.pad_token_id),
        ).bool()

        input_ids = tokens.masked_fill(
            ~token_mask,
            self.pad_token_id,
        )
        attention_mask = token_mask.long()

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            use_cache=False,
        )

        # input_ids:
        #   [s0, s1, ..., sN-1]
        #
        # label_token_ids:
        #   [s1, s2, ..., sN]
        #
        # Therefore label s1 is predicted by the logit at the position of s0,
        # i.e. output position 0.
        logits = output.logits[
            :,
            : tokens.size(1),
            : self._semantic_vocab_size,
        ]

        return RecModelOutput(logits=logits)

    def _forward_prefix_target(self, batch: Batch) -> RecModelOutput:
        prefix_ids, prefix_mask = self._scoring_prefix(batch)
        target_token_ids = batch.get("target_token_ids")
        if target_token_ids is None:
            output = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                position_ids=self._position_ids(prefix_mask),
                use_cache=False,
            )
            return RecModelOutput(
                logits=output.logits[:, :, : self._semantic_vocab_size]
            )

        target_mask = batch.get(
            "target_token_mask", torch.ones_like(target_token_ids, dtype=torch.bool)
        ).bool()
        target_ids = target_token_ids.long().masked_fill(~target_mask, self.pad_token_id)
        input_ids = torch.cat((prefix_ids, target_ids), dim=1)
        attention_mask = torch.cat((prefix_mask, target_mask.long()), dim=1)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            use_cache=False,
        )

        prefix_width = prefix_ids.size(1)
        logits = output.logits[
            :, prefix_width - 1 : prefix_width - 1 + target_ids.size(1),
            : self._semantic_vocab_size,
        ]
        return RecModelOutput(logits=logits)

    def token_logprobs(
        self,
        batch: Batch,
        *,
        target_token_ids: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_token_ids is None:
            target_token_ids = batch["target_token_ids"]
        if target_token_mask is None:
            target_token_mask = batch.get("target_token_mask")
        if target_token_mask is None:
            target_token_mask = torch.ones_like(target_token_ids, dtype=torch.bool)

        work_batch = dict(batch)
        work_batch["target_token_ids"] = target_token_ids
        work_batch["target_token_mask"] = target_token_mask
        logits = self._forward_prefix_target(work_batch).logits
        if logits is None:
            raise RuntimeError("GPT2 trajectory scoring returned no logits")

        safe_targets = target_token_ids.long().masked_fill(~target_token_mask.bool(), 0)
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            dim=-1, index=safe_targets.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs.masked_fill(~target_token_mask.bool(), 0.0)

    def score_item_tokens(
        self,
        batch: Batch,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-token semantic-vocabulary log-probs from one teacher-forced pass."""
        logits = self.candidate_token_logits(
            batch,
            candidate_token_ids=candidate_token_ids,
            candidate_token_mask=candidate_token_mask,
        )
        mask = (
            torch.ones_like(candidate_token_ids, dtype=torch.bool)
            if candidate_token_mask is None else candidate_token_mask.bool()
        )
        targets = candidate_token_ids.long().masked_fill(~mask, 0)
        token_lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_lp.masked_fill(~mask, 0.0)

    def score_item_tokens_valid_actions(
        self,
        batch: Batch,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-token log-probs normalized over catalog children, as in TIGER.

        GFlow DB/SubTB selects this method automatically. All transitions use
        one teacher-forced forward, including when training with dropout.
        """
        if self.item_semantic_ids.numel() == 0:
            return self.score_item_tokens(
                batch,
                candidate_token_ids=candidate_token_ids,
                candidate_token_mask=candidate_token_mask,
            )
        logits = self.candidate_token_logits(
            batch,
            candidate_token_ids=candidate_token_ids,
            candidate_token_mask=candidate_token_mask,
        )
        bsz, n_candidates, sid_len = candidate_token_ids.shape
        targets = candidate_token_ids.reshape(bsz * n_candidates, sid_len)
        mask = (
            torch.ones_like(targets, dtype=torch.bool)
            if candidate_token_mask is None
            else candidate_token_mask.reshape_as(targets).bool()
        )
        if self._catalog_action_index is None:
            self._catalog_action_index = _CatalogActionIndex(self.item_semantic_ids)
        token_lp = self._catalog_action_index.token_logprobs(
            logits.reshape(bsz * n_candidates, sid_len, -1), targets, mask,
        )
        return token_lp.reshape(bsz, n_candidates, sid_len)

    def score_items_valid_actions(
        self,
        batch: Batch,
        candidate_item_ids: torch.Tensor | None = None,
        candidate_token_ids: torch.Tensor | None = None,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Semantic-ID log-probs with catalog normalization, used by GFlow TB."""
        if candidate_token_ids is None:
            return self.score_items(
                batch,
                candidate_item_ids=candidate_item_ids,
                candidate_token_ids=candidate_token_ids,
                candidate_token_mask=candidate_token_mask,
            )
        if candidate_token_ids.ndim == 2:
            token_lp = self.score_item_tokens_valid_actions(
                batch,
                candidate_token_ids=candidate_token_ids.unsqueeze(1),
                candidate_token_mask=(None if candidate_token_mask is None else candidate_token_mask.unsqueeze(1)),
            ).squeeze(1)
            return token_lp.sum(dim=-1)
        if candidate_token_ids.ndim != 3:
            raise ValueError("candidate_token_ids must have shape [B, L] or [B, N, L]")
        return self.score_item_tokens_valid_actions(
            batch,
            candidate_token_ids=candidate_token_ids,
            candidate_token_mask=candidate_token_mask,
        ).sum(dim=-1)

    def candidate_token_logits(
        self,
        batch: Batch,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced next-token logits for candidate semantic-ID prefixes.

        Shape [B, N, L, V] over the semantic vocabulary: ``[..., d, :]`` is the
        model's distribution over token ``d`` given the history and the
        candidate's first ``d`` tokens (causal attention: token ``d`` cannot
        influence it). ``L`` may be shorter than the SID length, which is how
        the policy candidate sampler decodes level by level.
        """
        if candidate_token_ids.ndim != 3:
            raise ValueError("candidate_token_ids must have shape [B, N, L]")
        if candidate_token_mask is not None and candidate_token_mask.shape != candidate_token_ids.shape:
            raise ValueError("candidate_token_mask must match candidate_token_ids")
        bsz, n_candidates, sid_len = candidate_token_ids.shape
        if "_gpt2_history_grad_state" in batch:
            return self._candidate_logits_from_grad_history(
                batch, candidate_token_ids, candidate_token_mask,
            )
        if "_gpt2_history_state" in batch and candidate_token_mask is None:
            return self._candidate_logits_from_cached_history(batch, candidate_token_ids)
        work_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == (bsz,):
                work_batch[key] = value.repeat_interleave(n_candidates, dim=0)
            else:
                work_batch[key] = value
        work_batch["target_token_ids"] = candidate_token_ids.reshape(bsz * n_candidates, sid_len)
        work_batch["target_token_mask"] = (
            torch.ones(bsz * n_candidates, sid_len, dtype=torch.bool, device=candidate_token_ids.device)
            if candidate_token_mask is None
            else candidate_token_mask.reshape(bsz * n_candidates, sid_len).bool()
        )
        logits = self._forward_prefix_target(work_batch).logits
        if logits is None:
            raise RuntimeError("GPT2 candidate scoring returned no logits")
        return logits.reshape(bsz, n_candidates, sid_len, -1)

    @torch.no_grad()
    def cache_history_encoding(self, batch: Batch) -> Batch:
        """A copy of ``batch`` carrying the history's key/value cache, so
        repeated candidate scoring of the same histories (level-by-level
        decoding in the policy candidate sampler) reads the history once
        instead of once per candidate per level. Inference only."""
        prefix_ids, prefix_mask = self._scoring_prefix(batch)
        output = self.model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            position_ids=self._position_ids(prefix_mask),
            use_cache=True,
        )
        cached = dict(batch)
        cached["_gpt2_history_state"] = (
            output.past_key_values,
            prefix_mask,
            output.logits[:, -1, : self._semantic_vocab_size],
        )
        return cached

    def cache_history_encoding_grad(self, batch: Batch) -> Batch:
        """Prefill one differentiable history for repeated training candidates.

        Store immutable tensor references rather than a mutable HF cache.
        Each candidate decoder creates its own cache from these K/V tensors,
        preserving gradients through the shared history and making chunk
        checkpoint recomputation independent of preceding decoder calls.
        The existing no-grad inference cache remains separate.
        """
        prefix_ids, prefix_mask = self._scoring_prefix(batch)
        output = self.model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            position_ids=self._position_ids(prefix_mask),
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("Differentiable GPT2 history prefill requires use_cache=True.")
        # Transformers 5.x yields (key, value, optional sliding metadata);
        # earlier DynamicCache and legacy tuple caches yield (key, value).
        key_values = tuple((layer[0], layer[1]) for layer in output.past_key_values)
        cached = dict(batch)
        cached["_gpt2_history_grad_state"] = (
            key_values, prefix_mask, output.logits[:, -1, : self._semantic_vocab_size],
        )
        return cached

    def _candidate_logits_from_grad_history(
        self,
        batch: Batch,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        from transformers.cache_utils import DynamicCache

        key_values, prefix_mask, first_logits = batch["_gpt2_history_grad_state"]
        bsz, n_candidates, sid_len = candidate_token_ids.shape
        if prefix_mask.size(0) != bsz:
            raise ValueError("Cached GPT2 history batch size does not match candidate tokens.")
        if sid_len < 1:
            raise ValueError("Candidate semantic IDs must have at least one token.")
        first = first_logits.repeat_interleave(n_candidates, dim=0).unsqueeze(1)
        if sid_len == 1:
            return first.reshape(bsz, n_candidates, 1, -1)

        # DynamicCache.update exists in both the legacy key_cache API and
        # current layer-based API. Fresh objects avoid deepcopy of non-leaf
        # tensors and never append candidate states to the shared prefill.
        cache = DynamicCache()
        for layer_idx, (keys, values) in enumerate(key_values):
            cache.update(
                keys.repeat_interleave(n_candidates, dim=0),
                values.repeat_interleave(n_candidates, dim=0), layer_idx,
            )
        mask = (torch.ones_like(candidate_token_ids, dtype=torch.bool)
                if candidate_token_mask is None else candidate_token_mask.bool())
        tokens = candidate_token_ids.long().masked_fill(~mask, self.pad_token_id)
        tokens = tokens.reshape(bsz * n_candidates, sid_len)[:, :-1]
        token_mask = mask.reshape(bsz * n_candidates, sid_len)[:, :-1].long()
        attention_mask = torch.cat((prefix_mask.repeat_interleave(n_candidates, dim=0), token_mask), dim=1)
        output = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask)[:, -tokens.size(1):],
            past_key_values=cache,
            use_cache=True,
        )
        logits = torch.cat((first, output.logits[:, :, : self._semantic_vocab_size]), dim=1)
        return logits.reshape(bsz, n_candidates, sid_len, -1)

    def _candidate_logits_from_cached_history(
        self, batch: Batch, candidate_token_ids: torch.Tensor
    ) -> torch.Tensor:
        import copy

        past, prefix_mask, first_logits = batch["_gpt2_history_state"]
        bsz, n_candidates, sid_len = candidate_token_ids.shape
        first = first_logits.repeat_interleave(n_candidates, dim=0).unsqueeze(1)
        if sid_len == 1:  # token 0 depends on the history alone
            return first.reshape(bsz, n_candidates, 1, -1)
        cache = copy.deepcopy(past)  # the forward below appends to it
        cache.batch_repeat_interleave(n_candidates)
        tokens = candidate_token_ids.reshape(bsz * n_candidates, sid_len)[:, :-1].long()
        attention_mask = torch.cat(
            (prefix_mask.repeat_interleave(n_candidates, dim=0), torch.ones_like(tokens)), dim=1
        )
        output = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask)[:, -tokens.size(1):],
            past_key_values=cache,
            use_cache=True,
        )
        logits = torch.cat((first, output.logits[:, :, : self._semantic_vocab_size]), dim=1)
        return logits.reshape(bsz, n_candidates, sid_len, -1)

    def _apply_trie_constraint(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        constrained = logits.new_full(
            logits.shape,
            -torch.inf,
        )

        rows_by_prefix: dict[
            tuple[int, ...],
            list[int],
        ] = defaultdict(list)

        for row, prefix in enumerate(generated.tolist()):
            rows_by_prefix[
                tuple(int(token) for token in prefix)
            ].append(row)

        allowed_next_tokens = self._get_allowed_next_tokens()
        mapping = allowed_next_tokens[step]

        for prefix, rows in rows_by_prefix.items():
            allowed = mapping.get(prefix)
            if not allowed:
                raise RuntimeError(
                    f"Invalid generated semantic prefix: {prefix}"
                )

            row_idx = torch.as_tensor(
                rows,
                device=logits.device,
                dtype=torch.long,
            )
            token_idx = torch.as_tensor(
                allowed,
                device=logits.device,
                dtype=torch.long,
            )

            constrained[
                row_idx[:, None],
                token_idx[None, :],
            ] = logits[
                row_idx[:, None],
                token_idx[None, :],
            ]

        return constrained

    @torch.inference_mode()
    def _generate_greedy(
        self,
        batch: Batch,
        *,
        constrain_valid_sequences: bool,
    ) -> torch.Tensor:
        prefix_ids, prefix_mask = self._prefix(batch)
        output = self.model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            position_ids=self._position_ids(prefix_mask),
            use_cache=True,
        )
        logits = output.logits[:, -1, : self._semantic_vocab_size]
        past_key_values = output.past_key_values
        generated = torch.empty(
            (prefix_ids.size(0), 0), dtype=torch.long, device=prefix_ids.device
        )
        attention_mask = prefix_mask
        next_position = prefix_mask.sum(dim=1).long()

        for step in range(self._sem_id_len):
            if constrain_valid_sequences:
                logits = self._apply_trie_constraint(logits, generated, step)
            next_token = logits.argmax(dim=-1)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if step + 1 == self._sem_id_len:
                break

            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.size(0), 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            output = self.model(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                position_ids=next_position[:, None],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, : self._semantic_vocab_size]
            next_position = next_position + 1

        return generated.unsqueeze(1)

    @torch.inference_mode()
    def generate(
        self,
        batch: Batch,
        *,
        num_beams: int | None = None,
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        constrain_valid_sequences: bool = True,
        exclude_seen_items: bool = False,
        do_sample: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        if exclude_seen_items:
            raise NotImplementedError(
                "exclude_seen_items is not implemented for GPT2RecModel. "
                "Disable it rather than silently evaluating a different protocol."
            )

        num_beams = int(num_beams or self._num_beams)
        num_return_sequences = int(
            num_return_sequences or self._num_return_sequences
        )
        max_new_tokens = int(max_new_tokens or self._sem_id_len)
        do_sample = bool(do_sample)

        if max_new_tokens != self._sem_id_len:
            raise ValueError(
                f"GPT2RecModel must generate exactly "
                f"{self._sem_id_len} semantic tokens."
            )

        if num_beams <= 0 or num_return_sequences <= 0:
            raise ValueError(
                "num_beams and num_return_sequences must be positive"
            )

        if not do_sample and num_return_sequences > num_beams:
            raise ValueError(
                "num_return_sequences cannot exceed num_beams "
                "when do_sample=False"
            )

        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be positive")

        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")

        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        if (
            constrain_valid_sequences
            and self.item_semantic_ids.numel() == 0
        ):
            raise ValueError(
                "Valid-sequence generation requires a non-empty "
                "semantic catalog"
            )

        if (
            self._fast_greedy_generation
            and not do_sample
            and num_beams == 1
            and num_return_sequences == 1
        ):
            return self._generate_greedy(
                batch,
                constrain_valid_sequences=constrain_valid_sequences,
            )

        prefix_ids, prefix_mask = self._prefix(batch)

        processors = LogitsProcessorList(
            [SemanticTokenLogitsProcessor(self._semantic_vocab_size)]
        )

        if constrain_valid_sequences:
            processors.append(
                GPT2ValidSequenceLogitsProcessor(
                    allowed_next_tokens=self._get_allowed_next_tokens(),
                    prompt_length=prefix_ids.size(1),
                    semantic_vocab_size=self._semantic_vocab_size,
                )
            )

        generate_kwargs: dict[str, Any] = {
            "input_ids": prefix_ids,
            "attention_mask": prefix_mask,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": self._sem_id_len,
            "do_sample": do_sample,
            "early_stopping": False,
            "logits_processor": processors,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": None,
            "use_cache": True,
        }

        if temperature is not None:
            generate_kwargs["temperature"] = float(temperature)

        if top_k is not None:
            generate_kwargs["top_k"] = int(top_k)

        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)

        old_use_cache = self.model.config.use_cache
        self.model.config.use_cache = True

        try:
            outputs = self.model.generate(**generate_kwargs)
        finally:
            self.model.config.use_cache = old_use_cache
        generated = outputs[:, prefix_ids.size(1) :]
        expected_rows = prefix_ids.size(0) * num_return_sequences

        if generated.shape != (expected_rows, self._sem_id_len):
            raise RuntimeError(
                "Unexpected generated shape: "
                f"{tuple(generated.shape)}, "
                f"expected ({expected_rows}, {self._sem_id_len})."
            )

        return generated.view(
            prefix_ids.size(0),
            num_return_sequences,
            self._sem_id_len,
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "codebook_size": self._codebook_size,
            "sem_id_len": self._sem_id_len,
            "semantic_vocab_size": self._semantic_vocab_size,
            "unified_vocab_size": self._unified_vocab_size,
            "embedding_dim": self.config.n_embd,
            "num_heads": self.config.n_head,
            "num_layers": self.config.n_layer,
            "dim_feedforward": self.config.n_inner,
            "max_position_embeddings": self.config.n_positions,
            "num_beams": self._num_beams,
            "num_return_sequences": self._num_return_sequences,
            "fast_greedy_generation": self._fast_greedy_generation,
            "tie_word_embeddings": self.config.tie_word_embeddings,
        }
