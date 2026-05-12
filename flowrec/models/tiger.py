import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import LogitsProcessor, LogitsProcessorList, T5Config, T5ForConditionalGeneration

from .base import FlowRecModel, Batch, RecModelOutput

log = logging.getLogger(__name__)


class TIGERLogitsProcessor(LogitsProcessor):
    """Constrained decoding processor that restricts generation to valid semantic IDs.

    Expects semantic_id_tensor to contain the exact vocabulary indices that will
    appear in the decoded sequence (i.e. the same representation as batch tokens).
    """

    def __init__(
        self,
        semantic_id_tensor: torch.Tensor,  # [num_items, sid_len]
        num_beams: int,
        visited_item_ids: torch.Tensor | None = None,  # [B, history_len]
        step_token_to_items: list[dict[int, torch.Tensor]] | None = None,
    ):
        self.semantic_ids = semantic_id_tensor
        self.num_beams = num_beams
        self.sid_len = semantic_id_tensor.size(1)
        self.num_items = semantic_id_tensor.size(0)
        self.visited_mask: torch.Tensor | None = None
        self.expanded_visited_mask: torch.Tensor | None = None

        # Per-step mapping: token_value → indices of items that carry that token
        # at that step.  Limits __call__ to O(codebook_size) GPU ops.
        # Accept a precomputed version to avoid rebuilding this on every call
        # when the item table never changes (the common case during training).
        if step_token_to_items is not None:
            self._step_token_to_items = step_token_to_items
        else:
            self._step_token_to_items = TIGERLogitsProcessor._compute_step_token_to_items(
                semantic_id_tensor
            )

        if visited_item_ids is not None and visited_item_ids.numel() > 0:
            B = visited_item_ids.size(0)
            self.visited_mask = torch.zeros(
                B, self.num_items,
                device=semantic_id_tensor.device,
                dtype=torch.bool,
            )
            # Vectorised: find all (user, history_pos) pairs where row_id >= 0
            # (-1 is padding; row IDs are indices into semantic_id_tensor).
            valid = visited_item_ids >= 0
            b_idx, h_idx = valid.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                item_idx = visited_item_ids[b_idx, h_idx].clamp(0, self.num_items - 1)
                self.visited_mask[b_idx, item_idx] = True
            self.expanded_visited_mask = self.visited_mask.repeat_interleave(self.num_beams, dim=0)

    @staticmethod
    def _compute_step_token_to_items(
        semantic_id_tensor: torch.Tensor,
    ) -> list[dict[int, torch.Tensor]]:
        """Build the per-step token→item-index mapping from an item semantic-ID table.

        This is a pure function of the item table and can be cached across calls
        so that constrained generation does not recompute it for every batch.
        """
        sid_len = semantic_id_tensor.size(1)
        result: list[dict[int, torch.Tensor]] = []
        for d in range(sid_len):
            tokens_at_d = semantic_id_tensor[:, d]
            groups: dict[int, torch.Tensor] = {}
            for t_val in tokens_at_d.unique().tolist():
                groups[int(t_val)] = (tokens_at_d == t_val).nonzero(as_tuple=True)[0]
            result.append(groups)
        return result

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids[:, 0] is the decoder_start_token; decoded positions start at 1.
        current_step = input_ids.shape[1] - 1

        if current_step >= self.sid_len:
            return scores

        BK = input_ids.size(0)
        device = input_ids.device
        mask = torch.full_like(scores, -float("inf"))
        token_groups = self._step_token_to_items[current_step]

        # Fast path: step 0 with no visited filter — every item is valid for every
        # beam, so every token that appears at position 0 is universally allowed.
        if current_step == 0 and self.visited_mask is None:
            for token_val in token_groups:
                mask[:, token_val] = 0.0
            return scores + mask

        # Compute which items are still valid for each beam based on decoded prefix.
        if current_step == 0:
            valid_items_mask = torch.ones(BK, self.num_items, device=device, dtype=torch.bool)
        else:
            decoded_tokens = input_ids[:, 1:]  # [BK, current_step]
            prefixes = self.semantic_ids[:, :current_step].to(device)  # [num_items, current_step]
            valid_items_mask = (prefixes.unsqueeze(0) == decoded_tokens.unsqueeze(1)).all(dim=-1)

        if self.visited_mask is not None:
            assert self.expanded_visited_mask is not None
            valid_items_mask = valid_items_mask & ~self.expanded_visited_mask

        # Allow token t for beam b iff at least one item with next-token == t is
        # still valid for b.  At most codebook_size (≤256) iterations instead of
        # the original O(B × num_beams) Python loop.
        for token_val, item_indices in token_groups.items():
            any_valid = valid_items_mask[:, item_indices].any(dim=-1)  # [BK]
            mask[any_valid, token_val] = 0.0

        return scores + mask


class TigerModel(FlowRecModel):
    """TIGER generative retrieval model (arXiv:2305.05065).

    A T5 encoder-decoder that maps a user's interaction history (encoded as
    flattened semantic-ID tokens) to the semantic ID of the next item.

    Token representation
    --------------------
    Semantic IDs are expected to arrive **pre-offset**: each level's tokens
    already occupy a disjoint slice of the vocabulary (as produced by
    RQVAEItemTokenizer with level_offset_tokens=True). Batch padding is tracked
    by masks and replaced with the model's special pad token before the T5 call,
    so semantic token 0 can remain a valid TIGER token.

    Vocabulary layout (derived at runtime from the actual token range):
      [0, sem_vocab_size)        — semantic tokens (pre-offset from tokenizer)
      [sem_vocab_size, +user_ids_count) — user tokens
      last 32 slots              — special tokens (PAD, EOS, decoder-start)
    """

    model_type = "tiger"

    def __init__(
        self,
        codebook_size: int,
        sem_id_len: int,
        user_ids_count: int,
        semantic_id_mapping: dict[str, list[int]] | None = None,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_beams: int = 100,
        num_return_sequences: int = 20,
        d_kv: int = 64,
        layer_norm_eps: float = 1e-6,
        activation: str = "relu",
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__()

        self._codebook_size = codebook_size
        self._sem_id_len = sem_id_len
        self.user_ids_count = user_ids_count
        self._num_beams = num_beams
        self._num_return_sequences = num_return_sequences

        # Semantic-ID tensor for constrained decoding; shape [num_items, sid_len].
        if semantic_id_mapping is not None:
            sorted_items = sorted(semantic_id_mapping.keys())
            sid_list = [semantic_id_mapping[item] for item in sorted_items]
            self._semantic_item_to_row = {
                item_id: row_id for row_id, item_id in enumerate(sorted_items)
            }
            self.register_buffer("item_semantic_ids", torch.tensor(sid_list, dtype=torch.long))
            # Derive vocabulary size from the actual token range so we never
            # allocate a vocab that's too small for the real token values.
            max_token = max(t for sid in semantic_id_mapping.values() for t in sid)
            sem_vocab_size = max_token + 1
        else:
            self._semantic_item_to_row = {}
            self.register_buffer("item_semantic_ids", torch.empty(0, sem_id_len, dtype=torch.long))
            sem_vocab_size = codebook_size * sem_id_len

        unified_vocab_size = sem_vocab_size + user_ids_count + 32
        # User tokens immediately follow the semantic token range.
        self._user_token_offset = sem_vocab_size
        self._semantic_vocab_size = sem_vocab_size
        self._unified_vocab_size = unified_vocab_size

        self.pad_token_id = unified_vocab_size - 1
        self.eos_token_id = unified_vocab_size - 2
        self.decoder_start_token_id = unified_vocab_size - 3

        t5_cfg = T5Config(
            vocab_size=unified_vocab_size,
            d_model=embedding_dim,
            d_kv=d_kv,
            d_ff=dim_feedforward,
            num_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            is_encoder_decoder=True,
            use_cache=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            layer_norm_epsilon=layer_norm_eps,
            feed_forward_proj=activation,
            tie_word_embeddings=False,
            **kwargs,
        )
        self.config = t5_cfg
        self.model = T5ForConditionalGeneration(config=t5_cfg)
        self._init_weights(initializer_range)

        # Lazily populated on the first generate() call once the model is on its
        # target device.  Keyed per step: token_value → item indices on device.
        self._step_token_to_items_cache: list[dict[int, torch.Tensor]] | None = None
        self._batch_range_validated = False

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        """Truncated-normal weight initialisation matching the TIGER paper."""
        bound = 2 * initializer_range
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param.data)
                else:
                    nn.init.trunc_normal_(param.data, std=initializer_range, a=-bound, b=bound)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            elif "codebook" in name or "bos_embedding" in name:
                nn.init.trunc_normal_(param.data, std=initializer_range, a=-bound, b=bound)

    def _get_step_token_to_items(self) -> list[dict[int, torch.Tensor]]:
        """Return the cached per-step token→item mapping, building it on first call."""
        if self._step_token_to_items_cache is None:
            self._step_token_to_items_cache = (
                TIGERLogitsProcessor._compute_step_token_to_items(self.item_semantic_ids)
            )
        return self._step_token_to_items_cache

    def _fill_padding(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        fill_value: int,
    ) -> torch.Tensor:
        """Replace positions where mask is False with fill_value.

        Tokens are expected to be pre-offset vocabulary IDs (as produced by
        RQVAEItemTokenizer with level_offset_tokens=True plus offset_by_one=1).
        No further offset arithmetic is applied here.

        Args:
            tokens:     [..., sid_len] integer tensor of vocabulary IDs.
            mask:       [..., sid_len] boolean validity mask.
            fill_value: Value to write at padding positions.
        """
        return tokens.masked_fill(~mask.bool(), fill_value)

    def validate_batch_token_range(self, batch: Batch) -> None:
        """Verify target token IDs fall within the semantic token range.

        Guards against a semantic_id_mapping / datamodule offset mismatch where
        the mapping was built without offset but the batch carries offset tokens
        (or vice-versa), which would silently corrupt loss computation.
        """
        if "target_token_ids" not in batch:
            return
        max_target = int(batch["target_token_ids"].max().item())
        if max_target >= self._user_token_offset:
            raise ValueError(
                f"target_token_ids contain token {max_target}, but semantic token range "
                f"ends at {self._user_token_offset - 1}. "
                "Semantic-ID mapping and datamodule offset are inconsistent."
            )

    def forward(self, batch: Batch) -> RecModelOutput:
        """Encode history + user token, decode target semantic ID, return logits."""
        b = batch["user_ids"].size(0)

        # Flatten history: [B, history_len, sid_len] → [B, history_len * sid_len].
        # Padding positions become pad_token_id so the encoder ignores them.
        hist_tokens = self._fill_padding(
            batch["history_token_ids"], batch["history_token_mask"], self.config.pad_token_id
        )
        input_ids = hist_tokens.view(b, -1)

        # Append user token; TIGER hashes user IDs into a fixed user-token table.
        user_ids = batch.get("hashed_user_ids", batch["user_ids"])
        user_tokens = (user_ids % self.user_ids_count) + self._user_token_offset
        input_ids = torch.cat([input_ids, user_tokens.unsqueeze(1)], dim=1)

        flat_mask = batch["history_token_mask"].view(b, -1)
        user_mask = torch.ones((b, 1), device=flat_mask.device, dtype=flat_mask.dtype)
        attention_mask = torch.cat([flat_mask, user_mask], dim=1).long()

        if not self._batch_range_validated:
            self.validate_batch_token_range(batch)
            self._batch_range_validated = True

        decoder_input_ids = None
        if "target_token_ids" in batch:
            # Construct decoder_input_ids by right-shifting the targets: prepend
            # decoder_start_token_id and drop the last token.  This matches what
            # T5ForConditionalGeneration does internally when labels are passed,
            # but avoids the redundant internal CE computation (loss is computed
            # externally by the SFT training method).
            target = self._fill_padding(
                batch["target_token_ids"], batch["target_token_mask"], self.config.pad_token_id
            )
            start = torch.full(
                (b, 1), self.decoder_start_token_id,
                dtype=target.dtype, device=target.device,
            )
            decoder_input_ids = torch.cat([start, target[:, :-1]], dim=1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return RecModelOutput(logits=outputs.logits)

    def token_logprobs(
        self,
        batch: Batch,
        *,
        target_token_ids: torch.Tensor | None = None,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-token log-probabilities for a semantic-ID trajectory.

        Shape: [B, sem_id_len].
        """
        if target_token_ids is None:
            target_token_ids = batch["target_token_ids"]
        if target_token_mask is None:
            target_token_mask = batch.get("target_token_mask")

        forward_batch = {**batch, "target_token_ids": target_token_ids}
        if target_token_mask is not None:
            forward_batch["target_token_mask"] = target_token_mask

        output = self.forward(forward_batch)
        if output.logits is None:
            raise ValueError("forward() did not return logits.")

        log_probs = torch.log_softmax(output.logits, dim=-1)  # [B, sid_len, V]
        effective_mask = (
            target_token_mask
            if target_token_mask is not None
            else torch.ones_like(target_token_ids, dtype=torch.bool)
        )
        # Use pad_token_id for the gather index at padding positions (value is
        # discarded by the caller via the mask).
        target_ids = self._fill_padding(target_token_ids, effective_mask, self.config.pad_token_id)
        return torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)

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
        """Beam-search generation of semantic-ID recommendations.

        Returns:
            Tensor of shape [B, K, sid_len] with vocabulary-indexed token IDs
            (pre-offset, as stored in the semantic ID file), where K = num_return_sequences.
        """
        num_beams = num_beams or self._num_beams
        num_return_sequences = num_return_sequences or self._num_return_sequences
        max_new_tokens = max_new_tokens or self._sem_id_len
        constrained = constrained if constrained is not None else True

        b = batch["user_ids"].size(0)

        hist_tokens = self._fill_padding(
            batch["history_token_ids"], batch["history_token_mask"], self.config.pad_token_id
        )
        user_ids = batch.get("hashed_user_ids", batch["user_ids"])
        user_tokens = (user_ids % self.user_ids_count) + self._user_token_offset
        input_ids = torch.cat([hist_tokens.view(b, -1), user_tokens.unsqueeze(1)], dim=1)

        flat_mask = batch["history_token_mask"].view(b, -1)
        user_mask = torch.ones((b, 1), device=flat_mask.device, dtype=flat_mask.dtype)
        attention_mask = torch.cat([flat_mask, user_mask], dim=1).long()

        processors = LogitsProcessorList()
        if constrained and self.item_semantic_ids.numel() > 0:
            visited_item_ids = self._history_semantic_row_ids(batch, device=input_ids.device)
            try:
                if visited_item_ids is not None and visited_item_ids.numel() > 0:
                    max_idx = self.item_semantic_ids.size(0) - 1
                    out_of_range = (visited_item_ids >= 0) & (visited_item_ids > max_idx)
                    if out_of_range.any():
                        log.warning(
                            "%d visited semantic row IDs exceed semantic-ID table size (%d); ignoring them.",
                            out_of_range.sum().item(),
                            self.item_semantic_ids.size(0),
                        )
                        visited_item_ids = visited_item_ids.masked_fill(out_of_range, -1)
                    visited_item_ids = visited_item_ids.to(input_ids.device)

                processors.append(
                    TIGERLogitsProcessor(
                        self.item_semantic_ids,  # registered buffer, already on correct device
                        num_beams=num_beams,
                        visited_item_ids=visited_item_ids,
                        step_token_to_items=self._get_step_token_to_items(),
                    )
                )
            except Exception:
                log.warning("Constrained generation failed; falling back to unconstrained.", exc_info=True)
                constrained = False

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            logits_processor=processors,
            early_stopping=False,
            do_sample=False,
        )

        # outputs[:, 0] is the decoder start token; strip it.
        # Return pre-offset vocabulary IDs as-is (callers can map back via semantic_id_mapping).
        predictions = outputs[:, 1 : 1 + self._sem_id_len].view(b, num_return_sequences, self._sem_id_len)
        return predictions

    def _history_semantic_row_ids(
        self,
        batch: Batch,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return visited-item rows in ``item_semantic_ids`` space.

        FlowRec's integer item IDs reserve 0 for padding and are assigned in
        dataset order.  The constrained decoder's item table is sorted by raw
        item key, so those IDs are not interchangeable.  New collators provide
        ``history_semantic_row_ids`` directly; raw-item fallback keeps older
        batches usable as long as raw IDs are present.
        """
        row_ids = batch.get("history_semantic_row_ids")
        if torch.is_tensor(row_ids):
            return row_ids.to(device)

        raw_histories = batch.get("history_raw_items")
        if raw_histories is None:
            return None

        rows: list[list[int]] = []
        max_len = 0
        for history in raw_histories:
            row = [self._semantic_item_to_row.get(str(item), -1) for item in history]
            rows.append(row)
            max_len = max(max_len, len(row))

        if not rows:
            return None

        padded = [([-1] * (max_len - len(row)) + row) for row in rows]
        return torch.tensor(padded, dtype=torch.long, device=device)

    def config_dict(self) -> dict[str, Any]:
        return {
            "codebook_size": self._codebook_size,
            "sem_id_len": self._sem_id_len,
            "user_ids_count": self.user_ids_count,
            "embedding_dim": self.config.d_model,
            "num_heads": self.config.num_heads,
            "num_encoder_layers": self.config.num_layers,
            "num_decoder_layers": self.config.num_decoder_layers,
            "dim_feedforward": self.config.d_ff,
            "dropout": self.config.dropout_rate,
            "num_beams": self._num_beams,
            "num_return_sequences": self._num_return_sequences,
            "semantic_vocab_size": self._semantic_vocab_size,
            "unified_vocab_size": self._unified_vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
