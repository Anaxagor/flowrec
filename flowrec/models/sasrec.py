from __future__ import annotations
 
from dataclasses import dataclass, asdict
 
import torch
import torch.nn as nn
 
from flowrec.models.base import (
    FlowRecModel,
    RecModelOutput,
    Batch,
    require_batch_keys,
)
 
 
@dataclass
class SASRecConfig:
    item_vocab_size: int = 0
    max_history_len: int = 20
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    pos_drop: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-8
    pad_item_id: int = 0
    exclude_history_in_generate: bool = False
 
 
class PointWiseFeedForward(nn.Module):
    """
    Classical SASRec point-wise feed-forward block.
    """
 
    def __init__(self, hidden_units: int, dropout: float):
        super().__init__()
 
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
 
        self.relu = nn.ReLU()
 
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
 
        x = x.transpose(-1, -2)
        x = self.conv1(x)
        x = self.dropout1(x)
 
        x = self.relu(x)
 
        x = self.conv2(x)
        x = self.dropout2(x)
 
        x = x.transpose(-1, -2)
 
        return x + residual
 
 
class SASRec(FlowRecModel):
    model_type = "sasrec"
    supported_train_layouts = ("prefix_target", "dense_causal")
 
    def __init__(self, config: SASRecConfig | dict):
        super().__init__()
 
        if isinstance(config, dict):
            config = SASRecConfig(**config)
 
        self.cfg = config
 
        self.item_vocab_size = config.item_vocab_size
        self.max_history_len = config.max_history_len
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        self.pad_item_id = config.pad_item_id
 
        self.item_embedding = nn.Embedding(
            self.item_vocab_size,
            self.d_model,
            padding_idx=self.pad_item_id,
        )
 
        self.pos_embedding = nn.Embedding(
            self.max_history_len,
            self.d_model,
        )
 
        self.emb_dropout = nn.Dropout(config.pos_drop)
 
        # SASRec blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
 
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
 
        for _ in range(self.num_layers):
            self.attention_layernorms.append(
                nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
            )
 
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=self.nhead,
                    dropout=self.dropout_rate,
                    batch_first=False,
                )
            )
 
            self.forward_layernorms.append(
                nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
            )
 
            self.forward_layers.append(PointWiseFeedForward(self.d_model, self.dropout_rate))
 
        self.last_layernorm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
 
        self.apply(self._init_weights)
 
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
 
            if module.bias is not None:
                module.bias.data.zero_()
 
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
 
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
 
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
 
    def _sanitize_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.long()
        pad_id = int(getattr(self, "pad_item_id", 0))
        valid = item_ids.ge(0) & item_ids.lt(int(self.item_vocab_size))
        return torch.where(valid, item_ids, torch.full_like(item_ids, pad_id))


    def _encode_history(
        self,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns sequence hidden states:
        shape = [B, T, D]
        """
 
        device = history_item_ids.device
        batch_size, seq_len = history_item_ids.shape
        if seq_len > self.max_history_len:
            raise ValueError(
                f"history length {seq_len} exceeds max_history_len={self.max_history_len}"
            )
 
        history_item_ids = self._sanitize_item_ids(history_item_ids)
        history_mask = history_mask & history_item_ids.ne(int(getattr(self, "pad_item_id", 0)))

        # item embeddings
        seqs = self.item_embedding(history_item_ids)
        seqs *= self.d_model ** 0.5
 
        # Position valid items by their order in the history rather than by
        # their padded tensor column. This makes equivalent left- and
        # right-padded histories use identical positional embeddings. For the
        # existing right-padded dense-causal batches this is unchanged:
        # valid prefix positions remain 0, 1, ..., history_length - 1.
        positions = history_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
 
        seqs = seqs + self.pos_embedding(positions)
 
        seqs = self.emb_dropout(seqs)
 
        # padding mask
        timeline_mask = ~history_mask
 
        seqs = seqs.masked_fill(
            timeline_mask.unsqueeze(-1),
            0.0,
        )
 
        # Causal + padding mask. With left padding, a padded query at the
        # beginning would otherwise have every key masked, which can yield
        # backend-dependent NaNs. Valid queries attend only to valid keys at or
        # before their timestep; padded queries attend only to themselves and
        # are zeroed again after the block.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        valid_query_mask = history_mask.unsqueeze(-1)
        valid_attention_mask = causal_mask.unsqueeze(0) | timeline_mask.unsqueeze(1)
        padded_query_mask = ~torch.eye(
            seq_len,
            device=device,
            dtype=torch.bool,
        ).unsqueeze(0)
        attention_mask = torch.where(
            valid_query_mask,
            valid_attention_mask,
            padded_query_mask,
        )
        attention_mask = (
            attention_mask.unsqueeze(1)
            .expand(batch_size, self.nhead, seq_len, seq_len)
            .reshape(batch_size * self.nhead, seq_len, seq_len)
        )
 
        # transformer blocks
        for i in range(self.num_layers):
            seqs_t = seqs.transpose(0, 1)
 
            q = self.attention_layernorms[i](seqs_t)
 
            mha_outputs, _ = self.attention_layers[i](
                q,
                seqs_t,
                seqs_t,
                attn_mask=attention_mask,
                need_weights=False,
            )
 
            seqs = (q + mha_outputs).transpose(0, 1)
 
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
 
            seqs = seqs.masked_fill(
                timeline_mask.unsqueeze(-1),
                0.0,
            )
 
 
        seqs = self.last_layernorm(seqs)
 
        return seqs
 
 
    def dense_item_hidden_states(self, batch: Batch) -> torch.Tensor:
        """Return dense-causal predictor states [B, T, D] without full-catalog logits.

        SFT's scalable cross-entropy consumes these states directly together with
        ``item_output_embedding_weight()``.  Keeping this as a model capability
        avoids allocating the [B, T, item_vocab_size] tensor during SCE training.
        """
        require_batch_keys(batch, ["input_item_ids", "attention_mask"])
        return self._encode_history(
            batch["input_item_ids"],
            batch["attention_mask"].to(torch.bool),
        )

    def item_output_embedding_weight(self) -> torch.Tensor:
        """Return the tied item/output embedding matrix [V, D]."""
        return self.item_embedding.weight

    def forward(self, batch: Batch) -> RecModelOutput:

        # Dense-causal batches: the causal mask in _encode_history makes every
        # hidden state a valid next-item predictor, so one forward pass scores
        # all positions at once. Sequence-relative positional IDs make valid
        # states independent of whether the tensor is left- or right-padded.
        if "input_item_ids" in batch:
            seq_out = self.dense_item_hidden_states(batch)
            logits = torch.matmul(
                seq_out,
                self.item_output_embedding_weight().transpose(0, 1),
            )
            return RecModelOutput(logits=logits)

        require_batch_keys(batch, ["history_item_ids", "history_mask"])

        history_item_ids = batch["history_item_ids"]
        history_mask = batch["history_mask"].to(torch.bool)
 
        seq_out = self._encode_history(
            history_item_ids,
            history_mask,
        )
 
        # last valid timestep
        seq_len = history_item_ids.size(1)
 
        positions = torch.arange(
            seq_len,
            device=history_item_ids.device,
        ).unsqueeze(0)
 
        last_idx = positions.masked_fill(
            ~history_mask,
            -1,
        ).max(dim=1).values
 
        if (last_idx < 0).any():
            raise ValueError("Found sequence with no valid history items.")
 
        batch_idx = torch.arange(
            history_item_ids.size(0),
            device=history_item_ids.device,
        )
 
        final_hidden = seq_out[batch_idx, last_idx]
 
        scores = torch.matmul(
            final_hidden,
            self.item_embedding.weight.transpose(0, 1),
        )
 
        return RecModelOutput(scores=scores)
 
    def score_items(
        self,
        batch: Batch,
        candidate_item_ids: torch.Tensor | None = None,
    ):
        scores = self.forward(batch).scores

        if candidate_item_ids is None:
            return scores

        candidate_item_ids = candidate_item_ids.to(scores.device).long()
        valid = candidate_item_ids.ge(0) & candidate_item_ids.lt(scores.size(1))
        safe_candidate_item_ids = candidate_item_ids.clamp(0, max(scores.size(1) - 1, 0))

        if candidate_item_ids.ndim == 1:
            # Case 1: one candidate per user: [B]
            if candidate_item_ids.size(0) == scores.size(0):
                out = scores.gather(1, safe_candidate_item_ids[:, None]).squeeze(1)
                return out.masked_fill(~valid, -torch.inf)

            # Case 2: same candidate set for every user: [N]
            out = scores[:, safe_candidate_item_ids]
            return out.masked_fill(~valid.unsqueeze(0), -torch.inf)

        if candidate_item_ids.ndim == 2:
            # Case 3: per-user candidates: [B, N]
            out = scores.gather(dim=1, index=safe_candidate_item_ids)
            return out.masked_fill(~valid, -torch.inf)

        raise ValueError("candidate_item_ids must have shape [N], [B], or [B, N].")

    def generate(
        self,
        batch: Batch,
        *,
        num_beams: int | None = None,
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        constrain_valid_sequences: bool = False,
        exclude_seen_items: bool = False,
    ):
        scores = self.forward(batch).scores
        scores[:, self.pad_item_id] = -torch.inf

        if (
            exclude_seen_items or self.cfg.exclude_history_in_generate
        ) and "history_item_ids" in batch:
            history = batch["history_item_ids"].to(scores.device).long()
            history_mask = batch.get("history_mask")
            if history_mask is None:
                history_mask = history.ne(self.pad_item_id)
            else:
                history_mask = history_mask.to(scores.device).bool()
            valid = (
                history_mask
                & history.ge(0)
                & history.lt(self.item_vocab_size)
                & history.ne(self.pad_item_id)
            )
            if valid.any():
                rows = torch.arange(scores.size(0), device=scores.device).unsqueeze(1)
                scores[rows.expand_as(history)[valid], history[valid]] = -torch.inf
 
        _, top_k_items = torch.topk(
            scores,
            k=int(num_return_sequences or 1),
            dim=1,
        )
 
        return top_k_items
 
    def config_dict(self) -> dict:
        return asdict(self.cfg)
 
 
def sasrec_from_config(config: dict):
    cfg = SASRecConfig(**config)
    model = SASRec(cfg)
    return model
