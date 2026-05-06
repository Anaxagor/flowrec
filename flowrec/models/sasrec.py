from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowrec.models.base import FlowRecModel, RecModelOutput, Batch, require_batch_keys


@dataclass
class SASRecConfig:
    item_vocab_size: int = 0
    max_history_len: int = 20
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 2
    dim_ffn: int | None = None
    dropout: float = 0.1
    pos_drop: float = 0.1
    use_learned_pos: bool = True


class SASRec(FlowRecModel):
    model_type = "sasrec"

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
        dim_ffn = config.dim_ffn or 4 * self.d_model
        self.dropout = config.dropout

        self.item_embedding = nn.Embedding(self.item_vocab_size, self.d_model, padding_idx=0)
        self.score_projection = None
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_history_len, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=dim_ffn,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

        self.score_projection = nn.Linear(self.d_model, self.item_vocab_size)

    def forward(self, batch: Batch) -> RecModelOutput:
        require_batch_keys(batch, ["history_item_ids", "history_mask"])
        history_item_ids = batch["history_item_ids"]
        history_mask = batch["history_mask"].to(torch.bool)

        if self.item_embedding is None:
            raise RuntimeError("Call set_item_vocab_size after datamodule.setup()")

        item_emb = self.item_embedding(history_item_ids)
        item_emb = self.dropout(item_emb)

        pos_emb = self.pos_embedding[:, :history_item_ids.size(1)]
        # positional embedding should be 0 on padding positions
        pos_emb = pos_emb.masked_fill(~history_mask.unsqueeze(-1), 0.0)
        x = item_emb + pos_emb
        x = x.masked_fill(~history_mask.unsqueeze(-1), 0.0)
        x = self.layer_norm(x)

        src_key_padding_mask = (~history_mask)
        seq_len = history_item_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=history_item_ids.device),
            diagonal=1
        ).bool()

        seq_out = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        assert torch.isfinite(seq_out).all(), "NaN in seq_out"

        positions = torch.arange(seq_len, device=history_mask.device).unsqueeze(0)
        last_idx = positions.masked_fill(~history_mask, -1).max(dim=1).values
        if (last_idx < 0).any():
            raise ValueError("Found sequence with no valid history items.")
        batch_idx = torch.arange(history_item_ids.size(0), device=history_item_ids.device)
        pooled = seq_out[batch_idx, last_idx]

        pooled = self.layer_norm(pooled)
        scores = self.score_projection(pooled)
        return RecModelOutput(scores=scores)

    def score_items(self, batch: Batch, candidate_item_ids: torch.Tensor | None = None):
        scores = self.forward(batch).scores
        candidate_scores = scores[torch.arange(scores.size(0)), candidate_item_ids]
        return candidate_scores

    def generate(self, batch: Batch, num_return_sequences: int = 1):
        scores = self.forward(batch).scores
        _, top_k_items = torch.topk(scores, k=num_return_sequences, dim=1)
        return top_k_items

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def sasrec_from_config(config: dict):
    cfg = SASRecConfig(**config)
    model = SASRec(cfg)
    return model
