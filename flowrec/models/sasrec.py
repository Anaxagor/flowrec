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
 
        self.item_embedding = nn.Embedding(
            self.item_vocab_size,
            self.d_model,
            padding_idx=0,
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
                nn.LayerNorm(self.d_model, eps=1e-8)
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
                nn.LayerNorm(self.d_model, eps=1e-8)
            )
 
            self.forward_layers.append(PointWiseFeedForward(self.d_model, self.dropout_rate))
 
        self.last_layernorm = nn.LayerNorm(self.d_model, eps=1e-8)
 
        self.apply(self._init_weights)
 
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
 
            if module.bias is not None:
                module.bias.data.zero_()
 
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
 
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
 
        # positional embeddings
        positions = torch.arange(
            seq_len,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
 
        seqs = seqs + self.pos_embedding(positions)
 
        seqs = self.emb_dropout(seqs)
 
        # padding mask
        timeline_mask = ~history_mask
 
        seqs = seqs.masked_fill(
            timeline_mask.unsqueeze(-1),
            0.0,
        )
 
        # causal mask
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
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
                key_padding_mask=timeline_mask,
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
 
 
    def forward(self, batch: Batch) -> RecModelOutput:
 
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
