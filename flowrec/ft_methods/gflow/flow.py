from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from flowrec.models.base import Batch


LogZContext = Literal["global", "semantic_history", "vector"]


@dataclass(frozen=True, slots=True)
class SemanticHistoryLogZConfig:
    """Configuration for logZ(history semantic IDs)."""

    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.0
    pad_token_id: int = 0
    init_value: float = 0.0


class GlobalLogZ(nn.Module):
    """Trainable scalar logZ baseline."""

    def __init__(self, init_value: float = 0.0) -> None:
        super().__init__()
        self.log_z = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.log_z


class VectorLogZ(nn.Module):
    """Trainable vector logZ baseline summed into one scalar per context.

    This keeps the trajectory-balance interface scalar while allowing logZ to
    be parameterized by a small set of independent trainable components, e.g.
    one component per semantic-ID position.
    """

    def __init__(self, vector_size: int, init_value: float = 0.0) -> None:
        super().__init__()
        if vector_size <= 0:
            raise ValueError("vector_size must be positive for VectorLogZ.")
        value = float(init_value) / float(vector_size)
        self.log_z_vector = nn.Parameter(
            torch.full((int(vector_size),), value, dtype=torch.float32)
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        log_z = self.log_z_vector.sum()
        user_ids = batch.get("user_ids")
        if torch.is_tensor(user_ids):
            return log_z.expand(user_ids.size(0))
        return log_z


class SemanticHistoryLogZ(nn.Module):
    """Predict logZ from a user's history semantic-token IDs.

    The model is intentionally small: embed semantic tokens, masked-mean pool
    over all valid history tokens, and run a two-layer MLP. The output is [B],
    so trajectory balance uses one partition estimate per user context.
    """

    def __init__(self, cfg: SemanticHistoryLogZConfig) -> None:
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be positive for SemanticHistoryLogZ.")
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self._init_output(float(cfg.init_value))

    def forward(self, batch: Batch) -> torch.Tensor:
        if "history_token_ids" not in batch:
            raise KeyError("SemanticHistoryLogZ requires batch['history_token_ids'].")

        token_ids = batch["history_token_ids"]
        if not torch.is_tensor(token_ids):
            raise TypeError("batch['history_token_ids'] must be a torch.Tensor.")
        token_ids = token_ids.long()

        mask = batch.get("history_token_mask")
        if mask is None or not torch.is_tensor(mask):
            mask = token_ids.ne(int(self.cfg.pad_token_id))
        mask = mask.to(token_ids.device).bool()

        max_token = int(token_ids.max().item()) if token_ids.numel() > 0 else -1
        if max_token >= self.cfg.vocab_size:
            raise ValueError(
                f"history_token_ids contain token {max_token}, but logZ vocab_size is "
                f"{self.cfg.vocab_size}."
            )

        embeddings = self.token_embedding(token_ids)  # [B, H, L, D]
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=(1, 2))
        denom = weights.sum(dim=(1, 2)).clamp_min(1.0)
        pooled = pooled / denom
        return self.net(pooled).squeeze(-1)

    def _init_output(self, init_value: float) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, init_value)
