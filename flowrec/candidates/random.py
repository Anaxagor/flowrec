from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flowrec.models.base import Batch, FlowRecModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


@dataclass(frozen=True, slots=True)
class RandomSemanticCandidateConfig:
    """Configuration for simple random semantic-ID candidate sampling."""

    num_negative_samples: int = 1
    include_positive: bool = True


class RandomSemanticIDCandidateSampler:
    """Build positive + random-negative semantic-ID trajectories.

    The first candidate is the observed target trajectory when include_positive=True.
    Negatives are sampled from the model's valid item semantic-ID table.

    This is intentionally simple and serves as the first GFlowGR trajectory sampler.
    """

    def __init__(self, cfg: RandomSemanticCandidateConfig | None = None) -> None:
        self.cfg = cfg or RandomSemanticCandidateConfig()

    def sample(
        self,
        batch: Batch,
        *,
        model: FlowRecModel | None = None,
        item_semantic_ids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> SemanticTrajectoryBatch:
        if "target_token_ids" not in batch:
            raise KeyError("RandomSemanticIDCandidateSampler requires batch['target_token_ids'].")

        target_token_ids = batch["target_token_ids"]
        if not torch.is_tensor(target_token_ids):
            raise TypeError("batch['target_token_ids'] must be a torch.Tensor.")
        target_token_ids = target_token_ids.long()
        device = target_token_ids.device
        bsz, sid_len = target_token_ids.shape

        target_token_mask = batch.get("target_token_mask")
        if target_token_mask is None or not torch.is_tensor(target_token_mask):
            target_token_mask = torch.ones_like(target_token_ids, dtype=torch.bool, device=device)
        else:
            target_token_mask = target_token_mask.to(device).bool()

        neg_k = int(self.cfg.num_negative_samples)
        if neg_k < 0:
            raise ValueError("num_negative_samples must be non-negative.")

        candidates: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        is_positive_parts: list[torch.Tensor] = []
        candidate_rows: list[torch.Tensor] = []

        if self.cfg.include_positive:
            candidates.append(target_token_ids.unsqueeze(1))
            masks.append(target_token_mask.unsqueeze(1))
            is_positive_parts.append(torch.ones((bsz, 1), device=device, dtype=torch.bool))
            candidate_rows.append(torch.full((bsz, 1), -1, device=device, dtype=torch.long))

        if neg_k > 0:
            table = self._resolve_item_semantic_ids(model=model, item_semantic_ids=item_semantic_ids)
            table = table.to(device).long()
            if table.ndim != 2 or table.size(1) != sid_len:
                raise ValueError(
                    "item_semantic_ids must have shape [num_items, sid_len] matching target_token_ids. "
                    f"Got {tuple(table.shape)} and target sid_len={sid_len}."
                )

            valid_rows = self._valid_rows(table)
            if valid_rows.numel() == 0:
                raise ValueError("No valid item semantic IDs are available for random sampling.")

            row_positions = torch.randint(
                low=0,
                high=int(valid_rows.numel()),
                size=(bsz, neg_k),
                device=device,
                generator=generator,
            )
            sampled_rows = valid_rows[row_positions]
            negative_tokens = table[sampled_rows]

            negative_tokens, sampled_rows = self._resample_matches(
                table=table,
                valid_rows=valid_rows,
                negative_tokens=negative_tokens,
                sampled_rows=sampled_rows,
                target_token_ids=target_token_ids,
                generator=generator,
            )

            negative_mask = torch.ones_like(negative_tokens, dtype=torch.bool)
            candidates.append(negative_tokens)
            masks.append(negative_mask)
            is_positive_parts.append(torch.zeros((bsz, neg_k), device=device, dtype=torch.bool))
            candidate_rows.append(sampled_rows)

        if not candidates:
            raise ValueError("No candidates were built. Enable include_positive or num_negative_samples > 0.")

        return SemanticTrajectoryBatch(
            candidate_token_ids=torch.cat(candidates, dim=1),
            candidate_token_mask=torch.cat(masks, dim=1),
            is_positive=torch.cat(is_positive_parts, dim=1),
            candidate_item_row_ids=torch.cat(candidate_rows, dim=1) if candidate_rows else None,
            extra={"sampler": "random_semantic_id"},
        )

    def _resolve_item_semantic_ids(
        self,
        *,
        model: FlowRecModel | None,
        item_semantic_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if item_semantic_ids is not None:
            return item_semantic_ids
        if model is not None and hasattr(model, "item_semantic_ids"):
            table = getattr(model, "item_semantic_ids")
            if torch.is_tensor(table):
                return table
        raise ValueError(
            "Random negative sampling requires item_semantic_ids or a model with item_semantic_ids."
        )

    def _valid_rows(self, table: torch.Tensor) -> torch.Tensor:
        return torch.arange(table.size(0), device=table.device, dtype=torch.long)

    def _resample_matches(
        self,
        *,
        table: torch.Tensor,
        valid_rows: torch.Tensor,
        negative_tokens: torch.Tensor,
        sampled_rows: torch.Tensor,
        target_token_ids: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, neg_k, _ = negative_tokens.shape
        for _ in range(3):
            matches = negative_tokens.eq(target_token_ids.unsqueeze(1)).all(dim=-1)
            if not matches.any():
                break
            n_matches = int(matches.sum().item())
            repl_pos = torch.randint(
                low=0,
                high=int(valid_rows.numel()),
                size=(n_matches,),
                device=negative_tokens.device,
                generator=generator,
            )
            repl_rows = valid_rows[repl_pos]
            negative_tokens[matches] = table[repl_rows]
            sampled_rows[matches] = repl_rows
        return negative_tokens, sampled_rows
