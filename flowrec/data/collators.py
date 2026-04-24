from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

HistoryPadding = Literal["left", "right"]


@dataclass(slots=True)
class SequenceBatchCollator:
    """Collate FlowRec sequence examples into tensors.

    Outputs item-level fields for SASRec-style models and, when semantic IDs are
    provided, token-level fields for TIGER/GPT4Rec-style generative models.
    """

    max_history_len: int = 20
    history_padding: HistoryPadding = "left"
    pad_item_id: int = 0
    pad_token_id: int = 0
    semantic_ids: dict[str, list[int]] | None = None
    sid_len: int | None = None
    include_raw: bool = True

    def __post_init__(self) -> None:
        if self.history_padding not in {"left", "right"}:
            raise ValueError("history_padding must be 'left' or 'right'")
        if self.semantic_ids is not None and self.sid_len is None:
            self.sid_len = max((len(v) for v in self.semantic_ids.values()), default=0)
        if self.semantic_ids is not None and not self.sid_len:
            raise ValueError("sid_len must be positive when semantic_ids are provided")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        history_item_ids, history_mask = self._collate_item_histories(
            [ex["history_item_ids"] for ex in examples]
        )

        batch: dict[str, Any] = {
            "user_ids": torch.tensor([int(ex["user_id"]) for ex in examples], dtype=torch.long),
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "target_item_ids": torch.tensor(
                [int(ex["target_item_id"]) for ex in examples], dtype=torch.long
            ),
            "target_ratings": torch.tensor(
                [float(ex["target_rating"]) for ex in examples], dtype=torch.float32
            ),
            "target_timestamps": torch.tensor(
                [int(ex["target_timestamp"]) for ex in examples], dtype=torch.long
            ),
            "target_positions": torch.tensor(
                [int(ex["target_position"]) for ex in examples], dtype=torch.long
            ),
        }

        if self.include_raw:
            batch.update(
                {
                    "raw_user_ids": [ex["raw_user_id"] for ex in examples],
                    "history_raw_items": [ex["history_raw_items"] for ex in examples],
                    "target_raw_items": [ex["target_raw_item"] for ex in examples],
                    "splits": [ex["split"] for ex in examples],
                }
            )

        if self.semantic_ids is not None:
            history_token_ids, history_token_mask = self._collate_history_semantic_ids(
                [ex["history_raw_items"] for ex in examples]
            )
            target_token_ids, target_token_mask = self._collate_target_semantic_ids(
                [ex["target_raw_item"] for ex in examples]
            )
            batch.update(
                {
                    "history_token_ids": history_token_ids,
                    "history_token_mask": history_token_mask,
                    "target_token_ids": target_token_ids,
                    "target_token_mask": target_token_mask,
                }
            )

        return batch

    def _collate_item_histories(
        self,
        histories: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        masks: list[list[bool]] = []

        for history in histories:
            clipped = list(history[-self.max_history_len :])
            pad_len = self.max_history_len - len(clipped)
            if self.history_padding == "left":
                row = [self.pad_item_id] * pad_len + clipped
                mask = [False] * pad_len + [True] * len(clipped)
            else:
                row = clipped + [self.pad_item_id] * pad_len
                mask = [True] * len(clipped) + [False] * pad_len
            rows.append(row)
            masks.append(mask)

        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    def _sid_tensor(self, raw_item_id: str) -> tuple[list[int], list[bool]]:
        assert self.semantic_ids is not None
        assert self.sid_len is not None

        tokens = list(self.semantic_ids.get(raw_item_id, []))[: self.sid_len]
        pad_len = self.sid_len - len(tokens)
        return tokens + [self.pad_token_id] * pad_len, [True] * len(tokens) + [False] * pad_len

    def _collate_target_semantic_ids(
        self,
        raw_items: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        for raw_item in raw_items:
            tokens, mask = self._sid_tensor(raw_item)
            token_rows.append(tokens)
            mask_rows.append(mask)
        return torch.tensor(token_rows, dtype=torch.long), torch.tensor(mask_rows, dtype=torch.bool)

    def _collate_history_semantic_ids(
        self,
        histories: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.sid_len is not None

        batch_tokens: list[list[list[int]]] = []
        batch_masks: list[list[list[bool]]] = []

        pad_tokens = [self.pad_token_id] * self.sid_len
        pad_mask = [False] * self.sid_len

        for history in histories:
            clipped = list(history[-self.max_history_len :])
            token_rows: list[list[int]] = []
            mask_rows: list[list[bool]] = []
            for raw_item in clipped:
                tokens, mask = self._sid_tensor(raw_item)
                token_rows.append(tokens)
                mask_rows.append(mask)

            pad_len = self.max_history_len - len(token_rows)
            if self.history_padding == "left":
                token_rows = [pad_tokens[:] for _ in range(pad_len)] + token_rows
                mask_rows = [pad_mask[:] for _ in range(pad_len)] + mask_rows
            else:
                token_rows = token_rows + [pad_tokens[:] for _ in range(pad_len)]
                mask_rows = mask_rows + [pad_mask[:] for _ in range(pad_len)]

            batch_tokens.append(token_rows)
            batch_masks.append(mask_rows)

        return torch.tensor(batch_tokens, dtype=torch.long), torch.tensor(batch_masks, dtype=torch.bool)
