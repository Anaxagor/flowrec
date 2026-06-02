from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
    user_hash_mod: int | None = None
    _semantic_item_to_row: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.history_padding not in {"left", "right"}:
            raise ValueError("history_padding must be 'left' or 'right'")
        if self.semantic_ids is not None and self.sid_len is None:
            self.sid_len = max((len(v) for v in self.semantic_ids.values()), default=0)
        if self.semantic_ids is not None and not self.sid_len:
            raise ValueError("sid_len must be positive when semantic_ids are provided")
        self._semantic_item_to_row = (
            {item_id: row for row, item_id in enumerate(sorted(self.semantic_ids))}
            if self.semantic_ids is not None
            else {}
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        history_item_ids, history_mask = self._collate_item_histories(
            [ex["history_item_ids"] for ex in examples]
        )
        history_ratings = self._collate_history_float_values(
            self._history_feature_lists(
                examples,
                key="history_ratings",
                default=0.0,
            )
        )
        history_verified_purchase = self._collate_history_bool_values(
            self._history_feature_lists(
                examples,
                key="history_verified_purchase",
                default=False,
            )
        )

        batch: dict[str, Any] = {
            "user_ids": torch.tensor([int(ex["user_id"]) for ex in examples], dtype=torch.long),
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "history_ratings": history_ratings,
            "history_verified_purchase": history_verified_purchase,
            "target_item_ids": torch.tensor(
                [int(ex["target_item_id"]) for ex in examples], dtype=torch.long
            ),
            "target_ratings": torch.tensor(
                [float(ex["target_rating"]) for ex in examples], dtype=torch.float32
            ),
            "target_verified_purchase": torch.tensor(
                [_optional_bool(ex.get("target_verified_purchase")) for ex in examples],
                dtype=torch.bool,
            ),
            "target_timestamps": torch.tensor(
                [int(ex["target_timestamp"]) for ex in examples], dtype=torch.long
            ),
            "target_positions": torch.tensor(
                [int(ex["target_position"]) for ex in examples], dtype=torch.long
            ),
        }
        if self.user_hash_mod is not None:
            batch["hashed_user_ids"] = torch.tensor(
                [_hash_user_id(int(ex["user_id"]), self.user_hash_mod) for ex in examples],
                dtype=torch.long,
            )

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
            histories = [ex["history_raw_items"] for ex in examples]
            history_token_ids, history_token_mask = self._collate_history_semantic_ids(
                histories
            )
            target_token_ids, target_token_mask = self._collate_target_semantic_ids(
                [ex["target_raw_item"] for ex in examples]
            )
            history_semantic_row_ids = self._collate_history_semantic_row_ids(histories)
            batch.update(
                {
                    "history_token_ids": history_token_ids,
                    "history_token_mask": history_token_mask,
                    "history_semantic_row_ids": history_semantic_row_ids,
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

    def _history_feature_lists(
        self,
        examples: list[dict[str, Any]],
        *,
        key: str,
        default: Any,
    ) -> list[list[Any]]:
        """Return history-aligned feature lists for each example.

        Newly-built examples carry history_ratings and history_verified_purchase.
        For backwards compatibility with older cached examples, missing feature
        lists are filled with a neutral value.  Non-empty mismatched feature
        lists are rejected to avoid silently misaligning ratings with item IDs.
        """
        features: list[list[Any]] = []
        for ex in examples:
            history_len = len(ex["history_item_ids"])
            values = list(ex.get(key, []))
            if not values:
                values = [default for _ in range(history_len)]
            elif len(values) != history_len:
                raise ValueError(
                    f"Example field {key!r} has length {len(values)}, but "
                    f"history_item_ids has length {history_len}."
                )
            features.append(values)
        return features

    def _collate_history_float_values(
        self,
        histories: list[list[Any]],
    ) -> torch.Tensor:
        rows: list[list[float]] = []
        for history in histories:
            clipped = [float(x) for x in history[-self.max_history_len :]]
            pad_len = self.max_history_len - len(clipped)
            if self.history_padding == "left":
                row = [0.0] * pad_len + clipped
            else:
                row = clipped + [0.0] * pad_len
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    def _collate_history_bool_values(
        self,
        histories: list[list[Any]],
    ) -> torch.Tensor:
        rows: list[list[bool]] = []
        for history in histories:
            clipped = [_optional_bool(x) for x in history[-self.max_history_len :]]
            pad_len = self.max_history_len - len(clipped)
            if self.history_padding == "left":
                row = [False] * pad_len + clipped
            else:
                row = clipped + [False] * pad_len
            rows.append(row)
        return torch.tensor(rows, dtype=torch.bool)

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

    def _collate_history_semantic_row_ids(
        self,
        histories: list[list[str]],
    ) -> torch.Tensor:
        rows: list[list[int]] = []

        for history in histories:
            clipped = list(history[-self.max_history_len :])
            row = [self._semantic_item_to_row.get(str(item), -1) for item in clipped]
            pad_len = self.max_history_len - len(row)
            if self.history_padding == "left":
                row = [-1] * pad_len + row
            else:
                row = row + [-1] * pad_len
            rows.append(row)

        return torch.tensor(rows, dtype=torch.long)


def _optional_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", ""}:
            return False
    return bool(value)


def _hash_user_id(user_id: int, modulo: int) -> int:
    try:
        import murmurhash

        return int(murmurhash.hash(str(user_id)) % modulo)
    except ImportError:
        digest = hashlib.blake2b(str(user_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % modulo
