from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Literal

import torch

HistoryPadding = Literal["left", "right"]


@dataclass(slots=True)
class SequenceBatchCollator:
    """Collate FlowRec sequence examples into tensors.

    The public batch format is unchanged, but the padded history and semantic-ID
    paths are built with tensor scatter/gather rather than nested Python loops.
    This matters most when semantic IDs are enabled because the old code looped
    over batch × history × sid_len in Python.
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
    _semantic_token_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    _semantic_token_mask: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.history_padding not in {"left", "right"}:
            raise ValueError("history_padding must be 'left' or 'right'")
        if self.semantic_ids is not None and self.sid_len is None:
            self.sid_len = max((len(v) for v in self.semantic_ids.values()), default=0)
        if self.semantic_ids is not None and not self.sid_len:
            raise ValueError("sid_len must be positive when semantic_ids are provided")

        if self.semantic_ids is None:
            self._semantic_item_to_row = {}
            self._semantic_token_ids = None
            self._semantic_token_mask = None
            return

        assert self.sid_len is not None
        ordered_items = sorted(self.semantic_ids)
        # Keep the public row ids 0-based for backward compatibility.  Gather
        # tensors reserve row 0 for missing/padded items, so gathered rows use
        # public_row + 1 internally.
        self._semantic_item_to_row = {item_id: row for row, item_id in enumerate(ordered_items)}

        token_rows = torch.full(
            (len(ordered_items) + 1, self.sid_len),
            int(self.pad_token_id),
            dtype=torch.long,
        )
        mask_rows = torch.zeros((len(ordered_items) + 1, self.sid_len), dtype=torch.bool)
        for row, item_id in enumerate(ordered_items, start=1):
            tokens = [int(x) for x in self.semantic_ids.get(item_id, [])[: self.sid_len]]
            if not tokens:
                continue
            n = min(len(tokens), self.sid_len)
            token_rows[row, :n] = torch.as_tensor(tokens[:n], dtype=torch.long)
            mask_rows[row, :n] = True
        self._semantic_token_ids = token_rows
        self._semantic_token_mask = mask_rows

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        history_lists = [ex["history_item_ids"] for ex in examples]
        history_item_ids, history_mask = self._collate_item_histories(history_lists)
        history_ratings = self._collate_history_float_values(
            self._history_feature_lists(examples, key="history_ratings", default=0.0)
        )
        history_verified_purchase = self._collate_history_bool_values(
            self._history_feature_lists(examples, key="history_verified_purchase", default=False)
        )

        batch: dict[str, Any] = {
            "user_ids": torch.as_tensor([int(ex["user_id"]) for ex in examples], dtype=torch.long),
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "history_ratings": history_ratings,
            "history_verified_purchase": history_verified_purchase,
            "target_item_ids": torch.as_tensor(
                [int(ex["target_item_id"]) for ex in examples], dtype=torch.long
            ),
            "target_ratings": torch.as_tensor(
                [float(ex["target_rating"]) for ex in examples], dtype=torch.float32
            ),
            "target_verified_purchase": torch.as_tensor(
                [_optional_bool(ex.get("target_verified_purchase")) for ex in examples],
                dtype=torch.bool,
            ),
            "target_timestamps": torch.as_tensor(
                [int(ex["target_timestamp"]) for ex in examples], dtype=torch.long
            ),
            "target_positions": torch.as_tensor(
                [int(ex["target_position"]) for ex in examples], dtype=torch.long
            ),
        }
        # user_hash_mod <= 0 means the model carries no user token - skip hashing.
        if self.user_hash_mod is not None and int(self.user_hash_mod) > 0:
            batch["hashed_user_ids"] = torch.as_tensor(
                [_hash_user_id(int(ex["user_id"]), self.user_hash_mod) for ex in examples],
                dtype=torch.long,
            )

        if "example_index" in examples[0]:
            if not all("example_index" in ex for ex in examples):
                raise ValueError("example_index must be present on every example in a batch")
            batch["example_indices"] = torch.as_tensor(
                [int(ex["example_index"]) for ex in examples],
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
            history_semantic_row_ids = self._collate_history_semantic_row_ids(histories)
            history_token_ids, history_token_mask = self._tokens_from_row_ids(history_semantic_row_ids)
            target_token_ids, target_token_mask = self._collate_target_semantic_ids(
                [ex["target_raw_item"] for ex in examples]
            )
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

    def _collate_padded_1d(
        self,
        histories: list[list[Any]],
        *,
        pad_value: int | float | bool,
        dtype: torch.dtype,
        transform: Callable[[Any], Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad ragged 1D histories using one scatter operation."""
        batch_size = len(histories)
        width = int(self.max_history_len)
        out = torch.full((batch_size, width), pad_value, dtype=dtype)
        mask = torch.zeros((batch_size, width), dtype=torch.bool)
        if batch_size == 0 or width <= 0:
            return out, mask

        lengths = torch.as_tensor(
            [min(len(history), width) for history in histories],
            dtype=torch.long,
        )
        total = int(lengths.sum().item())
        if total == 0:
            return out, mask

        values: list[Any] = []
        for history, length in zip(histories, lengths.tolist(), strict=False):
            if length <= 0:
                continue
            clipped = history[-length:]
            if transform is None:
                values.extend(clipped)
            else:
                values.extend(transform(x) for x in clipped)

        row_idx = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), lengths)
        start_idx = torch.where(
            lengths > 0,
            width - lengths if self.history_padding == "left" else torch.zeros_like(lengths),
            torch.zeros_like(lengths),
        )
        segment_starts = torch.cumsum(lengths, dim=0) - lengths
        rel_idx = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(segment_starts, lengths)
        col_idx = start_idx[row_idx] + rel_idx

        out[row_idx, col_idx] = torch.as_tensor(values, dtype=dtype)
        mask[row_idx, col_idx] = True
        return out, mask

    def _collate_item_histories(
        self,
        histories: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._collate_padded_1d(
            histories,
            pad_value=int(self.pad_item_id),
            dtype=torch.long,
            transform=int,
        )

    def _history_feature_lists(
        self,
        examples: list[dict[str, Any]],
        *,
        key: str,
        default: Any,
    ) -> list[list[Any]]:
        """Return history-aligned feature lists for each example."""
        features: list[list[Any]] = []
        for ex in examples:
            history_len = len(ex["history_item_ids"])
            values = list(ex.get(key, []))
            if not values:
                values = [default] * history_len
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
        values, _ = self._collate_padded_1d(
            histories,
            pad_value=0.0,
            dtype=torch.float32,
            transform=float,
        )
        return values

    def _collate_history_bool_values(
        self,
        histories: list[list[Any]],
    ) -> torch.Tensor:
        values, _ = self._collate_padded_1d(
            histories,
            pad_value=False,
            dtype=torch.bool,
            transform=_optional_bool,
        )
        return values

    def _collate_target_semantic_ids(
        self,
        raw_items: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._semantic_token_ids is not None
        assert self._semantic_token_mask is not None
        gather_rows = torch.as_tensor(
            [self._semantic_item_to_row.get(str(raw_item), -1) + 1 for raw_item in raw_items],
            dtype=torch.long,
        ).clamp_min_(0)
        return self._semantic_token_ids[gather_rows], self._semantic_token_mask[gather_rows]

    def _tokens_from_row_ids(self, row_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._semantic_token_ids is not None
        assert self._semantic_token_mask is not None
        gather_rows = (row_ids + 1).clamp_min(0)
        return self._semantic_token_ids[gather_rows], self._semantic_token_mask[gather_rows]

    def _collate_history_semantic_ids(
        self,
        histories: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_ids = self._collate_history_semantic_row_ids(histories)
        return self._tokens_from_row_ids(row_ids)

    def _collate_history_semantic_row_ids(
        self,
        histories: list[list[str]],
    ) -> torch.Tensor:
        row_ids, _ = self._collate_padded_1d(
            histories,
            pad_value=-1,
            dtype=torch.long,
            transform=lambda item: self._semantic_item_to_row.get(str(item), -1),
        )
        return row_ids


@dataclass(slots=True)
class DenseCausalCollator:
    """Collate dense-causal train rows for decoder-only-style training.

    Each example carries a window of inputs plus per-position labels (the
    inputs shifted by one) and a label mask. Inputs, labels, and label masks
    use the same configured padding side so their positions remain aligned.

    Batch schema:
      - user_ids:        [B]     long
      - input_item_ids:  [B, T]  long, padded with pad_item_id
      - attention_mask:  [B, T]  bool, True at real input positions
      - label_item_ids:  [B, T]  long, padded with pad_item_id
      - label_mask:      [B, T]  bool, True where the loss applies
    """

    max_seq_len: int = 20
    history_padding: HistoryPadding = "right"
    pad_item_id: int = 0
    include_raw: bool = True

    def __post_init__(self) -> None:
        if self.history_padding not in {"left", "right"}:
            raise ValueError("history_padding must be 'left' or 'right'")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        batch_size = len(examples)
        width = int(self.max_seq_len)
        input_item_ids = torch.full((batch_size, width), int(self.pad_item_id), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.bool)
        label_item_ids = torch.full((batch_size, width), int(self.pad_item_id), dtype=torch.long)
        label_mask = torch.zeros((batch_size, width), dtype=torch.bool)

        for row, ex in enumerate(examples):
            inputs = ex["history_item_ids"]
            labels = ex["label_item_ids"]
            mask = ex["label_mask"]
            if not (len(inputs) == len(labels) == len(mask)):
                raise ValueError(
                    "Dense example misaligned: "
                    f"inputs={len(inputs)}, labels={len(labels)}, mask={len(mask)}."
                )
            # Keep the most recent positions when a row exceeds the collator
            # width; input/label alignment is preserved because all three lists
            # are clipped identically.
            length = min(len(inputs), width)
            if length == 0:
                continue
            start = width - length if self.history_padding == "left" else 0
            end = start + length
            input_item_ids[row, start:end] = torch.as_tensor(
                [int(x) for x in inputs[-length:]], dtype=torch.long
            )
            attention_mask[row, start:end] = True
            label_item_ids[row, start:end] = torch.as_tensor(
                [int(x) for x in labels[-length:]], dtype=torch.long
            )
            label_mask[row, start:end] = torch.as_tensor(
                [bool(x) for x in mask[-length:]], dtype=torch.bool
            )

        batch: dict[str, Any] = {
            "user_ids": torch.as_tensor([int(ex["user_id"]) for ex in examples], dtype=torch.long),
            "input_item_ids": input_item_ids,
            "attention_mask": attention_mask,
            "label_item_ids": label_item_ids,
            "label_mask": label_mask,
        }

        if self.include_raw:
            batch.update(
                {
                    "raw_user_ids": [ex["raw_user_id"] for ex in examples],
                    "history_raw_items": [ex["history_raw_items"] for ex in examples],
                    "splits": [ex["split"] for ex in examples],
                }
            )

        return batch


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
