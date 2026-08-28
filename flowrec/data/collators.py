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
    """Collate dense-causal train rows.

    The original item-level dense batch is preserved for existing models.

    When ``semantic_ids`` are provided, the collator additionally builds a
    semantic-token causal stream for decoder-only generative models:

        item_0_tokens, item_1_tokens, ..., item_N_tokens

    and returns the one-token shifted inputs/labels together with a loss mask.

    Batch schema always contains:
      - user_ids:        [B]
      - input_item_ids:  [B, T]
      - attention_mask:  [B, T]
      - label_item_ids:  [B, T]
      - label_mask:      [B, T]

    With semantic IDs it additionally contains:
      - input_token_ids:       [B, S]
      - token_attention_mask:  [B, S]
      - label_token_ids:       [B, S]
      - token_label_mask:      [B, S]

    ``hashed_user_ids`` is emitted when ``user_hash_mod`` is configured so
    train and evaluation use exactly the same user-token mapping.
    """

    max_seq_len: int = 20
    history_padding: HistoryPadding = "right"

    pad_item_id: int = 0
    pad_token_id: int = 0

    semantic_ids: dict[str, list[int]] | None = None
    sid_len: int | None = None

    include_raw: bool = True
    user_hash_mod: int | None = None

    _semantic_token_lookup: dict[str, tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.history_padding not in {"left", "right"}:
            raise ValueError("history_padding must be 'left' or 'right'")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.user_hash_mod is not None and self.user_hash_mod <= 0:
            raise ValueError("user_hash_mod must be positive")

        if self.semantic_ids is None:
            if self.sid_len is not None:
                raise ValueError(
                    "sid_len was provided without semantic_ids"
                )
            return

        if self.sid_len is None:
            lengths = {
                len(tokens)
                for tokens in self.semantic_ids.values()
            }
            if len(lengths) != 1:
                raise ValueError(
                    "All semantic IDs must have the same length"
                )
            self.sid_len = next(iter(lengths), 0)

        if self.sid_len <= 0:
            raise ValueError(
                "sid_len must be positive when semantic_ids are provided"
            )

        lookup: dict[str, tuple[int, ...]] = {}

        for raw_item, tokens in self.semantic_ids.items():
            token_tuple = tuple(int(token) for token in tokens)

            if len(token_tuple) != self.sid_len:
                raise ValueError(
                    f"Semantic ID for item {raw_item!r} has "
                    f"length {len(token_tuple)}, expected {self.sid_len}"
                )

            lookup[str(raw_item)] = token_tuple

        self._semantic_token_lookup = lookup

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        batch = self._collate_item_level(examples)

        if self.user_hash_mod is not None:
            batch["hashed_user_ids"] = torch.as_tensor(
                [
                    _hash_user_id(
                        int(ex["user_id"]),
                        self.user_hash_mod,
                    )
                    for ex in examples
                ],
                dtype=torch.long,
            )

        if self.semantic_ids is not None:
            batch.update(
                self._collate_semantic_tokens(examples)
            )

        if "example_index" in examples[0]:
            if not all(
                "example_index" in ex
                for ex in examples
            ):
                raise ValueError(
                    "example_index must be present on every "
                    "example in a batch"
                )

            batch["example_indices"] = torch.as_tensor(
                [
                    int(ex["example_index"])
                    for ex in examples
                ],
                dtype=torch.long,
            )

        if self.include_raw:
            batch.update(
                {
                    "raw_user_ids": [
                        ex["raw_user_id"]
                        for ex in examples
                    ],
                    "history_raw_items": [
                        ex["history_raw_items"]
                        for ex in examples
                    ],
                    "splits": [
                        ex["split"]
                        for ex in examples
                    ],
                }
            )

        return batch

    def _collate_item_level(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Preserve the original dense-collator behaviour."""
        batch_size = len(examples)
        width = int(self.max_seq_len)

        input_item_ids = torch.full(
            (batch_size, width),
            int(self.pad_item_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, width),
            dtype=torch.bool,
        )
        label_item_ids = torch.full(
            (batch_size, width),
            int(self.pad_item_id),
            dtype=torch.long,
        )
        label_mask = torch.zeros(
            (batch_size, width),
            dtype=torch.bool,
        )

        for row, ex in enumerate(examples):
            inputs = ex["history_item_ids"]
            labels = ex["label_item_ids"]
            mask = ex["label_mask"]

            if not (
                len(inputs)
                == len(labels)
                == len(mask)
            ):
                raise ValueError(
                    "Dense example misaligned: "
                    f"inputs={len(inputs)}, "
                    f"labels={len(labels)}, "
                    f"mask={len(mask)}."
                )

            length = min(len(inputs), width)
            if length == 0:
                continue

            start = (
                width - length
                if self.history_padding == "left"
                else 0
            )
            end = start + length

            input_item_ids[row, start:end] = torch.as_tensor(
                [int(x) for x in inputs[-length:]],
                dtype=torch.long,
            )

            attention_mask[row, start:end] = True

            label_item_ids[row, start:end] = torch.as_tensor(
                [int(x) for x in labels[-length:]],
                dtype=torch.long,
            )

            label_mask[row, start:end] = torch.as_tensor(
                [bool(x) for x in mask[-length:]],
                dtype=torch.bool,
            )

        return {
            "user_ids": torch.as_tensor(
                [
                    int(ex["user_id"])
                    for ex in examples
                ],
                dtype=torch.long,
            ),
            "input_item_ids": input_item_ids,
            "attention_mask": attention_mask,
            "label_item_ids": label_item_ids,
            "label_mask": label_mask,
        }

    def _collate_semantic_tokens(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Build the semantic-token causal stream."""
        assert self.sid_len is not None

        batch_size = len(examples)
        width = int(self.max_seq_len)
        sid_len = int(self.sid_len)
        
        batch_item_width = max(
            (
                min(len(ex["history_raw_items"]), width)
                for ex in examples
            ),
            default=0,
        )

        # Allocate semantic tensors only to the longest stream in this batch.
        # Each row contains its history plus one final target item, followed by
        # the one-token causal shift.
        token_width = (batch_item_width + 1) * sid_len - 1

        input_token_ids = torch.full(
            (batch_size, token_width),
            int(self.pad_token_id),
            dtype=torch.long,
        )
        label_token_ids = torch.full(
            (batch_size, token_width),
            int(self.pad_token_id),
            dtype=torch.long,
        )
        token_attention_mask = torch.zeros(
            (batch_size, token_width),
            dtype=torch.bool,
        )
        token_label_mask = torch.zeros(
            (batch_size, token_width),
            dtype=torch.bool,
        )

        for row, ex in enumerate(examples):
            raw_inputs = list(
                ex["history_raw_items"]
            )[-width:]

            raw_labels = list(
                ex["label_raw_items"]
            )[-width:]

            item_label_mask = list(
                ex["label_mask"]
            )[-width:]

            if not (
                len(raw_inputs)
                == len(raw_labels)
                == len(item_label_mask)
            ):
                raise ValueError(
                    "Dense semantic example is misaligned: "
                    f"inputs={len(raw_inputs)}, "
                    f"labels={len(raw_labels)}, "
                    f"mask={len(item_label_mask)}."
                )

            if not raw_inputs:
                continue

            # The item-level dense representation contains:
            #
            # inputs = [i0, i1, ..., iN]
            # labels = [i1, i2, ..., iN+1]
            #
            # Therefore the complete autoregressive stream is:
            #
            # [i0, i1, ..., iN, iN+1]
            stream_items = (
                raw_inputs
                + [raw_labels[-1]]
            )

            stream_tokens: list[int] = []

            for raw_item in stream_items:
                semantic_id = (
                    self._semantic_token_lookup.get(
                        str(raw_item)
                    )
                )

                if semantic_id is None:
                    raise ValueError(
                        "Missing semantic ID for item "
                        f"{raw_item!r}"
                    )

                stream_tokens.extend(semantic_id)

            inputs = stream_tokens[:-1]
            labels = stream_tokens[1:]

            # The first sid_len - 1 labels only reconstruct the
            # remainder of the first history item. They are context,
            # not supervised next-item targets.
            semantic_label_mask: list[bool] = (
                [False] * (sid_len - 1)
            )

            for valid_target in item_label_mask:
                semantic_label_mask.extend(
                    [bool(valid_target)] * sid_len
                )

            if len(semantic_label_mask) != len(labels):
                raise RuntimeError(
                    "Semantic label mask is misaligned: "
                    f"{len(semantic_label_mask)} != "
                    f"{len(labels)}"
                )

            length = len(inputs)

            if length > token_width:
                raise RuntimeError(
                    "Semantic stream exceeds allocated "
                    f"width: {length} > {token_width}"
                )

            start = (
                token_width - length
                if self.history_padding == "left"
                else 0
            )
            end = start + length

            input_token_ids[row, start:end] = (
                torch.as_tensor(
                    inputs,
                    dtype=torch.long,
                )
            )

            label_token_ids[row, start:end] = (
                torch.as_tensor(
                    labels,
                    dtype=torch.long,
                )
            )

            token_attention_mask[row, start:end] = True

            token_label_mask[row, start:end] = (
                torch.as_tensor(
                    semantic_label_mask,
                    dtype=torch.bool,
                )
            )

        return {
            "input_token_ids": input_token_ids,
            "token_attention_mask": token_attention_mask,
            "label_token_ids": label_token_ids,
            "token_label_mask": token_label_mask,
        }


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
