"""Vectorized prefix-target batch assembly.

``SequenceBatchCollator`` builds every batch from per-example Python dicts:
list slicing, per-element transforms, and raw-item -> semantic-row dict
lookups. At Yambda scale (10M+ examples, batch 2048, history ~50) that Python
work is the training bottleneck, and the per-worker ``to_dict(records)``
materialization in ``SequenceDataset`` multiplies RAM by the worker count.

This module flattens a ``SequenceDataset``'s example DataFrame ONCE into
contiguous tensors (items, ratings, verified flags + offsets) and assembles
batches with pure tensor gather/scatter from example indices. The produced
batches are IDENTICAL — same keys, dtypes, shapes, and values — to what
``SequenceBatchCollator`` emits, so every downstream consumer (SFT/TIGER,
GFlow via ``example_indices``, reward models reading ``history_ratings`` /
``target_ratings``, candidate samplers reading ``target_token_ids``) is
unaffected. Equivalence is enforced by tests/test_tensorized_batches.py.

The fast path deliberately does NOT cover:
- ``include_raw_in_batch=True`` (batches carry Python string lists; the
  legacy collator handles that path unchanged), and
- dense-causal training rows (``DenseCausalCollator``).

``maybe_tensorized_parts`` returns None in those cases and the datamodules
fall back to the legacy dataset/collator pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .collators import SequenceBatchCollator, _hash_user_id, _optional_bool
from .datasets import SequenceDataset

_ARRAYS_CACHE_ATTR = "_tensorized_arrays_cache"


@dataclass
class SequenceExampleArrays:
    """Flat tensors for one split's examples; built once per SequenceDataset."""

    split: str
    user_ids: torch.Tensor  # [N] long
    target_item_ids: torch.Tensor  # [N] long
    target_ratings: torch.Tensor  # [N] float32
    target_verified: torch.Tensor  # [N] bool
    target_timestamps: torch.Tensor  # [N] long
    target_positions: torch.Tensor  # [N] long
    history_offsets: torch.Tensor  # [N + 1] long, into the flat arrays
    history_items: torch.Tensor  # [total] long
    history_ratings: torch.Tensor  # [total] float32
    history_verified: torch.Tensor  # [total] bool

    @property
    def num_examples(self) -> int:
        return int(self.user_ids.numel())

    @classmethod
    def for_dataset(cls, dataset: SequenceDataset) -> "SequenceExampleArrays":
        cached = getattr(dataset, _ARRAYS_CACHE_ATTR, None)
        if cached is None:
            cached = cls.build(dataset.examples, split=str(dataset.split))
            setattr(dataset, _ARRAYS_CACHE_ATTR, cached)
        return cached

    @classmethod
    def build(cls, examples: Any, *, split: str) -> "SequenceExampleArrays":
        n = len(examples)
        histories = examples["history_item_ids"].to_numpy()
        lengths = np.fromiter((len(h) for h in histories), dtype=np.int64, count=n)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])

        flat_items = np.fromiter(
            chain.from_iterable(histories), dtype=np.int64, count=total
        )
        flat_ratings = _flat_history_feature(
            examples,
            key="history_ratings",
            lengths=lengths,
            total=total,
            default=0.0,
            dtype=np.float32,
            transform=None,
        )
        flat_verified = _flat_history_feature(
            examples,
            key="history_verified_purchase",
            lengths=lengths,
            total=total,
            default=False,
            dtype=np.bool_,
            transform=_cached_optional_bool(),
        )

        target_verified = np.fromiter(
            map(_cached_optional_bool(), examples["target_verified_purchase"].to_numpy()),
            dtype=np.bool_,
            count=n,
        )
        return cls(
            split=split,
            user_ids=torch.from_numpy(examples["user_id"].to_numpy(np.int64)),
            target_item_ids=torch.from_numpy(examples["target_item_id"].to_numpy(np.int64)),
            target_ratings=torch.from_numpy(examples["target_rating"].to_numpy(np.float32)),
            target_verified=torch.from_numpy(target_verified),
            target_timestamps=torch.from_numpy(examples["target_timestamp"].to_numpy(np.int64)),
            target_positions=torch.from_numpy(examples["target_position"].to_numpy(np.int64)),
            history_offsets=torch.from_numpy(offsets),
            history_items=torch.from_numpy(flat_items),
            history_ratings=torch.from_numpy(flat_ratings),
            history_verified=torch.from_numpy(flat_verified),
        )


class TensorizedSequenceDataset(Dataset):
    """Index-yielding view over a SequenceDataset for the vectorized collator.

    ``__getitem__`` returns the integer example index; all data lives in the
    shared ``SequenceExampleArrays`` held by the collator, so DataLoader
    workers share read-only pages instead of materializing per-worker record
    copies. ``examples``/``split`` proxy the source dataset for any code that
    inspects ``loader.dataset``.
    """

    def __init__(self, source: SequenceDataset) -> None:
        self.source = source
        self._length = len(source)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    @property
    def examples(self) -> Any:
        return self.source.examples

    @property
    def split(self) -> Any:
        return self.source.split


class TensorizedSequenceCollator:
    """Assemble SequenceBatchCollator-identical batches from example indices."""

    def __init__(
        self,
        base: SequenceBatchCollator,
        arrays: SequenceExampleArrays,
        *,
        item_id_to_semantic_row: torch.Tensor | None,
    ) -> None:
        if base.include_raw:
            raise ValueError(
                "TensorizedSequenceCollator does not support include_raw batches."
            )
        if base.semantic_ids is not None and item_id_to_semantic_row is None:
            raise ValueError(
                "item_id_to_semantic_row is required when semantic IDs are enabled."
            )
        self.base = base
        self.arrays = arrays
        self.item_id_to_semantic_row = item_id_to_semantic_row

        if base.user_hash_mod is not None:
            unique_users, inverse = torch.unique(arrays.user_ids, return_inverse=True)
            hashed_unique = torch.as_tensor(
                [
                    _hash_user_id(int(user), int(base.user_hash_mod))
                    for user in unique_users.tolist()
                ],
                dtype=torch.long,
            )
            self._hashed_user_ids: torch.Tensor | None = hashed_unique[inverse]
        else:
            self._hashed_user_ids = None

    def __call__(self, indices: Sequence[int]) -> dict[str, Any]:
        if len(indices) == 0:
            raise ValueError("Cannot collate an empty batch")
        arrays = self.arrays
        base = self.base
        idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = int(idx.numel())
        width = int(base.max_history_len)

        offsets_hi = arrays.history_offsets[idx + 1]
        lengths = (offsets_hi - arrays.history_offsets[idx]).clamp(max=width)
        total = int(lengths.sum().item())

        history_item_ids = torch.full(
            (batch_size, width), int(base.pad_item_id), dtype=torch.long
        )
        history_mask = torch.zeros((batch_size, width), dtype=torch.bool)
        history_ratings = torch.zeros((batch_size, width), dtype=torch.float32)
        history_verified = torch.zeros((batch_size, width), dtype=torch.bool)
        if base.semantic_ids is not None:
            history_semantic_row_ids = torch.full(
                (batch_size, width), -1, dtype=torch.long
            )

        if total > 0 and width > 0:
            row_idx = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.long), lengths
            )
            segment_starts = torch.cumsum(lengths, dim=0) - lengths
            rel_idx = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(
                segment_starts, lengths
            )
            start_col = (
                width - lengths
                if base.history_padding == "left"
                else torch.zeros_like(lengths)
            )
            col_idx = start_col[row_idx] + rel_idx
            # The last `length` history entries, matching history[-length:].
            flat_pos = (offsets_hi - lengths)[row_idx] + rel_idx

            flat_items = arrays.history_items[flat_pos]
            history_item_ids[row_idx, col_idx] = flat_items
            history_mask[row_idx, col_idx] = True
            history_ratings[row_idx, col_idx] = arrays.history_ratings[flat_pos]
            history_verified[row_idx, col_idx] = arrays.history_verified[flat_pos]
            if base.semantic_ids is not None:
                assert self.item_id_to_semantic_row is not None
                history_semantic_row_ids[row_idx, col_idx] = self.item_id_to_semantic_row[
                    flat_items
                ]

        batch: dict[str, Any] = {
            "user_ids": arrays.user_ids[idx],
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "history_ratings": history_ratings,
            "history_verified_purchase": history_verified,
            "target_item_ids": arrays.target_item_ids[idx],
            "target_ratings": arrays.target_ratings[idx],
            "target_verified_purchase": arrays.target_verified[idx],
            "target_timestamps": arrays.target_timestamps[idx],
            "target_positions": arrays.target_positions[idx],
        }
        if self._hashed_user_ids is not None:
            batch["hashed_user_ids"] = self._hashed_user_ids[idx]
        batch["example_indices"] = idx

        if base.semantic_ids is not None:
            assert self.item_id_to_semantic_row is not None
            history_token_ids, history_token_mask = base._tokens_from_row_ids(
                history_semantic_row_ids
            )
            target_rows = self.item_id_to_semantic_row[arrays.target_item_ids[idx]]
            target_token_ids, target_token_mask = base._tokens_from_row_ids(target_rows)
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


def maybe_tensorized_parts(
    dataset: Any,
    collator: Any,
    *,
    id2item: Sequence[str],
    enabled: bool = True,
) -> tuple[TensorizedSequenceDataset, TensorizedSequenceCollator] | None:
    """Return a vectorized (dataset, collate_fn) pair, or None to use legacy.

    The fast path applies to prefix-target batches without raw payloads; the
    legacy pair keeps handling dense-causal rows and include_raw batches.

    Semantic-ID lookups go through ``id2item`` (item_id -> raw item) instead
    of the per-example raw strings the legacy collator uses. Both are
    equivalent because factorize_interaction_ids and the example builders
    guarantee ``raw_item == id2item[item_id]`` for every history/target entry.
    """
    if not enabled:
        return None
    if not isinstance(dataset, SequenceDataset) or not isinstance(
        collator, SequenceBatchCollator
    ):
        return None
    if collator.include_raw or len(dataset) == 0:
        return None

    arrays = SequenceExampleArrays.for_dataset(dataset)
    item_id_to_semantic_row: torch.Tensor | None = None
    if collator.semantic_ids is not None:
        item_id_to_semantic_row = torch.as_tensor(
            [
                collator._semantic_item_to_row.get(str(raw_item), -1)
                for raw_item in id2item
            ],
            dtype=torch.long,
        )
    return (
        TensorizedSequenceDataset(dataset),
        TensorizedSequenceCollator(
            collator, arrays, item_id_to_semantic_row=item_id_to_semantic_row
        ),
    )


def _cached_optional_bool():
    """_optional_bool with a value cache; example payloads repeat a few values."""
    cache: dict[Any, bool] = {}

    def convert(value: Any) -> bool:
        try:
            return cache[value]
        except KeyError:
            result = _optional_bool(value)
            cache[value] = result
            return result
        except TypeError:
            return _optional_bool(value)

    return convert


def _flat_history_feature(
    examples: Any,
    *,
    key: str,
    lengths: np.ndarray,
    total: int,
    default: Any,
    dtype: Any,
    transform: Any,
) -> np.ndarray:
    """Flatten a history-aligned feature column with legacy default semantics.

    Mirrors SequenceBatchCollator._history_feature_lists: an empty per-row
    list falls back to `default * history_len`; a non-empty length mismatch is
    an error.
    """
    n = len(lengths)
    if key not in examples.columns:
        return np.full(total, default, dtype=dtype)

    column = examples[key].to_numpy()
    feature_lengths = np.fromiter((len(v) for v in column), dtype=np.int64, count=n)
    aligned = feature_lengths == lengths
    if not bool(aligned.all()):
        empty = feature_lengths == 0
        bad = ~(aligned | empty)
        if bool(bad.any()):
            row = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"Example field {key!r} has length {int(feature_lengths[row])}, but "
                f"history_item_ids has length {int(lengths[row])}."
            )
        # Replace empty rows by default-filled lists of the history length.
        column = column.copy()
        for row in np.flatnonzero(empty & (lengths > 0)):
            column[row] = [default] * int(lengths[row])

    values = chain.from_iterable(column)
    if transform is not None:
        values = map(transform, values)
    return np.fromiter(values, dtype=dtype, count=total)
