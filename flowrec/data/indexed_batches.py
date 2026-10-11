"""Tensorized training batches whose indices refer to reward-cache rows.

The reward cache follows the fine-tuning subset's order, whereas sequence
arrays follow the full dataset's order. Keep those two index spaces separate,
including when a subset repeats examples or is not sorted.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .tensor_batches import maybe_tensorized_parts


class ExamplePositions(Dataset):
    """Only a length crosses the worker boundary; no source DataFrame."""

    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


class IndexedTensorCollator:
    def __init__(self, base: Any, source_indices: torch.Tensor | None) -> None:
        self.base = base
        self.source_indices = source_indices

    def __call__(self, positions: Sequence[int]) -> dict[str, Any]:
        cache_indices = torch.as_tensor(positions, dtype=torch.long)
        source = (
            cache_indices
            if self.source_indices is None
            else self.source_indices[cache_indices]
        )
        batch = self.base(source)
        # The tensorized collator emits global indices. Rewards are indexed
        # by the subset position, never by the original training row.
        batch["example_indices"] = cache_indices
        return batch


class IndexedRecordCollator:
    """Legacy path for raw payloads, dense layouts, and custom datasets."""

    def __init__(
        self, dataset: Dataset, base: Any, source_indices: torch.Tensor | None
    ) -> None:
        self.dataset = dataset
        self.base = base
        self.source_indices = source_indices

    def __call__(self, positions: Sequence[int]) -> dict[str, Any]:
        records = []
        for position in positions:
            position = int(position)
            source = position if self.source_indices is None else int(self.source_indices[position])
            record = dict(self.dataset[source])
            record["example_index"] = position
            records.append(record)
        batch = self.base(records)
        batch["example_indices"] = torch.as_tensor(positions, dtype=torch.long)
        return batch


def create_indexed_parts(
    dm: Any,
    subset_indices: Sequence[int] | torch.Tensor | None,
    *,
    collator: Any | None = None,
    tensorized: bool = True,
) -> tuple[ExamplePositions, Any]:
    """Build worker-safe parts for a reward-cache-indexed DataLoader.

    ``None`` uses the complete training split without an identity index map.
    Explicit subsets retain their order and duplicates. The fast path only
    retains flat sequence tensors and the semantic lookup tables in workers;
    the main process may continue to own the source DataFrame.

    Pass a custom/dense collator explicitly to use its existing record path.
    Otherwise the prefix-target collator matches the GFlow loader's original
    behavior. ``tensorized=False`` or ``dm.cfg.tensorized_batches=False``
    also preserves that legacy path.
    """
    source = dm.train_dataset
    if source is None:
        raise RuntimeError("Call setup() before constructing the indexed train loader.")
    source_indices = None
    if subset_indices is not None:
        source_indices = torch.as_tensor(subset_indices, dtype=torch.long, device="cpu")
        if source_indices.ndim != 1:
            raise ValueError("subset_indices must be one-dimensional.")
        if source_indices.numel() and (
            int(source_indices.min()) < 0 or int(source_indices.max()) >= len(source)
        ):
            raise IndexError("subset_indices contains an index outside the training dataset.")
    length = len(source) if source_indices is None else int(source_indices.numel())
    base = dm._make_collator() if collator is None else collator
    parts = maybe_tensorized_parts(
        source,
        base,
        id2item=getattr(dm, "id2item", ()),
        enabled=bool(tensorized and getattr(dm.cfg, "tensorized_batches", True)),
    )
    if parts is not None:
        _, tensor_collator = parts
        return ExamplePositions(length), IndexedTensorCollator(tensor_collator, source_indices)
    return ExamplePositions(length), IndexedRecordCollator(source, base, source_indices)
