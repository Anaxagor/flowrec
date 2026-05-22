from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


@dataclass(slots=True)
class InteractionRecord:
    """Normalized user-item interaction record.

    The Amazon Reviews 2023 raw review files expose both `asin` and
    `parent_asin`. In FlowRec we typically use `parent_asin` as the item key
    because it joins cleanly to metadata and groups variants of the same item.
    """

    raw_user_id: str
    raw_item_id: str
    timestamp: int
    rating: float
    verified_purchase: bool | None = None
    order_idx: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SequenceExample:
    """A next-item prediction training/evaluation example."""

    raw_user_id: str
    user_id: int
    history_raw_items: list[str]
    history_item_ids: list[int]
    target_raw_item: str
    target_item_id: int
    target_rating: float
    target_timestamp: int
    split: SplitName
    target_position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_user_id": self.raw_user_id,
            "user_id": self.user_id,
            "history_raw_items": self.history_raw_items,
            "history_item_ids": self.history_item_ids,
            "target_raw_item": self.target_raw_item,
            "target_item_id": self.target_item_id,
            "target_rating": self.target_rating,
            "target_timestamp": self.target_timestamp,
            "split": self.split,
            "target_position": self.target_position,
        }


@dataclass(slots=True)
class DatasetStats:
    num_users: int
    num_items: int
    num_train_examples: int
    num_val_examples: int
    num_test_examples: int
    num_interactions: int

    # Optional post-split cold-item filtering diagnostics.  These default to
    # neutral values so older callers that only populate the core fields remain
    # compatible.
    post_filter_cold_items: bool = False
    cold_item_reference: str | None = None
    num_items_before_cold_filter: int = 0
    num_cold_items_removed: int = 0
    num_train_examples_dropped_cold: int = 0
    num_val_examples_dropped_cold: int = 0
    num_test_examples_dropped_cold: int = 0
    num_train_history_items_removed_cold: int = 0
    num_val_history_items_removed_cold: int = 0
    num_test_history_items_removed_cold: int = 0


class SequenceDataset(Dataset):
    """Thin PyTorch dataset around pre-built sequence examples.

    Padding and tensor conversion are intentionally handled by collators, not by
    the dataset. This keeps examples easy to inspect and reusable across SFT,
    GFlowNet, DPO, and RL fine-tuning methods.
    """

    def __init__(self, examples: list[SequenceExample], split: SplitName) -> None:
        self.examples = examples
        self.split = split

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx].to_dict()
