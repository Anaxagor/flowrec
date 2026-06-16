from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from torch.utils.data import Dataset

SplitName = Literal["train", "val", "test"]

# Canonical interaction DataFrame columns after preprocessing:
#   raw_user_id: string
#   raw_item_id: string
#   timestamp: int64
#   rating: float
#   verified_purchase: object/bool with optional missing values
#   order_idx: int64
#   user_id: int64, optional until ID factorization
#   item_id: int64, optional until ID factorization; 0 is reserved for padding
#   position: int64, per-user chronological position
#
# Canonical sequence-example DataFrame columns consumed by SequenceDataset and
# SequenceBatchCollator:
#   raw_user_id, user_id, history_raw_items, history_item_ids,
#   target_raw_item, target_item_id, target_rating, target_timestamp,
#   target_verified_purchase, history_ratings, history_verified_purchase,
#   split, target_position


@dataclass(slots=True)
class DatasetStats:
    num_users: int
    num_items: int
    num_train_examples: int
    num_val_examples: int
    num_test_examples: int
    num_interactions: int

    # Optional post-split cold-item filtering diagnostics. These default to
    # neutral values so callers that only need the core fields remain compatible.
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
    """Thin PyTorch dataset around a sequence-example DataFrame.

    The dataset no longer owns InteractionRecord or SequenceExample objects.  It
    keeps the split examples as a pandas DataFrame and returns the same dict
    schema as the old dataclass-backed dataset, so existing collators and model
    code do not need to change.
    """

    def __init__(self, examples: pd.DataFrame, split: SplitName) -> None:
        self.examples = examples.reset_index(drop=True).copy()
        self.split = split
        if "split" not in self.examples.columns:
            self.examples["split"] = split
        self._records: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return int(len(self.examples))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # DataLoader workers repeatedly indexing through pandas row objects is
        # slower than list indexing. Materialize dicts once per worker process;
        # this is a shallow conversion of list-valued columns, not a second copy
        # of the whole sequence tensor payload.
        if self._records is None:
            self._records = self.examples.to_dict(orient="records")
        return self._records[idx]
