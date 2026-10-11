"""Read only the next minibatches' reward factors in DataLoader workers.

No CUDA work or candidate sampling belongs here. Workers return compact CPU
tensors; DataLoader pins them and Trainer transfers them with the rest of the
batch. Reward transforms and exact full-catalog flows stay in GFlowMethod.
"""

from __future__ import annotations

import time
from typing import Any


PREPARED_REWARDS_KEY = "_gflow_prepared_rewards"
REWARD_PREPARE_SECONDS_KEY = "_gflow_reward_prepare_seconds"


class RewardPrefetchCollator:
    """Attach raw factors for the cache-row indices emitted by a collator."""

    def __init__(self, collator: Any, row_reader: Any) -> None:
        self.collator = collator
        # This reader must be path-only when pickled, and CPU-only when used.
        self.row_reader = row_reader

    def __call__(self, examples: Any) -> dict[str, Any]:
        batch = self.collator(examples)
        started = time.perf_counter()
        batch[PREPARED_REWARDS_KEY] = self.row_reader.prepare_rows(batch["example_indices"])
        batch[REWARD_PREPARE_SECONDS_KEY] = time.perf_counter() - started
        return batch
