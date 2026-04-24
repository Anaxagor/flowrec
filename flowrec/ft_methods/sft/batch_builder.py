from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flowrec.models.base import Batch


@dataclass(slots=True)
class SFTBatchBuilder:
    """Identity batch builder for supervised fine-tuning.

    GFlowGR and preference methods will build augmented candidate trajectories or
    chosen/rejected pairs.  Vanilla SFT does not need any additional structure:
    it trains on the target item already present in the datamodule batch.
    """

    def build(self, batch: Batch, **_: Any) -> Batch:
        return batch
