"""Training-split item popularity shared by evaluators and data modules."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def training_item_counts_by_raw_id(
    interactions: pd.DataFrame,
    *,
    split_strategy: str,
    item2id: Mapping[str, int],
) -> dict[str, int]:
    """Count source interactions belonging to train without validation leakage."""
    if interactions.empty:
        return {}
    if split_strategy == "leave_one_out":
        lengths = interactions.groupby("user_id", sort=False)["position"].transform("size")
        mask = interactions["position"].lt(lengths - 2)
    else:
        raise ValueError(f"Unsupported split_strategy={split_strategy!r}")
    counts = (
        interactions.loc[mask, "raw_item_id"]
        .astype(str)
        .loc[lambda values: values.isin(item2id)]
        .value_counts()
    )
    return {str(item): int(count) for item, count in counts.items()}
