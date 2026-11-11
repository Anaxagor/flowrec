"""Bound temporary history-expansion tables without changing source examples.

The original sequence builder remains the oracle for every chunk. This only
limits its intermediate long table; the final datamodule still contains all
examples and can remain large. The scoped patch is used by CPU preparation
and leaves the shared preprocessing modules/files unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import time


HISTORY_COLUMNS = ("history_raw_items", "history_item_ids", "history_ratings",
                   "history_verified_purchase")


def _attach_in_chunks(examples, events, *, chunk_size, original):
    import numpy as np

    total = len(examples)
    if total <= chunk_size:
        # In particular, preserve the original empty-frame column dtypes.
        original(examples, events)
        return
    started = time.monotonic()
    event_users = events["user_id"].to_numpy(copy=False)
    sorted_users = len(event_users) < 2 or not bool(np.any(event_users[1:] < event_users[:-1]))
    for column in HISTORY_COLUMNS:
        # Allocate object slots, not millions of disposable empty Python lists.
        examples[column] = np.empty(total, dtype=object)
    positions = {column: examples.columns.get_loc(column) for column in HISTORY_COLUMNS}
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = examples.iloc[start:end].copy()
        users = chunk["user_id"].to_numpy(copy=False)
        if sorted_users:
            # _prepare_events sorts by user. A contiguous slice keeps every
            # required event (including duplicate keys) in its original order.
            lo = int(np.searchsorted(event_users, users.min(), side="left"))
            hi = int(np.searchsorted(event_users, users.max(), side="right"))
            relevant_events = events.iloc[lo:hi]
        else:
            # Support callers with arbitrary event ordering without changing
            # merge/group ordering or assuming consecutive user IDs.
            relevant_events = events.loc[events["user_id"].isin(np.unique(users))]
        original(chunk, relevant_events)
        for column in HISTORY_COLUMNS:
            examples.iloc[start:end, positions[column]] = chunk[column].to_numpy(dtype=object, copy=False)
        print(json.dumps({"type": "exp8_history_chunk", "processed": end,
                          "num_examples": total, "chunk_rows": end - start,
                          "event_rows": len(relevant_events),
                          "elapsed_sec": round(time.monotonic() - started, 3)}, sort_keys=True), flush=True)


@contextmanager
def bounded_history_build(chunk_size):
    """Temporarily run original history construction in bounded row chunks.

    Zero retains the original implementation. Calls restore the previous
    function even after exceptions, and nested scopes are supported.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("history chunk size must be a nonnegative integer")
    if chunk_size == 0:
        yield
        return
    from flowrec.data import sequence_builder

    previous = sequence_builder._attach_history_lists

    def attach(examples, events):
        _attach_in_chunks(examples, events, chunk_size=chunk_size, original=previous)

    sequence_builder._attach_history_lists = attach
    try:
        yield
    finally:
        sequence_builder._attach_history_lists = previous
