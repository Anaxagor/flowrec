"""Repeated time-aware payload queries preserve cutoff and aggregation semantics."""

import warnings

import numpy as np
import pandas as pd
import pytest

from flowrec.data.interaction_store import InteractionLogStore


def _store() -> InteractionLogStore:
    # Deliberately unsorted, with duplicate timestamps, absent pairs, invalid
    # numeric payloads, and all-NaN prefixes. First/last must not skip NaNs.
    events = pd.DataFrame(
        [
            (0, 1, "listen", 30, "bad"),
            (1, 1, "listen", 20, "80"),
            (0, 1, "listen", 10, "10"),
            (0, 1, "listen", 20, "30"),
            (0, 1, "listen", 20, "50"),
            (0, 2, "listen", 10, None),
            (0, 2, "listen", 20, "bad"),
            (0, 2, "listen", 30, "90"),
            (0, 1, "like", 15, "100"),
        ],
        columns=["user_id", "item_id", "event_type", "timestamp", "played_ratio_pct"],
    )
    return InteractionLogStore(events, num_users=3, item_vocab_size=4)


def _reference(store, event_type, users, items, times, column, agg, strict):
    out = []
    for user, item, time in zip(users, items, times):
        frame = store.events
        past = frame.timestamp < time if strict else frame.timestamp <= time
        selected = frame.loc[
            (frame.event_type == event_type)
            & (frame.user_id == user)
            & (frame.item_id == item)
            & past
        ].sort_values("timestamp", kind="stable")
        if selected.empty:
            out.append(-7.0)
            continue
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=float)
        if agg == "first":
            out.append(values[0])
        elif agg == "last":
            out.append(values[-1])
        else:
            out.append(getattr(np, "nan" + agg)(values))
    return np.asarray(out)


@pytest.mark.parametrize("agg", ["first", "last", "max", "min", "mean"])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("column", ["played_ratio_pct", "timestamp"])
def test_repeated_payload_queries_match_past_event_reference(agg, strict, column):
    store = _store()
    users = [0, 0, 0, 0, 0, 0, 1, 2]
    items = [1, 1, 1, 2, 2, 3, 1, 1]
    times = np.array([10, 20, 40, 20, 30, 50, 20, 50])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN evidence
        for shifted in (times, times + 1, times - 20):
            actual = store.payload_for_pairs_before(
                "listen", users, items, shifted, column,
                agg=agg, strict=strict, default=-7.0,
            )
            expected = _reference(
                store, "listen", users, items, shifted, column, agg, strict
            )
            np.testing.assert_equal(actual, expected)


def test_numeric_payload_conversion_is_reused_across_query_batches(monkeypatch):
    store = _store()
    convert = pd.to_numeric
    calls = []

    def counted(values, *args, **kwargs):
        calls.append((values.name, len(values)))
        return convert(values, *args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", counted)
    for agg in ("last", "max", "first", "mean", "min"):
        store.payload_for_pairs_before(
            "listen", [0], [1], [25], "played_ratio_pct", agg=agg
        )
    assert calls == [("played_ratio_pct", 8)]

    # Independent event types and columns must not share a converted array.
    assert store.payload_for_pairs_before(
        "like", [0], [1], [25], "played_ratio_pct"
    ).tolist() == [100.0]
    assert store.payload_for_pairs_before(
        "listen", [0], [1], [25], "timestamp"
    ).tolist() == [20.0]
    assert calls == [("played_ratio_pct", 8), ("played_ratio_pct", 1), ("timestamp", 8)]


def test_absent_event_type_empty_queries_and_missing_column():
    store = _store()
    assert store.payload_for_pairs_before(
        "missing", [0], [1], [100], "timestamp", default=-7.0
    ).tolist() == [-7.0]
    assert store.payload_for_pairs_before(
        "listen", [], [], [], "timestamp"
    ).shape == (0,)
    with pytest.raises(KeyError, match="payload column"):
        store.payload_for_pairs_before("listen", [0], [1], [100], "missing")
