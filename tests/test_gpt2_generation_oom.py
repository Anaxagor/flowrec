"""Evaluation OOM retries must preserve the full ranking protocol and rows."""
import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/exp1-standard-params"))
import run_gpt2_sft_all as runner


def batch(rows):
    ids = torch.tensor(rows)
    return {
        "user_ids": ids,
        "target_item_ids": ids % 5 + 1,
        "recommendations": torch.stack((ids % 5 + 1, (ids + 1) % 5 + 1), dim=1),
        "history_token_ids": ids[:, None].repeat(1, 4),
        "raw_user_ids": [str(row) for row in rows],
        "raw_target_items": tuple(str(row % 5 + 1) for row in rows),
        "constant": "metadata",
        "scalar": torch.tensor(7),
    }


class Generator:
    def __init__(self, limit=None, *, error=None, fail_at_row=None):
        self.limit = limit
        self.error = error
        self.fail_at_row = fail_at_row
        self.calls = []
        self.successful_rows = []
        self.failed_allocations = []

    def eval(self):
        return self

    def generate(self, chunk, **kwargs):
        rows = chunk["user_ids"].tolist()
        self.calls.append(rows)
        assert kwargs == dict(num_beams=20, num_return_sequences=2,
                              constrain_valid_sequences=True)
        assert chunk["raw_user_ids"] == [str(row) for row in rows]
        assert chunk["raw_target_items"] == tuple(str(row % 5 + 1) for row in rows)
        assert chunk["history_token_ids"][:, 0].tolist() == rows
        assert chunk["constant"] == "metadata" and chunk["scalar"].item() == 7
        if self.error is not None:
            raise self.error
        limit = self.limit
        if self.fail_at_row is not None and self.fail_at_row in rows:
            limit = 1
        if limit is not None and len(rows) > limit:
            # Simulate an activation owned by the failing generate frame. It
            # must be released BEFORE the caller attempts to empty the cache.
            activation = torch.zeros(8)
            self.failed_allocations.append(weakref.ref(activation))
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        self.successful_rows.extend(rows)
        return chunk["recommendations"].unsqueeze(-1)


def evaluate(model, batches, *, max_batches=None):
    return runner.evaluate_generation_ranking_metrics(
        model, batches, device="cpu",
        lookup=runner.SemanticIdItemLookup({str(i): [i] for i in range(1, 6)},
                                         {str(i): i for i in range(1, 6)}),
        item_popularity=torch.tensor([0, 10, 20, 30, 40, 50]),
        num_catalog_items=5, ks=(1, 2), num_beams=20,
        constrain_valid_sequences=True, max_batches=max_batches,
    )


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("max_batches", [None, 1])
def test_generation_backoff_preserves_metrics_rows_and_loader_limit(monkeypatch, limit, max_batches):
    batches = [batch(range(5)), batch(range(5, 9))]
    model = Generator(limit)
    releases = []

    def release_cache():
        assert all(ref() is None for ref in model.failed_allocations)
        releases.append(True)

    monkeypatch.setattr(torch.cuda, "empty_cache", release_cache)
    expected = evaluate(Generator(), batches, max_batches=max_batches)
    actual = evaluate(model, batches, max_batches=max_batches)
    assert actual == expected
    assert model.successful_rows == list(range(9 if max_batches is None else 5))
    assert len(releases) == len(model.failed_allocations) > 0
    # The final short chunk of the first batch must not reduce the learned cap.
    if limit == 2 and max_batches is None:
        assert model.calls == [[0, 1, 2, 3, 4], [0, 1], [2, 3], [4], [5, 6], [7, 8]]


def test_later_chunk_oom_does_not_repeat_successful_examples(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = Generator(2, fail_at_row=3)
    batches = [batch(range(5)), batch(range(5, 9))]
    assert evaluate(model, batches) == evaluate(Generator(), batches)
    assert model.successful_rows == list(range(9))
    assert model.calls[:5] == [[0, 1, 2, 3, 4], [0, 1], [2, 3], [2], [3]]
    assert all(len(rows) == 1 for rows in model.calls[3:])


def test_single_example_oom_is_raised(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = Generator(0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        evaluate(model, [batch(range(5))])
    assert [len(rows) for rows in model.calls] == [5, 2, 1]
    assert model.successful_rows == []


def test_unrelated_runtime_error_is_not_retried():
    model = Generator(error=RuntimeError("invalid semantic catalog"))
    with pytest.raises(RuntimeError, match="invalid semantic catalog"):
        evaluate(model, [batch(range(5))])
    assert len(model.calls) == 1


def test_no_oom_preserves_original_batches():
    model = Generator()
    metrics = evaluate(model, [batch(range(5)), batch(range(5, 9))])
    assert metrics["num_examples"] == 9
    assert model.calls == [list(range(5)), list(range(5, 9))]
