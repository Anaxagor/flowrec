"""Unified semantic-ID candidate sampler: three modes, one exclusion contract."""

import pytest
import torch

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler

# Catalog: items 1..5 (0 = pad) mapped to rows 0..4 of the semantic table.
TABLE = torch.tensor(
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.long
)
ITEM_TO_ROW = torch.tensor([-1, 0, 1, 2, 3, 4], dtype=torch.long)


class FakeScorer(torch.nn.Module):
    """Deterministic per-item scores, shared across the batch."""

    def __init__(self, scores_by_item: list[float]) -> None:
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores_by_item, dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return self.scores.device

    def score_items(self, batch, candidate_item_ids: torch.Tensor) -> torch.Tensor:
        bsz = batch["target_token_ids"].size(0)
        return self.scores[candidate_item_ids].unsqueeze(0).expand(bsz, -1)


def _batch(target_item: int, history_items: list[int]) -> dict:
    row = int(ITEM_TO_ROW[target_item])
    history = torch.tensor([history_items], dtype=torch.long)
    return {
        "target_token_ids": TABLE[row].unsqueeze(0),
        "target_token_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_item_ids": torch.tensor([target_item]),
        "history_item_ids": history,
        "history_mask": history.ne(0),
    }


def _sampler(mode: str, *, scorer=None, exclude_history: bool = True, **cfg) -> SemanticIDCandidateSampler:
    return SemanticIDCandidateSampler(
        CandidateSamplerConfig(mode=mode, exclude_history=exclude_history, **cfg),
        item_semantic_ids=TABLE,
        item_id_to_semantic_row=ITEM_TO_ROW,
        scorer=scorer,
    )


class TestRandomMode:
    def test_negatives_exclude_target_and_history(self) -> None:
        sampler = _sampler("random", num_negatives=2)
        batch = _batch(target_item=1, history_items=[2, 3])
        seen = set()
        for _ in range(50):
            out = sampler.sample(batch)
            items = out.extra["candidate_item_ids"][0].tolist()
            assert out.is_positive[0].tolist() == [True, False, False]
            assert items[0] == 1
            negatives = [i for i in items[1:] if i >= 0]
            assert set(negatives).isdisjoint({1, 2, 3})
            seen.update(negatives)
        # The full allowed catalog {4, 5} is reachable.
        assert seen == {4, 5}

    def test_history_included_when_flag_off(self) -> None:
        sampler = _sampler("random", num_negatives=3, exclude_history=False)
        batch = _batch(target_item=1, history_items=[2, 3])
        seen = set()
        for _ in range(50):
            out = sampler.sample(batch)
            seen.update(i for i in out.extra["candidate_item_ids"][0, 1:].tolist() if i >= 0)
        assert seen == {2, 3, 4, 5}
        assert 1 not in seen  # the positive is still never a negative

    def test_exclude_history_without_mapping_raises(self) -> None:
        sampler = SemanticIDCandidateSampler(
            CandidateSamplerConfig(mode="random", num_negatives=1, exclude_history=True),
            item_semantic_ids=TABLE,
        )
        with pytest.raises(ValueError, match="exclude_history"):
            sampler.sample(_batch(target_item=1, history_items=[2]))

    def test_without_mapping_negatives_avoid_target_tokens(self) -> None:
        sampler = SemanticIDCandidateSampler(
            CandidateSamplerConfig(mode="random", num_negatives=4, exclude_history=False),
            item_semantic_ids=TABLE,
        )
        batch = _batch(target_item=1, history_items=[])
        for _ in range(20):
            out = sampler.sample(batch)
            negatives = out.candidate_token_ids[0, 1:]
            assert not negatives.eq(TABLE[0]).all(dim=-1).any()

    def test_unfillable_slots_are_marked_invalid(self) -> None:
        # Catalog has 5 items; target + 3 history leave one allowed item.
        sampler = _sampler("random", num_negatives=3)
        batch = _batch(target_item=1, history_items=[2, 3, 4])
        out = sampler.sample(batch)
        items = out.extra["candidate_item_ids"][0, 1:].tolist()
        assert sorted(i for i in items if i >= 0) == [5]
        rows = out.candidate_item_row_ids[0, 1:]
        assert (rows.ge(0).sum() == 1).item()
        # Invalid slots carry an all-False token mask.
        invalid_mask = out.candidate_token_mask[0, 1:][rows.lt(0)]
        assert not invalid_mask.any()


class TestTopNMode:
    def test_returns_highest_scoring_allowed_items(self) -> None:
        scorer = FakeScorer([0.0, 10.0, 8.0, 6.0, 4.0, 2.0])  # indexed by item id
        sampler = _sampler("top_n", scorer=scorer, num_negatives=2)
        # target item 2 (score 8) and history item 3 (score 6) are excluded,
        # so the hardest allowed negatives are items 1 (10) and 4 (4).
        batch = _batch(target_item=2, history_items=[3])
        out = sampler.sample(batch)
        assert out.extra["candidate_item_ids"][0].tolist() == [2, 1, 4]

    def test_deterministic(self) -> None:
        scorer = FakeScorer([0.0, 1.0, 5.0, 3.0, 4.0, 2.0])
        sampler = _sampler("top_n", scorer=scorer, num_negatives=3)
        batch = _batch(target_item=1, history_items=[])
        first = sampler.sample(batch).extra["candidate_item_ids"]
        for _ in range(5):
            assert torch.equal(sampler.sample(batch).extra["candidate_item_ids"], first)

    def test_requires_scorer(self) -> None:
        with pytest.raises(ValueError, match="scorer"):
            _sampler("top_n")


class TestCurriculumMode:
    def test_ramps_from_easy_to_hard(self) -> None:
        scorer = FakeScorer([0.0, 100.0, 0.0, 50.0, -100.0, 0.0])
        sampler = _sampler(
            "curriculum", scorer=scorer, num_negatives=1,
            curriculum_steps=100, temperature=1e-4,
        )
        batch = _batch(target_item=3, history_items=[])
        # step 0: alpha=-1 -> near-deterministically the easiest (lowest score).
        early = sampler.sample(batch, step=0).extra["candidate_item_ids"][0, 1].item()
        assert early == 4
        # step >= curriculum_steps: alpha=+1 -> the hardest allowed item.
        late = sampler.sample(batch, step=100).extra["candidate_item_ids"][0, 1].item()
        assert late == 1

    def test_no_warmup_means_hard_from_start(self) -> None:
        scorer = FakeScorer([0.0, 100.0, 0.0, 0.0, -100.0, 0.0])
        sampler = _sampler(
            "curriculum", scorer=scorer, num_negatives=1,
            curriculum_steps=0, temperature=1e-4,
        )
        batch = _batch(target_item=3, history_items=[])
        assert sampler.sample(batch, step=0).extra["candidate_item_ids"][0, 1].item() == 1

    def test_exclusions_apply(self) -> None:
        scorer = FakeScorer([0.0, 100.0, 90.0, 80.0, 0.0, -10.0])
        sampler = _sampler(
            "curriculum", scorer=scorer, num_negatives=1,
            curriculum_steps=0, temperature=1e-4,
        )
        # Highest-scoring items 1 (target) and 2, 3 (history) are excluded.
        batch = _batch(target_item=1, history_items=[2, 3])
        assert sampler.sample(batch, step=10).extra["candidate_item_ids"][0, 1].item() == 4


class TestContract:
    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            SemanticIDCandidateSampler(CandidateSamplerConfig(mode="pool"))

    def test_rows_and_items_stay_consistent(self) -> None:
        sampler = _sampler("random", num_negatives=3)
        out = sampler.sample(_batch(target_item=1, history_items=[2]))
        rows = out.candidate_item_row_ids[0]
        items = out.extra["candidate_item_ids"][0]
        for row, item in zip(rows.tolist(), items.tolist()):
            if row >= 0 and item >= 0:
                assert int(ITEM_TO_ROW[item]) == row

    def test_table_resolved_from_model(self) -> None:
        class ModelWithTable(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("item_semantic_ids", TABLE.clone())

        sampler = SemanticIDCandidateSampler(
            CandidateSamplerConfig(mode="random", num_negatives=1, exclude_history=False)
        )
        out = sampler.sample(_batch(target_item=1, history_items=[]), model=ModelWithTable())
        assert out.num_candidates == 2
