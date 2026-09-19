"""Policy mode of the candidate sampler: negatives drawn from the recommendation
model itself (stochastic beam search over the catalog trie), no teacher."""

import itertools

import pytest
import torch

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler

# A 3-level catalog with shared prefixes. Row 6 maps to NO catalog item (as the
# rows of a SID table the cold-item filter removed), item 0 is the pad.
TABLE = torch.tensor(
    [[1, 4, 7], [1, 4, 8], [1, 5, 7], [2, 4, 7], [2, 6, 9], [3, 5, 8], [3, 6, 9], [3, 6, 7]], dtype=torch.long
)
ITEM_TO_ROW = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 7], dtype=torch.long)   # items 1..7; row 6 unmapped
VOCAB = 10


class FakePolicy(torch.nn.Module):
    """Autoregressive toy model: the logits of token d depend on the previous
    token only (fixed table), so the exact item distribution is enumerable."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("transition", torch.randn(VOCAB, VOCAB, generator=gen) * 1.5)
        self.item_semantic_ids = TABLE
        self.calls: list[tuple[int, ...]] = []
        self.modes: list[bool] = []

    def candidate_token_logits(self, batch, *, candidate_token_ids, candidate_token_mask=None):
        self.calls.append(tuple(candidate_token_ids.shape))
        self.modes.append(self.training)
        bsz, n, length = candidate_token_ids.shape
        previous = torch.cat([candidate_token_ids.new_zeros(bsz, n, 1), candidate_token_ids[..., :-1]], dim=-1)
        return self.transition[previous]                       # [B, N, L, V]

    def item_distribution(self, scale: float = 1.0) -> dict[int, float]:
        """Exact P(row) under trie-renormalized, ``scale``-tempered token logits."""
        valid = [r for r in range(TABLE.size(0)) if bool((ITEM_TO_ROW == r).any())]
        probs: dict[int, float] = {}
        for row in valid:
            p, prev = 1.0, 0
            for depth in range(TABLE.size(1)):
                prefix = TABLE[row, :depth].tolist()
                children = sorted({int(TABLE[r, depth]) for r in valid if TABLE[r, :depth].tolist() == prefix})
                logits = self.transition[prev, children] * scale
                p *= float(torch.softmax(logits, dim=0)[children.index(int(TABLE[row, depth]))])
                prev = int(TABLE[row, depth])
            probs[row] = p
        return probs


def _batch(target_items: list[int], history: list[list[int]] | None = None) -> dict:
    rows = ITEM_TO_ROW[torch.tensor(target_items)]
    batch = {
        "target_token_ids": TABLE[rows],
        "target_token_mask": torch.ones(len(target_items), TABLE.size(1), dtype=torch.bool),
        "target_item_ids": torch.tensor(target_items),
    }
    if history is not None:
        hist = torch.tensor(history, dtype=torch.long)
        batch.update(history_item_ids=hist, history_mask=hist.ne(0))
    return batch


def _sampler(num_negatives: int, **cfg) -> SemanticIDCandidateSampler:
    return SemanticIDCandidateSampler(
        CandidateSamplerConfig(mode="policy", num_negatives=num_negatives, **cfg),
        item_semantic_ids=TABLE, item_id_to_semantic_row=ITEM_TO_ROW,
    )


class TestPolicyMode:
    def test_negatives_are_distinct_catalog_items_excluding_target_and_history(self) -> None:
        model = FakePolicy()
        sampler = _sampler(3)
        for seed in range(30):
            torch.manual_seed(seed)
            out = sampler.sample(_batch([1, 5], history=[[2, 0], [3, 4]]), model=model, step=0)
            rows, items = out.candidate_item_row_ids, out.extra["candidate_item_ids"]
            assert rows.shape == (2, 4) and bool(out.is_positive[:, 0].all())
            for b, banned in enumerate(({1, 2}, {5, 3, 4})):
                negatives = items[b, 1:].tolist()
                assert len(set(negatives)) == 3 and not set(negatives) & banned
                assert all(1 <= i <= 7 for i in negatives)             # never the unmapped row 6
                assert torch.equal(out.candidate_token_ids[b, 1:], TABLE[rows[b, 1:]])
        assert 6 not in {int(r) for r in rows.reshape(-1)}

    def test_first_draw_follows_the_models_own_item_distribution(self) -> None:
        model = FakePolicy(seed=3)
        sampler = _sampler(1, exclude_history=False, policy_extra_beams=1)
        target_row = int(ITEM_TO_ROW[1])
        exact = model.item_distribution()
        exact.pop(target_row)
        total = sum(exact.values())
        torch.manual_seed(0)
        out = sampler.sample(_batch([1] * 20000), model=model)
        counts = torch.bincount(out.candidate_item_row_ids[:, 1], minlength=TABLE.size(0)).float() / 20000
        for row, p in exact.items():
            assert abs(float(counts[row]) - p / total) < 0.012, (row, float(counts[row]), p / total)
        assert out.extra["policy_fallback_fraction"] == 0.0

    def test_always_on_policy_whatever_the_training_step(self) -> None:
        model = FakePolicy(seed=3)
        sampler = _sampler(1, exclude_history=False, policy_extra_beams=1)
        target_row = int(ITEM_TO_ROW[1])
        exact = {r: p for r, p in model.item_distribution().items() if r != target_row}
        total = sum(exact.values())
        for step in (0, 10, 10_000):                                  # no ramp: the step plays no role
            torch.manual_seed(step)
            out = sampler.sample(_batch([1] * 20000), model=model, step=step)
            counts = torch.bincount(out.candidate_item_row_ids[:, 1], minlength=TABLE.size(0)).float() / 20000
            assert all(abs(float(counts[row]) - p / total) < 0.012 for row, p in exact.items()), step
            assert "curriculum_alpha" not in out.extra

    def test_curriculum_and_temperature_are_refused(self) -> None:
        with pytest.raises(ValueError, match="always on-policy"):
            _sampler(1, curriculum_steps=100)
        with pytest.raises(ValueError, match="always on-policy"):
            _sampler(1, temperature=2.0)

    def test_only_valid_catalog_items_can_be_decoded(self) -> None:
        """The model strongly prefers tokens that lead OUT of the catalog: an
        unmapped row (6), a prefix no item has, a token no level uses."""

        class Rogue(FakePolicy):
            def candidate_token_logits(self, batch, *, candidate_token_ids, candidate_token_mask=None):
                logits = super().candidate_token_logits(batch, candidate_token_ids=candidate_token_ids).clone()
                logits[..., 0] += 30.0                                  # pad token
                logits[..., 9] += 30.0                                  # [3, 6, 9] is the unmapped row 6
                if logits.size(-2) > 2:
                    logits[..., 2, 4] += 30.0                           # token 4 is never a last-level token
                return logits

        model, sampler = Rogue(), _sampler(3, policy_extra_beams=0)
        valid = {tuple(TABLE[int(r)].tolist()) for r in ITEM_TO_ROW[1:]}
        for seed in range(50):
            torch.manual_seed(seed)
            out = sampler.sample(_batch([1, 4]), model=model)
            for tokens in out.candidate_token_ids.reshape(-1, TABLE.size(1)).tolist():
                assert tuple(tokens) in valid, tokens
            assert 6 not in out.candidate_item_row_ids.reshape(-1).tolist()
            assert bool(out.extra["candidate_item_ids"].ge(1).all())

    def test_an_invalid_negative_raises(self, monkeypatch) -> None:
        sampler = _sampler(2)
        sampler._policy_catalog = None
        from flowrec.candidates import sampler as module

        monkeypatch.setattr(module._PolicyCatalog, "rows_of",
                            lambda self, leaves: torch.full((leaves.size(0),), 6, device=leaves.device))
        # row 6 has no catalog item: nothing survives the filter, the uniform fill takes over ...
        out = sampler.sample(_batch([1]), model=FakePolicy())
        assert 6 not in out.candidate_item_row_ids.tolist()[0] and out.extra["policy_fallback_fraction"] == 1.0
        # ... and if a bad row ever got through, the per-batch invariant refuses it
        monkeypatch.setattr(module.SemanticIDCandidateSampler, "_draw",
                            lambda self, weights, k, gen: torch.full((weights.size(0), k), 6, dtype=torch.long))
        with pytest.raises(RuntimeError, match="not an allowed catalog item"):
            sampler.sample(_batch([1]), model=FakePolicy())

    def test_model_is_sampled_in_eval_mode_and_restored(self) -> None:
        model = FakePolicy().train()
        _sampler(2).sample(_batch([1]), model=model)
        assert model.modes and not any(model.modes) and model.training
        assert [shape[-1] for shape in model.calls] == [1, 2, 3]      # one decode per level

    def test_every_slot_is_filled_when_the_beam_runs_short(self) -> None:
        # 6 negatives requested, only 5 allowed items exist besides the target + 1 history item.
        model = FakePolicy()
        out = _sampler(5, policy_extra_beams=0).sample(_batch([1], history=[[2]]), model=model)
        negatives = out.extra["candidate_item_ids"][0, 1:].tolist()
        assert sorted(negatives) == [3, 4, 5, 6, 7]

    def test_history_crowding_the_beam_is_resolved_on_policy_not_by_uniform_filling(self) -> None:
        """The model's favourite items are all in the user's history: the first,
        narrow search comes back short, the wider one must complete it from the
        model's own distribution over the remaining items."""
        model = FakePolicy(seed=3)
        probs = model.item_distribution()
        by_prob = sorted(probs, key=probs.get, reverse=True)
        row_to_item = {int(r): i for i, r in enumerate(ITEM_TO_ROW.tolist()) if r >= 0}
        history_rows, target_row = by_prob[:3], by_prob[3]
        allowed = {r: p for r, p in probs.items() if r not in history_rows and r != target_row}
        total = sum(allowed.values())
        sampler = _sampler(1, policy_extra_beams=0)
        torch.manual_seed(4)
        n = 20000
        out = sampler.sample(_batch([row_to_item[target_row]] * n,
                                    history=[[row_to_item[r] for r in history_rows]] * n), model=model)
        assert out.extra["policy_fallback_fraction"] == 0.0 and out.extra["policy_retry_fraction"] > 0.5
        counts = torch.bincount(out.candidate_item_row_ids[:, 1], minlength=TABLE.size(0)).float() / n
        assert set(counts.nonzero().flatten().tolist()) <= set(allowed)
        for row, p in allowed.items():
            assert abs(float(counts[row]) - p / total) < 0.012, (row, float(counts[row]), p / total)

    def test_continued_search_keeps_the_without_replacement_law(self) -> None:
        """K=2 with a single beam to spare and the model's favourite item in the
        history: many examples need the continuation, and ordered pairs must
        still follow P(a, b) = p_a * p_b / (1 - p_a) over the allowed items."""
        model = FakePolicy(seed=5)
        probs = model.item_distribution()
        by_prob = sorted(probs, key=probs.get, reverse=True)
        row_to_item = {int(r): i for i, r in enumerate(ITEM_TO_ROW.tolist()) if r >= 0}
        history_row, target_row = by_prob[0], by_prob[1]
        p = {r: v for r, v in probs.items() if r not in (history_row, target_row)}
        z = sum(p.values())
        p = {r: v / z for r, v in p.items()}
        n = 40000
        torch.manual_seed(7)
        out = _sampler(2, policy_extra_beams=0).sample(
            _batch([row_to_item[target_row]] * n, history=[[row_to_item[history_row]]] * n), model=model)
        assert out.extra["policy_retry_fraction"] > 0.3 and out.extra["policy_fallback_fraction"] == 0.0
        pairs = out.candidate_item_row_ids[:, 1:]
        assert bool(pairs[:, 0].ne(pairs[:, 1]).all())
        for a, b in itertools.permutations(sorted(p, key=p.get, reverse=True)[:3], 2):
            share = float((pairs[:, 0].eq(a) & pairs[:, 1].eq(b)).float().mean())
            assert abs(share - p[a] * p[b] / (1 - p[a])) < 0.01, (a, b, share, p[a] * p[b] / (1 - p[a]))

    def test_requires_a_model_with_candidate_token_logits(self) -> None:
        with pytest.raises(ValueError, match="candidate_token_logits"):
            _sampler(1).sample(_batch([1]), model=None)
        with pytest.raises(ValueError, match="item_id_to_semantic_row"):
            SemanticIDCandidateSampler(CandidateSamplerConfig(mode="policy"), item_semantic_ids=TABLE)

    def test_top_beams_are_sampled_without_replacement_exactly(self) -> None:
        """Pairs drawn without replacement: P(a then b) = p_a * p_b / (1 - p_a)."""
        model = FakePolicy(seed=5)
        sampler = _sampler(2, exclude_history=False, policy_extra_beams=1)
        target_row = int(ITEM_TO_ROW[1])
        p = {r: v for r, v in model.item_distribution().items() if r != target_row}
        z = sum(p.values())
        p = {r: v / z for r, v in p.items()}
        torch.manual_seed(2)
        out = sampler.sample(_batch([1] * 30000), model=model)
        pairs = out.candidate_item_row_ids[:, 1:]
        for a, b in itertools.permutations(sorted(p, key=p.get, reverse=True)[:3], 2):
            share = float((pairs[:, 0].eq(a) & pairs[:, 1].eq(b)).float().mean())
            assert abs(share - p[a] * p[b] / (1 - p[a])) < 0.012, (a, b)
