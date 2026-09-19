"""Factorized reward cache: row reconstruction must equal the reward models'
own math, and select_candidates / composition must match the dense cache
semantics (target-row substitution for negative rows, power, exact reward sum).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from flowrec.ft_methods.gflow.factorized_cache import (
    CACHE_KIND,
    FACTORS_FILENAME,
    ComposedFactorizedRewardCache,
    FactorizedRewardCache,
    exact_similarity_rows,
    hierarchical_similarity_rows,
    open_reward_cache,
    scatter_sparse_rows,
)
from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel, transform_sasrec_scores
from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch

EPS = 1e-6
CPU = torch.device("cpu")


def _write_factorized(root: Path, name: str, factors: dict, *, mode=None, config=None,
                      num_items: int, num_examples: int, **manifest_extra) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save(factors, d / FACTORS_FILENAME)
    (d / "manifest.json").write_text(json.dumps({
        "kind": CACHE_KIND, "factors_path": FACTORS_FILENAME,
        "num_examples": num_examples, "num_catalog_items": num_items,
        "reward_epsilon": EPS,
        "component": {"name": name.split("_")[0], "mode": mode, "config": config or {}},
        **manifest_extra,
    }))
    return d


def _write_dense(root: Path, name: str, rewards: torch.Tensor, target_rows: torch.Tensor,
                 powers=(0.25, 4.0)) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    stored = rewards.float().clamp_min(EPS)
    payload = {
        "example_start": 0, "num_examples": int(rewards.size(0)),
        "target_semantic_row_ids": target_rows, "full_rewards": stored,
        "reward_sum": stored.sum(dim=1, keepdim=True),
    }
    for p in powers:
        slug = format(float(p), ".12g").replace("-", "m").replace(".", "p")
        payload[f"reward_sum_power_{slug}"] = stored.pow(p).clamp_min(EPS).sum(dim=1, keepdim=True)
    torch.save(payload, d / "shard_00000.pt")
    (d / "manifest.json").write_text(json.dumps({
        "num_examples": int(rewards.size(0)), "stored_power_sums": list(powers),
        "shards": [{"path": "shard_00000.pt", "example_start": 0, "num_examples": int(rewards.size(0))}],
    }))
    return d


# --------------------------------------------------------------------------- #
# Kernels vs the reward models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level_stride", [0, 256])
@pytest.mark.parametrize("level_weights", [None, (2.0, 1.0, 1.0, 0.5)])
def test_hierarchical_rows_match_similarity_model(level_stride, level_weights):
    torch.manual_seed(0)
    n_levels, K, dim, L, N, B = 3, 8, 4, 4, 37, 5
    codebooks = torch.randn(n_levels, K, dim)
    # 3 RQ levels + a collision suffix level (exact match only), some rows
    # carrying out-of-range codes to exercise the validity masks.
    codes = torch.randint(0, K, (N, L))
    codes[:, 3] = torch.randint(0, 3, (N,))
    codes[0, 1] = K + 5
    offset = 1
    tokens = codes + offset + (torch.arange(L) * level_stride).view(1, -1)
    target = tokens[torch.randint(0, N, (B,))].clone()
    target[1, 2] = tokens[3, 2]  # partial prefix agreement

    cfg = SimilarityRewardConfig(mode="hierarchical", token_offset=offset,
                                 level_stride=level_stride, level_weights=level_weights)
    model = SimilarityRewardModel(cfg, codebooks=codebooks)
    traj = SemanticTrajectoryBatch(
        candidate_token_ids=tokens.unsqueeze(0).expand(B, -1, -1),
        candidate_token_mask=torch.ones((B, N, L), dtype=torch.bool),
        is_positive=torch.zeros((B, N), dtype=torch.bool),
    )
    expected = model._hierarchical_similarity({"target_token_ids": target}, traj)
    got = hierarchical_similarity_rows(
        target, tokens, model.code_dist, token_offset=offset, level_stride=level_stride,
        level_weights=level_weights, chunk_rows=2,
    )
    assert torch.allclose(got, expected, atol=1e-6, rtol=0)


def test_exact_rows_match_similarity_model():
    torch.manual_seed(1)
    N, L, B = 20, 4, 3
    tokens = torch.randint(1, 6, (N, L))
    target = tokens[[4, 9, 9]].clone()
    target[2, 0] = 99
    model = SimilarityRewardModel(SimilarityRewardConfig(mode="exact"))
    traj = SemanticTrajectoryBatch(
        candidate_token_ids=tokens.unsqueeze(0).expand(B, -1, -1),
        candidate_token_mask=torch.ones((B, N, L), dtype=torch.bool),
        is_positive=torch.zeros((B, N), dtype=torch.bool),
    )
    expected = model._exact_similarity({"target_token_ids": target}, traj)
    assert torch.equal(exact_similarity_rows(target, tokens, chunk_rows=2), expected)


def test_transform_function_equals_reward_model_method():
    class _Stub(torch.nn.Module):
        def score_items(self, batch, candidate_item_ids=None):
            return torch.zeros(1)

    scores = torch.tensor([[-30.0, -1.0, 0.0, 2.5, float("-inf")]])
    for transform in ("sigmoid", "softplus", "identity", "exp", "minmax_batch", "rank_percentile"):
        cfg = SASRecRewardConfig(transform=transform, temperature=0.7, score_shift=0.3, reward_scale=1.5)
        model = SASRecRewardModel(_Stub(), cfg)
        assert torch.equal(model.transform_scores(scores), transform_sasrec_scores(scores, cfg))


def test_scatter_sparse_rows_densifies_csr():
    indptr = torch.tensor([0, 2, 2, 3])
    cols = torch.tensor([1, 3, 0])
    vals = torch.tensor([1.2, 0.6, 1.25])
    full = scatter_sparse_rows(indptr, cols, vals, torch.tensor([2, 0, 1]), num_items=4, fill_value=1.0, device=CPU)
    assert torch.equal(full, torch.tensor([[1.25, 1.0, 1.0, 1.0], [1.0, 1.2, 1.0, 0.6], [1.0, 1.0, 1.0, 1.0]]))


# --------------------------------------------------------------------------- #
# Caches: reconstruction + select_candidates semantics vs the dense cache
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic(tmp_path):
    torch.manual_seed(2)
    E, N, d, L = 6, 11, 5, 4
    hidden = torch.randn(E, d)
    item_table = torch.randn(N, d)
    row_valid = torch.ones(N, dtype=torch.bool)
    row_valid[7] = False
    target_rows = torch.tensor([0, 3, 7, 10, 5, 1])
    sasrec_cfg = SASRecRewardConfig(transform="sigmoid", temperature=1.0)
    scores = (hidden @ item_table.T).masked_fill(~row_valid.unsqueeze(0), -float("inf"))
    sasrec_dense = transform_sasrec_scores(scores, sasrec_cfg).clamp_min(EPS)

    indptr = torch.tensor([0, 1, 1, 3, 3, 4, 4])
    cols = torch.tensor([2, 0, 9, 10])
    vals = torch.tensor([1.2, 0.6, 1.25, 1.2])
    inter_dense = scatter_sparse_rows(indptr, cols, vals, torch.arange(E), num_items=N, fill_value=1.0, device=CPU)

    dirs = {
        "sasrec": _write_factorized(
            tmp_path, "sasrec", {"hidden": hidden.half(), "item_table": item_table, "row_valid": row_valid,
                                 "target_semantic_row_ids": target_rows},
            config=asdict(sasrec_cfg), num_items=N, num_examples=E),
        "interaction": _write_factorized(
            tmp_path, "interaction", {"indptr": indptr, "cols": cols, "vals": vals,
                                      "target_semantic_row_ids": target_rows},
            num_items=N, num_examples=E, fill_value=1.0),
    }
    # float16 hidden: the dense reference is what the stored (rounded) factor gives.
    scores16 = (hidden.half().float() @ item_table.T).masked_fill(~row_valid.unsqueeze(0), -float("inf"))
    sasrec_dense16 = transform_sasrec_scores(scores16, sasrec_cfg).clamp_min(EPS)
    dense = {
        "sasrec": _write_dense(tmp_path, "dense_sasrec", sasrec_dense16, target_rows),
        "interaction": _write_dense(tmp_path, "dense_interaction", inter_dense, target_rows),
    }
    return {"dirs": dirs, "dense": dense, "E": E, "N": N, "target_rows": target_rows,
            "sasrec_rows": sasrec_dense16, "interaction_rows": inter_dense}


def test_component_rows_match_dense_reference(synthetic, tmp_path):
    idx = torch.tensor([5, 0, 2])
    sasrec = FactorizedRewardCache(synthetic["dirs"]["sasrec"])
    assert torch.allclose(sasrec.component_rows(idx, CPU), synthetic["sasrec_rows"][idx], atol=1e-6)
    inter = FactorizedRewardCache(synthetic["dirs"]["interaction"])
    assert torch.equal(inter.component_rows(idx, CPU), synthetic["interaction_rows"][idx])


@pytest.mark.parametrize("power", [1.0, 0.25, 4.0])
def test_select_candidates_equals_dense_cache(synthetic, power):
    idx = torch.tensor([4, 1, 5])
    rows = torch.tensor([[-1, 2, 9], [-1, -1, 0], [3, 10, -1]])
    for key in ("sasrec", "interaction"):
        fact = FactorizedRewardCache(synthetic["dirs"][key], power=power)
        dense = _PrecomputedRewardCache(synthetic["dense"][key], power=power)
        f_sel, f_sum, f_full = fact.select_candidates(idx, rows, CPU, include_full=True)
        d_sel, d_sum, d_full = dense.select_candidates(idx, rows, CPU, include_full=True)
        assert torch.allclose(f_full, d_full, rtol=1e-5, atol=1e-7), key
        assert torch.allclose(f_sel, d_sel, rtol=1e-5, atol=1e-7), key
        assert torch.allclose(f_sum, d_sum, rtol=1e-5, atol=1e-6), key
        # negative rows resolved to the example's target row
        target = synthetic["target_rows"][idx]
        assert torch.allclose(f_sel[:, 0][:2], f_full[torch.arange(2), target[:2]])
        _, _, none_full = fact.select_candidates(idx, rows, CPU, include_full=False)
        assert none_full is None


def test_positive_component_rows(tmp_path):
    target_rows = torch.tensor([2, -1, 0])
    d = _write_factorized(tmp_path, "positive", {"target_semantic_row_ids": target_rows},
                          config={"positive_reward": 1.0, "negative_reward": 1e-3, "reward_epsilon": EPS},
                          num_items=4, num_examples=3)
    rows = FactorizedRewardCache(d).component_rows(torch.tensor([0, 1, 2]), CPU)
    expected = torch.full((3, 4), 1e-3)
    expected[0, 2] = 1.0
    expected[2, 0] = 1.0
    assert torch.equal(rows, expected)


@pytest.mark.parametrize("composition", ["multiply", "mixed"])
@pytest.mark.parametrize("power", [1.0, 4.0])
def test_composition_matches_reference_and_accepts_dense_components(synthetic, composition, power):
    idx = torch.tensor([3, 0])
    rows = torch.tensor([[-1, 6], [1, -1]])
    weights = {"sasrec": 1.0, "interaction": 2.0}
    composed = ComposedFactorizedRewardCache(
        [("sasrec", synthetic["dirs"]["sasrec"], weights["sasrec"]),
         ("interaction", synthetic["dirs"]["interaction"], weights["interaction"])],
        composition=composition, power=power,
    )
    mixed_layout = ComposedFactorizedRewardCache(
        [("sasrec", _PrecomputedRewardCache(synthetic["dense"]["sasrec"]), weights["sasrec"]),
         ("interaction", synthetic["dirs"]["interaction"], weights["interaction"])],
        composition=composition, power=power,
    )
    stored = {"sasrec": synthetic["sasrec_rows"][idx], "interaction": synthetic["interaction_rows"][idx]}
    if composition == "multiply":
        ref = torch.ones_like(stored["sasrec"])
        for k, w in weights.items():
            ref = ref * stored[k].clamp_min(EPS).pow(w)
    else:
        ref = torch.zeros_like(stored["sasrec"])
        for k, w in weights.items():
            ref = ref + w * (stored[k] / stored[k].amax(dim=1, keepdim=True)).clamp_min(EPS)
    ref = ref.clamp_min(EPS).pow(power).clamp_min(EPS)
    target = synthetic["target_rows"][idx]
    resolved = torch.where(rows >= 0, rows, target.unsqueeze(1))

    for cache in (composed, mixed_layout):
        sel, reward_sum, full = cache.select_candidates(idx, rows, CPU, include_full=True)
        assert torch.allclose(full, ref, rtol=1e-5, atol=1e-7)
        assert torch.allclose(sel, ref.gather(1, resolved), rtol=1e-5, atol=1e-7)
        assert torch.allclose(reward_sum, ref.sum(dim=1), rtol=1e-5, atol=1e-6)
    assert "**" in composed.describe()["form"]


def test_open_reward_cache_dispatches_on_manifest_kind(synthetic):
    assert isinstance(open_reward_cache(synthetic["dirs"]["interaction"]), FactorizedRewardCache)
    assert isinstance(open_reward_cache(synthetic["dense"]["interaction"]), _PrecomputedRewardCache)


def test_rejects_bad_inputs(synthetic, tmp_path):
    cache = FactorizedRewardCache(synthetic["dirs"]["interaction"])
    with pytest.raises(IndexError):
        cache.select_candidates(torch.tensor([99]), torch.tensor([[0]]), CPU)
    with pytest.raises(IndexError):
        cache.select_candidates(torch.tensor([0]), torch.tensor([[synthetic["N"]]]), CPU)
    with pytest.raises(ValueError):
        FactorizedRewardCache(synthetic["dirs"]["interaction"], power=0.0)
    with pytest.raises(ValueError):
        FactorizedRewardCache(synthetic["dense"]["interaction"])  # dense manifest, wrong reader
    with pytest.raises(ValueError):
        ComposedFactorizedRewardCache(
            [("interaction", FactorizedRewardCache(synthetic["dirs"]["interaction"], power=2.0), 1.0)],
            composition="multiply", power=1.0,
        )


# --------------------------------------------------------------------------- #
# Lookup-time sasrec transform (the factors are transform-agnostic)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("temperature", [1.0, 2.0, 4.0])
def test_sasrec_transform_override_is_the_teacher_softmax(synthetic, temperature):
    """exp(s / T) at lookup time from a cache BUILT with sigmoid: normalized
    over the catalog it is the teacher's softmax at that temperature."""
    base = FactorizedRewardCache(synthetic["dirs"]["sasrec"])
    view = base.with_config(transform="exp", temperature=temperature)
    idx = torch.arange(synthetic["E"])
    rows = view.component_rows(idx, CPU)

    factors = torch.load(synthetic["dirs"]["sasrec"] / "factors.pt")
    scores = (factors["hidden"].float() @ factors["item_table"].T)
    scores = scores.masked_fill(~factors["row_valid"].unsqueeze(0), -float("inf"))
    expected = torch.exp(scores / temperature).clamp_min(EPS)
    assert torch.allclose(rows, expected, rtol=1e-5, atol=1e-7)

    valid = factors["row_valid"]
    softmax = torch.softmax(scores[:, valid] / temperature, dim=1)
    assert torch.allclose(rows[:, valid] / rows[:, valid].sum(1, keepdim=True), softmax, atol=1e-5)

    # the original keeps its own transform and shares the factors with the view
    assert torch.allclose(base.component_rows(idx, CPU), synthetic["sasrec_rows"], atol=1e-6)
    assert view.factors is base.factors
    assert view.describe()["config_overrides"] == {"transform": "exp", "temperature": temperature}
    assert base.describe()["config_overrides"] == {}
    # opening with overrides is equivalent to the view
    opened = FactorizedRewardCache(synthetic["dirs"]["sasrec"],
                                   config_overrides={"transform": "exp", "temperature": temperature})
    assert torch.equal(opened.component_rows(idx, CPU), rows)


def test_config_overrides_are_restricted(synthetic):
    with pytest.raises(ValueError, match="only supported for the sasrec"):
        FactorizedRewardCache(synthetic["dirs"]["interaction"], config_overrides={"transform": "exp"})
    sasrec = FactorizedRewardCache(synthetic["dirs"]["sasrec"])
    with pytest.raises(ValueError, match="Unsupported sasrec config_overrides"):
        sasrec.with_config(reward_epsilon=1e-3)
    with pytest.raises(ValueError, match="temperature must be positive"):
        sasrec.with_config(temperature=0.0)


def test_per_component_exponent_modulates_a_peaked_base(synthetic):
    """sasrec(exp) x interaction ** gamma: the composed cache's component
    weights are exponents under 'multiply'."""
    sasrec = FactorizedRewardCache(synthetic["dirs"]["sasrec"]).with_config(transform="exp", temperature=2.0)
    inter = FactorizedRewardCache(synthetic["dirs"]["interaction"])
    idx = torch.arange(synthetic["E"])
    cand = torch.arange(synthetic["N"]).unsqueeze(0).expand(synthetic["E"], -1)
    for gamma in (1.0, 4.0):
        composed = ComposedFactorizedRewardCache(
            [("sasrec", sasrec, 1.0), ("interaction", inter, gamma)], composition="multiply", power=1.0)
        selected, reward_sum, full = composed.select_candidates(idx, cand, CPU, include_full=True)
        expected = (sasrec.component_rows(idx, CPU) * inter.component_rows(idx, CPU).pow(gamma)).clamp_min(EPS)
        assert torch.allclose(full, expected, rtol=1e-5)
        assert torch.allclose(selected, expected, rtol=1e-5)
        assert torch.allclose(reward_sum, expected.sum(1), rtol=1e-5)
