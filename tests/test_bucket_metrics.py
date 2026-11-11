"""Hand-calculated, paired checks of popularity and history diagnostics."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "exp5_result_analysis"))
from bucket_metrics import analyze_buckets


@pytest.fixture
def example():
    cat = pd.DataFrame({"item_id": list("abcde"), "popularity": [100, 30, 10, 2, 0]})
    specs = [
        dict(run_id="init", model="gpt2", stage="init", seed=None),
        dict(run_id="sft", model="gpt2", stage="sft", seed=42, baseline_run_id="init"),
        dict(run_id="gflow", model="gpt2", stage="gflow", seed=42,
             baseline_run_id="init", sft_run_id="sft"),
    ]
    common = dict(user_ids=np.array(["u1", "u2", "u3", "u4"]),
                  target_item_ids=np.array(list("bcda")),
                  history_lengths=np.array([1, 5, 5, 5]),
                  full_history_lengths=np.array([1, 5, 6, 20]),
                  history_fingerprints=np.array(["hist1", "hist2", "hist3", "hist4"]))
    bundles = {rid: {**deepcopy(common), "recommendations": np.array([list(s) for s in lists])}
               for rid, lists in [("init", ["ab", "ac", "ba", "ab"]),
                                  ("sft", ["bd", "cd", "da", "ca"]),
                                  ("gflow", ["bd", "cd", "de", "ac"]) ]}
    return specs, bundles, cat


def analyze(example, **kwargs):
    return analyze_buckets(*example, ks=(1, 2), history_edges=(5, 10, 20), **kwargs)


def select(frame, **values):
    mask = np.ones(len(frame), dtype=bool)
    for key, value in values.items():
        mask &= frame[key].eq(value).to_numpy()
    result = frame.loc[mask]
    assert len(result) == 1
    return result.iloc[0]


def test_known_ranking_and_exposure(example):
    result = analyze(example)
    initial = select(result["summary"], run_id="init", k=2)
    tuned = select(result["summary"], run_id="gflow", k=2)
    assert initial.hr == pytest.approx(.75)
    assert initial.ndcg == pytest.approx((1 + 2 / np.log2(3)) / 4)
    assert initial.coverage == pytest.approx(.6)
    assert initial.arp == pytest.approx(62.5)
    assert tuned.hr == pytest.approx(1)
    assert tuned.ndcg == pytest.approx(1)
    assert tuned.coverage == pytest.approx(1)
    assert tuned.arp == pytest.approx(19.5)
    counts = np.array([4, 3, 1, 0, 0], dtype=float)
    expected_gini = np.abs(counts[:, None] - counts[None, :]).sum() / (2 * 5 * counts.sum())
    assert initial.exposure_gini == pytest.approx(expected_gini)
    probabilities = counts[counts > 0] / counts.sum()
    assert initial.effective_catalog == pytest.approx(np.exp(-(probabilities * np.log(probabilities)).sum()))
    change = select(result["comparisons"], run_id="gflow", reference_run_id="init", k=2, cohort_type="overall")
    assert change.delta_hr == pytest.approx(.25)
    assert change.delta_ndcg == pytest.approx(1 - initial.ndcg)
    assert change.hit_gains == 1
    assert change.hit_losses == 0
    assert change.rank_improvements_both_hit == 2
    assert change.mean_rank_improvement_both_hit == pytest.approx(2 / 3)
    assert change.mean_list_overlap == pytest.approx(3 / 8)
    sft_change = select(result["comparisons"], run_id="gflow", reference_run_id="sft", k=2, cohort_type="overall")
    assert sft_change.delta_hr == 0
    assert sft_change.delta_ndcg == pytest.approx((1 - 1 / np.log2(3)) / 4)


def test_popularity_is_training_only_tie_aware_and_zero_separate(example):
    specs, bundles, cat = deepcopy(example)
    cat.loc[:, "popularity"] = [1, 1, 2, 4, 0]
    original = analyze((specs, bundles, cat))["catalog_buckets"].set_index("item_id")
    assert original.popularity_bucket.to_dict() == dict(a="tail", b="tail", c="middle", d="head", e="unseen")
    assert original.loc["a", "positive_popularity_percentile"] == pytest.approx(.375)
    assert np.isnan(original.loc["e", "positive_popularity_percentile"])
    for bundle in bundles.values():
        bundle["target_item_ids"][:] = "a"
    changed = analyze((specs, bundles, cat))["catalog_buckets"].set_index("item_id")
    pd.testing.assert_frame_equal(original, changed)
    cat.loc[:, "popularity"] = [1, 1, 1, 1, 0]
    tied = analyze((specs, bundles, cat))["catalog_buckets"]
    assert set(tied.loc[tied.popularity > 0, "popularity_bucket"]) == {"middle"}


def test_full_history_boundaries_and_empty_groups(example):
    result = analyze(example)
    histories = result["cohorts"].query("run_id == 'init' and k == 2 and cohort_type == 'history_length'").set_index("history_bucket")
    assert histories.num_users.to_dict() == {"0-5": 2, "6-10": 1, "11-20": 1, "21+": 0}
    assert histories.loc["21+", ["hr", "ndcg", "coverage", "arp", "exposure_gini", "effective_catalog"]].isna().all()
    assert histories.loc["0-5", "mean_input_history_length"] == 3
    assert histories.loc["11-20", "mean_history_length"] == 20
    unseen = select(result["cohorts"], run_id="init", k=2, cohort_type="target_popularity", target_popularity_bucket="unseen")
    assert unseen.num_users == 0 and np.isnan(unseen.hr)
    attribution = select(result["accuracy_attribution"], run_id="gflow", reference_run_id="init", k=2,
                         cohort_type="target_popularity", target_popularity_bucket="unseen")
    assert attribution.delta_hr_contribution == 0 and attribution.delta_ndcg_contribution == 0


def test_target_popularity_does_not_filter_or_rerank_lists(example):
    result = analyze(example)
    middle = select(result["cohorts"], run_id="init", k=1, cohort_type="target_popularity", target_popularity_bucket="middle")
    # u1 targets b, but a is the unmodified first recommendation. Filtering to
    # middle-popularity recommendations would incorrectly move b to rank 1.
    assert middle.num_users == 1
    assert middle.hr == 0
    assert middle.arp == 100
    assert middle.coverage == pytest.approx(1 / 5)
    item_bucket = select(result["popularity_exposure"], run_id="init", k=2, cohort_type="overall", popularity_bucket="tail")
    assert item_bucket.catalog_items == 2
    assert item_bucket.exposures == 1
    assert item_bucket.coverage == pytest.approx(.5)
    assert item_bucket.exposure_share == pytest.approx(1 / 8)
    assert item_bucket.exposure_gini == pytest.approx(.5)
    assert item_bucket.effective_catalog == pytest.approx(1)


def test_each_partition_and_per_item_contributions_reconstruct_total(example):
    result = analyze(example)
    overall = result["comparisons"].query("cohort_type == 'overall'")
    for _, comparison in overall.iterrows():
        selection = dict(run_id=comparison.run_id, reference_run_id=comparison.reference_run_id, k=comparison.k)
        attr = result["accuracy_attribution"]
        deltas = result["item_deltas"]
        for key, value in selection.items():
            attr = attr.loc[attr[key].eq(value)]
            deltas = deltas.loc[deltas[key].eq(value)]
        for _, frame in attr.groupby("cohort_type"):
            assert frame.num_users.sum() == 4
            assert frame.delta_hr_contribution.sum() == pytest.approx(comparison.delta_hr)
            assert frame.delta_ndcg_contribution.sum() == pytest.approx(comparison.delta_ndcg)
        assert deltas.delta_hr_contribution.sum() == pytest.approx(comparison.delta_hr)
        assert deltas.delta_ndcg_contribution.sum() == pytest.approx(comparison.delta_ndcg)
    improved_d = select(result["item_deltas"], run_id="gflow", reference_run_id="init", k=2, item_id="d")
    assert improved_d.target_count == 1 and improved_d.hit_gains == 1
    assert improved_d.hr_before == 0 and improved_d.hr_after == 1
    assert improved_d.exposure_status == "new"
    exposure_only = select(result["items"], run_id="gflow", k=2, item_id="e")
    assert exposure_only.target_count == 0 and np.isnan(exposure_only.hr)


def test_pairwise_overlap_equals_explicit_enumeration(example):
    result = analyze(example)
    for rid, bundle in example[1].items():
        for k in (1, 2):
            lists = bundle["recommendations"][:, :k]
            expected = np.mean([len(set(a) & set(b)) / k for a, b in combinations(lists, 2)])
            assert select(result["summary"], run_id=rid, k=k).mean_pairwise_overlap == pytest.approx(expected)


def test_permuted_users_align_and_original_bundles_are_unchanged(example):
    expected = analyze(example)
    copy = deepcopy(example)
    original = deepcopy(copy[1])
    order = np.array([2, 0, 3, 1])
    copy[1]["gflow"] = {key: array[order] for key, array in copy[1]["gflow"].items()}
    actual = analyze(copy)
    for key in expected:
        pd.testing.assert_frame_equal(expected[key], actual[key])
    for rid, bundle in example[1].items():
        for name in bundle:
            np.testing.assert_array_equal(bundle[name], original[rid][name])


@pytest.mark.parametrize("field", ["user_ids", "target_item_ids", "history_lengths", "full_history_lengths", "history_fingerprints"])
def test_alignment_mismatches_fail(example, field):
    bundle = example[1]["gflow"]
    if field == "history_lengths":
        bundle[field][0] = 0
    elif field == "full_history_lengths":
        bundle[field][0] = 2
    else:
        bundle[field][0] = "wrong" if field != "target_item_ids" else "e"
    with pytest.raises(ValueError, match="mismatched"):
        analyze(example)


@pytest.mark.parametrize("problem", ["duplicate_user", "duplicate_recommendation", "unknown_recommendation", "missing_fingerprint", "missing_full_history", "wrong_sft_seed", "wrong_baseline", "negative_popularity", "fractional_history"])
def test_invalid_inputs_fail(example, problem):
    specs, bundles, cat = example
    bundle = bundles["gflow"]
    if problem == "duplicate_user":
        bundle["user_ids"][0] = bundle["user_ids"][1]
    elif problem == "duplicate_recommendation":
        bundle["recommendations"][0] = ["a", "a"]
    elif problem == "unknown_recommendation":
        bundle["recommendations"][0, 0] = "z"
    elif problem == "missing_fingerprint":
        del bundle["history_fingerprints"]
    elif problem == "missing_full_history":
        del bundle["full_history_lengths"]
    elif problem == "wrong_sft_seed":
        specs[1]["seed"] = 43
    elif problem == "wrong_baseline":
        specs[2]["baseline_run_id"] = "sft"
    elif problem == "negative_popularity":
        cat.loc[0, "popularity"] = -1
    elif problem == "fractional_history":
        bundle["history_lengths"] = np.array([.5, 5, 5, 5])
    with pytest.raises(ValueError):
        analyze(example)


def test_padding_preserves_rank_and_is_not_a_catalog_item(example):
    example[1]["init"]["recommendations"][0] = ["", "b"]
    result = analyze(example)
    first = select(result["summary"], run_id="init", k=1)
    second = select(result["summary"], run_id="init", k=2)
    assert first.hr == .25
    assert second.hr == .75
    assert second.ndcg == pytest.approx((1 + 2 / np.log2(3)) / 4)
    assert second.valid_slot_rate == pytest.approx(7 / 8)


def test_history_fallback_is_explicit_and_bootstrap_is_reproducible(example):
    for bundle in example[1].values():
        del bundle["full_history_lengths"]
    first = analyze(example, bootstrap_samples=100, seed=23)
    second = analyze(example, bootstrap_samples=100, seed=23)
    pd.testing.assert_frame_equal(first["comparisons"], second["comparisons"])
    assert set(first["summary"].history_length_source) == {"history_lengths (clipped fallback)"}
    row = select(first["comparisons"], run_id="gflow", reference_run_id="init", k=2, cohort_type="overall")
    assert 0 <= row.hr_ci_low <= row.delta_hr <= row.hr_ci_high <= 1


def test_bootstrap_resamples_paired_differences(example):
    result = analyze(example, bootstrap_samples=100, seed=23)
    first_comparison = select(result["comparisons"], run_id="sft", reference_run_id="init", k=1, cohort_type="overall")
    paired_differences = np.array([1, 1, 1, -1])
    rng = np.random.default_rng(23)
    means = paired_differences[rng.integers(4, size=(100, 4))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    assert first_comparison.hr_ci_low == pytest.approx(low)
    assert first_comparison.hr_ci_high == pytest.approx(high)
    assert first_comparison.ndcg_ci_low == pytest.approx(low)
    assert first_comparison.ndcg_ci_high == pytest.approx(high)


def test_compact_item_exports_omit_inactive_catalog_items(example):
    specs, bundles, cat = example
    cat = pd.concat([cat, pd.DataFrame({"item_id": ["f"], "popularity": [0]})], ignore_index=True)
    result = analyze((specs, bundles, cat))
    assert "f" in set(result["catalog_buckets"].item_id)
    assert "f" not in set(result["items"].item_id)
    assert "f" not in set(result["item_deltas"].item_id)
    assert select(result["summary"], run_id="gflow", k=2).coverage == pytest.approx(5 / 6)
