"""Small, independently computed checks of recommendation-change analysis."""

from copy import deepcopy
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "exp6_result_analysis"))
from analysis import analyze_bundles
from beauty_sft_analysis import _metric_validation, _validate_cache_identity


@pytest.fixture
def example():
    catalog = pd.DataFrame({"item_id": list("abcde"), "popularity": [100, 30, 10, 2, 0]})
    specs = [
        {"run_id": "gpt2_init", "model": "gpt2", "stage": "init", "seed": None},
        {"run_id": "gpt2_sft_42", "model": "gpt2", "stage": "sft", "seed": 42,
         "baseline_run_id": "gpt2_init"},
    ]
    common = {
        "user_ids": np.array(["u1", "u2", "u3", "u4"]),
        "target_item_ids": np.array(list("bcda")),
        "history_lengths": np.array([1, 3, 7, 20]),
    }
    bundles = {
        "gpt2_init": {**deepcopy(common), "recommendations": np.array([list(x) for x in ["ab", "ac", "ba", "ab"]])},
        "gpt2_sft_42": {**deepcopy(common), "recommendations": np.array([list(x) for x in ["bd", "cd", "de", "ac"]])},
    }
    return specs, bundles, catalog


def analyze(example, **kwargs):
    return analyze_bundles(*example, ks=(1, 2), bootstrap_samples=100, seed=17, **kwargs)


def row(frame, run_id, k=2):
    matches = frame.loc[(frame["run_id"] == run_id) & (frame["k"] == k)]
    assert len(matches) == 1
    return matches.iloc[0]


def test_hand_computed_ranking_and_exposure_metrics(example):
    result = analyze(example)
    initial = row(result["summary"], "gpt2_init")
    tuned = row(result["summary"], "gpt2_sft_42")
    assert initial["hr"] == pytest.approx(3 / 4)
    assert initial["ndcg"] == pytest.approx((2 / np.log2(3) + 1) / 4)
    assert initial["coverage"] == pytest.approx(3 / 5)
    assert initial["arp"] == pytest.approx(62.5)
    assert tuned["hr"] == pytest.approx(1.0)
    assert tuned["ndcg"] == pytest.approx(1.0)
    assert tuned["coverage"] == pytest.approx(1.0)
    assert tuned["arp"] == pytest.approx(19.5)

    # Zero-exposure catalog items must count in the Gini denominator.
    counts = np.array([4, 3, 1, 0, 0], dtype=float)
    expected_gini = np.abs(counts[:, None] - counts[None, :]).sum() / (2 * len(counts) * counts.sum())
    probabilities = counts[counts > 0] / counts.sum()
    expected_entropy = -(probabilities * np.log(probabilities)).sum()
    assert initial["exposure_gini"] == pytest.approx(expected_gini)
    assert initial["exposure_entropy"] == pytest.approx(expected_entropy)
    assert initial["effective_catalog"] == pytest.approx(np.exp(expected_entropy))

    change = row(result["comparisons"], "gpt2_sft_42")
    assert change["delta_ndcg"] == pytest.approx(1 - initial["ndcg"])
    assert change["delta_coverage"] == pytest.approx(2 / 5)
    assert change["delta_arp"] == pytest.approx(19.5 - 62.5)
    assert change["mean_list_overlap"] == pytest.approx(3 / 8)
    assert change["new_catalog_items"] == 2
    assert change["lost_catalog_items"] == 0


def test_accuracy_attribution_partitions_users_and_explains_total_delta(example):
    result = analyze(example)
    groups = result["accuracy_attribution"].query("run_id == 'gpt2_sft_42' and k == 2").set_index("target_group")
    # User u3's target d was never recommended before; b, c, a were already covered.
    assert groups.loc["newly covered targets", "num_users"] == 1
    assert groups.loc["newly covered targets", "delta_ndcg_contribution"] == pytest.approx(1 / 4)
    assert groups.loc["previously covered targets", "num_users"] == 3
    assert groups.loc["previously covered targets", "delta_ndcg_contribution"] == pytest.approx((2 - 2 / np.log2(3)) / 4)
    assert groups.loc["never covered targets", "num_users"] == 0
    assert groups["num_users"].sum() == 4
    assert groups["delta_ndcg_contribution"].sum() == pytest.approx(
        row(result["comparisons"], "gpt2_sft_42")["delta_ndcg"]
    )


def test_bootstrap_resamples_paired_user_differences(example):
    samples = 1000
    result = analyze_bundles(*example, ks=(2,), bootstrap_samples=samples, seed=17)
    deltas = np.array([1 - 1 / np.log2(3), 1 - 1 / np.log2(3), 1, 0])
    rng = np.random.default_rng(17)
    resampled_means = deltas[rng.integers(4, size=(samples, 4))].mean(axis=1)
    expected_low, expected_high = np.quantile(resampled_means, [0.025, 0.975])
    change = row(result["comparisons"], "gpt2_sft_42")
    assert change["ndcg_ci_low"] == pytest.approx(expected_low)
    assert change["ndcg_ci_high"] == pytest.approx(expected_high)


def test_item_table_accounts_for_new_lost_and_zero_exposures(example):
    specs, bundles, catalog = example
    bundles["gpt2_sft_42"]["recommendations"][3] = ["c", "e"]
    result = analyze((specs, bundles, catalog))
    items = result["items"].query("run_id == 'gpt2_sft_42' and k == 2").set_index("item_id")
    assert set(items.index) == set(catalog["item_id"])
    assert items["exposures"].to_dict() == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 2}
    assert items["baseline_exposures"].to_dict() == {"a": 4, "b": 3, "c": 1, "d": 0, "e": 0}
    assert items.loc["a", "status"] == "lost"
    assert items.loc["d", "status"] == "new"
    assert items.loc["e", "status"] == "new"
    change = row(result["comparisons"], "gpt2_sft_42")
    assert change["new_catalog_items"] == 2
    assert change["lost_catalog_items"] == 1
    assert change["delta_coverage"] == pytest.approx(1 / 5)


@pytest.mark.parametrize("k", [1, 2])
def test_population_overlap_equals_explicit_user_pair_enumeration(example, k):
    result = analyze(example)
    for run_id, bundle in example[1].items():
        recommendations = bundle["recommendations"][:, :k]
        expected = np.mean([
            len(set(left) & set(right)) / k
            for left, right in combinations(recommendations, 2)
        ])
        assert row(result["summary"], run_id, k)["mean_pairwise_overlap"] == pytest.approx(expected)


def test_pairing_uses_user_identity_when_rows_are_shuffled(example):
    for bundle in example[1].values():
        bundle["history_fingerprints"] = np.array(["context1", "context2", "context3", "context4"])
    expected = analyze(example)
    specs, bundles, catalog = deepcopy(example)
    order = np.array([2, 0, 3, 1])
    bundles["gpt2_sft_42"] = {key: value[order] for key, value in bundles["gpt2_sft_42"].items()}
    actual = analyze((specs, bundles, catalog))
    for key in ["summary", "comparisons", "items"]:
        pd.testing.assert_frame_equal(actual[key], expected[key])
    sort_keys = [key for key in ["run_id", "k", "user_id"] if key in expected["users"].columns]
    assert "user_id" in sort_keys
    pd.testing.assert_frame_equal(
        actual["users"].sort_values(sort_keys).reset_index(drop=True),
        expected["users"].sort_values(sort_keys).reset_index(drop=True),
    )


@pytest.mark.parametrize("problem", ["missing", "different_contents", "wrong_shape"])
def test_equal_history_lengths_do_not_hide_mismatched_contexts(example, problem):
    specs, bundles, catalog = example
    for bundle in bundles.values():
        bundle["history_fingerprints"] = np.array(["context1", "context2", "context3", "context4"])
    changed = bundles["gpt2_sft_42"]
    if problem == "missing":
        del changed["history_fingerprints"]
    elif problem == "different_contents":
        changed["history_fingerprints"][0] = "wrong"
    else:
        changed["history_fingerprints"] = changed["history_fingerprints"][:-1]
    with pytest.raises(ValueError, match="[Hh]istory"):
        analyze((specs, bundles, catalog))


@pytest.mark.parametrize("num_users,total_users,expected_rows", [(4, 4, 2), (3, 4, 0)])
def test_recorded_metrics_are_compared_only_for_full_test(example, num_users, total_users, expected_rows):
    specs, _, _ = example
    specs[0]["recorded_test_metrics"] = {"HR@2": 0.75, "Coverage@2": 0.5}
    tables = analyze(example)
    provenance = [{"num_users": num_users, "total_test_users": total_users} for _ in specs]
    validation = _metric_validation(specs, tables, provenance)
    assert len(validation) == expected_rows
    if expected_rows:
        metrics = validation.set_index("metric")
        assert metrics.loc["HR", "difference"] == pytest.approx(0)
        assert metrics.loc["Coverage", "difference"] == pytest.approx(0.1)


@pytest.mark.parametrize("changed_key,new_value", [
    ("checkpoint", "/different/checkpoint.pt"),
    ("model", "tiger"),
    ("seed", 43),
    ("baseline_run_id", "different_baseline"),
    ("recorded_test_metrics", {"NDCG@2": 0.9}),
])
def test_cached_recommendations_cannot_be_relabeled_by_manifest(example, changed_key, new_value):
    spec = deepcopy(example[0][1])
    spec.update(checkpoint="/original/checkpoint.pt", recorded_test_metrics={"NDCG@2": 1.0})
    source = deepcopy(spec)
    source.update(num_users=4, total_test_users=4, fingerprint="cache-fingerprint")
    _validate_cache_identity(spec, source)
    spec[changed_key] = new_value
    with pytest.raises(ValueError, match=changed_key):
        _validate_cache_identity(spec, source)


def test_comparison_uses_explicit_baseline_not_first_seed(example):
    specs, bundles, catalog = example
    specs[0]["seed"] = 42
    specs.insert(1, {"run_id": "gpt2_init_43", "model": "gpt2", "stage": "init", "seed": 43})
    specs[2]["baseline_run_id"] = "gpt2_init_43"
    specs[2]["seed"] = 43
    bundles["gpt2_init_43"] = deepcopy(bundles["gpt2_sft_42"])
    result = analyze((specs, bundles, catalog))
    change = row(result["comparisons"], "gpt2_sft_42")
    assert change["baseline_run_id"] == "gpt2_init_43"
    assert change["delta_ndcg"] == pytest.approx(0)
    assert change["delta_coverage"] == pytest.approx(0)
    assert change["delta_arp"] == pytest.approx(0)
    assert change["mean_list_overlap"] == pytest.approx(1)
    assert change["new_catalog_items"] == 0
    assert change["lost_catalog_items"] == 0
    assert change["ndcg_ci_low"] == pytest.approx(0)
    assert change["ndcg_ci_high"] == pytest.approx(0)


def test_cross_model_overlap_ignores_order_within_recommendation_lists(example):
    specs, bundles, catalog = example
    specs.append({"run_id": "tiger_init", "model": "tiger", "stage": "init", "seed": None})
    bundles["tiger_init"] = deepcopy(bundles["gpt2_init"])
    bundles["tiger_init"]["recommendations"] = bundles["tiger_init"]["recommendations"][:, ::-1]
    result = analyze((specs, bundles, catalog))
    cross = result["cross_model"]
    matched = cross.loc[(cross["k"] == 2) & cross.apply(
        lambda r: {r["run_a"], r["run_b"]} == {"gpt2_init", "tiger_init"}, axis=1
    )]
    assert len(matched) == 1
    assert matched.iloc[0]["mean_list_overlap"] == pytest.approx(1)
    assert matched.iloc[0]["catalog_jaccard"] == pytest.approx(1)
    assert matched.iloc[0]["exposure_spearman"] == pytest.approx(1)


@pytest.mark.parametrize("problem", ["missing_user", "duplicate_user", "changed_target", "unknown_prediction", "duplicate_prediction"])
def test_invalid_or_noncomparable_runs_fail_loudly(example, problem):
    specs, bundles, catalog = example
    changed = bundles["gpt2_sft_42"]
    if problem == "missing_user":
        bundles["gpt2_sft_42"] = {key: value[:-1] for key, value in changed.items()}
    elif problem == "duplicate_user":
        changed["user_ids"][1] = changed["user_ids"][0]
    elif problem == "changed_target":
        changed["target_item_ids"][0] = "e"
    elif problem == "unknown_prediction":
        changed["recommendations"][0, 0] = "x"
    elif problem == "duplicate_prediction":
        changed["recommendations"][0, 1] = changed["recommendations"][0, 0]
    with pytest.raises(ValueError):
        analyze((specs, bundles, catalog))


def test_empty_prediction_slots_preserve_rank_and_do_not_extend_catalog(example):
    specs, bundles, catalog = example
    bundles["gpt2_sft_42"]["recommendations"] = np.array([
        ["", "b"], ["", ""], ["d", ""], ["", "a"],
    ])
    result = analyze((specs, bundles, catalog))
    tuned = row(result["summary"], "gpt2_sft_42")
    assert tuned["hr"] == pytest.approx(3 / 4)
    assert tuned["ndcg"] == pytest.approx((2 / np.log2(3) + 1) / 4)
    assert tuned["coverage"] == pytest.approx(3 / 5)
    assert tuned["arp"] == pytest.approx((30 + 2 + 100) / 3)
    assert row(result["summary"], "gpt2_sft_42", 1)["hr"] == pytest.approx(1 / 4)
    items = result["items"].query("run_id == 'gpt2_sft_42' and k == 2")
    assert items["exposures"].sum() == 3
    assert "" not in set(items["item_id"])


@pytest.mark.parametrize("ks", [(0,), (3,), (-1,)])
def test_invalid_cutoff_fails_instead_of_silently_truncating(example, ks):
    with pytest.raises(ValueError):
        analyze_bundles(*example, ks=ks, bootstrap_samples=0)
