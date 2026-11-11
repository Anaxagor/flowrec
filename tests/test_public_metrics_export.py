"""Public exports preserve source reports and select comparable raw runs."""
from __future__ import annotations

import csv
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "export_public_metrics", Path(__file__).resolve().parents[1] / "scripts/export_public_metrics.py"
)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def candidate(number, ndcg, coverage, **changes):
    return {
        "record_id": f"candidate_{number:04d}", "source_workbook": "fixture",
        "source_sheet": "Metrics", "source_row": str(number), "dataset": "beauty",
        "model": "tiger", "initialization": "base", "method": "gflow_tb",
        "source_namespace": "experiment_a", "calibration_protocol": "fixed",
        "evaluation_stage": "test", "seed": "42", "repeat_index": "1",
        "ndcg_at_10": str(ndcg), "coverage_at_10": str(coverage), **changes,
    }


def by_role(rows):
    return {row["selection"]: row for row in rows}


def test_extremes_and_pareto_compromise_exclude_dominated_points():
    rows = [candidate(1, ".9", ".1"), candidate(2, ".1", ".9"),
            candidate(3, ".7", ".7"), candidate(4, ".6", ".5")]
    selected = by_role(export.select_candidates(rows))
    assert selected["best_ndcg"]["source_row"] == "1"
    assert selected["best_coverage"]["source_row"] == "2"
    assert selected["pareto"]["source_row"] == "3"
    assert Decimal(selected["pareto"]["pareto_distance_squared"]) == Decimal("0.125")
    assert all(row["candidate_count"] == 4 and row["pareto_front_size"] == 3 for row in selected.values())
    assert selected["pareto"]["pareto_record_ids"] == "candidate_0001|candidate_0002|candidate_0003"


def test_normalization_uses_frontier_not_dominated_outlier():
    front = [candidate(1, ".9", ".1"), candidate(2, ".1", ".9"), candidate(3, ".7", ".7")]
    before = by_role(export.select_candidates(front))["pareto"]
    after = by_role(export.select_candidates(front + [candidate(4, "0", "0")]))["pareto"]
    assert before["record_id"] == after["record_id"]
    assert before["pareto_distance_squared"] == after["pareto_distance_squared"]


def test_primary_ties_use_other_metric_then_numeric_source_row():
    rows = [candidate(10, ".8", ".7"), candidate(2, ".8", ".7"), candidate(1, ".8", ".6")]
    selected = by_role(export.select_candidates(rows))
    assert {row["source_row"] for row in selected.values()} == {"2"}
    assert selected["best_ndcg"]["primary_tie_count"] == 3
    assert selected["best_coverage"]["primary_tie_count"] == 2
    assert selected["pareto"]["primary_tie_count"] == 2
    assert selected["pareto"]["pareto_front_size"] == 2
    assert selected["pareto"]["pareto_unique_point_count"] == 1


def test_pareto_distance_tie_prefers_ndcg():
    rows = [candidate(1, ".1", ".9"), candidate(2, ".9", ".1")]
    chosen = by_role(export.select_candidates(rows))["pareto"]
    assert chosen["source_row"] == "2"
    assert chosen["primary_tie_count"] == 2


def test_decimal_ranking_does_not_collapse_distinct_float_rounding_values():
    rows = [candidate(1, "0.50000000000000000000", ".5"),
            candidate(2, "0.50000000000000000001", ".5")]
    chosen = by_role(export.select_candidates(rows))
    assert {row["source_row"] for row in chosen.values()} == {"2"}
    assert chosen["best_ndcg"]["primary_tie_count"] == 1


def test_identical_points_and_same_seed_repetitions_are_preserved():
    rows = [candidate(3, ".5", ".5", repeat_index="2"), candidate(2, ".5", ".5")]
    selected = export.select_candidates(rows)
    assert len(selected) == 3
    assert {row["record_id"] for row in selected} == {"candidate_0002"}
    assert all(row["candidate_count"] == row["primary_tie_count"] == row["pareto_front_size"] == 2
               for row in selected)
    assert all(Decimal(row["pareto_distance_squared"]) == 0 for row in selected)
    assert all(row["candidate_record_ids"] == "candidate_0002|candidate_0003" for row in selected)


@pytest.mark.parametrize("field,value", [
    ("dataset", "instruments"), ("model", "gpt2"), ("initialization", "sft"),
    ("method", "prefix_kl_q_pi"), ("source_namespace", "experiment_b"),
    ("calibration_protocol", "periodic_10"), ("evaluation_stage", "validation"),
    ("split_protocol", "global_time"), ("dataset_version", "2023"),
    ("initial_checkpoint_id", "checkpoint_b"),
    ("history", "last_128_targets"), ("evaluation_point", "validation_selected_best"),
])
def test_incomparable_protocol_groups_are_separate(field, value):
    rows = [candidate(1, ".1", ".1"), candidate(2, ".9", ".9", **{field: value})]
    selected = export.select_candidates(rows)
    assert len(selected) == 6
    assert all(row["candidate_count"] == 1 for row in selected)
    assert {row["record_id"] for row in selected} == {"candidate_0001", "candidate_0002"}


def test_gflow_losses_are_pooled_and_sft_only_is_one_reference():
    rows = [candidate(1, ".9", ".1"), candidate(2, ".1", ".9", method="gflow_db"),
            candidate(3, ".7", ".7", method="gflow_subtb"), candidate(4, ".99", ".99", method="sft_only")]
    selected = export.select_candidates(rows)
    assert len(selected) == 4
    assert [row["record_id"] for row in selected if row["selection"] == "baseline"] == ["candidate_0004"]
    assert all(row["candidate_count"] == 3 for row in selected if row["method_family"] == "gflow")


def test_multiple_sft_references_are_not_ranked_as_if_hyperparameters():
    with pytest.raises(ValueError, match="Multiple SFT-only references"):
        export.select_candidates([candidate(1, ".1", ".1", method="sft_only"),
                                  candidate(2, ".9", ".9", method="sft_only")])


@pytest.mark.parametrize("value", ["", "NaN", "Infinity", "-0.1", "1.1"])
def test_invalid_candidate_metric_is_rejected(value):
    with pytest.raises(ValueError, match="ndcg_at_10"):
        export.select_candidates([candidate(1, value, ".5")])


def test_duplicate_candidate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        export.select_candidates([candidate(1, ".1", ".2"), candidate(1, ".2", ".3")])


def test_reported_roles_and_ci_are_not_recomputed():
    row = {"record_id": "reported_1", "source_selection": "best_ndcg|best_coverage|pareto",
           "ndcg_at_10": "0.1", "ndcg_at_10_ci95_half_width": "0.0020",
           "ndcg_at_10_reported": "0.100 ± 0.0020"}
    reference = {**row, "record_id": "reference_1", "source_selection": "baseline", "ndcg_at_10": "0.9"}
    selected, references = export.expand_reported_selections([row, reference])
    assert [r["selection"] for r in selected] == list(export.SELECTORS)
    assert all({k: v for k, v in r.items() if k != "selection"} == row for r in selected)
    assert references == [reference]


def test_main_replaces_pooled_distillation_with_separate_start_selections():
    source = {"record_id": "reported_1", "source_selection": "best_ndcg|best_coverage|pareto",
              "method": "prefix_ce", "initialization": "pooled", "ndcg_at_10": "0.95",
              "selection_scope": export.POOLED_INITIALIZATION_SCOPE}
    separate = {**source, "record_id": "reported_2", "method": "gflow",
                "initialization": "tuned_checkpoint", "selection_scope": "separate_initialization"}
    reported, _ = export.expand_reported_selections([source, separate])
    candidates = export.select_candidates([
        candidate(2, ".3", ".9", method="prefix_kl_q_pi", reward="sasrec"),
        candidate(3, ".9", ".3", method="prefix_kl_q_pi", initialization="sft"),
        candidate(4, ".8", ".8", method="q_distillation_mse"),
        candidate(5, ".6", ".6", method="q_distillation_mse", initialization="sft"),
        candidate(6, ".7", ".7", method="q_distillation_mse", source_namespace="experiment_b"),
    ])
    result = export.separate_initialization_selections(reported, candidates, list(source))
    assert len(reported) == 6  # Original source roles remain available separately.
    assert len(result) == 18  # Three original GFlow roles plus five separate groups.
    assert not any(r["selection_scope"] == export.POOLED_INITIALIZATION_SCOPE for r in result)
    assert {r["record_id"] for r in result if r["method"] == "gflow"} == {"reported_2"}
    assert all(r["ndcg_at_10"] == "0.95" for r in result if r["method"] == "gflow")
    for number in (2, 3, 4, 5, 6):
        group = [r for r in result if r["record_id"] == f"candidate_{number:04d}"]
        assert {r["selection"] for r in group} == set(export.SELECTORS)
        assert all(r["source_row"] == str(number) and r["source_sheet"] == "Metrics" for r in group)
        assert all(r["selection_basis"] == "recomputed_from_partial_candidate_snapshot" for r in group)
        assert all(r["ndcg_at_10_ci95_half_width"] == "" for r in group)
        assert all(r["ndcg_at_10_reported"] == r["ndcg_at_10"] for r in group)
    base = next(r for r in result if r["record_id"] == "candidate_0002")
    assert base["initialization"] == "base"  # No inference that base means tuned.
    assert json.loads(base["parameters_json"])["seed"] == 42
    assert json.loads(base["parameters_json"])["loss"] == "prefix_kl_q_pi"
    assert base["candidate_record_ids"] == "candidate_0002"


def test_new_grpo_ce_groups_are_additional_and_keep_historical_roles():
    historical = [
        {"record_id": f"reported_{method}", "dataset": "beauty", "model": "tiger",
         "method": method, "initialization": "tuned_checkpoint",
         "source_selection": "best_ndcg|best_coverage|pareto", "ndcg_at_10": "0.1230",
         "ndcg_at_10_ci95_half_width": "0.0020", "ndcg_at_10_reported": "0.1230 ± 0.0020",
         "selection_scope": "separate_initialization", "parameters_json": "{}"}
        for method in ("grpo", "ce")
    ]
    authored, _ = export.expand_reported_selections(historical)
    runs = [
        candidate(1, ".9", ".8", method="grpo", initialization="base"),
        candidate(2, ".8", ".9", method="grpo", initialization="sft"),
        candidate(3, ".7", ".7", method="ce", source_namespace="ce_protocol_a",
                  parameters_json='{"estimator":"exact"}'),
        candidate(4, ".6", ".6", method="ce", source_namespace="ce_protocol_b",
                  parameters_json='{"estimator":"sampled"}'),
    ]
    selected = export.separate_initialization_selections(
        authored, export.select_candidates(runs), list(historical[0]))
    assert len(selected) == 18
    for original in authored:
        kept = next(row for row in selected if row["record_id"] == original["record_id"]
                    and row["selection"] == original["selection"])
        assert all(kept[key] == value for key, value in original.items())
        assert kept["source_namespace"] == "reported_summary"
    added = [row for row in selected if row["record_id"].startswith("candidate_")]
    assert len(added) == 12
    assert {(row["method"], row["initialization"], row["source_namespace"]) for row in added} == {
        ("grpo", "base", "experiment_a"), ("grpo", "sft", "experiment_a"),
        ("ce", "base", "ce_protocol_a"), ("ce", "base", "ce_protocol_b"),
    }
    assert all(row["selection_basis"] == "recomputed_from_partial_candidate_snapshot" for row in added)
    assert all(row["ndcg_at_10_ci95_half_width"] == "" for row in added)


def test_ce_estimators_compete_as_configurations_without_averaging_seeds():
    runs = [candidate(1, ".9", ".1", method="ce", seed="42",
                      parameters_json='{"estimator":"exact","ce_target":"item"}'),
            candidate(2, ".1", ".9", method="ce", seed="43", repeat_index="2",
                      parameters_json='{"estimator":"sampled","ce_target":"item"}')]
    selected = export.select_candidates(runs)
    assert len(selected) == 3
    roles = by_role(selected)
    assert roles["best_ndcg"]["seed"] == "42"
    assert roles["best_coverage"]["seed"] == "43"
    assert roles["best_ndcg"]["ndcg_at_10"] == ".9"
    assert roles["best_coverage"]["coverage_at_10"] == ".9"
    assert all(row["candidate_count"] == 2 for row in selected)
    assert all(row["candidate_record_ids"] == "candidate_0001|candidate_0002" for row in selected)


def test_candidate_scientific_parameters_and_source_identity_reach_main_view():
    params = {"loss": "item_ce", "estimator": "sampled", "ce_target": "item",
              "importance_correction": True, "learning_rate": 0.00003, "num_samples": 10,
              "best_epoch": 3, "early_stopped": False, "source_protocol": "matched",
              "source_seed": 42, "optimizer_steps": 120, "target_grad_ratio": 0.1,
              "runtime_scope": "setup_fit_final_evaluation"}
    refs = [{"source_name": "ce_reports", "sheet": "results.csv", "row": 8}]
    run = candidate(8, ".4", ".7", method="ce", reward="sasrec", reward_power="4",
                    aux_sft_gradient_ratio="0.1", sft_weight="1", sampling="reward",
                    history="last_128_targets", evaluation_point="validation_selected_best",
                    runtime_seconds="12.50", epochs="4", parameters_json=json.dumps(params),
                    source_references=json.dumps(refs), input_digest_sha256="a" * 64)
    selected = export.separate_initialization_selections([], export.select_candidates([run]), [])
    for row in selected:
        saved = json.loads(row["parameters_json"])
        assert all(saved[key] == value for key, value in params.items())
        assert saved["reward_power"] == 4 and saved["rho"] == .1 and saved["seed"] == 42
        assert saved["runtime_seconds"] == 12.5 and saved["epochs"] == 4
        assert saved["history"] == row["history"] == "last_128_targets"
        assert saved["evaluation_point"] == "validation_selected_best"
        assert saved["sampling"] == "reward" and saved["sft_weight"] == 1
        assert row["ndcg_at_10"] == ".4" and row["ndcg_at_10_ci95_half_width"] == ""
        assert json.loads(row["source_references"]) == refs
        assert row["input_digest_sha256"] == "a" * 64


def test_grpo_reported_beta_does_not_imply_reward_power_or_other_coefficients():
    run = candidate(1, ".4", ".7", method="grpo", seed="", reward_power="",
                    parameters_json='{"reward_label":"S × H × I, β=1","reported_beta":1}')
    selected = export.separate_initialization_selections([], export.select_candidates([run]), [])
    params = json.loads(selected[0]["parameters_json"])
    assert params["reported_beta"] == 1 and params["reward_label"] == "S × H × I, β=1"
    assert params["reward_power"] is None and params["seed"] is None
    assert "beta" not in params and "kl_beta" not in params


@pytest.mark.parametrize("changes,match", [
    ({"execution_host": "fixture-host"}, "Unsupported candidate fields"),
    ({"parameters_json": '{"execution_host":"fixture-host"}'}, "unsupported scientific parameter"),
    ({"parameters_json": '{"reward":{"private_extra":"fixture"}}'}, "Invalid scientific parameter"),
    ({"parameters_json": '{"learning_rate":NaN}'}, "Invalid scientific parameter"),
    ({"parameters_json": '{"importance_correction":"true"}'}, "Invalid scientific parameter"),
    ({"source_references": '[{"source_name":"source","sheet":"Metrics","row":2,"extra":"x"}]'},
     "allow only source_name"),
    ({"source_references": '[{"source_name":"directory/source","sheet":"Metrics","row":2}]'},
     "public source identifier"),
    ({"source_references": '[{"source_name":"source","sheet":"Metrics","row":true}]'},
     "positive integer"),
    ({"input_digest_sha256": "not-a-digest"}, "SHA256"),
])
def test_candidate_schema_rejects_non_scientific_or_malformed_fields(changes, match):
    with pytest.raises(ValueError, match=match):
        export.select_candidates([candidate(1, ".5", ".5", **changes)])


def test_conflicting_scientific_parameter_is_not_silently_overwritten():
    run = candidate(1, ".5", ".5", method="grpo", seed="42", parameters_json='{"seed":43}')
    with pytest.raises(ValueError, match="Conflicting scientific parameter"):
        export.separate_initialization_selections([], export.select_candidates([run]), [])


@pytest.mark.parametrize("tags", ["", "best_accuracy", "baseline|pareto", "pareto|pareto"])
def test_invalid_source_roles_fail_instead_of_silently_dropping_rows(tags):
    with pytest.raises(ValueError):
        export.expand_reported_selections([{"record_id": "reported_1", "source_selection": tags}])


def csv_file(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def public_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    reported = {"record_id": "reported_1", "dataset": "beauty", "model": "tiger",
                "method": "gflow", "source_selection": "best_ndcg|best_coverage|pareto",
                "source_row": "2", "source_sheet": "Metrics", "evaluation_split": "test",
                "source_label": '=HYPERLINK("https://example.org", "literal source text")',
                "parameters_json": '{"reward":"sasrec","seed":42}',
                "ndcg_at_10": "0.100", "ndcg_at_10_ci95_half_width": "0.0020",
                "ndcg_at_10_reported": "0.100 ± 0.0020", "coverage_at_10": "0.5"}
    reference = {**reported, "record_id": "reference_1", "source_selection": "baseline"}
    csv_file(inputs / "reported_metrics.csv", [reported, reference])
    csv_file(inputs / "candidate_metrics.csv", [candidate(2, ".9", ".1"), candidate(3, ".1", ".9")])
    csv_file(inputs / "dataset_statistics.csv", [{"dataset": "beauty", "users": "100", "items": "50"}])
    (inputs / "provenance.json").write_text(json.dumps({"schema_version": 1, "source_policy": "separate snapshots"}))
    return inputs


def test_csv_export_preserves_inputs_and_source_precision(public_inputs, tmp_path, monkeypatch):
    before = {p.name: p.read_bytes() for p in public_inputs.iterdir()}
    def no_workbook(*args, **kwargs):
        raise AssertionError("CSV-only export must not use openpyxl")
    monkeypatch.setattr(export, "write_workbook", no_workbook)
    output = tmp_path / "csv"
    counts = export.export_metrics(public_inputs, output, skip_xlsx=True)
    assert counts == {"reported_rows": 2, "selected_roles": 3, "reference_rows": 1,
                      "candidate_rows": 2, "candidate_selection_rows": 3, "reported_selection_roles": 3}
    _, selected = export.read_csv(output / "selected_metrics.csv")
    assert all(row["ndcg_at_10"] == "0.100" and row["ndcg_at_10_reported"] == "0.100 ± 0.0020" for row in selected)
    assert {p.name: p.read_bytes() for p in public_inputs.iterdir()} == before
    assert not (output / "metrics.xlsx").exists()


def test_workbook_is_literal_numeric_and_reproducible(public_inputs, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    first, second = tmp_path / "first", tmp_path / "second"
    export.export_metrics(public_inputs, first)
    export.export_metrics(public_inputs, second)
    assert (first / "metrics.xlsx").read_bytes() == (second / "metrics.xlsx").read_bytes()
    book = openpyxl.load_workbook(first / "metrics.xlsx", data_only=False)
    assert book.properties.creator == "FlowRec"
    assert book.properties.lastModifiedBy == "FlowRec"
    assert book.properties.created.year == book.properties.modified.year == 2000
    assert all(sheet.freeze_panes and sheet.auto_filter.ref for sheet in book)
    assert all(cell.data_type != "f" and cell.hyperlink is None and cell.comment is None
               for sheet in book for row in sheet for cell in row)
    values = [cell.value for row in book["Selected metrics"] for cell in row]
    assert "0.100 ± 0.0020" in values
    assert '=HYPERLINK("https://example.org", "literal source text")' in values
    raw = book["Candidate runs"]
    ndcg_column = next(cell.column for cell in raw[1] if cell.value == "NDCG@10")
    assert raw.cell(2, ndcg_column).data_type == "n"
    assert raw.cell(2, ndcg_column).value == pytest.approx(0.9)
    assert book["Dataset statistics"]["B2"].data_type == "n"
    selected = book["Selected metrics"]
    ci_column = next(cell.column for cell in selected[1] if cell.value == "NDCG@10 95% CI half-width")
    assert selected.cell(2, ci_column).data_type == "n"
    assert selected.cell(2, ci_column).value == pytest.approx(0.002)
    with ZipFile(first / "metrics.xlsx") as archive:
        assert all(info.date_time == (2000, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert not any("externalLinks" in name or "comments" in name for name in archive.namelist())
