#!/usr/bin/env python3
"""Build public CSV views and an optional workbook from scientific metric inputs.

Reported selections are transcribed and retained alongside new CE/GRPO candidate
groups. Candidate selectors operate on individual runs without averaging seeds
or repetitions. Only XLSX export requires openpyxl.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DEFAULT_DIR = Path(__file__).resolve().parents[1] / "results" / "icde2027"
SELECTORS = ("best_ndcg", "best_coverage", "pareto")
METRICS = tuple(f"{metric}_at_{k}" for k in (5, 10, 20)
                for metric in ("hr", "ndcg", "coverage", "arp"))
CANDIDATE_GROUP_FIELDS = (
    "dataset", "model", "initialization", "method_family",
    "source_namespace", "calibration_protocol",
    "evaluation_stage", "dataset_version", "split_protocol", "initial_checkpoint_id",
    "history", "evaluation_point",
)
SELECTION_FIELDS = (
    "method_family", "selection", "selection_basis", "candidate_count",
    "primary_tie_count", "pareto_front_size", "pareto_unique_point_count",
    "candidate_record_ids", "primary_tied_record_ids", "pareto_record_ids",
    "pareto_distance_squared",
)
MAIN_PROVENANCE_FIELDS = (
    "selection_basis", "source_namespace", "calibration_protocol", "primary_tie_count",
    "candidate_record_ids", "primary_tied_record_ids", "pareto_record_ids",
    "source_references", "input_digest_sha256",
)
POOLED_INITIALIZATION_SCOPE = "pooled_tuned_and_sft_start_stages"
DISTILLATION_METHODS = {"prefix_kl_q_pi": "prefix_ce", "q_distillation_mse": "q_mse"}
MAIN_CANDIDATE_METHODS = {**DISTILLATION_METHODS, "grpo": "grpo", "ce": "ce"}
CANDIDATE_INPUT_FIELDS = {
    "record_id", "source_workbook", "source_sheet", "source_row", "registry_id",
    "registry_source_row", "dataset", "model", "initialization", "method", "reward",
    "reward_power", "aux_sft_gradient_ratio", "sft_weight", "sampling", "seed",
    "repeat_index", "evaluation_stage", "dataset_version", "split_protocol",
    "initial_checkpoint_id", *METRICS, "runtime_seconds", "epochs",
    "calibration_protocol", "source_namespace", "parameters_json", "history",
    "evaluation_point", "source_references", "input_digest_sha256",
}
PARAMETER_STRING_FIELDS = {
    "loss", "reward", "reward_label", "sampling", "source_namespace",
    "calibration_protocol", "history", "evaluation_point", "estimator", "ce_target",
    "calibration_sampling", "target_selection", "train_layout", "best_status",
    "timing_status", "runtime_scope", "checkpoint_selection_metric", "starting_checkpoint_role",
    "source_protocol", "variant", "sampling_algorithm",
}
PARAMETER_NUMBER_FIELDS = {
    "reward_power", "rho", "sft_weight", "seed", "source_seed", "repeat_index",
    "best_epoch", "num_negatives", "num_samples", "dropout", "reported_seed_count",
    "reported_beta", "curriculum_epochs", "curriculum_steps", "proposal_q_mix",
    "lambda_distill", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "train_num_examples", "catalog_size", "test_num_examples",
    "validation_num_examples", "max_history_len", "train_targets_per_user",
    "train_pool_last_n", "source_trial", "epochs_completed", "best_val_ndcg5",
    "best_time_seconds", "fit_time_seconds", "setup_time_seconds", "evaluation_time_seconds",
    "resumed_from_epoch", "runtime_seconds", "epochs",
    "optimizer_steps", "micro_steps", "target_grad_ratio", "interaction_exponent",
}
PARAMETER_BOOLEAN_FIELDS = {"importance_correction", "early_stopped"}
FIXED_TIME = datetime(2000, 1, 1)
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
NUMERIC_FIELDS = {
    "source_row", "registry_source_row", "candidate_count", "primary_tie_count",
    "pareto_front_size", "pareto_unique_point_count", "pareto_distance_squared",
    "seed", "repeat_index", "reward_power", "aux_sft_gradient_ratio", "sft_weight",
    "runtime_seconds", "epochs", "users", "items", "training_targets",
    "validation_targets", "test_targets",
}
NUMERIC_PARAMETERS = PARAMETER_NUMBER_FIELDS


def validate_candidate_fields(fields: list[str] | dict[str, Any]) -> None:
    unknown = set(fields) - CANDIDATE_INPUT_FIELDS
    if unknown:
        raise ValueError(f"Unsupported candidate fields: {', '.join(sorted(unknown))}")


def scientific_parameters(row: dict[str, Any]) -> dict[str, Any]:
    """Accept only typed scientific settings from the public candidate schema."""
    raw = row.get("parameters_json") or "{}"
    try:
        params = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate parameters_json must be a JSON object") from exc
    allowed = PARAMETER_STRING_FIELDS | PARAMETER_NUMBER_FIELDS | PARAMETER_BOOLEAN_FIELDS
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError("Candidate parameters_json contains unsupported scientific parameter keys")
    for key, value in params.items():
        if value is None:
            continue
        if key in PARAMETER_STRING_FIELDS:
            valid = isinstance(value, str)
        elif key in PARAMETER_BOOLEAN_FIELDS:
            valid = isinstance(value, bool)
        else:
            valid = not isinstance(value, bool) and isinstance(value, (int, float, str))
            try:
                valid = valid and Decimal(str(value)).is_finite()
            except InvalidOperation:
                valid = False
        if not valid:
            raise ValueError(f"Invalid scientific parameter type/value: {key}")
    return params


def validate_source_references(row: dict[str, Any]) -> None:
    raw = row.get("source_references") or "[]"
    try:
        references = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate source_references must be a JSON list") from exc
    if not isinstance(references, list):
        raise ValueError("Candidate source_references must be a JSON list")
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"source_name", "sheet", "row"}:
            raise ValueError("Source references allow only source_name, sheet and row")
        name, sheet, number = (reference[key] for key in ("source_name", "sheet", "row"))
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Source reference source_name must be a public source identifier")
        if not isinstance(sheet, str) or not re.fullmatch(r"[\w .()+-]+", sheet):
            raise ValueError("Source reference sheet must be a plain sheet label")
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError("Source reference row must be a positive integer")
    digest = row.get("input_digest_sha256")
    if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Candidate input_digest_sha256 must be a SHA256 hexadecimal digest")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"Missing or duplicate CSV headers in {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Inconsistent CSV row width in {path.name}")
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expand_reported_selections(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Preserve source roles and every original metric and uncertainty string."""
    selected, references = [], []
    for row in rows:
        tags = row.get("source_selection", "").split("|")
        if len(tags) != len(set(tags)) or set(tags) - {*SELECTORS, "baseline"}:
            raise ValueError(f"Invalid source_selection for {row.get('record_id', '')}")
        if "baseline" in tags:
            if tags != ["baseline"]:
                raise ValueError("A reference row cannot also carry selection roles")
            references.append(dict(row))
        else:
            for selector in SELECTORS:
                if selector in tags:
                    selected.append({**row, "selection": selector})
    return selected, references


def metric_value(row: dict[str, Any], field: str) -> Decimal:
    try:
        value = Decimal(str(row[field]))
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field} for {row.get('record_id', '')}") from exc
    if not value.is_finite() or not Decimal(0) <= value <= Decimal(1):
        raise ValueError(f"{field} must be a finite fraction for {row.get('record_id', '')}")
    return value


def stable_key(row: dict[str, Any]) -> tuple[Any, ...]:
    try:
        number = int(row["source_row"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each candidate needs an integer source_row") from exc
    return (number, str(row.get("source_workbook", "")), str(row.get("source_sheet", "")),
            str(row.get("record_id", "")))


def method_family(method: str) -> str:
    return "gflow" if method in {"gflow_tb", "gflow_db", "gflow_subtb"} else method


def select_candidates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Select extremes and a normalized Pareto compromise within each protocol.

    Front size counts physical rows, including identical points. Distances use
    min/max over the frontier only; a constant coordinate contributes zero.
    """
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    identities: set[str] = set()
    for original in rows:
        validate_candidate_fields(original)
        scientific_parameters(original)
        validate_source_references(original)
        row = {**original, "method_family": method_family(original["method"])}
        record_id = row.get("record_id", "")
        if not record_id or record_id in identities:
            raise ValueError("Candidate record_id values must be nonempty and unique")
        identities.add(record_id)
        stable_key(row)
        metric_value(row, "ndcg_at_10")
        metric_value(row, "coverage_at_10")
        key = tuple(row.get(field, "") for field in CANDIDATE_GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    result = []
    with localcontext() as context:
        context.prec = 50
        for key in sorted(groups):
            candidates = sorted(groups[key], key=stable_key)
            points = {row["record_id"]: (metric_value(row, "ndcg_at_10"),
                                          metric_value(row, "coverage_at_10"))
                      for row in candidates}
            frontier = [row for row in candidates if not any(
                q[0] >= points[row["record_id"]][0]
                and q[1] >= points[row["record_id"]][1]
                and q != points[row["record_id"]]
                for q in points.values()
            )]
            minima = [min(points[row["record_id"]][i] for row in frontier) for i in (0, 1)]
            maxima = [max(points[row["record_id"]][i] for row in frontier) for i in (0, 1)]
            distances = {
                row["record_id"]: sum(
                    ((maxima[i] - points[row["record_id"]][i]) / (maxima[i] - minima[i])) ** 2
                    if maxima[i] != minima[i] else Decimal(0)
                    for i in (0, 1)
                ) for row in frontier
            }
            roles = ("baseline",) if key[3] == "sft_only" else SELECTORS
            for selector in roles:
                if selector in {"best_ndcg", "best_coverage"}:
                    axis = 0 if selector == "best_ndcg" else 1
                    maximum = max(point[axis] for point in points.values())
                    tied = [row for row in candidates if points[row["record_id"]][axis] == maximum]
                    winner = min(tied, key=lambda row: (-points[row["record_id"]][1 - axis], stable_key(row)))
                elif selector == "pareto":
                    nearest = min(distances.values())
                    tied = [row for row in frontier if distances[row["record_id"]] == nearest]
                    winner = min(tied, key=lambda row: (-points[row["record_id"]][0],
                                                       -points[row["record_id"]][1], stable_key(row)))
                else:
                    # A reference is never selected by its performance.
                    if len(candidates) != 1:
                        raise ValueError("Multiple SFT-only references in one protocol; identify them separately")
                    winner = candidates[0]
                    tied = candidates
                result.append({
                    **winner, "selection": selector,
                    "selection_basis": "reported_test_candidates" if winner.get("evaluation_stage") == "test"
                    else f"reported_{winner.get('evaluation_stage') or 'unspecified'}_candidates",
                    "candidate_count": len(candidates), "primary_tie_count": len(tied),
                    "pareto_front_size": len(frontier),
                    "pareto_unique_point_count": len({points[row["record_id"]] for row in frontier}),
                    "candidate_record_ids": "|".join(row["record_id"] for row in candidates),
                    "primary_tied_record_ids": "|".join(row["record_id"] for row in tied),
                    "pareto_record_ids": "|".join(row["record_id"] for row in frontier),
                    "pareto_distance_squared": str(distances.get(winner["record_id"], "")),
                })
    return result


def separate_initialization_selections(
    reported: list[dict[str, str]], candidate_selected: list[dict[str, Any]],
    reported_fields: list[str],
) -> list[dict[str, Any]]:
    """Keep authored selections and add independent CE/GRPO candidate groups.

    The original pooled selections are still exported separately in full. The
    Prefix-CE/Q-MSE replacement rows describe only the available per-run snapshot.
    New CE/GRPO groups do not replace historical selections with unverified
    protocol or repetition identities.
    """
    result = [{**row, "selection_basis": "source_reported_separate_initialization",
               "source_namespace": "reported_summary"}
              for row in reported if row.get("selection_scope") != POOLED_INITIALIZATION_SCOPE]
    for row in candidate_selected:
        if row["method"] not in MAIN_CANDIDATE_METHODS or row["selection"] not in SELECTORS:
            continue
        params = scientific_parameters(row)
        defaults: dict[str, Any] = {
            "loss": row["method"], "reward": row.get("reward"), "sampling": row.get("sampling"),
            "source_namespace": row.get("source_namespace"),
            "calibration_protocol": row.get("calibration_protocol"),
        }
        for key, value in defaults.items():
            params.setdefault(key, value)
        for source, target in (("reward_power", "reward_power"), ("aux_sft_gradient_ratio", "rho"),
                               ("sft_weight", "sft_weight"), ("seed", "seed"),
                               ("repeat_index", "repeat_index"), ("runtime_seconds", "runtime_seconds"),
                               ("epochs", "epochs")):
            value = row.get(source, "")
            if value != "":
                converted_value = workbook_value(f"parameter_{target}", value)
                if target in params and params[target] is not None:
                    try:
                        same = Decimal(str(params[target])) == Decimal(str(converted_value))
                    except InvalidOperation:
                        same = params[target] == converted_value
                    if not same:
                        raise ValueError(f"Conflicting scientific parameter and candidate column: {target}")
                params[target] = converted_value
            elif target not in {"runtime_seconds", "epochs"}:
                params.setdefault(target, None)
        for field in ("history", "evaluation_point"):
            if row.get(field):
                if field in params and params[field] != row[field]:
                    raise ValueError(f"Conflicting scientific parameter and candidate column: {field}")
                params[field] = row[field]
        converted = {field: "" for field in reported_fields}
        converted.update({
            "record_id": row["record_id"], "dataset": row["dataset"], "model": row["model"],
            "method": MAIN_CANDIDATE_METHODS[row["method"]], "initialization": row["initialization"],
            "history": row.get("history") or "not_reported", "evaluation_split": row.get("evaluation_stage", ""),
            "selection": row["selection"],
            "selection_scope": "same_candidate_protocol_group",
            "source_row": row["source_row"], "source_sheet": row.get("source_sheet", ""),
            "source_label": f"{row['model']} {MAIN_CANDIDATE_METHODS[row['method']]} from {row['initialization']}",
            "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "candidate_count": row["candidate_count"], "pareto_front_size": row["pareto_front_size"],
            "notes": "partial_candidate_snapshot;single_run_metrics",
            **{field: row.get(field, "") for field in MAIN_PROVENANCE_FIELDS},
            "selection_basis": "recomputed_from_partial_candidate_snapshot",
        })
        for metric in METRICS:
            converted[metric] = row.get(metric, "")
            converted[f"{metric}_reported"] = row.get(metric, "")
            converted[f"{metric}_ci95_half_width"] = ""
        result.append(converted)
    return result


def display_header(field: str) -> str:
    match = re.fullmatch(r"(hr|ndcg|coverage|arp)_at_(5|10|20)(.*)", field)
    if match:
        metric, k, suffix = match.groups()
        label = {"hr": "HR", "ndcg": "NDCG", "coverage": "Coverage", "arp": "ARP"}[metric]
        return f"{label}@{k}" + {"": "", "_ci95_half_width": " 95% CI half-width",
                                  "_reported": " (reported)"}.get(suffix, suffix)
    return field.replace("_", " ").capitalize()


def reported_workbook_view(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Use source display strings so reported CI notation survives unchanged."""
    parameters = [json.loads(row.get("parameters_json") or "{}") for row in rows]
    parameter_keys = sorted({key for record in parameters for key in record})
    fields = ["dataset", "model", "initialization", "history", "method", "selection",
              "selection_basis", *[f"{metric}_reported" for metric in METRICS],
              *[f"parameter_{key}" for key in parameter_keys],
              *[field for metric in METRICS for field in (metric, f"{metric}_ci95_half_width")],
              "candidate_count", "pareto_front_size", "selection_scope", "evaluation_split",
              "record_id", "source_sheet", "source_row", "source_label", "notes",
              *[field for field in MAIN_PROVENANCE_FIELDS if field != "selection_basis"]]
    result = []
    for row, params in zip(rows, parameters):
        record = {field: row.get(field, "") for field in fields}
        record["selection"] = row.get("selection", "baseline")
        for metric in METRICS:
            display = row.get(f"{metric}_reported", "")
            if display == "":
                display = row.get(metric, "")
                ci = row.get(f"{metric}_ci95_half_width", "")
                if ci and display:
                    display += f" ± {ci}"
            record[f"{metric}_reported"] = display
        for key, value in params.items():
            record[f"parameter_{key}"] = value
        result.append(record)
    return fields, result


def workbook_value(field: str, value: Any) -> Any:
    """Convert scientific numbers without interpreting identifiers or formulas."""
    numeric = field in NUMERIC_FIELDS or field in METRICS or (
        field.endswith("_ci95_half_width") and field.removesuffix("_ci95_half_width") in METRICS
    ) or (field.startswith("parameter_") and field.removeprefix("parameter_") in NUMERIC_PARAMETERS)
    if numeric and isinstance(value, str) and NUMBER.fullmatch(value):
        number = Decimal(value)
        if number.is_finite():
            # Preserve long integer text instead of losing identifier precision.
            if number == number.to_integral_value():
                return int(number) if len(str(abs(int(number)))) <= 15 else value
            as_float = float(number)
            if as_float not in (float("inf"), float("-inf")):
                return as_float
    return value


def write_workbook(
    path: Path, tables: list[tuple[str, list[str], list[dict[str, Any]]]],
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Install optional openpyxl or pass --skip-xlsx") from exc

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.creator = "FlowRec"
    workbook.properties.lastModifiedBy = "FlowRec"
    workbook.properties.title = "FlowRec reported metrics"
    workbook.properties.created = FIXED_TIME
    workbook.properties.modified = FIXED_TIME
    for name, fields, rows in tables:
        sheet = workbook.create_sheet(name)
        sheet.freeze_panes = "C2"
        for col, field in enumerate(fields, 1):
            cell = sheet.cell(1, col, display_header(field))
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="16324F")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        for index, row in enumerate(rows, 2):
            for col, field in enumerate(fields, 1):
                value = row.get(field)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if value == "":
                    value = None
                value = workbook_value(field, value)
                if isinstance(value, str) and len(value) > 32767:
                    raise ValueError(f"Excel cell exceeds 32767 characters: {name}/{field}")
                cell = sheet.cell(index, col, value)
                # Source text always stays literal, including leading '=', '+',
                # '-' and '@'. No source formulas or hyperlinks are carried over.
                if isinstance(value, str):
                    cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if index % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="EDF3F8")
            sheet.row_dimensions[index].height = 30
        sheet.row_dimensions[1].height = 42
        sheet.auto_filter.ref = sheet.dimensions
        for col, field in enumerate(fields, 1):
            width = 23 if field in METRICS or field.endswith("_reported") else 19
            if field in {"source_label", "notes", "value", "selection_scope"}:
                width = 65
            elif field.endswith("record_ids"):
                width = 40
            sheet.column_dimensions[get_column_letter(col)].width = width

    buffer = BytesIO()
    workbook.save(buffer)
    # openpyxl versions can update core.xml's modified time during save. Fix
    # both XML dates and ZIP metadata so identical inputs produce identical ZIPs.
    with ZipFile(BytesIO(buffer.getvalue())) as source, ZipFile(path, "w", compression=ZIP_DEFLATED,
                                                              compresslevel=9) as target:
        for filename in sorted(source.namelist()):
            content = source.read(filename)
            if filename == "docProps/core.xml":
                content = re.sub(rb"(<dcterms:(?:created|modified)\b[^>]*>)[^<]*(</dcterms:(?:created|modified)>)",
                                 rb"\g<1>2000-01-01T00:00:00Z\g<2>", content)
            info = ZipInfo(filename, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            target.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)


def export_metrics(input_dir: Path, output_dir: Path, *, skip_xlsx: bool = False) -> dict[str, int]:
    reported_fields, reported = read_csv(input_dir / "reported_metrics.csv")
    candidate_fields, candidates = read_csv(input_dir / "candidate_metrics.csv")
    validate_candidate_fields(candidate_fields)
    stats_fields, statistics = read_csv(input_dir / "dataset_statistics.csv")
    provenance = json.loads((input_dir / "provenance.json").read_text(encoding="utf-8"))
    reported_selected, references = expand_reported_selections(reported)
    candidate_selected = select_candidates(candidates)
    selected = separate_initialization_selections(reported_selected, candidate_selected, reported_fields)
    selection_fields = candidate_fields + [field for field in SELECTION_FIELDS if field not in candidate_fields]
    main_fields = ["selection", *reported_fields, *MAIN_PROVENANCE_FIELDS]
    counts = {"reported_rows": len(reported), "selected_roles": len(selected),
              "reported_selection_roles": len(reported_selected),
              "reference_rows": len(references), "candidate_rows": len(candidates),
              "candidate_selection_rows": len(candidate_selected)}
    output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_xlsx:
        selected_view = reported_workbook_view(selected)
        reported_selected_view = reported_workbook_view(reported_selected)
        reference_view = reported_workbook_view(references)
        guide = [
            {"topic": "Reported selections", "value": "Source-author selections, preserved without reranking against the separate partial candidate snapshot."},
            {"topic": "Separate starting checkpoints", "value": "Selected metrics retain authored selections under reported_summary. New CE/GRPO candidate namespaces are additional groups and do not replace historical authored selections. Pooled Prefix-CE/Q-MSE roles retain their existing separate-start candidate views. All original roles remain in Reported selections."},
            {"topic": "Selection split", "value": "Reported selections use test metrics. These are descriptive post-hoc selections, not validation-selected held-out estimates."},
            {"topic": "Uncertainty", "value": "Reported ± values are Student-t 95% CI half-widths where supplied; per-row seed counts are not consistently available. Blank uncertainty is unknown, not zero."},
            {"topic": "Candidate selection", "value": "Individual runs; no seed/repetition averaging. Groups retain dataset, model, initialization, method family, source namespace, calibration protocol, history and evaluation point, plus any known evaluation metadata."},
            {"topic": "Pareto", "value": "Maximize NDCG@10 and Coverage@10. Normalize over the nondominated frontier; choose nearest to (1,1). Ties prefer NDCG, then Coverage, then source row. Equal points and tie counts are retained."},
            {"topic": "Metric units", "value": "HR/NDCG/Coverage are fractions. ARP is mean item popularity in source units. Raw CSVs retain exact numeric text; workbook metric strings preserve supplied CI notation."},
            {"topic": "Missing metadata", "value": "Available split protocols, starting-checkpoint identities and scientific settings are retained per row; empty values remain unknown. Source namespaces identify batches, not proof of identical evaluation populations."},
        ]
        metadata = [{"key": key, "value": value} for key, value in sorted(provenance.items())]
        metadata += [{"key": f"input_sha256:{name}", "value": hashlib.sha256((input_dir / name).read_bytes()).hexdigest()}
                     for name in ("reported_metrics.csv", "candidate_metrics.csv", "dataset_statistics.csv")]
        write_workbook(output_dir / "metrics.xlsx", [
            ("Readme", ["topic", "value"], guide),
            ("Selected metrics", *selected_view),
            ("Reported selections", *reported_selected_view),
            ("Reference metrics", *reference_view),
            ("Candidate selections", selection_fields, candidate_selected),
            ("Candidate runs", candidate_fields, candidates),
            ("Dataset statistics", stats_fields, statistics),
            ("Provenance", ["key", "value"], metadata),
        ])
    write_csv(output_dir / "selected_metrics.csv", main_fields, selected)
    write_csv(output_dir / "reported_selections.csv", ["selection", *reported_fields], reported_selected)
    write_csv(output_dir / "reference_metrics.csv", reported_fields, references)
    write_csv(output_dir / "candidate_selections.csv", selection_fields, candidate_selected)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--skip-xlsx", action="store_true", help="Export CSVs using only the Python standard library")
    args = parser.parse_args()
    counts = export_metrics(args.input_dir, args.output_dir, skip_xlsx=args.skip_xlsx)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
