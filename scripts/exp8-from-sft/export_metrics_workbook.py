#!/usr/bin/env python3
"""Export final results, epoch histories and available events to one workbook.

Run after collect_metrics.py. Requires openpyxl; input files remain untouched.
Histories/events keep their original run_dir. Unfinished CE runs are explicitly
marked no_final_result and never added to the final-results sheet.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts/exp8-from-sft/all_metrics.csv"
OUTPUT = ROOT / "artifacts/exp8-from-sft/all_metrics.xlsx"
EXPERIMENTS = (ROOT / "experiments/exp8-from-sft", ROOT / "experiments/exp4-gflow-tiger-from-sft")
IDENTITY = ("protocol", "source_id", "source_seed", "model", "dataset", "arm", "variant", "seed",
            "combo", "objective", "sampling", "rho", "power", "num_samples", "num_negatives",
            "configuration_id", "status", "run_dir", "source_result", "source_metadata")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
TEXT_COLUMNS = {"protocol", "source_id", "model", "dataset", "arm", "variant", "combo", "objective",
                "configuration_id", "status", "sampling", "run_dir", "source_result", "source_metadata",
                "source_history", "source_event"}


def local_run_dir(value):
    marker = "experiments/"
    index = value.find(marker)
    if index < 0:
        raise ValueError(f"Run directory has no experiments/ component: {value!r}")
    return value[index:].rstrip("/")


def flat(obj, prefix=""):
    """Expand nested objects; lists remain JSON. Detect ambiguous key collisions."""
    out = {}
    def add(value, key):
        if isinstance(value, dict) and value:
            for subkey, child in value.items():
                add(child, f"{key}.{subkey}" if key else subkey)
        else:
            if key in out:
                raise ValueError(f"Ambiguous flattened field: {key}")
            out[key] = (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(value, (dict, list)) else value)
    add(obj, prefix)
    return out


def to_cell(sheet, column, value):
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        value = str(value)
    if isinstance(value, str):
        if len(value) > 32767:
            raise ValueError(f"Value exceeds Excel's 32767-character limit in {column}")
        if column not in TEXT_COLUMNS and NUMBER.fullmatch(value):
            if re.fullmatch(r"[+-]?\d+", value):
                # Preserve identifiers and integers Excel cannot represent exactly.
                if len(value.lstrip("+-")) <= 15:
                    value = int(value)
            else:
                candidate = float(value)
                if math.isfinite(candidate):
                    value = candidate
        elif value in ("True", "False"):
            value = value == "True"
        if isinstance(value, str) and value.startswith("="):
            cell = WriteOnlyCell(sheet, value=value)
            cell.data_type = "s"
            return cell
    return value


def rows_for_csv():
    with METRICS.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("CSV has no header or duplicate columns")
        return columns, list(reader)


def history_records(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"history"} or not isinstance(data["history"], list):
        raise ValueError(f"Unexpected history format: {path}")
    for index, record in enumerate(data["history"]):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid history entry {index}: {path}")
        yield index, record


def event_records(path):
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source, 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid event at line {index}: {path}")
            yield index, record


def unfinished_run_identity(run):
    """Recover the recorded identity of a CE run without a final result.

    The absence of a local final result says nothing about the state of its original process.
    Read explicit source/run metadata rather than infer model or settings from
    directory names; paths are used to check that metadata belongs to this run.
    """
    directory = ROOT / run
    try:
        parts = directory.relative_to(ROOT / "experiments/exp8-from-sft").parts
    except ValueError as exc:
        raise ValueError(f"No final result or supported unfinished-run metadata: {directory}") from exc
    if len(parts) < 5 or parts[2] != "ce":
        raise ValueError(f"No final result or supported unfinished CE identity: {directory}")
    source_path = ROOT / "experiments/exp8-from-sft" / parts[0] / parts[1] / "source_identity.json"
    identity_path = directory / "identity.json"
    resolved_path = directory / "resolved_config.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if (not isinstance(source, dict) or not isinstance(identity, dict)
            or not isinstance(resolved, dict) or resolved.get("identity") != identity
            or identity.get("from_sft_source") != source):
        raise ValueError(f"Unfinished CE source/config identity mismatch: {directory}")
    entry = identity.get("entry")
    if (not isinstance(entry, dict) or not source.get("model")
            or parts[0] != source.get("source_id")
            or parts[1] != f"sft_seed_{source.get('source_seed')}"
            or parts[3] != source.get("dataset") or entry.get("dataset") != source.get("dataset")
            or parts[-1] != f"seed_{entry.get('seed')}"
            or identity.get("model") != source.get("model")):
        raise ValueError(f"Unfinished CE identity disagrees with its path/source: {directory}")
    row = {key: identity[key] for key in IDENTITY if key in identity}
    row.update({key: entry[key] for key in IDENTITY if key in entry})
    row.update({key: source[key] for key in ("source_id", "source_seed", "model", "dataset")})
    row.update(protocol="exp8_from_sft", arm="ce", variant="ce", status="no_final_result",
               run_dir=run, source_metadata=str(identity_path.relative_to(ROOT)))
    return row


def discover(paths, index, kind):
    headers, count = set(), 0
    for path in paths:
        run = str(path.parent.relative_to(ROOT))
        if run not in index:
            raise ValueError(f"No run identity matches {kind}: {path}")
        read = history_records if kind == "history" else event_records
        for _, record in read(path):
            headers.update(flat(record, kind).keys())
            count += 1
    return sorted(headers), count


def append(sheet, columns, record):
    sheet.append([to_cell(sheet, col, record.get(col)) for col in columns])


def main():
    columns, results = rows_for_csv()
    index = {}
    for row in results:
        if not row.get("run_dir"):
            if row.get("arm") != "init":
                raise ValueError(f"Non-baseline result has no run_dir: {row.get('source_result')}")
            continue
        key = local_run_dir(row["run_dir"])
        if key in index:
            raise ValueError(f"Multiple result rows share a run_dir: {key}")
        index[key] = row
    histories = sorted(p for root in EXPERIMENTS if root.exists() for p in root.rglob("training_history.json"))
    events = sorted(p for root in EXPERIMENTS if root.exists() for p in root.rglob("training_log.jsonl"))
    history_dirs = {str(path.parent.relative_to(ROOT)) for path in histories}
    if set(index) - history_dirs:
        raise ValueError(f"History/result mismatch: missing={sorted(set(index)-history_dirs)}")
    event_dirs = {str(path.parent.relative_to(ROOT)) for path in events}
    unfinished = sorted((history_dirs | event_dirs) - set(index))
    for run in unfinished:
        index[run] = unfinished_run_identity(run)
    history_columns, history_count = discover(histories, index, "history")
    event_columns, event_count = discover(events, index, "event")
    wb = Workbook(write_only=True)
    summary = wb.create_sheet("results")
    summary.freeze_panes = "A2"
    if "result_json" in columns:
        raise ValueError("CSV already contains reserved result_json column")
    summary_columns = columns + ["result_json"]
    summary.append(summary_columns)
    for row in results:
        record = dict(row)
        # Excel numbers have finite display/storage precision; preserve the
        # source CSV strings too so no original significant digits are lost.
        record["result_json"] = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        append(summary, summary_columns, record)
    for title, paths, dynamic in (("history", histories, history_columns), ("events", events, event_columns)):
        sheet = wb.create_sheet(title)
        sheet.freeze_panes = "A2"
        source_column = "source_history" if title == "history" else "source_event"
        columns_out = list(IDENTITY) + [source_column, "record_index", f"{title}_json"] + dynamic
        sheet.append(columns_out)
        for path in paths:
            result = index[str(path.parent.relative_to(ROOT))]
            read = history_records if title == "history" else event_records
            for record_index, record in read(path):
                data = {key: result.get(key) for key in IDENTITY}
                data[source_column] = str(path.relative_to(ROOT))
                data["record_index"] = record_index
                data[f"{title}_json"] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                data.update(flat(record, title if title == "history" else "event"))
                append(sheet, columns_out, data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".all_metrics.", suffix=".xlsx", dir=OUTPUT.parent)
    os.close(fd)
    try:
        wb.save(tmp)
        os.replace(tmp, OUTPUT)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(results)} final rows, "
          f"{len(histories)} histories / {history_count} entries, "
          f"{len(events)} event files / {event_count} entries; "
          f"{len(unfinished)} runs marked no_final_result in history/events")


if __name__ == "__main__":
    main()
