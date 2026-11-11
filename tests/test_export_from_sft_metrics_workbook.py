"""Completed metrics and unfinished CE events stay distinguishable in Excel."""
import csv
import importlib.util
import json
from pathlib import Path

from openpyxl import load_workbook
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_from_sft_workbook", ROOT / "scripts/exp7-from-sft/export_metrics_workbook.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    experiments = tmp_path / "experiments/exp8-from-sft"
    source_root = experiments / "gpt2_beauty_2023/sft_seed_42"
    source = dict(source_id="gpt2_beauty_2023", source_seed=42, model="gpt2", dataset="beauty_2023")
    write_json(source_root / "source_identity.json", source)
    directory = source_root / "ce/beauty_2023/sasrec/curriculum_smooth_is_n3_epochs5_qmix0p1/pow_0p25/rho_0p1/seed_42"
    identity = dict(from_sft_source=source, model="gpt2", sampling="curriculum_smooth_is", num_samples=3,
                    entry=dict(dataset="beauty_2023", combo="sasrec", power=.25, rho=.1, seed=42))
    write_json(directory / "identity.json", identity)
    write_json(directory / "resolved_config.json", {"identity": identity})
    event = directory / "training_log.jsonl"
    event.write_text('{"type":"validation_end","val/sft/ndcg@5":0.17}\n')
    metrics = tmp_path / "all_metrics.csv"
    output = tmp_path / "all_metrics.xlsx"
    complete = source_root / "sft/cell/seed_42"
    write_json(complete / "training_history.json", {"history": [{"epoch": 0, "val/ndcg@5": .1}]})
    row = dict(source, arm="sft", variant="sft", status="completed", seed=42,
               protocol="exp8_from_sft", run_dir=str(complete), source_result="results/completed.json")
    with metrics.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(export, "ROOT", tmp_path)
    monkeypatch.setattr(export, "EXPERIMENTS", (experiments,))
    monkeypatch.setattr(export, "METRICS", metrics)
    monkeypatch.setattr(export, "OUTPUT", output)
    return directory, identity, output, event


def sheet_rows(sheet):
    rows = iter(sheet.values)
    header = next(rows)
    return [dict(zip(header, row)) for row in rows]


def test_unfinished_events_export_without_becoming_final_results(inputs):
    directory, identity, output, event = inputs
    before = {p: p.read_bytes() for p in directory.iterdir()}
    export.main()
    workbook = load_workbook(output, read_only=True, data_only=True)
    try:
        final = sheet_rows(workbook["results"])
        assert len(final) == 1 and final[0]["status"] == "completed"
        history = sheet_rows(workbook["history"])
        assert len(history) == 1 and history[0]["status"] == "completed"
        events = sheet_rows(workbook["events"])
        assert len(events) == 1
        record = events[0]
        assert record["status"] == "no_final_result"
        assert record["model"] == "gpt2" and record["dataset"] == "beauty_2023"
        assert record["sampling"] == "curriculum_smooth_is" and record["num_samples"] == 3
        assert record["power"] == .25 and record["rho"] == .1
        assert record["source_result"] is None
        assert record["run_dir"] == str(directory.relative_to(export.ROOT))
        assert record["event.val/sft/ndcg@5"] == .17
    finally:
        workbook.close()
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("change", ["source", "config", "entry_dataset", "entry_seed"])
def test_mismatched_unfinished_identity_preserves_existing_workbook(inputs, change):
    directory, identity, output, event = inputs
    output.write_bytes(b"previous complete export")
    if change == "source":
        identity["from_sft_source"]["model"] = "tiger"
    elif change == "config":
        write_json(directory / "resolved_config.json", {"identity": {"wrong": True}})
    else:
        identity["entry"]["dataset" if change == "entry_dataset" else "seed"] = (
            "instruments" if change == "entry_dataset" else 43)
    if change != "config":
        write_json(directory / "identity.json", identity)
        write_json(directory / "resolved_config.json", {"identity": identity})
    with pytest.raises(ValueError, match="identity"):
        export.main()
    assert output.read_bytes() == b"previous complete export"


def test_malformed_event_is_not_silently_dropped(inputs):
    directory, identity, output, event = inputs
    event.write_text('{"type":"validation_end"}\n{"truncated":\n')
    with pytest.raises(json.JSONDecodeError):
        export.main()
    assert not output.exists()
