"""Native legacy baselines survive summary-CSV cleanup without double counting."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_collect_from_sft_metrics", ROOT / "scripts/exp7-from-sft/collect_metrics.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    legacy = tmp_path / "experiments/exp4-gflow-tiger-from-sft"
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(collector, "LEGACY_RESULTS", legacy)
    path = legacy / "beauty/sft_only_lr3e-05/sft_seed_42/baseline/init_metrics.json"
    path.parent.mkdir(parents=True)
    data = {
        "dataset": "beauty",
        "checkpoint": "/home/user/flowrec/checkpoints/exp2-tuned_params/finetuned/tiger/beauty/sft_only_lr3e-05/seed_42/best.pt",
        "validation": {"NDCG@5": .04, "num_examples": 22363.0},
        "test": {"NDCG@5": .027, "num_examples": 22363.0},
    }
    path.write_text(json.dumps(data))
    return path, data


def csv_baseline():
    return {"protocol": "legacy_tiger_gflow", "arm": "init", "variant": "init",
            "dataset": "beauty", "model": "tiger", "source_id": "tiger_beauty",
            "source_seed": "42", "source_result": "artifacts/legacy.csv",
            "available_splits": "test,validation", "completeness": "complete_csv",
            "test.NDCG@5": "0.027000", "csv_only_field": "preserved"}


def test_native_baseline_collected_without_csv(baseline):
    path, _ = baseline
    before = path.read_bytes()
    rows = collector.merge_legacy_baseline_json([])
    assert len(rows) == 1
    assert rows[0]["protocol"] == "legacy_tiger_gflow"
    assert rows[0]["arm"] == "init" and rows[0]["status"] == "completed"
    assert rows[0]["source_seed"] == 42
    assert rows[0]["source_result"] == str(path.relative_to(collector.ROOT))
    assert rows[0]["validation.NDCG@5"] == .04
    assert rows[0]["test.NDCG@5"] == .027
    assert rows[0]["available_splits"] == "test,validation"
    assert path.read_bytes() == before


def test_matching_csv_baseline_is_merged_with_provenance(baseline):
    path, _ = baseline
    rows = collector.merge_legacy_baseline_json([csv_baseline()])
    assert len(rows) == 1
    assert rows[0]["csv_only_field"] == "preserved"
    assert rows[0]["supplemental_source_result"] == "artifacts/legacy.csv"
    assert rows[0]["source_result"] == str(path.relative_to(collector.ROOT))
    assert rows[0]["completeness"] == "complete_json"


def test_conflicting_csv_metrics_are_not_overwritten(baseline):
    row = csv_baseline()
    row["test.NDCG@5"] = ".5"
    with pytest.raises(ValueError, match="disagree on test.NDCG@5"):
        collector.merge_legacy_baseline_json([row])


@pytest.mark.parametrize("change", ["dataset", "checkpoint_seed", "source_seed"])
def test_baseline_identity_must_agree_with_path(baseline, change):
    path, data = baseline
    if change == "dataset":
        data["dataset"] = "instruments"
    elif change == "checkpoint_seed":
        data["checkpoint"] = data["checkpoint"].replace("seed_42", "seed_43")
    else:
        data["source_seed"] = 43
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity disagrees"):
        collector.merge_legacy_baseline_json([])
