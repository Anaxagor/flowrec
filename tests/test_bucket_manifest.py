"""Guard the workbook selection and prevent accidental protocol substitution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/exp5_result_analysis"))
from bucket_manifest import build_manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_missing_weights_keep_all_selected_runs_and_beauty_alias(tmp_path):
    specs = build_manifest(tmp_path)
    assert len(specs) == 19
    assert sum(s["stage"] == "init" for s in specs) == 4
    assert sum(s["stage"] == "gflow" for s in specs) == 11
    assert not any(s["available"] for s in specs)
    assert all(not Path(s["checkpoint"]).is_absolute() for s in specs)
    assert len({s["run_id"] for s in specs}) == len(specs)
    assert len({s["checkpoint"] for s in specs}) == len(specs)
    selected = next(s for s in specs if s["dataset"] == "beauty" and "best_ndcg" in s["selection_labels"])
    assert selected["selection_labels"] == ["best_ndcg", "pareto"]
    assert selected["selection_cells"] == ["B28", "B30"]
    assert selected["recorded_test_metrics"]["NDCG@10"] == pytest.approx(0.0360486376228515)
    assert selected["source_workbook_sha256"] == "db1760c3338407950af599e5351931d86fd1fc59f50e62d2d83c025afc25058e"
    by_id = {s["run_id"]: s for s in specs}
    for s in specs:
        assert by_id[s["baseline_run_id"]]["dataset"] == s["dataset"]
        if s["stage"] == "gflow":
            assert by_id[s["sft_run_id"]]["stage"] == "sft"
            assert by_id[s["sft_run_id"]]["seed"] == 42


def test_seed_aliases_do_not_duplicate_primary_models(tmp_path):
    specs = build_manifest(tmp_path, seeds=(42, 43, 44))
    assert len(specs) == 19
    assert {s["seed"] for s in specs if s["stage"] == "gflow"} == {42}
    assert {s["seed"] for s in specs if s["stage"] == "init"} == {None}
    assert sum(s["stage"] == "sft" for s in specs) == 4
    assert build_manifest(tmp_path, seeds=(43,)) == specs


def test_does_not_substitute_legacy_yambda_controls(tmp_path):
    legacy = tmp_path / "experiments/exp2-sft-finetune-yambda/yambda_50m_listens/sft_lr3e-05/seed_42/checkpoints/best.pt"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    selected = next(s for s in build_manifest(tmp_path) if s["dataset"] == "yambda_50m_listens" and s["stage"] == "sft")
    assert not selected["available"]
    assert selected["checkpoint"] == "checkpoints/exp2/loo_split/sft-finetuned/tiger/yambda_50m_listens/best.pt"




def test_rejects_wrong_initial_checkpoint_evidence(tmp_path):
    spec = next(s for s in build_manifest(tmp_path) if s["stage"] == "gflow")
    write_json(tmp_path / spec["resolved_config"], {"init_checkpoint": "checkpoints/wrong.pt"})
    with pytest.raises(ValueError, match="unexpected initial"):
        build_manifest(tmp_path)


def test_rejects_last8_in_matched_yambda_run(tmp_path):
    spec = next(s for s in build_manifest(tmp_path) if s["dataset"] == "yambda_50m_listens" and s["stage"] == "gflow")
    write_json(tmp_path / spec["resolved_config"], {"data": {"datamodule_train_last_n": 8}})
    with pytest.raises(ValueError, match="expected matched"):
        build_manifest(tmp_path)


def test_csv_matching_uses_run_identity_not_historical_row_number(tmp_path):
    spec = next(s for s in build_manifest(tmp_path) if s["dataset"] == "instruments" and s["stage"] == "gflow")
    csv_path = tmp_path / spec["source_metrics_csv"]
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("dataset,variant,split,run_dir,NDCG@10,combo,power,objective,target_grad_ratio,num_negatives\n"
                        f"instruments,gflow,test,/home/other/flowrec/{spec['run_dir']},0.125,interaction_x_sim_hier,1,db,1,10\n")
    specs = build_manifest(tmp_path)
    selected = next(s for s in specs if s["run_id"] == spec["run_id"])
    assert selected["recorded_test_metrics"] == {"NDCG@10": 0.125}
    csv_path.write_text(csv_path.read_text().replace(",1,db,1,10", ",4,db,1,10"))
    with pytest.raises(ValueError, match="selected power"):
        build_manifest(tmp_path)


def test_primary_sft_is_represented_once_with_its_actual_seed(tmp_path):
    directory = tmp_path / "checkpoints/exp2/loo_split/sft-finetuned/tiger/yambda_50m_listens"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"selected weights")
    write_json(directory / "provenance.json", {
        "model": "tiger", "dataset": "yambda_50m_listens", "seed": 44,
        "source_checkpoint": "experiments/selected-primary/checkpoints/best.pt",
    })
    specs = build_manifest(tmp_path, (42, 43, 44))
    selected = [s for s in specs if s["dataset"] == "yambda_50m_listens" and s["stage"] == "sft"]
    assert len(selected) == 1
    assert selected[0]["seed"] == 44
    assert selected[0]["checkpoint"] == str((directory / "best.pt").relative_to(tmp_path))
    assert selected[0]["available"]
    assert all(s["sft_run_id"] == selected[0]["run_id"]
               for s in specs if s["dataset"] == "yambda_50m_listens" and s["stage"] == "gflow")


def test_initial_evidence_accepts_same_base_in_another_checkout(tmp_path):
    spec = next(s for s in build_manifest(tmp_path) if s["stage"] == "gflow")
    write_json(tmp_path / spec["resolved_config"], {
        "init_checkpoint": "/cluster/flowrec/" + spec["init_checkpoint"],
        "cli_args": {"prefix_from_dense": True},
    })
    assert next(s for s in build_manifest(tmp_path) if s["run_id"] == spec["run_id"])["evidence_available"]


def test_unverifiable_initial_evidence_fails_explicitly(tmp_path):
    spec = next(s for s in build_manifest(tmp_path) if s["stage"] == "gflow")
    write_json(tmp_path / spec["resolved_config"], {
        "init_checkpoint": "checkpoints/obsolete/gpt2/beauty/best.pt",
        "cli_args": {"prefix_from_dense": True},
    })
    with pytest.raises(ValueError, match="result cannot be verified"):
        build_manifest(tmp_path)
