"""Preparation and restart guards without loading models or dataset tensors."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts/exp8-from-sft"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp8_runtime_guards_test", HERE / "from_sft_runtime.py")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)
import ce_from_sft as ce  # noqa: E402
import prepared_data  # noqa: E402


@pytest.fixture
def session(tmp_path, monkeypatch):
    args = SimpleNamespace(
        lr=3e-5, weight_decay=.01, batch_size=32, grad_accum=64,
        max_epochs=500, patience=5, min_epochs=10, validate_every=1,
        validate_at_start=True, max_grad_norm=.5, amp_dtype="float16",
        val_sample_fraction=None, val_sample_seed=0, val_sample_resample=False,
        monitor="val/sft/ndcg@5", val_batch_size=64, test_batch_size=64,
        eval_num_beams=20, test_ks=[5,10,20], max_train_batches=None,
        max_eval_batches=None, smoke=False, curriculum_epochs=10., curriculum_steps=None,
        sampler_temperature=1., sasrec_transform="sigmoid", sasrec_temperature=1.,
        max_candidate_batch=128, gflow_weight_mode="fixed_grad_ratio",
        gflow_calibration_batches=5, gflow_calibration_seed=1234, gflow_calibration_eps=1e-12,
        seed=42, device="cuda", num_workers=2, checkpoint_every_updates=100,
        cache_batch_size=256, cache_verify_batches=1, resource_profile="small",
        models=["tiger","gpt2"], datasets=["beauty","instruments"],
        arms=["sft","gflow","ce"], progress_bar=True,
    )
    bound = SimpleNamespace(
        args=args, output_root=tmp_path/"output", cache_root=tmp_path/"cache",
        baseline_path=tmp_path/"output/baseline/init_metrics.json",
        source_identity={"source_id":"tiger_beauty", "source_seed":42,
                         "source_fingerprint":"source-v1", "checkpoint_sha256":"weights-v1"},
        source={"source_id":"tiger_beauty", "source_seed":42, "dataset":"beauty", "model":"tiger"},
        config={"data":{"reviews_path":"reviews.parquet"}},
        inputs_version="data-v1", ce_identity={"checkpoint":"weights-v1", "eval_beams":20},
    )
    monkeypatch.setattr(runtime,"input_identity",lambda s: {
        "source":deepcopy(s.source_identity), "data":s.inputs_version})
    monkeypatch.setattr(ce,"baseline_identity",lambda s,*_a: deepcopy(s.ce_identity))
    # Snapshot serialization has separate tests; isolate marker/baseline
    # guards from real datamodules and installed ML library versions here.
    monkeypatch.setattr(prepared_data,"snapshot_ready",lambda _session: True)
    return bound


def cell():
    return {"source_id":"tiger_beauty", "model":"tiger", "dataset":"beauty", "source_seed":42,
            "seed":42, "arm":"gflow", "combo":["sasrec"], "power":1., "rho":.1,
            "objective":"tb", "num_negatives":3}


def markers(session):
    for kind in ("cpu","gpu"):
        runtime.write_json(session.cache_root/f"prepared_{kind}.json", {
            "identity":runtime.input_identity(session), "status":"completed"})


@pytest.fixture
def yambda_session(session, monkeypatch):
    session.source.update(model="gpt2", dataset="yambda_50m_listens",
                          source_id="gpt2_yambda50_trial36")
    session.source_identity["source_id"] = session.source["source_id"]
    path = session.cache_root/session.source["dataset"]/"source_assets/teacher_id_translation.json"
    runtime.write_json(path, {"item_id_translation": [0, 2, 1]})
    expected = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    markers(session)
    for kind in ("cpu", "gpu"):
        marker = session.cache_root/f"prepared_{kind}.json"
        record = runtime.read_json(marker)
        runtime.write_json(marker, {**record, "teacher_translation": expected})
    # Other tests exercise baseline compatibility; focus on the catalog mapping here.
    monkeypatch.setattr(runtime, "baseline_ready", lambda _session: True)
    return session, path


def baselines(session):
    splits={"validation":{"NDCG@5":.1}, "test":{"NDCG@5":.2}}
    runtime.write_json(session.baseline_path,{"from_sft_source":session.source_identity, **splits})
    path=ce_path(session)
    runtime.write_json(path,{"identity":session.ce_identity, "splits":splits})


def ce_path(session):
    return session.output_root/"ce"/session.source["dataset"]/"baseline"/(
        "smoke.json" if session.args.smoke else "init_metrics.json")


def test_restart_identity_ignores_execution_only_changes(session):
    before=runtime.training_identity(session,cell())
    session.args.num_workers=0
    session.args.device="cpu"
    session.args.resource_profile="yambda"
    session.args.models=["tiger"]
    session.args.datasets=["beauty"]
    session.args.arms=["gflow"]
    session.args.progress_bar=False
    session.args.checkpoint_every_updates=1
    session.args.cache_batch_size=64
    session.args.cache_verify_batches=2
    session.args.output_root=Path("/another/mount/output")
    session.args.cache_root=Path("/another/mount/cache")
    assert runtime.training_identity(session,cell())==before


@pytest.mark.parametrize("name,value", [("lr",1e-5),("batch_size",16),("grad_accum",128),
                                       ("max_epochs",200),("validate_at_start",False)])
def test_restart_identity_detects_changed_optimization(session,name,value):
    before=runtime.training_identity(session,cell())
    setattr(session.args,name,value)
    assert runtime.training_identity(session,cell())!=before


def test_restart_identity_detects_seed_cell_source_and_input_changes(session):
    selected=cell()
    before=runtime.training_identity(session,selected)
    assert runtime.training_identity(session,{**selected,"seed":43})!=before
    assert runtime.training_identity(session,{**selected,"source_seed":43})!=before
    assert runtime.training_identity(session,{**selected,"rho":1.})!=before
    session.source_identity["checkpoint_sha256"]="replaced-weights"
    assert runtime.training_identity(session,selected)!=before
    session.source_identity["checkpoint_sha256"]="weights-v1"
    session.inputs_version="data-v2"
    assert runtime.training_identity(session,selected)!=before


def test_complete_preparation_requires_matching_both_baselines(session):
    markers(session); baselines(session)
    assert runtime.baseline_ready(session)
    runtime.require_prepared(session)


@pytest.mark.parametrize("kind",["cpu","gpu"])
@pytest.mark.parametrize("damage",["missing","no_status","incomplete","wrong_source"])
def test_preparation_rejects_missing_incomplete_or_changed_marker(session,kind,damage):
    markers(session); baselines(session)
    path=session.cache_root/f"prepared_{kind}.json"
    record=runtime.read_json(path)
    if damage=="missing": path.unlink()
    else:
        if damage=="no_status": record.pop("status")
        elif damage=="incomplete": record["status"]="running"
        else: record["identity"]["source"]["checkpoint_sha256"]="other"
        runtime.write_json(path,record)
    with pytest.raises(RuntimeError,match=f"{kind} preparation"):
        runtime.require_prepared(session)


def test_changed_live_source_cannot_reuse_old_preparation(session):
    markers(session); baselines(session)
    session.source_identity["source_fingerprint"]="source-v2"
    with pytest.raises(RuntimeError,match="preparation"):
        runtime.require_prepared(session)


def test_cpu_stage_does_not_require_gpu_marker_or_baseline(session):
    runtime.write_json(session.cache_root/"prepared_cpu.json", {
        "identity":runtime.input_identity(session), "status":"completed"})
    runtime.require_prepared(session,kinds=("cpu",))


@pytest.mark.parametrize("kinds", [("cpu",), ("cpu","gpu")])
def test_missing_prepared_snapshot_blocks_ready_markers(session,monkeypatch,kinds):
    markers(session); baselines(session)
    monkeypatch.setattr(prepared_data,"snapshot_ready",lambda _session: False)
    with pytest.raises(RuntimeError,match="Missing prepared data snapshot"):
        runtime.require_prepared(session,kinds=kinds)


def test_matching_teacher_catalog_mapping_allows_both_prepared_stages(yambda_session):
    session, _path = yambda_session
    runtime.require_prepared(session)


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
@pytest.mark.parametrize("damage", ["changed", "deleted"])
def test_changed_or_deleted_teacher_catalog_mapping_blocks_prepared_stage(yambda_session,kind,damage):
    session, path = yambda_session
    if damage == "changed":
        runtime.write_json(path, {"item_id_translation": [0, 1, 2]})
    else:
        path.unlink()
    with pytest.raises(RuntimeError, match="SASRec ID translation"):
        runtime.require_prepared(session,kinds=(kind,))


@pytest.mark.parametrize("model,dataset", [("gpt2", "beauty"), ("tiger", "yambda_500m_listens")])
def test_other_sources_need_no_teacher_catalog_mapping(session,model,dataset):
    session.source.update(model=model,dataset=dataset)
    markers(session)
    assert runtime.teacher_translation_identity(session) is None
    runtime.require_prepared(session,kinds=("cpu",))
    assert not list(session.cache_root.rglob("teacher_id_translation.json"))


@pytest.mark.parametrize("damage",[
    "common_missing","ce_missing","common_source","common_empty_split",
    "common_missing_split","ce_identity","ce_missing_split","ce_different_split",
])
def test_incomplete_or_mismatched_baselines_block_training(session,damage):
    markers(session); baselines(session)
    common=runtime.read_json(session.baseline_path)
    ce_record=runtime.read_json(ce_path(session))
    if damage=="common_missing": session.baseline_path.unlink()
    elif damage=="ce_missing": ce_path(session).unlink()
    elif damage.startswith("common_"):
        if damage=="common_source": common["from_sft_source"]["checkpoint_sha256"]="other"
        elif damage=="common_empty_split": common["test"]={}
        else: common.pop("validation")
        runtime.write_json(session.baseline_path,common)
    else:
        if damage=="ce_identity": ce_record["identity"]["checkpoint"]="other"
        elif damage=="ce_missing_split": ce_record["splits"].pop("test")
        else: ce_record["splits"]["validation"]["NDCG@5"]=-1.
        runtime.write_json(ce_path(session),ce_record)
    assert not runtime.baseline_ready(session)
    with pytest.raises(RuntimeError,match="baseline"):
        runtime.require_prepared(session)


def test_smoke_baseline_uses_its_own_ce_file(session):
    session.args.smoke=True
    markers(session); baselines(session)
    assert runtime.baseline_ready(session)
    ce_path(session).rename(ce_path(session).with_name("init_metrics.json"))
    assert not runtime.baseline_ready(session)
