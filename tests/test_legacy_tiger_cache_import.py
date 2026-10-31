"""Cache import cannot rebuild rewards or mutate legacy experiment payloads."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts/exp8-from-sft"
sys.path.insert(0, str(HERE))
import finetune_from_sft as runner
import legacy_tiger_cache_import as importer
import from_sft_runtime as runtime


@pytest.fixture
def session(tmp_path, monkeypatch):
    import legacy_tiger_cache_validation as validation
    origin = tmp_path / "legacy"
    output = tmp_path / "new"
    inputs = {"source": "weights42", "data": "full"}
    for key in runtime.COMPONENTS:
        directory = origin / key
        runtime.write_json(directory / "manifest.json", {"num_examples": 3, "component": key})
        (directory / "factors.pt").write_bytes(b"fixed teacher factors " + key.encode())
    session = SimpleNamespace(
        source={"source_id": "tiger_beauty", "model": "tiger", "dataset": "beauty",
                "expected_counts": {"num_finetune_examples": 3}},
        args=SimpleNamespace(smoke=False, device="cuda"),
        config={"data": {"source": "amazon"}}, cache_root=output,
        config_path=output / "config.json", indices=lambda _dm: [0, 1, 2],
        cache_dirs=lambda: {key: output / "reward_factors" / key for key in runtime.COMPONENTS},
        base=SimpleNamespace(control=SimpleNamespace(file_lock=lambda _path: nullcontext()),
                             cache_config=lambda *_args: {"component_config": "exact"}),
    )
    monkeypatch.setattr(importer, "legacy_cache_root", lambda _s: origin)
    monkeypatch.setattr(runtime, "input_identity", lambda _s: deepcopy(inputs))
    monkeypatch.setattr(validation, "validate_component", lambda _s, _dm, _key, directory:
                        {**runtime.read_json(directory / "manifest.json"), "legacy_validation": {"rows": 3}})

    def check_factor(s, key, count):
        directory = s.cache_dirs()[key]
        assert count == 3
        assert runtime.read_json(directory / "exp8_input_identity.json") == inputs
        assert (directory / "factors.pt").is_file()
    monkeypatch.setattr(runtime, "check_factor", check_factor)
    monkeypatch.setattr(runtime, "build_factor", lambda *_a: pytest.fail("must never rebuild factors"))
    session.inputs = inputs
    session.origin = origin
    return session


def test_import_private_copy_with_provenance_leaves_old_files_unchanged(session):
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in session.origin.rglob("*") if p.is_file()}
    importer._import_component(session, None, "sasrec", session.inputs)
    destination = session.cache_dirs()["sasrec"]
    record = runtime.read_json(destination / "manifest.json")
    assert record["imported_from"]["historical_asset_hashes_available"] is False
    assert record["legacy_validation"] == {"rows": 3}
    assert record["imported_from"]["factor_sha256"] == importer._digest(destination / "factors.pt")
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in session.origin.rglob("*") if p.is_file()}
    assert before == after
    (session.origin / "sasrec/factors.pt").write_bytes(b"later overwrite")
    assert (destination / "factors.pt").read_bytes() == b"fixed teacher factors sasrec"


def test_repeat_import_reuses_completed_destination(session, monkeypatch):
    import legacy_tiger_cache_validation as validation
    importer._import_component(session, None, "sasrec", session.inputs)
    monkeypatch.setattr(validation, "validate_component", lambda *_a: pytest.fail("revalidated old cache"))
    importer._import_component(session, None, "sasrec", session.inputs)


@pytest.mark.parametrize("mutation", ["payload", "manifest", "input"])
def test_changed_source_during_validation_never_commits(session, monkeypatch, mutation):
    import legacy_tiger_cache_validation as validation
    def validate(_s, _dm, _key, directory):
        manifest = runtime.read_json(directory / "manifest.json")
        if mutation == "payload":
            (directory / "factors.pt").write_bytes(b"overwritten by other process")
        elif mutation == "manifest":
            runtime.write_json(directory / "manifest.json", {"changed": True})
        else:
            session.inputs["data"] = "replaced"
        return manifest
    monkeypatch.setattr(validation, "validate_component", validate)
    with pytest.raises(RuntimeError, match="changed"):
        importer._import_component(session, None, "sasrec", deepcopy(session.inputs))
    assert not session.cache_dirs()["sasrec"].exists()
    assert not list((session.cache_root / "reward_factors").glob(".sasrec.import-*"))


def test_incompatible_cache_or_missing_payload_never_rebuilds(session, monkeypatch):
    import legacy_tiger_cache_validation as validation
    def reject(*_args):
        raise RuntimeError("user/target alignment differs")
    monkeypatch.setattr(validation, "validate_component", reject)
    with pytest.raises(RuntimeError, match="alignment"):
        importer._import_component(session, None, "sasrec", session.inputs)
    (session.origin / "sasrec/factors.pt").unlink()
    with pytest.raises(FileNotFoundError, match="never rebuilds"):
        importer._import_component(session, None, "sasrec", session.inputs)
    assert not session.cache_dirs()["sasrec"].exists()


def test_gpu_missing_import_refuses_before_normal_preparer(session, monkeypatch):
    monkeypatch.setattr(runtime, "require_prepared", lambda *_a, **_k: None)
    monkeypatch.setattr(runtime, "prepare", lambda *_a: pytest.fail("entered normal preparer"))
    with pytest.raises(FileNotFoundError):
        importer.prepare_reusing_legacy(session, "gpu")


def test_gpu_validates_all_factors_before_baseline_phase(session, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "require_prepared", lambda *_a, **_k: calls.append("cpu-ready"))
    monkeypatch.setattr(runtime, "check_factor", lambda _s, name, _n: calls.append(name))
    monkeypatch.setattr(runtime, "prepare", lambda _s, kind: calls.append(kind))
    importer.prepare_reusing_legacy(session, "gpu")
    assert calls == ["cpu-ready", *runtime.COMPONENTS, "gpu"]


def test_cpu_imports_all_components_and_saves_snapshot_without_gpu(session, monkeypatch):
    import prepared_data
    calls = []
    dm = object()
    session.setup_data = lambda **_kw: dm
    session.audit_data = lambda _dm: {"counts": "matched"}
    monkeypatch.setattr(runtime, "compare_tiger_source", lambda *_a: {"comparison": "all rows"})
    def import_component(s, _dm, key, _identity):
        assert s.args.device == "cpu"
        calls.append(key)
    monkeypatch.setattr(importer, "_import_component", import_component)
    monkeypatch.setattr(prepared_data, "save_prepared_data", lambda *_a: calls.append("snapshot"))
    importer.prepare_reusing_legacy(session, "cpu")
    assert calls == [*runtime.COMPONENTS, "snapshot"]
    assert session.args.device == "cuda"
    assert runtime.read_json(session.cache_root / "prepared_cpu.json")["identity"] == session.inputs
    assert not (session.cache_root / "prepared_gpu.json").exists()


def test_reuse_can_complete_an_earlier_normal_cpu_preparation(session, monkeypatch):
    import prepared_data
    for key in ("interaction", "similarity"):
        importer._import_component(session, None, key, session.inputs)
    runtime.write_json(session.cache_root / "prepared_cpu.json",
                       {"identity": session.inputs, "status": "completed", "teacher_translation": None})
    session.setup_data = lambda **_kw: object()
    session.audit_data = lambda _dm: {"counts": "matched"}
    monkeypatch.setattr(runtime, "compare_tiger_source", lambda *_a: {"comparison": "all rows"})
    monkeypatch.setattr(prepared_data, "snapshot_ready", lambda *_a: True)
    monkeypatch.setattr(prepared_data, "save_prepared_data", lambda *_a: None)
    importer.prepare_reusing_legacy(session, "cpu")
    assert (session.cache_dirs()["sasrec"] / "factors.pt").is_file()
    session.setup_data = lambda **_kw: pytest.fail("ready snapshot was rebuilt")
    importer.prepare_reusing_legacy(session, "cpu")


def test_reuse_flag_keeps_same_98_training_cells():
    flags = ["--count", "--source-ids", "tiger_beauty", "tiger_instruments", "--arms", "sft", "ce"]
    grid = runner.build_grid(runner.parse_args(flags))
    assert len(grid) == 98
    assert runner.build_grid(runner.parse_args([*flags, "--reuse-tiger-gflow-caches"])) == grid


@pytest.mark.parametrize("flags", [["--source-ids", "gpt2_beauty"],
                                  ["--source-ids", "tiger_yambda50_matched"],
                                  ["--source-ids", "tiger_beauty", "--smoke"]])
def test_reuse_cli_rejects_unsupported_sources_and_smoke(flags):
    with pytest.raises(SystemExit):
        runner.parse_args(["--count", "--reuse-tiger-gflow-caches", *flags])
