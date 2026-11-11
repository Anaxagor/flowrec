"""Saved CE data remain reusable after unrelated code/runtime updates."""
from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp6-ce-distillation"))
import ce_yambda_prepared_data as prepared
import ce_yambda_data_runtime as runtime
from flowrec.data.yambda import YambdaDataModule

SPEC = importlib.util.spec_from_file_location(
    "ce_prepared_compatibility_fixture", ROOT / "tests/test_ce_yambda_prepared_data.py")
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


@pytest.fixture
def compact(tmp_path):
    directory, dm, _ = helpers.fixture.__wrapped__(tmp_path)
    identity = {"version": 1, "model": "tiger", "dataset": "yambda_500m_listens",
        "data_config": {"max_history_len": 4, "train_target_policy": "last_n"},
        "subset": {"random_n": 50, "seed": 42, "pool_last_n": 128},
        "source_trial": None, "source_audit": {"catalog_sha256": "fixed-catalog"},
        "assets": {"events.parquet": {"bytes": 10, "mtime_ns": 100}},
        "code": {"scripts/gpt2_matched_finetune.py": "new-source"},
        "runtime": {"python": [3, 12], "torch": "2.7", "numpy": "2.0"}}
    prepared.prepare(directory, dm, [3, 1], identity, audit={"unchanged": True})
    return directory, dm, identity


def change_manifest(directory, change):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    return manifest


def set_path(record, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        record = record[key]
    record[keys[-1]] = value


@pytest.mark.parametrize("changed", ["serializer", "outer_runtime", "code", "input_runtime", "all"])
def test_existing_code_runtime_provenance_never_changes_saved_data(compact, monkeypatch, changed):
    directory, dm, identity = compact

    def old_provenance(manifest):
        saved = manifest["identity"]
        if changed in ("serializer", "all"):
            saved["serializer_sha256"] = "old-serializer-hash"
        if changed in ("outer_runtime", "all"):
            saved["runtime"] = {"python": [3, 11], "torch": "2.6", "numpy": "1.26"}
        if changed in ("code", "all"):
            saved["inputs"]["code"] = {"scripts/gpt2_matched_finetune.py": "old-source"}
        if changed in ("input_runtime", "all"):
            saved["inputs"]["runtime"] = {"python": [3, 11], "torch": "2.6", "numpy": "1.26"}

    change_manifest(directory, old_provenance)
    original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in directory.iterdir() if path.is_file()}
    expected_train = helpers._batch(dm, [3, 1], [0, 1])
    expected_val = next(iter(dm.val_dataloader()))
    expected_test = next(iter(dm.test_dataloader()))
    monkeypatch.setattr(YambdaDataModule, "setup", lambda *_a, **_k: pytest.fail("raw setup called"))
    with pytest.warns(RuntimeWarning, match="code/runtime"):
        loaded = prepared.load(directory, dm.cfg, identity)
    helpers.examples.assert_batches_equal(expected_train, helpers._batch(loaded, None, [0, 1]))
    helpers.examples.assert_batches_equal(expected_val, next(iter(loaded.val_dataloader())))
    helpers.examples.assert_batches_equal(expected_test, next(iter(loaded.test_dataloader())))
    assert loaded._ce_prepared_source_indices.tolist() == [3, 1]
    assert loaded._ce_prepared_audit == {"unchanged": True}
    assert original == {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("path,value", [
    ("model", "gpt2"), ("dataset", "yambda_50m_listens"), ("version", 2),
    ("data_config.max_history_len", 50), ("data_config.train_target_policy", "all"),
    ("subset.random_n", 15), ("subset.seed", 43), ("subset.pool_last_n", 64),
    ("source_trial", 4), ("source_audit.catalog_sha256", "changed-catalog"),
])
def test_semantic_mismatches_still_fail_with_exact_field(compact, path, value):
    directory, dm, identity = compact
    different = copy.deepcopy(identity)
    set_path(different, path, value)
    with pytest.raises(RuntimeError, match="CE prepared-data inputs changed") as exc:
        prepared.load(directory, dm.cfg, different)
    assert "inputs." + path in str(exc.value)


@pytest.mark.parametrize("field,value", [("bytes", 20), ("mtime_ns", 200)])
def test_raw_asset_changes_still_fail(compact, field, value):
    directory, dm, identity = compact
    different = copy.deepcopy(identity)
    different["assets"]["events.parquet"][field] = value
    with pytest.raises(RuntimeError, match="CE prepared-data inputs changed") as exc:
        prepared.load(directory, dm.cfg, different)
    assert "inputs.assets.events.parquet." + field in str(exc.value)


@pytest.mark.parametrize("value", [None, [], "bad", {"schema": 1},
                                     {"schema": 1, "inputs": None}, {"schema": 1, "inputs": []}])
def test_malformed_saved_identity_is_rejected(compact, value):
    directory, dm, identity = compact
    change_manifest(directory, lambda m: m.update(identity=value))
    with pytest.raises(RuntimeError, match="Invalid CE prepared-data identity"):
        prepared.load(directory, dm.cfg, identity)


def test_payload_schema_remains_a_hard_compatibility_boundary(compact):
    directory, dm, identity = compact
    change_manifest(directory, lambda m: m["identity"].update(schema=999))
    with pytest.raises(RuntimeError, match="CE prepared-data inputs changed") as exc:
        prepared.load(directory, dm.cfg, identity)
    assert "schema" in str(exc.value)


def test_uncommitted_manifest_is_rejected(compact):
    directory, dm, identity = compact
    change_manifest(directory, lambda m: m.update(status="building"))
    with pytest.raises(RuntimeError, match="Incomplete CE prepared-data manifest"):
        prepared.load(directory, dm.cfg, identity)


@pytest.mark.parametrize("damage", ["tensor_size", "metadata_checksum"])
def test_ignored_code_change_cannot_bypass_artifact_integrity(compact, monkeypatch, damage):
    directory, dm, identity = compact
    manifest = change_manifest(directory, lambda m: m["identity"].update(serializer_sha256="old"))
    if damage == "tensor_size":
        artifact = directory / manifest["artifacts"]["tensors"]["file"]
        artifact.write_bytes(b"truncated")
        expected_error = "artifact changed"
    else:
        artifact = directory / manifest["artifacts"]["metadata"]["file"]
        before = artifact.stat()
        with artifact.open("r+b") as stream:
            byte = stream.read(1)
            stream.seek(0)
            stream.write(bytes([byte[0] ^ 1]))
        os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
        monkeypatch.setattr(prepared.pickle, "load", lambda *_a: pytest.fail("damaged metadata read"))
        expected_error = "checksum mismatch"
    with pytest.raises(RuntimeError, match=expected_error):
        prepared.load(directory, dm.cfg, identity)


def test_cpu_preparation_reuses_old_code_cache_without_rebuilding(compact, monkeypatch, capsys):
    directory, dm, identity = compact
    change_manifest(directory, lambda m: m["identity"].update(serializer_sha256="previous-version"))
    previous = (directory / "manifest.json").read_bytes()
    args = SimpleNamespace()
    base = SimpleNamespace(control=SimpleNamespace(file_lock=lambda *_a: nullcontext()),
        tiger_runner=SimpleNamespace(build_data_config=lambda *_a, **_k: dm.cfg,
            build_data_module=lambda *_a, **_k: pytest.fail("raw data rebuild called"),
            seed_everything=lambda *_a: pytest.fail("preparation was restarted")))
    monkeypatch.setattr(runtime, "directory", lambda *_a: directory)
    monkeypatch.setattr(runtime, "identity", lambda *_a: identity)
    with pytest.warns(RuntimeWarning, match="code/runtime"):
        runtime.prepare_once(args, "tiger", base, SimpleNamespace(), {}, directory / "source.json")
    assert "already ready" in capsys.readouterr().out
    assert (directory / "manifest.json").read_bytes() == previous
