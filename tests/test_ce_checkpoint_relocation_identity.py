"""Resume path migration retains all scientific and content identity guards."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ce_checkpoint_identity", ROOT / "scripts/exp7-ce-distillation/checkpoint_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
PROTOCOL = ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def relocated(tmp_path):
    folder = tmp_path / "checkpoints/exp2/loo_split/initial/tiger/beauty"
    folder.mkdir(parents=True)
    checkpoint, sidecar = folder / "best.pt", folder / "best.pt.json"
    checkpoint.write_bytes(b"selected weights")
    sidecar.write_text('{"model": "tiger"}')
    historical = "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt"
    origin = "artifacts/checkpoint_archive/2026-10-30/exp2-tuned_params/initial/tiger/beauty/best.pt"
    record = {"source_checkpoint": origin, "sha256": sha(checkpoint),
              "metadata": {"best.pt.json": origin + ".json"},
              "metadata_sha256": {"best.pt.json": sha(sidecar)}}
    (folder / "provenance.json").write_text(json.dumps(record))
    old = {"checkpoint": "/cluster/flowrec/" + historical,
           "config": {"optimizer": {"lr": 3e-5}, "dataset": {"train_last_n": 128}},
           "assets": {"/cluster/flowrec/" + historical: {"bytes": 16, "mtime_ns": 42},
                      "/cluster/flowrec/" + historical + ".json": {"bytes": 18, "mtime_ns": 43}}}
    new = copy.deepcopy(old)
    new["checkpoint"] = str(checkpoint)
    new["assets"] = {str(checkpoint): old["assets"][old["checkpoint"]],
                     str(sidecar): old["assets"][old["checkpoint"] + ".json"]}
    return tmp_path, old, new, checkpoint, sidecar


def matches(root, old, new):
    return identity.same_after_checkpoint_relocation(old, new, root=root, protocol_path=PROTOCOL)


def test_relocation_uses_only_current_content_and_provenance(relocated, monkeypatch):
    root, old, new, _, _ = relocated
    original = Path.open
    def only_current(path, *args, **kwargs):
        assert "checkpoint_archive" not in str(path)
        assert "exp2-tuned_params" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", only_current)
    assert matches(root, old, new)


@pytest.mark.parametrize("asset", ["checkpoint", "sidecar"])
def test_current_asset_bytes_must_match_provenance(relocated, asset):
    root, old, new, checkpoint, sidecar = relocated
    path = checkpoint if asset == "checkpoint" else sidecar
    content = path.read_bytes()
    path.write_bytes(b"x" * len(content))
    assert not matches(root, old, new)


@pytest.mark.parametrize("change", ["lr", "subset", "asset_stats", "source"])
def test_path_migration_does_not_relax_other_identity_fields(relocated, change):
    root, old, new, checkpoint, _ = relocated
    if change == "lr":
        new["config"]["optimizer"]["lr"] = 1e-3
    elif change == "subset":
        new["config"]["dataset"]["train_last_n"] = 15
    elif change == "asset_stats":
        new["assets"] = copy.deepcopy(new["assets"])
        new["assets"][str(checkpoint)]["mtime_ns"] += 1
    else:
        old["checkpoint"] = "/cluster/flowrec/checkpoints/exp2-tuned_params/initial/tiger/instruments/best.pt"
    assert not matches(root, old, new)


def test_only_exact_pre_migration_protocol_hash_is_compatible(tmp_path):
    current_hash = sha(PROTOCOL)
    assert identity._protocol_path_only_change(identity.PREVIOUS_PROTOCOL_SHA256, current_hash, PROTOCOL)
    assert not identity._protocol_path_only_change("0" * 64, current_hash, PROTOCOL)
    changed = json.loads(PROTOCOL.read_text())
    changed["scientific_setting"] = 123
    altered = tmp_path / "protocol.json"
    altered.write_text(json.dumps(changed, indent=2) + "\n")
    assert not identity._protocol_path_only_change(identity.PREVIOUS_PROTOCOL_SHA256, sha(altered), altered)


def test_legacy_full_history_path_cannot_alias_last128(tmp_path):
    old = "checkpoints/exp2-tuned_params/initial/gpt2/yambda_50m_listens/best_model.pt"
    new = "checkpoints/exp2/loo_split/initial/gpt2/yambda_50m_listens/last_128/best.pt"
    assert identity._flat_initial_name(old, tmp_path) != identity._flat_initial_name(new, tmp_path)
