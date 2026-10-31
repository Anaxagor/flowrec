"""Legacy reuse checks actual factor payloads, row identity and reward assets."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp8-from-sft"))
import legacy_tiger_cache_validation as validation
from flowrec.models.sasrec import SASRec, SASRecConfig


def write(path, value):
    path.write_text(json.dumps(value, default=str))


@pytest.fixture(params=["sasrec", "interaction", "similarity"])
def legacy(tmp_path, request):
    name = request.param
    raw = tmp_path / "reviews.parquet"
    reviews = []
    for user in range(2):
        for position in range(160):
            row = {"user_id": f"u{user}", "asin": f"i{position % 5}",
                   "timestamp": 1000 + 2 * position, "rating": 5., "verified_purchase": True}
            reviews.append(row)
            if position % 7 == 0:
                reviews.append({**row, "timestamp": row["timestamp"] - 1,
                                "rating": 1., "verified_purchase": False})
    pd.DataFrame(reviews).to_parquet(raw)
    semantic = tmp_path / "semantic.json"
    write(semantic, {f"i{i}": [i, (i+1) % 5] for i in range(5)})
    rqvae = tmp_path / "rqvae.pt"
    torch.manual_seed(5)
    torch.save({"quantizer.codebooks": torch.randn(2, 8, 4)}, rqvae)
    config = {"seed": 42, "data": {"reviews_path": str(raw), "semantic_id_path": str(semantic)},
        "dataset": {"positive_criterion": "verified_purchase == True", "min_user_interactions": 3,
                    "max_history_len": 5, "filter_reviews_to_metadata": False},
        "dataloader": {"batch_size": 7, "train_batch_size": 7, "validation_batch_size": 7,
                       "num_workers": 0, "persistent_workers": False, "include_raw_in_batch": False},
        "model": {"use_user_token": False}}
    config_path = tmp_path / "session_config.json"
    write(config_path, config)
    runner = validation.factors.tiger_runner
    cfg = runner.build_data_config(config, config_path=config_path)
    dm = runner.build_data_module(cfg)
    dm.setup()
    teacher_path = tmp_path / "sasrec.pt"
    teacher = SASRec(SASRecConfig(item_vocab_size=dm.item_vocab_size, max_history_len=5,
                                d_model=4, nhead=1, num_layers=1, dropout=0., pos_drop=0.))
    torch.save({"model_state_dict": teacher.state_dict(), "model_config": asdict(teacher.cfg)}, teacher_path)
    rewards = {"sasrec": {"transform": "sigmoid", "temperature": 1.}, "interaction": {},
               "similarity": {"mode": "hierarchical", "level_stride": 0, "rqvae_model_path": str(rqvae)}}
    config["method"] = {"name": "gflow", "reward_model": "multiply", "reward_epsilon": 1e-6,
                        "reward": {"components": [{"name": key, "weight": 1., "reward": value}
                                                   for key, value in rewards.items()]},
                        "sasrec": {"checkpoint_path": str(teacher_path)}}

    def cache_config(run_config, component, _args):
        result = deepcopy(run_config)
        result["method"] = {"name": "gflow", "reward_model": component, "reward_epsilon": 1e-6,
                            "reward": deepcopy(rewards[component]), "sasrec": deepcopy(config["method"]["sasrec"])}
        return result

    write(config_path, config)
    old_config = tmp_path / f"cache_config_{name}.json"
    write(old_config, cache_config(config, name, None))
    directory = tmp_path / "legacy_factors"
    validation.factors.main(["--config", str(old_config), "--output-dir", str(directory),
                             "--batch-size", "32", "--num-workers", "0", "--device", "cpu",
                             "--verify-batches", "1"], prepared_data=dm)
    session = SimpleNamespace(source={"model": "tiger", "dataset": "instruments", "source_id": "tiger_instruments"},
        args=SimpleNamespace(smoke=False, train_last_n=0, train_random_n=0, train_pool_last_n=None),
        config=config, config_path=config_path,
        base=SimpleNamespace(cache_config=cache_config, tiger_runner=runner))
    return SimpleNamespace(session=session, dm=dm, name=name, directory=directory,
                           old_config=old_config, teacher=teacher_path, root=tmp_path)


def rewrite_manifest(legacy, change):
    path = legacy.directory / "manifest.json"
    record = json.loads(path.read_text())
    change(record)
    write(path, record)


def rewrite_payload(legacy, change):
    path = legacy.directory / "factors.pt"
    payload = torch.load(path, weights_only=True)
    change(payload)
    torch.save(payload, path)
    rewrite_manifest(legacy, lambda m: m.update(factor_bytes=path.stat().st_size))


def validate(legacy):
    return validation.validate_component(legacy.session, legacy.dm, legacy.name, legacy.directory)


def test_current_factors_pass_without_builds_writes_or_rng_changes(legacy, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("validation attempted preprocessing or factor build")

    monkeypatch.setattr(legacy.dm, "setup", forbidden)
    monkeypatch.setattr(legacy.session.base.tiger_runner, "build_data_module", forbidden)
    monkeypatch.setattr(validation.factors, "main", forbidden)
    before = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
              for path in legacy.root.rglob("*") if path.is_file()}
    rng = torch.random.get_rng_state().clone()
    result = validate(legacy)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert result["legacy_validation"]["all_rows_checked"] == len(legacy.dm.train_dataset) > 256
    assert result["legacy_validation"]["historical_input_hashes_available"] is False
    assert result["legacy_validation"]["artifacts"]["factors"]["sha256"]
    after = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
             for path in legacy.root.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("key", ["user_ids", "target_item_ids", "target_timestamps", "target_semantic_row_ids", "target_token_ids"])
def test_every_row_identity_vector_is_checked_beyond_first256(legacy, key):
    def corrupt(payload):
        payload[key][257] += 1
    rewrite_payload(legacy, corrupt)
    with pytest.raises(ValueError, match="all .* rows"):
        validate(legacy)


@pytest.mark.parametrize("damage", ["subset", "rows", "data", "method", "component_config", "floor"])
def test_semantic_config_changes_are_rejected(legacy, damage):
    if damage == "method":
        record = json.loads(legacy.old_config.read_text())
        record["method"]["sasrec"]["checkpoint_path"] = "/different/teacher.pt"
        write(legacy.old_config, record)
    else:
        def change(record):
            if damage == "subset": record["per_user_last_n"] = 1
            elif damage == "rows": record["num_examples"] -= 1
            elif damage == "data": record["data_config"]["max_history_len"] += 1
            elif damage == "component_config": record["component"]["config"]["unrecognized_option"] = 1
            else: record["reward_epsilon"] = 1e-3
        rewrite_manifest(legacy, change)
    with pytest.raises(ValueError, match="Legacy factor import"):
        validate(legacy)


def test_operational_batch_settings_do_not_prevent_reuse(legacy):
    def change(record):
        record["data_config"].update(batch_size=256, eval_batch_size=128, num_workers=4,
                                      persistent_workers=True, include_raw_in_batch=True)
    rewrite_manifest(legacy, change)
    assert validate(legacy)["legacy_validation"]["all_rows_checked"] > 256


def test_changed_component_payload_is_detected_against_current_reward_assets(legacy):
    def corrupt(payload):
        if legacy.name == "sasrec": payload["item_table"][0, 0] += 1
        elif legacy.name == "similarity": payload["code_dist"][0, 0, 1] += .1
        else:
            # First probe has no previous positive but does have auxiliary
            # evidence. Changing its fill changes the full-catalogue rewards.
            payload["vals"].mul_(.5)
    rewrite_payload(legacy, corrupt)
    with pytest.raises(ValueError, match="SASRec item table|codebook distances|interaction rewards"):
        validate(legacy)


def test_nonfinite_factors_are_rejected_even_outside_hidden_probes(legacy):
    def corrupt(payload):
        if legacy.name == "sasrec": payload["hidden"][257, 0] = float("nan")
        elif legacy.name == "similarity": payload["code_dist"][0, 0, 1] = float("nan")
        else: payload["vals"][-1] = float("nan")
    rewrite_payload(legacy, corrupt)
    with pytest.raises(ValueError, match="nonfinite|codebook distances|columns or values"):
        validate(legacy)


def test_sasrec_hidden_probe_rejects_same_item_table_but_changed_sequence_weights(legacy):
    if legacy.name != "sasrec":
        pytest.skip("SASRec-specific sequence-weight validation")
    checkpoint = torch.load(legacy.teacher, weights_only=False)
    key = next(key for key in checkpoint["model_state_dict"] if key.endswith("weight") and "item_embedding" not in key)
    checkpoint["model_state_dict"][key].add_(10)
    torch.save(checkpoint, legacy.teacher)
    with pytest.raises(ValueError, match="SASRec hidden states"):
        validate(legacy)
