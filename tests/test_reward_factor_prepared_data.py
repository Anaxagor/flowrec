"""Factor precomputation borrows prepared data without duplicating its history."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import precompute_reward_factors as precompute
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.gflow.factorized_cache import FACTORS_FILENAME, FactorizedRewardCache
from precompute_reward_cache import subset_indices


@pytest.fixture
def prepared(tmp_path):
    events = tmp_path / "events.parquet"
    rows = []
    for user, count in ((1, 180), (2, 70), (3, 14)):
        for position in range(count):
            row = {"uid": user, "item_id": position % 11, "timestamp": 1000 + position,
                   "played_ratio_pct": 100., "event_type": "listen"}
            rows.append(row)
            if position % 10 == 0:
                rows.append({**row, "played_ratio_pct": 20.})
    pd.DataFrame(rows).to_parquet(events)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({str(i): [i + 1, i + 17] for i in range(11)}))
    cfg = YambdaDataConfig(
        interactions_path=events, semantic_id_path=semantic, hf_recent_months=None,
        positive_criterion="played_ratio_pct >= 50", min_user_interactions=3,
        drop_items_missing_embedding=False, offset_semantic_ids_by_one=False,
        train_target_policy="last_n", train_last_n=128, max_history_len=50,
        num_workers=0, persistent_workers=False, include_raw_in_batch=False,
        batch_size=9, eval_batch_size=11, seed=42,
    )
    data = YambdaDataModule(cfg)
    data.setup()
    return data


@pytest.mark.parametrize("component", ["similarity", "interaction"])
@pytest.mark.parametrize("selection", ["last128", "sampled50"])
def test_prepared_factors_equal_standalone_without_reloading_or_tensorizing(
    tmp_path, prepared, monkeypatch, capsys, component, selection,
):
    config = {
        "seed": 42,
        "data": {"source": "yambda", "interactions_path": str(prepared.cfg.interactions_path),
                 "semantic_id_path": str(prepared.cfg.semantic_id_path)},
        "dataset": asdict(prepared.cfg), "model": {"use_user_token": False},
        "method": {"name": "gflow", "reward_model": component,
                   "reward": {"mode": "exact"} if component == "similarity" else {"dataset": "yambda"}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, default=str))
    common = ["--config", str(config_path), "--device", "cpu", "--batch-size", "8",
              "--num-workers", "0", "--per-user-pool-last-n", "128", "--verify-batches", "1"]
    subset = {"per_user_pool_last_n": 128}
    if selection == "sampled50":
        common += ["--per-user-random-n", "50", "--per-user-random-seed", "42"]
        subset.update(per_user_random_n=50, per_user_random_seed=42)
    else:
        common += ["--per-user-last-n", "128"]
        subset.update(per_user_last_n=128)
    indices = subset_indices(prepared.train_dataset, **subset)
    original_frame = prepared.train_dataset.examples.copy(deep=True)
    original_cfg = asdict(prepared.cfg)

    def unused_loader(*args, **kwargs):
        raise AssertionError("These factor components must not tensorize the full training layout")

    monkeypatch.setattr(YambdaDataModule, "train_dataloader", unused_loader)
    fresh_dir = tmp_path / "fresh"
    precompute.main([*common, "--output-dir", str(fresh_dir)])

    def duplicate_setup(*args, **kwargs):
        raise AssertionError("A prepared datamodule must not be rebuilt or set up again")

    monkeypatch.setattr(precompute.tiger_runner, "build_data_config", duplicate_setup)
    monkeypatch.setattr(precompute.tiger_runner, "build_data_module", duplicate_setup)
    monkeypatch.setattr(YambdaDataModule, "setup", duplicate_setup)
    reused_dir = tmp_path / "reused"
    precompute.main([*common, "--output-dir", str(reused_dir)], prepared_data=prepared)
    assert "reusing prepared dataset; skipping data setup" in capsys.readouterr().out

    fresh = torch.load(fresh_dir / FACTORS_FILENAME, weights_only=True)
    reused = torch.load(reused_dir / FACTORS_FILENAME, weights_only=True)
    assert fresh.keys() == reused.keys()
    for name in fresh:
        torch.testing.assert_close(reused[name], fresh[name], rtol=0, atol=0)
    selected = original_frame.iloc[indices]
    assert reused["user_ids"].tolist() == selected.user_id.tolist()
    assert reused["target_item_ids"].tolist() == selected.target_item_id.tolist()
    assert reused["target_timestamps"].tolist() == selected.target_timestamp.tolist()

    fresh_manifest = json.loads((fresh_dir / "manifest.json").read_text())
    manifest = json.loads((reused_dir / "manifest.json").read_text())
    assert manifest["data_config"] == precompute.jsonable(original_cfg)
    # Manifest provenance describes the borrowed dataset, while cache batch
    # size is explicitly independent from the training/validation settings.
    assert manifest["data_config"]["batch_size"] == 9
    assert manifest["batch_size"] == fresh_manifest["data_config"]["batch_size"] == 8
    for key in ("training_context", "num_catalog_items", "num_examples", "component",
                "per_user_last_n", "per_user_pool_last_n", "per_user_random_n", "per_user_random_seed"):
        assert manifest[key] == fresh_manifest[key]
    assert manifest["num_examples"] == len(indices)
    cache = FactorizedRewardCache(reused_dir)
    reference = FactorizedRewardCache(fresh_dir)
    rows = torch.arange(len(indices))
    torch.testing.assert_close(cache.component_rows(rows, torch.device("cpu")),
                               reference.component_rows(rows, torch.device("cpu")), rtol=0, atol=0)
    assert manifest["verification"]["max_abs_diff"] < 1e-6
    pd.testing.assert_frame_equal(prepared.train_dataset.examples, original_frame)
    assert asdict(prepared.cfg) == original_cfg


@pytest.mark.parametrize("data,message", [
    (SimpleNamespace(), "original data configuration"),
    (SimpleNamespace(cfg=YambdaDataConfig(), train_dataset=None), "already be set up"),
    (SimpleNamespace(cfg=YambdaDataConfig(), train_dataset=object(), semantic_ids=None, sid_len=None),
     "semantic_id_path"),
])
def test_prepared_data_requires_initialized_data(tmp_path, data, message):
    config = tmp_path / "config.json"
    config.write_text("{}")
    with pytest.raises(ValueError, match=message):
        precompute.main(["--config", str(config), "--output-dir", str(tmp_path / "out"),
                         "--device", "cpu"], prepared_data=data)
