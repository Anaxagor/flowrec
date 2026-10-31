"""Tiny matched Yambda data -> real factor precompute -> GPT-2 CE training.

The source architecture is reduced only in these temporary checkpoints. Data
construction, frozen source audit, target selection, reward factors, training,
checkpoint reload, and final metrics use the production experiment code.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "scripts", ROOT / "scripts/exp2-tuned-params",
               ROOT / "scripts/exp7-ce-distillation"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import ce_distill_gpt2_yambda as experiment
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache
from precompute_reward_cache import per_user_random_indices
from tuning_gpt2 import audit_matched_data


@pytest.fixture(params=experiment.DATASETS)
def matched_inputs(tmp_path, monkeypatch, request):
    dataset = request.param
    matched, base = experiment.load_runtime()
    source = base.matched_source
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    events = []
    for user, count in ((1, 190), (2, 14)):
        for position in range(count):
            events.append({"uid": user, "item_id": position % 32,
                           "timestamp": 1000 + position * 10,
                           "event_type": "listen", "played_ratio_pct": 100.})
        events.extend([
            {"uid": user, "item_id": 1, "timestamp": 1005,
             "event_type": "like", "played_ratio_pct": 0.},
            {"uid": user, "item_id": 2, "timestamp": 1015,
             "event_type": "listen", "played_ratio_pct": 20.},
        ])
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(events).to_parquet(events_path)
    semantic_root = tmp_path / "semantic_ids"
    semantic_path = semantic_root / dataset / "semantic_ids.json"
    semantic_path.parent.mkdir(parents=True)
    semantic_path.write_text(json.dumps({
        str(i): [i // 16 + 1, 17 + i % 16] for i in range(32)
    }))
    cfg = YambdaDataConfig(
        interactions_path=events_path, semantic_id_path=semantic_path,
        hf_recent_months=None, min_user_interactions=3,
        min_item_interactions=0, drop_items_missing_embedding=False,
        allowed_event_types=("listen",), min_played_ratio_pct=50.,
        offset_semantic_ids_by_one=False, user_hash_mod=None,
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_prefix_from_dense=False, train_window_len=50, train_window_stride=1,
        train_target_policy="last_n", train_last_n=128, train_all_prefixes=False,
        max_history_len=50, history_padding="right", num_workers=0,
        persistent_workers=False, include_raw_in_batch=False, batch_size=4,
        seed=42,
    )
    dense = YambdaDataModule(cfg)
    dense.setup()
    architecture = {"embedding_dim": 8, "num_layers": 1, "num_heads": 2,
                    "dim_feedforward": 16}
    model = base.gpt2_runner.build_model(
        base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML),
        {"model": {**architecture, "codebook_size": 16,
                   "max_position_embeddings": 205, "dropout": 0.}},
        dense,
    )
    paths = source.trial_paths(tmp_path, dataset)
    paths["checkpoint"].parent.mkdir(parents=True)
    recorded = {
        "dataset": dataset, "seed": 42, "model_config": model.config_dict(),
        "hparams": {"dropout": 0.}, "data_config": asdict(cfg),
        "resume_identity": {"data_audit": audit_matched_data(dense)},
    }
    paths["resolved_config"].write_text(json.dumps(recorded, default=str))
    paths["sidecar"].write_text(json.dumps({"model_config": model.config_dict()}))
    torch.save({"model_config": model.config_dict(),
                "model_state_dict": model.state_dict()}, paths["checkpoint"])
    monkeypatch.setitem(source.TRIALS, dataset,
                        {**source.TRIALS[dataset], "architecture": architecture})
    monkeypatch.setattr(base, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(base, "SEMANTIC_IDS_ROOT", semantic_root)
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    try:
        yield dataset, matched, base, dense, paths
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("sampling", ["reward", "curriculum_smooth_is"])
@pytest.mark.parametrize("target_count", [15, 50])
def test_selected_trial_data_factors_and_complete_ce_run(
    tmp_path, monkeypatch, matched_inputs, sampling, target_count,
):
    dataset, matched, raw_base, dense, paths = matched_inputs
    args = experiment.parse_args([
        "--datasets", dataset, "--combos", "interaction", "--powers", "1",
        "--rho-targets", "0.1", "--seeds", "43", "--sampling", sampling,
        "--curriculum-epochs", "1", "--device", "cpu", "--batch-size", "2",
        "--grad-accum", "1", "--num-workers", "0", "--cache-batch-size", "16",
        "--cache-verify-batches", "1", "--max-epochs", "1", "--min-epochs", "0",
        "--max-train-batches", "2", "--max-eval-batches", "1",
        "--calibration-batches", "1", "--cache-root", str(tmp_path / "factors"),
        "--output-root", str(tmp_path / "results"),
        "--train-random-n", str(target_count),
    ])
    b = experiment.matched_args(args, dataset, 43, matched)
    base = experiment.runtime_for(raw_base, b)
    config = experiment.build_run_config(base, dataset, b)
    dm = base.tiger_runner.build_data_module(
        base.tiger_runner.build_data_config(config, config_path=tmp_path / "run.json"))
    dm.setup()
    audit = experiment.audit_data(base, dm, b)
    assert audit["all_fingerprints_equal"]
    assert audit["source_trial"] == experiment.SOURCE_TRIALS[dataset]
    assert dm.cfg.seed == 42 and b.seed == 43
    assert not dm.cfg.train_prefix_from_dense
    assert dm.interaction_store.counts_by_type() == {"like": 2, "listen": 2}
    for column in ("history_item_ids", "target_item_id", "target_timestamp"):
        assert dm.train_dataset.examples[column].tolist() == dense.train_dataset.examples[column].tolist()
    for split in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(dm, split).examples,
                                      getattr(dense, split).examples)
    indices = base.finetune_indices(dm.train_dataset, b)
    if dataset == experiment.DATASETS[0]:
        assert indices is None
        selected = dm.train_dataset.examples
    else:
        assert indices == per_user_random_indices(dm.train_dataset, target_count, 42)
        selected = dm.train_dataset.examples.iloc[indices]
        assert selected.groupby("user_id").size().max() == target_count
        assert selected.groupby("user_id").size().min() == dm.train_dataset.examples.groupby("user_id").size().min()
        other = experiment.matched_args(args, dataset, 44, matched)
        assert base.finetune_indices(dm.train_dataset, other) == indices
    assert selected.history_item_ids.map(len).max() == 50
    model, _ = base.build_gpt2_from_tuned_checkpoint(dataset, dm)
    assert base.tuned_checkpoint_paths(dataset)["checkpoint"] == paths["checkpoint"]

    # Exercise the real prepared-data path; CPU interaction factors must not
    # start a subprocess or request the SASRec teacher.
    def forbidden(*unused, **kwargs):
        pytest.fail("Interaction-only cache phase must not invoke this path")

    monkeypatch.setattr(raw_base.subprocess, "run", forbidden)
    monkeypatch.setattr(raw_base, "matched_teacher_translation", forbidden)
    experiment.build_caches(args, matched, raw_base)
    cache_build = experiment.shared.dataset_output_root(args, dataset) / "cache_build"
    for filename in ("data_reward_config.json", "matched_prefix_audit.json", "interaction/timing.json"):
        assert (cache_build / filename).is_file()
    cache_dir = experiment.component_cache_dir(base, dataset, "interaction", b)
    manifest_path = cache_dir / "manifest.json"
    manifest = experiment.check_component_manifest(
        base, config, manifest_path, "interaction", len(selected), b)
    assert manifest["num_examples"] == len(selected)
    assert manifest["training_context"]["context"] == "matched_prefix"
    if dataset == experiment.DATASETS[1]:
        assert manifest["per_user_random_n"] == target_count
        wrong_count = 15 if target_count == 50 else 50
        mismatched = json.loads(manifest_path.read_text())
        mismatched["per_user_random_n"] = wrong_count
        manifest_path.write_text(json.dumps(mismatched))
        with pytest.raises(RuntimeError, match="cache subset per_user_random_n"):
            experiment.check_component_manifest(base, config, manifest_path, "interaction", len(selected), b)
        manifest_path.write_text(json.dumps(manifest))
    cache = FactorizedRewardCache(cache_dir)
    experiment.verify_caches(base, dm, indices, {"interaction": cache}, model)
    original_timestamp = cache.factors["target_timestamps"][-1].clone()
    cache.factors["target_timestamps"][-1] += 1
    with pytest.raises(RuntimeError, match="target_timestamps"):
        experiment.verify_caches(base, dm, indices, {"interaction": cache}, model)
    cache.factors["target_timestamps"][-1] = original_timestamp
    # A source audit drift must be rejected even if row counts still agree.
    changed = json.loads(manifest_path.read_text())
    changed["training_context"]["data_audit"]["train_contexts_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="training_context"):
        experiment.check_component_manifest(base, config, manifest_path, "interaction", len(selected), b)
    manifest_path.write_text(json.dumps(manifest))

    entry = experiment.build_grid(args)[0]
    experiment.gpt2.run_entry(args, entry, matched, raw_base, profile=experiment)
    run_dir = experiment.run_dir_for(args, entry)
    result = experiment.read_json(run_dir / "result.json")
    resolved = experiment.read_json(run_dir / "resolved_config.json")
    calibration = experiment.read_json(run_dir / "calibration.json")
    assert result["status"] == "completed" and result["model"] == "gpt2"
    assert result["source_trial"] == experiment.SOURCE_TRIALS[dataset]
    assert result["target_selection"] == ("last128" if dataset == experiment.DATASETS[0] else f"sampled{target_count}")
    assert result["num_examples"] == len(selected)
    assert result["sampling"] == sampling and result["ce_target"] == "q"
    assert result["micro_steps"] == 2
    assert result["evaluation_scope"] == "limited"
    assert result["splits"]["test"]
    assert calibration["weight"] > 0
    assert resolved["identity"]["trainer"]["amp_dtype"] == "bfloat16"
    assert resolved["identity"]["trainer"]["val_sample_fraction"] == (
        None if dataset == experiment.DATASETS[0] else .25)
    for checkpoint in ("last.pt", "best.pt"):
        payload = torch.load(run_dir / "checkpoints" / checkpoint,
                             map_location="cpu", weights_only=False)
        assert all(torch.isfinite(value).all() for value in payload["model_state_dict"].values()
                   if torch.is_tensor(value))
    saved_audit = experiment.read_json(
        run_dir / getattr(experiment, "DATA_AUDIT_FILENAME", "dense_prefix_audit.json"))
    assert saved_audit["source_trial"] == experiment.SOURCE_TRIALS[dataset]
    assert saved_audit["all_fingerprints_equal"]
