"""Real tiny Beauty 2023 data -> factors -> corrected GPT-2 CE -> reload/eval."""
from __future__ import annotations

from dataclasses import replace
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

import ce_distill_gpt2_beauty2023 as experiment
from flowrec.data.datamodule import AmazonReviewDataModule
from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache


@pytest.fixture
def tiny_inputs(tmp_path, monkeypatch):
    matched, base = experiment.load_runtime()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    reviews = tmp_path / "reviews.parquet"
    rows = [dict(user_id=f"user-{user}", parent_asin=str(i % 32), timestamp=1000 + i,
                 rating=5., verified_purchase=True)
            for user, count in [(0, 66), (1, 14)] for i in range(count)]
    rows.append(dict(user_id="user-0", parent_asin="31", timestamp=1001,
                     rating=2., verified_purchase=False))
    pd.DataFrame(rows).to_parquet(reviews)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame({"parent_asin": [str(i) for i in range(32)]}).to_parquet(metadata)
    semantic_root = tmp_path / "semantic_ids"
    semantic_path = semantic_root / "beauty_2023/semantic_ids.json"
    semantic_path.parent.mkdir(parents=True)
    semantic_path.write_text(json.dumps({str(i): [i // 16, i % 16] for i in range(32)}))
    source_yaml = base.runner.load_yaml(base.dataset_yaml_path("beauty_2023"))
    source_yaml.update(min_user_interactions=3, min_item_interactions=0,
                       processed_cache_dir=str(tmp_path / "processed"))
    source_yaml["datasets"]["beauty_2023"].update(
        reviews_path=str(reviews), metadata_path=str(metadata))
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(json.dumps(source_yaml))
    monkeypatch.setattr(base, "dataset_yaml_path", lambda dataset: yaml_path)
    monkeypatch.setattr(base, "SEMANTIC_IDS_ROOT", semantic_root)
    profile = {"model": {"embedding_dim": 8, "num_layers": 1, "num_heads": 2,
                          "dim_feedforward": 16, "codebook_size": 16,
                          "max_position_embeddings": 85, "dropout": 0.}}
    paths = {"checkpoint": tmp_path / "model/best_model.pt",
             "sidecar": tmp_path / "model/best_model.pt.json",
             "best_trial": tmp_path / "model/best_trial.json"}
    paths["checkpoint"].parent.mkdir()
    paths["sidecar"].write_text(json.dumps({"model_config": profile["model"]}))
    paths["best_trial"].write_text(json.dumps({"best_params": {"dropout": 0.}}))
    monkeypatch.setattr(base, "tuned_checkpoint_paths", lambda dataset, args=None: paths)
    args = experiment.parse_args(["--device", "cpu", "--batch-size", "2",
                                  "--grad-accum", "1", "--num-workers", "0"])
    b = experiment.matched_args(args, "beauty_2023", 42, matched)
    config = experiment.build_run_config(base, "beauty_2023", b)
    cfg = base.tiger_runner.build_data_config(config, config_path=tmp_path / "run.json")
    dm = AmazonReviewDataModule(cfg)
    dm.setup()
    model = base.gpt2_runner.build_model(base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML),
                                        profile, dm)
    torch.save({"model_config": model.config_dict(), "model_state_dict": model.state_dict()},
               paths["checkpoint"])
    paths["sidecar"].write_text(json.dumps({"model_config": model.config_dict()}))
    try:
        yield matched, base, dm, paths
    finally:
        torch.set_num_threads(old_threads)


def test_real_factors_corrected_loss_checkpoint_and_final_metrics(tmp_path, monkeypatch, tiny_inputs):
    matched, base, prefix, paths = tiny_inputs
    args = experiment.parse_args([
        "--combos", "interaction", "--powers", "1", "--rho-targets", "0.1",
        "--curriculum-epochs", "1", "--device", "cpu", "--batch-size", "2",
        "--grad-accum", "1", "--num-workers", "0", "--cache-batch-size", "16",
        "--cache-verify-batches", "1", "--max-epochs", "1", "--min-epochs", "0",
        "--max-train-batches", "2", "--max-eval-batches", "1", "--calibration-batches", "1",
        "--cache-root", str(tmp_path / "factors"), "--output-root", str(tmp_path / "results"),
    ])
    b = experiment.matched_args(args, "beauty_2023", 42, matched)
    config = experiment.build_run_config(base, "beauty_2023", b)
    # Resolve the GFlow launcher profile independently, then compare actual
    # rows and tokenized examples. This catches context or tokenization drift even if
    # the two pipelines happen to retain identical example counts.
    gflow_args = base.dataset_args(matched.parse_args([
        "--datasets", "beauty_2023", "--num-candidates", "3", "--prefix-from-dense",
        "--seed", "42", "--device", "cpu", "--batch-size", "2",
        "--grad-accum", "1", "--num-workers", "0",
    ]), "beauty_2023")
    gflow_config = base.build_run_config("beauty_2023", gflow_args,
                                        config["model"]["user_ids_count"])
    gflow = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(
        gflow_config, config_path=tmp_path / "gflow.json"))
    gflow.setup()
    assert prefix.item2id == gflow.item2id
    for split in ("train_dataset", "val_dataset", "test_dataset"):
        ce_dataset, gflow_dataset = getattr(prefix, split), getattr(gflow, split)
        pd.testing.assert_frame_equal(ce_dataset.examples, gflow_dataset.examples)
        for index in range(len(ce_dataset)):
            ce_example, gflow_example = ce_dataset[index], gflow_dataset[index]
            assert ce_example.keys() == gflow_example.keys()
            for key, value in ce_example.items():
                other = gflow_example[key]
                if torch.is_tensor(value):
                    assert torch.equal(value, other), (split, index, key)
                else:
                    assert value == other, (split, index, key)
    assert base.finetune_indices(prefix.train_dataset, b) == base.finetune_indices(
        gflow.train_dataset, gflow_args)
    dense = AmazonReviewDataModule(replace(prefix.cfg, train_example_layout="dense_causal",
                                           train_prefix_from_dense=False))
    dense.setup()
    oracle = []
    for row in dense.train_dataset.examples.itertuples():
        for offset, active in enumerate(row.label_mask):
            if active:
                oracle.append((row.user_id, row.target_position - len(row.label_mask) + offset + 1,
                               row.label_item_ids[offset], tuple(row.history_item_ids[:offset + 1])))
    actual = [(row.user_id, row.target_position, row.target_item_id, tuple(row.history_item_ids))
              for row in prefix.train_dataset.examples.itertuples()]
    assert sorted(actual) == sorted(oracle)
    assert prefix.item2id == dense.item2id
    assert prefix.train_dataset.examples.history_item_ids.map(len).max() == 20
    for split in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(prefix, split).examples,
                                      getattr(dense, split).examples)
    indices = base.finetune_indices(prefix.train_dataset, b)
    assert indices == list(range(len(prefix.train_dataset)))

    def forbidden(*unused, **kwargs):
        pytest.fail("CPU interaction factors must use prepared data without a subprocess or SASRec teacher")

    monkeypatch.setattr(base.subprocess, "run", forbidden)
    monkeypatch.setattr(base, "matched_teacher_translation", forbidden)
    experiment.build_caches(args, matched, base)
    cache_dir = experiment.component_cache_dir(base, "beauty_2023", "interaction", b)
    path = cache_dir / "manifest.json"
    manifest = experiment.check_component_manifest(base, config, path, "interaction", len(indices), b)
    assert manifest["num_examples"] == len(indices)
    assert manifest["training_context"]["context"] == "dense_window"
    cache = FactorizedRewardCache(cache_dir)
    model, _ = base.build_gpt2_from_tuned_checkpoint("beauty_2023", prefix)
    experiment.verify_caches(base, prefix, indices, {"interaction": cache}, model)
    changed = dict(manifest, training_context=None)
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="training_context"):
        experiment.check_component_manifest(base, config, path, "interaction", len(indices), b)
    path.write_text(json.dumps(manifest))

    entry = experiment.build_grid(args)[0]
    experiment.run_entry(args, entry, matched, base)
    run_dir = experiment.run_dir_for(args, entry)
    result = experiment.read_json(run_dir / "result.json")
    resolved = experiment.read_json(run_dir / "resolved_config.json")
    calibration = experiment.read_json(run_dir / "calibration.json")
    assert result["status"] == "completed" and result["model"] == "gpt2"
    assert result["dataset"] == "beauty_2023" and result["num_examples"] == len(indices)
    assert result["sampling"] == "curriculum_smooth_is" and result["ce_target"] == "q"
    assert result["importance_correction"] and result["num_samples"] == 3
    assert result["micro_steps"] == 2 and result["evaluation_scope"] == "limited"
    for split in ("validation", "test"):
        assert result["splits"][split]["num_examples"] == len(getattr(prefix, "val_dataset" if split == "validation" else "test_dataset"))
        for metric in ("HR", "NDCG", "Coverage", "ARP"):
            for k in (5, 10, 20):
                assert result["splits"][split][f"{metric}@{k}"] >= 0
    assert calibration["weight"] > 0
    assert resolved["method"]["sampling"] == "curriculum_smooth_is"
    assert resolved["method"]["proposal_q_mix"] == .1
    assert resolved["identity"]["trainer"]["val_sample_fraction"] is None
    logs = [json.loads(line) for line in (run_dir / "training_log.jsonl").read_text().splitlines()]
    assert any("importance_weight_mean" in json.dumps(row) for row in logs)
    for name in ("last.pt", "best.pt"):
        payload = torch.load(run_dir / "checkpoints" / name, map_location="cpu", weights_only=False)
        assert all(torch.isfinite(value).all() for value in payload["model_state_dict"].values()
                   if torch.is_tensor(value))
    saved = experiment.read_json(run_dir / "dense_prefix_audit.json")
    assert saved["validation"]["equal"] and saved["test"]["equal"]
    # A completed matching run must be reusable without loading any data/model.
    monkeypatch.setattr(base.tiger_runner, "build_data_module", forbidden)
    experiment.run_entry(args, entry, matched, base)
