"""Amazon fine-tuning expands base dense windows and retains complete held-out data."""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon_fullhist as matched
from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.dense_prefix import dense_prefix_signature, prefixes_from_dense_windows


def amazon_config(tmp_path):
    rows = [dict(user_id=str(user), asin=str(i % 31), timestamp=1000 + i,
                 rating=float(i % 5 + 1), verified_purchase=True)
            for user, length in enumerate([5, 23, 66]) for i in range(length)]
    rows[-1]["asin"] = "cold"
    rows.append(dict(user_id="2", asin="0", timestamp=1055, rating=1., verified_purchase=False))
    reviews = tmp_path / "reviews.parquet"
    pd.DataFrame(rows).to_parquet(reviews)
    sids = tmp_path / "sids.json"
    sids.write_text(json.dumps({**{str(i): [i // 8, i % 8] for i in range(31)}, "cold": [4, 1]}))
    return AmazonReviewDataConfig(
        reviews_path=reviews, semantic_id_path=sids, positive_criterion="verified_purchase == True",
        max_history_len=20, min_user_interactions=3, min_item_interactions=0,
        train_example_layout="dense_causal", train_target_policy="all", history_padding="left",
        processed_cache_dir=tmp_path / "processed", num_workers=0, persistent_workers=False,
        batch_size=8, include_raw_in_batch=False,
    )


@pytest.mark.parametrize("cold_filter", [False, True])
def test_amazon_dense_expansion_retains_catalog_eval_and_cached_audit(tmp_path, cold_filter):
    cfg = replace(amazon_config(tmp_path), filter_cold_items_after_split=cold_filter)
    dense = AmazonReviewDataModule(cfg)
    dense.setup()
    prefix = AmazonReviewDataModule(replace(cfg, train_example_layout="prefix_target", train_prefix_from_dense=True))
    prefix.setup()
    expected = prefixes_from_dense_windows(dense.train_dataset.examples, dense.interactions)
    pd.testing.assert_frame_equal(prefix.train_dataset.examples, expected)
    assert len(prefix.train_dataset) == sum(sum(mask) for mask in dense.train_dataset.examples.label_mask)
    assert prefix.train_dataset.examples.groupby("user_id").size().tolist() == [2, 20, 63]
    assert prefix.item2id == dense.item2id and prefix.semantic_ids == dense.semantic_ids
    assert len(prefix.interaction_store) == len(dense.interaction_store) == 1
    for name in ("val", "test"):
        pd.testing.assert_frame_equal(getattr(prefix, f"{name}_dataset").examples,
                                      getattr(dense, f"{name}_dataset").examples)
        for left, right in zip(getattr(prefix, f"{name}_dataloader")(), getattr(dense, f"{name}_dataloader")()):
            for key in left:
                if isinstance(left[key], torch.Tensor):
                    torch.testing.assert_close(left[key], right[key])
    reloaded = AmazonReviewDataModule(prefix.cfg)
    reloaded.setup()
    pd.testing.assert_frame_equal(reloaded.train_dataset.examples, prefix.train_dataset.examples)
    assert reloaded.dense_prefix_audit == prefix.dense_prefix_audit
    assert reloaded.summary()["training_context"] == dense_prefix_signature(asdict(prefix.cfg))
    assert dense._processed_cache_path() != prefix._processed_cache_path()
    legacy = AmazonReviewDataModule(replace(prefix.cfg, train_prefix_from_dense=False))
    legacy.setup()
    assert legacy._processed_cache_path() != prefix._processed_cache_path()
    assert legacy.train_dataset.examples.history_item_ids.tolist() != prefix.train_dataset.examples.history_item_ids.tolist()


def test_dense_reward_precompute_records_context_and_preserves_target_alignment(tmp_path, monkeypatch):
    import precompute_reward_cache

    cfg = replace(amazon_config(tmp_path), train_example_layout="prefix_target", train_prefix_from_dense=True)
    dm = AmazonReviewDataModule(cfg)
    dm.setup()
    config = {"data": {"source": "amazon", "reviews_path": str(cfg.reviews_path),
                       "semantic_id_path": str(cfg.semantic_id_path)},
              "dataset": asdict(cfg),
              "model": {"use_user_token": False, "codebook_size": 8, "embedding_dim": 8,
                        "num_heads": 2, "num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 16},
              "method": {"name": "gflow", "reward_model": "similarity", "reward": {"mode": "exact"}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, default=str))
    output = tmp_path / "rewards"
    monkeypatch.setattr(sys, "argv", ["precompute_reward_cache.py", "--config", str(path),
                                     "--output-dir", str(output), "--device", "cpu",
                                     "--batch-size", "8", "--num-workers", "0"])
    precompute_reward_cache.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_context"] == dense_prefix_signature(config["dataset"])
    assert manifest["num_examples"] == len(dm.train_dataset)
    matched.base.gflow_ablation.verify_cache_alignment(dm, list(range(len(dm.train_dataset))), output)


@pytest.mark.parametrize("dataset", ["beauty", "instruments", "beauty_2023", "yambda_50m_listens", "yambda_500m_listens"])
def test_reject_old_completed_prefix_runs_even_if_counts_match(tmp_path, dataset):
    args = matched.parse_args(["--datasets", dataset])
    (tmp_path / "resolved_config.json").write_text(json.dumps({"cli_args": {"prefix_from_dense": False}}))
    (tmp_path / "result.json").write_text(json.dumps({"status": "completed"}))
    with pytest.raises(RuntimeError, match="Different training context"):
        matched.base.check_existing_training_context(tmp_path, args)
