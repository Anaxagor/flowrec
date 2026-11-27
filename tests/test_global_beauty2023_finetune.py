"""Beauty 2023 global arrays and their compact, model-specific Amazon inputs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
import finetune_beauty2023 as experiment
import ce_yambda_prepared_data as compact
from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.indexed_batches import create_indexed_parts
from sasrec_config_runner import training_item_popularity


@pytest.mark.parametrize("model", ["gpt2", "tiger"])
def test_global_beauty_array_settings(model, tmp_path):
    import gflow_finetune_gpt2_amazon as gpt
    import gflow_finetune_yambda500m as tiger

    grid = experiment.grid(model, experiment.DATASET)
    assert [entry["seed"] for entry in grid["sft"]] == [42, 43, 44]
    for arm in ("gflow", "ce"):
        assert {(row["power"], row["rho"]) for row in grid[arm]} == {
            (power, rho) for power in (.25, 1., 16.) for rho in (1., .1)}
        assert {row["combo"] for row in grid[arm]} == {"sasrec_x_interaction_x_similarity"}
        assert {row["seed"] for row in grid[arm]} == {42}
    assert len(grid["gflow"]) == 18 and {e["objective"] for e in grid["gflow"]} == {"tb", "db", "subtb"}
    assert all(e["num_negatives"] == 3 and e["sampler"] == "teacher" for e in grid["gflow"])
    assert len(grid["ce"]) == 6
    assert all(e["num_samples"] == 3 and e["sampling"] == "reward_negatives"
               and not e["importance_correction"] for e in grid["ce"])
    cli = SimpleNamespace(model=model, dataset=experiment.DATASET, device="cpu", num_workers=0, smoke=False)
    args = experiment.gflow.training_args(gpt if model == "gpt2" else tiger, cli, 44, tmp_path, tmp_path)
    assert (args.batch_size, args.grad_accum) == (128, 16)
    assert not args.train_last_n and not args.train_random_n
    assert args.teacher_sampler_label == "sasrec_curriculum" and args.strict_negatives
    if model == "gpt2":
        assert args.prefix_from_dense


@pytest.mark.parametrize("model", ["gpt2", "tiger"])
def test_amazon_compact_prefixes_match_source_and_keep_training_reward_window(tmp_path, model):
    reviews = tmp_path / "reviews.parquet"
    rows = [dict(user_id=str(u), parent_asin=str((u + t) % 9), timestamp=t,
                 rating=5., verified_purchase=True) for u in range(3) for t in range(20)]
    rows += [{**rows[t], "verified_purchase": False, "rating": 2.} for t in (1, 12, 18)]
    pd.DataFrame(rows).to_parquet(reviews)
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({str(i): [i, 16 + i] for i in range(9)}))
    cfg = AmazonReviewDataConfig(reviews_path=reviews, semantic_id_path=ids, min_user_interactions=2,
        item_key="parent_asin", filter_reviews_to_metadata=False, min_history_len=1, max_history_len=4,
        train_prefix_from_dense=model == "gpt2", history_padding="right" if model == "gpt2" else "left",
        positive_criterion="verified_purchase == True", split_strategy="global_time",
        global_val_timestamp=12, global_test_timestamp=16, global_eval_target_mode="last_per_user",
        num_workers=0, batch_size=8, eval_batch_size=8, include_raw_in_batch=False)
    dm = AmazonReviewDataModule(cfg)
    dm.setup()
    experiment.gflow.store_reward_events(dm, tmp_path)
    assert pd.read_parquet(tmp_path / "positive.parquet").timestamp.max() == 11
    assert pd.read_parquet(tmp_path / "auxiliary.parquet").timestamp.tolist() == [1]

    source = {"model": model}
    compact.prepare(tmp_path / "prefixes", dm, None, source)
    loaded = compact.load(tmp_path / "prefixes", cfg, source)
    assert isinstance(loaded, AmazonReviewDataModule)
    assert loaded._ce_prepared_source_indices.tolist() == list(range(len(dm.train_dataset)))
    assert loaded.train_item_counts_by_raw_id == dm.train_item_counts_by_raw_id
    assert training_item_popularity(loaded).sum() == 36
    assert loaded.interactions.empty
    positions = list(range(len(dm.train_dataset)))
    for original, prepared in [
        (create_indexed_parts(dm, None, tensorized=True)[1](positions),
         create_indexed_parts(loaded, None, tensorized=True)[1](positions)),
        (next(iter(dm.val_dataloader())), next(iter(loaded.val_dataloader()))),
        (next(iter(dm.test_dataloader())), next(iter(loaded.test_dataloader()))),
    ]:
        assert original.keys() == prepared.keys()
        for key in original:
            if torch.is_tensor(original[key]):
                torch.testing.assert_close(original[key], prepared[key])
            else:
                assert original[key] == prepared[key]
    experiment.gflow.restore_reward_events(loaded, tmp_path)
    assert len(loaded.interaction_store.events) == 1
