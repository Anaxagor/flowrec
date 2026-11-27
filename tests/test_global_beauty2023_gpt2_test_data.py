"""The checkpoint evaluator consumes the complete native test split and training popularity."""
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
import json

import pandas as pd
import pytest
import torch

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule

data = import_module("scripts.global-time.gpt2_beauty2023_final_data")


def test_heldout_cache_preserves_native_batches_and_train_popularity(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews.parquet"
    metadata = tmp_path / "meta.parquet"
    semantic = tmp_path / "semantic.json"
    items = ["A", "B", "C"]
    rows = [{"user_id": user, "parent_asin": items[i % 3], "timestamp": timestamp,
             "rating": 5.0, "verified_purchase": True}
            for user in ("u1", "u2") for i, timestamp in enumerate((1, 2, 3, 4, 10, 11, 20, 21))]
    pd.DataFrame(rows).to_parquet(reviews)
    pd.DataFrame({"parent_asin": items}).to_parquet(metadata)
    semantic.write_text(json.dumps({item: [i, i] for i, item in enumerate(items)}))
    monkeypatch.setattr(data, "SID_SHA256", sha256(semantic.read_bytes()).hexdigest())
    cfg = AmazonReviewDataConfig(reviews_path=str(reviews), metadata_path=str(metadata),
        semantic_id_path=str(semantic), item_key="parent_asin", positive_criterion="verified_purchase == True",
        split_strategy="global_time", global_val_timestamp=10, global_test_timestamp=20,
        global_eval_target_mode="last_per_user", train_example_layout="dense_causal",
        history_padding="right", min_item_interactions=1, min_user_interactions=1,
        include_raw_in_batch=False, num_workers=0, pin_memory=False, test_batch_size=1)
    original = AmazonReviewDataModule(cfg)
    original.setup()
    (tmp_path / "study_config.json").write_text(json.dumps({"data_config": asdict(cfg)}))
    (tmp_path / "data_summary.json").write_text(json.dumps(original.summary()))
    paths = (tmp_path, tmp_path / "prepared", tmp_path / "node-scratch")
    with pytest.raises(FileNotFoundError, match="CPU-prepared"):
        data.load(*paths)
    artifact = data.prepare(*paths)
    monkeypatch.setattr(AmazonReviewDataModule, "setup", lambda *args: pytest.fail("unexpected raw rebuild"))
    assert data.prepare(*paths) == artifact
    dm = data.load(*paths, num_workers=0)
    assert dm.train_dataset is dm.val_dataset is None
    assert dm.train_item_counts_by_raw_id == original.train_item_counts_by_raw_id == {"A": 4, "B": 2, "C": 2}
    assert dm.item2id == original.item2id and dm.semantic_ids == original.semantic_ids
    assert dm.test_dataset.examples.target_timestamp.tolist() == [21, 21]
    for old, new in zip(original.test_dataloader(), dm.test_dataloader(), strict=True):
        assert new.keys() == old.keys()
        for key in old:
            if torch.is_tensor(old[key]):
                torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
            else:
                assert new[key] == old[key]
