"""Score the exact archived weights on every test row through the evaluation CLI."""
from dataclasses import asdict
import hashlib
from importlib import import_module
import json

import pandas as pd
import pytest
import torch

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule

job = import_module("scripts.global-time.evaluate_gpt2_beauty2023_best")


def test_stored_best_full_test(tmp_path, monkeypatch):
    source = tmp_path / "initial"
    source.mkdir()
    reviews = tmp_path / "reviews.parquet"
    metadata = tmp_path / "metadata.parquet"
    sids = tmp_path / "sids.json"
    rows = [dict(user_id=str(user), parent_asin=str((user + t) % 24), timestamp=t,
                 rating=5., verified_purchase=True) for user in range(3) for t in range(34)]
    pd.DataFrame(rows).to_parquet(reviews)
    pd.DataFrame({"parent_asin": list(map(str, range(24)))}).to_parquet(metadata)
    sids.write_text(json.dumps({str(i): [i // 6, 6 + i % 6] for i in range(24)}))
    monkeypatch.setattr(job.data, "SID_SHA256", hashlib.sha256(sids.read_bytes()).hexdigest())
    cfg = AmazonReviewDataConfig(
        reviews_path=str(reviews), metadata_path=str(metadata), semantic_id_path=str(sids),
        item_key="parent_asin", positive_criterion="verified_purchase == True", split_strategy="global_time",
        global_val_timestamp=25, global_test_timestamp=30, global_eval_target_mode="last_per_user",
        train_example_layout="dense_causal", max_history_len=4, history_padding="right",
        num_workers=0, pin_memory=False, eval_batch_size=2, include_raw_in_batch=False)
    dm = AmazonReviewDataModule(cfg)
    dm.setup()
    trial = dict(best_trial_number=0, best_epoch=3, best_value_val_generation_ndcg10=.03,
                 metric=job.MONITOR, dataset="beauty_2023",
                 model=dict(embedding_dim=8, codebook_size=12, num_heads=2, num_layers=1, dim_feedforward=16))
    study = dict(monitor=job.MONITOR, data_config=asdict(cfg), regime={"model_config": {}})
    model = job.gpt2.build_model(study["regime"]["model_config"], trial, dm)
    best = source / "best.pt"
    torch.save(dict(model_type="gpt2", method_name="sft", model_state_dict=model.state_dict(),
                    model_config=model.config_dict(), trainer_config={"monitor": job.MONITOR},
                    trainer_state={"epoch": 3, "best_epoch": 3, "best_metric": .03}), best)
    hashes = {}
    for name, meta in (("study_config.json", study), ("best_trial.json", trial), ("data_summary.json", dm.summary())):
        (source / name).write_text(json.dumps(meta))
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    record = dict(model="gpt2", dataset="beauty_2023", split="global_split", stage="initial", seed=42, trial=0,
                  checkpoint=str((job.SOURCE / "best.pt").relative_to(job.ROOT)),
                  sha256=hashlib.sha256(best.read_bytes()).hexdigest(), metadata_sha256=hashes)
    (source / "provenance.json").write_text(json.dumps(record))
    common = ["--source-dir", str(source), "--prepared-dir", str(tmp_path / "prepared"),
              "--scratch-dir", str(tmp_path / "scratch")]
    job.main(["prepare", *common])
    def forbidden(*args, **kwargs):
        pytest.fail("An archived-checkpoint test may only iterate the test loader")
    monkeypatch.setattr(AmazonReviewDataModule, "train_dataloader", forbidden)
    monkeypatch.setattr(AmazonReviewDataModule, "val_dataloader", forbidden)
    oracle = job.gpt2.evaluate_generation_ranking_metrics(
        model.eval(), dm.test_dataloader(), device="cpu", ks=(5, 10, 20), num_beams=20,
        constrain_valid_sequences=True, lookup=job.gpt2.SemanticIdItemLookup(dm.semantic_ids, dm.item2id),
        item_popularity=job.runner.training_item_popularity(dm), num_catalog_items=dm.num_items)
    job.main(["evaluate", *common, "--device", "cpu", "--test-batch-size", "2"])
    result = json.loads((source / "test_metrics.json").read_text())
    assert result["test_metrics"] == oracle
    assert result["expected_test_examples"] == oracle["num_examples"] == 3
    assert result["best_epoch_one_based"] == 4
    assert result["checkpoint_sha256"] == record["sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()
    assert set(oracle) == {"num_examples"} | {f"{m}@{k}" for m in ("HR", "NDCG", "Coverage", "ARP")
                                               for k in (5, 10, 20)}

    selected = job.load_source(source, tmp_path / "scratch")
    selected[-1]["model_state_dict"]["item_semantic_ids"] = model.item_semantic_ids.flip([0])
    with pytest.raises(ValueError, match="semantic catalog"):
        job.evaluate(dm, selected, "cpu")
    (source / "best_trial.json").write_text(json.dumps({**trial, "best_epoch": 4}))
    with pytest.raises(ValueError, match="checksum"):
        job.main(["evaluate", *common, "--device", "cpu"])
