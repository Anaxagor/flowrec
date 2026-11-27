"""A prepared RL cell must reproduce ranking batches and the histories used for rewards."""
import json

import pandas as pd
import pytest
import torch

from scripts import rl_backfill_inputs as inputs
from scripts import run_tiger


@pytest.mark.parametrize("native", [False, True])
def test_shared_cache_preserves_heldout_and_reward_events(tmp_path, monkeypatch, native):
    if native:
        compact = inputs.import_module("scripts.exp7-ce-distillation.ce_yambda_prepared_data")
        monkeypatch.delattr(compact, "PreparedAmazonDataModule", raising=False)
    reviews = tmp_path / "reviews.parquet"
    metadata = tmp_path / "metadata.parquet"
    sids = tmp_path / "sids.json"
    checkpoint = tmp_path / "best.pt"
    rows = [dict(user_id=str(u), parent_asin=str((u + t) % 9), timestamp=t,
                 rating=5., verified_purchase=True) for u in range(3) for t in range(20)]
    rows += [{**rows[t], "verified_purchase": False, "rating": 2.} for t in (1, 12, 18)]
    pd.DataFrame(rows).to_parquet(reviews)
    pd.DataFrame({"parent_asin": [str(i) for i in range(9)]}).to_parquet(metadata)
    sids.write_text(json.dumps({str(i): [i, 16 + i] for i in range(9)}))

    def tiny(split, model, dataset, *, batch_size, eval_batch_size, num_workers, **_):
        config = {"data": {"source": "amazon", "reviews_path": str(reviews), "metadata_path": str(metadata),
                           "semantic_id_path": str(sids)},
                  "dataset": {"min_user_interactions": 2, "min_item_interactions": 1, "item_key": "parent_asin",
                              "min_history_len": 1, "max_history_len": 4, "split_strategy": "global_time",
                              "global_val_timestamp": 12, "global_test_timestamp": 16,
                              "global_eval_target_mode": "last_per_user", "train_target_policy": "all",
                              "positive_criterion": "verified_purchase == True", "metadata_ids_only": True},
                  "dataloader": {"train_batch_size": batch_size, "validation_batch_size": eval_batch_size,
                                 "num_workers": num_workers, "pin_memory": False, "include_raw_in_batch": False},
                  "model": {"num_heads": 2, "d_kv": 4, "embedding_dim": 8, "codebook_size": 16,
                            "dim_feedforward": 16, "num_decoder_layers": 1, "num_encoder_layers": 1,
                            "use_user_token": False, "user_ids_count": 0}, "seed": 42}
        return checkpoint, config, {"num_val_examples": 3, "num_test_examples": 3}

    monkeypatch.setattr(inputs, "input_config", tiny)
    kwargs = dict(batch_size=3, eval_batch_size=3, num_workers=0)
    cell = ("global", "tiger", "beauty_2023")
    _, config, _ = tiny(*cell, **kwargs)
    native = run_tiger.build_data_module(run_tiger.build_data_config(config, config_path=checkpoint))
    native.setup()
    model = run_tiger.build_model(config, native, config_path=checkpoint)
    torch.save({"model_type": "tiger", "model_config": model.config_dict(),
                "model_state_dict": model.state_dict()}, checkpoint)
    raw = inputs.load_inputs(*cell, **kwargs).data
    marker = inputs.prepare_inputs_cache(*cell, cache_root=tmp_path)
    monkeypatch.setattr(inputs, "_open_raw", lambda *_: pytest.fail("prepared worker rebuilt raw input"))
    assert inputs.prepare_inputs_cache(*cell, cache_root=tmp_path) == marker
    cached = inputs.load_inputs(*cell, **kwargs, cache_root=tmp_path, reward="S×H×I").data
    assert set(cached.interactions.columns) >= {"position", "item_id_prefilter", "timestamp"}
    pd.testing.assert_frame_equal(cached.interactions, raw.interactions[cached.interactions.columns])
    pd.testing.assert_frame_equal(cached.interaction_store.events, raw.interaction_store.events, check_dtype=False)
    for method in ("train_dataloader", "val_dataloader", "test_dataloader"):
        original = next(iter(getattr(raw, method)()))
        prepared = next(iter(getattr(cached, method)()))
        for key in original:
            if torch.is_tensor(original[key]):
                torch.testing.assert_close(original[key], prepared[key])
    without_rewards = inputs.load_inputs(*cell, **kwargs, cache_root=tmp_path).data
    assert without_rewards.interactions.empty and without_rewards.interaction_store is None


def test_global_yambda50_uses_the_two_distinct_test_catalogs():
    kwargs = dict(batch_size=3, eval_batch_size=3, num_workers=0)
    gpt, gc, gs = inputs.input_config("global", "gpt2", inputs.Y50, **kwargs)
    tiger, tc, ts = inputs.input_config("global", "tiger", inputs.Y50, **kwargs)
    assert gpt.parent.name == "full_history" and gs["num_test_examples"] == 7044
    assert tiger.parent.name == inputs.Y50 and ts["num_test_examples"] == 6868
    assert gc["dataset"]["train_prefix_from_dense"] and not tc["dataset"]["train_prefix_from_dense"]


def test_global_hpo_backfill_can_rebuild_profiles_from_promoted_metadata(monkeypatch):
    read = inputs._read
    def require_promoted(path):
        assert path.is_relative_to(inputs.ROOT / "checkpoints/exp2/global_split/initial")
        return read(path)
    monkeypatch.setattr(inputs, "_read", require_promoted)
    kwargs = dict(batch_size=3, eval_batch_size=3, num_workers=0)
    path, config, summary = inputs.input_config("global", "tiger", inputs.BEAUTY, **kwargs)
    assert path.name == "trial_31.pt" and config["model"]["d_kv"] == 64
    assert summary["num_test_examples"] == 158576
    assert config["dataset"]["train_target_policy"] == "all"
    assert not config["dataset"]["train_prefix_from_dense"]
    path, config, summary = inputs.input_config("global", "gpt2", inputs.Y500, **kwargs)
    assert path.name == "trial_17.pt" and config["model"]["dropout"] == pytest.approx(0.2605724812002705)
    assert summary["num_test_examples"] == 70663
    assert config["dataset"]["train_last_n"] == 128
    assert not config["dataset"]["train_prefix_from_dense"]
