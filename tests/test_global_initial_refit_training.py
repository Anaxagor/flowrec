"""Initial-model refits train a fresh student up to the selected epoch and score only its terminal weights."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/global-time"))
import refit_initial_training as refit


@pytest.mark.parametrize("family,dataset", [("sasrec", "beauty_2023"), ("tiger", "beauty_2023"),
    ("gpt2", "beauty_2023"), ("gpt2", "yambda_500m_listens"), ("sasrec_sce", "yambda_500m_listens")])
def test_initial_refit_and_recover_final_test(tmp_path, monkeypatch, family, dataset):
    # More than 20 warm items so the real ranking evaluators can score the complete catalog.
    reviews, meta, sids = (tmp_path / name for name in ("reviews.parquet", "metadata.parquet", "sid.json"))
    rows = [dict(user_id=f"user-{u}", parent_asin=str((i + u) % 24), timestamp=1000 + i,
                 rating=5., verified_purchase=True) for u in range(4) for i in range(32)]
    pd.DataFrame(rows).to_parquet(reviews)
    pd.DataFrame({"parent_asin": [str(i) for i in range(24)]}).to_parquet(meta)
    sids.write_text(json.dumps({str(i): [i // 8, i % 8] for i in range(24)}))
    params = dict(lr=.001, weight_decay=.01, dropout=0., num_layers=1)
    if family.startswith("sasrec"):
        params.update(d_model=8, nhead=2, pos_drop=0.)
        if family == "sasrec_sce":
            params.update(n_buckets_mult=.25, bucket_size_x_mult=.25, bucket_size_y=4, mix_x=False)
    else:
        params.update(embedding_dim=8)
        params.update(dict(num_heads=2, ff_mult=2) if family == "gpt2" else dict(d_kv=4))
    spec = dict(model=family, dataset=dataset, params=params, seed=43,
                trial="trial_0001", best_epoch=1, epochs=2, validation_ndcg10=.4)
    builder = refit._sasrec if family.startswith("sasrec") else getattr(refit, "_" + family)
    recipe = builder(spec, 0, "cpu")
    if dataset.startswith("yambda"):
        events = pd.DataFrame(rows).rename(columns={"user_id": "uid", "parent_asin": "item_id"})
        events["event_type"], events["played_ratio_pct"] = "listen", 90
        events.to_parquet(reviews)
        pd.DataFrame({"item_id": list(range(24))}).to_parquet(meta)
        files = dict(interactions_path=reviews, embeddings_path=meta, hf_recent_months=None,
                     train_window_len=3)
    else:
        files = dict(reviews_path=reviews, metadata_path=meta, item_key="parent_asin", processed_cache_dir=None)
    cfg = replace(recipe.data_config, **files, semantic_id_path=None if family.startswith("sasrec") else sids,
                  batch_size=32, test_batch_size=4, global_val_timestamp=1025, global_test_timestamp=1030,
                  min_user_interactions=1, min_item_interactions=1, max_history_len=3)
    monkeypatch.setattr(refit, "_recipe", lambda *args: (replace(recipe, data_config=cfg), tmp_path / "data"))
    out = tmp_path / "output"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        refit.prepare_initial(spec, tmp_path)
        score = refit._test
        def interrupted(recipe, model, dm, device):
            assert not model.training
            state = torch.load(out / "checkpoints/last.pt", map_location="cpu", weights_only=True)
            assert state["trainer_state"]["epoch"] == 1 and state["epoch_progress"]["epoch_complete"]
            # Validation is never run; no checkpoint selection has taken place during the refit.
            assert state["trainer_state"]["best_metric"] is None
            raise RuntimeError("simulated eviction during test")
        monkeypatch.setattr(refit, "_test", interrupted)
        with pytest.raises(RuntimeError, match="simulated eviction"):
            refit.run_initial(spec, tmp_path, out, device="cpu", num_workers=0)
        last = out / "checkpoints/last.pt"
        modified = last.stat().st_mtime_ns
        assert not (out / "checkpoints/best.pt").exists()
        assert not (out / "result.json").exists()
        monkeypatch.setattr(refit, "_test", score)
        # A copied JSON can provide extra precision after the trial was selected from its YAML.
        precise = dict(spec, validation_ndcg10=.40000037)
        result = refit.run_initial(precise, tmp_path, out, device="cpu", num_workers=0)
        assert last.stat().st_mtime_ns == modified
        assert result["epochs_completed"] == 2
        assert result["seed"] == spec["seed"]
        assert result["validation_ndcg10"] == precise["validation_ndcg10"]
        assert result["test_metrics"]["num_examples"] == 4
        assert {f"{metric}@{k}" for metric in ("HR", "NDCG", "Coverage", "ARP")
                for k in (5, 10, 20)} <= result["test_metrics"].keys()
        assert refit.run_initial(spec, tmp_path, out, device="cpu", num_workers=0) == result
    finally:
        torch.set_num_threads(previous_threads)
