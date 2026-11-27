"""The dropout-0 continuation sees every original TIGER training prefix before validation."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
import tiger_yambda500m_sft_data as continuation


def test_continuation_scratch_preserves_original_batches_and_cutoff(tmp_path, monkeypatch):
    # Both user sequences are longer than 128 targets before validation.
    val, test = continuation.CUTOFFS
    times = [*range(val - 150, val), val, test]
    rows = [dict(uid=u, timestamp=t, item_id=(i + u) % 7, event_type="listen", played_ratio_pct=90)
            for u in (1, 2) for i, t in enumerate(times)]
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    paths = {key: tmp_path / name for key, name in continuation.ASSETS.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(paths["interactions_path"])
    pd.DataFrame({"item_id": range(7)}).to_parquet(paths["embeddings_path"])
    paths["semantic_id_path"].write_text(json.dumps({str(i): [i + 1, i + 2] for i in range(7)}))
    cfg = YambdaDataConfig(**paths, dataset_size="500m", event_file="multi_event", hf_recent_months=None,
            global_val_timestamp=val, global_test_timestamp=test, split_strategy="global_time",
            global_eval_target_mode="last_per_user", train_target_policy="last_n", train_last_n=128,
            max_history_len=50, min_user_interactions=1, min_item_interactions=1,
            positive_criterion="event_type == 'listen' and played_ratio_pct >= 50", user_hash_mod=1,
            include_raw_in_batch=True, batch_size=13, eval_batch_size=1, num_workers=0, persistent_workers=False)
    original = YambdaDataModule(cfg)
    original.setup()
    counts = {"train": 2 * 128, "val": 2, "test": 2}
    monkeypatch.setattr(continuation, "COUNTS", counts)
    monkeypatch.setattr(continuation, "NUM_ITEMS", 7)
    archived = asdict(replace(cfg, **{k: str(tmp_path / "previous-checkout" / rel)
                                      for k, rel in continuation.ASSETS.items()}))
    workspace = tmp_path / "ws"
    continuation.prepare(workspace, archived)
    def prohibit_rebuild(*args):
        pytest.fail("A complete cache must load without rebuilding examples")
    monkeypatch.setattr(YambdaDataModule, "setup", prohibit_rebuild)
    continuation.prepare(workspace, archived)
    dm = continuation.load(workspace, archived, num_workers=0)
    assert dm.semantic_ids == original.semantic_ids and dm.item2id == original.item2id
    assert dm._ce_prepared_audit["latest_training_target"] == val - 1
    assert dm._ce_prepared_source_indices.tolist() == list(range(256))
    assert dm.train_dataset[0]["target_timestamp"] == val - 128
    assert dm.train_dataset[127]["target_timestamp"] == val - 1
    assert [dm.val_dataset[i]["target_timestamp"] for i in range(len(dm.val_dataset))] == [val, val]
    for reference, cached in ((original.train_dataloader(), dm.train_dataloader()),
                              (original.val_dataloader(), dm.val_dataloader())):
        assert cached.batch_size == reference.batch_size
        for old, new in zip(reference, cached, strict=True):
            for key in old:
                if torch.is_tensor(old[key]):
                    torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="original TIGER global"):
        continuation.load(workspace, dict(archived, train_last_n=50))
