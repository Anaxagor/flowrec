"""Restore interaction evidence without changing matched checkpoint inputs."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "scripts/exp2-tuned-params",
             ROOT / "scripts/exp4-reward-ablations"):
    sys.path.insert(0, str(path))

import gpt2_matched_finetune as selected  # noqa: E402
import gflow_finetune_gpt2_amazon as experiment  # noqa: E402
import tuning_gpt2 as tuning  # noqa: E402
from flowrec.data.dense_prefix import dense_prefix_signature  # noqa: E402
from flowrec.data.interaction_store import partition_interactions  # noqa: E402
from flowrec.data.yambda import (  # noqa: E402
    YambdaDataConfig, YambdaDataModule, normalize_yambda_flat_frame,
)


@pytest.mark.parametrize("allowed", [["listen"], ["listen", "like"], None])
@pytest.mark.parametrize("criterion", [None, "played_ratio_pct >= 10 or event_type == 'like'"])
def test_converted_predicate_preserves_normalizer_selection_with_missing_ratios(allowed, criterion):
    raw = pd.DataFrame({
        "uid": [1] * 7, "item_id": list(range(7)), "timestamp": list(range(7)),
        "event_type": ["listen", "listen", "listen", "listen", "like", "dislike", "unlike"],
        "played_ratio_pct": [100., 50., 49., None, None, None, None],
    })
    config = dict(allowed_event_types=allowed, min_played_ratio_pct=50., positive_criterion=criterion)
    original = normalize_yambda_flat_frame(raw, allowed_event_types=allowed, min_played_ratio_pct=50.)
    original, _ = partition_interactions(original, criterion)
    _, converted = selected.input_config({"data_config": config, "matched_data_audit": {}}, ROOT)
    all_events = normalize_yambda_flat_frame(raw)
    positives, auxiliary = partition_interactions(all_events, converted["positive_criterion"])
    cols = ["raw_user_id", "raw_item_id", "timestamp", "order_idx"]
    pd.testing.assert_frame_equal(positives[cols], original[cols])
    assert len(positives) + len(auxiliary) == len(raw)
    assert converted["allowed_event_types"] is None and converted["min_played_ratio_pct"] is None
    assert config["min_played_ratio_pct"] == 50.  # Saved tuning protocol is immutable.


def test_auxiliary_restoration_keeps_dense_audit_and_cache_subprocess_inputs(tmp_path):
    epoch = int(pd.Timestamp("2026-08-15", tz="UTC").timestamp())
    rows = []
    for user, length in ((1, 145), (2, 12)):
        for index in range(length):
            base = dict(uid=user, item_id=index % 13, timestamp=epoch + index * 100,
                        is_organic=True)
            rows.append({**base, "event_type": "listen", "played_ratio_pct": 100.})
            if index % 5 == 0:
                rows.extend([
                    {**base, "event_type": "listen", "played_ratio_pct": 25.},
                    {**base, "event_type": "like", "played_ratio_pct": None},
                    {**base, "event_type": "dislike", "played_ratio_pct": None},
                ])
    # A newer auxiliary event must not move the date-window anchor when the
    # filtering strategy changes: both loaders anchor before event filtering.
    rows.append(dict(uid=1, item_id=0, timestamp=epoch + 86400 * 25,
                     event_type="like", played_ratio_pct=None, is_organic=True))
    rows.append(dict(uid=1, item_id=0, timestamp=epoch - 86400 * 100,
                     event_type="listen", played_ratio_pct=100., is_organic=True))
    path = tmp_path / "multi_event.parquet"
    pd.DataFrame(rows).to_parquet(path)
    sid_path = tmp_path / "semantic_ids.json"
    sid_path.write_text(json.dumps({str(i): [i + 1, i + 20] for i in range(13)}))
    config = YambdaDataConfig(
        interactions_path=path, semantic_id_path=sid_path, event_file="multi_event",
        hf_recent_months=2, allowed_event_types=("listen",), min_played_ratio_pct=50.,
        positive_criterion=None, min_user_interactions=5, min_item_interactions=0,
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_target_policy="last_n", train_last_n=128, max_history_len=50,
        train_window_len=50, train_window_stride=1, history_padding="right",
        num_workers=0, persistent_workers=False, offset_semantic_ids_by_one=False,
    )
    dense = YambdaDataModule(config)
    dense.setup()
    assert dense.interaction_store.events.empty
    expected = tuning.audit_matched_data(dense)
    saved = {"data_config": asdict(config), "matched_data_audit": expected}
    data, values = selected.input_config(saved, ROOT)
    old_partition = {**values, "positive_criterion": None,
                     "allowed_event_types": ["listen"], "min_played_ratio_pct": 50.}
    assert dense_prefix_signature(values) != dense_prefix_signature(old_partition)
    assert dense_prefix_signature(values)["data_audit"] == dense_prefix_signature(old_partition)["data_audit"]
    run_config = {
        "experiment_name": "trial36", "data": data, "dataset": values,
        "dataloader": {"num_workers": 0, "persistent_workers": False},
        "model": {"use_user_token": False},
        "method": {"component_keys": ["interaction"], "sasrec": {},
                   "reward": {"components": [{"name": "interaction", "reward": {}}]}},
    }
    # Exactly the JSON consumed by the separate factor-cache precompute process.
    cache_config = experiment.cache_config(run_config, "interaction")
    serialized = json.loads(json.dumps(cache_config, default=str))
    cfg = experiment.tiger_runner.build_data_config(serialized, config_path=tmp_path / "cache.json")
    prefix = YambdaDataModule(cfg)
    prefix.setup()
    result = selected.audit_prefix_data(prefix, saved)
    assert result["all_fingerprints_equal"]
    assert prefix.item2id == dense.item2id and prefix.semantic_ids == dense.semantic_ids
    assert prefix.train_dataset.examples.groupby("user_id").size().tolist() == [128, 9]
    assert prefix.train_dataset.examples.history_item_ids.str.len().max() == 50
    for split in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(prefix, split).examples, getattr(dense, split).examples)
    aux = prefix.interaction_store.events
    assert set(aux.event_type) == {"listen", "like", "dislike"}
    assert (aux.loc[aux.event_type.eq("listen"), "played_ratio_pct"] == 25.).all()
    assert saved["data_config"]["positive_criterion"] is None
    assert saved["data_config"]["allowed_event_types"] == ("listen",)
