"""Check the global SASRec score columns against an independently built dense split."""
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import pytest

from flowrec.data.post_split_filtering import filter_splits_to_train_item_catalog, remap_interactions_to_catalog
from flowrec.data.sequence_builder import SequenceSplitConfig, build_sequence_splits, factorize_interaction_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
from yambda_gflow_teacher import dense_teacher_ids


@pytest.mark.parametrize("window", [2, 50])
def test_raw_teacher_ids_match_original_training_layout(window):
    rows = []
    for user, tracks, times in [
        ("a", ["early", "a1", "a2", "a3", "a4", "a5", "late"], [1, 2, 3, 4, 5, 6, 11]),
        ("b", ["singleton", "val", "test"], [2, 10, 20]),
        ("c", ["late", "b1", "early", "another"], [3, 6, 7, 22]),
    ]:
        for pos, (item, stamp) in enumerate(zip(tracks, times)):
            rows.append(dict(raw_user_id=user, raw_item_id=item, timestamp=stamp, order_idx=pos,
                             rating=1.0, verified_purchase=False))
    events, users, _, _, raw_catalog = factorize_interaction_ids(pd.DataFrame(rows))
    student_cfg = SequenceSplitConfig(split_strategy="global_time", global_val_timestamp=10,
        global_test_timestamp=20, max_history_len=2, train_target_policy="last_n", train_last_n=1)
    teacher_cfg = replace(student_cfg, train_example_layout="dense_causal", max_history_len=window,
                          train_target_policy="all")
    def catalog(cfg):
        return filter_splits_to_train_item_catalog(build_sequence_splits(events, cfg=cfg),
                                                   old_id2item=raw_catalog)
    student, teacher = catalog(student_cfg), catalog(teacher_cfg)
    assert set(teacher.item2id) - set(student.item2id)
    assert "singleton" not in teacher.item2id
    table, teacher_size = dense_teacher_ids(remap_interactions_to_catalog(events, student.item2id),
        val_cutoff=10, student_vocab_size=len(student.id2item), num_users=len(users))
    assert teacher_size == len(teacher.id2item)
    assert table.tolist() == [0, *[teacher.item2id[item] for item in student.id2item[1:]]]
