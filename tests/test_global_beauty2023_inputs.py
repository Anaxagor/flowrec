"""The SASRec catalog may have the right size and still use the wrong item IDs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/global-time"))
from beauty2023_global_inputs import Beauty2023Profile


@pytest.mark.parametrize("student_ids", [[1, 2], [2, 1]])
def test_teacher_orders_observed_items_from_dense_training_windows(tmp_path, student_ids):
    teacher = tmp_path / "teacher.pt"
    torch.save({"model_config": {"item_vocab_size": 3},
                "model_state_dict": {"item_embedding.weight": torch.zeros(3, 2)}}, teacher)
    profile = SimpleNamespace(teacher_checkpoint=teacher)
    # Original item 2 belongs only to a user with no second pre-cutoff event.
    # Original items 1 and 3 become teacher IDs 1 and 2 respectively.
    events = pd.DataFrame({"user_id": [0, 0, 1, 1], "position": [0, 1, 0, 1],
                           "timestamp": [1, 2, 1, 10], "item_id_prefilter": [1, 3, 2, 3],
                           "item_id": [*student_ids, 0, student_ids[1]]})
    dm = SimpleNamespace(interactions=events, global_val_cutoff=10, item_vocab_size=3,
                         num_users=2, item2id={"a": student_ids[0], "b": student_ids[1]})
    destination = tmp_path / "translation.json"
    if student_ids == [2, 1]:
        with pytest.raises(ValueError, match="item catalogs differ"):
            Beauty2023Profile.verify_teacher_catalog(profile, dm, destination)
        assert not destination.exists()
    else:
        Beauty2023Profile.verify_teacher_catalog(profile, dm, destination)
        record = json.loads(destination.read_text())
        assert record["item_id_translation"] == [0, 1, 2] and record["teacher_item_vocab_size"] == 3
