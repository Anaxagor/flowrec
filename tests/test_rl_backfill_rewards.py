from types import SimpleNamespace
import json

import pandas as pd
import torch

from flowrec.data.interaction_store import InteractionLogStore
from flowrec.models.sasrec import SASRec
from flowrec.trajectories.semantic import SemanticTrajectoryBatch
from scripts.rl_backfill_rewards import build_backfill_reward


def test_grpo_product_uses_generated_items_in_teacher_order_and_training_window(tmp_path):
    checkpoint = tmp_path / "checkpoints/exp2/global_split/initial/sasrec/beauty_2023/best.pt"
    checkpoint.parent.mkdir(parents=True)
    teacher = SASRec(dict(item_vocab_size=4, d_model=8, nhead=2, num_layers=1, max_history_len=2)).eval()
    torch.save({"model_config": teacher.config_dict(), "model_state_dict": teacher.state_dict()}, checkpoint)
    torch.save({"quantizer.codebooks": torch.tensor([[[0.], [1.]], [[0.], [1.]]])}, tmp_path / "model.pt")
    (tmp_path / "semantic_id_metadata.json").write_text(json.dumps(
        {"rqvae_config": {"codebook_size": 2}, "level_offset_tokens": True}))
    # SASRec observed all three pre-filter items. The student observes items
    # 1 and 3 only. Its ID 2 must score SASRec item 3, including in histories.
    events = pd.DataFrame({"user_id": [0, 0, 0], "item_id": [1, 0, 2],
                           "timestamp": [1, 2, 3], "position": [0, 1, 2], "item_id_prefilter": [1, 2, 3]})
    reviews = pd.DataFrame({"user_id": [0, 0], "item_id": [2, 2], "timestamp": [8, 10],
                            "event_type": ["review", "review"], "rating": [2., 5.]})
    data = SimpleNamespace(cfg=SimpleNamespace(semantic_id_path=tmp_path / "semantic_ids.json",
                                               split_strategy="global_time", offset_semantic_ids_by_one=True),
                           num_users=1, item_vocab_size=3, global_val_cutoff=10, interactions=events,
                           item2id={"a": 2, "b": 1}, semantic_ids={"a": [1, 4], "b": [1, 3]},
                           interaction_store=InteractionLogStore(reviews, num_users=1, item_vocab_size=3))
    method = build_backfill_reward("S×H×I", dataset="beauty_2023", split="global", data=data, input_root=tmp_path)
    tokens = torch.tensor([[[1, 3], [1, 4]]])
    trajectories = SemanticTrajectoryBatch(tokens, torch.ones_like(tokens, dtype=torch.bool),
                                          torch.tensor([[True, False]]), torch.tensor([[1, 0]]))
    batch = {"history_item_ids": torch.tensor([[0, 2]]), "history_mask": torch.tensor([[False, True]]),
             "target_token_ids": tokens[:, 0], "target_item_ids": torch.tensor([1]),
             "user_ids": torch.tensor([0]), "target_timestamps": torch.tensor([11])}
    sasrec, similarity, interaction = method.components
    expected = teacher.score_items({**batch, "history_item_ids": torch.tensor([[0, 3]])}, torch.tensor([[1, 3]]))
    torch.testing.assert_close(sasrec(batch, trajectories), torch.nn.functional.softplus(expected))
    torch.testing.assert_close(similarity(batch, trajectories), torch.tensor([[1., .5]]))
    assert interaction.store.events["timestamp"].tolist() == [8]
    torch.testing.assert_close(method(batch, trajectories),
                               sasrec(batch, trajectories) * similarity(batch, trajectories) * interaction(batch, trajectories))


def test_loo_teacher_uses_ids_before_student_cold_item_filter(tmp_path):
    checkpoint = tmp_path / "checkpoints/exp2/loo_split/initial/sasrec_sce/yambda_500m_listens/best.pt"
    checkpoint.parent.mkdir(parents=True)
    teacher = SASRec(dict(item_vocab_size=4, d_model=8, nhead=2, num_layers=1, max_history_len=2))
    torch.save({"model_config": teacher.config_dict(), "model_state_dict": teacher.state_dict()}, checkpoint)
    stats = dict(name="yambda_500m_listens", num_items_before_cold_filter=3, num_interactions=3,
                 num_users=1, num_cold_items_removed=0, positive_criterion="listen", split_strategy="leave_one_out")
    (checkpoint.parent / "training_summary.json").write_text(json.dumps({"runs": [{"dataset": stats}]}))
    data = SimpleNamespace(cfg=SimpleNamespace(split_strategy="leave_one_out"), item_vocab_size=3,
                           summary=lambda: stats, item2id={"a": 1, "b": 2}, semantic_ids={"a": [1], "b": [2]},
                           interactions=pd.DataFrame({"item_id": [0, 1, 2], "item_id_prefilter": [1, 2, 3]}))
    reward = build_backfill_reward("S", dataset=stats["name"], split="loo", data=data, input_root=tmp_path)
    assert reward.sasrec_model.translation.tolist() == [0, 2, 3]
