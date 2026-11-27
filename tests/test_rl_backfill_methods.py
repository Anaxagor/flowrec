import math

import pytest
import torch
from torch import nn

from flowrec.ft_methods.rl_backfill import OnPolicyBackfillMethod
from flowrec.models.gpt2 import GPT2RecModel
from flowrec.models.tiger import TigerModel
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class PositiveReward(nn.Module):
    def forward(self, batch, trajectories):
        assert not self.training
        items = trajectories.extra["candidate_item_ids"]
        assert items.shape == (2, 17)
        torch.testing.assert_close(items[:, 0], batch["target_item_ids"])
        assert all(len(set(group)) == 17 for group in items.tolist())
        assert trajectories.candidate_token_mask.all()
        assert trajectories.extra["policy_fallback_fraction"] == 0
        return trajectories.is_positive.float()


@pytest.mark.parametrize("model_type", ["tiger", "gpt2"])
@pytest.mark.parametrize("algorithm", ["grpo", "sdpo"])
def test_backfill_differentiates_policy_on_generated_group(model_type, algorithm):
    torch.manual_seed(11)
    catalog = torch.tensor([[1 + i // 8, 5 + (i // 4) % 2, 9 + i % 4] for i in range(20)])
    kwargs = dict(
        semantic_id_mapping={f"item_{i:02}": row.tolist() for i, row in enumerate(catalog)},
        codebook_size=4, sem_id_len=3, embedding_dim=8, num_heads=2, dim_feedforward=16,
        dropout=0.3,
    )
    if model_type == "tiger":
        model = TigerModel(
            **kwargs, num_encoder_layers=1, num_decoder_layers=1, d_kv=4, use_user_token=False, user_ids_count=0,
        )
    else:
        model = GPT2RecModel(**kwargs, num_layers=1, max_position_embeddings=32)
    # The last two tokenizer rows are outside the split's item mapping.
    mapping = torch.tensor([-1, *range(18)])
    batch = {
        "user_ids": torch.tensor([0, 1]),
        "history_token_ids": catalog[torch.tensor([[2, 3], [5, 6]])],
        "history_token_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        "target_item_ids": torch.tensor([1, 5]),
        "target_token_ids": catalog[[0, 4]],
        "target_token_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    method = OnPolicyBackfillMethod(
        model, algorithm=algorithm, item_id_to_semantic_row=mapping,
        reward_model=PositiveReward() if algorithm == "grpo" else None,
    )
    model.train()
    result = method.training_step(model, batch)
    assert model.training and not method.reference.training
    assert result.loss.item() == pytest.approx(0.0 if algorithm == "grpo" else math.log(17), abs=1e-6)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in method.reference.parameters())
    if algorithm == "grpo":
        mapped = catalog[:18].expand(2, -1, -1)
        candidates = SemanticTrajectoryBatch(
            candidate_token_ids=mapped, candidate_token_mask=torch.ones_like(mapped, dtype=torch.bool),
            is_positive=mapped.eq(batch["target_token_ids"][:, None]).all(dim=-1),
        )
        with torch.no_grad():
            probabilities = method._token_logprobs(method.reference, batch, candidates).sum(-1).exp()
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
