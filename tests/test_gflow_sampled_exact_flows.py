import pytest
import torch

from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class FakeTokenModel:
    item_semantic_ids = torch.tensor([[1, 1], [1, 2], [2, 1]], dtype=torch.long)

    def score_item_tokens_valid_actions(self, batch, *, candidate_token_ids, candidate_token_mask):
        return torch.zeros_like(candidate_token_ids, dtype=torch.float32)


@pytest.mark.parametrize("objective", ["db", "subtb"])
def test_sampled_trajectories_use_full_catalogue_analytical_flows(objective: str):
    model = FakeTokenModel()
    method = GFlowMethod(
        GFlowConfig(
            objective=objective,
            tb_logz_source="reward_cache",
            db_activation_checkpointing=False,
        )
    )
    full_rows = torch.tensor([[0, 1, 2]])
    full_rewards = torch.tensor([[3.0, 5.0, 7.0]])
    sampled_tokens = model.item_semantic_ids[torch.tensor([[0, 2]])]
    trajectories = SemanticTrajectoryBatch(
        candidate_token_ids=sampled_tokens,
        candidate_token_mask=torch.ones_like(sampled_tokens, dtype=torch.bool),
        is_positive=torch.zeros(1, 2, dtype=torch.bool),
        extra={"cache_top_rows": full_rows, "cache_top_rewards": full_rewards},
    )

    _, actual = method._candidate_token_logprobs_and_prefix_flows(
        model, {}, trajectories, torch.tensor([[3.0, 7.0]])
    )
    expected = method._get_exact_flow_index(model).prefix_log_flows(
        sampled_tokens,
        method._get_exact_flow_index(model).dense_rewards(
            full_rows, full_rewards, reward_epsilon=1e-6
        ),
        reward_epsilon=1e-6,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, :, 0].exp(), torch.tensor([[15.0, 15.0]]))
