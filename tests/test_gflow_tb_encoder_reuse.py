import torch

from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.models.base import RecModelOutput
from flowrec.trajectories.semantic import SemanticTrajectoryBatch


class FakeSFT(torch.nn.Module):
    def compute_loss(self, model, batch, *, stage="train", step=None):
        return MethodOutput(
            loss=torch.tensor(0.5),
            model_output=RecModelOutput(
                logits=torch.zeros(1, 1, 1),
                extra={
                    "encoder_last_hidden_state": torch.randn(1, 2, 3),
                    "encoder_attention_mask": torch.ones(1, 2, dtype=torch.long),
                },
            ),
        )


class ReuseMethod(GFlowMethod):
    def _precomputed_reward_cache_tb_inputs(self, model, batch):
        trajectories = SemanticTrajectoryBatch(
            candidate_token_ids=torch.tensor([[[1, 11]]]),
            candidate_token_mask=torch.tensor([[[True, True]]]),
            is_positive=torch.tensor([[True]]),
        )
        return trajectories, torch.ones(1, 1), torch.zeros(1), torch.ones(1, 1), {}


class FakeModel:
    def score_items(self, batch, **kwargs):
        assert "_tiger_encoder_last_hidden_state" in batch
        assert "_tiger_encoder_attention_mask" in batch
        return torch.zeros(1, 1)


def test_tb_encoder_reuse_round_trip_and_scoring():
    cfg = GFlowConfig.from_dict(
        {"objective": "tb", "tb_logz_source": "reward_cache", "tb_reuse_sft_encoder": True}
    )
    assert cfg.to_dict()["tb_reuse_sft_encoder"] is True
    method = ReuseMethod(cfg, sft_method=FakeSFT())
    method.compute_loss(FakeModel(), {"user_ids": torch.tensor([0])})


if __name__ == "__main__":
    test_tb_encoder_reuse_round_trip_and_scoring()
    print("TB encoder reuse tests passed")
