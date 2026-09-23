"""S-DPO uses the configured curriculum at the trainer's current step."""

import pytest
import torch
import torch.nn.functional as F

from flowrec.candidates.sampler import CandidateSamplerConfig, SemanticIDCandidateSampler
from flowrec.ft_methods.preference_sdpo.config import SDPOConfig
from flowrec.ft_methods.preference_sdpo.method import SDPOMethod, softmax_preference_loss


TABLE = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
ITEM_TO_ROW = torch.tensor([-1, 0, 1, 2, 3, 4])


class ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("item_semantic_ids", TABLE.clone())

    def sequence_logprobs(self, batch, *, target_token_ids, target_token_mask):
        return self.weight * target_token_ids[:, 0].float()

    def score_items(self, batch, *, candidate_token_ids, candidate_token_mask):
        return self.weight * candidate_token_ids[:, :, 0].float()


class ToyTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scores", torch.tensor([0.0, 100.0, 0.0, 50.0, -100.0, 0.0]))

    @property
    def device(self):
        return self.scores.device

    def score_items(self, batch, candidate_item_ids):
        return self.scores[candidate_item_ids].unsqueeze(0).expand(batch["target_token_ids"].size(0), -1)


def _batch():
    return {
        "target_token_ids": TABLE[2:3],
        "target_token_mask": torch.ones((1, 2), dtype=torch.bool),
        "target_item_ids": torch.tensor([3]),
        "history_item_ids": torch.tensor([[2]]),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
    }


def test_curriculum_step_reaches_sdpo_negative_sampler():
    cfg = SDPOConfig(
        num_negatives=1,
        negative_sampler="curriculum",
        curriculum_steps=100,
        sampling_temperature=1e-4,
        exclude_history_in_sampling=True,
    )
    sampler = SemanticIDCandidateSampler(
        CandidateSamplerConfig(
            mode="curriculum", num_negatives=1, include_positive=False,
            exclude_history=True, curriculum_steps=100, temperature=1e-4,
        ),
        item_semantic_ids=TABLE,
        item_id_to_semantic_row=ITEM_TO_ROW,
        scorer=ToyTeacher(),
    )
    policy = ToyPolicy()
    method = SDPOMethod(cfg, model=policy, negative_sampler=sampler)

    easy = method.compute_loss(policy, _batch(), step=0)
    hard = method.compute_loss(policy, _batch(), step=100)
    assert easy.extra["rejected_token_ids"][0, 0, 0].item() == 4
    assert hard.extra["rejected_token_ids"][0, 0, 0].item() == 1
    assert easy.loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    easy.loss.backward()
    assert policy.weight.grad is not None and policy.weight.grad.abs().item() > 0


def test_one_negative_softmax_matches_pairwise_dpo():
    chosen = torch.tensor([0.1, -0.2])
    rejected = torch.tensor([[0.5], [-0.3]])
    assert torch.allclose(
        softmax_preference_loss(chosen, rejected),
        F.softplus(rejected[:, 0] - chosen),
    )


def test_sdpo_draws_distinct_negatives_for_multiway_loss():
    policy = ToyPolicy()
    method = SDPOMethod(SDPOConfig(num_negatives=3), model=policy)
    output = method.compute_loss(policy, _batch(), step=7)
    negatives = output.extra["rejected_token_ids"][0, :, 0].tolist()
    assert len(negatives) == len(set(negatives)) == 3
    assert 3 not in negatives
    assert output.loss.isfinite()


def test_curriculum_configuration_requires_positive_steps():
    with pytest.raises(ValueError, match="curriculum_steps"):
        SDPOConfig(negative_sampler="curriculum", curriculum_steps=0)
