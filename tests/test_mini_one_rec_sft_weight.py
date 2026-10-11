"""MiniOneRec can take a fixed sft_weight or gradient-norm calibration."""

import pytest
import torch

from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY
from flowrec.ft_methods.mini_one_rec.config import MiniOneRecConfig
from flowrec.ft_methods.mini_one_rec.method import MiniOneRecMethod
from flowrec.models.base import RecModelOutput

TABLE = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])


class ToyPolicy(torch.nn.Module):
    dense_supervision = "tokens"

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.bias = torch.nn.Parameter(torch.zeros(8))
        self.register_buffer("item_semantic_ids", TABLE.clone())

    def forward(self, batch):
        ids = batch["target_token_ids"]
        batch_size, length = ids.shape
        logits = self.weight * torch.ones(batch_size, length, 8, device=ids.device)
        return RecModelOutput(logits=logits + self.bias)

    def generate(self, batch, *, num_return_sequences, **kwargs):
        target = batch["target_token_ids"]
        other = torch.tensor([[2, 2]], device=target.device, dtype=target.dtype).expand_as(target)
        sequences = [target, other]
        while len(sequences) < int(num_return_sequences):
            sequences.append(other)
        return torch.stack(sequences[: int(num_return_sequences)], dim=1)

    def score_item_tokens(self, batch, *, candidate_token_ids, candidate_token_mask):
        return self.weight * candidate_token_ids[:, :, 0].float().unsqueeze(-1).expand_as(
            candidate_token_ids
        )


def _batch():
    return {
        "target_token_ids": TABLE[2:3],
        "target_token_mask": torch.ones((1, 2), dtype=torch.bool),
        "target_item_ids": torch.tensor([3]),
        "history_item_ids": torch.tensor([[2]]),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
    }


def _method(**kwargs) -> MiniOneRecMethod:
    cfg = MiniOneRecConfig(
        num_rollouts=2,
        rollout_strategy="sample",
        rollout_constrain_valid_sequences=False,
        compute_train_metrics=False,
        compute_eval_metrics=False,
        evaluate_generation=False,
        **kwargs,
    )
    return MiniOneRecMethod(cfg)


def test_default_sft_weight_is_one():
    assert MiniOneRecConfig().sft_weight == pytest.approx(1.0)
    assert MiniOneRecConfig().calibrate_sft_weight is False


def test_fixed_sft_weight_is_used_without_calibration():
    policy = ToyPolicy()
    method = _method(sft_weight=2.0)
    assert method.maybe_calibrate_sft_weight(policy, [_batch()]) is None
    assert method.cfg.sft_weight == pytest.approx(2.0)

    output = method.compute_loss(policy, _batch(), step=0)
    components = output.extra[LOSS_COMPONENTS_KEY]
    assert set(components) == {"grpo", "sft"}
    expected = method.cfg.grpo_weight * components["grpo"] + 2.0 * components["sft"]
    assert torch.allclose(output.loss, expected)


def test_calibrate_sft_weight_rejects_a_fixed_weight():
    with pytest.raises(ValueError, match="not both"):
        MiniOneRecConfig(sft_weight=1.0, calibrate_sft_weight=True)


def test_from_dict_accepts_nested_calibration_block():
    cfg = MiniOneRecConfig.from_dict(
        {
            "num_rollouts": 4,
            "calibrate_sft_weight": {
                "num_batches": 4,
                "target_ratio": 0.5,
                "seed": 7,
            },
        }
    )
    assert cfg.calibrate_sft_weight is True
    assert cfg.sft_weight == 0.0
    assert cfg.calibration_num_batches == 4
    assert cfg.calibration_target_ratio == pytest.approx(0.5)
    assert cfg.calibration_seed == 7


def test_maybe_calibrate_fills_sft_weight_from_gradient_norms():
    policy = ToyPolicy()
    method = _method(
        sft_weight=0.0,
        calibrate_sft_weight=True,
        calibration_num_batches=2,
        calibration_target_ratio=1.0,
    )
    result = method.maybe_calibrate_sft_weight(
        policy,
        [_batch(), _batch()],
        seed=0,
    )
    assert result is not None
    assert method.cfg.calibrate_sft_weight is False
    assert method.cfg.sft_weight == pytest.approx(float(result["weight"]))
    assert method.cfg.sft_weight > 0
    assert result["effective_ratio_check"] == pytest.approx(1.0, rel=1e-5)

    output = method.compute_loss(policy, _batch(), step=0)
    components = output.extra[LOSS_COMPONENTS_KEY]
    expected = method.cfg.grpo_weight * components["grpo"] + method.cfg.sft_weight * components["sft"]
    assert torch.allclose(output.loss, expected)
