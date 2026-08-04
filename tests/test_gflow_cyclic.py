from __future__ import annotations

import math

import pytest
import torch

from flowrec.ft_methods.gflow.cyclic import (
    APPEND_ACTION,
    DELETE_ACTION,
    STOP_ACTION,
    analytical_delete_probabilities,
    analytical_state_log_flows,
    sample_analytical_cyclic_trajectories,
)
from flowrec.ft_methods.gflow.exact_flows import CatalogPrefixFlowIndex
from flowrec.ft_methods.gflow.losses.cyclic import (
    cyclic_detailed_balance_loss,
    cyclic_subtrajectory_balance_loss,
    cyclic_trajectory_balance_loss,
)
from flowrec.ft_methods.base import MethodOutput
from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.cyclic import CyclicSemanticTrajectoryBatch
from flowrec.ft_methods.gflow.method import GFlowMethod
from flowrec.models.base import RecModelOutput


def _catalog() -> tuple[CatalogPrefixFlowIndex, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[1, 1], [1, 2], [2, 1]], dtype=torch.long)
    rewards = torch.tensor([[3.0, 5.0, 7.0]])
    return CatalogPrefixFlowIndex(tokens), tokens, rewards


def test_prefix_ranges_sum_rewards_and_sample_catalog_rows() -> None:
    index, _, rewards = _catalog()
    csum = index.sorted_reward_cumsum(rewards)
    prefixes = torch.tensor([[[0, 0], [1, 0], [2, 0]]])
    lengths = torch.tensor([[0, 1, 1]])

    lo, hi = index.prefix_ranges(prefixes, lengths)

    assert torch.equal(lo, torch.tensor([[0, 0, 2]]))
    assert torch.equal(hi, torch.tensor([[3, 2, 3]]))
    assert torch.allclose(
        index.range_sums(csum, lo, hi), torch.tensor([[15.0, 8.0, 7.0]], dtype=torch.float64)
    )
    assert torch.equal(
        index.sample_item_rows_from_ranges(csum, lo[:, 2:], hi[:, 2:]),
        torch.tensor([[2]]),
    )


def test_analytical_visit_flows_and_delete_cap_are_closed_form() -> None:
    base = torch.full((1, 3), math.log(10.0))
    depths = torch.tensor([[0, 1, 2]])

    flows = analytical_state_log_flows(
        base, depths, sid_len=2, cycle_flow_ratio=0.5
    ).exp()

    assert torch.allclose(flows, torch.tensor([[15.0, 20.0, 15.0]]))
    assert analytical_delete_probabilities(0.5, delete_probability_cap=0.4) == pytest.approx(
        (0.25, 1.0 / 3.0)
    )
    with pytest.raises(ValueError, match="cap"):
        analytical_delete_probabilities(0.5, delete_probability_cap=0.3)
    with pytest.raises(ValueError, match="cap"):
        analytical_delete_probabilities(0.5, delete_probability_cap=1.0 / 3.0)


def test_analytical_sampler_has_valid_bidirectional_steps_and_backward_terms() -> None:
    index, tokens, rewards = _catalog()
    kappa = 0.25
    batch = sample_analytical_cyclic_trajectories(
        flow_index=index,
        dense_rewards=rewards,
        item_tokens=tokens,
        num_trajectories=128,
        cycle_flow_ratio=kappa,
        delete_probability_cap=0.49,
        max_steps=128,
        reward_epsilon=1e-6,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.all(batch.terminal_item_row_ids.ge(0))
    assert torch.any(batch.action_types.eq(DELETE_ACTION) & batch.step_mask)
    assert torch.all(batch.action_types[batch.step_mask].ge(APPEND_ACTION))
    assert torch.all(batch.action_types[batch.step_mask].le(STOP_ACTION))

    current_depths = batch.state_depths[..., :-1]
    next_depths = batch.state_depths[..., 1:]
    append = batch.step_mask & batch.action_types.eq(APPEND_ACTION)
    delete = batch.step_mask & batch.action_types.eq(DELETE_ACTION)
    stop = batch.step_mask & batch.action_types.eq(STOP_ACTION)
    assert torch.all(next_depths[append] == current_depths[append] + 1)
    assert torch.all(next_depths[delete] == current_depths[delete] - 1)
    assert torch.all(next_depths[stop] == current_depths[stop])
    assert torch.all(current_depths[stop] == tokens.size(1))

    current_base = batch.state_base_log_flows[..., :-1]
    next_base = batch.state_base_log_flows[..., 1:]
    next_flow = batch.state_log_flows[..., 1:]
    assert torch.allclose(
        batch.backward_logprobs[append],
        (next_base + math.log1p(kappa) - next_flow)[append],
    )
    assert torch.allclose(
        batch.backward_logprobs[delete],
        (current_base + math.log(kappa) - next_flow)[delete],
    )
    assert torch.allclose(
        batch.root_backward_logprobs,
        torch.full_like(batch.root_backward_logprobs, -math.log1p(kappa)),
    )


def test_analytical_sampler_watchdog_fails_instead_of_truncating() -> None:
    index, tokens, rewards = _catalog()

    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        sample_analytical_cyclic_trajectories(
            flow_index=index,
            dense_rewards=rewards,
            item_tokens=tokens,
            num_trajectories=128,
            cycle_flow_ratio=1.0,
            max_steps=3,
            reward_epsilon=1e-6,
            generator=torch.Generator().manual_seed(0),
        )


def test_cyclic_balance_losses_vanish_for_balanced_trajectory() -> None:
    forward = torch.zeros((1, 1, 3))
    backward = torch.zeros_like(forward)
    flows = torch.zeros((1, 1, 4))
    step_mask = torch.tensor([[[True, True, False]]])
    rewards = torch.ones((1, 1))
    root_backward = torch.zeros((1, 1))

    tb_loss, _ = cyclic_trajectory_balance_loss(
        forward, backward, step_mask, rewards, torch.zeros(()), root_backward
    )
    db_loss, _ = cyclic_detailed_balance_loss(forward, backward, flows, step_mask)
    subtb_loss, _ = cyclic_subtrajectory_balance_loss(
        forward, backward, flows, step_mask, subtb_lambda=0.5
    )

    assert tb_loss.item() == pytest.approx(0.0)
    assert db_loss.item() == pytest.approx(0.0)
    assert subtb_loss.item() == pytest.approx(0.0)


class _FakeSFT(torch.nn.Module):
    def compute_loss(self, model, batch, *, stage="train", step=None):
        return MethodOutput(
            loss=torch.tensor(0.5),
            model_output=RecModelOutput(
                logits=torch.zeros(1, 1, 1),
                extra={
                    "encoder_last_hidden_state": torch.zeros(1, 1, 2),
                    "encoder_attention_mask": torch.ones(1, 1, dtype=torch.long),
                },
            ),
        )


class _CyclicDispatchMethod(GFlowMethod):
    def _sample_cyclic_trajectories(self, model, batch):
        trajectories = CyclicSemanticTrajectoryBatch(
            state_token_ids=torch.zeros(1, 1, 3, 2, dtype=torch.long),
            state_depths=torch.tensor([[[0, 1, 1]]]),
            action_types=torch.tensor([[[APPEND_ACTION, STOP_ACTION]]], dtype=torch.int8),
            step_mask=torch.tensor([[[True, True]]]),
            terminal_item_row_ids=torch.zeros(1, 1, dtype=torch.long),
            terminal_token_ids=torch.ones(1, 1, 2, dtype=torch.long),
            terminal_rewards=torch.ones(1, 1),
            state_base_log_flows=torch.zeros(1, 1, 3),
            state_log_flows=torch.zeros(1, 1, 3),
            backward_logprobs=torch.zeros(1, 1, 2),
            root_backward_logprobs=torch.zeros(1, 1),
            is_positive=torch.zeros(1, 1, dtype=torch.bool),
        )
        return trajectories, torch.zeros(1), {}

    def _cyclic_forward_logprobs(self, model, batch, trajectories):
        assert "_tiger_encoder_last_hidden_state" in batch
        return torch.zeros(1, 1, 2), {}

    def _cached_positive_mask(self, batch, candidate_tokens):
        return torch.zeros(candidate_tokens.shape[:2], dtype=torch.bool)


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
def test_method_dispatches_all_cyclic_objectives(objective: str) -> None:
    method = _CyclicDispatchMethod(
        GFlowConfig(
            objective=objective,
            environment="cyclic",
            reward_cache_path="unused",
            tb_logz_source="reward_cache",
            db_activation_checkpointing=False,
        ),
        sft_method=_FakeSFT(),
    )
    output = method.compute_loss(object(), {"user_ids": torch.tensor([0])})
    assert torch.isfinite(output.loss)
    assert output.metrics["gflow/cyclic_active"].item() == pytest.approx(1.0)
