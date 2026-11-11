import math

import pytest
import torch

from flowrec.ft_methods.gflow.config import GFlowConfig
from flowrec.ft_methods.gflow.exact_flows import CatalogPrefixFlowIndex
from flowrec.ft_methods.gflow.losses.prefix_ce import prefix_ce_loss


def test_child_targets_include_unsampled_siblings_and_only_catalog_actions():
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0]])
    paths = catalog[[0, 2]].unsqueeze(0)
    rewards = torch.tensor([[1., 3., 2.]], requires_grad=True)
    logs, valid = CatalogPrefixFlowIndex(catalog).child_log_flows(paths, rewards)
    torch.testing.assert_close(logs.exp(), torch.tensor([[[[4., 2.], [1., 3.]],
                                                        [[4., 2.], [2., 0.]]]]))
    assert valid.tolist() == [[[[True, True], [True, True]], [[True, True], [True, False]]]]
    assert not logs.requires_grad


def test_masking_zero_flow_nodes_and_scale_invariance():
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0]])
    paths = catalog.unsqueeze(0)
    index = CatalogPrefixFlowIndex(catalog)
    rewards = torch.tensor([[1., 3., 0.]])
    logs, valid = index.child_log_flows(paths, rewards)
    logits = torch.zeros(1, 3, 2, 3, requires_grad=True)
    # The third vocabulary entry, and second child of prefix 1, are invalid.
    logits.data[..., 2] = 1000.
    loss, metrics = prefix_ce_loss(logits, logs, valid)
    assert loss.item() == pytest.approx(math.log(2.))
    scaled, _ = index.child_log_flows(paths, rewards * 100)
    other, _ = prefix_ce_loss(logits, scaled, valid)
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[..., 2].count_nonzero() == 0
    assert logits.grad[0, 2, 1].count_nonzero() == 0
    assert metrics['prefix_ce_kl_level_0'].item() == pytest.approx(math.log(2.))


def test_chain_rule_and_exact_solution():
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0]])
    paths = catalog.unsqueeze(0)
    index = CatalogPrefixFlowIndex(catalog)
    logs, valid = index.child_log_flows(paths, torch.tensor([[1., 3., 2.]]))
    logits = torch.tensor([[[[.7, -.2], [.3, -.4]], [[.7, -.2], [.3, -.4]],
                            [[.7, -.2], [.2, 100.]]]], requires_grad=True)
    q = torch.tensor([1/6, 3/6, 2/6])
    loss, _ = prefix_ce_loss(logits, logs, valid, candidate_weights=q.unsqueeze(0))
    root_lp = logits[0, 0, 0].log_softmax(-1)
    inner_lp = logits[0, 0, 1].log_softmax(-1)
    item_ce = -(q * torch.stack([root_lp[0] + inner_lp[0], root_lp[0] + inner_lp[1], root_lp[1]])).sum()
    torch.testing.assert_close(2 * loss, item_ce)
    teacher = logs.clone().requires_grad_()
    optimum = logs.masked_fill(~valid, 0).clone().requires_grad_()
    _, metrics = prefix_ce_loss(optimum, teacher, valid)
    assert metrics['prefix_ce_kl'].abs() < 1e-6
    optimum_loss, _ = prefix_ce_loss(optimum, teacher, valid)
    optimum_loss.backward()
    assert teacher.grad is None
    assert optimum.grad.abs().max() < 1e-6
