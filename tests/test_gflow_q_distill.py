import pytest
import torch

from flowrec.ft_methods.gflow.losses.q_distill import q_distill_loss


def test_exact_q_targets_are_raw_child_log_flows_and_detached():
    flows = torch.tensor([[[6., 4., 1.]]]).log().requires_grad_()
    q = torch.tensor([[[4., 1.]]]).log().requires_grad_()
    loss, _ = q_distill_loss(q, flows, torch.ones_like(q, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.tensor(0.))
    loss, _ = q_distill_loss(q + .5, flows, torch.ones_like(q, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.tensor(.25))
    loss.backward()
    torch.testing.assert_close(q.grad, torch.tensor([[[.5, .5]]]))
    assert flows.grad is None


def test_q_distill_rejects_partial_paths():
    with pytest.raises(ValueError):
        q_distill_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 3),
                       torch.tensor([[[True, False]]]))
