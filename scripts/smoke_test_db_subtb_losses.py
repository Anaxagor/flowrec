"""CPU smoke tests for analytical detailed- and subtrajectory-balance losses."""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.ft_methods.gflow.exact_flows import CatalogPrefixFlowIndex
from flowrec.ft_methods.gflow.losses.detailed_balance import detailed_balance_loss
from flowrec.ft_methods.gflow.losses.subtrajectory_balance import (
    subtrajectory_balance_loss,
)


def main() -> None:
    catalog = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.long)
    rewards = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    candidates = catalog.unsqueeze(0)
    token_mask = torch.ones_like(candidates, dtype=torch.bool)

    index = CatalogPrefixFlowIndex(catalog)
    prefix_log_flows = index.prefix_log_flows(
        candidates,
        rewards.float(),
        reward_epsilon=1e-9,
    ).double()
    token_logprobs = prefix_log_flows[..., 1:] - prefix_log_flows[..., :-1]

    db_loss, db_metrics = detailed_balance_loss(
        token_logprobs,
        prefix_log_flows[..., :-1],
        prefix_log_flows[..., 1:],
        rewards,
        candidate_token_mask=token_mask,
    )
    subtb_loss, subtb_metrics = subtrajectory_balance_loss(
        token_logprobs,
        prefix_log_flows,
        rewards,
        candidate_token_mask=token_mask,
        subtb_lambda=0.9,
    )

    torch.testing.assert_close(db_loss, torch.zeros_like(db_loss), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        subtb_loss,
        torch.zeros_like(subtb_loss),
        atol=1e-12,
        rtol=0.0,
    )
    assert float(db_metrics["exact_db_active"]) == 1.0
    assert float(subtb_metrics["exact_subtb_active"]) == 1.0
    print("analytical DB and SubTB zero-residual checks passed")


if __name__ == "__main__":
    main()
