from __future__ import annotations

import torch


def detailed_balance_loss(*args, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raise NotImplementedError("Detailed balance loss is not implemented yet.")
