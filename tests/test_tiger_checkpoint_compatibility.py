from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flowrec.models.base import FlowRecModel, RecModelOutput


class _TinyCheckpointModel(FlowRecModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return RecModelOutput()


def test_generic_loader_accepts_trainer_model_state_dict(tmp_path: Path) -> None:
    source = _TinyCheckpointModel()
    path = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": source.state_dict(), "trainer_state": {}}, path)
    target = _TinyCheckpointModel()
    target.weight.data.zero_()

    payload = target.load_checkpoint(path)

    assert target.weight.item() == pytest.approx(source.weight.item())
    assert "trainer_state" in payload
