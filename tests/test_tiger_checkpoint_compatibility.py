from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.models.tiger import TigerModel
from scripts.run_tiger import build_model


def _checkpoint_compatible_tiger() -> TigerModel:
    return TigerModel(
        codebook_size=2,
        sem_id_len=2,
        user_ids_count=8,
        use_user_token=False,
        semantic_id_mapping={"a": [0, 1], "b": [1, 0]},
        embedding_dim=8,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        d_kv=32,
        dropout=0.0,
        enable_flow_head=True,
        detach_flow_head_trunk=False,
    )


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


def test_tiger_config_preserves_checkpoint_critical_fields() -> None:
    model = _checkpoint_compatible_tiger()

    config = model.config_dict()

    assert config["d_kv"] == 32
    assert config["use_user_token"] is False
    assert config["enable_flow_head"] is True
    assert config["detach_flow_head_trunk"] is False
    assert "flow_head.1.weight" in model.state_dict()


def test_tiger_strict_load_patches_only_flow_head_compatibility() -> None:
    source = _checkpoint_compatible_tiger()
    legacy_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("flow_head.")
    }
    legacy_state["flow_head.weight"] = torch.zeros(1, 8)
    legacy_state["flow_head.bias"] = torch.zeros(1)

    target = _checkpoint_compatible_tiger()
    target.load_state_dict(legacy_state, strict=True)

    invalid_state = dict(legacy_state)
    invalid_state["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected.weight"):
        target.load_state_dict(invalid_state, strict=True)


def test_tiger_runner_forwards_checkpoint_critical_model_fields(tmp_path: Path) -> None:
    data = SimpleNamespace(
        sid_len=2,
        semantic_ids={"a": [0, 1], "b": [1, 0]},
        train_item_counts_by_raw_id={},
    )
    model = build_model(
        {
            "dataset": {"num_codebooks": 2},
            "model": {
                "codebook_size": 2,
                "user_ids_count": 8,
                "use_user_token": False,
                "embedding_dim": 8,
                "num_heads": 1,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 16,
                "d_kv": 32,
                "enable_flow_head": True,
                "detach_flow_head_trunk": True,
            },
        },
        data,
        config_path=tmp_path / "config.json",
    )

    assert model.d_kv == 32
    assert model.use_user_token is False
    assert model.enable_flow_head is True
    assert model.detach_flow_head_trunk is True
