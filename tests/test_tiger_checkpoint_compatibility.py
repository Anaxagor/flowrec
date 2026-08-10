from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.models.tiger import TigerModel
from scripts import run_tiger as tiger_runner
from scripts.run_tiger import build_data_config, build_method, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP2_CONFIG_DIR = PROJECT_ROOT / "configs" / "exp2-tuned-params"


def _load_exp2_tiger_config(dataset: str) -> dict:
    return json.loads((EXP2_CONFIG_DIR / f"tiger_{dataset}.json").read_text())


@pytest.mark.parametrize(
    ("dataset", "max_grad_norm", "validation_batch_size", "optimizer"),
    [
        ("beauty", 0.5, 4096, {"lr": 0.0011966748408121942, "weight_decay": 0.01}),
        (
            "instruments",
            2.0,
            1024,
            {"lr": 0.0018511868508356536, "weight_decay": 0.01},
        ),
    ],
)
def test_exp2_tiger_config_is_complete(
    dataset: str,
    max_grad_norm: float,
    validation_batch_size: int,
    optimizer: dict[str, float],
) -> None:
    cfg = _load_exp2_tiger_config(dataset)

    assert set(cfg) == {
        "experiment_name",
        "seed",
        "data",
        "dataset",
        "dataloader",
        "model",
        "method",
        "training",
        "optimizer",
    }
    assert set(cfg["data"]) == {
        "source",
        "reviews_path",
        "metadata_path",
        "semantic_id_path",
        "processed_cache_dir",
        "rebuild_processed_cache",
    }
    assert cfg["data"]["reviews_path"] == (
        f"data/raw/dataframe_format/reviews_{dataset}_"
        f"{'2014' if dataset == 'beauty' else '2018'}.parquet"
    )
    assert cfg["data"]["metadata_path"] == (
        f"data/raw/dataframe_format/meta_{dataset}_"
        f"{'2014' if dataset == 'beauty' else '2018'}.parquet"
    )
    assert cfg["data"]["semantic_id_path"] == (
        f"datasets/semantic_ids/exp2-tuned-params/{dataset}/semantic_ids.json"
    )
    assert cfg["dataset"]["positive_criterion"] == "verified_purchase == True"
    assert cfg["dataset"]["metadata_item_key"] == "asin"
    assert cfg["dataset"]["offset_semantic_ids_by_one"] is True
    assert cfg["dataset"]["train_example_layout"] == "prefix_target"
    assert cfg["dataloader"] == {
        "train_batch_size": 2048,
        "validation_batch_size": validation_batch_size,
        "test_batch_size": 64,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "train_drop_last": False,
        "include_raw_in_batch": False,
    }
    assert cfg["model"] == {
        **cfg["model"],
        "embedding_dim": 128,
        "num_heads": 6,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
        "d_kv": 32,
        "use_user_token": False,
        "user_ids_count": 0,
        "enable_flow_head": True,
        "detach_flow_head_trunk": False,
    }
    assert cfg["method"] == {"name": "sft"}
    assert "evaluation" not in cfg
    assert cfg["training"]["generation_num_beams"] == 1
    assert cfg["training"]["generation_num_return_sequences"] == 1
    assert cfg["training"]["test_evaluate_generation"] is True
    assert cfg["training"]["test_generation_num_beams"] == 20
    assert cfg["training"]["test_generation_num_return_sequences"] == 20
    assert cfg["training"]["test_generation_topk"] == [5, 10, 20]
    assert cfg["training"]["max_grad_norm"] == max_grad_norm
    assert cfg["training"]["save_best"] is True
    assert cfg["training"]["init_checkpoint_path"] == (
        f"checkpoints/exp2-tuned_params/initial/tiger/{dataset}/best.pt"
    )
    assert cfg["training"]["init_checkpoint_strict"] is True
    assert cfg["optimizer"] == {
        **optimizer,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    }


@pytest.mark.parametrize(
    ("dataset", "validation_batch_size"),
    [("beauty", 4096), ("instruments", 1024)],
)
def test_exp2_tiger_config_parses_through_runner_builders(
    dataset: str, validation_batch_size: int
) -> None:
    config_path = EXP2_CONFIG_DIR / f"tiger_{dataset}.json"
    cfg = _load_exp2_tiger_config(dataset)

    data_cfg = build_data_config(cfg, config_path=config_path)
    assert data_cfg.reviews_path == PROJECT_ROOT / cfg["data"]["reviews_path"]
    assert data_cfg.metadata_path == PROJECT_ROOT / cfg["data"]["metadata_path"]
    assert data_cfg.semantic_id_path == PROJECT_ROOT / cfg["data"]["semantic_id_path"]
    assert data_cfg.processed_cache_dir == PROJECT_ROOT / cfg["data"]["processed_cache_dir"]
    assert data_cfg.metadata_item_key == "asin"
    assert data_cfg.positive_criterion == "verified_purchase == True"
    assert data_cfg.offset_semantic_ids_by_one is True
    assert data_cfg.batch_size == 2048
    assert data_cfg.eval_batch_size == validation_batch_size
    assert data_cfg.test_batch_size == 64

    data = SimpleNamespace(
        sid_len=4,
        semantic_ids={"item": [1, 2, 3, 4]},
        train_item_counts_by_raw_id={"item": 1},
    )
    model = build_model(cfg, data, config_path=config_path)
    method = build_method(
        cfg,
        model,
        data=data,
        train_loader=None,
        config_path=config_path,
    )

    assert model.config_dict()["embedding_dim"] == 128
    assert model.config_dict()["d_kv"] == 32
    assert model.config_dict()["use_user_token"] is False
    assert model.config_dict()["has_flow_head"] is True
    assert method.cfg.generation_num_beams == 1
    assert method.cfg.generation_num_return_sequences == 1


def test_exp2_tiger_runner_uses_best_checkpoint_and_test_beam_20(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = EXP2_CONFIG_DIR / "tiger_beauty.json"
    captured: dict[str, object] = {}

    class FakeData:
        semantic_ids = {"item": [1, 2, 3, 4]}
        sid_len = 4
        cfg = SimpleNamespace(train_example_layout="prefix_target")

        def setup(self):
            return None

        def train_dataloader(self):
            return "train-loader"

        def val_dataloader(self):
            return "validation-loader"

        def test_dataloader(self):
            return "test-loader"

        def summary(self):
            return {}

    class FakeModel:
        supported_train_layouts = ("prefix_target",)

        def config_dict(self):
            return {"num_beams": 20, "num_return_sequences": 20}

        def num_parameters(self, trainable_only: bool = False):
            return 1

        def load_state_dict(self, state_dict, *, strict: bool):
            captured["init_checkpoint_loaded"] = strict

    class FakeLogger:
        def __call__(self, *args, **kwargs):
            return None

        def log_asset(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class FakeTrainer:
        def __init__(self, method):
            self.method = method
            self.state = SimpleNamespace(
                epoch=0,
                global_step=1,
                best_metric=1.0,
                best_epoch=0,
            )

        @classmethod
        def from_optimizer_config(cls, *, method, **kwargs):
            instance = cls(method)
            captured["trainer"] = instance
            return instance

        def fit(self, train_loader, val_loader):
            assert (train_loader, val_loader) == ("train-loader", "validation-loader")
            return SimpleNamespace(
                best_checkpoint_path=str(tmp_path / "best.pt"),
                last_checkpoint_path=str(tmp_path / "last.pt"),
                val_metrics={"val/sft/ndcg@1": 1.0},
            )

        def load_checkpoint(self, path, *, load_optimizer, load_scheduler):
            captured["loaded_checkpoint"] = path
            assert load_optimizer is False
            assert load_scheduler is False

        def test(self, loader):
            captured["test_loader"] = loader
            captured["test_generation"] = (
                self.method.cfg.evaluate_generation,
                self.method.cfg.generation_num_beams,
                self.method.cfg.generation_num_return_sequences,
                self.method.cfg.generation_topk,
            )
            return {"test/sft/ndcg@20": 1.0}

    monkeypatch.setattr(
        tiger_runner,
        "parse_args",
        lambda: SimpleNamespace(
            config=config_path,
            run_name=None,
            device="cpu",
            seed=None,
            max_epochs=None,
            max_steps=None,
            experiments_dir=tmp_path / "experiments",
            plain_run_dir=False,
        ),
    )
    monkeypatch.setattr(tiger_runner, "create_run_dir", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(tiger_runner, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(tiger_runner, "build_data_module", lambda *args, **kwargs: FakeData())
    monkeypatch.setattr(tiger_runner, "build_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(tiger_runner, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        tiger_runner,
        "build_experiment_logger",
        lambda *args, **kwargs: FakeLogger(),
    )
    monkeypatch.setattr(tiger_runner, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(tiger_runner, "write_text_summary", lambda *args, **kwargs: None)

    tiger_runner.main()

    assert captured["init_checkpoint_loaded"] is True
    assert captured["loaded_checkpoint"] == str(tmp_path / "best.pt")
    assert captured["test_loader"] == "test-loader"
    assert captured["test_generation"] == (True, 20, 20, (5, 10, 20))
    assert captured["trainer"].method.cfg.generation_num_beams == 1


def test_tiger_runner_skips_initialization_when_init_checkpoint_is_null(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = EXP2_CONFIG_DIR / "tiger_beauty.json"
    config = deepcopy(_load_exp2_tiger_config("beauty"))
    config["training"]["init_checkpoint_path"] = None
    initial_loads = []

    class StopAfterModelBuild(Exception):
        pass

    class FakeData:
        semantic_ids = {"item": [1, 2, 3, 4]}
        sid_len = 4
        cfg = SimpleNamespace(train_example_layout="prefix_target")

        def setup(self):
            return None

        def train_dataloader(self):
            return None

        def val_dataloader(self):
            return None

        def test_dataloader(self):
            return None

    def stop_after_model_build(*args, **kwargs):
        raise StopAfterModelBuild

    monkeypatch.setattr(
        tiger_runner,
        "parse_args",
        lambda: SimpleNamespace(
            config=config_path,
            run_name=None,
            device="cpu",
            seed=None,
            max_epochs=None,
            max_steps=None,
            experiments_dir=tmp_path / "experiments",
            plain_run_dir=False,
        ),
    )
    monkeypatch.setattr(tiger_runner, "load_json", lambda path: config)
    monkeypatch.setattr(tiger_runner, "create_run_dir", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(tiger_runner, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(tiger_runner, "build_data_module", lambda *args, **kwargs: FakeData())
    monkeypatch.setattr(
        tiger_runner,
        "build_model",
        stop_after_model_build,
    )
    monkeypatch.setattr(
        tiger_runner,
        "load_model_state_dict",
        lambda *args, **kwargs: initial_loads.append((args, kwargs)),
    )

    with pytest.raises(StopAfterModelBuild):
        tiger_runner.main()

    assert initial_loads == []


def _checkpoint_compatible_tiger(*, enable_flow_head: bool | None = None) -> TigerModel:
    flow_head_config = (
        {} if enable_flow_head is None else {"enable_flow_head": enable_flow_head}
    )
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
        detach_flow_head_trunk=False,
        **flow_head_config,
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


def test_tiger_default_flow_head_matches_checkpoint_contract() -> None:
    model = _checkpoint_compatible_tiger()

    assert model.enable_flow_head is True
    assert isinstance(model.flow_head, nn.Sequential)
    assert [type(layer) for layer in model.flow_head] == [
        nn.LayerNorm,
        nn.Linear,
        nn.GELU,
        nn.Linear,
    ]
    assert model.flow_head[1].in_features == 8
    assert model.flow_head[1].out_features == 16
    assert model.flow_head[3].in_features == 16
    assert model.flow_head[3].out_features == 1
    assert torch.count_nonzero(model.flow_head[3].weight) == 0
    assert torch.count_nonzero(model.flow_head[3].bias) == 0


def test_tiger_disabled_flow_head_has_no_attribute() -> None:
    model = _checkpoint_compatible_tiger(enable_flow_head=False)

    assert not hasattr(model, "flow_head")
    assert model.config_dict()["has_flow_head"] is False


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


def test_tiger_load_state_dict_preserves_metadata() -> None:
    source = _checkpoint_compatible_tiger()
    state_dict = source.state_dict()
    state_dict._metadata["model"] = {"checkpoint_contract": "tiger"}
    target = _checkpoint_compatible_tiger()
    received_metadata = []

    def capture_metadata(module, incoming_state_dict, prefix, local_metadata, *args):
        if prefix == "model.":
            received_metadata.append(local_metadata)

    handle = target.model.register_load_state_dict_pre_hook(capture_metadata)
    try:
        target.load_state_dict(state_dict, strict=True)
    finally:
        handle.remove()

    assert received_metadata == [{"checkpoint_contract": "tiger"}]


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


def test_tiger_runner_defaults_to_an_enabled_flow_head(tmp_path: Path) -> None:
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
            },
        },
        data,
        config_path=tmp_path / "config.json",
    )

    assert model.enable_flow_head is True
