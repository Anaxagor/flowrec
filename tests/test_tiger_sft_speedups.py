from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from flowrec.data import datamodule as datamodule_module
from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.base import FlowRecMethod, MethodOutput
from flowrec.models.base import FlowRecModel, RecModelOutput
from flowrec.models.tiger import TigerModel
from flowrec.training.trainer import GenerationCatalogMetricsAccumulator, Trainer, TrainerConfig
from scripts.run_tiger import build_data_config


def _tiger(*, use_user_token: bool = True, user_ids_count: int = 8) -> TigerModel:
    return TigerModel(
        codebook_size=2,
        sem_id_len=2,
        user_ids_count=user_ids_count,
        use_user_token=use_user_token,
        semantic_id_mapping={"a": [0, 1], "b": [1, 0]},
        item_popularity_mapping={"a": 3, "b": 7},
        embedding_dim=8,
        num_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        d_kv=8,
        dropout=0.0,
    )


def _encoder_batch() -> dict[str, torch.Tensor]:
    return {
        "user_ids": torch.tensor([1, 3]),
        "hashed_user_ids": torch.tensor([5, 7]),
        "history_token_ids": torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
        "history_token_mask": torch.tensor(
            [[[True, True], [True, False]], [[True, True], [False, False]]]
        ),
    }


def test_user_token_is_compatible_by_default_and_can_be_disabled() -> None:
    default_model = _tiger()
    history_model = _tiger(use_user_token=False)

    default_ids, default_mask = default_model._encoder_inputs(_encoder_batch())
    history_ids, history_mask = history_model._encoder_inputs(_encoder_batch())

    assert default_model.use_user_token is True
    assert default_ids.shape == default_mask.shape == (2, 5)
    assert torch.equal(default_ids[:, -1], torch.tensor([7, 9]))
    assert history_ids.shape == history_mask.shape == (2, 4)
    assert history_model.config_dict()["use_user_token"] is False


def test_semantic_only_vocabulary_works_without_a_user_token() -> None:
    model = _tiger(use_user_token=False, user_ids_count=0)
    input_ids, attention_mask = model._encoder_inputs(_encoder_batch())

    assert input_ids.shape == attention_mask.shape == (2, 4)
    assert model.config.vocab_size == model.config_dict()["semantic_vocab_size"] + 32


def test_user_token_requires_a_nonempty_user_vocabulary() -> None:
    with pytest.raises(ValueError, match="user_ids_count"):
        _tiger(use_user_token=True, user_ids_count=0)


class _CatalogModel(nn.Module):
    method_name = "sft"

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "item_semantic_ids",
            torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.long),
        )
        self.register_buffer(
            "item_popularity",
            torch.tensor([2.0, 10.0, 20.0]),
            persistent=False,
        )


def test_native_generation_metrics_include_arp_at_standard_cutoffs() -> None:
    generated = torch.tensor(
        [
            [[1, 1], [2, 2], [3, 3], [9, 9], [1, 1]] * 4,
            [[2, 2], [2, 2], [9, 9], [1, 1], [3, 3]] * 4,
        ],
        dtype=torch.long,
    )
    accumulator = GenerationCatalogMetricsAccumulator(_CatalogModel())
    accumulator.update(
        {
            "generated_token_ids": generated,
            "generation_topk": (5, 10, 20),
            "method_batch": {"user_ids": torch.tensor([7, 8])},
        }
    )

    metrics = accumulator.compute()

    for k in (5, 10, 20):
        assert f"sft/generation_arp@{k}" in metrics
        assert metrics[f"sft/generation_coverage@{k}"] == 1.0
    assert metrics["sft/generation_arp_valid_fraction@5"] == pytest.approx(0.8)


def test_arp_averages_multiple_contexts_per_user_before_users() -> None:
    accumulator = GenerationCatalogMetricsAccumulator(_CatalogModel())
    accumulator.update(
        {
            "generated_token_ids": torch.tensor(
                [[[1, 1]], [[3, 3]], [[2, 2]]], dtype=torch.long
            ),
            "generation_topk": (1,),
            "method_batch": {"user_ids": torch.tensor([7, 7, 8])},
        }
    )

    # user 7: mean(2, 20) = 11; user 8: 10; ARP = mean(11, 10).
    assert accumulator.compute()["sft/generation_arp@1"] == pytest.approx(10.5)


def test_tiger_popularity_is_catalog_aligned_and_not_checkpointed() -> None:
    model = _tiger()

    assert model.item_popularity.tolist() == [3.0, 7.0]
    assert "item_popularity" not in model.state_dict()


def test_build_data_config_forwards_batches_filters_and_cache(tmp_path: Path) -> None:
    config = {
        "data": {
            "reviews_path": str(tmp_path / "reviews.json.gz"),
            "processed_cache_dir": str(tmp_path / "cache"),
            "rebuild_processed_cache": True,
        },
        "dataset": {
            "max_sequence_length": 20,
            "use_verified_only": True,
            "keep_raw_reviews": True,
            "deduplicate_consecutive": True,
        },
        "dataloader": {
            "train_batch_size": 2048,
            "validation_batch_size": 1024,
            "test_batch_size": 64,
            "num_workers": 0,
        },
        "model": {"user_ids_count": 0},
    }

    data_config = build_data_config(config, config_path=tmp_path / "config.json")

    assert data_config.eval_batch_size == 1024
    assert data_config.test_batch_size == 64
    assert data_config.use_verified_only is True
    assert data_config.keep_raw_reviews is True
    assert data_config.deduplicate_consecutive is True
    assert data_config.processed_cache_dir == str(tmp_path / "cache")
    assert data_config.rebuild_processed_cache is True


def test_existing_single_eval_batch_setting_remains_shared(tmp_path: Path) -> None:
    config = {
        "data": {"reviews_path": str(tmp_path / "reviews.json.gz")},
        "dataset": {"max_sequence_length": 20},
        "dataloader": {"validation_batch_size": 128, "num_workers": 0},
        "model": {"user_ids_count": 8},
    }

    data_config = build_data_config(config, config_path=tmp_path / "config.json")

    assert data_config.eval_batch_size == 128
    assert data_config.test_batch_size is None


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    reviews_path = tmp_path / "reviews.jsonl.gz"
    semantic_ids_path = tmp_path / "semantic_ids.json"
    items = [f"item-{idx}" for idx in range(6)]
    with gzip.open(reviews_path, "wt", encoding="utf-8") as handle:
        for user_idx in range(2):
            for position, item in enumerate(items):
                handle.write(
                    json.dumps(
                        {
                            "user_id": f"user-{user_idx}",
                            "asin": item,
                            "timestamp": user_idx * 100 + position,
                            "rating": 5.0,
                        }
                    )
                    + "\n"
                )
    semantic_ids_path.write_text(
        json.dumps({item: [idx, idx + 1] for idx, item in enumerate(items)}),
        encoding="utf-8",
    )
    return reviews_path, semantic_ids_path


def test_processed_cache_preserves_interaction_store_pipeline(tmp_path: Path, monkeypatch) -> None:
    reviews_path, semantic_ids_path = _write_dataset(tmp_path)
    cfg = AmazonReviewDataConfig(
        reviews_path=reviews_path,
        semantic_id_path=semantic_ids_path,
        item_key="asin",
        filter_reviews_to_metadata=False,
        min_user_interactions=2,
        filter_cold_items_after_split=False,
        processed_cache_dir=tmp_path / "processed-cache",
        max_history_len=5,
    )
    original = AmazonReviewDataModule(cfg)
    original.setup()

    def fail_raw_load(*args, **kwargs):
        raise AssertionError("raw interactions should not be parsed on a cache hit")

    monkeypatch.setattr(datamodule_module, "load_interactions_frame", fail_raw_load)
    cached = AmazonReviewDataModule(cfg)
    cached.setup()

    assert cached.summary() == original.summary()
    assert len(cached.train_dataset) == len(original.train_dataset)
    assert cached.user2id == original.user2id
    assert cached.item2id == original.item2id
    assert cached.interaction_store is not None
    assert cached.interaction_store.to_summary_dict() == original.interaction_store.to_summary_dict()
    assert cached.interaction_store.events.equals(original.interaction_store.events)


def test_validation_and_test_loaders_use_independent_batch_sizes(tmp_path: Path) -> None:
    data = AmazonReviewDataModule(
        AmazonReviewDataConfig(
            reviews_path=tmp_path / "unused.json.gz",
            eval_batch_size=7,
            test_batch_size=3,
            num_workers=0,
        )
    )
    data.val_dataset = [{}]
    data.test_dataset = [{}]

    assert data.val_dataloader().batch_size == 7
    assert data.test_dataloader().batch_size == 3


class _TinyModel(FlowRecModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return RecModelOutput()


class _TinyMethod(FlowRecMethod):
    method_name = "tiny"

    def compute_loss(self, model, batch, *, stage="train", step=None):
        value = model.weight * batch["value"].float().mean()
        return MethodOutput(loss=value, metrics={"value": value.detach()})


def test_original_persistent_loader_can_be_evaluated_twice() -> None:
    model = _TinyModel()
    trainer = Trainer(
        model=model,
        method=_TinyMethod(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        cfg=TrainerConfig(use_amp=False, progress_bar=False),
        device="cpu",
    )
    loader = DataLoader(
        [{"value": torch.tensor(float(i))} for i in range(8)],
        batch_size=2,
        num_workers=1,
        persistent_workers=True,
    )

    assert trainer.evaluate(loader, stage="val")["val/value"] == pytest.approx(3.5)
    assert trainer.evaluate(loader, stage="val")["val/value"] == pytest.approx(3.5)
