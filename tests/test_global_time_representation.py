"""Catalog, serialization and split stability checks without downloaded models."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from flowrec.item_representation.base import ItemEmbeddingStore, SemanticIDStore
from flowrec.item_representation.semantic_id import RQVAEItemTokenizer
from scripts import global_time_representation as shared


def _load_script(stem):
    spec = importlib.util.spec_from_file_location(
        f"global_{stem}", shared.PROJECT_ROOT / "scripts/global-time" / f"{stem}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embeddings = _load_script("build_amazon_embeddings")
tokenizers = _load_script("train_tokenizer_all")


def _assets(root: Path, name: str, family: str = "amazon") -> Path:
    config_dir = root / "configs"
    config_dir.mkdir()
    config_name = {
        "amazon": "dataset_amazon_2023.yaml" if name == "beauty_2023" else "dataset_amazon.yaml",
        "yambda": "dataset_yambda.yaml",
        "vk_lsvd": "dataset_vk_lsvd.yaml",
    }[family]
    for source in shared.DEFAULT_CONFIG_DIR.glob("dataset_*.yaml"):
        config = yaml.safe_load(source.read_text())
        config["datasets"] = {}
        if source.name == config_name:
            selected = config
        (config_dir / source.name).write_text(yaml.safe_dump(config))
    # Cold item 3 passes the pre-split 2-core. Auxiliary item 9, metadata /
    # embedding-missing item 8, and singleton item 7 must all be excluded.
    rows = []
    for user in (1, 2):
        events = [(1, 1), (2, 1), (3, 2), (4, 1), (5, 2), (6, 9), (7, 8),
                  (10, 1), (11, 3 if user == 1 else 2),
                  (20, 2), (21, 3 if user == 1 else 1)]
        if user == 1:
            events.append((8, 7))
        for timestamp, item in events:
            rows.append({"user_id": user, "item_id": item, "timestamp": timestamp})
    frame = pd.DataFrame(rows)
    source_path = root / "interactions.parquet"
    selected.update(min_user_interactions=2, min_item_interactions=2,
                    num_workers=0, persistent_workers=False, pin_memory=False)
    entry = {"global_val_timestamp": 10, "global_test_timestamp": 20}
    if family == "amazon":
        key = "parent_asin" if name == "beauty_2023" else "asin"
        frame[key] = frame.item_id.astype(str)
        frame["verified_purchase"] = frame.item_id != 9
        frame["rating"] = 5.0
        metadata_path = root / "metadata.parquet"
        pd.DataFrame({key: ["1", "2", "3", "7", "9"],
                      "title": ["one", "two", "three", "seven", "nine"]}).to_parquet(metadata_path)
        entry.update(reviews_path=str(source_path), metadata_path=str(metadata_path),
                     item_key=key, metadata_item_key=key)
    elif family == "yambda":
        frame["uid"] = frame.user_id
        frame["event_type"] = "listen"
        frame["played_ratio_pct"] = np.where(frame.item_id == 9, 10, 100)
        frame["track_length_seconds"] = 100
        frame["is_organic"] = True
        emb_path = root / "embeddings.parquet"
        pd.DataFrame({"item_id": [1, 2, 3, 7, 9],
                      "embed": [[float(i), 1., 2., 3.] for i in [1, 2, 3, 7, 9]]}).to_parquet(emb_path)
        selected.update(embeddings_path=str(emb_path), hf_recent_months=None)
        entry.update(interactions_path=str(source_path), dataset_size="50m")
    else:
        frame["timespent"] = np.where(frame.item_id == 9, 0, 10)
        frame["dislike"] = False
        for split, mask in (("train", frame.timestamp < 10),
                            ("val", (frame.timestamp >= 10) & (frame.timestamp < 20)),
                            ("test", frame.timestamp >= 20)):
            path = root / f"{split}.parquet"
            frame.loc[mask].to_parquet(path)
            entry[f"{split}_paths"] = [str(path)]
        emb_path = root / "embeddings.npz"
        np.savez(emb_path, item_id=np.array([1, 2, 3, 7, 9]),
                 embedding=np.arange(20, dtype=np.float32).reshape(5, 4))
        selected["embeddings_path"] = str(emb_path)
        entry.pop("global_val_timestamp")
        entry.pop("global_test_timestamp")
    frame.to_parquet(source_path)
    selected["datasets"] = {name: entry}
    (config_dir / config_name).write_text(yaml.safe_dump(selected))
    return config_dir


def _target(config_dir, name, embeddings_dir=None):
    return shared.collect_targets(config_dir, embeddings_dir or config_dir, [name])[0]


def test_default_targets_and_representation_settings_are_shared():
    targets = shared.collect_targets(shared.DEFAULT_CONFIG_DIR, shared.DEFAULT_EMBEDDINGS_DIR, None)
    assert [t.name for t in targets if t.family == "amazon"] == ["beauty", "instruments", "beauty_2023"]
    assert len(targets) == 7
    assert shared.DEFAULT_SEMANTIC_CONFIG == shared.embedding_script.DEFAULT_CONFIG_DIR / "semantic_ids.yaml"
    for path in (shared.DEFAULT_EMBEDDINGS_DIR, shared.DEFAULT_SEMANTIC_IDS_DIR, shared.DEFAULT_REPORT_PATH):
        assert "global-time" in path.parts


def test_exp2_targets_do_not_require_a_vk_config():
    targets = shared.collect_targets(
        shared.DEFAULT_CONFIG_DIR.parent / "exp2-tuned-params",
        shared.DEFAULT_EMBEDDINGS_DIR, None,
    )
    assert len(targets) == 5
    assert {target.family for target in targets} == {"amazon", "yambda"}


@pytest.mark.parametrize("name", ["beauty", "instruments", "beauty_2023"])
def test_amazon_embeddings_use_final_catalog(tmp_path, name):
    config_dir = _assets(tmp_path, name)
    output_dir = tmp_path / "embeddings"
    embeddings.main(["--config-dir", str(config_dir), "--datasets", name,
                     "--output-dir", str(output_dir), "--embedder", "hashing", "--hashing-dim", "8"])
    store = ItemEmbeddingStore.load(output_dir / f"{name}_content_embeddings.pt")
    assert set(store.item_ids) == {"1", "2"}
    assert store.embeddings.shape == (2, 8)
    assert store.metadata["catalog_summary"]["global_val_cutoff"] == 10
    assert store.metadata["catalog_scope"] == "post_split_train"


@pytest.mark.parametrize("family,name", [("amazon", "beauty"), ("yambda", "yambda_50m_listens"),
                                         ("vk_lsvd", "vk_lsvd_ur0.01_ir0.01")])
def test_reused_ids_preserve_tokens_and_evaluation_targets(tmp_path, family, name):
    config_dir = _assets(tmp_path, name, family)
    source_dir, output_dir = tmp_path / "source", tmp_path / "sids"
    original = {"1": [0, 256, 512, 771], "2": [0, 256, 512, 779], "3": [2, 257, 516, 768]}
    SemanticIDStore(original).save_json(source_dir / name / "semantic_ids.json")
    target = _target(config_dir, name)
    data = shared.build_catalog_datamodule(target)
    assert set(data.item2id) == {"1", "2"}
    tokenizers.main(["--config-dir", str(config_dir), "--datasets", name,
                     "--output-dir", str(output_dir), "--report-path", str(tmp_path / "report.json"),
                     "--reuse-semantic-ids-dir", str(source_dir)])
    sid_path = output_dir / name / "semantic_ids.json"
    assert SemanticIDStore.load_json(sid_path).semantic_ids == {k: original[k] for k in ("1", "2")}
    assert not (output_dir / name / "model.pt").exists()
    loaded = type(data)(replace(data.cfg, semantic_id_path=sid_path))
    loaded.setup()
    assert loaded.item2id == data.item2id
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(loaded, f"{split}_dataset").examples,
                                      getattr(data, f"{split}_dataset").examples)
    # Cold last targets are dropped, never replaced by earlier warm targets.
    assert len(data.val_dataset.examples) == 1
    assert len(data.test_dataset.examples) == 1
    entry = json.loads((output_dir / name / "catalog.json").read_text())
    assert entry["tokenizer_fit_scope"] == "unchanged_from_source"


@pytest.mark.parametrize("family,name", [("amazon", "beauty"), ("yambda", "yambda_50m_listens"),
                                         ("vk_lsvd", "vk_lsvd_ur0.01_ir0.01")])
def test_train_and_reload_tokenizer(tmp_path, family, name):
    config_dir = _assets(tmp_path, name, family)
    emb_dir = tmp_path / "embeddings"
    if family == "amazon":
        # Extra pre-split embeddings are harmless: fitting still uses only the final catalog.
        ItemEmbeddingStore(["1", "2", "3"], torch.eye(3, 4)).save(emb_dir / f"{name}_content_embeddings.pt")
    settings = yaml.safe_load(shared.DEFAULT_SEMANTIC_CONFIG.read_text())
    settings.update(rqvae_epochs=1, rqvae_hidden_dims=[4], rqvae_latent_dim=2,
                    num_codebooks=1, codebook_size=2, rqvae_batch_size=2, rqvae_val_ratio=0.)
    semantic_path = tmp_path / "semantic.yaml"
    semantic_path.write_text(yaml.safe_dump(settings))
    output_dir = tmp_path / "tokenizers"
    tokenizers.main(["--config-dir", str(config_dir), "--datasets", name,
                     "--semantic-config", str(semantic_path), "--embeddings-dir", str(emb_dir),
                     "--output-dir", str(output_dir), "--report-path", str(tmp_path / "report.json"),
                     "--device", "cpu"])
    saved = output_dir / name
    for filename in ("model.pt", "normalizer.pt", "config.json", "training_log.jsonl",
                     "semantic_ids.json", "semantic_ids.pt", "catalog.json"):
        assert (saved / filename).is_file()
    loaded = RQVAEItemTokenizer.load(saved)
    assert set(loaded.semantic_id_store.semantic_ids) == {"1", "2"}
    entry = json.loads((saved / "catalog.json").read_text())
    assert entry["num_catalog_items"] == entry["num_items"] == 2
    assert entry["catalog_scope"] == "post_split_train"


def test_reuse_rejects_missing_training_ids(tmp_path):
    config_dir = _assets(tmp_path, "beauty")
    source_dir = tmp_path / "source"
    SemanticIDStore({"1": [1, 2]}).save_json(source_dir / "beauty/semantic_ids.json")
    with pytest.raises(ValueError, match="existing SIDs miss 1 training items"):
        tokenizers.export_existing_ids(_target(config_dir, "beauty"), source_dir=source_dir,
                                       output_dir=tmp_path / "out", seed=42)


def test_catalog_rejects_non_temporal_config():
    target = _target(shared.embedding_script.DEFAULT_CONFIG_DIR, "beauty")
    with pytest.raises(ValueError, match="expected a temporal split"):
        shared.build_catalog_datamodule(target)


def test_embeddings_require_full_training_catalog_coverage():
    with pytest.raises(ValueError, match="Embeddings missing for 1 training items"):
        shared.restrict_embeddings(ItemEmbeddingStore(["1"], torch.ones(1, 2)), {"1", "2"})
