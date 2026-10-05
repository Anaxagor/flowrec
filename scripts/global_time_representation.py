"""Shared catalog selection for the global-time representation scripts.

Use the existing dataset config builders and datamodules, including their
preprocessing order and post-split cold-item filter. No old semantic IDs are
loaded while deriving the catalog for a new tokenizer.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, replace
from pathlib import Path

from flowrec.data.datamodule import AmazonReviewDataModule
from flowrec.data.vk_lsvd import VKLSVDDataModule
from flowrec.data.yambda import YambdaDataModule
from flowrec.item_representation.base import ItemEmbeddingStore
from scripts import sasrec_config_runner as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp1-standard_params"
DEFAULT_SEMANTIC_CONFIG = PROJECT_ROOT / "configs/exp1-standard_params/semantic_ids.yaml"
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "datasets/embeddings/global-time/exp1-standard_params"
DEFAULT_SEMANTIC_IDS_DIR = PROJECT_ROOT / "datasets/semantic_ids/global-time/exp1-standard_params"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts/global-time/exp1-standard_params/semantic_id_training_report.json"


def _load_exp1_script(stem: str):
    name = f"_global_time_{stem}"
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "scripts/exp1-standard-params" / f"{stem}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


embedding_script = _load_exp1_script("build_amazon_embeddings")
tokenizer_script = _load_exp1_script("train_tokenizer_all")


def collect_targets(config_dir: Path, embeddings_dir: Path, requested: list[str] | None):
    selected = None if requested is None or "all" in requested else set(requested)
    targets, available = [], []
    for family, filename in (
        ("amazon", "dataset_amazon.yaml"),
        ("amazon", "dataset_amazon_2023.yaml"),
        ("yambda", "dataset_yambda.yaml"),
        ("vk_lsvd", "dataset_vk_lsvd.yaml"),
    ):
        path = config_dir / filename
        # Exp2 has no VK-LSVD config. Enumerate the datasets present in the tree.
        if not path.is_file():
            continue
        config = runner.load_yaml(path)
        for name in config.get("datasets", {}):
            available.append(name)
            if selected is None or name in selected:
                targets.append(tokenizer_script.TokenizerTarget(
                    name=name, family=family, dataset_config=config,
                    embeddings_path=(
                        embeddings_dir / f"{name}_content_embeddings.pt"
                        if family == "amazon" else None
                    ),
                ))
    if selected is not None and selected - set(available):
        raise KeyError(f"Unknown datasets {sorted(selected - set(available))}; available: {available}")
    if not targets:
        raise ValueError(f"No datasets selected in {config_dir}")
    return targets


def validate_catalog_inputs(targets) -> None:
    builders = tokenizer_script.run_sasrec_all
    validators = {
        "amazon": builders.validate_amazon_inputs,
        "yambda": builders.validate_yambda_inputs,
        "vk_lsvd": builders.validate_vk_lsvd_inputs,
    }
    for target in targets:
        validators[target.family]([target.name], target.dataset_config)


def build_catalog_datamodule(target, *, seed: int = 42):
    builders = tokenizer_script.run_sasrec_all
    factory, config_builder = {
        "amazon": (AmazonReviewDataModule, builders.build_amazon_data_config),
        "yambda": (YambdaDataModule, builders.build_yambda_data_config),
        "vk_lsvd": (VKLSVDDataModule, builders.build_vk_lsvd_data_config),
    }[target.family]
    cfg = config_builder(target.name, target.dataset_config, seed=seed)
    # Representation training must not depend on an earlier tokenizer's coverage.
    cfg = replace(cfg, semantic_id_path=None)
    allowed_splits = {"global_time", "provided"} if target.family == "vk_lsvd" else {"global_time"}
    if cfg.split_strategy not in allowed_splits or not cfg.filter_cold_items_after_split:
        raise ValueError(
            f"{target.name}: expected a temporal split with filter_cold_items_after_split=true"
        )
    print(f"[{target.name}] deriving the catalog after preprocessing and {cfg.split_strategy} splitting")
    data = factory(cfg)
    data.setup()
    if not data.item2id:
        raise ValueError(f"{target.name}: no training items remain after preprocessing and splitting")
    return data


def catalog_metadata(data) -> dict:
    return {
        "catalog_scope": "post_split_train",
        "catalog_config": runner.to_jsonable(asdict(data.cfg)),
        "catalog_summary": data.summary(),
    }


def restrict_embeddings(store: ItemEmbeddingStore, item_ids: set[str]) -> ItemEmbeddingStore:
    """Preserve embedding order and require coverage of the final catalog."""
    missing = item_ids - set(store.item_ids)
    if missing:
        raise ValueError(
            f"Embeddings missing for {len(missing)} training items; examples: {sorted(missing)[:5]}"
        )
    if store.num_items == len(item_ids):
        return store
    return store.subset(item_ids)
