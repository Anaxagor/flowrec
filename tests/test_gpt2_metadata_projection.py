"""Amazon fine-tuning reads catalog IDs while reusing existing reward factors."""

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon_fullhist as fullhist

from flowrec.data.datamodule import AmazonReviewDataConfig
from flowrec.data.dense_prefix import dense_prefix_signature
from flowrec.ft_methods.gflow.factorized_cache import CACHE_KIND, FACTORS_FILENAME

g = fullhist.base
AMAZON_DATASETS = ("beauty", "instruments", "beauty_2023")


@pytest.mark.parametrize("dataset", AMAZON_DATASETS)
@pytest.mark.parametrize("mode", [[], ["--sft-only"], ["--cache-only"]],
                         ids=["gflow", "sft", "cache"])
def test_amazon_entrypoints_and_reward_builders_project_metadata(dataset, mode, tmp_path):
    args = fullhist.parse_args(["--datasets", dataset, *mode])
    config = g.build_run_config(dataset, args, user_ids_count=1)
    data = g.tiger_runner.build_data_config(config, config_path=tmp_path / "run.json")
    assert config["dataset"]["metadata_ids_only"] is True
    assert data.metadata_ids_only and data.filter_reviews_to_metadata

    # Cache subprocesses reconstruct their own datamodule from these configs.
    for component in config["method"]["component_keys"]:
        cached = g.cache_config(config, component)
        cache_data = g.tiger_runner.build_data_config(cached, config_path=tmp_path / "cache.json")
        assert asdict(cache_data) == asdict(data)


@pytest.mark.parametrize("dataset", ["yambda_50m_listens", "yambda_500m_listens"])
def test_yambda_configuration_does_not_acquire_amazon_metadata_option(dataset, tmp_path):
    args = fullhist.parse_args(["--datasets", dataset])
    config = g.build_run_config(dataset, args, user_ids_count=1)
    data = g.tiger_runner.build_data_config(config, config_path=tmp_path / "run.json")
    assert "metadata_ids_only" not in config["dataset"]
    assert "metadata_ids_only" not in asdict(data)


@pytest.mark.parametrize("dataset", AMAZON_DATASETS)
def test_base_gpt2_data_and_default_amazon_config_keep_full_metadata(dataset):
    assert not AmazonReviewDataConfig(reviews_path="reviews.parquet").metadata_ids_only
    path = g.dataset_yaml_path(dataset)
    group = next(group for group in g.gpt2_runner.GROUPS if group.data_config_name == path.name)
    reference, _ = g.gpt2_runner.build_data_config(
        group, dataset, g.runner.load_yaml(path), seed=42,
        semantic_ids_dir=g.SEMANTIC_IDS_ROOT, user_hash_mod=1,
    )
    assert not reference.metadata_ids_only


@pytest.mark.parametrize("component,mode", [("sasrec", None), ("interaction", None),
                                            ("sim_hier", "hierarchical")])
def test_metadata_projection_reuses_old_factor_manifests_without_recalculation(
    tmp_path, monkeypatch, component, mode,
):
    args = fullhist.parse_args([
        "--datasets", "beauty_2023", "--num-candidates", "3",
        "--cache-root", str(tmp_path / "cache"), "--require-existing-caches",
    ])
    config = g.build_run_config("beauty_2023", args, user_ids_count=1)
    old_config = copy.deepcopy(config)
    del old_config["dataset"]["metadata_ids_only"]
    old_data = asdict(g.tiger_runner.build_data_config(old_config, config_path=tmp_path / "old.json"))
    del old_data["metadata_ids_only"]  # Existing manifests predate this option.
    assert dense_prefix_signature(config["dataset"]) == dense_prefix_signature(old_config["dataset"])
    cache_dir = g.component_cache_dir("beauty_2023", component, args)
    cache_dir.mkdir(parents=True)
    # No factors are read by the manifest compatibility check itself.
    torch.save({"target_semantic_row_ids": torch.tensor([0, 1])}, cache_dir / FACTORS_FILENAME)
    manifest = {
        "kind": CACHE_KIND, "num_examples": 2, "num_catalog_items": 2,
        "component": {"name": "similarity" if mode else component, "mode": mode},
        "data_config": old_data,
        "training_context": dense_prefix_signature(old_config["dataset"]),
        **g.finetune_subset_spec(args),
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, default=str))
    before = manifest_path.read_bytes()
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: pytest.fail("Must reuse calculated factors"))

    actual = g.ensure_component_cache("beauty_2023", component, config, tmp_path / "config", 2, args)
    assert actual == json.loads(before)
    assert manifest_path.read_bytes() == before
    generated = json.loads((tmp_path / "config" / f"cache_config_{component}.json").read_text())
    assert generated["dataset"]["metadata_ids_only"] is True

    # The memory-only option must not weaken checks against incompatible contexts.
    incompatible = copy.deepcopy(config)
    incompatible["dataset"]["max_history_len"] += 1
    with pytest.raises(RuntimeError, match="different training_context"):
        g.ensure_component_cache("beauty_2023", component, incompatible,
                                 tmp_path / "config", 2, args)
