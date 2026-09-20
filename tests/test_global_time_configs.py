"""Parity guard for the global-time-split config tree (configs/global-time).

The tree duplicates the leave-one-out configs because the runners read every
file from one ``--config-dir``. These tests keep the copies honest: the two
trees may differ ONLY in the split block, the pinned cutoffs and the output
roots, and every global-time output root must be disjoint from the
leave-one-out one.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import sasrec_config_runner as common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "configs"
GLOBAL_TIME = PROJECT_ROOT / "configs" / "global-time"
EXP1 = "exp1-standard_params"
EXP2 = "exp2-tuned-params"

DATASET_YAMLS = ("dataset_amazon.yaml", "dataset_amazon_2023.yaml", "dataset_yambda.yaml")
MODEL_YAMLS = ("gpt2_model.yaml", "sasrec_model.yaml", "tiger_model.yaml")
TRAINER_YAMLS = (
    "trainer_gpt2.yaml",
    "trainer_sasrec.yaml",
    "trainer_sasrec_amazon_2023.yaml",
    "trainer_sasrec_vk_lsvd.yaml",
    "trainer_sasrec_yambda.yaml",
    "trainer_tiger.yaml",
)
# Shared dataset-YAML keys the global-time tree is allowed to change or add.
SPLIT_KEYS = {
    "split_strategy",
    "global_history_policy",
    "global_eval_target_mode",
    "filter_cold_items_after_split",
}
CUTOFF_KEYS = {"global_val_timestamp", "global_test_timestamp"}
OUTPUT_KEYS = {"checkpoints_dir", "experiments_dir", "artifacts_dir", "output_dir"}
OUTPUT_ROOTS = ("checkpoints/", "experiments/", "artifacts/")
DATASETS = ("beauty", "instruments", "beauty_2023", "yambda_50m_listens", "yambda_500m_listens")


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The dataclasses in the scripts resolve their module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_sasrec_all = _load_script("run_sasrec_all", "scripts/exp1-standard-params/run_sasrec_all.py")
run_tiger_sft_all = _load_script("run_tiger_sft_all", "scripts/exp1-standard-params/run_tiger_sft_all.py")
run_gpt2_sft_all = _load_script("run_gpt2_sft_all", "scripts/exp1-standard-params/run_gpt2_sft_all.py")


def _dataset_cutoffs(tree: str) -> dict[str, dict[str, int]]:
    cutoffs: dict[str, dict[str, int]] = {}
    for name in DATASET_YAMLS:
        for dataset, entry in common.load_yaml(GLOBAL_TIME / tree / name)["datasets"].items():
            cutoffs[dataset] = {key: entry[key] for key in sorted(CUTOFF_KEYS)}
    return cutoffs


def _reroot(value: Any) -> Any:
    """The documented output-root rule: <root>/... -> <root>/global-time/..."""
    if isinstance(value, dict):
        return {key: _reroot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_reroot(item) for item in value]
    if isinstance(value, str) and value.startswith(OUTPUT_ROOTS) and " " not in value:
        head, rest = value.split("/", 1)
        return f"{head}/global-time/{rest}"
    return value


@pytest.mark.parametrize("tree", [EXP1, EXP2])
@pytest.mark.parametrize("name", DATASET_YAMLS)
def test_dataset_yamls_differ_only_in_the_split(tree: str, name: str) -> None:
    base = common.load_yaml(BASE / tree / name)
    variant = common.load_yaml(GLOBAL_TIME / tree / name)

    assert variant["split_strategy"] == "global_time"
    assert variant["global_history_policy"] == "rolling"
    assert variant["global_eval_target_mode"] == "last_per_user"
    assert variant["filter_cold_items_after_split"] is True

    assert set(variant["datasets"]) == set(base["datasets"])
    for dataset, entry in variant["datasets"].items():
        assert CUTOFF_KEYS <= set(entry), f"{dataset}: cutoffs must be pinned"
        assert entry["global_val_timestamp"] < entry["global_test_timestamp"]
        stripped = {key: value for key, value in entry.items() if key not in CUTOFF_KEYS}
        assert stripped == base["datasets"][dataset]

    def shared(config: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in config.items() if k not in SPLIT_KEYS | {"datasets"}}

    assert shared(variant) == shared(base)


def test_exp1_and_exp2_pin_the_same_cutoffs() -> None:
    assert _dataset_cutoffs(EXP1) == _dataset_cutoffs(EXP2)
    assert set(_dataset_cutoffs(EXP1)) == set(DATASETS)


def test_vk_lsvd_keeps_its_provided_temporal_split() -> None:
    name = "dataset_vk_lsvd.yaml"
    variant = common.load_yaml(GLOBAL_TIME / EXP1 / name)
    assert variant == common.load_yaml(BASE / EXP1 / name)
    assert variant["split_strategy"] == "provided"


@pytest.mark.parametrize("name", MODEL_YAMLS)
def test_model_yamls_are_verbatim_copies(name: str) -> None:
    assert common.load_yaml(GLOBAL_TIME / EXP1 / name) == common.load_yaml(BASE / EXP1 / name)


@pytest.mark.parametrize("name", TRAINER_YAMLS)
def test_trainer_yamls_differ_only_in_output_roots(name: str) -> None:
    base = common.load_yaml(BASE / EXP1 / name)
    variant = common.load_yaml(GLOBAL_TIME / EXP1 / name)
    assert variant == _reroot(base)

    def output_dirs(config: dict[str, Any]) -> list[str]:
        found = [config[key] for key in OUTPUT_KEYS if key in config]
        for block in config.get("variants", {}).values():
            found.extend(block[key] for key in OUTPUT_KEYS if key in block)
        return found

    assert output_dirs(variant), name
    for path in output_dirs(variant):
        assert path.split("/")[1] == "global-time", path
    assert not set(output_dirs(variant)) & set(output_dirs(base))


def test_the_tokenizer_config_is_deliberately_absent() -> None:
    # The existing semantic IDs cover the whole pre-split catalog and are
    # reused. Retraining through this tree would tokenize only the
    # train-observed Yambda catalog, and TIGER / GPT-2 would then drop the
    # cold items BEFORE the split while SASRec drops them after it.
    assert not (GLOBAL_TIME / EXP1 / "semantic_ids.yaml").exists()


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_exp1_runner_resolves_the_same_global_time_split(dataset: str) -> None:
    config_dir = GLOBAL_TIME / EXP1
    expected = _dataset_cutoffs(EXP1)[dataset]
    semantic_ids_dir = PROJECT_ROOT / "datasets" / "semantic_ids" / EXP1

    sasrec_plan = next(
        plan for plan in run_sasrec_all.build_group_plans(config_dir, [dataset])
    )
    builder = {
        "amazon": run_sasrec_all.build_amazon_data_config,
        "amazon_2023": run_sasrec_all.build_amazon_data_config,
        "yambda": run_sasrec_all.build_yambda_data_config,
    }[sasrec_plan.group.key]
    configs = {"sasrec": builder(dataset, sasrec_plan.dataset_config, seed=42)}
    for model, module in (("tiger", run_tiger_sft_all), ("gpt2", run_gpt2_sft_all)):
        plan = next(plan for plan in module.build_group_plans(config_dir, [dataset]))
        built = module.build_data_config(
            plan.group,
            dataset,
            plan.dataset_config,
            seed=42,
            semantic_ids_dir=semantic_ids_dir,
            user_hash_mod=1,
        )
        configs[model] = built[0]

    for model, cfg in configs.items():
        assert cfg.split_strategy == "global_time", model
        assert cfg.global_val_timestamp == expected["global_val_timestamp"], model
        assert cfg.global_test_timestamp == expected["global_test_timestamp"], model
        assert cfg.global_history_policy == "rolling", model
        assert cfg.global_eval_target_mode == "last_per_user", model
        assert cfg.filter_cold_items_after_split is True, model
    assert configs["sasrec"].train_example_layout == "dense_causal"
    assert configs["tiger"].train_example_layout == "prefix_target"


def test_finetune_protocol_matches_the_leave_one_out_protocol() -> None:
    name = "finetune_base_matched.json"
    base = json.loads((BASE / EXP2 / name).read_text(encoding="utf-8"))
    variant = json.loads((GLOBAL_TIME / EXP2 / name).read_text(encoding="utf-8"))

    # Split-independent sections are identical ...
    for section in ("shared_schedule", "gflow_grid", "cluster_resources"):
        assert variant[section] == base[section], section
    assert set(variant["datasets"]) == set(base["datasets"]) == set(DATASETS)
    for dataset, block in variant["datasets"].items():
        expected = copy.deepcopy(base["datasets"][dataset])
        expected.pop("base_sft_data")
        actual = copy.deepcopy(block)
        actual.pop("base_sft_data")
        assert actual == expected, dataset
    # ... and every path moves under global-time/, nothing else changes.
    for section in ("arms", "teacher_free_experiment"):
        assert variant[section] == _reroot(base[section]), section

    assert variant["split"]["cutoffs"] == _dataset_cutoffs(EXP2)
    shared = common.load_yaml(GLOBAL_TIME / EXP2 / "dataset_yambda.yaml")
    for key in SPLIT_KEYS:
        assert variant["split"][key] == shared[key], key


def test_finetune_protocol_outputs_never_touch_leave_one_out_results() -> None:
    variant = json.loads(
        (GLOBAL_TIME / EXP2 / "finetune_base_matched.json").read_text(encoding="utf-8")
    )

    def paths(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [path for item in value.values() for path in paths(item)]
        if isinstance(value, list):
            return [path for item in value for path in paths(item)]
        if isinstance(value, str) and value.startswith(OUTPUT_ROOTS) and " " not in value:
            return [value]
        return []

    found = paths(variant["arms"]) + paths(variant["teacher_free_experiment"])
    assert found
    for path in found:
        assert path.split("/")[1] == "global-time", path


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/raw/dataframe_format/reviews_beauty_2014.parquet").is_file()
    or not (PROJECT_ROOT / "datasets/semantic_ids" / EXP1 / "beauty/semantic_ids.json").is_file(),
    reason="needs the local Beauty 2014 parquet files and exp1 semantic IDs",
)
def test_beauty_models_share_one_evaluation_set() -> None:
    """SASRec (dense, no semantic IDs) and TIGER (prefix rows, semantic IDs)
    must score the same users on the same targets: the reused semantic IDs
    cover the whole pre-split catalog, so nothing is dropped before the split."""
    from dataclasses import replace

    from flowrec.data.datamodule import AmazonReviewDataModule
    from flowrec.data.sequence_builder import SequenceSplitConfig, resolve_global_time_cutoffs

    config_dir = GLOBAL_TIME / EXP1
    sasrec_plan = next(plan for plan in run_sasrec_all.build_group_plans(config_dir, ["beauty"]))
    sasrec_cfg = run_sasrec_all.build_amazon_data_config("beauty", sasrec_plan.dataset_config, seed=42)
    tiger_plan = next(plan for plan in run_tiger_sft_all.build_group_plans(config_dir, ["beauty"]))
    tiger_cfg, _ = run_tiger_sft_all.build_data_config(
        tiger_plan.group,
        "beauty",
        tiger_plan.dataset_config,
        seed=42,
        semantic_ids_dir=PROJECT_ROOT / "datasets" / "semantic_ids" / EXP1,
        user_hash_mod=1,
    )

    targets: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for model, cfg in (("sasrec", sasrec_cfg), ("tiger", tiger_cfg)):
        data = AmazonReviewDataModule(replace(cfg, num_workers=0, persistent_workers=False))
        data.setup()
        assert data.global_val_cutoff == cfg.global_val_timestamp
        assert data.global_test_cutoff == cfg.global_test_timestamp
        # The pinned cutoffs ARE the study-wide rule (80% / 90% interaction
        # quantiles) evaluated on this stream, not hand-picked dates.
        rule = SequenceSplitConfig(
            split_strategy="global_time", global_val_ratio=0.1, global_test_ratio=0.1
        )
        assert resolve_global_time_cutoffs(data.interactions, cfg=rule) == (
            cfg.global_val_timestamp,
            cfg.global_test_timestamp,
        )
        targets[model] = {
            split: sorted(
                zip(frame["raw_user_id"].astype(str), frame["target_raw_item"].astype(str))
            )
            for split, frame in (("val", data.val_dataset.examples), ("test", data.test_dataset.examples))
        }
        # One target per user and window.
        for pairs in targets[model].values():
            assert len({user for user, _ in pairs}) == len(pairs) > 0

    assert targets["sasrec"] == targets["tiger"]
