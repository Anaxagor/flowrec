"""VK-LSVD in the exp1 pipeline: config resolution for every runner and a CPU
smoke run of tokenizer -> SASRec -> TIGER -> GPT-2 on a synthetic subsample."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flowrec.data.vk_lsvd import VKLSVDDataConfig
from scripts import sasrec_config_runner as common

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "exp1-standard-params"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vk_lsvd_smoke_assets import SMOKE_DATASET, build_smoke_config_dir  # noqa: E402

VK_DATASETS = ["vk_lsvd_ur0.01_ir0.01", "vk_lsvd_ur0.01_ip0.01"]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_sasrec_all = _load_script("run_sasrec_all")
run_tiger_sft_all = _load_script("run_tiger_sft_all")
run_gpt2_sft_all = _load_script("run_gpt2_sft_all")
train_tokenizer_all = _load_script("train_tokenizer_all")


# ---------------------------------------------------------------------------
# Config resolution against the committed exp1 YAMLs
# ---------------------------------------------------------------------------


def test_sasrec_runner_has_a_vk_lsvd_group_with_the_provided_split() -> None:
    plans = run_sasrec_all.build_group_plans(run_sasrec_all.DEFAULT_CONFIG_DIR, None)
    plan = next(plan for plan in plans if plan.group.key == "vk_lsvd")
    assert plan.dataset_names == VK_DATASETS
    assert plan.group.trainer_config_name == "trainer_sasrec_vk_lsvd.yaml"

    for name in VK_DATASETS:
        cfg = run_sasrec_all.build_vk_lsvd_data_config(name, plan.dataset_config, seed=43)
        assert isinstance(cfg, VKLSVDDataConfig)
        assert cfg.use_hf is False
        assert cfg.interactions_root == (
            PROJECT_ROOT / "data/raw/VK-LSVD/subsamples" / name.removeprefix("vk_lsvd_")
        )
        assert cfg.embeddings_path == PROJECT_ROOT / "data/raw/VK-LSVD/item_embeddings.npz"
        assert cfg.drop_items_missing_embedding is True
        assert cfg.positive_criterion == "timespent >= 1 and not dislike"
        assert cfg.split_strategy == "provided"
        assert cfg.global_eval_target_mode == "last_per_user"
        assert cfg.train_example_layout == "dense_causal"
        assert cfg.history_padding == "left"
        assert cfg.include_raw_in_batch is False
        assert cfg.max_history_len == 50
        assert cfg.min_user_interactions == 5 and cfg.min_item_interactions == 5
        assert cfg.deduplicate_consecutive is True
        assert cfg.filter_cold_items_after_split is True
        assert cfg.seed == 43
    run_sasrec_all.validate_train_layout([plan])

    checkpoint_root = common.configured_path(plan.trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT)
    assert checkpoint_root == PROJECT_ROOT / "checkpoints/exp1-standard_params/sasrec/vk_lsvd"
    assert plan.trainer_config["metrics_summary_name"] == "sasrec_{dataset}_metrics_summary.csv"
    assert plan.trainer_config["save_interaction_store"] is False
    sce = run_sasrec_all.build_group_plans(run_sasrec_all.DEFAULT_CONFIG_DIR, VK_DATASETS, variant="sce")
    assert "sasrec_sce" in str(common.configured_path(sce[0].trainer_config, "checkpoints_dir", project_root=PROJECT_ROOT))


def test_vk_lsvd_preflight_reports_missing_root_and_files(tmp_path: Path) -> None:
    plan = next(
        plan
        for plan in run_sasrec_all.build_group_plans(run_sasrec_all.DEFAULT_CONFIG_DIR, None)
        if plan.group.key == "vk_lsvd"
    )
    dataset = json.loads(json.dumps(plan.dataset_config))
    missing_root = tmp_path / "nowhere"
    dataset["datasets"] = {"vk": {"interactions_root": str(missing_root)}}
    with pytest.raises(FileNotFoundError, match="nowhere"):
        run_sasrec_all.validate_vk_lsvd_inputs(["vk"], dataset)

    root = tmp_path / "subsample"
    for split, week in (("train", 0), ("validation", 25), ("test", 26)):
        (root / split).mkdir(parents=True)
        (root / split / f"week_{week:02}.parquet").touch()
    dataset["datasets"] = {"vk": {"interactions_root": str(root)}}
    npz = tmp_path / "item_embeddings.npz"
    dataset["embeddings_path"] = str(npz)
    with pytest.raises(FileNotFoundError, match="item_embeddings.npz"):
        run_sasrec_all.validate_vk_lsvd_inputs(["vk"], dataset)
    npz.touch()
    run_sasrec_all.validate_vk_lsvd_inputs(["vk"], dataset)

    dataset["use_hf"] = True
    with pytest.raises(ValueError, match="offline-only"):
        run_sasrec_all.validate_vk_lsvd_inputs(["vk"], dataset)


@pytest.mark.parametrize(
    ("script", "layout", "padding"),
    [(run_tiger_sft_all, "prefix_target", "left"), (run_gpt2_sft_all, "dense_causal", "right")],
)
def test_generative_runners_resolve_vk_lsvd_configs(script, layout: str, padding: str) -> None:
    plans = script.build_group_plans(script.DEFAULT_CONFIG_DIR, VK_DATASETS)
    assert [plan.group.key for plan in plans] == ["vk_lsvd"]
    plan = plans[0]
    assert plan.dataset_names == VK_DATASETS
    for name in VK_DATASETS:
        cfg, test_eval_batch_size = script.build_data_config(
            plan.group,
            name,
            plan.dataset_config,
            seed=42,
            semantic_ids_dir=script.DEFAULT_SEMANTIC_IDS_DIR,
            user_hash_mod=1,
        )
        assert isinstance(cfg, VKLSVDDataConfig)
        assert cfg.train_example_layout == layout
        assert cfg.history_padding == padding
        assert cfg.split_strategy == "provided"
        assert cfg.positive_criterion == "timespent >= 1 and not dislike"
        assert cfg.semantic_id_path == script.DEFAULT_SEMANTIC_IDS_DIR / name / "semantic_ids.json"
        assert cfg.user_hash_mod == 1
        assert test_eval_batch_size == 64
        if layout == "prefix_target":
            assert cfg.train_target_policy == "last_n" and cfg.train_last_n == 128
            assert cfg.batch_size == 2048 and cfg.eval_batch_size == 1024
        else:
            assert cfg.batch_size == 512 and cfg.eval_batch_size == 128
            assert cfg.include_raw_in_batch is False
    # Default selection keeps the large-first order with VK-LSVD after Yambda.
    keys = [plan.group.key for plan in script.build_group_plans(script.DEFAULT_CONFIG_DIR, None)]
    assert keys == ["yambda", "vk_lsvd", "amazon_2023", "amazon"]


def test_gpt2_model_profiles_exist_for_vk_lsvd() -> None:
    model_config = common.load_yaml(run_gpt2_sft_all.DEFAULT_CONFIG_DIR / run_gpt2_sft_all.MODEL_CONFIG_NAME)
    for name in VK_DATASETS:
        profile = run_gpt2_sft_all.model_profile(model_config, name)
        assert profile["model"]["codebook_size"] == 256
        assert "lr" in profile["optimizer"]
        assert profile["training"]["val_sample_fraction"] == 0.1


def test_tokenizer_targets_include_the_vk_lsvd_family() -> None:
    targets = train_tokenizer_all.collect_targets(
        train_tokenizer_all.DEFAULT_CONFIG_DIR, train_tokenizer_all.DEFAULT_EMBEDDINGS_DIR, None
    )
    families = {target.name: target.family for target in targets}
    for name in VK_DATASETS:
        assert families[name] == "vk_lsvd"
    vk_only = train_tokenizer_all.collect_targets(
        train_tokenizer_all.DEFAULT_CONFIG_DIR, train_tokenizer_all.DEFAULT_EMBEDDINGS_DIR, VK_DATASETS[:1]
    )
    assert [target.name for target in vk_only] == VK_DATASETS[:1]
    assert vk_only[0].dataset_config is not None
    assert vk_only[0].dataset_config["embeddings_path"] == "data/raw/VK-LSVD/item_embeddings.npz"


# ---------------------------------------------------------------------------
# End-to-end CPU smoke run of every exp1 model on a synthetic subsample
# ---------------------------------------------------------------------------


def _run(module, argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", *argv])
    module.main()


def _metrics_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_exp1_model_trains_on_the_synthetic_vk_lsvd_subsample(tmp_path: Path, monkeypatch) -> None:
    config_dir = build_smoke_config_dir(tmp_path, n_users=200, n_items=400)
    semantic_ids_dir = tmp_path / "semantic_ids"
    report_path = tmp_path / "tokenizer_report.json"
    outputs = tmp_path / "outputs"
    common_args = ["--config-dir", str(config_dir), "--datasets", SMOKE_DATASET, "--device", "cpu"]

    # 1. Semantic IDs from the upstream content embeddings, catalog-restricted.
    _run(
        train_tokenizer_all,
        [*common_args, "--output-dir", str(semantic_ids_dir), "--report-path", str(report_path), "--epochs", "2"],
        monkeypatch,
    )
    report = json.loads(report_path.read_text())
    entry = report["datasets"][0]
    assert entry["family"] == "vk_lsvd"
    assert entry["num_items"] == entry["num_catalog_items"]
    assert entry["embedding_dim"] == 64
    semantic_ids = json.loads((semantic_ids_dir / SMOKE_DATASET / "semantic_ids.json").read_text())
    assert len(semantic_ids) == entry["num_items"]

    # 2. SASRec (dense-causal, full-catalog CE) through the shared sweep,
    # which enforces the exp1 three-seed protocol.
    _run(run_sasrec_all, [*common_args, "--max-steps", "2", "--seeds", "42", "43", "44"], monkeypatch)
    sasrec_rows = _metrics_rows(outputs / "artifacts" / f"sasrec_{SMOKE_DATASET}_metrics_summary.csv")
    assert [row["split"] for row in sasrec_rows] == ["validation", "test"] * 3
    for seed in (42, 43, 44):
        assert (outputs / "trainer_sasrec_vk_lsvd" / "checkpoints" / SMOKE_DATASET / f"seed_{seed}").is_dir()

    # 3. TIGER (prefix-target) and 4. GPT-2 (dense-causal over tokens).
    generative_args = [
        *common_args,
        "--semantic-ids-dir",
        str(semantic_ids_dir),
        "--seeds",
        "42",
        "--max-steps",
        "2",
        "--max-eval-batches",
        "1",
    ]
    _run(run_tiger_sft_all, generative_args, monkeypatch)
    tiger_rows = _metrics_rows(outputs / "artifacts" / f"tiger_sft_{SMOKE_DATASET}_metrics_summary_seed-42.csv")
    assert [row["split"] for row in tiger_rows] == ["validation", "test"]

    _run(run_gpt2_sft_all, generative_args, monkeypatch)
    gpt2_rows = _metrics_rows(outputs / "artifacts" / f"gpt2_sft_{SMOKE_DATASET}_metrics_summary_seed-42.csv")
    assert [row["split"] for row in gpt2_rows] == ["validation", "test"]

    for rows in (sasrec_rows, tiger_rows, gpt2_rows):
        for row in rows:
            assert row["dataset"] == SMOKE_DATASET
            for k in (5, 10, 20):
                for metric in ("HR", "NDCG", "Coverage", "ARP"):
                    assert row[f"{metric}@{k}"] != ""
