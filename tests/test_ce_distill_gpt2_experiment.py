"""GPT-2 CE uses its causal-window prefixes and isolated reward factors."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp7-ce-distillation/ce_distill_gpt2.py"
spec = importlib.util.spec_from_file_location("ce_distill_gpt2_experiment", SCRIPT)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


@pytest.fixture(scope="module")
def runtime():
    return ce.load_runtime()


def test_default_and_extension_grids_match_tiger_but_model_outputs_are_isolated():
    args = ce.parse_args([])
    assert len(ce.build_grid(args)) == 16
    assert args.datasets == ["beauty", "instruments"]
    assert args.powers == [.25, 1., 4., 16.] and args.rho_targets == [1., .1]
    assert args.seeds == [42] and args.num_samples == 10
    assert args.output_root == ROOT / "experiments/exp7-ce-distillation-gpt2"
    assert args.checkpoint_every_updates == 0
    exact = ce.parse_args(["--estimator", "exact"])
    assert exact.checkpoint_every_updates == 1
    ablations = [combo for combo in ce.COMBOS if combo != ce.DEFAULT_COMBO]
    smooth = ce.parse_args(["--combos", *ablations, "--sampling", "curriculum_smooth"])
    assert len(ce.build_grid(smooth)) == 96
    paths = {ce.run_dir_for(exact, row) for row in ce.build_grid(exact)}
    assert not paths & {ce.run_dir_for(smooth, row) for row in ce.build_grid(smooth)}
    tiger = ce.shared.parse_args([])
    assert not {ce.run_dir_for(args, row) for row in ce.build_grid(args)} & {
        ce.run_dir_for(tiger, row) for row in ce.build_grid(tiger)}


@pytest.mark.parametrize("flag", ["--list", "--count", "--aggregate-only"])
@pytest.mark.parametrize("sampling", ["reward", "curriculum", "curriculum_smooth", "curriculum_smooth_is"])
def test_metadata_cli_requires_no_ml_packages(flag, sampling, tmp_path):
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), flag, "--sampling", sampling,
                             "--output-root", str(tmp_path)], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    if flag == "--count":
        assert result.stdout.strip() == "16"
    elif flag == "--list":
        assert "total: 16 runs" in result.stdout
    else:
        assert json.loads(result.stdout)["completed"] == 0


@pytest.mark.parametrize("dataset", ["beauty", "instruments"])
def test_model_data_and_schedule_are_from_matched_gpt2(dataset, runtime):
    from flowrec.data.dense_prefix import dense_prefix_signature
    matched, base = runtime
    args = ce.parse_args([])
    b = ce.matched_args(args, dataset, 43, matched)
    config = ce.build_run_config(base, dataset, b)
    assert b.prefix_from_dense and b.cache_format == "factorized"
    assert (b.batch_size, b.grad_accum, b.val_batch_size, b.test_batch_size) == (32, 64, 64, 64)
    assert (b.eval_num_beams, b.test_ks) == (20, [5, 10, 20])
    assert b.train_last_n == b.train_random_n == 0
    assert config["seed"] == 43
    assert config["model"] == {"use_user_token": True, "user_ids_count": 1}
    assert config["pretrained_model"]["num_layers"] == 4
    assert config["pretrained_model"]["max_position_embeddings"] == 85
    assert config["dataset"]["train_example_layout"] == "prefix_target"
    assert config["dataset"]["train_all_prefixes"] is True
    assert dense_prefix_signature(config["dataset"]) == {
        "version": 1, "context": "dense_window", "window_len": 20,
        "window_stride": 20, "target_policy": "all", "last_n": None}
    assert config["dataset"]["positive_criterion"] == "verified_purchase == True"
    assert str(base.tuned_checkpoint_paths(dataset)["checkpoint"]).endswith(
        f"checkpoints/exp2/loo_split/initial/gpt2/{dataset}/best.pt")
    assert ce.component_cache_dir(base, dataset, "similarity", b) == (
        ROOT / "artifacts/exp7-ce-distillation-gpt2" / dataset / "reward_factors/sim_hier")
    sft = ce.make_sft(base).cfg
    assert sft.generation_num_beams == 5 and sft.generation_topk == (5,)
    assert not sft.compute_train_metrics and not sft.generation_exclude_seen_items


def test_smoke_cache_root_cannot_replace_production_factors(tmp_path, runtime):
    matched, base = runtime
    args = ce.parse_args(["--output-root", str(tmp_path / "runs"), "--cache-root", str(tmp_path / "production"),
                          "--batch-size", "2", "--grad-accum", "1", "--smoke"])
    b = ce.matched_args(args, "beauty", 42, matched)
    assert args.output_root == tmp_path / "runs_smoke"
    assert b.cache_root == args.output_root / "cache"
    assert ce.component_cache_dir(base, "beauty", "sasrec", b) == (
        args.output_root / "cache/beauty/reward_factors/sasrec")
    assert (b.batch_size, b.grad_accum, base._smoke_examples(b)) == (2, 1, 16)
    args.sampling = "curriculum_smooth"
    assert ce.curriculum_steps_for(args, 10000, b.batch_size) == 80


def test_custom_cache_root_keeps_gpt2_component_layout(tmp_path, runtime):
    matched, base = runtime
    args = ce.parse_args(["--cache-root", str(tmp_path)])
    b = ce.matched_args(args, "instruments", 42, matched)
    assert ce.component_cache_dir(base, "instruments", "interaction", b) == (
        tmp_path / "instruments/reward_factors/interaction")


@pytest.mark.parametrize("combo", ce.COMBOS)
def test_reward_ablations_keep_component_keys_and_specs_in_sync(combo, runtime):
    matched, base = runtime
    args = ce.parse_args([])
    full = ce.build_run_config(base, "beauty", ce.matched_args(args, "beauty", 42, matched))
    config = ce.reward_config(full, {"combo": combo})
    components = combo.split("_x_")
    assert config["method"]["component_keys"] == [ce.INTERNAL_COMPONENTS[name] for name in components]
    assert [spec["name"] for spec in config["method"]["reward"]["components"]] == components
    assert ("sasrec" in config["method"]) == ("sasrec" in components)
    # The unpruned configuration still supplies sasrec metadata to the shared
    # cache builder, even when building an interaction/similarity-only arm.
    assert full["method"]["component_keys"] == ["sasrec", "interaction", "sim_hier"]
    assert "sasrec" in base.cache_config(full, "sim_hier")["method"]


def test_long_user_contexts_targets_catalog_and_heldout_match_dense_windows(tmp_path, runtime):
    from dataclasses import replace
    import pandas as pd
    from flowrec.data.datamodule import AmazonReviewDataModule
    matched, base = runtime
    args = ce.parse_args(["--num-workers", "0"])
    config = ce.build_run_config(base, "beauty", ce.matched_args(args, "beauty", 42, matched))
    reviews = tmp_path / "reviews.parquet"
    pd.DataFrame([dict(user_id="long", asin=str(i), timestamp=1000 + i,
                       rating=5., verified_purchase=True) for i in range(66)]).to_parquet(reviews)
    sids = tmp_path / "sids.json"
    sids.write_text(json.dumps({str(i): [i // 16, i % 16] for i in range(66)}))
    config["data"] = {"source": "amazon", "reviews_path": str(reviews), "semantic_id_path": str(sids)}
    config["dataset"].update(min_user_interactions=3, min_item_interactions=0,
                              processed_cache_dir=str(tmp_path / "processed"))
    cfg = base.tiger_runner.build_data_config(config, config_path=tmp_path / "config.json")
    prefix = AmazonReviewDataModule(cfg)
    prefix.setup()
    dense = AmazonReviewDataModule(replace(cfg, train_example_layout="dense_causal", train_prefix_from_dense=False))
    dense.setup()
    oracle = []
    for row in dense.train_dataset.examples.itertuples():
        for offset, active in enumerate(row.label_mask):
            if active:
                oracle.append((row.user_id, row.target_position - len(row.label_mask) + offset + 1,
                               row.label_item_ids[offset], tuple(row.history_item_ids[:offset + 1])))
    actual = [(row.user_id, row.target_position, row.target_item_id, tuple(row.history_item_ids))
              for row in prefix.train_dataset.examples.itertuples()]
    assert sorted(actual) == sorted(oracle) and len(actual) == 63
    for target, history in [("4", ["3"]), ("24", ["23"])]:
        row = prefix.train_dataset.examples.loc[lambda df: df.target_raw_item == target].iloc[0]
        assert row.history_raw_items == history
    assert prefix.item2id == dense.item2id and prefix.semantic_ids == dense.semantic_ids
    for split in ("val", "test"):
        pd.testing.assert_frame_equal(getattr(prefix, split + "_dataset").examples,
                                      getattr(dense, split + "_dataset").examples)
    assert prefix.dense_prefix_audit["validation"]["equal"]
    assert prefix.dense_prefix_audit["test"]["equal"]


def test_cache_checks_reject_other_context_or_subset_before_reading_factors(tmp_path):
    checked = []
    base = SimpleNamespace(finetune_subset_spec=lambda b: {"per_user_last_n": None},
                           factor_runner=SimpleNamespace(check_component_manifest=lambda *a, **k: checked.append(a)))
    config = {"dataset": {"train_prefix_from_dense": True, "max_history_len": 20, "train_target_policy": "all"}}
    path = tmp_path / "manifest.json"
    ce.write_json(path, {"training_context": None, "per_user_last_n": None})
    with pytest.raises(RuntimeError, match="different training_context"):
        ce.check_component_manifest(base, config, path, "interaction", 8, SimpleNamespace())
    from flowrec.data.dense_prefix import dense_prefix_signature
    context = dense_prefix_signature(config["dataset"])
    ce.write_json(path, {"training_context": context, "per_user_last_n": 10})
    with pytest.raises(RuntimeError, match="cache subset"):
        ce.check_component_manifest(base, config, path, "interaction", 8, SimpleNamespace())
    assert not checked
    ce.write_json(path, {"training_context": context, "per_user_last_n": None})
    ce.check_component_manifest(base, config, path, "interaction", 8, SimpleNamespace())
    assert len(checked) == 1


@pytest.mark.parametrize("drift", ["catalog", "target_tokens", "timestamps", "row_count"])
def test_factors_must_match_gpt2_catalog_and_each_other_beyond_spotcheck(drift):
    import torch
    catalog = torch.tensor([[1, 3], [2, 4]])
    factors = {"item_tokens": catalog.clone(), "target_semantic_row_ids": torch.tensor([0, 1]),
               "target_token_ids": catalog.clone(), "target_timestamps": torch.tensor([10, 20])}
    reference = SimpleNamespace(num_examples=2, num_items=2, factors=factors)
    other = SimpleNamespace(num_examples=2, num_items=2, factors={key: value.clone() for key, value in factors.items()})
    if drift == "catalog":
        other.factors["item_tokens"] = catalog.flip(0)
    elif drift == "target_tokens":
        other.factors["target_token_ids"][1, 0] += 1
    elif drift == "timestamps":
        other.factors["target_timestamps"][1] += 1
    else:
        other.num_examples = 3
    base = SimpleNamespace(factor_runner=SimpleNamespace(verify_cache_alignment=lambda *args: None))
    with pytest.raises(RuntimeError):
        ce.verify_caches(base, None, [0, 1], {"similarity": reference, "interaction": other},
                         SimpleNamespace(item_semantic_ids=catalog))


def test_asset_fingerprints_include_tuned_model_dropout_and_only_used_teachers(tmp_path, runtime):
    matched, base = runtime
    args = ce.parse_args([])
    config = ce.build_run_config(base, "beauty", ce.matched_args(args, "beauty", 42, matched))
    manifests = [tmp_path / "interaction/manifest.json"]
    paths = ce.asset_paths_for(base, "beauty", config, ("interaction",), manifests)
    assert base.tuned_checkpoint_paths("beauty")["best_trial"] in paths
    assert base.tuned_checkpoint_paths("beauty")["checkpoint"] in paths
    assert base.EXP1_GPT2_MODEL_YAML in paths
    assert base.dataset_yaml_path("beauty") in paths
    sasrec = base.abspath(base.DATASETS["beauty"]["sasrec_checkpoint"])
    assert sasrec not in paths and base.rqvae_model_path("beauty") not in paths
    all_paths = ce.asset_paths_for(base, "beauty", config, ce.COMPONENTS, manifests)
    assert sasrec in all_paths and base.rqvae_model_path("beauty") in all_paths
    assert base.abspath(base.DATASETS["beauty"]["sasrec_trial"]) in all_paths


def test_identity_separates_gpt2_reward_hard_smooth_and_exact(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path)])
    entry = ce.build_grid(args)[0]
    identities = []
    paths = []
    for sampling, estimator in [("reward", "sampled"), ("curriculum", "sampled"),
                                 ("curriculum_smooth", "sampled"), ("reward", "exact")]:
        args.sampling, args.estimator = sampling, estimator
        identity = ce.build_identity(args, entry, {}, {}, {}, 100)
        assert identity["model"] == "gpt2"
        assert "dense-causal" in identity["data_protocol"]
        identities.append(json.dumps(identity, sort_keys=True))
        paths.append(ce.run_dir_for(args, entry))
        if sampling != "reward":
            assert identity["curriculum"]["algorithm"] == ce.SAMPLING_ALGORITHMS[sampling]
        if estimator == "exact":
            assert identity["exact_algorithm"] == "branching_prefix"
    assert len(set(identities)) == len(set(paths)) == 4


def test_aggregation_selects_validation_not_test(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path), "--datasets", "beauty", "--powers", "1", "4",
                          "--rho-targets", "1", "--sampling", "curriculum_smooth", "--combos", "similarity"])
    for entry in ce.build_grid(args):
        run = ce.run_dir_for(args, entry)
        ce.write_json(run / "result.json", {**entry, "status": "completed", "sampling": args.sampling,
            "estimator": "sampled", "num_samples": 10, "lambda_distill": 1., "best_epoch": 1,
            "best_val_ndcg5": .1, "optimizer_steps": 2, "fit_sec_this_invocation": 1., "run_dir": str(run),
            "splits": {"validation": {"NDCG@5": .2 if entry["power"] == 1 else .1},
                       "test": {"NDCG@5": .01 if entry["power"] == 1 else 1.}}})
    ce.aggregate(args)
    summary = ce.read_json(ce.summary_dir_for(args) / "summary.json")
    assert summary["selected"]["beauty"]["similarity"]["power"] == 1
    assert summary["completed"] == summary["expected"] == 2
