"""Beauty 2023 CE keeps the matched GPT-2 data and corrected objective."""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp6-ce-distillation/ce_distill_gpt2_beauty2023.py"
spec = importlib.util.spec_from_file_location("ce_beauty2023_profile_tests", SCRIPT)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


@pytest.fixture(scope="module")
def runtime():
    return ce.load_runtime()


def test_grid_is_24_requested_corrected_ce_runs_and_separate_from_old_amazon():
    args = ce.parse_args([])
    combos = ["sasrec", "interaction", "similarity", "sasrec_x_interaction_x_similarity"]
    assert args.datasets == ["beauty_2023"] and args.combos == combos
    assert args.powers == [.25, 1., 16.] and args.rho_targets == [1., .1]
    assert args.seeds == [42] and args.num_samples == 3
    assert args.estimator == "sampled" and args.sampling == "curriculum_smooth_is"
    assert args.curriculum_epochs == 5. and args.proposal_q_mix == .1
    assert args.resume
    assert args.output_root == ROOT / "experiments/exp7-ce-distillation-gpt2-beauty2023"
    grid = ce.build_grid(args)
    assert [(row["dataset"], ce.combo_key(row), row["power"], row["rho"], row["seed"])
            for row in grid] == list(itertools.product(["beauty_2023"], combos, [.25, 1., 16.], [1., .1], [42]))
    paths = [ce.run_dir_for(args, row) for row in grid]
    assert len(set(paths)) == 24
    assert all("curriculum_smooth_is_n3_epochs5_qmix0p1" in str(path) for path in paths)
    legacy = ce.gpt2.parse_args([])
    assert legacy.datasets == ["beauty", "instruments"] and legacy.num_samples == 10
    assert legacy.sampling == "reward" and len(ce.gpt2.build_grid(legacy)) == 16


@pytest.mark.parametrize("flag", ["--count", "--list", "--aggregate-only"])
def test_metadata_actions_require_only_stdlib(flag, tmp_path):
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), flag,
                             "--output-root", str(tmp_path)], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    if flag == "--count":
        assert result.stdout.strip() == "24"
    elif flag == "--list":
        assert "total: 24 runs" in result.stdout and "beauty_2023" in result.stdout
        assert "curriculum_smooth_is" in result.stdout
    else:
        assert json.loads(result.stdout)["completed"] == 0


@pytest.mark.parametrize("flags", [
    ["--datasets", "beauty"], ["--datasets", "yambda_50m_listens"],
    ["--combos", "sasrec_x_interaction"], ["--estimator", "exact"],
    ["--sampling", "reward"], ["--sampling", "curriculum_smooth"],
    ["--sampling", "curriculum"], ["--proposal-q-mix", "0"],
])
def test_wrong_dataset_grid_or_uncorrected_objective_is_rejected(flags):
    with pytest.raises(SystemExit):
        ce.parse_args(flags)


def test_matching_gflow_candidates_batches_data_checkpoint_and_teacher(runtime, monkeypatch):
    from flowrec.data.dense_prefix import dense_prefix_signature
    matched, base = runtime
    paths = base.tuned_checkpoint_paths("beauty_2023")
    if not paths["checkpoint"].is_file() and not paths["sidecar"].is_file():
        # This comparison needs user-token settings, not weights. The recorded
        # tuner configuration supplies the architecture in a clean checkout.
        profile = {"model": json.loads(paths["best_trial"].read_text())["model"]}
        monkeypatch.setattr(base, "tuned_profile", lambda dataset, args=None: (profile, profile["model"]))
    args = ce.parse_args([])
    b = ce.matched_args(args, "beauty_2023", 43, matched)
    gflow = base.dataset_args(matched.parse_args([
        "--datasets", "beauty_2023", "--num-candidates", "3", "--prefix-from-dense",
        "--seed", "43", "--num-workers", str(args.num_workers),
    ]), "beauty_2023")
    assert list(b.num_candidates) == list(gflow.num_candidates) == [3]
    assert (b.batch_size, b.grad_accum, b.val_batch_size, b.test_batch_size) == (128, 16, 128, 128)
    assert (b.batch_size, b.grad_accum) == (gflow.batch_size, gflow.grad_accum)
    assert b.train_last_n == b.train_random_n == 0
    assert b.val_sample_fraction is None and not b.val_sample_resample
    assert b.prefix_from_dense and b.cache_format == "factorized"
    assert b.eval_num_beams == 20 and b.test_ks == [5, 10, 20]
    config = ce.build_run_config(base, "beauty_2023", b)
    profile, _ = base.tuned_profile("beauty_2023")
    model_yaml = base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML)
    gflow_config = base.build_run_config(
        "beauty_2023", gflow, base.gpt2_runner.user_ids_count(model_yaml, profile))
    # Compare the complete input configuration, including filters and token
    # settings, rather than only the expected layout/counts.
    for key in ("seed", "data", "dataset", "dataloader", "model"):
        assert config[key] == gflow_config[key], key
    assert base.finetune_subset_spec(b) == base.finetune_subset_spec(gflow)
    assert config["seed"] == 43
    assert config["dataset"]["train_example_layout"] == "prefix_target"
    assert config["dataset"]["train_all_prefixes"]
    assert config["dataset"]["metadata_ids_only"]
    assert config["dataset"]["positive_criterion"] == "verified_purchase == True"
    assert dense_prefix_signature(config["dataset"]) == {
        "version": 1, "context": "dense_window", "window_len": 20,
        "window_stride": 20, "target_policy": "all", "last_n": None}
    assert base.tuned_checkpoint_paths("beauty_2023")["checkpoint"] == (
        ROOT / "checkpoints/exp2/loo_split/initial/gpt2/beauty_2023/best.pt")
    assert config["method"]["sasrec"] == base.sasrec_method_cfg("beauty_2023")
    assert str(base.rqvae_model_path("beauty_2023")).endswith("beauty_2023/model.pt")
    assert ce.component_cache_dir(base, "beauty_2023", "similarity", b) == (
        ROOT / "artifacts/exp7-ce-distillation-gpt2-beauty2023/beauty_2023/reward_factors/sim_hier")


def test_smoke_uses_separate_cache_and_explicit_batch_overrides(runtime, tmp_path):
    matched, base = runtime
    args = ce.parse_args(["--smoke", "--output-root", str(tmp_path / "runs"),
                          "--cache-root", str(tmp_path / "prod"),
                          "--batch-size", "2", "--grad-accum", "1"])
    b = ce.matched_args(args, "beauty_2023", 42, matched)
    assert args.output_root == tmp_path / "runs_smoke"
    assert b.cache_root == args.output_root / "cache"
    assert (b.batch_size, b.grad_accum, base._smoke_examples(b)) == (2, 1, 16)
    assert ce.component_cache_dir(base, "beauty_2023", "interaction", b).is_relative_to(b.cache_root)


def test_identity_declares_q_target_and_importance_correction():
    args = ce.parse_args([])
    entry = ce.build_grid(args)[0]
    identity = ce.build_identity(args, entry, {}, {}, {}, 100)
    assert identity["model"] == "gpt2" and identity["ce_target"] == "q"
    assert identity["curriculum"]["importance_correction"]
    assert identity["curriculum"]["proposal_q_mix"] == .1


def test_cache_rejects_different_prefix_context_or_subset(tmp_path):
    from flowrec.data.dense_prefix import dense_prefix_signature
    config = {"dataset": {"train_prefix_from_dense": True, "max_history_len": 20,
                           "train_target_policy": "all"}}
    context = dense_prefix_signature(config["dataset"])
    path = tmp_path / "manifest.json"
    base = SimpleNamespace(finetune_subset_spec=lambda b: {"per_user_last_n": None},
        factor_runner=SimpleNamespace(check_component_manifest=lambda *args, **kwargs: args[0]))
    for manifest, message in [
        ({"training_context": None}, "training_context"),
        ({"training_context": context, "per_user_last_n": 10}, "cache subset"),
    ]:
        ce.write_json(path, manifest)
        with pytest.raises(RuntimeError, match=message):
            ce.check_component_manifest(base, config, path, "interaction", 8, SimpleNamespace())
