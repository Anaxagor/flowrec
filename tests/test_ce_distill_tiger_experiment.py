"""Protocol reuse, run isolation, and held-out selection for the CE arm."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp7-ce-distillation/ce_distill_tiger.py"
spec = importlib.util.spec_from_file_location("ce_distill_tiger_experiment", SCRIPT)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


@pytest.fixture
def runtime(monkeypatch):
    matched, base = ce.load_runtime()
    # matched.parse_args activates dataset-specific module globals. Restore
    # them so unrelated experiment tests retain their original assets.
    for key in {"DATASET", *(k for block in base.DATASETS.values() for k in block)}:
        monkeypatch.setattr(base, key, getattr(base, key))
    return matched, base


def test_default_grid_is_the_exp4_composite_power_and_strength_sweep():
    args = ce.parse_args([])
    grid = ce.build_grid(args)
    protocol = ce.read_json(ce.PROTOCOL)
    assert len(grid) == 16
    assert args.datasets == ["beauty", "instruments"]
    assert args.powers == protocol["gflow_grid"]["powers"] == [.25, 1., 4., 16.]
    assert args.rho_targets == protocol["gflow_grid"]["rho_targets"] == [1., .1]
    assert args.seeds == [protocol["shared_schedule"]["seed_gflow"]] == [42]
    assert args.num_samples == protocol["gflow_grid"]["num_negatives"] == 10
    assert args.combos == [ce.DEFAULT_COMBO] and args.sampling == "reward"
    assert ce.COMPONENTS == ("sasrec", "interaction", "similarity")
    assert grid[0] == dict(dataset="beauty", power=.25, rho=1., seed=42)
    assert grid[8] == dict(dataset="instruments", power=.25, rho=1., seed=42)
    assert len({ce.run_dir_for(args, entry) for entry in grid}) == len(grid)


@pytest.mark.parametrize("flag", ["--list", "--count", "--aggregate-only"])
@pytest.mark.parametrize("sampling", ["reward", "curriculum", "curriculum_smooth", "curriculum_smooth_is"])
def test_metadata_commands_work_without_installed_ml_packages(flag, sampling, tmp_path):
    # -S removes site-packages: even importing torch or pandas would fail.
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), flag, "--sampling", sampling, "--output-root", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    if flag == "--count":
        assert result.stdout.strip() == "16"
    elif flag == "--list":
        assert "total: 16 runs" in result.stdout
    else:
        summary = json.loads(result.stdout)
        assert summary["completed"] == 0 and summary["expected"] == 16
        assert summary["selected"] == {}


@pytest.mark.parametrize("dataset", ["beauty", "instruments"])
def test_training_data_assets_and_batches_match_exp4(dataset, runtime):
    matched, base = runtime
    args = ce.parse_args([])
    inherited = ce.matched_args(args, dataset, 43, matched)
    config = base.build_run_config(inherited)
    source = ce.read_json(base.abspath(base.SFT_CONFIG))
    protocol = ce.read_json(ce.PROTOCOL)
    assert base.TIGER_CHECKPOINT == f"checkpoints/exp2/loo_split/initial/tiger/{dataset}/best.pt"
    assert base.SASREC_CHECKPOINT == f"checkpoints/exp2/loo_split/initial/sasrec/{dataset}/best.pt"
    assert base.RQVAE_MODEL == f"datasets/semantic_ids/exp2-tuned-params/{dataset}/rqvae/model.pt"
    assert config["model"] == source["model"]
    assert config["seed"] == 43
    assert config["dataset"] == {
        **source["dataset"], "use_verified_only": False,
        "positive_criterion": "verified_purchase == True",
    }
    assert config["dataset"]["train_target_policy"] == "all"
    assert config["dataset"]["train_all_prefixes"] is True
    assert inherited.train_last_n == inherited.train_random_n == 0
    assert (inherited.batch_size, inherited.grad_accum, inherited.val_batch_size) == (32, 64, 64)
    assert config["dataloader"]["train_batch_size"] == 32
    assert inherited.eval_num_beams == 20 and inherited.test_ks == [5, 10, 20]
    assert inherited.val_sample_fraction is None
    assert str(inherited.cache_root) == protocol["arms"]["tiger_gflow"]["cache_roots"][dataset]
    assert inherited.output_root != Path(protocol["arms"]["tiger_gflow"]["output_roots"][dataset])
    rewards = config["method"]["reward"]["components"]
    assert [r["name"] for r in rewards] == list(ce.COMPONENTS)
    assert all(r["weight"] == 1. for r in rewards)
    assert rewards[0]["reward"] == {"transform": "sigmoid", "temperature": 1.}
    assert rewards[2]["reward"]["mode"] == "hierarchical"


def test_smoke_caches_never_replace_production_or_custom_caches(tmp_path, runtime):
    matched, base = runtime
    args = ce.parse_args(["--output-root", str(tmp_path / "runs"),
                          "--cache-root", str(tmp_path / "production"),
                          "--batch-size", "2", "--grad-accum", "1", "--smoke"])
    inherited = ce.matched_args(args, "beauty", 42, matched)
    assert args.output_root == tmp_path / "runs_smoke"
    assert inherited.cache_root == args.output_root / "reward_factors/beauty"
    assert base._smoke_examples(inherited) == 16
    assert inherited.output_root == args.output_root / "beauty"


def test_estimator_and_nearby_numeric_grid_values_have_separate_runs(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path), "--datasets", "beauty",
                          "--powers", "1", "1.0000001", "--rho-targets", "1", "1.0000001"])
    sampled_paths = {ce.run_dir_for(args, entry) for entry in ce.build_grid(args)}
    assert len(sampled_paths) == 4
    args.estimator = "exact"
    exact_paths = {ce.run_dir_for(args, entry) for entry in ce.build_grid(args)}
    assert len(exact_paths) == 4 and not (exact_paths & sampled_paths)


def test_all_reward_ablations_and_sampling_regimes_have_distinct_paths(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path), "--combos", *ce.COMBOS])
    grid = ce.build_grid(args)
    assert len(grid) == 112
    assert {ce.components_for(entry) for entry in grid} == {
        ("sasrec",), ("interaction",), ("similarity",),
        ("sasrec", "interaction"), ("sasrec", "similarity"),
        ("interaction", "similarity"), ce.COMPONENTS,
    }
    reward_paths = {ce.run_dir_for(args, entry) for entry in grid}
    assert len(reward_paths) == 112
    args.sampling = "curriculum"
    curriculum_paths = {ce.run_dir_for(args, entry) for entry in grid}
    assert len(curriculum_paths) == 112 and not reward_paths & curriculum_paths
    args.curriculum_epochs = 20
    assert not curriculum_paths & {ce.run_dir_for(args, entry) for entry in grid}
    args.sampling, args.curriculum_epochs = "curriculum_smooth", 10
    smooth_paths = {ce.run_dir_for(args, entry) for entry in grid}
    assert len(smooth_paths) == 112
    assert not smooth_paths & (reward_paths | curriculum_paths)
    assert ce.estimator_tag(args) == "curriculum_smooth_n10_epochs10"
    args.curriculum_epochs = 5
    assert not smooth_paths & {ce.run_dir_for(args, entry) for entry in grid}
    args.combos = list(ce.COMBOS[:-1])
    assert len(ce.build_grid(args)) == 96
    triple = ce.parse_args(["--estimator", "exact"])
    assert len(ce.build_grid(triple)) == 16


def test_original_paths_and_identity_are_unchanged_by_new_operational_flags(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path)])
    entry = ce.build_grid(args)[0]
    assert ce.run_dir_for(args, entry) == tmp_path / "beauty/sampled_n10/pow_0p25/rho_1/seed_42"
    config, trainer, assets = {"seed": 42}, {"max_epochs": 500}, {"checkpoint.pt": {"bytes": 17}}
    expected = {
        "version": 1, "entry": entry, "estimator": "sampled", "num_samples": 10,
        "rng_policy": "seed + epoch for model and training loader",
        "catalog_chunk_size": 16, "normalization": "valid_actions",
        "reward": "exp4 floored product sasrec * interaction * hierarchical similarity, then total power",
        "config": config, "trainer": trainer, "lr": 3e-5, "weight_decay": .01,
        "calibration_batches": 5, "calibration_seed": 1234, "assets": assets,
        "protocol_sha256": hashlib.sha256(ce.PROTOCOL.read_bytes()).hexdigest(),
    }
    assert ce.build_identity(args, entry, config, trainer, assets, 0) == expected

    args.checkpoint_every_updates = 2
    args.curriculum_epochs = 50  # unused under direct reward sampling
    assert ce.build_identity(args, entry, config, trainer, assets, 0) == expected
    assert ce.reward_config(config, entry) is config
    args.sampling = "curriculum"
    changed = ce.build_identity(args, entry, config, trainer, assets, 500)
    assert changed["version"] == 2 and changed["sampling"] == "curriculum"
    assert changed == {
        **expected, "version": 2, "combo": ce.DEFAULT_COMBO, "sampling": "curriculum",
        "reward": "exp4 floored product sasrec * interaction * similarity, then total power",
        "curriculum": {
            "epochs": 50, "steps": 500, "algorithm": "reward_halves",
            "phases": ["low reward half", "high reward half"],
            "probabilities": "q normalized within the active half", "replacement": True,
            "importance_correction": False,
        },
    }
    args.sampling = "curriculum_smooth"
    smooth = ce.build_identity(args, entry, config, trainer, assets, 500)
    assert smooth["sampling"] == "curriculum_smooth"
    assert smooth["curriculum"]["algorithm"] == "reward_halves_linear_mixture"
    assert smooth["curriculum"]["importance_correction"] is False
    assert smooth["curriculum"]["epochs"] == 50 and smooth["curriculum"]["steps"] == 500
    assert "global_microstep" in smooth["curriculum"]["high_half_probability"]


def test_importance_run_identity_paths_and_summary_separate_epsilon_and_old_modes(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path), "--sampling", "curriculum_smooth_is",
                          "--curriculum-epochs", "5"])
    entry = ce.build_grid(args)[0]
    corrected_path = ce.run_dir_for(args, entry)
    corrected_summary = ce.summary_dir_for(args)
    assert "curriculum_smooth_is_n10_epochs5_qmix0p1" in str(corrected_path)
    original = ce.build_identity(args, entry, {}, {}, {}, 100)
    assert original["version"] == 3
    assert original["ce_target"] == "q" and original["calibration_sampling"] == "reward"
    assert original["curriculum"]["importance_correction"] is True
    assert original["curriculum"]["full_q_support"] is True
    assert original["curriculum"]["proposal_q_mix"] == .1
    args.proposal_q_mix = .2
    assert ce.run_dir_for(args, entry) != corrected_path
    assert ce.summary_dir_for(args) != corrected_summary
    assert ce.build_identity(args, entry, {}, {}, {}, 100) != original
    for mode in ("reward", "curriculum", "curriculum_smooth"):
        args.sampling = mode
        assert ce.run_dir_for(args, entry) != corrected_path
        legacy = ce.build_identity(args, entry, {}, {}, {}, 100)
        args.proposal_q_mix = .9
        assert ce.build_identity(args, entry, {}, {}, {}, 100) == legacy


@pytest.mark.parametrize("fails", [False, True])
def test_corrected_calibration_uses_q_and_always_restores_training_config(fails):
    from types import SimpleNamespace
    from flowrec.ft_methods.reward_distillation import RewardDistillationConfig
    config = RewardDistillationConfig(sampling="curriculum_smooth_is", curriculum_steps=50, proposal_q_mix=.2)
    method = SimpleNamespace(cfg=config)
    model, loader = object(), object()

    def calibrator(actual_model, actual_method, actual_loader, **kwargs):
        assert (actual_model, actual_loader) == (model, loader)
        assert actual_method.cfg.sampling == "reward"
        assert actual_method.cfg.num_samples == config.num_samples
        assert kwargs == {"target_ratio": .1, "seed": 1234}
        if fails:
            raise RuntimeError("calibration failed")
        return {"weight": .7}

    if fails:
        with pytest.raises(RuntimeError, match="calibration failed"):
            ce.calibrate_distillation_weight(calibrator, model, method, loader, target_ratio=.1, seed=1234)
    else:
        result = ce.calibrate_distillation_weight(calibrator, model, method, loader, target_ratio=.1, seed=1234)
        assert result == {"weight": .7, "sampling": "reward", "ce_target": "q"}
    assert method.cfg is config


@pytest.mark.parametrize("options,expected", [
    ([], 41070), (["--max-train-batches", "3"], 30),
    (["--smoke"], 80), (["--curriculum-epochs", "0"], 0),
    (["--curriculum-epochs", ".5", "--max-train-batches", "3"], 2),
])
@pytest.mark.parametrize("sampling", ["curriculum", "curriculum_smooth", "curriculum_smooth_is"])
def test_curriculum_duration_counts_actual_training_microbatches(options, expected, sampling):
    args = ce.parse_args(["--sampling", sampling, *options])
    assert ce.curriculum_steps_for(args, 131413, 32) == expected
    # A short smoke cache can contain fewer examples than the eight-batch cap.
    assert ce.curriculum_steps_for(ce.parse_args(["--sampling", sampling, "--smoke"]), 35, 32) == 20
    assert ce.curriculum_steps_for(ce.parse_args([]), 131413, 32) == 0


@pytest.mark.parametrize("combo,required_assets", [
    ("interaction", {"TIGER"}), ("sasrec", {"TIGER", "SASREC"}),
    ("similarity", {"TIGER", "RQVAE"}),
    ("interaction_x_similarity", {"TIGER", "RQVAE"}),
])
def test_singletons_and_pairs_only_require_their_own_reward_assets(combo, required_assets, runtime):
    matched, base = runtime
    args = ce.parse_args(["--combos", combo, "--sampling", "curriculum"])
    entry = ce.build_grid(args)[0]
    inherited = ce.matched_args(args, "beauty", 42, matched)
    original = base.build_run_config(inherited)
    config = ce.reward_config(original, entry)
    components = ce.components_for(entry)
    assert [spec["name"] for spec in config["method"]["reward"]["components"]] == list(components)
    assert ("sasrec" in config["method"]) == ("sasrec" in components)
    assert len(original["method"]["reward"]["components"]) == 3
    dummy = SimpleNamespace(TIGER_CHECKPOINT="TIGER", SASREC_CHECKPOINT="SASREC", RQVAE_MODEL="RQVAE", abspath=Path)
    paths = ce.asset_paths_for(dummy, {"data": {}}, components, [])
    assert {str(path) for path in paths} == required_assets


@pytest.mark.parametrize("argv", [
    ["--powers", "0"], ["--rho-targets", "nan"],
    ["--seeds", "42", "42"], ["--datasets", "beauty", "beauty"],
    ["--cache-component", "sasrec"], ["--cache-only", "--index", "0"],
    ["--combos", "sasrec", "sasrec"], ["--combos", "interaction_x_sasrec"],
    ["--estimator", "exact", "--sampling", "curriculum"],
    ["--estimator", "exact", "--sampling", "curriculum_smooth"],
    ["--estimator", "exact", "--sampling", "curriculum_smooth_is"],
    ["--proposal-q-mix", "0"], ["--proposal-q-mix", "-0.1"],
    ["--proposal-q-mix", "1.01"], ["--proposal-q-mix", "nan"], ["--proposal-q-mix", "inf"],
    ["--sampling", "curriculum_smooth_is", "--proposal-q-mix", "1e-50"],
    ["--sampling", "curriculum_smooth_is", "--proposal-q-mix", "1e-40"],
    ["--curriculum-epochs", "nan"], ["--curriculum-epochs", "-1"],
    ["--checkpoint-every-updates", "-1"],
])
def test_invalid_grid_or_cache_selection_fails(argv):
    with pytest.raises(SystemExit):
        ce.parse_args(argv)


def test_asset_change_rejects_resume_and_preserves_identity(tmp_path):
    asset = tmp_path / "checkpoint.pt"
    asset.write_bytes(b"original checkpoint")
    identity = {"entry": {"dataset": "beauty", "seed": 42}, "assets": ce.fingerprint([asset])}
    destination = tmp_path / "identity.json"
    ce.write_json(destination, identity)
    ce.check_identity(destination, identity)
    asset.write_bytes(b"different checkpoint with a different size")
    changed = {**identity, "assets": ce.fingerprint([asset])}
    with pytest.raises(RuntimeError, match="choose a new --output-root"):
        ce.check_identity(destination, changed)
    assert ce.read_json(destination) == identity


@pytest.mark.parametrize("change", ["checkpoint_path", "protocol", "learning_rate", "subset"])
def test_resume_requires_exact_current_run_identity(tmp_path, change):
    checkpoint = str(tmp_path / "checkpoints/exp2/loo_split/initial/tiger/beauty/best.pt")
    identity = {
        "checkpoint": checkpoint,
        "protocol_sha256": "a" * 64,
        "config": {"learning_rate": 3e-5, "train_last_n": 128},
    }
    destination = tmp_path / "identity.json"
    ce.check_identity(destination, identity)
    ce.write_json(destination, identity)
    ce.check_identity(destination, identity)

    changed = {**identity, "config": dict(identity["config"])}
    if change == "checkpoint_path":
        changed["checkpoint"] = str(tmp_path / "checkpoints/exp2/loo_split/initial/tiger/instruments/best.pt")
    elif change == "protocol":
        changed["protocol_sha256"] = "b" * 64
    elif change == "learning_rate":
        changed["config"]["learning_rate"] = 1e-3
    else:
        changed["config"]["train_last_n"] = 15
    with pytest.raises(RuntimeError, match="choose a new --output-root"):
        ce.check_identity(destination, changed)
    assert ce.read_json(destination) == identity


def test_full_cache_metadata_and_catalog_alignment_are_checked(runtime):
    import torch

    _, base = runtime
    catalog = torch.tensor([[1, 3], [2, 4]])
    target_rows = torch.arange(300) % 2
    caches = {
        name: SimpleNamespace(num_examples=300, num_items=len(catalog), factors={
            "item_tokens": catalog.clone(), "target_semantic_row_ids": target_rows.clone(),
        }) for name in ce.COMPONENTS
    }
    checked = []
    stub_base = SimpleNamespace(verify_cache_alignment=lambda *args: checked.append(True))
    model = SimpleNamespace(item_semantic_ids=catalog)
    ce.verify_caches(stub_base, object(), list(range(300)), caches, model)
    assert checked
    # Corruption beyond the inherited first-256-target spot check is caught
    # by the full cross-component check, before q can supervise wrong rows.
    caches["sasrec"].factors["target_semantic_row_ids"][299] = 9
    with pytest.raises(RuntimeError, match="target_semantic_row_ids"):
        ce.verify_caches(stub_base, object(), list(range(300)), caches, model)
    caches["sasrec"].factors["target_semantic_row_ids"] = target_rows.clone()
    model.item_semantic_ids = catalog.flip(0)
    with pytest.raises(RuntimeError, match="catalog order"):
        ce.verify_caches(stub_base, object(), list(range(300)), caches, model)


@pytest.mark.parametrize("name", ["sasrec", "interaction"])
def test_cache_alignment_without_similarity_still_checks_catalog_and_target_rows(name):
    import torch

    catalog = torch.tensor([[1, 3], [2, 4]])
    cache = SimpleNamespace(num_examples=3, num_items=2, factors={
        "target_semantic_row_ids": torch.tensor([0, 1, 0]),
        "target_token_ids": catalog[[0, 1, 0]],
    })
    base = SimpleNamespace(verify_cache_alignment=lambda *args: None)
    model = SimpleNamespace(item_semantic_ids=catalog)
    ce.verify_caches(base, object(), [0, 1, 2], {name: cache}, model)
    cache.factors["target_token_ids"][2] = catalog[1]
    with pytest.raises(RuntimeError, match="catalog order"):
        ce.verify_caches(base, object(), [0, 1, 2], {name: cache}, model)


def test_aggregation_selects_full_validation_mean_and_excludes_incomplete_seed_groups(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path), "--datasets", "beauty",
                          "--powers", "1", "4", "16", "--rho-targets", "1",
                          "--seeds", "42", "43"])
    scores = {(1., 42): (.3, .01), (1., 43): (.5, .01),
              (4., 42): (.2, .99), (4., 43): (.3, .99), (16., 42): (.9, .99)}
    for entry in ce.build_grid(args):
        if (entry["power"], entry["seed"]) not in scores:
            continue
        val, test = scores[entry["power"], entry["seed"]]
        directory = ce.run_dir_for(args, entry)
        ce.write_json(directory / "result.json", {
            **entry, "status": "completed", "estimator": args.estimator,
            "num_samples": args.num_samples, "lambda_distill": .5,
            "best_epoch": 2, "best_val_ndcg5": 1. - val,
            "optimizer_steps": 100, "fit_sec_this_invocation": 20., "run_dir": str(directory),
            "splits": {"validation": {"NDCG@5": val}, "test": {"NDCG@5": test}},
        })
    ce.aggregate(args)
    output = tmp_path / "sampled_n10"
    summary = ce.read_json(output / "summary.json")
    assert summary["completed"] == 5 and summary["expected"] == 6
    assert summary["missing"] == [dict(dataset="beauty", power=16., rho=1., seed=43)]
    selected = summary["selected"]["beauty"]
    assert selected["power"] == 1. and selected["mean_validation_ndcg5"] == pytest.approx(.4)
    assert len(selected["runs"]) == 2
    with (output / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["split"] for row in rows} == {"validation", "test"}


@pytest.mark.parametrize("sampling", ["curriculum", "curriculum_smooth", "curriculum_smooth_is"])
def test_reward_ablations_select_power_within_each_combo_and_sampling_regime(tmp_path, sampling):
    args = ce.parse_args(["--output-root", str(tmp_path), "--datasets", "beauty",
                          "--combos", "interaction", "sasrec_x_similarity",
                          "--sampling", sampling, "--powers", "1", "4", "--rho-targets", "1"])
    for entry in ce.build_grid(args):
        # Different optimal powers: an aggregate must preserve both ablations.
        winning_power = 1 if entry["combo"] == "interaction" else 4
        value = .7 if entry["power"] == winning_power else .2
        directory = ce.run_dir_for(args, entry)
        record = {
            **entry, "sampling": sampling, "curriculum_epochs": 10., "curriculum_steps": 100,
            "status": "completed", "estimator": "sampled", "num_samples": 10,
            "lambda_distill": .5, "best_epoch": 2, "best_val_ndcg5": 1. - value,
            "optimizer_steps": 100, "fit_sec_this_invocation": 20., "run_dir": str(directory),
            "splits": {"validation": {"NDCG@5": value}, "test": {"NDCG@5": 1. - value}},
        }
        ce.write_json(directory / "result.json", record)
        # Other samplers remain separate even with much higher scores.
        for other in set(ce.SAMPLING_ALGORITHMS) - {sampling}:
            args.sampling = other
            direct = ce.run_dir_for(args, entry)
            ce.write_json(direct / "result.json", {**record, "sampling": other, "run_dir": str(direct),
                                                   "splits": {"validation": {"NDCG@5": 1.}, "test": {"NDCG@5": 1.}}})
        args.sampling = sampling
    ce.aggregate(args)
    summary = ce.read_json(ce.summary_dir_for(args) / "summary.json")
    assert summary["completed"] == summary["expected"] == 4
    assert summary["selected"]["beauty"]["interaction"]["power"] == 1
    assert summary["selected"]["beauty"]["sasrec_x_similarity"]["power"] == 4
    assert all(v["mean_validation_ndcg5"] == .7 for v in summary["selected"]["beauty"].values())
    suffix = "_qmix0p1" if sampling == "curriculum_smooth_is" else ""
    assert ce.summary_dir_for(args) == tmp_path / f"ablations/{sampling}_n10_epochs10{suffix}"
    assert summary["sampling_algorithm"] == ce.SAMPLING_ALGORITHMS[sampling]
    assert summary["curriculum_epochs"] == 10


def test_exact_algorithm_and_checkpoint_interval_are_explicit():
    exact, sampled = ce.parse_args(["--estimator", "exact"]), ce.parse_args([])
    assert exact.checkpoint_every_updates == 1 and sampled.checkpoint_every_updates == 0
    identity = ce.build_identity(exact, ce.build_grid(exact)[0], {}, {}, {}, 0)
    assert identity["exact_algorithm"] == "branching_prefix"
