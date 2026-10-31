"""Verify sampled-50 training, cache and SFT protocol compatibility."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_tiger_matched as runner  # noqa: E402
import precompute_reward_cache as cache_tools  # noqa: E402
import sft_finetune_control_matched as control_matched  # noqa: E402

control = control_matched.base


@pytest.mark.parametrize("flags", [["--num-workers", "-1"], ["--loader-prefetch-factor", "0"]])
def test_invalid_input_pipeline_settings_fail_before_loading_data(flags):
    with pytest.raises(SystemExit):
        runner.parse_args(["--dataset", "yambda_500m_listens", *flags])


COMMON_ARGS = [
    "--dataset", "yambda_500m_listens", "--train-last-n", "0",
    "--train-random-n", "50", "--train-subset-seed", "42",
    "--rho-targets", "1.0", "--gflow-target-grad-ratio", "1.0",
    "--gflow-weight-mode", "fixed_grad_ratio", "--num-candidates", "10",
    "--seed", "42", "--samplers", "teacher", "--sasrec-transform", "sigmoid",
    "--sasrec-temperature", "1.0",
    "--cache-root", "artifacts/exp4-reward_ablation/gflow_yambda500m_sampled50/reward_factors",
    "--artifacts-dir", "artifacts/exp4-reward_ablation",
    "--metrics-csv-name", "gflow_finetune_yambda_500m_sampled50_top3_metrics_summary.csv",
    "--control-init-baseline", "experiments/exp2-sft-finetune-matched/yambda_500m_listens/init_baseline.json",
]
CELLS = [
    ("sasrec", "4", "db", "best_ndcg"),
    ("sasrec_x_interaction", "16", "subtb", "best_coverage"),
    ("sasrec_x_similarity", "0.25", "db", "pareto"),
]
SFT_ARGS = [
    "--datasets", "yambda_500m_listens", "--seeds", "42", "--lr", "3e-5",
    "--train-random-n", "50", "--train-subset-seed", "42", "--device", "cuda",
    "--num-workers", "2", "--resume", "--output-root", "experiments/exp2-sft-finetune-sampled50",
    "--artifacts-root", "artifacts/exp2-tuned-params/finetune_sampled50",
    "--promote-root", "experiments/exp2-sft-finetune-sampled50/promoted/tiger",
]


def task_args(task, *, cache=False):
    if cache:
        component = ("sasrec", "interaction", "similarity")[task]
        args = ["--device", "cpu" if component == "interaction" else "cuda",
                "--num-workers", "0" if component == "interaction" else "8",
                "--cache-only", "--cache-components", component, "--combos", component,
                "--powers", "1", "--objectives", "db", "--output-root",
                f"experiments/exp4-gflow-yambda500m-sampled50-top3/cache_builds/{component}"]
    else:
        combo, power, objective, label = CELLS[task]
        args = ["--device", "cuda", "--num-workers", "2", "--combos", combo,
                "--powers", power, "--objectives", objective, "--loader-prefetch-factor", "2",
                "--reward-prefetch", "--index", "0", "--resume", "--output-root",
                f"experiments/exp4-gflow-yambda500m-sampled50-top3/seed_42/{label}"]
    return runner.parse_args([*COMMON_ARGS, *args])


@pytest.mark.parametrize("cell,combo,objective,power,label", [
    (0, ("sasrec",), "db", 4.0, "best_ndcg"),
    (1, ("sasrec", "interaction"), "subtb", 16.0, "best_coverage"),
    (2, ("sasrec", "similarity"), "db", 0.25, "pareto"),
])
def test_all_tasks_keep_selected_protocol_and_use_separate_sampled50_outputs(
    cell, combo, objective, power, label,
):
    seed = 42
    args = task_args(cell)
    entry, = runner.base.build_grid(args)
    assert (entry["combo"], entry["objective"], entry["power"], entry["rho"]) == (
        combo, objective, power, 1.0)
    assert entry["num_negatives"] == 10 and entry["sampler"] == "teacher"
    assert args.seed == seed and args.resume and args.index == "0"
    assert (args.train_last_n, args.train_random_n, args.train_subset_seed) == (0, 50, 42)
    assert args.output_root == Path("experiments/exp4-gflow-yambda500m-sampled50-top3") / f"seed_{seed}" / label
    assert args.cache_root == Path("artifacts/exp4-reward_ablation/gflow_yambda500m_sampled50/reward_factors")
    assert args.metrics_csv_name == "gflow_finetune_yambda_500m_sampled50_top3_metrics_summary.csv"
    assert args.device == "cuda" and args.num_workers == 2
    assert args.reward_prefetch and args.tensorized_train_loader and args.loader_prefetch_factor == 2
    assert args.gflow_weight_mode == "fixed_grad_ratio"
    assert (args.batch_size, args.grad_accum) == (256, 8)
    assert (args.validate_every, args.patience, args.min_epochs, args.curriculum_epochs) == (1, 5, 10, 10)
    assert args.val_sample_fraction == 0.25 and not args.val_sample_resample
    assert args.validate_at_start and args.eval_num_beams == 20 and args.test_ks == [5, 10, 20]
    assert not args.smoke and args.max_train_batches is args.max_eval_batches is None
    config = runner.base.build_run_config(args)
    # Sample AFTER constructing the original last-128 layout: keep the catalog
    # and semantic IDs compatible with the original SFT initialization.
    assert config["dataset"]["dataset_size"] == "500m"
    assert config["dataset"]["train_target_policy"] == "last_n"
    assert config["dataset"]["train_last_n"] == 128
    runner.check_uniform_micro_batch(args)


def test_sample_matches_cache_rows_for_every_configuration_and_extends_sampled15():
    dataset = SimpleNamespace(examples=pd.DataFrame({"user_id": [0] * 128 + [1] * 50 + [2] * 17}))
    args = task_args(0)
    selected = runner.base.finetune_indices(dataset, args)
    assert len(selected) == 50 + 50 + 17
    assert selected == sorted(set(selected))
    assert dataset.examples.iloc[selected].groupby("user_id").size().to_dict() == {0: 50, 1: 50, 2: 17}
    assert set(range(128, 195)) <= set(selected)  # retain all short users' rows
    assert min(selected) < 64 and selected[:50] != list(range(78, 128))
    original = dataset.examples.copy(deep=True)
    for task in range(3):
        training = task_args(task)
        cache = task_args(task % 3, cache=True)
        spec = runner.base.subset_spec(cache)
        assert spec == runner.base.subset_spec(training) == {
            "per_user_last_n": None, "per_user_random_n": 50, "per_user_random_seed": 42,
        }
        assert cache.cache_root == training.cache_root
        assert runner.base.finetune_indices(dataset, training) == cache_tools.subset_indices(dataset, **spec) == selected
    pd.testing.assert_frame_equal(dataset.examples, original)
    sampled15 = runner.parse_args(["--dataset", "yambda_500m_listens"])
    assert sampled15.train_random_n == 15  # the historical protocol was not changed
    assert set(runner.base.finetune_indices(dataset, sampled15)) < set(selected)
    assert selected != cache_tools.per_user_random_indices(dataset, 50, 43)
    # Curriculum is ten epochs of the selected examples, including partial batches.
    assert runner.base.curriculum_steps_for_batch(args, 32, len(selected)) == 10 * 4


@pytest.fixture
def sft_args(monkeypatch):
    """Resolve the matched SFT wrapper, intercepting only the training entry."""
    # main retargets these module globals; restore them for unrelated tests.
    monkeypatch.setattr(control, "DATASETS", control.DATASETS)
    monkeypatch.setattr(control, "DEFAULT_OUTPUT_ROOT", control.DEFAULT_OUTPUT_ROOT)
    resolved = []
    monkeypatch.setattr(control, "main", lambda argv: resolved.append(control.parse_args(argv)))
    control_matched.main(SFT_ARGS)
    args, = resolved
    return args


def test_sft_only_uses_one_seed_the_same_examples_and_initial_checkpoint(sft_args):
    args = sft_args
    gflow = task_args(0)
    dataset = "yambda_500m_listens"
    assert args.datasets == [dataset] and args.seeds == [42]
    assert args.lr == [gflow.lr] == [3e-5]
    assert args.train_random_n == 50 and args.train_subset_seed == 42
    dm = SimpleNamespace(train_dataset=SimpleNamespace(
        examples=pd.DataFrame({"user_id": [0] * 128 + [1] * 50 + [2] * 17}),
    ))
    assert control.finetune_subset_indices(dm, dataset, args) == runner.base.finetune_indices(dm.train_dataset, gflow)
    assert control.abspath(control.DATASETS[dataset]["tiger_checkpoint"]) == runner.base.abspath(runner.base.TIGER_CHECKPOINT)
    config = control.build_run_config(dataset, args.lr[0], args.seeds[0], args)
    gflow_config = runner.base.build_run_config(gflow)
    assert config["method"] == {"name": "sft"}
    assert config["training"]["init_checkpoint_path"] == str(runner.base.abspath(runner.base.TIGER_CHECKPOINT))
    for key in ("train_target_policy", "train_last_n", "dataset_size"):
        assert config["dataset"][key] == gflow_config["dataset"][key]
    assert config["model"] == gflow_config["model"]
    assert config["data"] == gflow_config["data"]


def test_sft_only_matches_training_and_evaluation_protocol(sft_args):
    args = sft_args
    gflow = task_args(0)
    for attr in ("batch_size", "grad_accum", "val_batch_size", "weight_decay", "max_grad_norm",
                 "max_epochs", "validate_every", "patience", "min_epochs", "validate_at_start",
                 "eval_num_beams", "test_ks", "num_workers", "device"):
        assert getattr(args, attr) == getattr(gflow, attr), attr
    config = control.build_run_config(args.datasets[0], args.lr[0], args.seeds[0], args)
    training = config["training"]
    assert training["amp_dtype"] == gflow.amp_dtype == "float16"
    assert training["val_sample_fraction"] == gflow.val_sample_fraction == 0.25
    assert training["val_sample_resample"] == gflow.val_sample_resample is False
    assert training["generation_num_beams"] == 5
    trainer = control.tiger_runner.build_trainer_config(config, Path("test-output"))
    assert trainer.val_sample_seed == 0 and trainer.monitor == "val/sft/ndcg@5"
    assert trainer.early_stopping_mode == "best"
    assert trainer.save_best and trainer.save_last
    assert not args.smoke and args.max_train_batches is args.max_eval_batches is None
    assert args.resume and not args.aggregate_only and not args.skip_init_eval


def test_sft_outputs_cannot_resume_sampled15_or_full128(sft_args):
    args = sft_args
    assert args.output_root == Path("experiments/exp2-sft-finetune-sampled50")
    assert args.artifacts_root == Path("artifacts/exp2-tuned-params/finetune_sampled50")
    assert args.promote_root == Path("experiments/exp2-sft-finetune-sampled50/promoted/tiger")
    assert args.output_root != task_args(0).output_root
    run_dir = control.run_dir_for(args.output_root, args.datasets[0], args.lr[0], args.seeds[0])
    assert str(run_dir).endswith("yambda_500m_listens/sft_lr3e-05/seed_42")
    assert all("sampled50" in str(path) for path in (args.output_root, args.artifacts_root, args.promote_root))


@pytest.mark.parametrize("n,seed", [(None, None), (15, 42), (50, 43)])
def test_wrong_subset_caches_are_rejected_even_if_example_counts_match(tmp_path, n, seed):
    args = task_args(0)
    manifest = {"kind": runner.base.CACHE_KIND, "component": {"name": "sasrec"},
                "num_examples": 117, "per_user_last_n": None,
                "per_user_random_n": 50, "per_user_random_seed": 42}
    assert runner.base.check_component_manifest(manifest, tmp_path, "sasrec", 117, args)
    with pytest.raises(RuntimeError, match="fine-tunes on 50 random examples"):
        runner.base.check_component_manifest(
            {**manifest, "per_user_random_n": n, "per_user_random_seed": seed},
            tmp_path, "sasrec", 117, args,
        )


@pytest.mark.parametrize("task,component", [(0, "sasrec"), (1, "interaction"), (2, "similarity")])
def test_cache_build_commands_receive_the_actual_sampled50_draw(monkeypatch, tmp_path, task, component):
    args = task_args(task, cache=True)
    assert args.cache_only and args.cache_components == [component]
    assert args.device == ("cpu" if component == "interaction" else "cuda")
    assert args.num_workers == (0 if component == "interaction" else 8)
    assert runner.base.components_for([e["combo"] for e in runner.base.build_grid(args)]) == [component]
    args.cache_root = tmp_path / "cache"
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        out = Path(cmd[cmd.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({
            "kind": runner.base.CACHE_KIND, "component": {"name": component, "mode": "hierarchical"},
            "num_examples": 117, **runner.base.subset_spec(args),
        }))

    monkeypatch.setattr(runner.base.subprocess, "run", fake_run)
    runner.base.ensure_component_cache(component, runner.base.build_run_config(args), tmp_path, args, 117)
    command, = commands
    assert command[command.index("--per-user-random-n") + 1] == "50"
    assert command[command.index("--per-user-random-seed") + 1] == "42"
    assert "--per-user-last-n" not in command
