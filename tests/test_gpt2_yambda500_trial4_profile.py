"""The compact 500M experiment has exactly the requested grid and OOM budget."""
import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_yambda as experiment


def profile(*extra):
    return experiment.parse_args(["--yambda500-base", "trial4", "--device", "cpu", *extra])


@pytest.mark.parametrize("targets", ["last128", "sampled50"])
def test_exact_requested_grid_with_calibrated_rho_and_three_negatives(targets):
    args = profile("--yambda500-targets", targets)
    grid = experiment.base.build_grid(args)
    assert len(grid) == 36
    expected = set(itertools.product(
        [("sasrec", "interaction", "sim_hier"), ("sasrec",), ("interaction",), ("sim_hier",)],
        [1.0, 0.25, 16.0], ["tb", "db", "subtb"],
    ))
    actual = {(tuple(row["components"]), row["power"], row["objective"]) for row in grid}
    assert actual == expected
    assert {row["num_negatives"] for row in grid} == {3}
    assert {row["rho"] for row in grid} == {0.1}
    assert all(set(row["weights"]) == {1.0} for row in grid)
    assert args.gflow_weight_mode == "fixed_grad_ratio"
    assert args.samplers == ["teacher"] and args.seed == 42


def test_matched_targets_and_frozen_validation_are_preserved():
    args = profile()
    assert args.datasets == ["yambda_500m_listens"]
    assert args.yambda500_base == "trial4" and args.yambda50_base == "legacy"
    assert args.yambda500_targets == "last128"
    assert args.train_last_n == args.train_pool_last_n == 128
    assert args.train_random_n == 0 and not args.prefix_from_dense
    assert args.val_sample_fraction == 0.25 and not args.val_sample_resample
    assert args.resume and args.validate_at_start
    assert (args.max_epochs, args.validate_every, args.patience, args.min_epochs) == (500, 1, 5, 10)
    assert args.curriculum_epochs == 10
    assert args.cache_format == "factorized"
    assert args.reward_prefetch and args.tensorized_train_loader
    assert args.amp_dtype == "bfloat16"
    assert args.num_workers == 4 and args.loader_prefetch_factor == 2


def test_sampled50_uses_a_fixed_subset_from_the_original_last128_pool():
    args = profile("--yambda500-targets", "sampled50")
    assert args.yambda500_targets == "sampled50"
    assert args.train_pool_last_n == 128
    assert args.train_last_n == 0 and args.train_random_n == 50
    assert args.train_subset_seed == 42
    assert args.val_sample_fraction == 0.25 and not args.val_sample_resample


@pytest.mark.parametrize("targets", ["last128", "sampled50"])
def test_sft_control_uses_the_same_training_protocol_as_gflow(targets):
    gflow = profile("--yambda500-targets", targets)
    sft = profile("--yambda500-targets", targets, "--sft-only")
    assert sft.sft_only and not gflow.sft_only
    assert {key: value for key, value in vars(sft).items() if key != "sft_only"} == {
        key: value for key, value in vars(gflow).items() if key != "sft_only"
    }
    assert experiment.base.run_batch_settings(sft, None) == experiment.base.run_batch_settings(gflow, 3)
    dataset = "yambda_500m_listens"
    assert experiment.base.tuned_checkpoint_paths(dataset, sft) == experiment.base.tuned_checkpoint_paths(dataset, gflow)


@pytest.mark.parametrize("negative_count", [3, None])
@pytest.mark.parametrize("targets", ["last128", "sampled50"])
def test_oom_backoff_reaches_batch_one_without_changing_effective_batch(negative_count, targets):
    args = profile("--yambda500-targets", targets)
    assert args.oom_retries == 8
    assert (args.val_batch_size, args.test_batch_size, args.eval_num_beams) == (256, 128, 20)
    for batch in (256, 128, 64, 32, 16, 8, 4, 2, 1):
        actual, accumulation = experiment.base.run_batch_settings(args, negative_count)
        assert actual == batch and actual * accumulation == 2048
        assert args.max_candidate_batch == batch * 4
        args = experiment.base.oom_retry_args(args, negative_count)


def test_paths_are_isolated_from_legacy_and_50m():
    new = profile()
    old = experiment.parse_args(["--datasets", "yambda_500m_listens"])
    fifty = experiment.parse_args([])
    for field in ("output_root", "cache_root", "metrics_csv"):
        value = getattr(new, field)
        assert "yambda500" in str(value) and "trial4" in str(value)
        assert value != getattr(old, field) and value != getattr(fifty, field)


def test_sampled50_paths_cannot_resume_or_reuse_last128_outputs():
    full, sampled = profile(), profile("--yambda500-targets", "sampled50")
    assert str(full.output_root).endswith("exp4-gflow-gpt2-yambda500-trial4-k3-rho0p1")
    assert str(full.cache_root).endswith(experiment.TRIAL4_NAME)
    assert str(full.metrics_csv).endswith(experiment.TRIAL4_NAME + "_metrics_summary.csv")
    assert str(sampled.output_root).endswith("exp4-gflow-gpt2-yambda500-trial4-k3-rho0p1-sampled50")
    assert str(sampled.cache_root).endswith(experiment.TRIAL4_NAME + "_sampled50")
    assert str(sampled.metrics_csv).endswith(experiment.TRIAL4_NAME + "_sampled50_metrics_summary.csv")
    for field in ("output_root", "cache_root", "metrics_csv"):
        assert getattr(sampled, field) != getattr(full, field)


@pytest.mark.parametrize("flags", [[], ["--datasets", "yambda_500m_listens"], ["--yambda50-base", "trial36"]])
def test_sampled50_requires_the_trial4_source(flags):
    with pytest.raises((SystemExit, ValueError)):
        experiment.parse_args([*flags, "--yambda500-targets", "sampled50"])


@pytest.mark.parametrize("flags", [
    ["--train-last-n", "50"],
    ["--train-random-n", "15"],
    ["--train-subset-seed", "17"],
])
def test_sampled50_cannot_silently_change_the_shared_subset(flags):
    with pytest.raises((SystemExit, ValueError)):
        profile("--yambda500-targets", "sampled50", *flags)


@pytest.mark.parametrize("flags", [
    ["--datasets", "yambda_50m_listens"],
    ["--yambda50-base", "trial36"],
    ["--teacher-free"],
    ["--train-random-n", "15"],
])
def test_incompatible_source_or_target_selection_is_rejected(flags):
    with pytest.raises((SystemExit, ValueError)):
        profile(*flags)


def test_explicit_smaller_batch_can_be_selected_for_a_retry():
    args = profile("--batch-size", "128", "--grad-accum", "16",
                   "--max-candidate-batch", "512", "--test-batch-size", "64")
    assert experiment.base.run_batch_settings(args, 3) == (128, 16)
    assert args.test_batch_size == 64
