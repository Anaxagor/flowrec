"""The selected trial's ablation profile keeps targets and optimization aligned."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_yambda as experiment


def test_default_is_selected_yambda50_trial_with_three_negatives():
    args = experiment.parse_args([])
    assert args.datasets == ["yambda_50m_listens"]
    assert args.yambda50_base == "trial36"
    grid = experiment.base.build_grid(args)
    assert len(grid) == 168
    assert {entry["num_negatives"] for entry in grid} == {3}
    assert args.train_pool_last_n == 128 and args.train_random_n == 0
    assert not args.prefix_from_dense
    assert args.cache_format == "factorized"
    assert args.reward_prefetch and args.tensorized_train_loader
    assert args.loader_prefetch_factor == 2
    assert args.resume and args.validate_at_start


@pytest.mark.parametrize("flags", [[], ["--sft-only"], ["--teacher-free"]])
def test_candidate_budget_retains_the_effective_batch_in_all_arms(flags):
    args = experiment.parse_args(flags)
    assert experiment.base.run_batch_settings(args, 3) == (128, 16)
    assert experiment.base.run_batch_settings(args, None) == (128, 16)
    assert args.max_candidate_batch == 512 and args.oom_retries == 7
    assert args.val_batch_size == 256 and args.test_batch_size == 64
    assert args.eval_num_beams == 20
    assert args.validate_every == 1 and args.patience == 5 and args.min_epochs == 10


def test_trial36_outputs_are_separate_from_legacy_and_teacherfree_runs():
    new = experiment.parse_args([])
    old = experiment.parse_args(["--yambda50-base", "legacy"])
    teacherfree = experiment.parse_args(["--teacher-free"])
    assert old.datasets == list(experiment.DATASETS)
    for field in ("output_root", "cache_root", "metrics_csv"):
        assert getattr(new, field) != getattr(old, field)
        assert "trial36" in str(getattr(new, field))
    assert new.cache_root == teacherfree.cache_root
    assert new.output_root != teacherfree.output_root
    assert new.metrics_csv != teacherfree.metrics_csv


def test_explicit_yambda500_keeps_existing_protocol():
    args = experiment.parse_args(["--datasets", "yambda_500m_listens"])
    assert args.yambda50_base == "legacy" and args.prefix_from_dense
    assert args.train_random_n == 15 and args.train_pool_last_n == 128
    assert args.num_candidates == [10]
    assert args.cache_root == Path(experiment.CACHE_ROOT)


@pytest.mark.parametrize("datasets", [["yambda_500m_listens"], list(experiment.DATASETS)])
def test_selected_trial_rejects_other_datasets(datasets):
    with pytest.raises(SystemExit):
        experiment.parse_args(["--yambda50-base", "trial36", "--datasets", *datasets])


def test_explicit_memory_override_preserves_accumulation():
    args = experiment.parse_args(["--batch-size", "64", "--grad-accum", "32",
                                  "--max-candidate-batch", "256", "--test-batch-size", "32"])
    assert experiment.base.run_batch_settings(args, 3) == (64, 32)
    assert args.test_batch_size == 32
