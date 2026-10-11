"""Selected full-scale jobs: resolve real training arguments; mock only Slurm."""

import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/slurm/gflow_finetune_yambda500m_last128_top3.sbatch"
CPU_WORKER = ROOT / "scripts/slurm/gflow_yambda500m_interaction_cache_cpu.sbatch"
SUBMIT = ROOT / "scripts/slurm/submit_gflow_yambda500m_last128_top3.sh"
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in (WORKER, CPU_WORKER, SUBMIT)),
    reason="Slurm scripts are gitignored",
)
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_tiger_matched as runner  # noqa: E402


def task_args(task, *, cache=False):
    result = subprocess.run(
        ["bash", str(WORKER)], cwd=ROOT, capture_output=True, text=True, check=True,
        env={**os.environ, "DRY_RUN": "1", "CACHE_ONLY": str(int(cache)),
             "SLURM_ARRAY_TASK_ID": str(task), "SLURM_CPUS_PER_TASK": "8"},
    )
    command = shlex.split(result.stdout.strip())
    assert command[:2] == ["python", "scripts/exp4-reward-ablations/gflow_finetune_tiger_matched.py"]
    return runner.parse_args(command[2:])


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("cell,combo,objective,power,label", [
    (0, ("sasrec",), "db", 4.0, "best_ndcg"),
    (1, ("sasrec", "interaction"), "subtb", 16.0, "best_coverage"),
    (2, ("sasrec", "similarity"), "db", 0.25, "pareto"),
])
def test_training_tasks_resolve_to_exact_selected_cells(seed, cell, combo, objective, power, label):
    task = (seed - 42) * 3 + cell
    args = task_args(task)
    grid = runner.base.build_grid(args)
    assert len(grid) == 1
    entry = grid[0]
    assert (entry["combo"], entry["objective"], entry["power"], entry["rho"]) == (
        combo, objective, power, 1.0)
    assert entry["num_negatives"] == 10 and entry["sampler"] == "teacher"
    assert args.seed == seed and args.resume and args.index == "0"
    assert args.device == "cuda"
    assert args.num_workers == 2
    assert args.reward_prefetch and args.tensorized_train_loader
    assert args.loader_prefetch_factor == 2
    assert args.gflow_weight_mode == "fixed_grad_ratio"
    assert args.output_root.name == label
    assert args.output_root.parent.name == f"seed_{seed}"
    assert args.train_last_n == args.train_random_n == 0
    assert runner.base.finetune_indices(range(260), args) == list(range(260))
    config = runner.base.build_run_config(args)
    assert config["dataset"]["dataset_size"] == "500m"
    assert config["dataset"]["train_target_policy"] == "last_n"
    assert config["dataset"]["train_last_n"] == 128
    assert (args.batch_size, args.grad_accum) == (256, 8)
    assert (args.validate_every, args.patience, args.min_epochs, args.curriculum_epochs) == (1, 5, 10, 10)
    assert args.validate_at_start and not args.val_sample_resample
    assert args.val_sample_fraction == 0.25
    assert not args.smoke and args.max_train_batches is args.max_eval_batches is None
    assert args.eval_num_beams == 20 and args.test_ks == [5, 10, 20]
    runner.check_uniform_micro_batch(args)
    protocol = runner.load_protocol()["arms"]["tiger_gflow"]
    for attr, mapping in (("cache_root", "cache_roots"), ("metrics_csv_name", "metrics_csv_names")):
        assert str(getattr(args, attr)) != protocol[mapping]["yambda_500m_listens"]


def test_training_workers_can_be_disabled_without_changing_training_protocol():
    result = subprocess.run(
        ["bash", str(WORKER)], cwd=ROOT, capture_output=True, text=True, check=True,
        env={**os.environ, "DRY_RUN": "1", "CACHE_ONLY": "0", "SLURM_ARRAY_TASK_ID": "1",
             "TRAIN_NUM_WORKERS": "0", "LOADER_PREFETCH_FACTOR": "1"},
    )
    args = runner.parse_args(shlex.split(result.stdout.strip())[2:])
    assert args.num_workers == 0 and args.loader_prefetch_factor == 1
    assert (args.batch_size, args.grad_accum, args.num_candidates) == (256, 8, [10])
    assert args.reward_prefetch and args.tensorized_train_loader


@pytest.mark.parametrize("flags", [["--num-workers", "-1"], ["--loader-prefetch-factor", "0"]])
def test_invalid_input_pipeline_settings_fail_before_loading_data(flags):
    with pytest.raises(SystemExit):
        runner.parse_args(["--dataset", "yambda_500m_listens", *flags])


@pytest.mark.parametrize("task,component", [(0, "sasrec"), (1, "interaction"), (2, "similarity")])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_cache_tasks_cover_the_training_components_with_identical_rows(task, component, seed):
    cache, training = task_args(task, cache=True), task_args((seed - 42) * 3 + task)
    assert cache.cache_only and cache.cache_components == [component]
    grid = runner.base.build_grid(cache)
    needed = runner.base.components_for([entry["combo"] for entry in grid])
    assert needed == [component]
    assert cache.device == ("cpu" if component == "interaction" else "cuda")
    assert cache.num_workers == (0 if component == "interaction" else 8)
    assert cache.cache_root == training.cache_root
    assert runner.base.subset_spec(cache) == runner.base.subset_spec(training) == {
        "per_user_last_n": None, "per_user_random_n": None, "per_user_random_seed": None,
    }


@pytest.fixture
def mock_slurm(tmp_path):
    log = tmp_path / "sbatch.jsonl"
    executable = tmp_path / "sbatch"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['SBATCH_LOG'])\n"
        "previous = log.read_text().splitlines() if log.exists() else []\n"
        "with log.open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_CACHE') == '1': sys.exit(1)\n"
        "if sys.argv[-1].endswith('interaction_cache_cpu.sbatch'):\n"
        "    if os.environ.get('FAIL_CPU_CACHE') == '1': sys.exit(1)\n"
        "    if any(os.environ.get(k) for k in ('SBATCH_GPUS', 'SBATCH_GRES')): sys.exit(2)\n"
        "print(str(1001 + len(previous)) + ';cluster')\n"
    )
    executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "SBATCH_LOG": str(log), "DRY_RUN": "0", "FAIL_CACHE": "0", "FAIL_CPU_CACHE": "0",
           "SBATCH_CONSTRAINT": "type_h", "SBATCH_TIMELIMIT": "3-00:00:00",
           "SBATCH_GPUS": "1", "SBATCH_GRES": "gpu:1"}
    return log, env


def test_submission_chains_three_caches_then_nine_training_jobs(mock_slurm, tmp_path):
    log, env = mock_slurm
    subprocess.run(["bash", str(SUBMIT)], cwd=tmp_path, env=env,
                   capture_output=True, text=True, check=True)
    cache, cpu_cache, train = [json.loads(line) for line in log.read_text().splitlines()]
    for command in (cache, cpu_cache, train):
        assert "--time=2-00:00:00" in command
        assert (ROOT / command[-1]) == (CPU_WORKER if command == cpu_cache else WORKER)
    assert [flag for flag in cache if flag.startswith("--constraint=")] == [
        "--constraint=type_a|type_b|type_c"]
    assert [flag for flag in train if flag.startswith("--constraint=")] == [
        "--constraint=type_e|type_f"]
    assert "--constraint=type_b" in cpu_cache
    assert "--cpus-per-task=1" in cpu_cache
    assert not any(flag.startswith(("--gpus", "--gres")) for flag in cpu_cache)
    assert "--export=ALL,CACHE_ONLY=1,DRY_RUN=0" in cache
    assert "--array=0,2" in cache
    assert "--export=ALL,CACHE_ONLY=1,DRY_RUN=0" in cpu_cache
    assert "--array=1" in cpu_cache
    assert "--export=ALL,CACHE_ONLY=0,DRY_RUN=0" in train
    assert "--array=0-8" in train
    assert "--dependency=afterok:1001:1002" in train


@pytest.mark.parametrize("failed_phase,count", [("FAIL_CACHE", 1), ("FAIL_CPU_CACHE", 2)])
def test_cache_submission_failure_does_not_submit_training(mock_slurm, failed_phase, count):
    log, env = mock_slurm
    result = subprocess.run(["bash", str(SUBMIT)], cwd=ROOT,
                            env={**env, failed_phase: "1"}, capture_output=True, text=True)
    assert result.returncode != 0
    assert len(log.read_text().splitlines()) == count


def test_dry_run_never_submits_jobs(mock_slurm):
    log, env = mock_slurm
    result = subprocess.run(["bash", str(SUBMIT)], cwd=ROOT,
                            env={**env, "DRY_RUN": "1"}, capture_output=True, text=True, check=True)
    assert not log.exists()
    assert sum(line.startswith("python ") for line in result.stdout.splitlines()) == 9
    cache, cpu_cache, train = [shlex.split(line) for line in result.stdout.splitlines()
                               if line.startswith("sbatch ")]
    assert "--constraint=type_a|type_b|type_c" in cache
    assert "--constraint=type_b" in cpu_cache
    assert cpu_cache[-1].endswith("interaction_cache_cpu.sbatch")
    assert "--constraint=type_e|type_f" in train


def test_direct_sbatch_uses_requested_resource_limits():
    directives = [line for line in WORKER.read_text().splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --time=2-00:00:00" in directives
    assert '#SBATCH --constraint="type_e|type_f"' in directives
    assert "#SBATCH --array=0-8" in directives


def test_cpu_cache_launcher_resolves_only_interaction_without_requesting_gpu():
    directives = [line for line in CPU_WORKER.read_text().splitlines() if line.startswith("#SBATCH")]
    assert not any("--gpu" in line or "--gres" in line for line in directives)
    assert "#SBATCH --cpus-per-task=1" in directives
    assert "#SBATCH --time=2-00:00:00" in directives
    assert "#SBATCH --constraint=type_b" in directives
    result = subprocess.run(["bash", str(CPU_WORKER)], capture_output=True, text=True, check=True,
                            env={**os.environ, "SLURM_SUBMIT_DIR": str(ROOT), "DRY_RUN": "1",
                                 "SLURM_CPUS_PER_TASK": "1", "SLURM_ARRAY_TASK_ID": "1"})
    args = runner.parse_args(shlex.split(result.stdout.strip())[2:])
    assert args.device == "cpu" and args.num_workers == 0
    assert args.cache_only and args.cache_components == ["interaction"]
    assert args.cache_root == task_args(1, cache=True).cache_root
    invalid = subprocess.run(["bash", str(CPU_WORKER)], capture_output=True, text=True,
                             env={**os.environ, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "0"})
    assert invalid.returncode == 2


@pytest.mark.parametrize("task,cache,expected", [(9, False, "0..8"), (3, True, "0..2")])
def test_worker_rejects_out_of_range_task(task, cache, expected):
    result = subprocess.run(["bash", str(WORKER)], capture_output=True, text=True,
                            env={**os.environ, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": str(task),
                                 "CACHE_ONLY": str(int(cache))})
    assert result.returncode == 2
    assert f"must be {expected}" in result.stderr


def test_nine_runs_keep_distinct_outputs_and_persist_all_seed_metrics(tmp_path):
    """Synthetic eval results exercise the actual writer used after fine-tuning."""
    configurations = [task_args(task) for task in range(9)]
    assert len({args.output_root for args in configurations}) == 9
    assert len({runner.base.metrics_csv_path(args) for args in configurations}) == 1
    csv_path = tmp_path / configurations[0].metrics_csv_name
    run_dirs = set()
    all_rows = []
    for args in configurations:
        entry, = runner.base.build_grid(args)
        run_dir = runner.base.run_dir_for(args.output_root, entry, args.seed)
        run_dirs.add(run_dir)
        identity = {
            **runner.base.csv_identity(entry), "dataset": args.dataset,
            "combo": runner.base.combo_key(entry["combo"]), "power": entry["power"],
            "objective": entry["objective"], "num_negatives": entry["num_negatives"],
            "seed": args.seed, "variant": "gflow",
        }
        metrics = {f"{metric}@{k}": args.seed / 100 for k in args.test_ks
                   for metric in ("HR", "NDCG", "Coverage", "ARP")}
        rows = runner.base.metric_rows(identity, {"validation": metrics, "test": metrics},
                                       {"status": "completed", "target_grad_ratio": entry["rho"],
                                        "run_dir": str(run_dir)})
        runner.base.upsert_metrics_csv(csv_path, rows)
        all_rows.extend(rows)
    assert len(run_dirs) == 9
    # Re-saving after a resume replaces matching rows, retaining other seeds.
    runner.base.upsert_metrics_csv(csv_path, all_rows[:2])
    with csv_path.open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 18
    assert {(row["combo"], row["seed"], row["split"]) for row in saved} == {
        (combo, str(seed), split)
        for combo in ("sasrec", "sasrec_x_interaction", "sasrec_x_similarity")
        for seed in (42, 43, 44) for split in ("validation", "test")
    }
    for row in saved:
        assert row["status"] == "completed"
        for k in (5, 10, 20):
            for metric in ("HR", "NDCG", "Coverage", "ARP"):
                assert float(row[f"{metric}@{k}"]) == int(row["seed"]) / 100
