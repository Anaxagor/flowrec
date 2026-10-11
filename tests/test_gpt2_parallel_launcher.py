"""Run the real Slurm shell entry points with stubbed cluster commands."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARRAY = ROOT / "scripts/slurm/tuning_gpt2_array.sbatch"
WORKERS = ROOT / "scripts/slurm/tuning_gpt2_workers_array.sbatch"
pytestmark = pytest.mark.skipif(
    not (ARRAY.is_file() and WORKERS.is_file()), reason="Slurm scripts are gitignored"
)


@pytest.fixture
def launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda_base = tmp_path / "conda"
    profile = conda_base / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text("conda() { return 0; }\n")
    log = tmp_path / "python_calls.jsonl"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "conda": '#!/bin/sh\nprintf "%s\\n" "$TEST_CONDA_BASE"\n',
        "date": "#!/bin/sh\necho 2026-10-09T12:00:00+03:00\n",
        "python": f"#!{sys.executable}\n" + (
            "import json, os, sys\n"
            "with open(os.environ['TEST_CALL_LOG'], 'a') as output:\n"
            "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        ),
    }
    for name, content in scripts.items():
        path = bin_dir / name
        path.write_text(content)
        path.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("GPT2_")}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}", TEST_CONDA_BASE=str(conda_base),
        TEST_CALL_LOG=str(log), SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_JOB_ID="123",
        SLURM_JOB_ID="124", CUDA_VISIBLE_DEVICES="0",
    )

    def run(task_id, *, script=ARRAY, **overrides):
        log.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                                env={**env, "SLURM_ARRAY_TASK_ID": str(task_id), **overrides})
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_default_array_runs_four_workers_per_yambda(launch):
    directive = next(line.split("=", 1)[1] for line in ARRAY.read_text().splitlines()
                     if line.startswith("#SBATCH --array="))
    indices, concurrency = directive.split("%")
    assert concurrency == "8"
    task_ids = []
    for entry in indices.split(","):
        bounds = [int(value) for value in entry.split("-")]
        task_ids.extend(range(bounds[0], bounds[-1] + 1))
    assert task_ids == [3, 4, 8, 9, 13, 14, 18, 19]
    workers = {"yambda_50m_listens": [], "yambda_500m_listens": []}
    for task_id in task_ids:
        result, calls = launch(task_id)
        assert result.returncode == 0, result.stderr
        args, = calls
        assert args[0] == "scripts/exp2-tuned-params/tuning_gpt2.py"
        dataset = args[args.index("--datasets") + 1]
        workers[dataset].append(int(args[args.index("--worker-id") + 1]))
        assert "--match-tiger" in args and "--fail-stale-trials" not in args
        assert args[args.index("--train-batch-size") + 1] == "256"
        assert args[args.index("--device") + 1] == "cuda"
        assert "--n-trials" not in args  # The study retains its shared budget.
    assert workers == {"yambda_50m_listens": [0, 1, 2, 3], "yambda_500m_listens": [0, 1, 2, 3]}


@pytest.mark.parametrize("script", [ARRAY, WORKERS])
def test_cluster_resources_select_one_80gb_gpu_without_memory_or_node_pinning(script):
    directives = [line for line in script.read_text().splitlines() if line.startswith("#SBATCH")]
    assert '#SBATCH --constraint="type_e|type_f"' in directives
    assert "#SBATCH --gpus=1" in directives
    assert "#SBATCH --account=proj_1752" in directives
    assert not any("--mem" in line or "--nodelist" in line or "--exclusive" in line for line in directives)


@pytest.mark.parametrize("task_id,dataset,worker", [
    (0, "beauty", "0"), (1, "instruments", "0"), (2, "beauty_2023", "0"),
    (5, "beauty", "1"), (54, "yambda_500m_listens", "10"),
])
def test_old_dataset_indices_and_additional_workers(launch, task_id, dataset, worker):
    result, calls = launch(task_id)
    assert result.returncode == 0, result.stderr
    args, = calls
    assert args[args.index("--datasets") + 1] == dataset
    assert args[args.index("--worker-id") + 1] == worker
    assert ("--match-tiger" in args) == dataset.startswith("yambda")


@pytest.mark.parametrize("task_id", [-1, "oops", ""])
def test_invalid_task_fails_without_starting_training(launch, task_id):
    result, calls = launch(task_id)
    assert result.returncode != 0 and not calls
    assert "nonnegative integer" in result.stderr


@pytest.mark.parametrize("script,first,later", [(ARRAY, 4, 9), (WORKERS, 0, 1)])
def test_recovery_runs_only_first_worker(launch, script, first, later):
    result, calls = launch(first, script=script, GPT2_TUNE_FAIL_STALE="1",
                           GPT2_TUNE_DATASET="yambda_500m_listens")
    assert result.returncode == 0 and "--fail-stale-trials" in calls[0]
    result, calls = launch(later, script=script, GPT2_TUNE_FAIL_STALE="1",
                           GPT2_TUNE_DATASET="yambda_500m_listens")
    assert result.returncode == 0 and not calls
    assert "worker 0 only" in result.stdout


def test_explicit_legacy_mode_and_matched_microbatch_override(launch):
    result, calls = launch(8, GPT2_MATCH_TIGER="0")
    assert result.returncode == 0 and "--match-tiger" not in calls[0]
    result, calls = launch(8, GPT2_TRAIN_BATCH_SIZE="64")
    assert result.returncode == 0
    assert calls[0][calls[0].index("--train-batch-size") + 1] == "64"
