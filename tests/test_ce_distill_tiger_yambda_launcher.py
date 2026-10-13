"""Check Yambda CE submissions without allocating Slurm resources or loading data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/slurm/submit_ce_distill_tiger_yambda.sh"
WORKER = ROOT / "scripts/slurm/ce_distill_tiger_yambda_array.sbatch"
DATASETS = ("yambda_50m_listens", "yambda_500m_listens")


@pytest.fixture
def clean_env():
    ignored = {
        "PYTHON_BIN", "ACCOUNT", "PARTITION", "TRAIN_TIME", "CACHE_GPU_TIME",
        "CACHE_CPU_TIME", "CACHE_TIME", "BASH_ENV", "FLOWREC_ROOT", "CONDA_ENV",
        "TRAIN_WORKERS", "CACHE_GPU_WORKERS", "CACHE_CPU_WORKERS",
    }
    prefixes = ("MAX_PARALLEL", "TRAIN_50M_", "TRAIN_500M_", "CACHE_GPU_50M_",
                "CACHE_GPU_500M_", "CACHE_CPU_50M_", "CACHE_CPU_500M_")
    env = {key: value for key, value in os.environ.items()
           if key not in ignored and not key.startswith(prefixes)}
    env["PYTHON_BIN"] = sys.executable
    return env


def plan(env, *arguments, cwd=None):
    result = subprocess.run(
        ["bash", str(HELPER), "--dry-run", *arguments], env=env, cwd=cwd or ROOT,
        capture_output=True, text=True, check=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines()
                if line.startswith("sbatch ")]
    assert commands, result.stdout
    return commands


def parts(command):
    boundary = next(i for i, value in enumerate(command) if value.endswith(WORKER.name))
    resources = dict(value[2:].split("=", 1) for value in command[1:boundary]
                     if value.startswith("--") and "=" in value)
    return resources, command[boundary + 1], command[boundary + 2:]


def values(args, flag):
    positions = [i for i, value in enumerate(args) if value == flag]
    if not positions:
        return []
    result = []
    for value in args[positions[-1] + 1:]:
        if value.startswith("--"):
            break
        result.append(value)
    return result


def dataset_commands(commands, dataset):
    return [command for command in commands if values(parts(command)[2], "--datasets") == [dataset]]


def assert_resources(commands):
    for command in commands:
        resources, phase, args = parts(command)
        dataset, = values(args, "--datasets")
        large = dataset == DATASETS[1]
        assert resources["account"] == "proj_1752"
        assert resources["partition"] == "rocky"
        assert "array" in resources
        assert not any(key.startswith("mem") or key in {"nodelist", "exclude", "exclusive"}
                       for key in resources)
        assert "--exclusive" not in command
        if phase == "cache-cpu":
            assert resources["gpus"] == "0"
            assert resources["constraint"] == ("type_b" if large else "type_d")
        else:
            assert resources["gpus"] == "1"
            expected = "type_h" if phase == "train" and large else "type_b" if large else "type_a|type_c"
            assert resources["constraint"] == expected


def test_default_plan_has_112_runs_and_separate_resource_pools(clean_env):
    commands = plan(clean_env)
    assert len(commands) == 6
    for dataset, parallel in zip(DATASETS, (4, 1)):
        selected = dataset_commands(commands, dataset)
        assert [parts(command)[1] for command in selected] == ["cache-gpu", "cache-cpu", "train"]
        assert [parts(command)[0]["array"] for command in selected] == [
            f"0-0%{parallel}", f"0-1%{parallel}", f"0-55%{parallel}"]
        gpu, cpu, train = [parts(command)[0] for command in selected]
        assert "dependency" not in gpu
        assert len(train["dependency"].split(":")) == 3
        if dataset == DATASETS[0]:
            assert "dependency" not in cpu
        else:
            # The large CPU cache cannot share its host with our GPU cache.
            assert cpu["dependency"].startswith("afterok:")
            assert len(cpu["dependency"].split(":")) == 2
            assert cpu["dependency"].split(":")[1] in train["dependency"].split(":")[1:]
    assert_resources(commands)


def test_three_seeds_are_168_runs_per_dataset(clean_env):
    commands = plan(clean_env, "--skip-cache", "--", "--seeds", "42", "43", "44")
    assert len(commands) == 2
    for dataset, parallel in zip(DATASETS, (4, 1)):
        command, = dataset_commands(commands, dataset)
        resources, phase, args = parts(command)
        assert phase == "train" and resources["array"] == f"0-167%{parallel}"
        assert "dependency" not in resources
        assert values(args, "--seeds") == ["42", "43", "44"]
    assert_resources(commands)


@pytest.mark.parametrize("combo,cache_phase", [("interaction", "cache-cpu"),
                                              ("similarity", "cache-cpu"),
                                              ("sasrec", "cache-gpu")])
def test_only_components_used_by_the_grid_get_cache_allocations(clean_env, combo, cache_phase):
    commands = plan(clean_env, "--", "--datasets", DATASETS[1], "--combos", combo,
                    "--powers", "1", "--rho-targets", ".1")
    assert len(commands) == 2
    assert [parts(command)[1] for command in commands] == [cache_phase, "train"]
    cache, train = [parts(command)[0] for command in commands]
    assert cache["array"] == train["array"] == "0-0%1"
    assert "dependency" not in cache
    assert len(train["dependency"].split(":")) == 2
    assert_resources(commands)


def test_literal_paths_and_custom_grid_survive_submission_quoting(clean_env, tmp_path):
    marker = tmp_path / "must-not-be-created"
    literal = str(tmp_path / f"CE 'quoted' $(touch {marker.name})")
    env = {**clean_env, "MAX_PARALLEL": "3", "TRAIN_TIME": "12:00:00"}
    commands = plan(env, "--skip-cache", "--", "--datasets", DATASETS[1],
                    "--combos", "sasrec", "interaction_x_similarity", "--powers", ".25", "4",
                    "--rho-targets", ".1", "--seeds", "43", "44", "--num-samples", "3",
                    "--catalog-chunk-size", "7", "--curriculum-epochs", "2.5", "--max-epochs", "19",
                    "--checkpoint-every-updates", "2", "--output-root", literal, cwd=tmp_path)
    command, = commands
    resources, phase, args = parts(command)
    assert phase == "train" and resources["array"] == "0-7%3"
    assert resources["time"] == "12:00:00"
    for flag, expected in [("--output-root", literal), ("--catalog-chunk-size", "7"),
                           ("--curriculum-epochs", "2.5"), ("--max-epochs", "19"),
                           ("--checkpoint-every-updates", "2"), ("--num-samples", "3")]:
        assert values(args, flag) == [expected]
    assert not marker.exists() and not (ROOT / marker.name).exists()
    assert_resources(commands)


def test_grid_sizing_requires_only_python_stdlib(clean_env, tmp_path):
    interpreter = tmp_path / "stdlib-python"
    interpreter.write_text(f"#!/bin/bash\nexec {shlex.quote(sys.executable)} -S \"$@\"\n")
    interpreter.chmod(0o755)
    commands = plan({**clean_env, "PYTHON_BIN": str(interpreter)}, "--skip-cache", cwd=tmp_path)
    assert [parts(command)[0]["array"] for command in commands] == ["0-55%4", "0-55%1"]


@pytest.mark.parametrize("arguments", [
    ["--", "--aggregate-only"], ["--", "--index", "2"], ["--", "--device", "cpu"],
    ["--", "--cache-only"], ["--", "--help"], ["--", "--datasets", "beauty"],
    ["--", "--powers", "0"],
])
def test_invalid_or_phase_arguments_fail_before_any_submission(clean_env, arguments):
    result = subprocess.run(["bash", str(HELPER), "--dry-run", *arguments], env=clean_env,
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "sbatch " not in result.stdout


def test_real_submission_path_chains_returned_array_ids(clean_env, tmp_path):
    # A fake executable covers submission/dependency wiring, including the
    # optional ';cluster' suffix of sbatch --parsable, without contacting Slurm.
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "submitted.jsonl"
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "p=Path(os.environ['SBATCH_TEST_LOG'])\n"
        "rows=p.read_text().splitlines() if p.exists() else []\n"
        "job_id=7001+len(rows)\n"
        "with p.open('a') as f: f.write(json.dumps({'id':str(job_id),'args':sys.argv[1:]})+'\\n')\n"
        "print(str(job_id)+';test-cluster')\n"
    )
    sbatch.chmod(0o755)
    env = {**clean_env, "PATH": str(binaries) + os.pathsep + clean_env["PATH"],
           "SBATCH_TEST_LOG": str(log)}
    subprocess.run(["bash", str(HELPER)], env=env, cwd=tmp_path,
                   capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 6
    for dataset in DATASETS:
        selected = [row for row in rows if values(parts(["sbatch", *row["args"]])[2], "--datasets") == [dataset]]
        gpu, cpu, train = selected
        cpu_resources = parts(["sbatch", *cpu["args"]])[0]
        train_resources = parts(["sbatch", *train["args"]])[0]
        assert train_resources["dependency"] == f"afterok:{gpu['id']}:{cpu['id']}"
        if dataset == DATASETS[1]:
            assert cpu_resources["dependency"] == f"afterok:{gpu['id']}"
        else:
            assert "dependency" not in cpu_resources


@pytest.fixture
def worker_env(clean_env, tmp_path):
    profile = tmp_path / "conda/etc/profile.d"
    profile.mkdir(parents=True)
    (profile / "conda.sh").write_text("")
    shell_setup = tmp_path / "bash_env"
    shell_setup.write_text('module() { :; }\nconda() { if [[ "$1" == info ]]; then printf "%s\\n" "$FAKE_CONDA"; fi; }\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    srun = binaries / "srun"
    srun.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                    "print('SRUN_JSON=' + json.dumps({'args':sys.argv[1:],'polars':os.environ.get('POLARS_MAX_THREADS')}))\n")
    srun.chmod(0o755)
    return {**clean_env, "BASH_ENV": str(shell_setup), "FAKE_CONDA": str(profile.parents[1]),
            "PATH": str(binaries) + os.pathsep + clean_env["PATH"], "SLURM_SUBMIT_DIR": str(ROOT),
            "SLURM_ARRAY_JOB_ID": "12345"}


def worker_calls(worker_env, phase, task, *arguments):
    result = subprocess.run(["bash", str(WORKER), phase, *arguments],
                            env={**worker_env, "SLURM_ARRAY_TASK_ID": str(task)},
                            capture_output=True, text=True, check=True)
    calls = [json.loads(line.removeprefix("SRUN_JSON=")) for line in result.stdout.splitlines()
             if line.startswith("SRUN_JSON=")]
    assert calls, result.stdout
    return calls


@pytest.mark.parametrize("phase,task,component,device", [
    ("cache-gpu", 0, "sasrec", "cuda"), ("cache-cpu", 0, "interaction", "cpu"),
    ("cache-cpu", 1, "similarity", "cpu"),
])
def test_worker_maps_cache_components_and_uses_identical_preprocessing(worker_env, phase, task, component, device):
    calls = worker_calls(worker_env, phase, task, "--datasets", DATASETS[1],
                         "--output-root", "/tmp/CE experiment")
    args = calls[-1]["args"]
    assert values(args, "--datasets") == [DATASETS[1]]
    assert values(args, "--cache-component") == [component]
    assert values(args, "--device") == [device]
    assert values(args, "--output-root") == ["/tmp/CE experiment"]
    assert "--cache-only" in args
    if device == "cpu":
        assert len(calls) == 1 and values(args, "--num-workers") == ["0"]


def test_worker_filtered_cpu_array_uses_selected_component(worker_env):
    calls = worker_calls(worker_env, "cache-cpu", 0, "--datasets", DATASETS[0],
                         "--combos", "similarity")
    assert values(calls[-1]["args"], "--cache-component") == ["similarity"]


def test_worker_preserves_no_resume_and_local_dataset_index(worker_env):
    calls = worker_calls(worker_env, "train", 55, "--datasets", DATASETS[1],
                         "--no-resume", "--num-workers", "0")
    args = calls[-1]["args"]
    assert values(args, "--index") == ["55"]
    assert values(args, "--datasets") == [DATASETS[1]]
    assert values(args, "--device") == ["cuda"]
    assert values(args, "--num-workers") == ["0"]
    assert [flag for flag in args if flag in {"--resume", "--no-resume"}][-1] == "--no-resume"


def test_worker_preprocessing_thread_count_matches_between_all_phases(worker_env):
    threads = {worker_calls(worker_env, phase, 0, "--datasets", dataset)[-1]["polars"]
               for dataset in DATASETS for phase in ("cache-gpu", "cache-cpu", "train")}
    assert len(threads) == 1
    assert int(threads.pop()) > 0
