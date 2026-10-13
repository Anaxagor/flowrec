"""Exercise the GPT-2 Slurm plan without submitting jobs or loading ML assets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/slurm/submit_ce_distill_gpt2.sh"
WORKER = ROOT / "scripts/slurm/ce_distill_gpt2_array.sbatch"
SINGLES_AND_PAIRS = [
    "sasrec", "interaction", "similarity", "sasrec_x_interaction",
    "sasrec_x_similarity", "interaction_x_similarity",
]


@pytest.fixture
def clean_env():
    ignored = {
        "PYTHON_BIN", "ACCOUNT", "PARTITION", "MAX_PARALLEL", "TRAIN_TIME", "EXACT_TIME",
        "EXACT_CATALOG_CHUNK_SIZE", "CACHE_GPU_TIME", "CACHE_CPU_TIME", "BASH_ENV",
        "FLOWREC_ROOT", "TRAIN_WORKERS", "CACHE_GPU_WORKERS", "CACHE_CPU_WORKERS",
    }
    env = {key: value for key, value in os.environ.items() if key not in ignored}
    env["PYTHON_BIN"] = sys.executable
    return env


def plan(env, *arguments, cwd=None):
    result = subprocess.run(
        ["bash", str(HELPER), "--dry-run", *arguments], env=env, cwd=cwd or ROOT,
        capture_output=True, text=True, check=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("sbatch ")]
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


def assert_resources(commands):
    for command in commands:
        resources, phase, _ = parts(command)
        assert resources["account"] == "proj_1752"
        assert resources["partition"] == "rocky"
        assert "array" in resources
        assert not any(key.startswith("mem") or key in {"nodelist", "exclude"} for key in resources)
        if phase == "cache-cpu":
            assert resources["gpus"] == "0"
            assert resources["constraint"] == "type_d"
        else:
            assert resources["gpus"] == "1"
            assert resources["constraint"] == "type_a|type_c"


def test_default_plan_has_cpu_and_gpu_caches_before_training(clean_env):
    commands = plan(clean_env)
    assert len(commands) == 3
    assert [parts(command)[1] for command in commands] == ["cache-gpu", "cache-cpu", "train"]
    assert [parts(command)[0]["array"] for command in commands] == ["0-1%8", "0-3%8", "0-15%8"]
    assert "dependency" not in parts(commands[0])[0]
    assert "dependency" not in parts(commands[1])[0]
    assert parts(commands[2])[0]["dependency"] == "afterok:DRY_cache_gpu:DRY_cache_cpu"
    assert_resources(commands)


@pytest.mark.parametrize("three_seeds,expected", [(False, [16, 96]), (True, [48, 288])])
def test_extensions_share_caches_and_size_both_experiments(clean_env, three_seeds, expected):
    arguments = ["--extensions"]
    if three_seeds:
        arguments += ["--", "--seeds", "42", "43", "44"]
    commands = plan(clean_env, *arguments)
    assert len(commands) == 4
    assert [parts(command)[1] for command in commands] == ["cache-gpu", "cache-cpu", "train", "train"]
    assert [parts(command)[0]["array"] for command in commands[:2]] == ["0-1%8", "0-3%8"]
    for command, count in zip(commands[2:], expected):
        resources, _, _ = parts(command)
        assert resources["array"] == f"0-{count - 1}%8"
        assert resources["dependency"] == "afterok:DRY_cache_gpu:DRY_cache_cpu"
    exact, smooth = [parts(command)[2] for command in commands[2:]]
    assert values(exact, "--estimator") == ["exact"]
    assert values(exact, "--sampling") == ["reward"]
    assert values(exact, "--combos") == ["sasrec_x_interaction_x_similarity"]
    assert values(exact, "--checkpoint-every-updates") == ["1"]
    assert values(smooth, "--estimator") == ["sampled"]
    assert values(smooth, "--sampling") == ["curriculum_smooth"]
    assert values(smooth, "--combos") == SINGLES_AND_PAIRS
    assert_resources(commands)


def test_explicit_smooth_ablation_is_one_96_task_array(clean_env):
    commands = plan(clean_env, "--skip-cache", "--", "--sampling", "curriculum_smooth",
                    "--combos", *SINGLES_AND_PAIRS)
    assert len(commands) == 1
    resources, phase, args = parts(commands[0])
    assert phase == "train" and resources["array"] == "0-95%8"
    assert "dependency" not in resources
    assert values(args, "--sampling") == ["curriculum_smooth"]


def test_trimmed_extension_grids_honor_options_and_literal_paths(clean_env, tmp_path):
    marker = tmp_path / "should-not-exist"
    literal_path = str(tmp_path / f"CE 'quoted' $(touch {marker.name})")
    env = {**clean_env, "MAX_PARALLEL": "3", "EXACT_TIME": "12:00:00"}
    commands = plan(env, "--extensions", "--skip-cache", "--",
                    "--datasets", "instruments", "--powers", ".25", "4", "--rho-targets", ".1",
                    "--seeds", "43", "44", "--catalog-chunk-size", "7", "--curriculum-epochs", "2.5",
                    "--max-epochs", "19", "--checkpoint-every-updates", "2", "--output-root", literal_path,
                    cwd=tmp_path)
    assert len(commands) == 2
    assert [parts(command)[0]["array"] for command in commands] == ["0-3%3", "0-23%3"]
    assert parts(commands[0])[0]["time"] == "12:00:00"
    for command in commands:
        resources, phase, args = parts(command)
        assert phase == "train" and "dependency" not in resources
        assert values(args, "--output-root") == [literal_path]
        assert values(args, "--catalog-chunk-size") == ["7"]
        assert values(args, "--curriculum-epochs") == ["2.5"]
        assert values(args, "--max-epochs") == ["19"]
        assert values(args, "--checkpoint-every-updates") == ["2"]
    assert not marker.exists()
    assert not (ROOT / marker.name).exists()
    assert_resources(commands)


def test_grid_sizing_works_without_site_packages_or_data(clean_env, tmp_path):
    # Metadata-only execution must not require torch/transformers/pandas.
    interpreter = tmp_path / "stdlib-python"
    interpreter.write_text(f"#!/bin/bash\nexec {shlex.quote(sys.executable)} -S \"$@\"\n")
    interpreter.chmod(0o755)
    commands = plan({**clean_env, "PYTHON_BIN": str(interpreter)}, "--extensions", cwd=tmp_path)
    assert [parts(command)[0]["array"] for command in commands[-2:]] == ["0-15%8", "0-95%8"]


@pytest.mark.parametrize("arguments", [
    ["--", "--aggregate-only"], ["--", "--index", "2"], ["--", "--device", "cpu"],
    ["--extensions", "--", "--estimator", "sampled"], ["--", "--powers", "0"],
])
def test_invalid_or_phase_arguments_fail_before_any_plan_is_submitted(clean_env, arguments):
    result = subprocess.run(["bash", str(HELPER), "--dry-run", *arguments], env=clean_env,
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "sbatch " not in result.stdout


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
    srun.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SRUN_JSON=' + json.dumps(sys.argv[1:]))\n")
    srun.chmod(0o755)
    return {**clean_env, "BASH_ENV": str(shell_setup), "FAKE_CONDA": str(profile.parents[1]),
            "PATH": str(binaries) + os.pathsep + clean_env["PATH"], "SLURM_SUBMIT_DIR": str(ROOT),
            "SLURM_ARRAY_JOB_ID": "12345"}


@pytest.mark.parametrize("phase,task,dataset,component,device", [
    ("cache-gpu", 1, "instruments", "sasrec", "cuda"),
    ("cache-cpu", 2, "instruments", "interaction", "cpu"),
    ("cache-cpu", 3, "instruments", "similarity", "cpu"),
])
def test_worker_maps_cache_array_to_dataset_component_without_running_models(worker_env, phase, task, dataset, component, device):
    result = subprocess.run(["bash", str(WORKER), phase, "--output-root", "/tmp/CE experiment"],
                            env={**worker_env, "SLURM_ARRAY_TASK_ID": str(task)},
                            capture_output=True, text=True, check=True)
    calls = [json.loads(line.removeprefix("SRUN_JSON=")) for line in result.stdout.splitlines()
             if line.startswith("SRUN_JSON=")]
    args = calls[-1]
    assert "scripts/exp7-ce-distillation/ce_distill_gpt2.py" in args
    assert values(args, "--datasets") == [dataset]
    assert values(args, "--cache-component") == [component]
    assert values(args, "--device") == [device]
    assert values(args, "--output-root") == ["/tmp/CE experiment"]
    assert "--cache-only" in args
    if device == "cpu":
        assert len(calls) == 1 and values(args, "--num-workers") == ["0"]


def test_worker_preserves_explicit_no_resume_and_array_index(worker_env):
    result = subprocess.run(["bash", str(WORKER), "train", "--no-resume", "--num-workers", "0"],
                            env={**worker_env, "SLURM_ARRAY_TASK_ID": "15"},
                            capture_output=True, text=True, check=True)
    args = json.loads([line.removeprefix("SRUN_JSON=") for line in result.stdout.splitlines()
                       if line.startswith("SRUN_JSON=")][-1])
    assert values(args, "--index") == ["15"]
    assert values(args, "--device") == ["cuda"]
    assert values(args, "--num-workers") == ["0"]
    assert [flag for flag in args if flag in {"--resume", "--no-resume"}][-1] == "--no-resume"
