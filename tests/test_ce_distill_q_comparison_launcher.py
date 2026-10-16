"""Check paired CE plans without contacting Slurm or importing ML packages."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/slurm/submit_ce_distill_q_comparison.sh"
COMBOS = [
    "sasrec", "interaction", "similarity", "sasrec_x_interaction",
    "sasrec_x_similarity", "interaction_x_similarity", "sasrec_x_interaction_x_similarity",
]


@pytest.fixture
def clean_env():
    ignored = {
        "PYTHON_BIN", "ACCOUNT", "PARTITION", "TRAIN_TIME", "EXACT_TIME",
        "EXACT_CATALOG_CHUNK_SIZE", "CACHE_GPU_TIME", "CACHE_CPU_TIME", "BASH_ENV",
        "FLOWREC_ROOT", "CONDA_ENV", "TRAIN_WORKERS", "CACHE_GPU_WORKERS", "CACHE_CPU_WORKERS",
    }
    prefixes = ("SBATCH_", "MAX_PARALLEL", "TRAIN_50M_", "TRAIN_500M_",
                "CACHE_GPU_50M_", "CACHE_GPU_500M_", "CACHE_CPU_50M_", "CACHE_CPU_500M_")
    env = {key: value for key, value in os.environ.items()
           if key not in ignored and not key.startswith(prefixes)}
    env["PYTHON_BIN"] = sys.executable
    return env


def invoke(env, *args, cwd=None):
    return subprocess.run(["bash", str(HELPER), *args], cwd=cwd or ROOT, env=env,
                          capture_output=True, text=True)


def plan(env, *args, cwd=None):
    result = invoke(env, "--dry-run", *args, cwd=cwd)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()
                if line.startswith("sbatch ")]
    assert commands, result.stdout
    return commands


def parts(command):
    boundary = next(i for i, value in enumerate(command) if value.endswith(".sbatch"))
    resources = dict(value[2:].split("=", 1) for value in command[1:boundary]
                     if value.startswith("--") and "=" in value)
    return resources, command[boundary], command[boundary + 1], command[boundary + 2:]


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


@pytest.mark.parametrize("profile", ["tiger", "gpt2", "tiger-yambda"])
def test_default_plan_pairs_all_112_cells_without_rebuilding_caches(clean_env, profile):
    commands = plan(clean_env, "--profile", profile, "--skip-cache")
    yambda = profile == "tiger-yambda"
    assert len(commands) == (4 if yambda else 2)
    samples = []
    for command in commands:
        resource, worker, phase, args = parts(command)
        assert phase == "train"
        assert worker.endswith("ce_distill_" + profile.replace("-", "_") + "_array.sbatch")
        assert resource["gpus"] == "1"
        assert resource["account"] == "proj_1752"
        assert resource["partition"] == "rocky"
        assert "dependency" not in resource
        assert not any(key.startswith("mem") or key in {"nodelist", "exclude", "exclusive"}
                       for key in resource)
        assert values(args, "--estimator") == ["sampled"]
        assert values(args, "--combos") == COMBOS
        assert values(args, "--curriculum-epochs") == ["5"]
        # Keep each runner's historical defaults: n10 Amazon / n3 Yambda and
        # its original output root; completed reward cells can be reused.
        assert "--num-samples" not in args and "--output-root" not in args
        assert "--cache-root" not in args
        mode, = values(args, "--sampling")
        samples.append(mode)
        assert values(args, "--proposal-q-mix") == (["0.1"] if mode == "curriculum_smooth_is" else [])
        if yambda:
            dataset, = values(args, "--datasets")
            large = dataset == "yambda_500m_listens"
            assert resource["array"] == ("0-55%1" if large else "0-55%4")
            assert resource["constraint"] == ("type_h" if large else "type_a|type_c")
        else:
            assert resource["array"] == "0-111%8"
            assert resource["constraint"] == "type_a|type_c"
    assert samples == (["reward", "reward", "curriculum_smooth_is", "curriculum_smooth_is"]
                       if yambda else ["reward", "curriculum_smooth_is"])


@pytest.mark.parametrize("profile,dataset", [
    ("tiger", "beauty"), ("gpt2", "instruments"), ("tiger-yambda", "yambda_50m_listens"),
])
def test_shared_grid_and_quoted_paths_are_forwarded_to_both_arms(clean_env, tmp_path, profile, dataset):
    output = str(tmp_path / "CE comparison $(do-not-run)")
    commands = plan(clean_env, "--profile", profile, "--proposal-q-mix", "0.2", "--",
                    "--datasets", dataset, "--combos", "sasrec", "interaction", "--powers", ".25", "4",
                    "--rho-targets", ".1", "--seeds", "42", "44", "--num-samples", "7",
                    "--curriculum-epochs", "3.5", "--output-root", output, cwd=tmp_path)
    assert len(commands) == 2
    for command in commands:
        resources, _, _, args = parts(command)
        assert resources["array"].startswith("0-7%")
        for flag, expected in {
            "--datasets": [dataset], "--combos": ["sasrec", "interaction"],
            "--powers": [".25", "4"], "--rho-targets": [".1"], "--seeds": ["42", "44"],
            "--num-samples": ["7"], "--curriculum-epochs": ["3.5"], "--output-root": [output],
        }.items():
            assert values(args, flag) == expected
    assert values(parts(commands[0])[3], "--proposal-q-mix") == []
    assert values(parts(commands[1])[3], "--proposal-q-mix") == ["0.2"]


def test_default_profile_and_explicit_skip_cache_have_the_same_plan(clean_env):
    assert plan(clean_env) == plan(clean_env, "--profile=tiger", "--skip-cache")


def test_resource_environment_applies_to_both_modes(clean_env):
    commands = plan({**clean_env, "ACCOUNT": "proj_1925", "TRAIN_TIME": "3-00:00:00", "MAX_PARALLEL": "2"})
    for command in commands:
        resources = parts(command)[0]
        assert resources["account"] == "proj_1925"
        assert resources["time"] == "3-00:00:00"
        assert resources["array"] == "0-111%2"


@pytest.mark.parametrize("args", [
    ["--profile", "unknown"], ["--profile"], ["--proposal-q-mix"],
    ["--", "--sampling", "reward"], ["--", "--sampling=reward"],
    ["--", "--estimator", "exact"], ["--", "--estimator=sampled"],
    ["--", "--proposal-q-mix", "0.1"], ["--", "--extensions"],
    ["--", "--cache-only"], ["--", "--index", "0"], ["--", "--device", "cpu"],
    ["--", "--datasets", "beauty", "beauty"], ["--", "--powers", "nan"],
])
def test_invalid_or_conflicting_arguments_do_not_print_a_partial_plan(clean_env, args):
    result = invoke(clean_env, "--dry-run", *args)
    assert result.returncode != 0
    assert "sbatch " not in result.stdout


@pytest.fixture
def fake_sbatch_env(clean_env, tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "submissions.jsonl"
    stub = binaries / "sbatch"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "p=Path(os.environ['CE_TEST_SUBMISSIONS'])\n"
        "rows=p.read_text().splitlines() if p.exists() else []\n"
        "with p.open('a') as f: f.write(json.dumps({'args':sys.argv[1:],"
        "'unexpected_env':{k:v for k,v in os.environ.items() if k.startswith('SBATCH_')}})+'\\n')\n"
        "print(str(8100+len(rows))+';test-cluster')\n"
    )
    stub.chmod(0o755)
    return {**clean_env, "PATH": str(binaries) + os.pathsep + clean_env["PATH"],
            "CE_TEST_SUBMISSIONS": str(log)}, log


@pytest.mark.parametrize("bad_mix", ["0", "-0.1", "1.1", "nan", "not-a-number", "1e-50", "1e-40"])
def test_invalid_second_arm_never_submits_first_arm(fake_sbatch_env, bad_mix):
    env, log = fake_sbatch_env
    result = invoke(env, "--proposal-q-mix", bad_mix)
    assert result.returncode != 0
    assert not log.exists()
    assert "sbatch " not in result.stdout


@pytest.mark.parametrize("profile,expected", [("tiger", 2), ("gpt2", 2), ("tiger-yambda", 4)])
def test_submission_wiring_uses_only_stub_sbatch_and_removes_inherited_resource_overrides(fake_sbatch_env, profile, expected):
    env, log = fake_sbatch_env
    env = {**env, "SBATCH_MEM_PER_NODE": "999G", "SBATCH_NODELIST": "a-fixed-node", "SBATCH_GPUS": "8"}
    result = invoke(env, "--profile", profile)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == expected
    for row in rows:
        assert row["unexpected_env"] == {}
        resources, _, phase, args = parts(["sbatch", *row["args"]])
        assert phase == "train"
        assert resources["gpus"] == "1"
        assert "dependency" not in resources
        assert values(args, "--estimator") == ["sampled"]
    modes = [values(parts(["sbatch", *row["args"]])[3], "--sampling")[0] for row in rows]
    assert modes == ["reward"] * (expected // 2) + ["curriculum_smooth_is"] * (expected // 2)
