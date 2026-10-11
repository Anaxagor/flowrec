"""Beauty 2023 arrays calculate factors before launching the two fine-tuning arms."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/submit_gpt2_beauty2023.sh"
pytestmark = pytest.mark.skipif(not LAUNCHER.is_file(), reason="Slurm scripts are gitignored")


@pytest.fixture
def launch(tmp_path):
    script = tmp_path / "scripts/slurm/submit_gpt2_beauty2023.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"
    shim = f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps({"tool": tool, "args": args, "env": {
        k: v for k, v in os.environ.items() if k.startswith("SBATCH_") or k in (
            "SCRIPT", "DATASETS", "CACHE_ONLY", "REBUILD_CACHES", "CACHE_COMPONENTS",
            "GRID_INDEX", "SFT_ONLY", "BLOCK_SIZE", "SEEDS", "EXTRA_ARGS")}}) + "\\n")
if tool == "sbatch":
    if os.environ.get("FAIL_SUBMISSION") == "1":
        sys.exit(1)
    print("12345;rocky")
elif "--list" in args:
    print("total: 168 runs (beauty_2023=168)")
elif "--cache-status" in args:
    sys.exit(1 if os.environ.get("MISSING_FACTORS") == "1" else 0)
elif args[:1] != ["-c"]:
    raise SystemExit("unexpected Python action")
'''
    for name in ("python", "sbatch"):
        path = bin_dir / name
        path.write_text(shim)
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALL_LOG": str(log),
           "DRY_RUN": "0", "REBUILD_CACHES": "1", "BLOCK_SIZE": "", "SEEDS": "42 43 44",
           "GFLOW_ONLY": "0",
           "DATASETS": "beauty instruments", "EXTRA_ARGS": "--no-prefix-from-dense --cache-format dense",
           "SCRIPT": "teacherfree.py", "CACHE_ONLY": "1", "SFT_ONLY": "1", "GRID_INDEX": "1",
           "SBATCH_CONSTRAINT": "type_a", "SBATCH_NODELIST": "cn-048",
           "SBATCH_EXCLUDE": "cn-019", "SBATCH_PREFER": "type_f", "SBATCH_BATCH": "type_e",
           "SBATCH_GRES": "gpu:h100:1", "SBATCH_GPUS_PER_NODE": "h100:1",
           "SBATCH_GPUS_PER_SOCKET": "h100:1", "SBATCH_GPUS_PER_TASK": "h100:1",
           "SBATCH_DEPENDENCY": "afterok:old-cache", "SBATCH_ACCOUNT": "proj_1825",
           "SBATCH_MEM_PER_NODE": "200G", "SBATCH_MEM_PER_CPU": "10G", "SBATCH_MEM_PER_GPU": "80G"}

    def run(**overrides):
        result = subprocess.run(["bash", str(script)], env={**env, **overrides},
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_factors_then_sft_and_full_gflow_grid(launch):
    result, calls = launch()
    assert result.returncode == 0, result.stderr
    submissions = [call for call in calls if call["tool"] == "sbatch"]
    assert len(submissions) == 3
    assert [next(arg for arg in call["args"] if arg.startswith("--array="))
            for call in submissions] == ["--array=0-2", "--array=0-2", "--array=0-167"]
    cache, sft, grid = submissions
    assert not any(arg.startswith("--dependency") for arg in cache["args"])
    for call in (sft, grid):
        assert "--dependency=afterok:12345" in call["args"]
        assert call["env"]["REBUILD_CACHES"] == "0"
    assert cache["env"]["REBUILD_CACHES"] == "1"
    assert cache["env"]["CACHE_COMPONENTS"] == "sasrec interaction sim_hier"
    assert [(call["env"]["CACHE_ONLY"], call["env"]["SFT_ONLY"], call["env"]["GRID_INDEX"])
            for call in submissions] == [("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")]
    for call in submissions:
        assert "--account=proj_1752" in call["args"] and "--gpus=1" in call["args"]
        assert "--constraint=type_e|type_f" in call["args"]
        assert not any(arg.startswith(("--nodelist", "--exclude", "--prefer",
                                       "--gres", "--gpus-per", "--mem")) for arg in call["args"])
        assert not any(key in call["env"] for key in (
            "SBATCH_CONSTRAINT", "SBATCH_NODELIST", "SBATCH_EXCLUDE", "SBATCH_PREFER", "SBATCH_BATCH",
            "SBATCH_GRES", "SBATCH_GPUS_PER_NODE", "SBATCH_GPUS_PER_SOCKET", "SBATCH_GPUS_PER_TASK",
            "SBATCH_DEPENDENCY", "SBATCH_MEM_PER_NODE", "SBATCH_MEM_PER_CPU", "SBATCH_MEM_PER_GPU"))
        assert call["env"]["DATASETS"] == "beauty_2023"
        assert call["env"]["SCRIPT"].endswith("gflow_finetune_gpt2_amazon_fullhist.py")
        assert call["env"]["EXTRA_ARGS"] == "--prefix-from-dense --cache-format factorized --num-candidates 3"


def test_failed_factor_submission_stops_before_training(launch):
    result, calls = launch(FAIL_SUBMISSION="1")
    assert result.returncode != 0
    submissions = [call for call in calls if call["tool"] == "sbatch"]
    assert len(submissions) == 1 and submissions[0]["env"]["CACHE_ONLY"] == "1"


def test_dry_run_submits_nothing(launch):
    result, calls = launch(DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(" sbatch ") == 3
    assert not any(call["tool"] == "sbatch" for call in calls)


def test_reuse_factors_and_custom_array_sizes(launch):
    result, calls = launch(REBUILD_CACHES="0", BLOCK_SIZE="5", SEEDS="43 44")
    assert result.returncode == 0, result.stderr
    submissions = [call for call in calls if call["tool"] == "sbatch"]
    assert all(call["env"]["REBUILD_CACHES"] == "0" for call in submissions)
    assert [next(arg for arg in call["args"] if arg.startswith("--array="))
            for call in submissions] == ["--array=0-2", "--array=0-1", "--array=0-33"]


def test_gflow_only_checks_factors_then_submits_one_array(launch):
    result, calls = launch(GFLOW_ONLY="1")
    assert result.returncode == 0, result.stderr
    assert any("--cache-status" in call["args"] for call in calls)
    submissions = [call for call in calls if call["tool"] == "sbatch"]
    assert len(submissions) == 1
    grid = submissions[0]
    assert "--array=0-167" in grid["args"]
    assert not any(arg.startswith("--dependency") for arg in grid["args"])
    assert grid["env"]["GRID_INDEX"] == "1"
    assert "--num-candidates 3 --require-existing-caches" in grid["env"]["EXTRA_ARGS"]
    assert grid["env"]["CACHE_ONLY"] == grid["env"]["SFT_ONLY"] == grid["env"]["REBUILD_CACHES"] == "0"


def test_gflow_only_missing_factors_does_not_submit_or_recalculate(launch):
    result, calls = launch(GFLOW_ONLY="1", MISSING_FACTORS="1")
    assert result.returncode != 0
    assert not any(call["tool"] == "sbatch" for call in calls)


def test_gflow_only_dry_run_has_no_factor_or_sft_jobs(launch):
    result, calls = launch(GFLOW_ONLY="1", DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(" sbatch ") == 1
    assert not any(call["tool"] == "sbatch" for call in calls)
