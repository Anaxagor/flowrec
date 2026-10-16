"""Training-only restart must not inherit GPU restrictions or rebuild factors."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/slurm/relaunch_gpt2_yambda50.sh"
pytestmark = pytest.mark.skipif(not LAUNCHER.exists(), reason="Slurm scripts unavailable")


@pytest.fixture
def launch(tmp_path):
    script = tmp_path / "scripts/slurm/relaunch_gpt2_yambda50.sh"
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
            "SCRIPT", "DATASETS", "CACHE_ONLY", "REBUILD_CACHES", "GRID_INDEX",
            "SFT_ONLY", "BLOCK_SIZE", "SEEDS", "EXTRA_ARGS")}}) + "\\n")
if tool == "sbatch":
    print("12345")
elif "--list" in args:
    print("total: 168 runs (yambda_50m_listens=168)")
elif "--cache-status" in args:
    sys.exit(int(os.environ.get("MISSING_FACTORS", "0")))
elif args[:1] != ["-c"]:
    raise SystemExit("unexpected Python action")
'''
    for name in ("python", "sbatch"):
        path = bin_dir / name
        path.write_text(shim)
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALL_LOG": str(log),
           "DRY_RUN": "0", "BLOCK_SIZE": "4", "SEEDS": "42 43 44",
           "CACHE_ONLY": "1", "REBUILD_CACHES": "1", "EXTRA_ARGS": "--force-cache",
           "SBATCH_CONSTRAINT": "type_e|type_f", "SBATCH_NODELIST": "cn-048",
           "SBATCH_EXCLUDE": "cn-019", "SBATCH_PREFER": "type_f", "SBATCH_BATCH": "type_e",
           "SBATCH_GRES": "gpu:h100:1", "SBATCH_GPUS_PER_NODE": "h100:1",
           "SBATCH_GPUS_PER_SOCKET": "h100:1", "SBATCH_GPUS_PER_TASK": "h100:1",
           "SBATCH_DEPENDENCY": "afterok:old-cache-job", "SBATCH_ACCOUNT": "proj_1825",
           "SBATCH_EXTRA": "--constraint=type_e", "MAX_PARALLEL": "1"}

    def run(**overrides):
        result = subprocess.run(["bash", str(script)], env={**env, **overrides},
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_submits_only_three_training_arrays_without_placement_restrictions(launch):
    result, calls = launch()
    assert result.returncode == 0, result.stderr
    submissions = [call for call in calls if call["tool"] == "sbatch"]
    assert len(submissions) == 3
    assert [next(arg for arg in call["args"] if arg.startswith("--array="))
            for call in submissions] == ["--array=0-41", "--array=0-2", "--array=0-2"]
    for call in submissions:
        assert "--account=proj_1752" in call["args"] and "--gpus=1" in call["args"]
        assert not any(arg.startswith(("--constraint", "--nodelist", "--exclude", "--prefer",
                                       "--dependency", "--gres", "--gpus-per")) for arg in call["args"])
        assert not any(key in call["env"] for key in (
            "SBATCH_CONSTRAINT", "SBATCH_NODELIST", "SBATCH_EXCLUDE", "SBATCH_PREFER", "SBATCH_BATCH",
            "SBATCH_GRES", "SBATCH_GPUS_PER_NODE", "SBATCH_GPUS_PER_SOCKET", "SBATCH_GPUS_PER_TASK",
            "SBATCH_DEPENDENCY"))
        assert call["env"]["CACHE_ONLY"] == call["env"]["REBUILD_CACHES"] == "0"
        assert call["env"]["DATASETS"] == "yambda_50m_listens"
        assert call["env"]["SCRIPT"].endswith("gflow_finetune_gpt2_yambda.py")
        assert "--yambda50-base legacy" in call["env"]["EXTRA_ARGS"]
        assert "--test-batch-size 64" in call["env"]["EXTRA_ARGS"]
        assert "--eval-num-beams 20 --test-ks 5 10 20" in call["env"]["EXTRA_ARGS"]
        assert "--force-cache" not in call["env"]["EXTRA_ARGS"]
    assert [(call["env"]["SFT_ONLY"], call["env"]["GRID_INDEX"]) for call in submissions] == [
        ("0", "1"), ("1", "0"), ("1", "0")]
    assert ["--yambda50-all-targets" in call["env"]["EXTRA_ARGS"] for call in submissions] == [False, False, True]
    cache_check = next(i for i, call in enumerate(calls) if "--cache-status" in call["args"])
    assert calls[cache_check]["args"][1:3] == ["--yambda50-base", "legacy"]
    assert all(i > cache_check for i, call in enumerate(calls) if call["tool"] == "sbatch")


def test_missing_factors_stop_before_any_submission(launch):
    result, calls = launch(MISSING_FACTORS="1")
    assert result.returncode != 0
    assert not any(call["tool"] == "sbatch" for call in calls)
    assert not any("--cache-only" in call["args"] for call in calls)


def test_dry_run_needs_no_factor_files_and_submits_nothing(launch):
    result, calls = launch(DRY_RUN="1", MISSING_FACTORS="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(" sbatch ") == 3
    assert not any(call["tool"] == "sbatch" or "--cache-status" in call["args"] for call in calls)


def test_grid_partition_and_control_seed_count_are_configurable(launch):
    result, calls = launch(BLOCK_SIZE="5", SEEDS="43 44")
    assert result.returncode == 0, result.stderr
    submissions = [call for call in calls if call["tool"] == "sbatch"]
    assert [next(arg for arg in call["args"] if arg.startswith("--array="))
            for call in submissions] == ["--array=0-33", "--array=0-1", "--array=0-1"]
