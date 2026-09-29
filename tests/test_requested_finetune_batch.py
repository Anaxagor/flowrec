"""Exact requested cluster batch, without contacting Slurm or querying nodes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_resources as resources

SUBMIT = ROOT / "scripts/slurm/submit_finetune_batch.sh"
pytestmark = pytest.mark.skipif(not SUBMIT.exists(), reason="Slurm scripts are gitignored")


def test_type_only_plan_does_not_query_node_inventory(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Node inventory must not be queried")

    monkeypatch.setattr(resources, "live_nodes", forbidden)
    monkeypatch.setattr(resources, "static_nodes", forbidden)
    assert resources.main(["plan", "--placement", "types", "--live", "--datasets",
                           "beauty", "instruments", "yambda_50m_listens", "yambda_500m_listens"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    for line in lines:
        flags, datasets, mode = line.split("\t")
        assert mode == "free" and "--nodelist" not in flags
        expected = ("type_e|type_f|type_h" if datasets == "yambda_500m_listens"
                    else "type_a|type_b|type_c|type_e|type_f|type_h")
        assert f"--constraint={expected}" in flags
    assert "--constraint=type_e|type_f|type_h" in lines[-1]


@pytest.mark.parametrize("mode,tasks", [(None, 201), ("teacher", 201), ("teacherfree", 369)])
def test_real_submission_arguments_with_mock_slurm(tmp_path, mode, tasks):
    log = tmp_path / "sbatch.jsonl"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(f"#!{sys.executable}\n" +
                     "import json, os, sys\nfrom pathlib import Path\n"
                     "p = Path(os.environ['SBATCH_LOG'])\n"
                     "old = p.read_text().splitlines() if p.exists() else []\n"
                     "with p.open('a') as f:\n"
                     "    f.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
                     "print(1001 + len(old))\n")
    sbatch.chmod(0o755)
    sinfo = tmp_path / "sinfo"
    sinfo.write_text('#!/bin/bash\necho queried > "$SINFO_LOG"\nexit 91\n')
    sinfo.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n")
    squeue.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
           "SBATCH_LOG": str(log), "SINFO_LOG": str(tmp_path / "sinfo.log"),
           "DRY_RUN": "0", "BLOCK_SIZE": "4", "MAX_PARALLEL": "20",
           "SEEDS": "42 43 44", "EXTRA_ARGS": "", "SBATCH_EXTRA": "", "FORCE": "0",
           "SBATCH_CONSTRAINT": "type_a", "SBATCH_NODELIST": "forbidden-node", "LIVE_NODES": "1"}
    if mode is None:
        env.pop("GFLOW_MODE", None)
    else:
        env["GFLOW_MODE"] = mode
    result = subprocess.run(["bash", str(SUBMIT)], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 12
    assert not (tmp_path / "sinfo.log").exists()
    assert f"array tasks submitted: {tasks}" in result.stdout
    for call in calls:
        argv, exported = call["argv"], call["env"]
        big = exported.get("DATASETS") == "yambda_500m_listens"
        constraints = [flag for flag in argv if flag.startswith("--constraint")]
        assert constraints == (["--constraint=type_e|type_f|type_h"] if big else
                               ["--constraint=type_a|type_b|type_c|type_e|type_f|type_h"])
        assert not any(flag.startswith("--nodelist") for flag in argv)
        assert "SBATCH_CONSTRAINT" not in exported and "SBATCH_NODELIST" not in exported
        script = ROOT / argv[-1]
        assert not any(line.startswith("#SBATCH") and any(option in line for option in
                       ("--constraint", "--nodelist")) for line in script.read_text().splitlines())
    tiger_controls = [c for c in calls if c["argv"][-1].endswith("sft_finetune_control_matched_array.sbatch")]
    assert {c["env"]["DATASETS"] for c in tiger_controls} == {"yambda_50m_listens", "yambda_500m_listens"}
    assert len(tiger_controls) == 3
    tiger500 = [c for c in tiger_controls if c["env"]["DATASETS"] == "yambda_500m_listens"]
    assert len(tiger500) == 2
    assert sum(c["env"]["EXTRA_ARGS"] == "--train-random-n 0" for c in tiger500) == 1
    assert len({next(f for f in c["argv"] if f.startswith("--job-name=")) for c in tiger500}) == 2
    # All requested GPT-2 controls are prefix runs, including the all-target diagnostic.
    assert not any(c["argv"][-1].endswith("sft_control_gpt2_amazon_dense.sbatch") for c in calls)
    gpt2_controls = [c for c in calls if c["env"].get("SFT_ONLY") == "1"]
    assert len(gpt2_controls) == 3
    yambda_controls = [c for c in gpt2_controls if c["env"].get("DATASETS") == "yambda_50m_listens"]
    assert len(yambda_controls) == 2
    assert sum("--yambda50-all-targets" in c["env"]["EXTRA_ARGS"] for c in yambda_controls) == 1
    assert all(c["env"]["SCRIPT"].endswith("gflow_finetune_gpt2_yambda.py") for c in yambda_controls)
    assert all("--array=0-2%20" in c["argv"] for c in yambda_controls)
    assert not any("--yambda50-all-targets" in c["env"].get("EXTRA_ARGS", "")
                   for c in calls if c not in yambda_controls)
    # All cache submissions come first, force rebuilding, and every training
    # array waits for every cache to succeed. Fine-tuning never forces a rebuild.
    assert all(c["env"].get("CACHE_ONLY") == "1" for c in calls[:3])
    assert all(c["env"]["REBUILD_CACHES"] == "1" for c in calls[:3])
    for call in calls[3:]:
        assert call["env"].get("CACHE_ONLY") != "1"
        assert call["env"]["REBUILD_CACHES"] == "0"
        assert "--dependency=afterok:1001:1002:1003" in call["argv"]
    # All GFlow grids use the array index for configuration selection.
    grids = [c for c in calls if any(flag.endswith("_grid") for flag in c["argv"])]
    assert len(grids) == 3
    assert all(any(flag.startswith("--dependency=afterok:") for flag in c["argv"]) for c in grids)
    assert all(any(flag.startswith("--array=") for flag in c["argv"]) for c in calls)
    job_names = {flag.split("=", 1)[1] for c in calls for flag in c["argv"] if flag.startswith("--job-name=")}
    if mode == "teacherfree":
        assert "flowrec_teacherfree_gpt2_grid" in job_names
        assert "flowrec_teacherfree_gpt2_yambda_yambda_50m_listens_grid" in job_names
        assert "flowrec_teacherfree_tiger_yambda_500m_listens_grid" in job_names
        assert "GFlow mode: teacherfree" in result.stdout
    else:
        assert "flowrec_gpt2_amazon_grid" in job_names
        assert "flowrec_gpt2_yambda_yambda_50m_listens_grid" in job_names
        assert "flowrec_tiger_gflow_yambda_yambda_500m_listens_grid" in job_names
        assert not any("teacherfree" in name for name in job_names)
        assert "GFlow mode: teacher (standard" in result.stdout
        # The standard ablation builds SASRec as well as interaction and
        # hierarchical similarity, before training. sim_embed is separate.
        assert "--array=0-2,4-6%20" in calls[0]["argv"]
        assert calls[1]["env"]["CACHE_COMPONENTS"] == "sasrec interaction similarity"
        gpt2_grid = next(c for c in grids if "--job-name=flowrec_gpt2_amazon_grid" in c["argv"])
        tiger_grid = next(c for c in grids if c["argv"][-1].endswith("gflow_finetune_tiger_matched_array.sbatch"))
        yambda_grid = next(c for c in grids if c["env"].get("DATASETS") == "yambda_50m_listens")
        assert "--array=0-41%20" in yambda_grid["argv"]
        assert "--array=0-83%20" in gpt2_grid["argv"]   # two datasets x 168 / four cells
        assert "--array=0-41%20" in tiger_grid["argv"]  # one dataset x 168 / four cells


@pytest.mark.parametrize("extra_args", ["", "--train-last-n 128"])
def test_dense_sbatch_dispatches_yambda_seed_without_losing_dataset(tmp_path, extra_args):
    profile = tmp_path / "etc/profile.d"
    profile.mkdir(parents=True)
    (profile / "conda.sh").write_text("")
    startup = tmp_path / "bash_env"
    startup.write_text('module() { :; }\nconda() { if [[ "$1" == info ]]; then echo "$MOCK_CONDA_BASE"; fi; }\n')
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                      "with open(os.environ['ARGV_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    python.chmod(0o755)
    log = tmp_path / "argv.jsonl"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "BASH_ENV": str(startup),
           "MOCK_CONDA_BASE": str(tmp_path), "ARGV_LOG": str(log), "SLURM_SUBMIT_DIR": str(ROOT),
           "DATASETS": "beauty instruments yambda_50m_listens", "SLURM_ARRAY_TASK_ID": "7",
           "SEEDS": "42 43 44", "EXTRA_ARGS": extra_args}
    subprocess.run(["bash", str(ROOT / "scripts/slurm/sft_control_gpt2_amazon_dense.sbatch")],
                   env=env, capture_output=True, text=True, check=True)
    call = json.loads(log.read_text().splitlines()[-1])
    assert call[0].endswith("sft_control_gpt2_amazon_dense.py")
    assert call[call.index("--datasets") + 1] == "yambda_50m_listens"
    assert call[call.index("--seed") + 1] == "43"
    assert ("--train-last-n" in call) == bool(extra_args)
    if extra_args:
        assert call[call.index("--train-last-n") + 1] == "128"


@pytest.mark.parametrize("family", ["gpt2", "tiger"])
@pytest.mark.parametrize("cache_only", ["0", "1"])
def test_cache_recalculation_flag_is_only_sent_by_cache_tasks(tmp_path, family, cache_only):
    profile = tmp_path / "etc/profile.d"
    profile.mkdir(parents=True)
    (profile / "conda.sh").write_text("")
    startup = tmp_path / "bash_env"
    startup.write_text('module() { :; }\nconda() { if [[ "$1" == info ]]; then echo "$MOCK_CONDA_BASE"; fi; }\n')
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                      "with open(os.environ['ARGV_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                      "if '--list' in sys.argv: print('total: 336')\n")
    python.chmod(0o755)
    log = tmp_path / "argv.jsonl"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "BASH_ENV": str(startup),
           "MOCK_CONDA_BASE": str(tmp_path), "ARGV_LOG": str(log), "SLURM_SUBMIT_DIR": str(ROOT),
           "DATASETS": "beauty" if family == "gpt2" else "yambda_500m_listens",
           "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "100", "SLURM_JOB_ID": "100",
           "CACHE_ONLY": cache_only, "CACHE_COMPONENTS": "interaction", "REBUILD_CACHES": "1",
           "SFT_ONLY": "0", "GRID_INDEX": "1", "BLOCK_SIZE": "4", "EXTRA_ARGS": ""}
    name = "gflow_finetune_gpt2_amazon_fullhist" if family == "gpt2" else "gflow_finetune_tiger_matched_array"
    subprocess.run(["bash", str(ROOT / "scripts/slurm" / f"{name}.sbatch")],
                   env=env, capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(("--force-cache" in call) == (cache_only == "1") for call in calls)
    if cache_only == "0":
        assert calls[-1][calls[-1].index("--index") + 1] == "0-3"
