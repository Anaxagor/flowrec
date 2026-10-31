"""Metadata grid/source selection stays exact and usable without ML packages."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp8-from-sft/finetune_from_sft.py"
spec = importlib.util.spec_from_file_location("exp8_grid_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def args(*flags):
    return runner.parse_args(["--count", *flags])


def test_default_grid_has_nine_complete_unique_seed42_source_sweeps():
    grid = runner.build_grid(args())
    assert len(grid) == 1089
    assert Counter(cell["arm"] for cell in grid) == {"sft":9, "gflow":648, "ce":432}
    assert len({runner.cell_key(cell) for cell in grid}) == len(grid)
    assert {cell["source_seed"] for cell in grid} == {42}
    assert {cell["seed"] for cell in grid} == {42}
    sources = {cell["source_id"] for cell in grid}
    assert sources == {
        "tiger_beauty", "tiger_instruments", "tiger_yambda50_matched", "tiger_yambda500_matched_r15",
        "gpt2_beauty", "gpt2_instruments", "gpt2_beauty_2023", "gpt2_yambda50_trial36",
        "gpt2_yambda500_trial4_sampled50",
    }
    for source in sources:
        cells = [cell for cell in grid if cell["source_id"] == source]
        assert Counter(cell["arm"] for cell in cells) == {"sft":1, "gflow":72, "ce":48}
        gflow = [cell for cell in cells if cell["arm"] == "gflow"]
        ce = [cell for cell in cells if cell["arm"] == "ce"]
        for arm in (gflow, ce):
            assert {tuple(cell["combo"]) for cell in arm} == {
                ("sasrec",), ("interaction",), ("similarity",), ("sasrec", "interaction", "similarity")}
            assert {cell["power"] for cell in arm} == {.25,1.,16.}
            assert {cell["rho"] for cell in arm} == {.1,1.}
        assert {cell["num_negatives"] for cell in gflow} == {3}
        assert {cell["objective"] for cell in gflow} == {"tb", "db", "subtb"}
        assert {cell["num_samples"] for cell in ce} == {3}
        assert {cell["sampling"] for cell in ce} == {"reward", "curriculum_smooth_is"}


@pytest.mark.parametrize("profile,sources", [("small",4),("large",1),("yambda",4)])
def test_resource_filters_preserve_cell_identity_and_prepare_indexing(profile,sources):
    selected = args("--resource-profile",profile)
    jobs = runner.source_jobs(selected)
    grid = runner.build_grid(selected)
    assert len(jobs) == sources and len(grid) == 121*sources
    complete = runner.build_grid(args())
    expected = [cell for cell in complete if runner.resource_profile(cell["dataset"]) == profile]
    assert grid == expected
    cpu = runner.source_jobs(args("--phase","prepare","--prepare-kind","cpu","--resource-profile",profile))
    gpu = runner.source_jobs(args("--phase","prepare","--prepare-kind","gpu","--resource-profile",profile))
    assert cpu == gpu == jobs


@pytest.mark.parametrize("flags,count", [
    (["--models","gpt2"],605), (["--models","tiger"],484),
    (["--datasets","beauty","instruments"],484),
    (["--models","gpt2","--datasets","beauty","instruments"],242),
    (["--arms","sft"],9), (["--arms","gflow"],648), (["--arms","ce"],432),
    (["--models","tiger","--datasets","beauty_2023"],0),
])
def test_dataset_model_and_arm_filters(flags,count):
    assert len(runner.build_grid(args(*flags))) == count


def test_training_seed42_is_required():
    with pytest.raises(SystemExit):
        args("--seeds","43")


def test_source_seed_aliases_are_normalized_and_deduplicated(monkeypatch):
    monkeypatch.setattr(runner.protocol, "primary_source_seed", lambda source_id, seed: 42)
    selected = args("--source-ids", "tiger_beauty", "--source-seeds", "42", "43", "44")
    jobs = runner.source_jobs(selected)
    assert jobs == [{"source_id": "tiger_beauty", "model": "tiger", "dataset": "beauty", "source_seed": 42}]
    assert len(runner.build_grid(selected)) == 121


@pytest.mark.parametrize("flags", [
    ["--models","gpt2","gpt2"], ["--datasets","beauty","beauty"],
    ["--seeds","42","42"], ["--source-seeds","42","42"],
    ["--source-ids","gpt2_beauty","gpt2_beauty"], ["--arms","ce","ce"],
])
def test_duplicates_are_errors(flags):
    with pytest.raises(SystemExit):
        args(*flags)


def test_missing_explicit_source_stays_selected_and_never_falls_back(monkeypatch):
    source = "tiger_beauty_2023"
    cells = runner.build_grid(args("--source-ids",source))
    assert len(cells) == 121
    assert {cell["source_id"] for cell in cells} == {source}
    calls = []
    def missing(source_id, seed, **kwargs):
        calls.append((source_id,seed))
        raise FileNotFoundError("missing selected SFT-only checkpoint")
    monkeypatch.setattr(runner.protocol,"resolve_source",missing)
    with pytest.raises(FileNotFoundError, match="missing selected"):
        runner.main(["--source-ids",source,"--run-index","0"])
    assert calls == [(source,42)]


@pytest.mark.parametrize("flag", ["--count", "--list"])
def test_metadata_commands_work_with_site_packages_disabled(flag):
    result = subprocess.run([sys.executable,"-S",str(SCRIPT),flag], cwd=ROOT,
                            capture_output=True,text=True,timeout=15)
    assert result.returncode == 0,result.stderr
    if flag == "--count":
        assert result.stdout.strip() == "1089"
    else:
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(rows) == 1089
        assert [row["index"] for row in rows] == list(range(1089))


def test_smoke_paths_are_isolated():
    normal = args()
    smoke = args("--smoke")
    assert smoke.output_root == normal.output_root / "smoke"
    assert smoke.cache_root == normal.cache_root / "smoke"
