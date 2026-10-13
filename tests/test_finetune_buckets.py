"""Check stage integration and reject stale/mixed experiment artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/exp6_result_analysis"))
from finetune_buckets import analyze_dataset, load_runs, main, parse_args, sha256, write_json


def test_discover_retains_missing_models_and_infers_dataset_on_later_stages(tmp_path):
    assert main(["--mode", "discover", "--repo-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                 "--datasets", "beauty"]) == 0
    args = parse_args(["--mode", "prepare", "--output-dir", str(tmp_path / "out")])
    specs = load_runs(args)
    assert args.datasets == ["beauty"]
    assert len(specs) == 4
    assert all(not s["available"] for s in specs)
    assert specs[2]["selection_labels"] == ["best_ndcg", "pareto"]


@pytest.fixture
def artifacts(tmp_path):
    folder = tmp_path / "beauty"
    prepared, cache = folder / "prepared", folder / "cache"
    prepared.mkdir(parents=True)
    cache.mkdir()
    pd.DataFrame({"item_id": list("abc"), "popularity": [10, 2, 0]}).to_csv(prepared / "catalog.csv", index=False)
    write_json(prepared / "metadata.json", {"fingerprint": "frozen-split"})
    specs = []
    arrays = dict(user_ids=np.array(["u1", "u2"]), target_item_ids=np.array(["a", "b"]),
                  history_lengths=np.array([2, 3]), full_history_lengths=np.array([2, 30]),
                  history_fingerprints=np.array(["context1", "context2"]))
    for stage, recs in (("init", [["a", "c"], ["a", "b"]]),
                        ("sft", [["a", "c"], ["b", "c"]]),
                        ("gflow", [["a", "b"], ["b", "a"]])):
        spec = dict(run_id=stage, dataset="beauty", model="gpt2", stage=stage,
                    seed=None if stage == "init" else 42, checkpoint=f"{stage}.pt",
                    init_checkpoint="init.pt", data_config="data.json", baseline_run_id="init",
                    sft_run_id="sft" if stage != "init" else None,
                    selection_labels=[stage], selection_cells=["B1"])
        specs.append(spec)
        output = cache / f"{stage}.npz"
        np.savez_compressed(output, **arrays, recommendations=np.array(recs))
        source = {**spec, "npz_sha256": sha256(output), "catalog_sha256": sha256(prepared / "catalog.csv"),
                  "num_users": 2, "total_test_users": 2,
                  "fingerprint_inputs": {"prepared_fingerprint": "frozen-split",
                      "prepared_metadata_sha256": sha256(prepared / "metadata.json"),
                      "num_beams": 2, "top_k": 2, "constrain_valid_sequences": True,
                      "exclude_seen_items": False, "dtype": "float32"}}
        write_json(output.with_suffix(".json"), source)
    args = parse_args(["--mode", "analyze", "--output-dir", str(tmp_path), "--ks", "1", "2", "--no-plots"])
    return specs, folder, args


def test_cached_analysis_exports_paired_tables(artifacts):
    specs, folder, args = artifacts
    analyze_dataset(specs, folder, args)
    comparisons = pd.read_csv(folder / "tables/comparisons.csv")
    selected = comparisons.query("run_id == 'gflow' and cohort_type == 'overall' and k == 1")
    assert set(selected.reference_stage) == {"init", "sft"}
    assert selected.set_index("reference_stage").loc["init", "delta_hr"] == .5
    attribution = pd.read_csv(folder / "tables/accuracy_attribution.csv")
    contributions = attribution.query("run_id == 'gflow' and reference_stage == 'init' and k == 1 and cohort_type == 'history_length'")
    assert contributions.delta_ndcg_contribution.sum() == .5
    assert (folder / "tables/item_deltas.csv.gz").is_file()
    assert json.loads((folder / "analysis_metadata.json").read_text())["subsample"] is False


def test_analysis_rejects_changed_preparation(artifacts):
    specs, folder, args = artifacts
    write_json(folder / "prepared/metadata.json", {"fingerprint": "another-split"})
    with pytest.raises(ValueError, match="different prepared test split"):
        analyze_dataset(specs, folder, args)


@pytest.mark.parametrize("mutation, expected", [("model", "manifest/cache mismatch"),
                                               ("beams", "Generation settings differ"),
                                               ("contents", "cache contents/catalog changed")])
def test_analysis_rejects_mixed_or_modified_cache(artifacts, mutation, expected):
    specs, folder, args = artifacts
    sidecar = folder / "cache/gflow.json"
    source = json.loads(sidecar.read_text())
    if mutation == "model":
        source["model"] = "tiger"
    elif mutation == "beams":
        source["fingerprint_inputs"]["num_beams"] = 3
    else:
        with (folder / "cache/gflow.npz").open("ab") as handle:
            handle.write(b"changed")
    write_json(sidecar, source)
    with pytest.raises(ValueError, match=expected):
        analyze_dataset(specs, folder, args)
