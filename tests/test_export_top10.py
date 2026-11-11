"""Top-10 exports retain raw IDs and reject incomplete or mixed artifacts."""
from pathlib import Path
import csv
import gzip
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/exp6_result_analysis"))
import export_top10 as export


def write_catalog(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "popularity"])
        writer.writerows(rows)


@pytest.fixture
def export_assets(tmp_path):
    prepared = tmp_path / "prepared/tiny_prep"
    catalog = [(f"{i:03d}", 0 if i == 0 else i) for i in range(12)]
    write_catalog(prepared / "catalog.csv", catalog)
    export.write_json(prepared / "metadata.json", {"num_users": 2, "total_test_users": 2,
        "fingerprint_inputs": {"max_users": None, "sample_seed": 42}})
    spec = {"run_id": "tiny_run", "preparation_id": "tiny_prep", "model": "tiger",
            "dataset": "beauty", "stage": "sft", "seed": 42,
            "checkpoint": "checkpoints/exp2-tuned_params/finetuned/tiger/beauty/sft_only_lr3e-05/seed_42/best.pt"}
    popularity = export.export_popularity(prepared, tmp_path / "popularity/tiny_prep.csv")
    cache = tmp_path / "cache/tiny_run.npz"
    cache.parent.mkdir()
    arrays = {"user_ids": np.asarray(["0007", "пользователь"]),
              "target_item_ids": np.asarray(["011", "000"]),
              "recommendations": np.asarray([[f"{i:03d}" for i in range(10)],
                                               [f"{i:03d}" for i in range(11, 1, -1)]])}
    np.savez_compressed(cache, **arrays)
    metadata = {"npz_sha256": export.sha256(cache), "catalog_sha256": export.sha256(prepared / "catalog.csv"),
                "num_users": 2, "total_test_users": 2,
                "fingerprint_inputs": {"prepared_metadata_sha256": export.sha256(prepared / "metadata.json")}}
    export.write_json(cache.with_suffix(".json"), metadata)
    manifest = {"runs": [spec], "missing_groups": [], "max_users": None}
    return tmp_path, prepared, spec, popularity, cache, arrays, manifest


def rewrite_cache(cache, arrays):
    np.savez_compressed(cache, **arrays)
    metadata = json.loads(cache.with_suffix(".json").read_text())
    metadata["npz_sha256"] = export.sha256(cache)
    export.write_json(cache.with_suffix(".json"), metadata)


def test_popularity_preserves_zero_counts_raw_ids_and_train_total(export_assets):
    _, prepared, _, popularity, _, _, _ = export_assets
    with popularity.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert rows[0] == {"item_id": "000", "train_count": "0", "train_fraction": "0.0"}
    assert rows[-1]["item_id"] == "011"
    assert sum(int(row["train_count"]) for row in rows) == 66
    assert sum(float(row["train_fraction"]) for row in rows) == pytest.approx(1)
    metadata = json.loads(popularity.with_suffix(".json").read_text())
    assert metadata["num_items"] == 12
    assert metadata["total_train_interactions"] == 66
    assert metadata["catalog_sha256"] == export.sha256(prepared / "catalog.csv")
    assert metadata["csv_sha256"] == export.sha256(popularity)


@pytest.mark.parametrize("rows", [[], [("a", -1)], [("a", 1), ("a", 2)]])
def test_invalid_popularity_catalog_is_rejected_without_export(tmp_path, rows):
    prepared = tmp_path / "prepared"
    write_catalog(prepared / "catalog.csv", rows)
    output = tmp_path / "popularity.csv"
    with pytest.raises(ValueError, match="Invalid training popularity catalog"):
        export.export_popularity(prepared, output)
    assert not output.exists()


def test_all_zero_popularity_has_finite_zero_fractions(tmp_path):
    prepared = tmp_path / "prepared"
    write_catalog(prepared / "catalog.csv", [("001", 0), ("002", 0)])
    export.write_json(prepared / "metadata.json", {})
    output = export.export_popularity(prepared, tmp_path / "popularity.csv")
    with output.open(newline="") as handle:
        assert [float(row["train_fraction"]) for row in csv.DictReader(handle)] == [0.0, 0.0]


def test_recommendation_export_keeps_every_example_rank_and_raw_id(export_assets):
    root, _, spec, popularity, cache, arrays, manifest = export_assets
    output = export.export_recommendations(cache, spec, root)
    with gzip.open(output, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["example_index", "user_id", "target_item_id", *export.RANK_COLUMNS]
        rows = list(reader)
    assert len(rows) == 2
    assert [row["example_index"] for row in rows] == ["0", "1"]
    assert [row["user_id"] for row in rows] == arrays["user_ids"].tolist()
    assert [row["target_item_id"] for row in rows] == ["011", "000"]
    assert [[row[column] for column in export.RANK_COLUMNS] for row in rows] == arrays["recommendations"].tolist()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["top_k"] == 10
    assert metadata["popularity_file"] == popularity.relative_to(root).as_posix()
    assert metadata["num_examples"] == metadata["total_test_examples"] == 2
    export.verify_exports(manifest, root)


@pytest.mark.parametrize("change,match", [
    ("duplicate", "Invalid top10"), ("outside_catalog", "Invalid top10"),
    ("short_ranking", "exactly 10"), ("duplicate_user", "missing or duplicated"),
    ("missing_example", "missing or duplicated"),
])
def test_invalid_recommendations_cannot_publish_an_output(export_assets, change, match):
    root, _, spec, _, cache, arrays, _ = export_assets
    if change == "duplicate":
        arrays["recommendations"][0, 9] = arrays["recommendations"][0, 0]
    elif change == "outside_catalog":
        arrays["recommendations"][0, 0] = "999"
    elif change == "short_ranking":
        arrays["recommendations"] = arrays["recommendations"][:, :9]
    elif change == "duplicate_user":
        arrays["user_ids"][1] = arrays["user_ids"][0]
    else:
        arrays = {key: value[:1] for key, value in arrays.items()}
    rewrite_cache(cache, arrays)
    with pytest.raises(ValueError, match=match):
        export.export_recommendations(cache, spec, root)
    assert not (root / "recommendations/tiny_run.csv.gz").exists()
    assert not list((root / "recommendations").glob("*.tmp"))


@pytest.mark.parametrize("change", ["cache", "popularity_csv", "catalog_hash", "prepared_hash"])
def test_mixed_cache_or_popularity_artifacts_are_rejected(export_assets, change):
    root, _, spec, popularity, cache, _, _ = export_assets
    if change == "cache":
        with cache.open("ab") as handle:
            handle.write(b"changed")
    elif change == "popularity_csv":
        with popularity.open("a") as handle:
            handle.write("999,1,0.5\n")
    else:
        sidecar = popularity.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        key = "catalog_sha256" if change == "catalog_hash" else "prepared_metadata_sha256"
        metadata[key] = "mismatched"
        export.write_json(sidecar, metadata)
    with pytest.raises(ValueError, match="cache changed|Popularity file does not match"):
        export.export_recommendations(cache, spec, root)


def test_manifest_groups_only_compatible_data_protocols(tmp_path):
    baseline = {"data": {"source": "yambda", "interactions_path": "data/events.parquet",
                         "semantic_id_path": "data/ids.json"},
                "dataset": {"max_history_len": 50, "offset_semantic_ids_by_one": False},
                "model": {"d_kv": 32, "use_user_token": False, "user_ids_count": 0}}
    configs = [baseline, {**baseline, "model": {**baseline["model"], "d_kv": 64}},
               {**baseline, "dataset": {**baseline["dataset"], "max_history_len": 20}},
               {**baseline, "dataset": {**baseline["dataset"], "offset_semantic_ids_by_one": True}}]
    runs = []
    for index, config in enumerate(configs):
        export.write_json(tmp_path / f"config_{index}.json", config)
        runs.append({"run_id": f"run_{index}", "dataset": "yambda_50m_listens", "model": "tiger",
                     "stage": "init", "checkpoint": f"checkpoints/exp2-tuned_params/run_{index}/best.pt",
                     "data_config": f"config_{index}.json"})
    manifest = export.freeze_manifest(runs, tmp_path, max_users=8, sample_seed=17)
    assert manifest["selection_split"] == "test"
    assert manifest["checkpoint_kinds"] == ["best"]
    assert manifest["evaluation_split"] == "test"
    assert len(manifest["preparations"]) == 3
    ids = [run["preparation_id"] for run in manifest["runs"]]
    assert ids[0] == ids[1]
    assert len(set(ids[1:])) == 3
    assert manifest["max_users"] == 8 and manifest["sample_seed"] == 17
    assert all(run["preserve_legacy_checkpoint_path"] for run in manifest["runs"])
    assert [run["checkpoint"] for run in manifest["runs"]] == [run["checkpoint"] for run in runs]
    assert all("preparation_id" not in run for run in runs)
    assert "beauty/gpt2/ce" in manifest["missing_groups"]
    assert "beauty/tiger/gflow" in manifest["missing_groups"]
    assert "beauty/tiger/sft2" in manifest["missing_groups"]


@pytest.mark.parametrize("change,match", [
    ("missing", "No such file"), ("manifest", "manifest/export metadata mismatch"),
    ("partial", "incomplete full-test export"), ("rows", "CSV example count mismatch"),
    ("popularity", "checksum mismatch"), ("missing_group", "No source runs for beauty/gpt2/ce"),
])
def test_verify_detects_missing_mismatched_and_partial_outputs(export_assets, change, match):
    root, _, spec, popularity, cache, _, manifest = export_assets
    output = export.export_recommendations(cache, spec, root)
    sidecar = output.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    if change == "missing":
        output.unlink()
    elif change == "manifest":
        manifest["runs"] = [{**spec, "seed": 43}]
    elif change == "partial":
        metadata["total_test_examples"] = 3
        export.write_json(sidecar, metadata)
    elif change == "rows":
        with gzip.open(output, "rt", encoding="utf-8") as handle:
            lines = handle.readlines()
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            handle.writelines(lines[:-1])
        metadata["csv_sha256"] = export.sha256(output)
        export.write_json(sidecar, metadata)
    elif change == "popularity":
        popularity.write_text("changed\n")
    else:
        manifest["missing_groups"] = ["beauty/gpt2/ce"]
    with pytest.raises(ValueError, match=match):
        export.verify_exports(manifest, root)


def test_verify_allows_an_explicit_smoke_subset(export_assets):
    root, prepared, spec, popularity, cache, _, manifest = export_assets
    prepared_meta = json.loads((prepared / "metadata.json").read_text())
    prepared_meta["fingerprint_inputs"]["max_users"] = 2
    prepared_meta["total_test_users"] = 20
    export.write_json(prepared / "metadata.json", prepared_meta)
    export.export_popularity(prepared, popularity)
    metadata = json.loads(cache.with_suffix(".json").read_text())
    metadata["total_test_users"] = 20
    metadata["fingerprint_inputs"]["prepared_metadata_sha256"] = export.sha256(prepared / "metadata.json")
    export.write_json(cache.with_suffix(".json"), metadata)
    manifest["max_users"] = 2
    export.export_recommendations(cache, spec, root)
    export.verify_exports(manifest, root)


@pytest.mark.parametrize("key,value", [("max_users", 1), ("sample_seed", 99)])
def test_verify_rejects_export_from_other_frozen_subset(export_assets, key, value):
    root, _, spec, _, cache, _, manifest = export_assets
    export.export_recommendations(cache, spec, root)
    manifest[key] = value
    with pytest.raises(ValueError, match="test subset differs"):
        export.verify_exports(manifest, root)


def test_verify_rejects_stale_preparation(export_assets):
    root, prepared, spec, _, cache, _, manifest = export_assets
    export.export_recommendations(cache, spec, root)
    export.write_json(prepared / "metadata.json", {"changed": True})
    with pytest.raises(ValueError, match="different test preparation"):
        export.verify_exports(manifest, root)



def test_old_from_base_manifest_requires_regeneration():
    with pytest.raises(ValueError, match="version-2 from-SFT.*regenerate"):
        export.validate_manifest({"schema_version": 1, "selection_split": "test"})


def test_discovery_options_separate_selection_from_evaluation():
    args = export.parse_args(["--mode", "discover"])
    assert args.selection_split == "test"
    assert args.checkpoint_kinds == ["best"]
    args = export.parse_args(["--mode", "discover", "--selection-split", "validation",
                              "--checkpoint-kinds", "best", "last"])
    assert args.selection_split == "validation"
    assert args.checkpoint_kinds == ["best", "last"]


@pytest.mark.parametrize("option", [["--sample-seed", "7"], ["--selection-split", "test"],
                                   ["--checkpoint-kinds", "last"]])
def test_frozen_options_are_not_silently_ignored(option):
    with pytest.raises(SystemExit):
        export.parse_args(["--mode", "generate", *option])


def test_preflight_rejects_missing_requested_family(tmp_path):
    with pytest.raises(FileNotFoundError, match="No source runs for beauty/tiger/gflow"):
        export.check_inputs({"runs": [], "preparations": [],
                             "missing_groups": ["beauty/tiger/gflow"]}, tmp_path)


def frozen_sft_manifest(root):
    from top10_manifest import BASE, SOURCE_IDS, first_checkpoint
    export.write_json(root / 'config.json', {
        'data': {'source': 'amazon', 'reviews_path': 'data/reviews.csv',
                 'metadata_path': 'data/items.csv', 'semantic_id_path': 'data/ids.json'},
        'dataset': {'max_history_len': 20}, 'model': {'use_user_token': False}})
    runs = [{'run_id': f'sft_seed_{seed}', 'model': 'tiger', 'dataset': 'beauty',
             'stage': 'sft', 'seed': seed, 'checkpoint': first_checkpoint('tiger', 'beauty', seed),
             'checkpoint_kind': 'best', 'source_id': SOURCE_IDS['tiger', 'beauty'],
             'init_checkpoint': BASE['tiger', 'beauty'], 'data_config': 'config.json'}
            for seed in (42, 43)]
    return export.freeze_manifest(runs, root)


def test_manifest_retains_all_sft_seeds_and_tracks_missing_stages(tmp_path):
    frozen = frozen_sft_manifest(tmp_path)
    export.validate_manifest(frozen)
    assert frozen['selection_split'] == 'test'
    assert frozen['checkpoint_kinds'] == ['best']
    assert [r['seed'] for r in frozen['runs']] == [42, 43]
    assert 'beauty/tiger/sft' not in frozen['missing_groups']
    assert 'beauty/tiger/sft2' in frozen['missing_groups']
    assert len(frozen['preparations']) == 1


@pytest.mark.parametrize('change,match', [
    ('missing_groups', 'completeness'), ('path_migration', 'paths must be preserved'),
    ('wrong_kind', 'checkpoint kind'), ('wrong_preparation', 'Preparation must reference')])
def test_manifest_rejects_scope_and_preparation_drift(tmp_path, change, match):
    frozen = frozen_sft_manifest(tmp_path)
    if change == 'missing_groups':
        frozen['missing_groups'] = []
    elif change == 'path_migration':
        frozen['runs'][0]['preserve_legacy_checkpoint_path'] = False
    elif change == 'wrong_kind':
        frozen['checkpoint_kinds'] = ['last']
    else:
        frozen['preparations'][0]['spec'] = dict(frozen['runs'][0], seed=99)
    with pytest.raises(ValueError, match=match):
        export.validate_manifest(frozen)
