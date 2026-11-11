"""Selected GPT-2 Yambda CE sources, frozen subset, and cache provenance."""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py"
spec = importlib.util.spec_from_file_location("ce_gpt2_yambda_profile", SCRIPT)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


@pytest.fixture(scope="module")
def runtime():
    return ce.load_runtime()


def test_requested_grid_keeps_reward_power_rho_order_and_isolates_models():
    args = ce.parse_args([])
    assert args.datasets == list(ce.DATASETS)
    assert args.combos == ["sasrec", "interaction", "similarity", ce.DEFAULT_COMBO]
    assert args.powers == [1., .25, 16.] and args.rho_targets == [1., .1]
    assert args.num_samples == 3 and args.seeds == [42]
    assert (args.train_random_n, args.train_subset_seed) == (15, 42)
    assert args.sampling == "reward" and args.curriculum_epochs == 5.
    assert args.checkpoint_every_updates == 10 and args.resume
    grid = ce.build_grid(args)
    expected = itertools.product(ce.DATASETS, ce.COMBOS, [1., .25, 16.], [1., .1], [42])
    assert [(row["dataset"], ce.combo_key(row), row["power"], row["rho"], row["seed"])
            for row in grid] == list(expected)
    assert len(grid) == 48
    assert args.output_root == ROOT / ce.DEFAULT_OUTPUT_ROOT
    ordinary = {ce.run_dir_for(args, row) for row in grid}
    args.sampling = "curriculum_smooth_is"
    corrected = {ce.run_dir_for(args, row) for row in grid}
    assert not ordinary & corrected
    assert all("qmix0p1" in str(path) for path in corrected)


@pytest.mark.parametrize("sampling", ["reward", "curriculum_smooth_is"])
@pytest.mark.parametrize("flag", ["--count", "--list", "--aggregate-only"])
@pytest.mark.parametrize("target_count", [15, 50])
def test_metadata_actions_need_only_the_standard_library(flag, sampling, target_count, tmp_path):
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), flag, "--sampling", sampling,
                             "--train-random-n", str(target_count),
                             "--output-root", str(tmp_path)], check=True,
                            cwd=tmp_path, capture_output=True, text=True)
    if flag == "--count":
        assert result.stdout.strip() == "48"
    elif flag == "--list":
        assert "total: 48 runs" in result.stdout
        assert "trial=36 targets=last128" in result.stdout
        assert f"trial=4 targets=sampled{target_count}" in result.stdout
    else:
        assert json.loads(result.stdout)["completed"] == 0


def test_large_catalog_exact_ce_is_rejected():
    with pytest.raises(SystemExit, match="reserved for small catalogs"):
        ce.parse_args(["--estimator", "exact"])


@pytest.mark.parametrize("count", [0, -1, 128, 129])
def test_full_or_invalid_training_subsets_are_rejected(count):
    with pytest.raises(SystemExit, match="1..127"):
        ce.parse_args(["--train-random-n", str(count)])


def test_negative_subset_seed_is_rejected():
    with pytest.raises(SystemExit, match="nonnegative"):
        ce.parse_args(["--train-subset-seed", "-1"])


@pytest.mark.parametrize("changes", [{"train_random_n": 0}, {"train_random_n": 128},
    {"ce_target_selection": "last128"}, {"train_pool_last_n": None},
    {"train_subset_seed": -1}, {"train_last_n": 8}])
def test_runtime_rejects_bypassing_the_frozen_random_subset(changes, runtime):
    matched, raw = runtime
    b = ce.matched_args(ce.parse_args([]), ce.DATASETS[1], 42, matched)
    for key, value in changes.items():
        setattr(b, key, value)
    base = ce.runtime_for(raw, b)
    with pytest.raises(RuntimeError):
        base.finetune_indices(None, b)
    with pytest.raises(RuntimeError):
        ce.build_run_config(base, ce.DATASETS[1], b)


def test_sampled15_50_and_seed_namespaces_keep_legacy_runs_and_50m_unchanged(runtime, tmp_path):
    import pandas as pd
    matched, raw = runtime
    dataset = ce.DATASETS[1]
    examples = SimpleNamespace(examples=pd.DataFrame({"user_id": [0] * 128 + [1] * 7,
                                                     "target_timestamp": list(range(128)) + list(range(7))}))
    entry = {"dataset": dataset, "combo": "sasrec", "power": 1., "rho": 1., "seed": 42}
    records, original_50m = [], []
    for count, subset_seed in ((15, 42), (50, 42), (15, 7)):
        args = ce.parse_args(["--train-random-n", str(count), "--train-subset-seed", str(subset_seed),
                              "--output-root", str(tmp_path / "runs"), "--cache-root", str(tmp_path / "factors")])
        b = ce.matched_args(args, dataset, 43, matched)
        base = ce.runtime_for(raw, b)
        config = ce.build_run_config(base, dataset, b)
        indices = base.finetune_indices(examples, b)
        assert len(indices) == count + 7 and indices == sorted(set(indices))
        assert examples.examples.iloc[indices].groupby("user_id").size().tolist() == [count, 7]
        assert base.finetune_indices(examples, b) == indices
        other_model_seed = ce.matched_args(args, dataset, 99, matched)
        assert base.finetune_indices(examples, other_model_seed) == indices
        namespace = "" if (count, subset_seed) == (15, 42) else f"sampled{count}_seed{subset_seed}"
        root = tmp_path / "runs" / dataset / namespace
        assert ce.run_dir_for(args, entry) == root / "sasrec/sampled_n3/pow_1/rho_1/seed_42"
        assert ce.shared.dataset_output_root(args, dataset) == root
        assert b.cache_root == tmp_path / "factors" / namespace
        assert ce.component_cache_dir(base, dataset, "interaction", b) == b.cache_root / dataset / "reward_factors/interaction"
        assert ce.teacher_translation_path(dataset, b) == b.cache_root / dataset / "source_assets/teacher_id_translation.json"
        identity = ce.build_identity(args, entry, config, {}, {}, 100)
        records.append((config, ce.run_dir_for(args, entry), b.cache_root, identity, ce.shared.summary_dir_for(args)))
        b50 = ce.matched_args(args, ce.DATASETS[0], 43, matched)
        config50 = ce.build_run_config(ce.runtime_for(raw, b50), ce.DATASETS[0], b50)
        original_50m.append((config50, b50.cache_root, ce.run_dir_for(args, {**entry, "dataset": ce.DATASETS[0]})))
    assert len({r[1] for r in records}) == len({r[2] for r in records}) == len({r[4] for r in records}) == 3
    assert records[0][0]["dataset"] == records[1][0]["dataset"] == records[2][0]["dataset"]
    assert records[0][0]["initial_model"] == records[1][0]["initial_model"]
    assert records[0][3] != records[1][3] != records[2][3]
    assert original_50m[0] == original_50m[1] == original_50m[2]
    identity_path = tmp_path / "legacy_identity.json"
    ce.write_json(identity_path, records[0][3])
    with pytest.raises(RuntimeError, match="Protocol/assets changed"):
        ce.shared.check_identity(identity_path, records[1][3])


@pytest.mark.parametrize("dataset", ce.DATASETS)
@pytest.mark.parametrize("target_count", [15, 50])
def test_selected_sources_batches_and_target_subsets_are_pinned(dataset, target_count, runtime):
    matched, raw = runtime
    args = ce.parse_args(["--device", "cpu", "--train-random-n", str(target_count)])
    b = ce.matched_args(args, dataset, 43, matched)
    base = ce.runtime_for(raw, b)
    assert b.datasets == [dataset] and b.seed == 43
    assert not b.prefix_from_dense and b.cache_format == "factorized"
    assert b.train_last_n == 0 and b.train_subset_seed == 42
    namespace = f"sampled{target_count}_seed42" if dataset == ce.DATASETS[1] and target_count != 15 else ""
    assert b.cache_root == ROOT / ce.DEFAULT_CACHE_ROOT / namespace
    assert b.eval_num_beams == 20 and b.test_ks == [5, 10, 20]
    assert base.tuned_checkpoint_paths(dataset)["checkpoint"] == (
        ROOT / "checkpoints/exp2/loo_split/initial/gpt2" / dataset / "last_128/best.pt")
    if dataset == ce.DATASETS[0]:
        assert (b.batch_size, b.grad_accum, b.test_batch_size) == (128, 16, 64)
        assert (b.train_random_n, b.train_pool_last_n) == (0, None)
        assert b.ce_target_selection == "last128"
        assert base.finetune_subset_spec(b) == {
            "per_user_last_n": None, "per_user_random_n": None,
            "per_user_random_seed": None, "per_user_pool_last_n": None}
        assert base.finetune_indices(SimpleNamespace(), b) is None
    else:
        assert (b.batch_size, b.grad_accum, b.test_batch_size) == (256, 8, 128)
        assert (b.train_random_n, b.train_pool_last_n) == (target_count, 128)
        assert b.ce_target_selection == f"sampled{target_count}"
        assert base.finetune_subset_spec(b) == {
            "per_user_last_n": None, "per_user_random_n": target_count,
            "per_user_random_seed": 42, "per_user_pool_last_n": 128}
    overrides = ce.trainer_overrides(args, b)
    assert overrides == {"amp_dtype": "bfloat16", "val_sample_seed": 0,
                         "val_sample_resample": False,
                         "val_sample_fraction": .25 if dataset == ce.DATASETS[1] else None}


@pytest.mark.parametrize("dataset", ce.DATASETS)
@pytest.mark.parametrize("target_count", [15, 50])
def test_smoke_and_explicit_batches_are_isolated(dataset, target_count, runtime, tmp_path):
    matched, base = runtime
    args = ce.parse_args(["--output-root", str(tmp_path / "runs"), "--cache-root", str(tmp_path / "prod"),
                         "--batch-size", "2", "--grad-accum", "1", "--smoke",
                         "--train-random-n", str(target_count)])
    b = ce.matched_args(args, dataset, 42, matched)
    assert args.output_root == tmp_path / "runs_smoke"
    namespace = f"sampled{target_count}_seed42" if dataset == ce.DATASETS[1] and target_count != 15 else ""
    assert b.cache_root == args.output_root / "cache" / namespace
    assert ce.component_cache_dir(base, dataset, "similarity", b) == (
        b.cache_root / dataset / "reward_factors/sim_hier")
    assert (b.batch_size, b.grad_accum, base._smoke_examples(b)) == (2, 1, 16)
    assert ce.teacher_translation_path(dataset, b) == b.cache_root / dataset / "source_assets/teacher_id_translation.json"


def test_runtime_binds_all_model_helpers_to_the_selected_namespace():
    b = SimpleNamespace()
    calls = []
    original = SimpleNamespace(
        tuned_profile=lambda ds, args: calls.append(("profile", ds, args)) or ({"model": {}}, {}),
        tuned_checkpoint_paths=lambda ds, args: calls.append(("paths", ds, args)) or {},
        build_gpt2_from_tuned_checkpoint=lambda ds, dm, args: calls.append(("model", ds, dm, args)) or ("model", {}),
    )
    base = ce.runtime_for(original, b)
    base.tuned_profile("yambda_50m_listens")
    base.tuned_profile("yambda_50m_listens")
    base.tuned_checkpoint_paths("yambda_50m_listens")
    base.build_gpt2_from_tuned_checkpoint("yambda_50m_listens", "data")
    assert calls == [("profile", "yambda_50m_listens", b), ("paths", "yambda_50m_listens", b),
                     ("model", "yambda_50m_listens", "data", b)]


def test_new_identity_records_source_and_target_protocol(tmp_path):
    args = ce.parse_args(["--output-root", str(tmp_path), "--sampling", "curriculum_smooth_is"])
    entry = ce.build_grid(args)[0]
    config = {"ce_data_protocol": {"source_trial": 36, "target_selection": "last128"}}
    identity = ce.build_identity(args, entry, config, {}, {}, 100)
    assert identity["model"] == "gpt2" and identity["source_trial"] == 36
    assert identity["cache_context_version"] == 2
    assert identity["ce_data_protocol"] == config["ce_data_protocol"]
    assert identity["ce_target"] == "q"
    assert identity["curriculum"]["importance_correction"]


def test_manifest_requires_teacher_translation_and_frozen_source(tmp_path, monkeypatch):
    dataset = ce.DATASETS[1]
    b = SimpleNamespace(datasets=[dataset], cache_root=tmp_path)
    source = tmp_path / "source.json"
    path = tmp_path / "sasrec/manifest.json"
    path.parent.mkdir()
    (path.parent / "factors.pt").touch()
    config = {"data": {"interactions_path": "/data/original.parquet"},
              "dataset": {"matched_tuning_data_audit": {"catalog_sha256": "catalog"}},
              "initial_model": {"trial": 4},
              "method": {"sasrec": {"checkpoint_path": "teacher.pt"}}}
    raw = SimpleNamespace(sasrec_method_cfg=lambda ds: {"checkpoint_path": "teacher.pt"},
                          cache_config=lambda cfg, key: cfg)
    wanted = json.loads(json.dumps(config))
    wanted["method"]["sasrec"]["item_id_translation_path"] = str(ce.teacher_translation_path(dataset, b))
    ce.write_json(source, wanted)
    manifest = {"config_path": str(source), "teacher_id_translation": False,
                "split": "train", "num_examples": 10}
    monkeypatch.setattr(ce.gpt2, "check_component_manifest", lambda *a, **kw: dict(manifest))
    with pytest.raises(RuntimeError, match="teacher item ID translation"):
        ce.check_component_manifest(raw, config, path, "sasrec", 10, b)
    manifest["teacher_id_translation"] = True
    assert ce.check_component_manifest(raw, config, path, "sasrec", 10, b) == manifest
    changed = json.loads(json.dumps(wanted))
    changed["initial_model"]["trial"] = 36
    ce.write_json(source, changed)
    with pytest.raises(RuntimeError, match="source initial_model"):
        ce.check_component_manifest(raw, config, path, "sasrec", 10, b)
    changed = json.loads(json.dumps(wanted))
    changed["data"]["interactions_path"] = "/data/other.parquet"
    ce.write_json(source, changed)
    with pytest.raises(RuntimeError, match="source data"):
        ce.check_component_manifest(raw, config, path, "sasrec", 10, b)
    ce.write_json(source, wanted)
    with pytest.raises(RuntimeError, match="cache row count"):
        ce.check_component_manifest(raw, config, path, "sasrec", 9, b)
    manifest["split"] = "val"
    with pytest.raises(RuntimeError, match="training reward cache"):
        ce.check_component_manifest(raw, config, path, "sasrec", 10, b)


def test_asset_identity_includes_trial_provenance_data_and_teacher_translation(tmp_path):
    paths = {key: tmp_path / name for key, name in (
        ("checkpoint", "best.pt"), ("resolved_config", "resolved_config.json"),
        ("sidecar", "best.pt.json"), ("data_audit", "data_audit.json"))}
    paths["sidecar"].touch()
    paths["data_audit"].touch()
    teacher = tmp_path / "sasrec/model.pt"
    base = SimpleNamespace(
        tuned_checkpoint_paths=lambda ds: paths, EXP1_GPT2_MODEL_YAML=tmp_path / "model.yaml",
        dataset_yaml_path=lambda ds: tmp_path / "dataset.yaml",
        abspath=lambda p: Path(p), DATASETS={"ds": {"sasrec_checkpoint": teacher,
                                                   "sasrec_trial": tmp_path / "sasrec/best_trial.json"}},
        rqvae_model_path=lambda ds: tmp_path / "rqvae.pt")
    config = {"data": {"source": "yambda", "interactions_path": str(tmp_path / "events.parquet"),
                       "embeddings_path": str(tmp_path / "embeddings.parquet"),
                       "semantic_id_path": str(tmp_path / "semantic.json")},
              "method": {"sasrec": {"item_id_translation_path": str(tmp_path / "translation.json")}}}
    all_assets = ce.asset_paths_for(base, "ds", config, ce.COMPONENTS, [tmp_path / "factors/manifest.json"])
    assert paths["resolved_config"] in all_assets and paths["checkpoint"] in all_assets
    assert tmp_path / "translation.json" in all_assets and teacher.parent / "training_summary.json" in all_assets
    assert tmp_path / "events.parquet" in all_assets and tmp_path / "embeddings.parquet" in all_assets
    interaction = ce.asset_paths_for(base, "ds", config, ["interaction"], [])
    assert teacher not in interaction and tmp_path / "translation.json" not in interaction
    assert tmp_path / "rqvae.pt" not in interaction
