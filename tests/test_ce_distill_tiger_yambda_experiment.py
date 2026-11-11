"""Yambda CE retains full contexts while sampling 500M training targets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp6-ce-distillation/ce_distill_tiger_yambda.py"
spec = importlib.util.spec_from_file_location("ce_tiger_yambda_experiment", SCRIPT)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


@pytest.fixture
def runtime(monkeypatch):
    matched, proxy = ce.load_runtime()
    base = proxy._base
    for key in {"DATASET", *(key for block in base.DATASETS.values() for key in block)}:
        monkeypatch.setattr(base, key, getattr(base, key))
    return matched, proxy, base


def test_default_grid_is_sampled50_for_500m_smooth_n3_five_epochs():
    args = ce.parse_args([])
    assert args.datasets == list(ce.DATASETS)
    assert args.combos == list(ce.COMBOS)
    assert args.powers == [.25, 1., 4., 16.] and args.rho_targets == [1., .1]
    assert args.seeds == [42]
    assert (args.estimator, args.sampling, args.num_samples, args.curriculum_epochs) == (
        "sampled", "curriculum_smooth", 3, 5.)
    assert args.checkpoint_every_updates == 10 and args.resume
    assert (args.train_random_n, args.train_subset_seed) == (50, 42)
    grid = ce.build_grid(args)
    assert len(grid) == len({ce.run_dir_for(args, row) for row in grid}) == 112
    assert all("curriculum_smooth_n3_epochs5" in str(ce.run_dir_for(args, row)) for row in grid)
    assert args.output_root == ROOT / "experiments/exp7-ce-distillation-tiger-yambda"
    for entry in grid:
        assert ("sampled50_seed42" in str(ce.run_dir_for(args, entry))) == (entry["dataset"] == ce.DATASETS[1])


@pytest.mark.parametrize("sampling", ["reward", "curriculum_smooth_is"])
def test_500m_compact_tiger_grid_matches_gpt2_python_protocol(sampling):
    """Compare the underlying grids without depending on a cluster launcher."""
    gpt_spec = importlib.util.spec_from_file_location(
        "ce_gpt2_yambda_compact_grid", ROOT / "scripts/exp6-ce-distillation/ce_distill_gpt2_yambda.py"
    )
    gpt = importlib.util.module_from_spec(gpt_spec)
    gpt_spec.loader.exec_module(gpt)
    reference = gpt.parse_args(["--datasets", "yambda_500m_listens"])
    tiger = ce.parse_args([
        "--datasets", "yambda_500m_listens",
        "--combos", "sasrec", "interaction", "similarity", "sasrec_x_interaction_x_similarity",
        "--powers", "1", ".25", "16", "--rho-targets", "1", ".1", "--seeds", "42",
        "--num-samples", "3", "--train-random-n", "50", "--train-subset-seed", "42",
        "--curriculum-epochs", "5", "--sampling", sampling, "--estimator", "sampled",
    ])
    assert len(gpt.build_grid(reference)) == 24
    assert ce.build_grid(tiger) == gpt.build_grid(reference)
    assert tiger.sampling == sampling
    assert tiger.num_samples == reference.num_samples == 3
    assert tiger.curriculum_epochs == reference.curriculum_epochs == 5


def test_50m_python_grid_keeps_all_56_default_cells():
    args = ce.parse_args(["--datasets", "yambda_50m_listens"])
    grid = ce.build_grid(args)
    assert len(grid) == 56
    assert {entry["dataset"] for entry in grid} == {"yambda_50m_listens"}
    assert {entry["power"] for entry in grid} == {.25, 1, 4, 16}


@pytest.mark.parametrize("dataset,expected", [(None, 112), *[(name, 56) for name in ce.DATASETS]])
@pytest.mark.parametrize("flag", ["--count", "--list", "--aggregate-only"])
@pytest.mark.parametrize("sampling", ["reward", "curriculum_smooth", "curriculum_smooth_is"])
def test_metadata_commands_do_not_import_ml_or_construct_yambda(dataset, expected, flag, sampling, tmp_path):
    command = [sys.executable, "-S", str(SCRIPT), flag, "--sampling", sampling, "--output-root", str(tmp_path)]
    if dataset:
        command += ["--datasets", dataset]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    if flag == "--count":
        assert result.stdout.strip() == str(expected)
    elif flag == "--list":
        assert f"total: {expected} runs" in result.stdout
    else:
        summary = json.loads(result.stdout)
        assert summary["expected"] == expected and summary["completed"] == 0


@pytest.mark.parametrize("dataset,fraction", [(ce.DATASETS[0], None), (ce.DATASETS[1], .25)])
def test_matched_assets_full_layout_and_fixed_validation(dataset, fraction, runtime, tmp_path):
    import pandas as pd
    import torch
    from flowrec.training import Trainer
    matched, proxy, base = runtime
    args = ce.parse_args(["--cache-root", str(tmp_path)])
    b = ce.matched_args(args, dataset, 42, matched)
    config = proxy.build_run_config(b)
    source = ce.read_json(base.abspath(base.SFT_CONFIG))
    assert config["model"] == source["model"]
    assert b.train_last_n == 0
    assert b.train_random_n == (50 if dataset == ce.DATASETS[1] else 0)
    synthetic = SimpleNamespace(examples=pd.DataFrame({"user_id": [0] * 128 + [1] * 7}))
    indices = proxy.finetune_indices(synthetic, b)
    if dataset == ce.DATASETS[1]:
        assert len(indices) == 57 and indices == sorted(indices)
        assert synthetic.examples.iloc[indices].groupby("user_id").size().tolist() == [50, 7]
        assert indices == proxy.finetune_indices(synthetic, b)
    else:
        assert indices is None
    assert (b.batch_size, b.grad_accum, b.val_batch_size) == (256, 8, 256)
    assert (b.eval_num_beams, b.test_ks) == (20, [5, 10, 20])
    layout = "sampled50_seed42" if dataset == ce.DATASETS[1] else "last128"
    assert b.cache_root == tmp_path / dataset / layout / "reward_factors"
    ce.assert_last128_config(config)
    assert config["dataset"]["train_all_prefixes"] == source["dataset"]["train_all_prefixes"]
    assert not config["dataset"].get("train_prefix_from_dense", False)
    assert config["ce_data_protocol"]["additional_per_user_random_n"] == (50 if dataset == ce.DATASETS[1] else None)
    assert config["ce_data_protocol"]["per_user_base_last_n"] == 128
    expected_teacher = "sasrec_sce" if dataset == ce.DATASETS[1] else "sasrec"
    assert f"initial/{expected_teacher}/{dataset}" in base.SASREC_CHECKPOINT
    cfg = ce.shared.build_trainer_config(args, b, tmp_path / "run", torch.device("cpu"))
    assert cfg.val_sample_fraction == fraction
    assert cfg.val_sample_seed == 0 and not cfg.val_sample_resample
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(100)), batch_size=10)
    trainer = SimpleNamespace(cfg=cfg, state=SimpleNamespace(epoch=0, global_step=0))
    first, _ = Trainer._maybe_sample_validation_loader(trainer, loader, stage="val")
    trainer.state.epoch, trainer.state.global_step = 9, 10000
    later, _ = Trainer._maybe_sample_validation_loader(trainer, loader, stage="val")
    if fraction is None:
        assert first is later is loader
    else:
        assert len(first.dataset) == 25 and first.dataset.indices == later.dataset.indices
    final_test, _ = Trainer._maybe_sample_validation_loader(trainer, loader, stage="test")
    assert final_test is loader


@pytest.mark.parametrize("dataset", ce.DATASETS)
def test_synthetic_yambda_matches_exp2_targets_catalog_and_retains_auxiliary(dataset, runtime, tmp_path):
    import pandas as pd
    matched, proxy, base = runtime
    args = ce.parse_args(["--num-workers", "0", "--cache-root", str(tmp_path / "cache")])
    b = ce.matched_args(args, dataset, 42, matched)
    config = proxy.build_run_config(b)
    source = ce.read_json(base.abspath(base.SFT_CONFIG))
    events = [dict(uid=user, item_id=pos % 31, timestamp=1000 + pos * 10,
                   event_type="listen", played_ratio_pct=90., is_organic=True)
              for user in (1, 2) for pos in range(220)]
    events += [dict(uid=user, item_id=7, timestamp=1501 + i,
                    event_type=kind, played_ratio_pct=20. if kind == "listen" else None, is_organic=False)
               for user in (1, 2) for i, kind in enumerate(("like", "dislike", "listen"))]
    interactions = tmp_path / "multi_event.parquet"
    pd.DataFrame(events).to_parquet(interactions)
    sid_path = tmp_path / "semantic_ids.json"
    sid_path.write_text(json.dumps({str(i): [i % 16, 256 + i // 16, 512, 768] for i in range(31)}))
    embeddings = tmp_path / "embeddings.parquet"
    pd.DataFrame({"item_id": range(31), "embed": [[1., 0.]] * 31}).to_parquet(embeddings)
    for run_config in (config, source):
        run_config["data"].update(interactions_path=str(interactions), semantic_id_path=str(sid_path),
                                  embeddings_path=str(embeddings) if dataset == ce.DATASETS[1] else None)
        run_config["dataset"]["hf_recent_months"] = None
        run_config["dataloader"].update(num_workers=0, persistent_workers=False)
    dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=tmp_path / "ce.json"))
    reference = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(source, config_path=tmp_path / "base.json"))
    dm.setup()
    reference.setup()
    assert len(dm.train_dataset) == 256
    assert dm.train_dataset.examples.groupby("user_id").size().tolist() == [128, 128]
    assert dm.train_dataset.examples.history_item_ids.map(len).max() == 50
    for split in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(dm, split + "_dataset").examples,
                                      getattr(reference, split + "_dataset").examples)
    assert dm.item2id == reference.item2id and dm.semantic_ids == reference.semantic_ids
    indices = proxy.finetune_indices(dm.train_dataset, b)
    if dataset == ce.DATASETS[1]:
        assert len(indices) == 100
        selected = dm.train_dataset.examples.iloc[indices]
        assert selected.groupby("user_id").size().tolist() == [50, 50]
        assert selected.history_item_ids.map(len).max() == 50
    else:
        assert indices is None
    assert len(dm.interaction_store) == 6
    if dataset == ce.DATASETS[0]:
        assert len(reference.interaction_store) == 0


def _write_factor_metadata(base, b, config, root, name):
    from dataclasses import asdict
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    source = root / ("source_" + name + ".json")
    ce.write_json(source, base.cache_config(config, name, b))
    data_config = asdict(base.tiger_runner.build_data_config(config, config_path=ROOT / "configs/test.json"))
    data_config = json.loads(json.dumps(data_config, default=str))
    manifest = {"kind": "factorized", "split": "train", "num_examples": 120 if b.train_random_n else 1000,
        "num_catalog_items": 31, "data": {"num_train_examples": 1000, "num_items": 31},
        **base.subset_spec(b),
        "data_config": data_config, "config_path": str(source),
        "component": {"name": name, "mode": "hierarchical" if name == "similarity" else None}}
    ce.write_json(path / "manifest.json", manifest)
    (path / "factors.pt").write_bytes(b"metadata test placeholder")
    return manifest


@pytest.mark.parametrize("count", [15, 50])
def test_reuse_requires_all_three_compatible_components_and_never_depends_on_combo(count, runtime, monkeypatch, tmp_path):
    matched, proxy, base = runtime
    dataset = ce.DATASETS[1]
    args = ce.parse_args(["--train-random-n", str(count)])
    b = ce.matched_args(args, dataset, 42, matched)
    config = proxy.build_run_config(b)
    legacy = tmp_path / f"legacy_sampled{count}"
    monkeypatch.setitem(ce.LEGACY_500M_CACHE_ROOTS, (count, 42), str(legacy))
    fallback = ROOT / ce.DEFAULT_CACHE_ROOT / dataset / f"sampled{count}_seed42/reward_factors"
    for name in ce.COMPONENTS:
        _write_factor_metadata(base, b, config, legacy, name)
    assert ce.select_cache_root(args, dataset, b, base) == legacy
    args.combos = ["sasrec"]
    assert ce.select_cache_root(args, dataset, b, base) == legacy
    manifest = ce.read_json(legacy / "interaction/manifest.json")
    manifest["per_user_random_n"], manifest["per_user_random_seed"] = None, None
    ce.write_json(legacy / "interaction/manifest.json", manifest)
    assert ce.select_cache_root(args, dataset, b, base) == fallback
    with pytest.raises(RuntimeError, match="random"):
        ce.check_cache_metadata(base, config, "interaction", legacy / "interaction", b)


def test_subset_count_seed_paths_and_identity_change_without_affecting_50m(runtime, tmp_path):
    import pandas as pd
    matched, proxy, base = runtime
    dataset = SimpleNamespace(examples=pd.DataFrame({"user_id": [0] * 128 + [1] * 128}))
    records = []
    unchanged = []
    for count, subset_seed in ((50, 42), (15, 42), (50, 7)):
        args = ce.parse_args(["--train-random-n", str(count), "--train-subset-seed", str(subset_seed),
                              "--cache-root", str(tmp_path / "caches")])
        b = ce.matched_args(args, ce.DATASETS[1], 99, matched)
        config = proxy.build_run_config(b)
        indices = proxy.finetune_indices(dataset, b)
        assert len(indices) == count * 2
        assert indices == sorted(indices) and len(set(indices)) == len(indices)
        assert b.train_subset_seed == subset_seed and b.seed == 99
        entry = {"dataset": ce.DATASETS[1], "power": 1., "rho": 1., "seed": 99}
        run_dir = ce.run_dir_for(args, entry)
        namespace = f"sampled{count}_seed{subset_seed}"
        assert run_dir.is_relative_to(args.output_root / ce.DATASETS[1] / namespace)
        assert b.output_root == args.output_root / ce.DATASETS[1] / namespace
        assert b.cache_root == tmp_path / "caches" / ce.DATASETS[1] / namespace / "reward_factors"
        identity = ce.shared.build_identity(args, entry, config, {}, {}, 10)
        assert identity["config"]["ce_data_protocol"]["subset"] == base.subset_spec(b)
        assert ce.result_metadata(args, entry, b, config)["target_selection"] == f"sampled{count}"
        records.append((indices, run_dir, identity, ce.summary_dir_for(args)))
        b50 = ce.matched_args(args, ce.DATASETS[0], 99, matched)
        unchanged.append((proxy.build_run_config(b50), b50.cache_root,
                          ce.run_dir_for(args, {**entry, "dataset": ce.DATASETS[0]})))
    assert len({str(record[1]) for record in records}) == 3
    assert len({str(record[3]) for record in records}) == 3
    assert records[0][0] != records[2][0]
    assert records[0][2] != records[1][2] != records[2][2]
    assert unchanged[0] == unchanged[1] == unchanged[2]


@pytest.mark.parametrize("drift", ["full128", "count", "seed", "truncated", "oversized"])
def test_sampled500_cache_rejects_incompatible_subset_and_exact_row_count(drift, runtime, tmp_path):
    matched, proxy, base = runtime
    b = ce.matched_args(ce.parse_args(["--cache-root", str(tmp_path)]), ce.DATASETS[1], 42, matched)
    config = proxy.build_run_config(b)
    manifest = _write_factor_metadata(base, b, config, tmp_path, "sasrec")
    if drift == "full128":
        manifest.update(per_user_random_n=None, per_user_random_seed=None, num_examples=1000)
    elif drift == "count":
        manifest["per_user_random_n"] = 7
    elif drift == "seed":
        manifest["per_user_random_seed"] = 7
    elif drift == "truncated":
        manifest["num_examples"] = 119
    else:
        manifest["num_examples"] = 121
    ce.write_json(tmp_path / "sasrec/manifest.json", manifest)
    with pytest.raises(RuntimeError):
        ce.check_cache_metadata(base, config, "sasrec", tmp_path / "sasrec", b, 120)


def test_sampled500_runtime_rejects_direct_full_layout_bypass(runtime):
    matched, proxy, base = runtime
    b = ce.matched_args(ce.parse_args([]), ce.DATASETS[1], 42, matched)
    for count in (0, 128):
        b.train_random_n = count
        with pytest.raises(RuntimeError, match="sampled targets"):
            proxy.build_run_config(b)
        with pytest.raises(RuntimeError, match="sampled targets"):
            proxy.finetune_indices(None, b)


def test_factor_precompute_receives_same_frozen_subset(runtime, monkeypatch, tmp_path):
    matched, proxy, base = runtime
    args = ce.parse_args(["--device", "cpu", "--cache-root", str(tmp_path / "cache"),
                          "--train-random-n", "7", "--train-subset-seed", "19"])
    b = ce.matched_args(args, ce.DATASETS[1], 99, matched)
    config = proxy.build_run_config(b)
    seen = []

    def fake_precompute(command, **kwargs):
        seen.append(command)
        _write_factor_metadata(base, b, config, b.cache_root, "interaction")

    monkeypatch.setattr(base.subprocess, "run", fake_precompute)
    destination = tmp_path / "output" / "sampled7_seed19" / "cache_build" / "interaction"
    destination.mkdir(parents=True)
    manifest = proxy.ensure_component_cache("interaction", config, destination, b, None)
    assert manifest["per_user_random_n"] == 7 and manifest["per_user_random_seed"] == 19
    assert len(seen) == 1
    command = seen[0]
    assert command[command.index("--per-user-random-n") + 1] == "7"
    assert command[command.index("--per-user-random-seed") + 1] == "19"
    assert "--per-user-last-n" not in command


def test_cache_only_source_configs_and_timings_use_subset_namespace(runtime, monkeypatch, tmp_path):
    matched, proxy, base = runtime
    seen = []

    def fake_cache(name, config, destination, b, count):
        seen.append((name, config, destination, b))
        return {"num_examples": 100}

    monkeypatch.setattr(ce, "load_runtime", lambda: (matched, proxy))
    monkeypatch.setattr(proxy, "ensure_component_cache", fake_cache)
    ce.main(["--datasets", ce.DATASETS[1], "--cache-only", "--cache-component", "interaction",
             "--train-random-n", "7", "--train-subset-seed", "19",
             "--output-root", str(tmp_path / "output"), "--cache-root", str(tmp_path / "cache")])
    assert len(seen) == 1
    name, config, destination, b = seen[0]
    assert destination == tmp_path / "output" / ce.DATASETS[1] / "sampled7_seed19/cache_build/interaction"
    assert (destination / "timing.json").is_file()
    assert config["ce_data_protocol"]["subset"] == base.subset_spec(b)
    assert config["ce_data_protocol"]["target_selection"] == "sampled7"


@pytest.mark.parametrize("drift", ["last8", "history", "auxiliary", "teacher", "missing_source", "missing_factor",
                                   "truncated_rows", "wrong_catalog"])
def test_cache_metadata_rejects_wrong_data_or_reward_protocol(drift, runtime, tmp_path):
    matched, proxy, base = runtime
    b = ce.matched_args(ce.parse_args([]), ce.DATASETS[0], 42, matched)
    config = proxy.build_run_config(b)
    manifest = _write_factor_metadata(base, b, config, tmp_path, "sasrec")
    if drift == "last8":
        manifest["per_user_last_n"] = 8
    elif drift == "history":
        manifest["data_config"]["max_history_len"] = 20
    elif drift == "auxiliary":
        manifest["data_config"]["allowed_event_types"] = ["listen"]
    elif drift == "teacher":
        source = ce.read_json(manifest["config_path"])
        source["method"]["sasrec"]["checkpoint_path"] = "/wrong/teacher.pt"
        ce.write_json(manifest["config_path"], source)
    elif drift == "missing_source":
        Path(manifest["config_path"]).unlink()
    elif drift == "truncated_rows":
        manifest["num_examples"] = 8
    elif drift == "wrong_catalog":
        manifest["num_catalog_items"] = 30
    else:
        (tmp_path / "sasrec/factors.pt").unlink()
    ce.write_json(tmp_path / "sasrec/manifest.json", manifest)
    with pytest.raises(RuntimeError):
        ce.check_cache_metadata(base, config, "sasrec", tmp_path / "sasrec", b)


def test_smoke_isolates_both_outputs_and_all_cache_rows(runtime, tmp_path):
    matched, proxy, base = runtime
    args = ce.parse_args(["--smoke", "--batch-size", "2", "--grad-accum", "1",
                          "--output-root", str(tmp_path / "runs"), "--cache-root", str(tmp_path / "production")])
    b = ce.matched_args(args, ce.DATASETS[1], 42, matched)
    assert args.output_root == tmp_path / "runs_smoke"
    assert b.cache_root == args.output_root / "cache" / ce.DATASETS[1] / "sampled50_seed42/reward_factors"
    assert base._smoke_examples(b) == 16
    assert ce.curriculum_steps_for(args, 10_004_891, b.batch_size) == 40


def test_head_middle_tail_alignment_catches_drift_outside_first256():
    import torch
    count = 2000
    rows = torch.arange(count)
    base = SimpleNamespace(
        tiger_runner=SimpleNamespace(item_id_to_semantic_row=lambda dm: rows),
        indexed_train_loader=lambda dm, selected, **kwargs: [{"target_item_ids": rows[selected]}])
    proxy = ce.YambdaRuntime(base)
    factors = SimpleNamespace(num_examples=count, factors={}, target_rows=lambda selected: rows[selected])
    dm = SimpleNamespace(train_dataset=range(count))
    proxy.verify_cache_alignment(dm, None, {"sasrec": factors})
    for corrupt in (1000, 1999):
        bad = rows.clone()
        bad[corrupt] = 0
        factors.target_rows = lambda selected: bad[selected]
        with pytest.raises(RuntimeError, match="head/middle/tail"):
            proxy.verify_cache_alignment(dm, None, {"sasrec": factors})


@pytest.mark.parametrize("key,column", [("user_ids", "user_id"), ("target_item_ids", "target_item_id"),
                                        ("target_timestamps", "target_timestamp")])
@pytest.mark.parametrize("sampled", [False, True])
def test_full_vector_alignment_catches_rows_outside_all_spotchecks(key, column, sampled):
    import pandas as pd
    import torch
    count = 2000
    rows = torch.arange(count * 2 if sampled else count)
    selected = list(range(0, count * 2, 2)) if sampled else None

    class Dataset:
        examples = pd.DataFrame({column: rows.numpy()})

        def __len__(self):
            return len(rows)

    base = SimpleNamespace(
        tiger_runner=SimpleNamespace(item_id_to_semantic_row=lambda dm: rows),
        indexed_train_loader=lambda dm, selected, **kwargs: [
            {"target_item_ids": rows[selected], key: rows[selected]}])
    proxy = ce.YambdaRuntime(base)
    cached_rows = rows[selected] if sampled else rows
    values = cached_rows.clone()
    values[500] = 0  # Outside head256, middle256, and tail256.
    factors = SimpleNamespace(num_examples=count, factors={key: values}, target_rows=lambda positions: cached_rows[positions])
    with pytest.raises(RuntimeError, match="training rows"):
        proxy.verify_cache_alignment(SimpleNamespace(train_dataset=Dataset()), selected, {"sasrec": factors})


@pytest.mark.parametrize("dataset", ce.DATASETS)
def test_run_identity_fingerprints_raw_events_and_embedding_catalog(dataset, runtime, tmp_path):
    matched, proxy, base = runtime
    args = ce.parse_args([])
    config = proxy.build_run_config(ce.matched_args(args, dataset, 42, matched))
    assets = ce.shared.asset_paths_for(proxy, config, ("interaction",), [tmp_path / "interaction/manifest.json"])
    assert Path(config["data"]["interactions_path"]) in assets
    assert Path(config["data"]["semantic_id_path"]) in assets
    if dataset == ce.DATASETS[1]:
        assert Path(config["data"]["embeddings_path"]) in assets
    else:
        assert config["data"]["embeddings_path"] is None
    identity = ce.shared.build_identity(args, {"dataset": dataset, "power": 1., "rho": 1., "seed": 42},
                                       config, {}, {}, 5 * 100)
    assert identity["curriculum"]["algorithm"] == "reward_halves_linear_mixture"
    assert identity["config"]["ce_data_protocol"]["additional_per_user_random_n"] == (
        50 if dataset == ce.DATASETS[1] else None)


def test_existing_amazon_defaults_are_unchanged():
    amazon = ce.shared.parse_args([])
    assert amazon.datasets == ["beauty", "instruments"]
    assert amazon.combos == [ce.DEFAULT_COMBO]
    assert (amazon.sampling, amazon.num_samples, amazon.curriculum_epochs, amazon.checkpoint_every_updates) == (
        "reward", 10, 10., 0)


@pytest.mark.parametrize("value", ["0", "-1", "128", "129", "1000"])
def test_no_cli_can_request_full_last128_or_invalid_subset(value):
    with pytest.raises(SystemExit, match="1..127"):
        ce.parse_args(["--train-random-n", value])


def test_no_cli_can_request_negative_subset_seed_or_unbounded_exact():
    with pytest.raises(SystemExit, match="nonnegative"):
        ce.parse_args(["--train-subset-seed", "-1"])
    with pytest.raises(SystemExit):
        ce.parse_args(["--estimator", "exact", "--sampling", "reward"])
