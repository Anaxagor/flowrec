"""Yambda CE retains all exp2 last-128 targets and the auxiliary evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp7-ce-distillation/ce_distill_tiger_yambda.py"
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


def test_default_grid_is_full_last128_smooth_n3_five_epochs():
    args = ce.parse_args([])
    assert args.datasets == list(ce.DATASETS)
    assert args.combos == list(ce.COMBOS)
    assert args.powers == [.25, 1., 4., 16.] and args.rho_targets == [1., .1]
    assert args.seeds == [42]
    assert (args.estimator, args.sampling, args.num_samples, args.curriculum_epochs) == (
        "sampled", "curriculum_smooth", 3, 5.)
    assert args.checkpoint_every_updates == 10 and args.resume
    grid = ce.build_grid(args)
    assert len(grid) == len({ce.run_dir_for(args, row) for row in grid}) == 112
    assert all("curriculum_smooth_n3_epochs5" in str(ce.run_dir_for(args, row)) for row in grid)
    assert args.output_root == ROOT / "experiments/exp7-ce-distillation-tiger-yambda"


@pytest.mark.parametrize("dataset,expected", [(None, 112), *[(name, 56) for name in ce.DATASETS]])
@pytest.mark.parametrize("flag", ["--count", "--list", "--aggregate-only"])
def test_metadata_commands_do_not_import_ml_or_construct_yambda(dataset, expected, flag, tmp_path):
    command = [sys.executable, "-S", str(SCRIPT), flag, "--output-root", str(tmp_path)]
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
    import torch
    from flowrec.training import Trainer
    matched, proxy, base = runtime
    args = ce.parse_args(["--cache-root", str(tmp_path)])
    b = ce.matched_args(args, dataset, 42, matched)
    config = proxy.build_run_config(b)
    source = ce.read_json(base.abspath(base.SFT_CONFIG))
    assert config["model"] == source["model"]
    assert b.train_last_n == b.train_random_n == 0
    assert proxy.finetune_indices(range(10_004_891), b) is None
    assert (b.batch_size, b.grad_accum, b.val_batch_size) == (256, 8, 256)
    assert (b.eval_num_beams, b.test_ks) == (20, [5, 10, 20])
    assert b.cache_root == tmp_path / dataset / "last128/reward_factors"
    ce.assert_last128_config(config)
    assert config["dataset"]["train_all_prefixes"] == source["dataset"]["train_all_prefixes"]
    assert not config["dataset"].get("train_prefix_from_dense", False)
    assert config["ce_data_protocol"]["additional_per_user_random_n"] is None
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
    manifest = {"kind": "factorized", "split": "train", "num_examples": 1000,
        "num_catalog_items": 31, "data": {"num_train_examples": 1000, "num_items": 31},
        "per_user_last_n": None, "per_user_random_n": None, "per_user_random_seed": None,
        "data_config": data_config, "config_path": str(source),
        "component": {"name": name, "mode": "hierarchical" if name == "similarity" else None}}
    ce.write_json(path / "manifest.json", manifest)
    (path / "factors.pt").write_bytes(b"metadata test placeholder")
    return manifest


def test_reuse_requires_all_three_compatible_components_and_never_depends_on_combo(runtime, monkeypatch, tmp_path):
    matched, proxy, base = runtime
    dataset = ce.DATASETS[1]
    args = ce.parse_args([])
    b = ce.matched_args(args, dataset, 42, matched)
    config = proxy.build_run_config(b)
    legacy = tmp_path / "legacy_last128"
    monkeypatch.setitem(ce.LEGACY_CACHE_ROOTS, dataset, str(legacy))
    fallback = ROOT / ce.DEFAULT_CACHE_ROOT / dataset / "last128/reward_factors"
    for name in ce.COMPONENTS:
        _write_factor_metadata(base, b, config, legacy, name)
    assert ce.select_cache_root(args, dataset, b, base) == legacy
    args.combos = ["sasrec"]
    assert ce.select_cache_root(args, dataset, b, base) == legacy
    manifest = ce.read_json(legacy / "interaction/manifest.json")
    manifest["per_user_random_n"], manifest["per_user_random_seed"] = 15, 42
    ce.write_json(legacy / "interaction/manifest.json", manifest)
    assert ce.select_cache_root(args, dataset, b, base) == fallback
    with pytest.raises(RuntimeError, match="random"):
        ce.check_cache_metadata(base, config, "interaction", legacy / "interaction", b)


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
    assert b.cache_root == args.output_root / "cache" / ce.DATASETS[1] / "last128/reward_factors"
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
def test_full_vector_alignment_catches_rows_outside_all_spotchecks(key, column):
    import pandas as pd
    import torch
    count = 2000
    rows = torch.arange(count)

    class Dataset:
        examples = pd.DataFrame({column: rows.numpy()})

        def __len__(self):
            return count

    base = SimpleNamespace(
        tiger_runner=SimpleNamespace(item_id_to_semantic_row=lambda dm: rows),
        indexed_train_loader=lambda dm, selected, **kwargs: [
            {"target_item_ids": rows[selected], key: rows[selected]}])
    proxy = ce.YambdaRuntime(base)
    values = rows.clone()
    values[500] = 0  # Outside head256, middle256, and tail256.
    factors = SimpleNamespace(num_examples=count, factors={key: values}, target_rows=lambda selected: rows[selected])
    with pytest.raises(RuntimeError, match="training rows"):
        proxy.verify_cache_alignment(SimpleNamespace(train_dataset=Dataset()), None, {"sasrec": factors})


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
    assert identity["config"]["ce_data_protocol"]["additional_per_user_random_n"] is None


def test_existing_amazon_defaults_are_unchanged():
    amazon = ce.shared.parse_args([])
    assert amazon.datasets == ["beauty", "instruments"]
    assert amazon.combos == [ce.DEFAULT_COMBO]
    assert (amazon.sampling, amazon.num_samples, amazon.curriculum_epochs, amazon.checkpoint_every_updates) == (
        "reward", 10, 10., 0)


def test_no_cli_can_silently_request_old_random15_or_unbounded_exact():
    with pytest.raises(SystemExit):
        ce.parse_args(["--train-random-n", "15"])
    with pytest.raises(SystemExit):
        ce.parse_args(["--estimator", "exact", "--sampling", "reward"])
