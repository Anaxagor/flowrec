"""Data-free tests for the three-seed runner of the tuned GPT-2 models
(run_gpt2_tuned_all.py).

They cover the tuned-YAML loading guards, the trainer regime derived from
the YAML's tuning_regime block (which must pass the exp1 runner's own
validation), checkpoint path resolution, the artifact writers, and the
per-seed merge. The CPU smoke run on real data is documented in the module
docstring of the script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts" / "exp2-tuned-params"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_tuned = _load("run_gpt2_tuned_all", SCRIPTS / "run_gpt2_tuned_all.py")
tuning = _load("tuning_gpt2", SCRIPTS / "tuning_gpt2.py")

METRIC_COLUMNS = [f"{m}@{k}" for k in (5, 10, 20) for m in ("HR", "NDCG", "Coverage", "ARP")]


@pytest.fixture(scope="module")
def regime():
    return tuning.load_regime()


def _write_tuned_yaml(tmp_path: Path, regime, dataset: str = "yambda_50m_listens", *,
                      global_matched: bool = False, data_config_dir: Path | None = None,
                      **overrides) -> Path:
    from dataclasses import replace

    ctx = tuning.RunContext(
        regime=regime,
        semantic_ids_dir=tuning.DEFAULT_SEMANTIC_IDS_DIR,
        output_root=tmp_path / "ckpt",
        config_out_dir=tmp_path / "configs",
    )
    if global_matched:
        ctx = replace(ctx, config_dir=data_config_dir or PROJECT_ROOT / run_tuned.GLOBAL_CONFIG_DIR,
                      required_split_strategy="global_time", global_yambda_regime="last_128")
    hparams = {**regime.baselines[dataset], "lr": 7.5e-4, "weight_decay": 3e-2, **overrides}
    return tuning.write_tuned_config(
        ctx=ctx, dataset_name=dataset, hparams=hparams, best_value=0.1, best_trial_number=3,
        best_epoch=7, trials_completed=10, trials_planned=40, study_name="s",
    )


def _paths(tmp_path: Path) -> "run_tuned.Paths":
    return run_tuned.Paths(
        config_dir=run_tuned.CONFIG_DIR,
        tuned_config_dir=tmp_path / "configs",
        semantic_ids_dir=run_tuned.DEFAULT_SEMANTIC_IDS_DIR,
        checkpoint_root=tmp_path / "ckpt",
        experiments_root=tmp_path / "exp",
        artifacts_root=tmp_path / "artifacts",
    )


@pytest.mark.parametrize("change, expected", [
    ({"embedding_dim": 256}, "embedding_dim"),
    ({"train_last_n": 64}, "data.train_last_n"),
])
@pytest.mark.parametrize("data_key", ["data", "data_config"])
def test_primary_yambda_eval_rejects_another_architecture_or_protocol(tmp_path, monkeypatch, change, expected, data_key):
    from dataclasses import replace

    root = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2"
    directory = root / "yambda_50m_listens/last_128"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"selected checkpoint")
    (directory / "best.pt.json").write_text(json.dumps({"model_config": {"embedding_dim": 768}}))
    data = {"split_strategy": "leave_one_out", "train_target_policy": "last_n", "train_last_n": 128}
    (directory / "resolved_config.json").write_text(json.dumps({data_key: data}))
    configured_data = {**data, **{k: v for k, v in change.items() if k != "embedding_dim"}}
    monkeypatch.setattr(run_tuned, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_tuned.run_all, "build_data_config", lambda *a, **kw: (argparse.Namespace(**configured_data), 1))
    model = {"datasets": {"yambda_50m_listens": {"model": {"embedding_dim": change.get("embedding_dim", 768)}}}}
    with pytest.raises(ValueError, match=expected):
        run_tuned.validate_primary_evaluation("yambda_50m_listens", None, {}, model,
                                             replace(_paths(tmp_path), checkpoint_root=root))


def test_global_eval_rejects_full_history_weights_copied_into_last128(tmp_path, monkeypatch):
    from dataclasses import replace

    root = tmp_path / "checkpoints/exp2/global_split/initial/gpt2"
    directory = root / "yambda_50m_listens/last_128"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"selected checkpoint")
    data = {"split_strategy": "global_time", "train_target_policy": "all", "train_last_n": 512}
    (directory / "study_config.json").write_text(json.dumps({"data_config": data}))
    monkeypatch.setattr(run_tuned, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_tuned.run_all, "build_data_config", lambda *a, **kw: (argparse.Namespace(**{**data, "train_last_n": 128}), 1))
    model = {"datasets": {"yambda_50m_listens": {"model": {"embedding_dim": 768}}}}
    with pytest.raises(ValueError, match="train_target_policy"):
        run_tuned.validate_primary_evaluation("yambda_50m_listens", None, {}, model,
                                             replace(_paths(tmp_path), checkpoint_root=root))


def _summary(dataset: str, seed, scale: float = 1.0) -> dict:
    metrics = {c: scale * (i + 1) / 100 for i, c in enumerate(METRIC_COLUMNS)}
    return {
        "seed": seed,
        "dataset": {"name": dataset},
        "validation_metrics": dict(metrics),
        "test_metrics": {k: v / 2 for k, v in metrics.items()},
        "best_checkpoint": f"/ckpt/{dataset}/seed_{seed}/best.pt",
    }


# ---------------------------------------------------------------------------
# Tuned config loading and the trainer regime
# ---------------------------------------------------------------------------

def test_tuned_yaml_loads_and_missing_or_incomplete_files_fail_loudly(tmp_path: Path, regime) -> None:
    path = _write_tuned_yaml(tmp_path, regime)
    doc = run_tuned.load_tuned_config("yambda_50m_listens", tmp_path / "configs")
    profile = run_tuned.run_all.model_profile(doc, "yambda_50m_listens")
    assert profile["model"]["embedding_dim"] == 512
    assert profile["optimizer"] == {"lr": 7.5e-4, "weight_decay": 3e-2}
    assert doc["tuning_regime"]["monitor"] == "val/sft/generation_ndcg@10"

    with pytest.raises(FileNotFoundError, match="tuning_gpt2.py"):
        run_tuned.load_tuned_config("beauty", tmp_path / "configs")

    stripped = yaml.safe_load(path.read_text())
    stripped.pop("tuning_regime")
    (tmp_path / "configs" / "gpt2_model_beauty_2023.yaml").write_text(
        yaml.safe_dump({**stripped, "datasets": {"beauty_2023": stripped["datasets"]["yambda_50m_listens"]}})
    )
    with pytest.raises(KeyError, match="tuning_regime"):
        run_tuned.load_tuned_config("beauty_2023", tmp_path / "configs")


def test_trainer_config_mirrors_the_tuning_regime_and_passes_the_exp1_runner_checks(tmp_path: Path, regime) -> None:
    _write_tuned_yaml(tmp_path, regime)
    paths = _paths(tmp_path)
    doc = run_tuned.load_tuned_config("yambda_50m_listens", tmp_path / "configs")
    trainer = run_tuned.build_trainer_config("yambda_50m_listens", doc, paths)

    spec = tuning.TUNING_SPECS["yambda_50m_listens"]
    assert trainer["max_epochs"] == spec.max_epochs
    assert trainer["early_stopping_patience"] == spec.patience
    assert trainer["monitor"] == "val/sft/generation_ndcg@10"
    assert trainer["monitor_mode"] == "max"
    assert trainer["weight_decay"] == pytest.approx(3e-2)          # tuned, not exp1's 1e-2
    assert trainer["min_delta"] == pytest.approx(regime.min_delta)
    assert trainer["gradient_accumulation_steps"] == regime.grad_accum
    assert trainer["seeds"] == [42, 43, 44]
    assert trainer["checkpoints_dir"] == str(tmp_path / "ckpt")
    assert trainer["metrics_summary_name"] == "gpt2_{dataset}_metrics_summary.csv"

    # The exp1 runner must accept the config verbatim (its whitelist raises on
    # unknown keys) and apply the profile's validation subsample.
    args = argparse.Namespace(max_epochs=None, max_steps=None, max_eval_batches=None)
    profile = run_tuned.run_all.model_profile(doc, "yambda_50m_listens")
    cfg = run_tuned.run_all.build_trainer_config(trainer, profile, tmp_path / "c", args, use_amp=False)
    assert cfg.val_sample_fraction == 0.1 and cfg.val_sample_resample is False
    assert cfg.early_stopping_patience == spec.patience and cfg.monitor_mode == "max"
    optimizer = run_tuned.run_all.optimizer_config(trainer, profile)
    assert optimizer["lr"] == pytest.approx(7.5e-4) and optimizer["weight_decay"] == pytest.approx(3e-2)
    assert optimizer["betas"] == tuple(regime.betas)


def test_amazon_profile_has_no_validation_subsample(tmp_path: Path, regime) -> None:
    _write_tuned_yaml(tmp_path, regime, dataset="beauty")
    doc = run_tuned.load_tuned_config("beauty", tmp_path / "configs")
    trainer = run_tuned.build_trainer_config("beauty", doc, _paths(tmp_path))
    args = argparse.Namespace(max_epochs=None, max_steps=None, max_eval_batches=None)
    cfg = run_tuned.run_all.build_trainer_config(
        trainer, run_tuned.run_all.model_profile(doc, "beauty"), tmp_path / "c", args, use_amp=False
    )
    assert cfg.val_sample_fraction is None
    assert cfg.max_epochs == tuning.TUNING_SPECS["beauty"].max_epochs


def test_matched_export_restores_data_and_validation_protocol(tmp_path, regime, monkeypatch):
    from dataclasses import asdict
    from flowrec.data.yambda import YambdaDataConfig

    path = _write_tuned_yaml(tmp_path, regime)
    doc = yaml.safe_load(path.read_text())
    recorded = asdict(YambdaDataConfig(
        interactions_path=tmp_path / "same_events.parquet",
        semantic_id_path=tmp_path / "same_ids.json",
        dataset_size="50m", hf_recent_months=2,
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_target_policy="last_n", train_last_n=128, max_history_len=50,
        train_window_len=50, train_window_stride=1, history_padding="right",
        batch_size=256, eval_batch_size=128, user_hash_mod=None,
        offset_semantic_ids_by_one=False, drop_items_missing_embedding=False,
    ))
    doc["tuning_data_config"] = run_tuned.runner.to_jsonable(recorded)
    doc["tuning_regime"].update(
        validate_every_epochs=5, max_steps=None, max_epochs=500,
        gradient_accumulation_steps=8, val_sample_fraction=0.25,
        val_sample_seed=0, val_sample_resample=False, monitor="val/sft/generation_ndcg@5",
    )
    doc["datasets"]["yambda_50m_listens"]["training"] = {
        "val_sample_fraction": 0.25, "val_sample_seed": 0, "val_sample_resample": False,
    }
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(run_tuned.run_all, "validate_inputs", lambda *a, **kw: None)
    plan, = run_tuned.build_plans(argparse.Namespace(datasets=["yambda_50m_listens"]), _paths(tmp_path))
    cfg, _ = run_tuned.run_all.build_data_config(
        plan.group, plan.dataset_name, plan.dataset_config, seed=43,
        semantic_ids_dir=tmp_path / "unrelated_ids", user_hash_mod=1,
    )
    assert cfg.train_dense_match_prefix and cfg.train_last_n == 128
    assert cfg.train_window_stride == 1 and cfg.train_window_len == 50
    assert cfg.user_hash_mod is None and not cfg.offset_semantic_ids_by_one
    assert not cfg.drop_items_missing_embedding and cfg.seed == 43
    assert cfg.semantic_id_path == tmp_path / "same_ids.json"
    assert cfg.interactions_path == tmp_path / "same_events.parquet"
    trainer = plan.trainer_config
    assert trainer["validate_every_epochs"] == 5 and trainer["max_epochs"] == 500
    assert trainer["gradient_accumulation_steps"] * cfg.batch_size == 2048
    assert trainer["val_sample_fraction"] == 0.25 and not trainer["val_sample_resample"]
    assert trainer["monitor"] == "val/sft/generation_ndcg@5"


def test_matched_export_cannot_overwrite_legacy_checkpoints(tmp_path, regime):
    from dataclasses import replace

    path = _write_tuned_yaml(tmp_path, regime)
    doc = yaml.safe_load(path.read_text())
    doc["tuning_data_config"] = {"train_dense_match_prefix": True}
    path.write_text(yaml.safe_dump(doc))
    paths = replace(_paths(tmp_path), checkpoint_root=PROJECT_ROOT / run_tuned.CHECKPOINTS_DIR)
    with pytest.raises(ValueError, match="separate --checkpoints-dir"):
        run_tuned.build_plans(argparse.Namespace(datasets=["yambda_50m_listens"]), paths)


@pytest.mark.parametrize("dataset", ["yambda_50m_listens", "yambda_500m_listens"])
def test_global_final_uses_the_matched_export_on_the_canonical_root(tmp_path, dataset, regime, monkeypatch):
    from dataclasses import replace

    yaml_path = _write_tuned_yaml(tmp_path, regime, dataset)
    export = yaml.safe_load(yaml_path.read_text())
    reference = PROJECT_ROOT / "checkpoints/exp2/global_split/initial/tiger" / dataset / "resolved_config.json"
    data = json.loads(reference.read_text())["data"]
    export["tuning_data_config"] = dict(data, train_example_layout="dense_causal", train_dense_match_prefix=True,
                                       train_window_len=50, train_window_stride=1, history_padding="right")
    paths = replace(_paths(tmp_path), config_dir=PROJECT_ROOT / "configs/global-time/exp2-tuned-params",
                    checkpoint_root=tmp_path / "checkpoints/exp2/global_split/initial/gpt2")
    monkeypatch.setattr(run_tuned, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_tuned.run_all, "validate_inputs", lambda *a, **kw: None)
    args = argparse.Namespace(datasets=[dataset])
    with pytest.raises(ValueError, match="last_128 data protocol"):
        run_tuned.build_plans(args, paths)
    yaml_path.write_text(yaml.safe_dump(export))
    plan, = run_tuned.build_plans(args, paths)
    cfg, _ = run_tuned.run_all.build_data_config(plan.group, dataset, plan.dataset_config, seed=42,
                                                 semantic_ids_dir=paths.semantic_ids_dir, user_hash_mod=1)
    assert cfg.train_last_n == 128 and cfg.train_dense_match_prefix
    assert cfg.global_val_timestamp == data["global_val_timestamp"]
    canonical_configs = tmp_path / "configs/global-time/exp2-tuned-params"
    assert run_tuned.tuned_config_path(dataset, canonical_configs, paths.checkpoint_root) == canonical_configs / "gpt2_last_128" / yaml_path.name


@pytest.mark.parametrize("dataset", ["yambda_50m_listens", "yambda_500m_listens"])
def test_match_tiger_final_consumes_global_hpo_export(tmp_path, dataset, regime, monkeypatch):
    path = _write_tuned_yaml(tmp_path, regime, dataset, global_matched=True)
    args = run_tuned.parse_args(["--match-tiger", "--datasets", dataset, "--seeds", "42",
                                 "--tuned-config-dir", str(path.parent)])
    paths = run_tuned.resolve_paths(args)
    monkeypatch.setattr(run_tuned.run_all, "validate_inputs", lambda *a, **kw: None)
    plan, = run_tuned.build_plans(args, paths)
    assert args.seeds == [42] and plan.dataset_name == dataset
    assert plan.model_config["tuning_protocol"]["name"] == "global_time_tiger_last_128"
    assert plan.trainer_config["monitor"] == plan.model_config["tuning_regime"]["monitor"]
    assert plan.trainer_config["val_sample_fraction"] == 0.25
    assert run_tuned.tuned_config_path(dataset, paths.tuned_config_dir, paths.checkpoint_root) == path
    assert paths.checkpoint_root == PROJECT_ROOT / "checkpoints/exp2/global_split/initial/gpt2"
    checkpoint_root = tmp_path / paths.checkpoint_root.relative_to(PROJECT_ROOT)
    monkeypatch.setattr(run_tuned, "PROJECT_ROOT", tmp_path)
    assert run_tuned.checkpoint_path_for(checkpoint_root, dataset, 42, which="best") == (
        checkpoint_root / dataset / "last_128/best.pt")
    assert paths.artifacts_root == PROJECT_ROOT / "artifacts/global-time/exp2-tuned-params/gpt2_last_128"
    assert paths.experiments_root == PROJECT_ROOT / "experiments/global-time/exp2-tuned-params/gpt2_last_128"


@pytest.mark.parametrize("dataset", ["yambda_50m_listens", "yambda_500m_listens"])
def test_match_tiger_final_uses_tiger_protocol_with_legacy_global_yaml(tmp_path, dataset, regime, monkeypatch):
    name = "dataset_yambda.yaml"
    source = tmp_path / name
    config = yaml.safe_load((PROJECT_ROOT / run_tuned.GLOBAL_CONFIG_DIR / name).read_text())
    for key in ("train_target_policy", "train_last_n", "train_dense_match_prefix"):
        config["gpt2"].pop(key, None)
    source.write_text(yaml.safe_dump(config))
    export = _write_tuned_yaml(tmp_path, regime, dataset, global_matched=True, data_config_dir=tmp_path)
    args = run_tuned.parse_args(["--match-tiger", "--datasets", dataset, "--config-dir", str(tmp_path),
                                 "--tuned-config-dir", str(export.parent)])
    paths = run_tuned.resolve_paths(args)
    monkeypatch.setattr(run_tuned.run_all, "validate_inputs", lambda *a, **kw: None)
    plan, = run_tuned.build_plans(args, paths)
    cfg, _ = run_tuned.run_all.build_data_config(plan.group, dataset, plan.dataset_config, seed=42,
                                                 semantic_ids_dir=paths.semantic_ids_dir, user_hash_mod=1)
    assert (cfg.train_target_policy, cfg.train_last_n, cfg.train_dense_match_prefix) == ("last_n", 128, True)
    assert (cfg.train_window_len, cfg.train_window_stride) == (50, 1)
    assert cfg.global_test_timestamp == config["datasets"][dataset]["global_test_timestamp"]

    config["datasets"][dataset]["global_test_timestamp"] += 1
    source.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="global_test_timestamp"):
        run_tuned.build_plans(args, paths)


def test_match_tiger_defaults_and_incompatible_cli_inputs():
    args = run_tuned.parse_args(["--match-tiger"])
    assert args.datasets == list(run_tuned.MATCHED_YAMBDA_DATASETS)
    assert (args.seeds or run_tuned.SEEDS) == (42, 43, 44)
    assert run_tuned.resolve_paths(args).tuned_config_dir == PROJECT_ROOT / run_tuned.GLOBAL_CONFIG_DIR / "gpt2_last_128"
    legacy = run_tuned.parse_args([])
    assert legacy.datasets == list(run_tuned.DATASETS) and not legacy.match_tiger
    assert run_tuned.resolve_paths(legacy).checkpoint_root == PROJECT_ROOT / run_tuned.CHECKPOINTS_DIR
    for extra in (["--datasets", "beauty"], ["--checkpoints-dir", run_tuned.CHECKPOINTS_DIR],
                  ["--checkpoints-dir", "checkpoints/exp2/global_split/initial/tiger"]):
        with pytest.raises(SystemExit):
            run_tuned.parse_args(["--match-tiger", *extra])


@pytest.mark.parametrize("fault", ["full_history", "loo_export", "wrong_dataset_config", "legacy_sample"])
def test_match_tiger_final_fails_before_input_loading_for_an_incompatible_study(tmp_path, regime, monkeypatch, fault):
    name = "yambda_50m_listens"
    path = _write_tuned_yaml(tmp_path, regime, name, global_matched=True)
    export = yaml.safe_load(path.read_text())
    options = ["--match-tiger", "--datasets", name, "--tuned-config-dir", str(path.parent),
               "--checkpoints-dir", str(tmp_path / "private")]
    if fault == "full_history":
        export["tuning_protocol"] = {"name": "global_time_full_history", "yambda_regime": "full_history"}
    elif fault == "loo_export":
        export["tuning_data_config"]["split_strategy"] = "leave_one_out"
    elif fault == "legacy_sample":
        export["datasets"][name]["training"]["val_sample_fraction"] = 0.1
    else:
        options += ["--config-dir", str(run_tuned.CONFIG_DIR)]
    path.write_text(yaml.safe_dump(export))
    monkeypatch.setattr(run_tuned.run_all, "validate_inputs", lambda *a, **kw: pytest.fail("validated input files"))
    args = run_tuned.parse_args(options)
    with pytest.raises(ValueError, match="global_time_tiger_last_128|global-time last_128 data protocol|fixed 25%"):
        run_tuned.build_plans(args, run_tuned.resolve_paths(args))


@pytest.mark.parametrize("legacy_export", [True, False])
def test_explicit_global_full_history_run_keeps_all_targets(tmp_path, regime, monkeypatch, legacy_export):
    from dataclasses import replace

    name = "yambda_50m_listens"
    path = _write_tuned_yaml(tmp_path, regime, name)
    legacy = PROJECT_ROOT / f"checkpoints/exp2/global_split/initial/gpt2/{name}/full_history/study_config.json"
    data = json.loads(legacy.read_text())["data_config"]
    paths = replace(_paths(tmp_path), config_dir=PROJECT_ROOT / "configs/global-time/exp2-tuned-params")
    if legacy_export:
        directory = paths.checkpoint_root / name
        directory.mkdir(parents=True)
        (directory / "study_config.json").write_text(legacy.read_text())
    else:
        doc = yaml.safe_load(path.read_text())
        doc.update(tuning_data_config=data, tuning_protocol={"yambda_regime": "full_history"})
        path.write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(run_tuned.run_all, "validate_inputs", lambda *a, **kw: None)
    plan, = run_tuned.build_plans(argparse.Namespace(datasets=[name]), paths)
    cfg, _ = run_tuned.run_all.build_data_config(plan.group, name, plan.dataset_config, seed=42,
                                                 semantic_ids_dir=paths.semantic_ids_dir, user_hash_mod=1)
    assert cfg.train_target_policy == "all" and not cfg.train_dense_match_prefix
    assert cfg.train_window_stride is None


# ---------------------------------------------------------------------------
# Checkpoints and artifact writers
# ---------------------------------------------------------------------------

def test_checkpoint_paths(tmp_path: Path) -> None:
    root = tmp_path / "ckpt"
    assert run_tuned.checkpoint_path_for(root, "beauty", 43, which="best") == root / "beauty" / "seed_43" / "best.pt"
    assert run_tuned.checkpoint_path_for(root, "beauty", 43, which="last") == root / "beauty" / "seed_43" / "last.pt"
    assert run_tuned.checkpoint_path_for(root, "beauty", None, which="incumbent") == root / "beauty" / "best_model.pt"
    with pytest.raises(ValueError):
        run_tuned.checkpoint_path_for(root, "beauty", None, which="best")


def test_dataset_outputs_are_seed_scoped_for_single_seed_runs(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    ks = (5, 10, 20)
    path = run_tuned.write_dataset_outputs(
        root, "beauty", [_summary("beauty", 42)], seeds=[42], ranking_ks=ks,
        metadata={"mode": "train"}, names=run_tuned.TRAIN_OUTPUTS,
    )
    assert path == root / "gpt2_beauty_metrics_summary_seed-42.csv"
    assert (root / "gpt2_beauty_metrics_aggregate_seed-42.csv").is_file()
    payload = json.loads((root / "gpt2_beauty_summary_seed-42.json").read_text())
    assert payload["seeds"] == [42] and payload["runs"][0]["seed"] == 42
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert [r["split"] for r in rows] == ["validation", "test"]
    assert set(METRIC_COLUMNS) <= set(rows[0])

    # Three seeds -> plain names; incumbent -> plain names with its own templates.
    path3 = run_tuned.write_dataset_outputs(
        root, "beauty", [_summary("beauty", s) for s in (42, 43, 44)], seeds=[42, 43, 44],
        ranking_ks=ks, metadata={}, names=run_tuned.TRAIN_OUTPUTS,
    )
    assert path3 == root / "gpt2_beauty_metrics_summary.csv"
    inc = run_tuned.write_dataset_outputs(
        root, "beauty", [_summary("beauty", "incumbent")], seeds=None, ranking_ks=ks,
        metadata={}, names=run_tuned.INCUMBENT_OUTPUTS,
    )
    assert inc == root / "gpt2_beauty_incumbent_eval_metrics_summary.csv"


def test_merge_folds_per_seed_payloads_into_plain_tables_with_mean_std(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    ks = (5, 10, 20)
    for dataset in ("beauty", "instruments"):
        for seed, scale in ((42, 1.0), (43, 2.0), (44, 3.0)):
            run_tuned.write_dataset_outputs(
                root, dataset, [_summary(dataset, seed, scale)], seeds=[seed], ranking_ks=ks,
                metadata={"mode": "train"}, names=run_tuned.TRAIN_OUTPUTS,
            )
    run_tuned.merge_seed_summaries(
        ["beauty", "instruments"], seeds=[42, 43, 44], artifacts_root=root, ranking_ks=ks,
        names=run_tuned.TRAIN_OUTPUTS, skip_missing=False,
    )
    with open(root / "gpt2_beauty_metrics_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and sorted({r["seed"] for r in rows}) == ["42", "43", "44"]
    with open(root / "gpt2_beauty_metrics_aggregate.csv") as f:
        agg = {r["split"]: r for r in csv.DictReader(f)}
    # HR@5 = scale * 0.01 over scales 1, 2, 3 -> mean 0.02, population std ~0.008165.
    assert float(agg["validation"]["HR@5_mean"]) == pytest.approx(0.02)
    assert float(agg["validation"]["HR@5_std"]) == pytest.approx(0.008165, abs=1e-5)
    assert agg["validation"]["num_seeds"] == "3"
    assert (root / "gpt2_all_metrics_summary.csv").is_file()
    assert (root / "gpt2_all_metrics_aggregate.csv").is_file()
    payload = json.loads((root / "gpt2_beauty_summary.json").read_text())
    assert payload["merged_seeds"] == [42, 43, 44] and payload["seeds"] is None

    # A missing seed aborts unless --skip-missing.
    with pytest.raises(FileNotFoundError, match="seed-45"):
        run_tuned.merge_seed_summaries(
            ["beauty"], seeds=[42, 45], artifacts_root=root, ranking_ks=ks,
            names=run_tuned.TRAIN_OUTPUTS, skip_missing=False,
        )
    run_tuned.merge_seed_summaries(
        ["beauty"], seeds=[42, 45], artifacts_root=root, ranking_ks=ks,
        names=run_tuned.TRAIN_OUTPUTS, skip_missing=True,
    )
    with open(root / "gpt2_beauty_metrics_summary.csv") as f:
        assert {r["seed"] for r in csv.DictReader(f)} == {"42"}
