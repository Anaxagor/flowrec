"""The new GPT-2 study preserves historical TIGER data and long schedules."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import optuna
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpt2_tiger_matched_tuning_test", ROOT / "scripts/exp2-tuned-params/tuning_gpt2.py"
)
tuning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tuning
SPEC.loader.exec_module(tuning)


@pytest.fixture(scope="module")
def regime():
    return tuning.load_regime()


def context(regime, tmp_path, dataset="yambda_50m_listens", *extra):
    args = tuning.parse_args([
        "--match-tiger", "--datasets", dataset, "--device", "cpu",
        "--output-root", str(tmp_path / "study"), "--num-workers", "0",
        "--no-progress-bar", *extra,
    ])
    return tuning.build_context(args, regime).for_dataset(dataset)


@pytest.mark.parametrize("old_class", ["YambdaDataConfig", "SequenceSplitConfig"])
def test_stale_data_code_fails_before_creating_study(regime, tmp_path, monkeypatch, old_class):
    from flowrec.data import sequence_builder, yambda

    module = yambda if old_class == "YambdaDataConfig" else sequence_builder
    cls = getattr(module, old_class)
    monkeypatch.setattr(cls, "__dataclass_fields__", {
        key: value for key, value in cls.__dataclass_fields__.items()
        if key != "train_dense_match_prefix"
    })
    ctx = context(regime, tmp_path)

    def unexpected_study(*args, **kwargs):
        pytest.fail("A study must not be created with incompatible data code")

    monkeypatch.setattr(tuning.optuna, "create_study", unexpected_study)
    with pytest.raises(RuntimeError, match=old_class) as error:
        tuning.tune_one_dataset(ctx, "yambda_50m_listens")
    message = str(error.value)
    assert str(yambda.__file__) in message and str(sequence_builder.__file__) in message
    assert "--describe-protocol" in message
    assert not ctx.output_root.exists()
    with pytest.raises(RuntimeError, match=old_class):
        tuning.describe_protocol("yambda_50m_listens", ctx)


def test_entrypoint_prefers_its_checkout_over_earlier_stale_package(tmp_path):
    shadow = tmp_path / "older-install"
    package = shadow / "flowrec"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('Imported stale flowrec package')\n")
    script = ROOT / "scripts/exp2-tuned-params/tuning_gpt2.py"
    code = f"""
import runpy
import sys
sys.path = [{str(shadow)!r}] + [p for p in sys.path if p != {str(ROOT)!r}] + [{str(ROOT)!r}]
runpy.run_path({str(script)!r}, run_name='bootstrap_check')
from flowrec.data import yambda
assert sys.path[0] == {str(ROOT)!r}
assert yambda.__file__ == {str(ROOT / 'flowrec/data/yambda.py')!r}
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_requested_gpu_is_checked_before_study_and_data_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning.torch.cuda, "is_available", lambda: False)
    args = tuning.parse_args([
        "--match-tiger", "--datasets", "yambda_50m_listens", "--device", "cuda",
        "--output-root", str(tmp_path / "out"),
    ])
    ctx = tuning.build_context(args, tuning.load_regime())

    def unexpected_setup(*args, **kwargs):
        pytest.fail("An unavailable GPU must fail before dataset or study setup")

    monkeypatch.setattr(tuning, "build_datamodule", unexpected_setup)
    monkeypatch.setattr(tuning.optuna, "create_study", unexpected_setup)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        tuning.run(args, ctx)
    assert not ctx.output_root.exists()


@pytest.mark.parametrize("name,epochs,cadence,clip,dtype,fraction", [
    ("yambda_50m_listens", 200, 1, 1.0, "bfloat16", None),
    ("yambda_500m_listens", 500, 5, 0.5, "float16", 0.25),
])
def test_settings_from_saved_tiger_regimes(regime, tmp_path, name, epochs, cadence, clip, dtype, fraction):
    ctx = context(regime, tmp_path, name)
    description = tuning.describe_protocol(name, ctx)
    trainer = description["trainer"]
    assert trainer["max_epochs"] == epochs
    assert trainer["validate_every_epochs"] == cadence
    assert trainer["early_stopping_patience"] == 10
    assert trainer["patience_in_epochs"] == 10 * cadence
    assert trainer["max_grad_norm"] == clip and trainer["amp_dtype"] == dtype
    assert trainer["val_sample_fraction"] == fraction
    assert trainer["val_sample_resample"] is False
    assert trainer["monitor"] == "val/sft/generation_ndcg@5"
    assert trainer["val_num_beams"] == 5
    assert trainer["gradient_accumulation_steps"] * description["data_config"]["batch_size"] == 2048
    assert trainer["max_steps"] is None  # Microbatch changes cannot impose a new epoch cap.
    assert description["trial_seed_policy"] == "fixed" and not description["performance_pruning"]
    assert not (tmp_path / "study").exists()  # Describe loads no data or study.
    tuning.validate_specs(ctx.regime, tuning.MATCHED_SPECS)
    assert 128 in ctx.spec(name).embedding_dim and 192 in ctx.spec(name).embedding_dim
    assert ctx.spec(name).num_layers[0] == 2
    assert tuning.TUNING_SPECS[name].max_epochs in (60, 30)  # No global mutation.


@pytest.mark.parametrize("name,offset,embedding_filter,user_mod", [
    ("yambda_50m_listens", False, False, None),
    ("yambda_500m_listens", True, True, 1),
])
def test_preprocessing_comes_from_portable_tiger_config(regime, tmp_path, name, offset, embedding_filter, user_mod):
    from scripts import run_tiger
    ctx = context(regime, tmp_path, name)
    _, actual = tuning.resolve_data_config(name, ctx)
    path = tuning.CONFIG_DIR / f"tiger_{name}_sft.json"
    expected = run_tiger.build_data_config(json.loads(path.read_text()), config_path=path)
    for key in (
        "split_strategy", "train_last_n", "train_target_policy", "max_history_len",
        "hf_recent_months", "positive_criterion", "allowed_event_types", "min_played_ratio_pct",
        "min_user_interactions", "min_item_interactions", "deduplicate_consecutive",
        "semantic_id_path", "filter_cold_items_after_split", "cold_item_train_source",
    ):
        assert getattr(actual, key) == getattr(expected, key), key
    assert actual.offset_semantic_ids_by_one == offset
    assert actual.drop_items_missing_embedding == embedding_filter
    assert actual.user_hash_mod == user_mod
    assert actual.train_dense_match_prefix and not actual.train_prefix_from_dense
    assert actual.train_example_layout == "dense_causal"
    assert actual.train_window_len == 50 and actual.train_window_stride == 1
    assert actual.history_padding == "right"


def test_isolated_defaults_and_invalid_combinations(regime, tmp_path):
    args = tuning.parse_args(["--match-tiger"])
    ctx = tuning.build_context(args, regime)
    assert args.datasets == list(tuning.MATCHED_DATASETS)
    assert ctx.output_root == tuning.MATCHED_CHECKPOINT_ROOT
    assert ctx.config_out_dir == tuning.MATCHED_CONFIG_DIR
    smaller = context(regime, tmp_path, "yambda_50m_listens", "--train-batch-size", "64")
    assert smaller.regime.grad_accum == 32 and smaller.max_steps is None
    with pytest.raises(ValueError, match="only Yambda"):
        tuning.build_context(tuning.parse_args(["--match-tiger", "--datasets", "beauty"]), regime)
    with pytest.raises(ValueError, match="divisor"):
        context(regime, tmp_path, "yambda_50m_listens", "--train-batch-size", "100")
    with pytest.raises(ValueError, match="leave-one-out"):
        replace(ctx, required_split_strategy="global_time").for_dataset("yambda_50m_listens")


def test_fingerprint_rejects_changed_precision_batch_or_data_semantics(regime, tmp_path):
    name = "yambda_500m_listens"
    ctx = context(regime, tmp_path, name)
    ctx.dataset_dir(name).mkdir(parents=True)
    study = optuna.create_study(direction="maximize")
    tuning.validate_study_protocol(study, name, ctx)
    protocol = study.user_attrs["protocol"]
    assert protocol["version"] == 2 and len(protocol["fingerprint_sha256"]) == 64
    assert protocol["regime"]["amp_dtype"] == "float16"
    tuning.validate_study_protocol(study, name, replace(ctx, num_workers=2, n_trials_override=100))
    for changed in (
        replace(ctx, regime=replace(ctx.regime, amp_dtype="bfloat16")),
        replace(ctx, train_batch_size=64, regime=replace(ctx.regime, grad_accum=32)),
        replace(ctx, validate_every_epochs=1),
        replace(ctx, max_epochs_override=30),
    ):
        with pytest.raises(ValueError, match="different split/training protocol"):
            tuning.validate_study_protocol(study, name, changed)


def synthetic_data(ctx, tmp_path, name):
    from flowrec.data.yambda import YambdaDataModule
    path = tmp_path / "events.parquet"
    rows = [dict(uid=user, item_id=i % 13, timestamp=1_700_000_000 + i,
                 played_ratio_pct=100, event_type="listen")
            for user, length in enumerate([145, 12]) for i in range(length)]
    pd.DataFrame(rows).to_parquet(path)
    sid_path = tmp_path / "semantic_ids.json"
    sid_path.write_text(json.dumps({str(i): [i // 4, 4 + i % 4, 8, 9] for i in range(13)}))
    _, cfg = tuning.resolve_data_config(name, ctx)
    cfg = replace(cfg, interactions_path=path, semantic_id_path=sid_path,
                  hf_recent_months=None, min_user_interactions=3, min_item_interactions=0,
                  embeddings_path=None, drop_items_missing_embedding=False,
                  batch_size=4, eval_batch_size=2, num_workers=0, persistent_workers=False)
    dm = YambdaDataModule(cfg)
    dm.setup()
    return dm


def test_audit_checks_supervision_and_input_changes(regime, tmp_path):
    name = "yambda_50m_listens"
    dm = synthetic_data(context(regime, tmp_path, name), tmp_path, name)
    audit = tuning.audit_matched_data(dm)
    assert audit["num_train_targets"] == 128 + 9
    assert audit["num_val_examples"] == audit["num_test_examples"] == 2
    frame = dm.train_dataset.examples
    row = frame.iloc[0]
    frame.at[0, "label_mask"] = [True] * len(row.label_mask)
    with pytest.raises(ValueError, match="one final target"):
        tuning.audit_matched_data(dm)


def test_audit_detects_changed_training_context_with_unchanged_targets(regime, tmp_path):
    name = "yambda_50m_listens"
    dm = synthetic_data(context(regime, tmp_path, name), tmp_path, name)
    before = tuning.audit_matched_data(dm)
    frame = dm.train_dataset.examples
    history = list(frame.at[0, "history_item_ids"])
    history[0], history[1] = history[1], history[0]
    frame.at[0, "history_item_ids"] = history
    after = tuning.audit_matched_data(dm)
    assert before["train_targets_sha256"] == after["train_targets_sha256"]
    assert before["train_contexts_sha256"] != after["train_contexts_sha256"]
    assert before["validation_sha256"] == after["validation_sha256"]


def test_sid_path_can_be_corrected_only_before_any_trial_starts(regime, tmp_path):
    name = "yambda_50m_listens"
    ctx = context(regime, tmp_path, name)
    ctx.dataset_dir(name).mkdir(parents=True)
    study = optuna.create_study(direction="maximize")
    missing = replace(ctx, semantic_ids_dir=tmp_path / "missing-sids")
    tuning.validate_study_protocol(study, name, missing)
    study.enqueue_trial(ctx.regime.baselines[name])
    tuning.validate_study_protocol(study, name, ctx)
    assert study.user_attrs["protocol"]["data_config"]["semantic_id_path"] == str(
        ctx.semantic_ids_dir / name / "semantic_ids.json"
    )
    trial = study.ask()
    with pytest.raises(ValueError, match="different split/training protocol"):
        tuning.validate_study_protocol(study, name, missing)
    study.tell(trial, 0.1)


@pytest.mark.parametrize("name,epochs", [("yambda_50m_listens", 1), ("yambda_500m_listens", 5)])
def test_cpu_training_uses_matched_generation_monitor_and_retains_checkpoint(regime, tmp_path, name, epochs):
    torch.set_num_threads(1)
    ctx = context(regime, tmp_path, name, "--max-epochs", str(epochs), "--debug-max-batches", "1")
    dm = synthetic_data(ctx, tmp_path, name)
    params = dict(embedding_dim=8, num_heads=2, num_layers=1, ff_mult=2,
                  dropout=0.0, lr=1e-3, weight_decay=0.01)
    result = tuning.train_once(ctx=ctx, dm=dm, dataset_name=name, hparams=params,
                               spec=ctx.spec(name), checkpoint_dir=tmp_path / "trial", seed=42)
    assert result["epochs_completed"] == epochs
    assert result["best_epoch"] == epochs - 1
    assert 0.0 <= result["value"] <= 1.0
    assert result["checkpoint"].is_file()
    assert (tmp_path / "trial/last.pt").is_file()


def test_real_gpt2_continues_interrupted_trial_with_original_total_epoch_budget(regime, tmp_path):
    torch.set_num_threads(1)
    name = "yambda_50m_listens"
    ctx = context(regime, tmp_path, name, "--max-epochs", "3", "--debug-max-batches", "1")
    dm = synthetic_data(ctx, tmp_path, name)
    ctx.dataset_dir(name).mkdir(parents=True)
    tuning.validate_study_protocol(optuna.create_study(direction="maximize"), name, ctx)
    tuning.runner.write_json(ctx.dataset_dir(name) / "data_audit.json", tuning.audit_matched_data(dm))
    params = dict(embedding_dim=8, num_heads=2, num_layers=1, ff_mult=2,
                  dropout=0.0, lr=1e-3, weight_decay=0.01)
    source = tmp_path / "interrupted"

    def interrupt_after_checkpoint(payload):
        if payload.get("type") == "epoch_end":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tuning.train_once(ctx=ctx, dm=dm, dataset_name=name, hparams=params,
                          spec=ctx.spec(name), checkpoint_dir=source, seed=42,
                          logger=interrupt_after_checkpoint)
    original = (source / "last.pt").read_bytes()
    recovery = tuning.inspect_resume_checkpoint(
        source, tuning.matched_resume_identity(ctx, name, params, 42),
    )
    assert recovery.completed_epochs == 1
    destination = tmp_path / "continued"
    result = tuning.train_once(ctx=ctx, dm=dm, dataset_name=name, hparams=params,
                               spec=ctx.spec(name), checkpoint_dir=destination, seed=42,
                               resume_from=recovery)
    assert result["epochs_completed"] == 3 and result["checkpoint"].is_file()
    state = torch.load(destination / "last.pt", weights_only=False)["trainer_state"]
    assert state["global_step"] == 3 and state["epoch"] == 2
    assert (source / "last.pt").read_bytes() == original


def test_promoted_yaml_embeds_reproducible_data_and_no_wrong_metric_name(regime, tmp_path):
    name = "yambda_500m_listens"
    ctx = context(regime, tmp_path, name)
    path = tuning.write_tuned_config(
        ctx=ctx, dataset_name=name, hparams=regime.baselines[name], best_value=0.12,
        best_trial_number=0, best_epoch=49, trials_completed=1, trials_planned=20,
        study_name=tuning.study_name_for(name, match_tiger=True),
    )
    doc = yaml.safe_load(path.read_text())
    assert "ndcg10" not in path.read_text()
    assert doc["tuning_regime"]["max_epochs"] == 500
    assert doc["tuning_regime"]["validate_every_epochs"] == 5
    assert doc["tuning_regime"]["val_sample_resample"] is False
    assert doc["tuning_regime"]["use_amp"] == ctx.regime.use_amp
    assert doc["tuning_data_config"]["train_dense_match_prefix"] is True
    assert doc["tuning_data_config"]["train_last_n"] == 128
    assert doc["tuning_data_config"]["interactions_path"].startswith("data/raw/")
    assert doc["tuning_protocol"]["supervised_target_items_per_update"] == 2048
    search = tuning.build_method(ctx).cfg
    final = tuning.run_all.build_method(doc).cfg
    for key in (
        "generation_topk", "generation_num_beams", "generation_num_return_sequences",
        "generation_constrain_valid_sequences", "generation_exclude_seen_items",
    ):
        assert getattr(final, key) == getattr(search, key), key
    assert final.generation_num_return_sequences <= final.generation_num_beams


def test_search_trials_share_seed_and_report_generic_metric(regime, tmp_path, monkeypatch):
    name = "yambda_50m_listens"
    ctx = context(regime, tmp_path, name)
    seeds = []
    def train(**kwargs):
        seeds.append(kwargs["seed"])
        folder = kwargs["checkpoint_dir"]
        folder.mkdir(parents=True)
        checkpoint = folder / "best.pt"
        checkpoint.write_bytes(b"test checkpoint")
        kwargs["logger"]({"type": "val_end", ctx.monitor: 0.1})
        return dict(value=0.1, best_epoch=0, epochs_completed=1, checkpoint=checkpoint)
    monkeypatch.setattr(tuning, "train_once", train)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    objective = tuning.make_objective(ctx=ctx, dataset_name=name, dm=None,
                                      spec=ctx.spec(name), study_name=study.study_name)
    study.optimize(objective, n_trials=2)
    assert seeds == [42, 42]
    best = json.loads((ctx.dataset_dir(name) / "best_trial.json").read_text())
    assert best["metric"] == ctx.monitor and best["best_value"] == 0.1
    assert "best_value_val_generation_ndcg10" not in best
    records = [json.loads(line) for line in (ctx.dataset_dir(name) / "epoch_metrics.jsonl").read_text().splitlines()]
    assert records[0]["value"] == 0.1 and records[0]["metric"] == ctx.monitor
    assert (ctx.dataset_dir(name) / "trials/trial_0/best.pt").exists()
