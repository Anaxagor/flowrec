"""The new CE adapter preserves SFT-only weights/data and exp7 estimators."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp8-from-sft/ce_from_sft.py"
spec = importlib.util.spec_from_file_location("ce_from_sft_tests", SCRIPT)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


@contextmanager
def no_lock(_path):
    yield


@pytest.fixture
def session(tmp_path):
    checkpoint = tmp_path / "sft_only" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"SFT ONLY WEIGHTS")
    source_config = checkpoint.parent / "resolved_config.json"
    source_config.write_text('{"method":"sft"}')
    data = tmp_path / "events.parquet"
    data.write_bytes(b"fixed data")
    b = SimpleNamespace(seed=42, batch_size=128, grad_accum=16, lr=3e-5,
        weight_decay=.01, max_epochs=500, patience=5, min_epochs=10,
        device="cpu", num_workers=0, val_sample_fraction=.25,
        val_sample_resample=False, amp_dtype="bfloat16", max_grad_norm=.5,
        validate_every=1, eval_num_beams=20, test_ks=[5, 10, 20],
        max_train_batches=None, max_eval_batches=None, smoke=False, resume=True)
    base = SimpleNamespace(control=SimpleNamespace(file_lock=no_lock),
        tiger_runner=SimpleNamespace(build_data_module=lambda _cfg: None),
        tuned_profile=lambda _dataset: ({"model": {"embedding_dim": 128}}, {}))
    return SimpleNamespace(base=base, args=b,
        source={"model": "gpt2", "dataset": "yambda_500m_listens"},
        checkpoint=checkpoint, source_assets=[checkpoint, source_config],
        source_identity={"id": "gpt2_yambda500_sft", "source_seed": 42,
                         "checkpoint": str(checkpoint)},
        config={"seed": 42, "data": {"interactions_path": str(data)},
                "dataset": {"train_prefix_from_dense": False, "train_target_policy": "last_n",
                            "train_last_n": 128, "max_history_len": 50},
                "model": {"use_user_token": True, "user_ids_count": 1},
                "method": {"name": "gflow", "component_keys": list(ce.COMPONENTS),
                           "reward": {"components": []}}},
        output_root=tmp_path / "new_experiment", cache_root=tmp_path / "cache",
        audit_data=lambda _dm: {"source_subset": "sampled15_seed42", "verified": True})


def cell(session, **overrides):
    return {"model": session.source["model"], "dataset": session.source["dataset"],
            "source_id": session.source_identity["id"], "source_seed": 42,
            "seed": 42, "arm": "ce", "combo": ("sasrec",),
            "power": .25, "rho": .1, "sampling": "reward", **overrides}


def test_two_estimators_have_48_separate_paths_and_fixed_q_sampling(session):
    paths = []
    for combo, power, rho, sampling in itertools.product(
        [*( (name,) for name in ce.COMPONENTS), ce.COMPONENTS], [.25, 1, 16], [.1, 1], ce.SAMPLINGS
    ):
        args = ce.ce_args(session, sampling)
        entry = ce.ce_entry(cell(session, combo=combo, power=power, rho=rho, sampling=sampling))
        paths.append(ce.tiger.run_dir_for(args, entry))
        assert args.num_samples == 3 and args.estimator == "sampled"
        assert args.lr == 3e-5 and args.curriculum_epochs == 5
        assert args.proposal_q_mix == .1
        assert ce.tiger.sampling_metadata(args)["ce_target"] == "q"
    assert len(set(paths)) == 48
    assert any("sampled_n3" in str(path) for path in paths)
    assert any("curriculum_smooth_is_n3_epochs5_qmix0p1" in str(path) for path in paths)


def test_config_and_trainer_preserve_bound_source_not_ce_dense_defaults(session):
    original = deepcopy(session.config)
    config = ce.source_config(session)
    assert config["dataset"] == original["dataset"]
    assert config["model"] == original["model"]
    assert config["data"] == original["data"]
    assert config["from_sft_source"] == session.source_identity
    assert config["pretrained_model"] == {"embedding_dim": 128}
    assert session.config == original
    assert ce.trainer_overrides(session)["val_sample_fraction"] == .25
    assert ce.trainer_overrides(session)["amp_dtype"] == "bfloat16"
    args = ce.ce_args(session)
    assert (args.batch_size, args.grad_accum) == (128, 16)
    session.args.checkpoint_every_updates = 0
    assert ce.ce_args(session).checkpoint_every_updates == 0


def test_gpt2_profile_uses_bound_loader_config_subset_and_source_audit(session, monkeypatch):
    loader = object()
    session.base.build_gpt2_from_tuned_checkpoint = loader
    selected_args = session.args
    captured = {}

    def run_entry(args, entry, matched, base, *, profile):
        assert base.build_gpt2_from_tuned_checkpoint is loader
        assert profile.matched_args(args, entry["dataset"], 42, matched) is selected_args
        captured["config"] = profile.build_run_config(base, entry["dataset"], selected_args)
        captured["audit"] = profile.audit_data(base, object(), selected_args)
        captured["identity"] = profile.build_identity(args, entry, captured["config"], {}, {}, 10)
        captured["metadata"] = profile.result_metadata(args, entry, selected_args, captured["config"])

    monkeypatch.setattr(ce.gpt2, "run_entry", run_entry)
    ce._gpt2_run(session, ce.ce_args(session, "curriculum_smooth_is"), ce.ce_entry(cell(session)))
    assert captured["config"]["dataset"]["train_prefix_from_dense"] is False
    assert captured["audit"]["source_subset"] == "sampled15_seed42"
    assert captured["identity"]["from_sft_source"] == session.source_identity
    assert captured["identity"]["ce_target"] == "q"
    assert captured["identity"]["calibration_sampling"] == "reward"
    assert captured["metadata"]["initial_checkpoint"] == str(session.checkpoint)


def test_tiger_hooks_use_source_and_restore_after_failure(session, monkeypatch):
    session.source["model"] = "tiger"
    original_config_builder = lambda _b: {"initial_checkpoint": "wrong original weights"}
    session.base.build_run_config = original_config_builder
    originals = {key: getattr(ce.tiger, key) for key in
                 ("build_identity", "asset_paths_for", "build_trainer_config", "verify_caches")}

    def run_entry(args, entry, matched, base, *, matched_args_fn, result_metadata_fn):
        assert matched_args_fn(args, entry["dataset"], 42, matched) is session.args
        config = base.build_run_config(session.args)
        assert config["dataset"] == session.config["dataset"]
        assert config["from_sft_source"] == session.source_identity
        assert session.checkpoint in ce.tiger.asset_paths_for(base, config, (), [])
        identity = ce.tiger.build_identity(args, entry, config, {}, {}, 10)
        assert identity["model"] == "tiger"
        assert identity["from_sft_source"] == session.source_identity
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(ce.tiger, "run_entry", run_entry)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        ce._tiger_run(session, ce.ce_args(session), ce.ce_entry(cell(session)))
    assert session.base.build_run_config is original_config_builder
    assert all(getattr(ce.tiger, key) is value for key, value in originals.items())


@pytest.mark.parametrize("model", ["gpt2", "tiger"])
def test_prepare_reuses_already_evaluated_splits_without_model_or_data_setup(session, model):
    session.source["model"] = model
    # No runner/model builders are available on this fake base: duplicate
    # setup or evaluation would fail the test immediately.
    splits = {"validation": {"NDCG@5": .1}, "test": {"NDCG@5": .09}}
    prepared = ce.prepare_ce(session, dm=object(), splits=splits)
    assert prepared["splits"] == splits
    assert prepared["from_sft_source"] == session.source_identity
    assert ce.prepare_ce(session) == prepared
    identity = prepared["identity"]
    config = ce.source_config(session)
    assert identity["checkpoint"] == ce.tiger.fingerprint([session.checkpoint])[str(session.checkpoint)]
    assert identity["dataset"] == config["dataset"]
    assert identity["eval_beams"] == 20 and identity["max_batches"] is None
    assert ("pretrained_model" in identity) == (model == "gpt2")
    # Replacing a source checkpoint must invalidate the initial evaluation.
    session.checkpoint.write_bytes(b"DIFFERENT SFT WEIGHTS")
    with pytest.raises(RuntimeError, match="baseline identity/content differs"):
        ce.prepare_ce(session, splits=splits)


def test_prepare_rejects_incomplete_shared_baseline(session):
    with pytest.raises(ValueError, match="both validation and test"):
        ce.prepare_ce(session, splits={"validation": {"NDCG@5": .1}})


def test_prepare_rejects_source_tiger_catalog_drift_before_evaluation(session):
    import torch
    session.source["model"] = "tiger"
    model = SimpleNamespace(item_semantic_ids=torch.tensor([[1, 2], [2, 1]]))
    session.base.tiger_runner.seed_everything = lambda _seed: None
    session.base.tiger_runner.build_model = lambda *_args, **_kwargs: model

    def load_different_catalog(loaded, _path):
        # Same architecture/table shape, but rows refer to other items.
        loaded.item_semantic_ids[:] = loaded.item_semantic_ids.flip(0)

    session.base.control.load_tiger_checkpoint = load_different_catalog
    with pytest.raises(RuntimeError, match="semantic IDs differ"):
        ce.prepare_ce(session, dm=object())
    assert not (session.output_root / "ce" / session.source["dataset"] /
                "baseline" / "init_metrics.json").exists()


def test_identity_fingerprints_source_config_and_factor_payloads(session, tmp_path):
    factor_dir = tmp_path / "sasrec"
    factor_dir.mkdir()
    manifest = factor_dir / "manifest.json"
    factors = factor_dir / "factors.pt"
    manifest.write_text("{}")
    factors.write_bytes(b"factor bytes")
    paths = ce.asset_paths(session, ce.source_config(session), ("sasrec",), [manifest])
    assert session.checkpoint in paths and session.source_assets[1] in paths
    assert manifest in paths and factors in paths
    assert len(paths) == len(set(paths))


def test_run_failure_reaches_array_and_missing_or_incomplete_result_is_rejected(session, monkeypatch):
    def fail(*_args):
        raise RuntimeError("failed CE step")

    monkeypatch.setattr(ce, "_gpt2_run", fail)
    with pytest.raises(RuntimeError, match="failed CE step"):
        ce.run_ce(session, cell(session))
    monkeypatch.setattr(ce, "_gpt2_run", lambda *_args: None)
    with pytest.raises(FileNotFoundError):
        ce.run_ce(session, cell(session))
    path = ce.tiger.run_dir_for(ce.ce_args(session), ce.ce_entry(cell(session))) / "result.json"
    ce.tiger.write_json(path, {"status": "failed"})
    with pytest.raises(RuntimeError, match="CE did not complete"):
        ce.run_ce(session, cell(session))


def test_run_rejects_wrong_source_and_seed_before_training(session):
    with pytest.raises(ValueError, match="bound SFT source"):
        ce.run_ce(session, cell(session, dataset="beauty"))
    with pytest.raises(ValueError, match="bound runtime"):
        ce.run_ce(session, cell(session, seed=43))


def test_ce_factory_uses_cpu_snapshot_and_refuses_raw_setup_fallback(session, monkeypatch):
    import prepared_data
    loaded = object()
    original_builder = session.base.tiger_runner.build_data_module
    monkeypatch.setattr(prepared_data, "load_prepared_data", lambda _session: loaded)
    with ce.prepared_datamodules(session):
        assert session.base.tiger_runner.build_data_module(object()) is loaded
    assert session.base.tiger_runner.build_data_module is original_builder
    monkeypatch.setattr(prepared_data, "load_prepared_data", lambda _session: None)
    with ce.prepared_datamodules(session):
        with pytest.raises(RuntimeError, match="run CPU preparation first"):
            session.base.tiger_runner.build_data_module(object())
    assert session.base.tiger_runner.build_data_module is original_builder
