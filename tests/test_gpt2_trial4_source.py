"""Selected 500M trial 4 keeps its matched inputs and rejects unsafe weights."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "scripts/exp2-tuned-params",
                  ROOT / "scripts/exp4-reward-ablations"):
    sys.path.insert(0, str(directory))

import gflow_finetune_gpt2_amazon as experiment
import gpt2_matched_finetune as source
from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from tuning_gpt2 import audit_matched_data

DATASET = "yambda_500m_listens"


@pytest.mark.parametrize("dataset,number", [("yambda_50m_listens", 36), (DATASET, 4)])
def test_selected_trial_prefers_original_cluster_path_and_keeps_its_metadata(tmp_path, dataset, number):
    canonical = tmp_path / "checkpoints/exp2/loo_split/initial/gpt2" / dataset / "last_128"
    trial = (tmp_path / "checkpoints/exp2-tuned_params/gpt2_tiger_matched" / dataset
             / "trials" / f"trial_{number}")
    canonical.mkdir(parents=True)
    (canonical / "best.pt").touch()
    (canonical / "resolved_config.json").write_text("{}")
    assert source.trial_paths(tmp_path, dataset)["checkpoint"] == canonical / "best.pt"

    trial.mkdir(parents=True)
    (trial / "best.pt").touch()
    paths = source.trial_paths(tmp_path, dataset)
    assert paths == {"checkpoint": trial / "best.pt", "sidecar": trial / "best.pt.json",
                     "resolved_config": trial / "resolved_config.json",
                     "data_audit": trial.parent.parent / "data_audit.json"}
    # A missing trial config must never be silently supplied by canonical.
    with pytest.raises(FileNotFoundError, match="no fallback"):
        source.load_recorded(tmp_path, dataset)

    (trial / "data_audit.json").write_text("{}")
    assert source.trial_paths(tmp_path, dataset)["data_audit"] == trial / "data_audit.json"


@pytest.mark.parametrize("dataset,family", [("yambda_50m_listens", "sasrec"), (DATASET, "sasrec_sce")])
def test_yambda_teacher_prefers_old_cluster_bundle_and_respects_explicit_binding(tmp_path, monkeypatch, dataset, family):
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    canonical = tmp_path / "checkpoints/exp2/loo_split/initial" / family / dataset
    legacy = tmp_path / "checkpoints/exp2-tuned_params/initial" / family / dataset
    canonical.mkdir(parents=True)
    (canonical / "best_trial.json").write_text(json.dumps({"best_params": {"d_model": 11}}))
    assert experiment.sasrec_asset_paths(dataset) == (canonical / "best.pt", canonical / "best_trial.json")
    legacy.mkdir(parents=True)
    (legacy / "best_model.pt").touch()
    (legacy / "best_trial.json").write_text(json.dumps({"best_params": {"d_model": 22}}))
    assert experiment.sasrec_method_cfg(dataset) == {
        "checkpoint_path": str(legacy / "best_model.pt"), "model": {"d_model": 22}}
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "trial.json").write_text(json.dumps({"best_params": {"d_model": 33}}))
    monkeypatch.setitem(experiment.DATASETS, dataset, {**experiment.DATASETS[dataset],
                        "sasrec_checkpoint": str(custom / "teacher.pt"),
                        "sasrec_trial": str(custom / "trial.json")})
    assert experiment.sasrec_method_cfg(dataset) == {
        "checkpoint_path": str(custom / "teacher.pt"), "model": {"d_model": 33}}


def selected_args(*extra):
    return experiment.parse_args(["--datasets", DATASET, "--yambda500-base", "trial4", *extra])


@pytest.fixture
def trial4(tmp_path, monkeypatch):
    events = tmp_path / "events.parquet"
    rows = []
    for user, size in ((1, 180), (2, 14)):
        for position in range(size):
            event = {"uid": user, "item_id": position % 11, "timestamp": 1000 + position,
                     "played_ratio_pct": 100., "event_type": "listen"}
            rows.append(event)
            if position % 10 == 0:
                rows.append({**event, "played_ratio_pct": 20.})
    pd.DataFrame(rows).to_parquet(events)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({str(i): [i + 1, i + 17] for i in range(11)}))
    cfg = YambdaDataConfig(
        interactions_path=events, semantic_id_path=semantic, hf_recent_months=None,
        min_user_interactions=3, min_item_interactions=0, drop_items_missing_embedding=False,
        allowed_event_types=("listen",), min_played_ratio_pct=50.,
        offset_semantic_ids_by_one=False, user_hash_mod=None,
        train_example_layout="dense_causal", train_dense_match_prefix=True,
        train_prefix_from_dense=False, train_window_len=50, train_window_stride=1,
        train_target_policy="last_n", train_last_n=128, train_all_prefixes=False,
        max_history_len=50, history_padding="right", num_workers=0, persistent_workers=False,
        include_raw_in_batch=False, batch_size=4, seed=42,
    )
    dense = YambdaDataModule(cfg)
    dense.setup()
    recorded = {"dataset": DATASET, "seed": 42,
                "model_config": {**source.TRIALS[DATASET]["architecture"], "codebook_size": 256,
                                 "max_position_embeddings": 205},
                "hparams": {"dropout": .1}, "data_config": asdict(cfg),
                "resume_identity": {"data_audit": audit_matched_data(dense)}}
    paths = source.trial_paths(tmp_path, DATASET)
    paths["resolved_config"].parent.mkdir(parents=True)
    paths["resolved_config"].write_text(json.dumps(recorded, default=str))
    paths["sidecar"].write_text(json.dumps({"model_config": recorded["model_config"]}))
    torch.save({"model_config": recorded["model_config"],
                "model_state_dict": {"weight": torch.tensor([1., 2.])}}, paths["checkpoint"])
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    return tmp_path, dense, recorded


def test_explicit_source_has_no_shared_best_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    assert experiment.tuned_checkpoint_paths(DATASET, selected_args())["checkpoint"] == (
        tmp_path / "checkpoints/exp2/loo_split/initial/gpt2"
        / DATASET / "last_128/best.pt")
    assert experiment.tuned_checkpoint_paths(DATASET)["checkpoint"].parent.name == "full_history"
    with pytest.raises(FileNotFoundError, match="no fallback"):
        source.load_recorded(tmp_path, DATASET)
    with pytest.raises(SystemExit):
        selected_args("--yambda50-base", "trial36")
    with pytest.raises(SystemExit):
        experiment.parse_args(["--datasets", "yambda_50m_listens", "--yambda500-base", "trial4"])
    for extra in (("--train-random-n", "15"), ("--train-last-n=64",)):
        with pytest.raises(SystemExit):
            selected_args(*extra)


@pytest.mark.parametrize("sft_only", [False, True])
def test_500m_matched_prefixes_use_all_128_targets_and_restore_auxiliary_events(trial4, sft_only):
    root, dense, _ = trial4
    args = selected_args("--seed", "43", "--num-workers", "0",
                         *(["--sft-only"] if sft_only else []))
    assert args.train_last_n == args.train_pool_last_n == 128 and args.train_random_n == 0
    assert not args.prefix_from_dense
    assert args.val_sample_fraction == .25 and not args.val_sample_resample
    config = experiment.build_run_config(DATASET, args, 1)
    assert config["initial_model"]["trial"] == 4 and config["initial_model"]["dataset"] == DATASET
    dm = YambdaDataModule(experiment.tiger_runner.build_data_config(config, config_path=root / "run.json"))
    dm.setup()
    audit = source.audit_prefix_data(dm, source.load_recorded(root, DATASET))
    assert audit["source_trial"] == 4 and audit["source_dataset"] == DATASET
    assert audit["all_fingerprints_equal"]
    assert len(experiment.finetune_indices(dm.train_dataset, args)) == len(dense.train_dataset)
    assert dm.train_dataset.examples.groupby("user_id").size().max() == 128
    assert dm.train_dataset.examples.history_item_ids.str.len().max() == 50
    assert len(dm.interaction_store.events) > 0
    assert dm.cfg.seed == 42 and args.seed == 43
    for split in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(dm, split).examples, getattr(dense, split).examples)


@pytest.mark.parametrize("sft_only", [False, True])
def test_sampled50_matches_tiger_and_cache_indices_without_changing_contexts(trial4, sft_only):
    from precompute_reward_cache import build_cache_loader, per_user_random_indices, per_user_tail_indices

    root, dense, _ = trial4
    args = selected_args("--yambda500-targets", "sampled50", "--seed", "43", "--num-workers", "0",
                         *(["--sft-only"] if sft_only else []))
    assert (args.train_pool_last_n, args.train_last_n, args.train_random_n, args.train_subset_seed) == (128, 0, 50, 42)
    config = experiment.build_run_config(DATASET, args, 1)
    dm = YambdaDataModule(experiment.tiger_runner.build_data_config(config, config_path=root / "run.json"))
    dm.setup()
    # The checkpoint's full last-128 layout is audited before any subsampling.
    assert source.audit_prefix_data(dm, source.load_recorded(root, DATASET))["all_fingerprints_equal"]
    indices = experiment.finetune_indices(dm.train_dataset, args)
    # TIGER uses this same draw on its already built last-128 prefix dataset.
    assert indices == per_user_random_indices(dm.train_dataset, 50, 42)
    assert indices != per_user_tail_indices(dm.train_dataset, 50)
    assert indices == sorted(set(indices))
    selected = dm.train_dataset.examples.iloc[indices]
    counts = dm.train_dataset.examples.groupby("user_id").size().clip(upper=50)
    pd.testing.assert_series_equal(selected.groupby("user_id").size(), counts)
    pd.testing.assert_frame_equal(selected, dense.train_dataset.examples.iloc[indices][selected.columns])
    assert selected.history_item_ids.str.len().max() == 50
    assert dm.item2id == dense.item2id and dm.semantic_ids == dense.semantic_ids
    for split in ("val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(getattr(dm, split).examples, getattr(dense, split).examples)

    cache_loader = build_cache_loader(dm, "train", batch_size=4, num_workers=0,
                                     **experiment.finetune_subset_spec(args))
    assert cache_loader.dataset.indices == indices
    # Training seed / epoch shuffle must not change the selected examples.
    other_seed = selected_args("--yambda500-targets", "sampled50", "--seed", "44")
    assert experiment.finetune_indices(dm.train_dataset, other_seed) == indices
    assert experiment.finetune_indices(dm.train_dataset, args) == indices


@pytest.mark.parametrize("reuse_data", [False, True])
def test_sampled50_factor_builder_uses_the_same_subset_and_rejects_old_cache(trial4, monkeypatch, reuse_data):
    import random
    import numpy as np
    import precompute_reward_factors
    from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache

    root, _, _ = trial4
    args = selected_args("--yambda500-targets", "sampled50", "--device", "cpu", "--num-workers", "0",
                         "--cache-format", "factorized", "--cache-root", str(root / "factors"))
    config = experiment.build_run_config(DATASET, args, 1)
    dm = YambdaDataModule(experiment.tiger_runner.build_data_config(config, config_path=root / "run.json"))
    dm.setup()
    indices = experiment.finetune_indices(dm.train_dataset, args)
    calls = []
    actual_precompute = precompute_reward_factors.main

    def precompute(cmd, **kwargs):
        assert not reuse_data, "A prepared datamodule must not launch a child process"
        calls.append(cmd)
        actual_precompute(cmd[2:])

    def precompute_in_process(argv, *, prepared_data):
        assert reuse_data and prepared_data is dm
        calls.append(["python", "precompute_reward_factors.py", *argv])
        actual_precompute(argv, prepared_data=prepared_data)

    monkeypatch.setattr(experiment.subprocess, "run", precompute)
    monkeypatch.setattr(precompute_reward_factors, "main", precompute_in_process)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    kwargs = {"prepared_data": dm} if reuse_data else {}
    manifest = experiment.ensure_component_cache(DATASET, "sim_hier", config, root / "out", len(indices), args, **kwargs)
    if reuse_data:
        assert random.getstate() == python_rng
        current_numpy = np.random.get_state()
        assert current_numpy[0] == numpy_rng[0] and current_numpy[2:] == numpy_rng[2:]
        np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng)
    assert len(calls) == 1
    command = calls[0]
    for flag, value in (("--per-user-random-n", "50"), ("--per-user-random-seed", "42"),
                        ("--per-user-pool-last-n", "128")):
        assert command[command.index(flag) + 1] == value
    assert "--per-user-last-n" not in command
    assert manifest["num_examples"] == len(indices)
    cache_dir = experiment.component_cache_dir(DATASET, "sim_hier", args)
    experiment.factor_runner.verify_cache_alignment(dm, indices, {"sim_hier": FactorizedRewardCache(cache_dir)})
    manifest.update(per_user_last_n=128, per_user_random_n=None, per_user_random_seed=None)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="per_user_"):
        experiment.ensure_component_cache(DATASET, "sim_hier", config, root / "out", len(indices), args, **kwargs)
    assert len(calls) == 1  # An old manifest fails before it can be used.


def test_cache_only_context_prepares_data_once_and_reuses_it_for_factors(trial4, monkeypatch):
    root, _, _ = trial4
    args = selected_args("--yambda500-targets", "sampled50", "--cache-only", "--device", "cpu",
                         "--num-workers", "0", "--cache-format", "factorized",
                         "--cache-root", str(root / "factors"), "--skip-sanity-check")
    calls = []
    actual_build = experiment.tiger_runner.build_data_module

    def build_once(cfg):
        assert not calls, "The factor builder must reuse the prepared datamodule"
        dm = actual_build(cfg)
        calls.append(dm)
        return dm

    def forbid_subprocess(*args, **kwargs):
        pytest.fail("Cache-only context must compute factors in the same process")

    monkeypatch.setattr(experiment, "validate_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(experiment, "build_gpt2_from_tuned_checkpoint",
                        lambda *a, **kw: (torch.nn.Linear(1, 1), {}))
    monkeypatch.setattr(experiment.tiger_runner, "build_data_module", build_once)
    monkeypatch.setattr(experiment.subprocess, "run", forbid_subprocess)
    result = experiment.build_dataset_context(DATASET, ["sim_hier"], args=args,
        device=torch.device("cpu"), output_root=root / "out", csv_path=root / "metrics.csv", need_teacher=False)
    assert result is None and len(calls) == 1
    manifest = json.loads((experiment.component_cache_dir(DATASET, "sim_hier", args) / "manifest.json").read_text())
    assert manifest["num_examples"] == len(experiment.finetune_indices(calls[0].train_dataset, args))


def test_sampled50_cannot_skip_or_resume_last128_results(tmp_path):
    args = selected_args("--yambda500-targets", "sampled50", "--resume")
    saved = {"dataset": DATASET, "cli_args": {**vars(args), "yambda500_targets": "last128"},
             "data": {"num_finetune_examples": 50, "finetune_per_user_pool_last_n": 128,
                      "finetune_per_user_last_n": None, "finetune_per_user_random_n": 50,
                      "finetune_subset_seed": 42}}
    # Even equal counts / overridden paths must not reuse another protocol.
    config_path = tmp_path / "resolved_config.json"
    config_path.write_text(json.dumps(saved, default=str))
    checkpoint = tmp_path / "checkpoints/last.pt"
    checkpoint.parent.mkdir()
    torch.save({"weight": torch.tensor([1.])}, checkpoint)
    with pytest.raises(RuntimeError, match="Different target selection"):
        experiment.check_existing_training_context(tmp_path, args)
    assert experiment.resumable_checkpoint(tmp_path, {"dataset": DATASET}, 50, args) is None
    saved["cli_args"]["yambda500_targets"] = "sampled50"
    config_path.write_text(json.dumps(saved, default=str))
    experiment.check_existing_training_context(tmp_path, args)
    assert experiment.resumable_checkpoint(tmp_path, {"dataset": DATASET}, 50, args) == checkpoint
    config_path.unlink()
    (tmp_path / "result.json").write_text(json.dumps({"status": "completed"}))
    with pytest.raises(RuntimeError, match="Cannot verify training context"):
        experiment.check_existing_training_context(tmp_path, args)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_trial4_checks_actual_weights_even_when_sidecar_is_valid(trial4, bad):
    root, _, recorded = trial4
    profile, model = experiment.tuned_profile(DATASET, selected_args())
    assert profile["model"]["embedding_dim"] == 256 and model["num_layers"] == 6
    checkpoint = source.trial_paths(root, DATASET)["checkpoint"]
    torch.save({"model_config": recorded["model_config"],
                "model_state_dict": {"weight": torch.tensor([1., bad])}}, checkpoint)
    with pytest.raises(ValueError, match="non-finite model weights"):
        experiment.tuned_profile(DATASET, selected_args())


def test_trial4_rejects_mismatched_actual_checkpoint_metadata(trial4):
    root, _, recorded = trial4
    checkpoint = source.trial_paths(root, DATASET)["checkpoint"]
    torch.save({"model_config": {**recorded["model_config"], "num_layers": 4},
                "model_state_dict": {"weight": torch.tensor([1.])}}, checkpoint)
    with pytest.raises(RuntimeError, match="differs"):
        experiment.tuned_profile(DATASET, selected_args())


@pytest.mark.parametrize("nll", [float("nan"), float("inf")])
def test_initial_sanity_stops_before_training_on_nonfinite_nll(trial4, monkeypatch, nll):
    root, _, _ = trial4
    args = selected_args("--num-workers", "0", "--device", "cpu")
    monkeypatch.setattr(experiment, "validate_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(experiment, "build_gpt2_from_tuned_checkpoint",
                        lambda *a, **kw: (torch.nn.Linear(1, 1), {}))
    monkeypatch.setattr(experiment.gpt2_beauty, "sanity_check_init_nll", lambda *a, **kw: nll)
    with pytest.raises(FloatingPointError, match="non-finite val NLL"):
        experiment.build_dataset_context(DATASET, [], args=args, device=torch.device("cpu"),
                                         output_root=root / "out", csv_path=root / "metrics.csv",
                                         need_teacher=False)


def test_resume_cannot_cross_legacy_and_trial4_sources(tmp_path):
    args = selected_args()
    args.resume, args.force = True, False
    saved = {"dataset": DATASET, "cli_args": {**vars(args), "yambda500_base": "legacy"},
             "data": {"num_finetune_examples": 100,
                      "finetune_per_user_pool_last_n": 128, "finetune_per_user_last_n": 128,
                      "finetune_per_user_random_n": 0, "finetune_subset_seed": args.train_subset_seed}}
    (tmp_path / "resolved_config.json").write_text(json.dumps(saved, default=str))
    checkpoint = tmp_path / "checkpoints/last.pt"
    checkpoint.parent.mkdir()
    torch.save({"weight": torch.tensor([1.])}, checkpoint)
    with pytest.raises(RuntimeError, match="Different initial checkpoint"):
        experiment.check_existing_training_context(tmp_path, args)
    assert experiment.resumable_checkpoint(tmp_path, {"dataset": DATASET}, 100, args) is None
    saved["cli_args"]["yambda500_base"] = "trial4"
    (tmp_path / "resolved_config.json").write_text(json.dumps(saved, default=str))
    experiment.check_existing_training_context(tmp_path, args)
    assert experiment.resumable_checkpoint(tmp_path, {"dataset": DATASET}, 100, args) == checkpoint
