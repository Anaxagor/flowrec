"""CE can resume the original selected Yambda checkpoints."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in ("scripts", "scripts/exp4-reward-ablations", "scripts/exp6-ce-distillation"):
    sys.path.insert(0, str(ROOT / folder))

import ce_yambda_checkpoint_paths as paths
import gpt2_matched_finetune as source

DATASETS = ("yambda_50m_listens", "yambda_500m_listens")


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def trial_paths(root, dataset, legacy):
    if legacy:
        directory = root / "checkpoints/exp2-tuned_params/gpt2_tiger_matched" / dataset
        trial = directory / "trials" / f"trial_{source.trial_number(dataset)}"
        audit = directory / "data_audit.json"
    else:
        trial = root / "checkpoints/exp2/loo_split/initial/gpt2" / dataset / "last_128"
        audit = trial / "data_audit.json"
    return {"checkpoint": trial / "best.pt", "sidecar": trial / "best.pt.json",
            "resolved_config": trial / "resolved_config.json", "data_audit": audit}


def teacher_paths(root, dataset, legacy):
    model = "sasrec" if dataset == DATASETS[0] else "sasrec_sce"
    directory = root / ("checkpoints/exp2-tuned_params/initial" if legacy else
                        "checkpoints/exp2/loo_split/initial") / model / dataset
    return directory / ("best_model.pt" if legacy else "best.pt"), directory / "best_trial.json"


def tiger_path(root, dataset, legacy):
    if legacy:
        suffix = "yambda_50m/checkpoints/best.pt" if dataset == DATASETS[0] else dataset + "/best.pt"
        return root / "checkpoints/exp2-tuned_params/initial/tiger" / suffix
    return root / "checkpoints/exp2/loo_split/initial/tiger" / dataset / "best.pt"


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("layout", ["legacy", "canonical", "both", "missing"])
def test_exact_selected_trial_layout_and_stable_legacy_preference(tmp_path, dataset, layout):
    legacy, canonical = (trial_paths(tmp_path, dataset, old) for old in (True, False))
    if layout in ("legacy", "both"):
        touch(legacy["checkpoint"])
    if layout in ("canonical", "both"):
        touch(canonical["checkpoint"])
    assert paths.matched_trial_paths(tmp_path, dataset) == (
        legacy if layout in ("legacy", "both") else canonical)


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_fallback_to_another_trial_or_full_history(tmp_path, monkeypatch, dataset):
    other = tmp_path / "checkpoints/exp2-tuned_params/gpt2_tiger_matched" / dataset / "trials/trial_99/best.pt"
    touch(other)
    touch(tmp_path / "checkpoints/exp2/loo_split/initial/gpt2" / dataset / "full_history/best.pt")
    monkeypatch.setattr(source, "trial_paths", paths.matched_trial_paths)
    assert not source.trial_paths(tmp_path, dataset)["checkpoint"].exists()
    with pytest.raises(FileNotFoundError, match="no fallback"):
        source.load_recorded(tmp_path, dataset)


def test_original_dataset_audit_path_is_preserved_if_trial_copy_also_exists(tmp_path):
    selected = trial_paths(tmp_path, DATASETS[1], True)
    touch(selected["checkpoint"])
    local_audit = touch(selected["checkpoint"].parent / "data_audit.json")
    assert paths.matched_trial_paths(tmp_path, DATASETS[1])["data_audit"] == local_audit
    touch(selected["data_audit"])
    assert paths.matched_trial_paths(tmp_path, DATASETS[1])["data_audit"] == selected["data_audit"]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("legacy", [False, True])
def test_gpt2_teacher_metadata_stays_with_selected_weights(tmp_path, dataset, legacy):
    specs = {name: {"sasrec_checkpoint": str(teacher_paths(tmp_path, name, False)[0]),
                    "sasrec_trial": str(teacher_paths(tmp_path, name, False)[1]),
                    "untouched": name} for name in DATASETS}
    original = copy.deepcopy(specs)
    for name in DATASETS:
        touch(teacher_paths(tmp_path, name, False)[0])
    if legacy:
        touch(teacher_paths(tmp_path, dataset, True)[0])
    base = SimpleNamespace(PROJECT_ROOT=tmp_path, DATASETS=specs,
                           matched_source=SimpleNamespace(trial_paths=source.trial_paths))
    paths.configure_gpt2(base)
    checkpoint, metadata = teacher_paths(tmp_path, dataset, legacy)
    assert Path(base.DATASETS[dataset]["sasrec_checkpoint"]) in (checkpoint, checkpoint.relative_to(tmp_path))
    assert Path(base.DATASETS[dataset]["sasrec_trial"]) in (metadata, metadata.relative_to(tmp_path))
    assert base.DATASETS[dataset]["untouched"] == dataset
    other = next(name for name in DATASETS if name != dataset)
    assert base.DATASETS[other] == original[other]
    assert base.matched_source.trial_paths(tmp_path, dataset) == paths.matched_trial_paths(tmp_path, dataset)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("legacy_model,legacy_teacher", [(False, False), (True, False), (False, True), (True, True)])
def test_tiger_selects_model_and_teacher_independently_without_changing_rqvae(
        tmp_path, dataset, legacy_model, legacy_teacher):
    checkpoint, metadata = teacher_paths(tmp_path, dataset, False)
    base = SimpleNamespace(PROJECT_ROOT=tmp_path, DATASET=dataset,
        TIGER_CHECKPOINT=str(tiger_path(tmp_path, dataset, False)),
        SASREC_CHECKPOINT=str(checkpoint), SASREC_TRIAL=str(metadata),
        RQVAE_MODEL=f"datasets/semantic_ids/exp2-tuned-params/{dataset}/model.pt")
    rqvae = base.RQVAE_MODEL
    touch(tiger_path(tmp_path, dataset, False))
    touch(checkpoint)
    if legacy_model:
        touch(tiger_path(tmp_path, dataset, True))
    if legacy_teacher:
        touch(teacher_paths(tmp_path, dataset, True)[0])
    paths.configure_tiger(base)
    model = tiger_path(tmp_path, dataset, legacy_model)
    teacher, trial = teacher_paths(tmp_path, dataset, legacy_teacher)
    assert Path(base.TIGER_CHECKPOINT) in (model, model.relative_to(tmp_path))
    assert Path(base.SASREC_CHECKPOINT) in (teacher, teacher.relative_to(tmp_path))
    assert Path(base.SASREC_TRIAL) in (trial, trial.relative_to(tmp_path))
    assert base.RQVAE_MODEL == rqvae


def recorded_trial(dataset):
    return {"dataset": dataset, "seed": 42,
        "model_config": {**source.TRIALS[dataset]["architecture"], "codebook_size": 256,
                         "sem_id_len": 3, "max_position_embeddings": 205},
        "hparams": {"dropout": .1},
        "data_config": {"train_dense_match_prefix": True, "train_example_layout": "dense_causal",
            "train_prefix_from_dense": False, "split_strategy": "leave_one_out",
            "train_target_policy": "last_n", "train_last_n": 128, "max_history_len": 50,
            "train_window_len": 50, "train_window_stride": 1, "seed": 42, "user_hash_mod": 1},
        "resume_identity": {"data_audit": {key: 1 if key.startswith("num_") else "saved-hash"
                                             for key in source.AUDIT_KEYS}}}


@pytest.mark.parametrize("invalid", [None, "architecture", "audit", "layout"])
def test_legacy_selection_keeps_original_source_validation(tmp_path, monkeypatch, invalid):
    dataset = DATASETS[1]
    selected = trial_paths(tmp_path, dataset, True)
    touch(selected["checkpoint"])
    recorded = recorded_trial(dataset)
    if invalid == "architecture":
        recorded["model_config"]["num_layers"] = 2
    elif invalid == "audit":
        del recorded["resume_identity"]["data_audit"]["train_contexts_sha256"]
    elif invalid == "layout":
        recorded["data_config"]["train_last_n"] = 50
    selected["resolved_config"].write_text(json.dumps(recorded))
    monkeypatch.setattr(source, "trial_paths", paths.matched_trial_paths)
    if invalid is not None:
        with pytest.raises(ValueError):
            source.load_recorded(tmp_path, dataset)
    else:
        result = source.load_recorded(tmp_path, dataset)
        assert result["matched_source_trial"] == 4
        assert result["matched_data_audit"] == recorded["resume_identity"]["data_audit"]


def test_partial_legacy_trial_never_borrows_canonical_metadata(tmp_path, monkeypatch):
    dataset = DATASETS[1]
    legacy = trial_paths(tmp_path, dataset, True)
    canonical = trial_paths(tmp_path, dataset, False)
    touch(legacy["checkpoint"])
    touch(canonical["checkpoint"])
    canonical["resolved_config"].write_text(json.dumps(recorded_trial(dataset)))
    monkeypatch.setattr(source, "trial_paths", paths.matched_trial_paths)
    with pytest.raises(FileNotFoundError, match="no fallback") as error:
        source.load_recorded(tmp_path, dataset)
    assert str(legacy["resolved_config"]) in str(error.value)


def test_tiger_checkpoint_aliases_preserve_existing_prepared_data_identity(tmp_path):
    import ce_yambda_data_runtime as prepared_runtime
    from flowrec.data.yambda import YambdaDataConfig

    dataset = DATASETS[1]
    teacher, trial = teacher_paths(tmp_path, dataset, False)
    base = SimpleNamespace(PROJECT_ROOT=tmp_path, DATASET=dataset,
        TIGER_CHECKPOINT=str(tiger_path(tmp_path, dataset, False)),
        SASREC_CHECKPOINT=str(teacher), SASREC_TRIAL=str(trial))
    cfg = YambdaDataConfig(interactions_path=touch(tmp_path / "events.parquet"),
        semantic_id_path=touch(tmp_path / "semantic_ids.json"), dataset_size="500m",
        train_example_layout="prefix_target", train_target_policy="last_n", train_last_n=128,
        train_prefix_from_dense=False, train_dense_match_prefix=False,
        split_strategy="leave_one_out", include_raw_in_batch=False, use_hf=False)
    args = SimpleNamespace(train_random_n=50, train_subset_seed=42)
    config = {"dataset": {}, "method": {"sasrec": {"checkpoint_path": base.SASREC_CHECKPOINT}},
              "sft_checkpoint": base.TIGER_CHECKPOINT}
    before = prepared_runtime.identity(args, "tiger", cfg, config)
    touch(tiger_path(tmp_path, dataset, True))
    touch(teacher_paths(tmp_path, dataset, True)[0])
    paths.configure_tiger(base)
    config["method"]["sasrec"]["checkpoint_path"] = base.SASREC_CHECKPOINT
    config["sft_checkpoint"] = base.TIGER_CHECKPOINT
    assert Path(base.SASREC_CHECKPOINT).name == "best_model.pt"
    assert prepared_runtime.identity(args, "tiger", cfg, config) == before


def test_real_ce_runtime_builds_legacy_trial4_config_before_data_setup(tmp_path, monkeypatch):
    import torch
    import ce_distill_gpt2_yambda as ce
    import gflow_finetune_gpt2_amazon as raw

    # Restore shared runtime globals after the test; the fixture never reads raw data.
    monkeypatch.setattr(raw, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(raw, "DATASETS", copy.deepcopy(raw.DATASETS))
    monkeypatch.setattr(source, "trial_paths", source.trial_paths)
    dataset = DATASETS[1]
    selected = trial_paths(tmp_path, dataset, True)
    touch(selected["checkpoint"])
    recorded = recorded_trial(dataset)
    selected["resolved_config"].write_text(json.dumps(recorded))
    selected["sidecar"].write_text(json.dumps({"model_config": recorded["model_config"]}))
    torch.save({"model_config": recorded["model_config"],
                "model_state_dict": {"weight": torch.tensor([1., 2.])}}, selected["checkpoint"])
    teacher, metadata = teacher_paths(tmp_path, dataset, True)
    touch(teacher)
    metadata.write_text(json.dumps({"best_params": {"d_model": 256, "nhead": 2,
                                                   "num_layers": 3, "dropout": .1}}))
    matched, runtime = ce.load_runtime()
    args = ce.parse_args(["--datasets", dataset, "--device", "cpu", "--train-random-n", "50"])
    b = ce.matched_args(args, dataset, 42, matched)
    config = ce.build_run_config(ce.runtime_for(runtime, b), dataset, b)
    assert config["initial_model"]["trial"] == 4
    assert config["initial_model"]["checkpoint"] == str(selected["checkpoint"])
    assert config["method"]["sasrec"]["checkpoint_path"] == str(teacher)
    assert config["method"]["sasrec"]["model"]["num_layers"] == 3
    assert config["dataset"]["train_example_layout"] == "prefix_target"
    assert config["dataset"]["train_last_n"] == 128
    assert config["ce_data_protocol"]["target_selection"] == "sampled50"
    assert config["dataset"]["matched_tuning_data_audit"] == recorded["resume_identity"]["data_audit"]
