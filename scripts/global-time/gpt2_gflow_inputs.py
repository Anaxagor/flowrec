"""Inputs of the promoted global-time Amazon GPT-2, TIGER and SASRec checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs/global-time"
DATASET_YAML = CONFIG_DIR / "exp2-tuned-params/dataset_amazon.yaml"
CHECKPOINTS = Path("checkpoints/exp2/global_split/initial")
# The promoted GPT-2 studies recorded these exp2 SIDs and their original codebooks.
SEMANTIC_IDS = Path("datasets/semantic_ids/exp2-tuned-params")
DATASETS = ("beauty", "instruments")
# Sampler-major grid order. The first four blocks retain the historical GPT-2 array IDs.
SAMPLERS = ("random", "policy", "teacher", "q", "q_curriculum", "policy_random")


def _dataset_config(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported global-time Amazon dataset {dataset!r}")
    config = yaml.safe_load(DATASET_YAML.read_text())
    return {**config, **config["datasets"][dataset]}


def _checkpoint_files(model: str) -> tuple[str, ...]:
    common = ("best.pt", "best_trial.json", "study_config.json", "data_summary.json")
    return (*common, "best.pt.json", "resolved_config.json") if model == "tiger" else common


def input_files(dataset: str, *, student: str = "gpt2") -> tuple[Path, ...]:
    """Repository-relative paths to stage together under one scratch root."""
    if student not in ("gpt2", "tiger"):
        raise ValueError(f"Unknown student {student!r}")
    config = _dataset_config(dataset)
    files = [Path(config["reviews_path"]), Path(config["metadata_path"]),
             SEMANTIC_IDS / dataset / "semantic_ids.json", SEMANTIC_IDS / dataset / "rqvae/model.pt"]
    for model in (student, "sasrec"):
        directory = CHECKPOINTS / model / dataset
        files.extend(directory / name for name in (*_checkpoint_files(model), "provenance.json"))
    return tuple(files)


def verify_sources(dataset: str, input_root: Path, *, student: str = "gpt2") -> dict:
    """Verify promoted weights, metadata, and the original preprocessing before cache construction."""
    config = _dataset_config(dataset)
    digests = {}
    for model in (student, "sasrec"):
        relative = CHECKPOINTS / model / dataset
        directory = input_root / relative
        provenance = json.loads((directory / "provenance.json").read_text())
        for key, expected in (("split", "global_split"), ("stage", "initial"), ("model", model),
                              ("dataset", dataset), ("checkpoint", str(relative / "best.pt"))):
            if provenance.get(key) != expected:
                raise ValueError(f"{directory}/provenance.json: expected {key}={expected}")
        for name in _checkpoint_files(model):
            expected = provenance["sha256"] if name == "best.pt" else provenance["metadata_sha256"][name]
            digest = hashlib.sha256()
            with (directory / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch for {directory / name}")
            digests[str(relative / name)] = expected
        if model == "tiger":
            settings_path = directory / "resolved_config.json"
            recorded = json.loads(settings_path.read_text())["data"]
        else:
            settings_path = directory / "study_config.json"
            recorded = json.loads(settings_path.read_text())["data_config"]
        expected_config = {key: config[key] for key in (
            "global_val_timestamp", "global_test_timestamp", "global_history_policy",
            "global_eval_target_mode", "filter_cold_items_after_split", "cold_item_train_source",
            "positive_criterion", "min_user_interactions", "min_item_interactions", "min_history_len",
            "max_history_len", "reviews_path", "metadata_path", "item_key", "filter_reviews_to_metadata",
            "deduplicate_consecutive", "offset_semantic_ids_by_one", "train_all_prefixes")}
        expected_config.update(split_strategy="global_time", drop_items_missing_semantic_id=False,
                               train_example_layout="prefix_target" if model == "tiger" else "dense_causal",
                               train_target_policy="all", seed=42,
                               train_window_len=None, train_window_stride=None,
                               semantic_id_path=str(SEMANTIC_IDS / dataset / "semantic_ids.json")
                               if model != "sasrec" else None)
        if model == "tiger":
            expected_config.update(train_prefix_from_dense=False, history_padding="left")
        for key, value in expected_config.items():
            if recorded[key] != value:
                raise ValueError(f"{settings_path}: {key}={recorded[key]!r}, expected {value!r}")
    return digests


def configure_inputs(base, input_root: Path) -> None:
    """Bind the existing GPT-2 runner to global-time configs and staged data."""
    root = Path(input_root).resolve()
    base.EXP2_DATASET_YAML = DATASET_YAML
    base.EXP1_GPT2_MODEL_YAML = CONFIG_DIR / "exp1-standard_params/gpt2_model.yaml"
    base.SEMANTIC_IDS_ROOT = root / SEMANTIC_IDS
    for dataset in DATASETS:
        teacher = root / CHECKPOINTS / "sasrec" / dataset
        # First three validation batches at B=64 yield 0.01675 and 0.07634.
        # The LOO lower bounds correspond to different held-out examples.
        base.DATASETS[dataset] = {
            "input_root": root,
            "gpt2_dir": str(root / CHECKPOINTS / "gpt2" / dataset),
            "sasrec_checkpoint": str(teacher / "best.pt"),
            "sasrec_trial": str(teacher / "best_trial.json"),
            "teacher_ndcg_threshold": 0.01 if dataset == "beauty" else 0.03,
            "training_window_only": True,
            "data_seed": 42,
            "processed_cache_dir": root / "processed" if root != REPO else None,
        }
