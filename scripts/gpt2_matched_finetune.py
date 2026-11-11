"""Recorded input protocol for explicitly selected matched Yambda checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

DATASET = "yambda_50m_listens"
TRIAL = 36
TRIAL_DIRECTORY = Path("checkpoints/exp2/loo_split/initial/gpt2") / DATASET / "last_128"
EXPECTED_ARCHITECTURE = {"embedding_dim": 768, "num_layers": 4, "num_heads": 2,
                         "dim_feedforward": 3072}
TRIALS = {
    DATASET: {"trial": TRIAL, "flag": "yambda50_base", "selection": "trial36",
              "architecture": EXPECTED_ARCHITECTURE},
    "yambda_500m_listens": {
        "trial": 4, "flag": "yambda500_base", "selection": "trial4",
        "architecture": {"embedding_dim": 256, "num_layers": 6, "num_heads": 4,
                         "dim_feedforward": 512}},
}
AUDIT_KEYS = ("num_train_targets", "num_items", "num_val_examples", "num_test_examples",
              "train_targets_sha256", "train_contexts_sha256", "validation_sha256",
              "test_sha256", "catalog_sha256", "semantic_ids_sha256")


def selected(dataset: str, args: Any = None) -> bool:
    spec = TRIALS.get(dataset)
    return bool(spec and getattr(args, spec["flag"], "legacy") == spec["selection"])


def trial_number(dataset: str = DATASET) -> int:
    return int(TRIALS[dataset]["trial"])


def trial_paths(project_root: Path, dataset: str = DATASET) -> dict[str, Path]:
    # Older cluster runs use the selected trial in its original tuning tree.
    # Keep its metadata with the same checkpoint: mixing it with the promoted
    # copy could silently change the training protocol on resume.
    tuning = project_root / "checkpoints/exp2-tuned_params/gpt2_tiger_matched" / dataset
    trial = tuning / "trials" / f"trial_{trial_number(dataset)}"
    if (trial / "best.pt").is_file():
        trial_audit = trial / "data_audit.json"
        return {"checkpoint": trial / "best.pt", "sidecar": trial / "best.pt.json",
                "resolved_config": trial / "resolved_config.json",
                "data_audit": trial_audit if trial_audit.is_file() else tuning / "data_audit.json"}
    canonical = project_root / "checkpoints/exp2/loo_split/initial/gpt2" / dataset / "last_128"
    canonical_paths = {"checkpoint": canonical / "best.pt", "sidecar": canonical / "best.pt.json",
                       "resolved_config": canonical / "resolved_config.json",
                       "data_audit": canonical / "data_audit.json"}
    return canonical_paths


def load_recorded(project_root: Path, dataset: str = DATASET) -> dict[str, Any]:
    trial = trial_number(dataset)
    paths = trial_paths(project_root, dataset)
    path = paths["resolved_config"]
    if not path.is_file():
        raise FileNotFoundError(f"Selected matched {dataset} trial {trial} requires {path}; no fallback to another checkpoint")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("dataset") != dataset:
        raise ValueError(f"{path} is not a {dataset} trial")
    architecture = recorded.get("model_config", {})
    for key, expected in TRIALS[dataset]["architecture"].items():
        if architecture.get(key) != expected:
            raise ValueError(f"Selected trial {trial} has unexpected {key}: {architecture.get(key)!r}, expected {expected}")
    data = recorded.get("data_config", {})
    expected_data = {"train_dense_match_prefix": True, "train_example_layout": "dense_causal",
                     "train_prefix_from_dense": False, "split_strategy": "leave_one_out",
                     "train_target_policy": "last_n", "train_last_n": 128,
                     "max_history_len": 50, "train_window_len": 50, "train_window_stride": 1}
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ValueError(f"Selected trial {trial} requires {key}={expected!r}, got {data.get(key)!r}")
    if "dropout" not in recorded.get("hparams", {}):
        raise ValueError(f"Selected trial {trial} lacks its recorded dropout")
    audit = (recorded.get("resume_identity") or {}).get("data_audit")
    if audit is None and paths["data_audit"].is_file():
        audit = json.loads(paths["data_audit"].read_text(encoding="utf-8"))
    if not audit or any(key not in audit for key in AUDIT_KEYS):
        raise ValueError(f"Selected trial {trial} requires its full saved data audit, including training histories")
    recorded["matched_data_audit"] = {key: audit[key] for key in AUDIT_KEYS}
    recorded["matched_source_trial"] = trial
    return recorded


def load_finite_checkpoint(path: Path) -> dict[str, Any]:
    """Fail before GPU allocation if the selected best checkpoint has NaN/Inf.

    Trial 4 later failed with NaN; its earlier best.pt is usable only if the
    saved model itself is finite. Optimizer state is not used for fine-tuning.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{path} lacks a non-empty model_state_dict")
    invalid = [name for name, tensor in state.items()
               if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item()]
    if invalid:
        raise ValueError(f"Selected checkpoint {path} has non-finite model weights: {invalid[:10]}")
    return checkpoint


def recorded_path(value: str | None, project_root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    elif not path.exists() and "/flowrec/" in str(path):
        path = project_root / str(path).split("/flowrec/", 1)[1]
    return str(path)


def input_config(recorded: Mapping[str, Any], project_root: Path) -> tuple[dict, dict]:
    """One prefix per matched dense target, with exactly its preceding history."""
    values = dict(recorded["data_config"])
    # The selected SFT run discarded non-listen events and partial listens.
    # Keep its exact positive predicate, but partition those other events
    # into the auxiliary store so the interaction reward has real evidence.
    # The normalizer's listen threshold does NOT apply to other event types.
    # Date/embedding/organic scope remains unchanged and precedes either path.
    predicates = []
    if values.get("positive_criterion") is not None:
        predicates.append(f"({values['positive_criterion']})")
    if values.get("allowed_event_types") is not None:
        predicates.append(f"(event_type in {list(values['allowed_event_types'])!r})")
    if values.get("min_played_ratio_pct") is not None:
        threshold = float(values["min_played_ratio_pct"])
        predicates.append(f"(event_type != 'listen' or played_ratio_pct >= {threshold!r})")
    if predicates:
        values.update(positive_criterion=" and ".join(predicates),
                      allowed_event_types=None, min_played_ratio_pct=None)
    data = {"source": "yambda"}
    for key in ("interactions_path", "semantic_id_path", "embeddings_path"):
        data[key] = recorded_path(values.pop(key, None), project_root)
    for key in ("use_hf", "hf_dataset_name", "streaming", "max_rows"):
        if key in values:
            data[key] = values.pop(key)
    values.update(train_example_layout="prefix_target", train_dense_match_prefix=False,
                  train_prefix_from_dense=False, history_padding="left",
                  matched_tuning_data_audit=dict(recorded["matched_data_audit"]))
    return data, values


def audit_prefix_data(dm: Any, recorded: Mapping[str, Any]) -> dict[str, Any]:
    """Compare complete ordered prefixes/catalog/held-out data to the saved study."""
    import pandas as pd

    def frame_hash(frame: Any, histories: bool) -> str:
        digest = hashlib.sha256()
        columns = [c for c in ("raw_user_id", "target_position", "target_raw_item", "target_timestamp")
                   if c in frame]
        if not columns:
            columns = ["user_id", "target_position", "target_item_id"]
        for start in range(0, len(frame), 100_000):
            subset = frame.iloc[start:start + 100_000][columns].copy()
            if histories:
                subset["history_item_ids"] = frame.iloc[start:start + 100_000].history_item_ids.map(tuple)
            digest.update(pd.util.hash_pandas_object(subset, index=False).to_numpy().tobytes())
        return digest.hexdigest()

    train = dm.train_dataset.examples
    actual = {
        "num_train_targets": len(train), "num_items": dm.num_items,
        "num_val_examples": len(dm.val_dataset), "num_test_examples": len(dm.test_dataset),
        "train_targets_sha256": frame_hash(train, False),
        "train_contexts_sha256": frame_hash(train, True),
        "validation_sha256": frame_hash(dm.val_dataset.examples, True),
        "test_sha256": frame_hash(dm.test_dataset.examples, True),
        "catalog_sha256": hashlib.sha256(json.dumps(sorted(dm.item2id.items())).encode()).hexdigest(),
        "semantic_ids_sha256": hashlib.sha256(json.dumps(sorted(dm.semantic_ids.items())).encode()).hexdigest(),
    }
    expected = recorded["matched_data_audit"]
    dataset = recorded.get("dataset", DATASET)
    trial = recorded.get("matched_source_trial", trial_number(dataset))
    mismatches = {key: {"expected": expected[key], "actual": actual[key]}
                  for key in AUDIT_KEYS if actual[key] != expected[key]}
    if mismatches:
        raise RuntimeError(f"Fine-tuning data differ from selected {dataset} trial {trial}: {mismatches}")
    return {"source_dataset": dataset, "source_trial": trial, "reference_layout": "dense_causal_match_prefix",
            "prefix_layout": "prefix_target", "all_fingerprints_equal": True, **actual}
