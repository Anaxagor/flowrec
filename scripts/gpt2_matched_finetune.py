"""Recorded input protocol for the explicitly selected Yambda 50M trial 36."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

DATASET = "yambda_50m_listens"
TRIAL = 36
TRIAL_DIRECTORY = Path("checkpoints/exp2-tuned_params/gpt2_tiger_matched") / DATASET / "trials/trial_36"
EXPECTED_ARCHITECTURE = {"embedding_dim": 768, "num_layers": 4, "num_heads": 2,
                         "dim_feedforward": 3072}
AUDIT_KEYS = ("num_train_targets", "num_items", "num_val_examples", "num_test_examples",
              "train_targets_sha256", "train_contexts_sha256", "validation_sha256",
              "test_sha256", "catalog_sha256", "semantic_ids_sha256")


def selected(dataset: str, args: Any = None) -> bool:
    return dataset == DATASET and getattr(args, "yambda50_base", "legacy") == "trial36"


def trial_paths(project_root: Path) -> dict[str, Path]:
    directory = project_root / TRIAL_DIRECTORY
    return {"checkpoint": directory / "best.pt", "sidecar": directory / "best.pt.json",
            "resolved_config": directory / "resolved_config.json",
            "data_audit": directory.parent.parent / "data_audit.json"}


def load_recorded(project_root: Path) -> dict[str, Any]:
    paths = trial_paths(project_root)
    path = paths["resolved_config"]
    if not path.is_file():
        raise FileNotFoundError(f"Selected matched trial 36 requires {path}; no fallback to another checkpoint")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("dataset") != DATASET:
        raise ValueError(f"{path} is not a {DATASET} trial")
    architecture = recorded.get("model_config", {})
    for key, expected in EXPECTED_ARCHITECTURE.items():
        if architecture.get(key) != expected:
            raise ValueError(f"Selected trial 36 has unexpected {key}: {architecture.get(key)!r}, expected {expected}")
    data = recorded.get("data_config", {})
    expected_data = {"train_dense_match_prefix": True, "train_example_layout": "dense_causal",
                     "train_prefix_from_dense": False, "split_strategy": "leave_one_out",
                     "train_target_policy": "last_n", "train_last_n": 128,
                     "max_history_len": 50, "train_window_len": 50, "train_window_stride": 1}
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ValueError(f"Selected trial 36 requires {key}={expected!r}, got {data.get(key)!r}")
    if "dropout" not in recorded.get("hparams", {}):
        raise ValueError("Selected trial 36 lacks its recorded dropout")
    audit = (recorded.get("resume_identity") or {}).get("data_audit")
    if audit is None and paths["data_audit"].is_file():
        audit = json.loads(paths["data_audit"].read_text(encoding="utf-8"))
    if not audit or any(key not in audit for key in AUDIT_KEYS):
        raise ValueError("Selected trial 36 requires its full saved data audit, including training histories")
    recorded["matched_data_audit"] = {key: audit[key] for key in AUDIT_KEYS}
    return recorded


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
    mismatches = {key: {"expected": expected[key], "actual": actual[key]}
                  for key in AUDIT_KEYS if actual[key] != expected[key]}
    if mismatches:
        raise RuntimeError(f"Fine-tuning data differ from selected trial 36: {mismatches}")
    return {"source_trial": TRIAL, "reference_layout": "dense_causal_match_prefix",
            "prefix_layout": "prefix_target", "all_fingerprints_equal": True, **actual}
