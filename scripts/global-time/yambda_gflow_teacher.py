"""Map the last-128 student catalog onto the global SASRec teacher's dense catalog."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def dense_teacher_ids(events, *, val_cutoff, student_vocab_size, num_users):
    """Recover the teacher's observed train catalog in pre-filter factorization order.

    Its disjoint dense windows cover every event of a user with at least two
    pre-cutoff positives. A singleton contributes neither an input nor a label.
    remap_interactions_to_catalog preserves all positions and original IDs,
    including tracks outside the student's shorter last-128 training histories.
    """
    users = events["user_id"].to_numpy()
    timestamps = events["timestamp"].to_numpy()
    positions = events["position"].to_numpy()
    prefilter = events["item_id_prefilter"].to_numpy()
    student_ids = events["item_id"].to_numpy()
    eligible = np.zeros(num_users, dtype=bool)
    observed = np.zeros(int(prefilter.max()) + 1, dtype=bool)
    student_to_prefilter = np.zeros(student_vocab_size, dtype=np.int64)

    # Bound temporary boolean/index arrays on the 72M-row positive stream.
    chunk = 1_000_000
    for start in range(0, len(events), chunk):
        rows = slice(start, start + chunk)
        second = (positions[rows] == 1) & (timestamps[rows] < val_cutoff)
        eligible[users[rows][second]] = True
    for start in range(0, len(events), chunk):
        rows = slice(start, start + chunk)
        in_window = (timestamps[rows] < val_cutoff) & eligible[users[rows]]
        observed[prefilter[rows][in_window]] = True
        student_to_prefilter[student_ids[rows]] = prefilter[rows]
    student_to_prefilter[0] = 0
    if not observed[student_to_prefilter[1:]].all():
        raise ValueError("The global SASRec training catalog is missing student items")
    # The original cold-item filter keeps first-occurrence ID order, closing gaps.
    return np.cumsum(observed)[student_to_prefilter], int(observed.sum()) + 1


def prepare_teacher_translation(dataset, dm, teacher_checkpoint, destination):
    from run_sasrec_all import build_yambda_data_config

    source = Path(__file__).resolve().parents[2] / "configs/global-time/exp2-tuned-params/dataset_yambda.yaml"
    cfg = build_yambda_data_config(dataset, yaml.safe_load(source.read_text()), seed=42)
    # Catalogs share the same positive stream before the model-specific train selection.
    for field in (
        "split_strategy", "global_val_timestamp", "global_test_timestamp", "cold_item_train_source",
        "filter_cold_items_after_split", "dataset_size", "dataset_type", "event_file", "max_rows",
        "hf_start_date", "hf_end_date", "hf_recent_months", "min_history_len", "positive_criterion",
        "min_user_interactions", "min_item_interactions", "deduplicate_consecutive",
        "drop_items_missing_embedding", "drop_items_missing_semantic_id", "min_played_ratio_pct",
        "allowed_event_types", "organic_filter",
    ):
        if getattr(cfg, field) != getattr(dm.cfg, field):
            raise ValueError(f"{dataset}: teacher and student input streams differ at {field}")
    if (cfg.split_strategy != "global_time" or cfg.cold_item_train_source != "observed"
            or not cfg.filter_cold_items_after_split or cfg.min_history_len != 1
            or cfg.train_example_layout != "dense_causal" or cfg.train_dense_match_prefix
            or (cfg.train_window_stride or cfg.max_history_len) != (cfg.train_window_len or cfg.max_history_len)):
        raise ValueError(f"{dataset}: unexpected global SASRec dense training layout")
    table, size = dense_teacher_ids(dm.interactions, val_cutoff=dm.global_val_cutoff,
                                    student_vocab_size=dm.item_vocab_size, num_users=dm.num_users)
    checkpoint = torch.load(teacher_checkpoint, mmap=True, map_location="cpu", weights_only=False)
    embedding = checkpoint["model_state_dict"]["item_embedding.weight"]
    if size != embedding.shape[0] or size != checkpoint["model_config"]["item_vocab_size"]:
        raise ValueError(f"{dataset}: reconstructed SASRec catalog has {size} rows, checkpoint has {embedding.shape[0]}")
    record = {
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_item_vocab_size": size,
        "target_catalog_sha256": hashlib.sha256(json.dumps(sorted(dm.item2id.items())).encode()).hexdigest(),
        "item_id_translation": table.tolist(),
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record) + "\n")
    temporary.replace(path)
    return path
