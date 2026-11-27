"""Prepare the original global Beauty 2023 GPT-2 test split on CPU."""
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path

import torch

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.data.tensor_batches import SequenceExampleArrays
from flowrec.mmap_scratch import stage_mmap_file

compact = import_module("scripts.exp7-ce-distillation.ce_yambda_prepared_data")
bounded_history_build = import_module("scripts.exp8-from-sft.bounded_preparation").bounded_history_build
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ("reviews_path", "metadata_path", "semantic_id_path")
SID_SHA256 = "5433d3ffe96629079ad06f46511f34e155cb0d0926cc8d19420e0f409ae843c9"
STATE = ("id2item", "item2id", "id2user", "user2id", "semantic_ids", "sid_len",
         "train_item_counts_by_raw_id", "stats", "global_val_cutoff", "global_test_cutoff",
         "post_split_filter_stats")


def _inputs(source_dir, prepared_dir):
    source = Path(source_dir)
    data = json.loads((source / "study_config.json").read_text())["data_config"]
    summary = json.loads((source / "data_summary.json").read_text())
    cfg = AmazonReviewDataConfig(**{**data, "metadata_ids_only": True})
    if (cfg.split_strategy != "global_time" or cfg.global_history_policy != "rolling"
            or cfg.global_eval_target_mode != "last_per_user" or cfg.train_example_layout != "dense_causal"
            or cfg.global_val_timestamp != summary["global_val_cutoff"]
            or cfg.global_test_timestamp != summary["global_test_cutoff"]):
        raise ValueError("Expected the original Beauty 2023 GPT-2 global split and cutoffs")
    assets = {key: (ROOT / getattr(cfg, key)).resolve(strict=True) for key in ASSETS}
    signatures = {}
    for key, path in assets.items():
        stat = path.stat()
        signatures[key] = [str(path), stat.st_size, stat.st_mtime_ns]
    identity = {"format": 1, "assets": signatures, "data": data, "source_summary": summary}
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cfg, assets, summary, Path(prepared_dir) / f"test-{digest}.pt"


def prepare(source_dir, prepared_dir, scratch_dir):
    """Audit all native splits, then store just the frozen test rows and catalog."""
    cfg, assets, expected, target = _inputs(source_dir, prepared_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with compact._lock(target.with_suffix(".lock")):
        if target.is_file():
            return target
        staged = {key: stage_mmap_file(path, scratch_dir,
                                       expected_sha256=SID_SHA256 if key == "semantic_id_path" else None)
                  for key, path in assets.items()}
        dm = AmazonReviewDataModule(replace(cfg, **staged, processed_cache_dir=None))
        with bounded_history_build(50_000):
            dm.setup()
        actual = dm.summary()
        for name in ("global_val_cutoff", "global_test_cutoff", "num_train_examples", "num_val_examples",
                     "num_test_examples", "num_interactions", "num_users", "num_items", "sid_len",
                     "positive_criterion", "post_split_filter"):
            if actual[name] != expected[name]:
                raise ValueError(f"Global Beauty 2023 {name}: built {actual[name]}, expected {expected[name]}")
        timestamps = dm.test_dataset.examples.target_timestamp
        if timestamps.empty or timestamps.min() < dm.global_test_cutoff or not dm.train_item_counts_by_raw_id:
            raise ValueError("Invalid test timestamps or missing training-split popularity")
        state = {name: getattr(dm, name) for name in STATE}
        state["test_arrays"] = SequenceExampleArrays.for_dataset(dm.test_dataset)
        temporary = target.with_suffix(".tmp")
        try:
            torch.save(state, temporary, pickle_protocol=4)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def load(source_dir, prepared_dir, scratch_dir, *, num_workers=2, test_batch_size=None):
    """Open all test examples on scratch without rebuilding any splits."""
    cfg, _, expected, target = _inputs(source_dir, prepared_dir)
    if not target.is_file():
        raise FileNotFoundError(f"Missing CPU-prepared global Beauty 2023 test split: {target}")
    local = stage_mmap_file(target, scratch_dir)
    state = torch.load(local, weights_only=False, map_location="cpu", mmap=True)
    dm = AmazonReviewDataModule(replace(cfg, num_workers=num_workers, persistent_workers=num_workers > 0,
                                        test_batch_size=test_batch_size or cfg.test_batch_size))
    for name in STATE:
        setattr(dm, name, state[name])
    arrays = state["test_arrays"]
    if arrays.split != "test" or arrays.num_examples != expected["num_test_examples"]:
        raise ValueError("Prepared test rows differ from the archived Beauty 2023 split")
    dm.test_dataset = compact.CompactSequenceDataset(arrays, id2item=dm.id2item, id2user=dm.id2user)
    dm._is_setup = True
    return dm
