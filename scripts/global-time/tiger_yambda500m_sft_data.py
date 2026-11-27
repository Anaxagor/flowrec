"""Prepare and reload the original global Yambda 500M training rows of TIGER dropout-0."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exp7-ce-distillation"))
sys.path.insert(0, str(ROOT / "scripts/exp8-from-sft"))

from flowrec.data.yambda import YambdaDataConfig, YambdaDataModule
from flowrec.mmap_scratch import stage_mmap_file
from bounded_preparation import bounded_history_build
import ce_yambda_prepared_data as compact


ASSETS = {
    "interactions_path": Path("data/raw/yambda/flat/500m/multi_event.parquet"),
    "embeddings_path": Path("data/raw/yambda/embeddings.parquet"),
    "semantic_id_path": Path("datasets/semantic_ids/exp2-tuned-params/yambda_500m_listens/semantic_ids.json"),
}
# From the global dropout-0 study's data_summary.json.
COUNTS = {"train": 9_560_668, "val": 71_867, "test": 70_663}
NUM_ITEMS = 371_818
CUTOFFS = (25_001_165, 25_510_950)


def _inputs(workspace, data_config, num_workers):
    sources = {}
    versions = {}
    for key, relative in ASSETS.items():
        path = Path(data_config[key])
        # Some copies of the trial record the original submitter's absolute checkout.
        if path.is_absolute() and path.parts[-len(relative.parts):] == relative.parts:
            path = relative
        path = (ROOT / path).resolve(strict=True)
        sources[key] = path
        stat = path.stat()
        versions[key] = [stat.st_size, stat.st_mtime_ns]
    cfg = YambdaDataConfig(**{**data_config, **sources, "num_workers": num_workers,
                              "persistent_workers": num_workers > 0, "include_raw_in_batch": False})
    if (cfg.split_strategy != "global_time" or (cfg.global_val_timestamp, cfg.global_test_timestamp) != CUTOFFS
            or cfg.dataset_size != "500m" or cfg.train_target_policy != "last_n" or cfg.train_last_n != 128
            or cfg.train_example_layout != "prefix_target" or cfg.max_history_len != 50 or cfg.train_prefix_from_dense
            or not cfg.filter_cold_items_after_split or cfg.cold_item_train_source != "observed"):
        raise ValueError("Expected the original TIGER global Yambda 500M last-128 training split")
    identity = {"protocol": "global_tiger_yambda500m_dropout0_original_train_v1",
                "assets": versions, "data": compact._input_config(cfg)}
    directory = Path(workspace).resolve() / "tiger_yambda500m_global_dropout_0" / "train"
    return directory, cfg, identity


def _check_counts(counts, full_train_count):
    if counts != COUNTS or full_train_count != COUNTS["train"]:
        raise ValueError(f"Expected all original dropout-0 splits {COUNTS}, found {counts}, pool={full_train_count}")


def prepare(workspace, data_config, *, num_workers=0):
    """Build native prefix-target examples on CPU and commit every original row to scratch."""
    directory, cfg, identity = _inputs(workspace, data_config, num_workers)
    directory.mkdir(exist_ok=True, parents=True)
    with compact._lock(directory / "cpu-prepare.lock"):
        manifest = compact.check(directory, identity)
        if manifest is not None:
            _check_counts(manifest["counts"], manifest["full_train_count"])
            return
        staged = {key: stage_mmap_file(getattr(cfg, key), directory.parent / "inputs") for key in ASSETS}
        dm = YambdaDataModule(replace(cfg, **staged))
        with bounded_history_build(50_000):
            dm.setup()
        counts = {split: len(getattr(dm, split + "_dataset")) for split in COUNTS}
        _check_counts(counts, len(dm.train_dataset))
        if dm.num_items != NUM_ITEMS or (dm.global_val_cutoff, dm.global_test_cutoff) != CUTOFFS:
            raise ValueError("The prepared catalog or time boundaries differ from the dropout-0 study")
        latest_target = int(dm.train_dataset.examples.target_timestamp.max())
        if latest_target >= dm.global_val_cutoff:
            raise ValueError(f"Training target {latest_target} crosses the validation boundary")
        summary = dm.summary()
        # Keep the source identity stable across compute nodes. All training tensors and
        # item/user/SID mappings are self-contained once the compact manifest is committed.
        dm.cfg = cfg
        dm.interactions = dm.interaction_store = dm.embedding_item_ids = None
        compact.prepare(directory, dm, None, identity,
                        audit={"latest_training_target": latest_target}, source_summary=summary)
        print(f"Prepared global dropout-0 splits {counts}, latest training target {latest_target}", flush=True)


def load(workspace, data_config, *, num_workers=2):
    """Open the original training and validation loaders from the CPU-prepared dataset."""
    directory, cfg, identity = _inputs(workspace, data_config, num_workers)
    dm = compact.load(directory, cfg, identity)
    if dm is None:
        raise FileNotFoundError(f"Missing CPU-prepared dropout-0 data at {directory}")
    counts = {split: len(getattr(dm, split + "_dataset")) for split in COUNTS}
    _check_counts(counts, dm._ce_prepared_full_train_count)
    return dm
