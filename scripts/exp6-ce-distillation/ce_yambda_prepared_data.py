"""Compact immutable Yambda inputs for CE workers, prepared once on CPU.

Only selected training prefixes, held-out prefixes, catalog metadata and
popularity counts are retained. Histories are flat mmap-backed tensors;
workers never deserialize the raw interactions or Python history lists.
The caller audits the full source pool and owns the lock around expensive
preprocessing. This module also locks publication of a snapshot generation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys
import time
import uuid
import warnings

import pandas as pd
import torch

from flowrec.data.datasets import DatasetStats, SequenceDataset
from flowrec.data.tensor_batches import SequenceExampleArrays
from flowrec.data.yambda import YambdaDataModule

SCHEMA_VERSION = 1
LOADER_FIELDS = frozenset({"batch_size", "eval_batch_size", "test_batch_size", "num_workers",
    "pin_memory", "persistent_workers", "train_drop_last", "include_raw_in_batch", "tensorized_batches"})
_SCALARS = {"user_id": "user_ids", "target_item_id": "target_item_ids",
    "target_rating": "target_ratings", "target_verified_purchase": "target_verified",
    "target_timestamp": "target_timestamps", "target_position": "target_positions"}
_ATTRIBUTES = ("user2id", "id2user", "item2id", "id2item", "semantic_ids", "sid_len",
    "train_item_counts_by_raw_id", "post_split_filter_stats", "dense_prefix_audit")


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(identity):
    return _json({"schema": SCHEMA_VERSION, "inputs": identity,
        "serializer_sha256": _hash_file(__file__),
        "runtime": {"python": list(sys.version_info[:2]),
                    **{name: version(name) for name in ("numpy", "pandas", "torch")}}})


def _compatible_identity(identity):
    """Separate the frozen data contract from informational build provenance.

    Existing schema-1 manifests contain code hashes and exact package versions.
    Those describe how the snapshot was built; changing a checkpoint resolver
    or updating a library does not change the examples already stored in it.
    Strip only these known fields, retaining all data/config/schema checks.
    """
    if not isinstance(identity, dict) or not isinstance(identity.get("inputs"), dict):
        raise RuntimeError("Invalid CE prepared-data identity")
    result = {key: value for key, value in identity.items()
              if key not in {"serializer_sha256", "runtime"}}
    result["inputs"] = {key: value for key, value in identity["inputs"].items()
                        if key not in {"code", "runtime"}}
    return result


def _different_fields(recorded, expected, prefix=""):
    if isinstance(recorded, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(recorded.keys() | expected.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in recorded or key not in expected:
                differences.append(path)
            else:
                differences.extend(_different_fields(recorded[key], expected[key], path))
        return differences
    return [] if recorded == expected else [prefix]


def _input_config(cfg):
    ignored = set(LOADER_FIELDS)
    if (cfg.split_strategy == "leave_one_out" and cfg.train_example_layout == "prefix_target"
            and cfg.train_target_policy == "last_n"):
        # This deterministic input protocol uses seed only for loader order;
        # the fixed sampled-target seed belongs to the caller's identity.
        ignored.add("seed")
    return _json({key: value for key, value in asdict(cfg).items() if key not in ignored})


@contextmanager
def _lock(path):
    import fcntl
    with Path(path).open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _record(path):
    stat = path.stat()
    return {"file": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": _hash_file(path)}


def _artifact(directory, record, *, verify_hash=False):
    name = record.get("file")
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise RuntimeError("Invalid CE prepared-data artifact filename")
    path = directory / name
    if not path.is_file():
        raise RuntimeError(f"Missing CE prepared-data artifact: {path}; rerun CPU preparation")
    stat = path.stat()
    if stat.st_size != record.get("bytes") or stat.st_mtime_ns != record.get("mtime_ns"):
        raise RuntimeError(f"CE prepared-data artifact changed: {path}; use a fresh preparation directory")
    if verify_hash and _hash_file(path) != record.get("sha256"):
        raise RuntimeError(f"CE prepared-data checksum mismatch: {path}")
    return path


def check(directory, identity, *, verify_hashes=False):
    """Return a valid committed manifest, or None when no snapshot exists."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable CE prepared-data manifest: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise RuntimeError(f"Incomplete CE prepared-data manifest: {path}")
    recorded, expected = manifest.get("identity"), _identity(identity)
    differences = _different_fields(_compatible_identity(recorded), _compatible_identity(expected))
    if differences:
        fields_changed = ", ".join(differences[:8])
        if len(differences) > 8:
            fields_changed += f", ... ({len(differences)} fields)"
        raise RuntimeError(f"CE prepared-data inputs changed: {path}; differing fields: "
                           f"{fields_changed}; use a fresh preparation directory")
    for name, record in manifest.get("artifacts", {}).items():
        _artifact(directory, record, verify_hash=verify_hashes or name == "metadata")
    if set(manifest.get("artifacts", {})) != {"metadata", "tensors"}:
        raise RuntimeError(f"Incomplete CE prepared-data manifest: {path}")
    if recorded != expected:
        warnings.warn(f"CE prepared-data code/runtime provenance differs: {path}; "
                      "data identity matches, reusing the stored examples. "
                      "Use a fresh preparation directory to apply preprocessing code changes.",
                      RuntimeWarning, stacklevel=2)
    return manifest


class CompactSequenceDataset(SequenceDataset):
    """SequenceDataset contract with numeric rows and flat mapped histories."""

    def __init__(self, arrays, *, id2item, id2user):
        self.split = arrays.split
        self._ce_prepared = True
        self._tensorized_arrays_cache = arrays
        self.examples = pd.DataFrame(
            {name: getattr(arrays, field).numpy() for name, field in _SCALARS.items()}, copy=False)
        self._records = None
        self._id2item, self._id2user = id2item, id2user

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        a = self._tensorized_arrays_cache
        lo, hi = int(a.history_offsets[index]), int(a.history_offsets[index + 1])
        items = a.history_items[lo:hi].tolist()
        user, target = int(a.user_ids[index]), int(a.target_item_ids[index])
        return {"user_id": user, "raw_user_id": self._id2user[user],
            "history_item_ids": items, "history_raw_items": [self._id2item[i] for i in items],
            "history_ratings": a.history_ratings[lo:hi].tolist(),
            "history_verified_purchase": a.history_verified[lo:hi].tolist(),
            "target_item_id": target, "target_raw_item": self._id2item[target],
            "target_rating": float(a.target_ratings[index]),
            "target_verified_purchase": bool(a.target_verified[index]),
            "target_timestamp": int(a.target_timestamps[index]),
            "target_position": int(a.target_positions[index]),
            "split": self.split, "example_index": index}


class PreparedYambdaDataModule(YambdaDataModule):
    """Read-only prepared inputs; setup must never fall back to raw data."""

    def setup(self, stage=None):
        if not getattr(self, "_ce_prepared", False) or not self._is_setup:
            raise RuntimeError("Prepared Yambda inputs are not initialized; run CPU preparation")

    def prepare_data(self):
        self.setup()

    def summary(self):
        summary = dict(self._ce_prepared_source_summary)
        summary.update(asdict(self.stats))
        summary["prepared_full_train_examples"] = self._ce_prepared_full_train_count
        return summary


def prepare(directory, dm, indices, identity, audit=None):
    """Publish a compact snapshot from an audited, initialized Yambda module."""
    if not isinstance(dm, YambdaDataModule) or not dm._is_setup:
        raise ValueError("CE preparation requires an initialized Yambda datamodule")
    if dm.cfg.train_example_layout != "prefix_target":
        raise ValueError("CE prepared inputs require prefix_target layout")
    if any(getattr(dm, split + "_dataset", None) is None for split in ("train", "val", "test")):
        raise ValueError("CE preparation requires all three dataset splits")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_count = len(dm.train_dataset)
    selected = (torch.arange(source_count, dtype=torch.long) if indices is None else
                torch.as_tensor(indices, dtype=torch.long, device="cpu"))
    if selected.ndim != 1 or (selected.numel() and
            (int(selected.min()) < 0 or int(selected.max()) >= source_count)):
        raise ValueError("CE preparation indices must be a one-dimensional valid source subset")
    selected_digest = hashlib.sha256(memoryview(selected.numpy()).cast("B")).hexdigest()
    with _lock(directory / "publish.lock"):
        existing = check(directory, identity)
        if existing is not None:
            if existing["input_config"] != _input_config(dm.cfg):
                raise RuntimeError("CE prepared-data input configuration differs")
            if (existing["full_train_count"] != source_count or
                    existing["source_indices_sha256"] != selected_digest):
                raise RuntimeError("CE prepared-data selected source indices differ")
            return existing
        started = time.perf_counter()
        arrays = {}
        for split in ("train", "val", "test"):
            frame = getattr(dm, split + "_dataset").examples
            if split == "train" and indices is not None:
                frame = frame.iloc[selected.numpy()]
            # Build only selected histories. Do not populate the full pool's
            # tensor cache, which would keep another large copy resident.
            built = SequenceExampleArrays.build(frame, split=split)
            arrays[split] = {field.name: getattr(built, field.name)
                             for field in fields(built) if field.name != "split"}
        attrs = {name: getattr(dm, name) for name in _ATTRIBUTES if hasattr(dm, name)}
        metadata = {"schema": SCHEMA_VERSION, "attributes": attrs, "audit": audit,
            "source_summary": dm.summary(), "stats": dm.stats,
            "full_train_count": source_count, "input_config": _input_config(dm.cfg)}
        generation = uuid.uuid4().hex
        meta_path, tensor_path = directory / f"{generation}.pkl", directory / f"{generation}.pt"
        manifest_path = directory / "manifest.json"
        temporary_manifest = directory / f"{generation}.manifest.tmp"
        committed = False
        try:
            with meta_path.open("xb") as stream:
                pickle.dump(metadata, stream, protocol=5)
                stream.flush(); os.fsync(stream.fileno())
            with tensor_path.open("xb") as stream:
                torch.save({"schema": SCHEMA_VERSION, "source_indices": selected, "arrays": arrays}, stream)
                stream.flush(); os.fsync(stream.fileno())
            manifest = {"status": "completed", "identity": _identity(identity),
                "input_config": metadata["input_config"], "full_train_count": source_count,
                "source_indices_sha256": selected_digest,
                "counts": {split: int(values["user_ids"].numel()) for split, values in arrays.items()},
                "artifacts": {"metadata": _record(meta_path), "tensors": _record(tensor_path)},
                "elapsed_sec": time.perf_counter() - started}
            with temporary_manifest.open("x") as stream:
                json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary_manifest, manifest_path)
            committed = True
            return manifest
        finally:
            if not committed:
                for path in (meta_path, tensor_path, temporary_manifest):
                    path.unlink(missing_ok=True)


def load(directory, cfg, identity):
    """Load compact mmap inputs, never reading raw interactions or setup()."""
    directory = Path(directory)
    manifest = check(directory, identity)
    if manifest is None:
        return None
    if manifest["input_config"] != _input_config(cfg):
        raise RuntimeError("CE prepared-data input configuration differs from this run")
    # Metadata are a locally generated pickle. Identity and its full checksum
    # are checked before deserializing; external snapshots are not accepted.
    with _artifact(directory, manifest["artifacts"]["metadata"]).open("rb") as stream:
        metadata = pickle.load(stream)
    tensors = torch.load(_artifact(directory, manifest["artifacts"]["tensors"]),
                         map_location="cpu", mmap=True, weights_only=True)
    if metadata.get("schema") != SCHEMA_VERSION or tensors.get("schema") != SCHEMA_VERSION:
        raise RuntimeError("CE prepared-data payload schema differs")
    if metadata["input_config"] != manifest["input_config"]:
        raise RuntimeError("CE prepared-data metadata config differs from manifest")
    dm = PreparedYambdaDataModule(cfg)
    for name, value in metadata["attributes"].items():
        if name not in _ATTRIBUTES:
            raise RuntimeError(f"Unexpected CE prepared-data attribute {name}")
        setattr(dm, name, value)
    for split in ("train", "val", "test"):
        arrays = SequenceExampleArrays(split=split, **tensors["arrays"][split])
        if arrays.num_examples != manifest["counts"][split]:
            raise RuntimeError(f"CE prepared-data {split} count differs from manifest")
        setattr(dm, split + "_dataset", CompactSequenceDataset(arrays, id2item=dm.id2item, id2user=dm.id2user))
    selected = tensors["source_indices"]
    if selected.ndim != 1 or selected.numel() != len(dm.train_dataset):
        raise RuntimeError("CE prepared-data selected-index provenance has incompatible length")
    if hashlib.sha256(memoryview(selected.numpy()).cast("B")).hexdigest() != manifest["source_indices_sha256"]:
        raise RuntimeError("CE prepared-data selected-index provenance checksum differs")
    if metadata["full_train_count"] != manifest["full_train_count"]:
        raise RuntimeError("CE prepared-data source pool count differs from manifest")
    dm._ce_prepared_source_indices = selected
    dm._ce_prepared_full_train_count = metadata["full_train_count"]
    dm._ce_prepared_source_summary = metadata["source_summary"]
    dm._ce_prepared_audit = metadata["audit"]
    dm._ce_prepared_manifest = str(directory / "manifest.json")
    counts = {split: len(getattr(dm, split + "_dataset")) for split in ("train", "val", "test")}
    means = {split: (float(tensors["arrays"][split]["history_offsets"][-1]) / count if count else 0.)
             for split, count in counts.items()}
    total = sum(counts.values())
    stats = metadata["stats"]
    if stats is None:
        stats = DatasetStats(num_users=dm.num_users, num_items=dm.num_items,
            num_train_examples=0, num_val_examples=0, num_test_examples=0, num_interactions=0)
    dm.stats = replace(stats, **{f"num_{split}_examples": count for split, count in counts.items()},
        **{f"mean_{split}_history_len": mean for split, mean in means.items()},
        mean_history_len=sum(means[s] * counts[s] for s in counts) / total if total else 0.)
    dm._ce_prepared = dm._is_setup = True
    return dm
