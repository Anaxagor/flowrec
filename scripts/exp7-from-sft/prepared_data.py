"""Persist audited CPU preprocessing for subsequent GPU array tasks.

The datamodule owns DataFrames, ID maps and datasets, not live DataLoaders or
collators. Protocol-5 pickle streams the Python graph; contiguous NumPy/CPU
tensor buffers go to a separate file and reload through a private writable
mmap. This avoids torch.save's large in-memory pickle buffer and repeated
history flattening on the GPU nodes. Only locally generated cache artifacts
with matching input/code/library identities are loaded.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
from importlib.metadata import version
import json
import mmap
import os
from pathlib import Path
import pickle
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
LOADER_FIELDS = frozenset({"batch_size", "eval_batch_size", "test_batch_size", "num_workers",
    "pin_memory", "persistent_workers", "train_drop_last", "include_raw_in_batch", "tensorized_batches"})


def cache_directory(session):
    return Path(session.cache_root) / "prepared_data"


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity():
    paths = [*sorted((ROOT / "flowrec/data").rglob("*.py")), ROOT / "scripts/run_tiger.py",
             ROOT / "scripts/exp7-from-sft/sft_source_protocol.py",
             ROOT / "scripts/exp7-from-sft/from_sft_runtime.py", Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): _hash_file(path) for path in paths}


def snapshot_identity(session):
    from from_sft_runtime import input_identity
    return _json({"schema": SCHEMA_VERSION, "inputs": input_identity(session),
                  "preprocessing_code": code_identity(),
                  "runtime": {"python": list(sys.version_info[:2]),
                              **{name: version(name) for name in ("numpy", "pandas", "torch")}}})


def _input_config(cfg):
    if not is_dataclass(cfg):
        raise TypeError("Prepared datamodule config must be a dataclass")
    return _json({key: value for key, value in asdict(cfg).items() if key not in LOADER_FIELDS})


def _fresh_config(session):
    return session.base.tiger_runner.build_data_config(deepcopy(session.config), config_path=session.config_path)


def _check_setup(dm):
    if not getattr(dm, "_is_setup", False) or any(
        getattr(dm, name, None) is None for name in ("train_dataset", "val_dataset", "test_dataset")
    ):
        raise RuntimeError("A prepared-data snapshot requires all three fully initialized splits")


def _warm_tensor_arrays(dm):
    if not getattr(dm.cfg, "tensorized_batches", False) or dm.cfg.include_raw_in_batch:
        return
    from flowrec.data.tensor_batches import SequenceExampleArrays
    for split in ("train", "val", "test"):
        if split == "train" and dm.cfg.train_example_layout != "prefix_target":
            continue
        SequenceExampleArrays.for_dataset(getattr(dm, split + "_dataset"))


def _snapshot_view(dm):
    """Shallow views retain source data and omit disposable lookup caches."""
    result = copy(dm)
    for key in tuple(vars(result)):
        if key.startswith("_exp8_prepared_"):
            delattr(result, key)
    for name in ("train_dataset", "val_dataset", "test_dataset"):
        dataset = copy(getattr(dm, name))
        dataset._records = None  # Per-worker Python records can dwarf the frame.
        setattr(result, name, dataset)
    if dm.interaction_store is not None:
        result.interaction_store = copy(dm.interaction_store)
        for key in vars(result.interaction_store):
            if key.endswith("_cache"):
                setattr(result.interaction_store, key, {})
    return result


def _tensor_from_numpy(array):
    import torch
    return torch.from_numpy(array)


class _StreamingPickler(pickle.Pickler):
    def reducer_override(self, obj):
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "cpu" or obj.requires_grad or obj.layout != torch.strided:
                raise ValueError("Prepared data may only contain ordinary CPU tensors without gradients")
            # NumPy protocol5 exports this storage as a buffer instead of
            # PyTorch's default nested torch.save/BytesIO representation.
            return _tensor_from_numpy, (obj.detach().numpy(),)
        return NotImplemented


class _DigestWriter:
    def __init__(self, stream):
        self.stream, self.digest, self.size = stream, hashlib.sha256(), 0

    def write(self, data):
        view = memoryview(data).cast("B")
        for start in range(0, len(view), 8 * 1024 * 1024):
            chunk = view[start:start + 8 * 1024 * 1024]
            written = self.stream.write(chunk)
            if written != len(chunk):
                raise OSError("Short prepared-data artifact write")
            self.digest.update(chunk)
            self.size += written
        return len(view)


def _artifact_record(path, digest):
    stat = path.stat()
    return {"file": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest}


def _artifact_path(directory, record):
    name = record.get("file")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise RuntimeError("Invalid prepared-data artifact filename")
    path = directory / name
    if not path.is_file():
        raise RuntimeError(f"Prepared-data artifact is missing: {path}; rerun CPU preparation")
    stat = path.stat()
    if stat.st_size != record.get("bytes") or stat.st_mtime_ns != record.get("mtime_ns"):
        raise RuntimeError(f"Prepared-data artifact changed: {path}; rerun CPU preparation with a fresh cache root")
    return path


def check_prepared_data(session, *, verify_hashes=False):
    """Cheap preflight; optional full streaming hashes for explicit audits."""
    directory = cache_directory(session)
    path = directory / "manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable prepared-data manifest: {path}") from exc
    if manifest.get("status") != "completed" or manifest.get("identity") != snapshot_identity(session):
        raise RuntimeError(f"Prepared data inputs/code/runtime changed: {path}; use a fresh cache root and CPU preparation")
    for kind in ("state", "buffers"):
        artifact = _artifact_path(directory, manifest[kind])
        if verify_hashes and _hash_file(artifact) != manifest[kind]["sha256"]:
            raise RuntimeError(f"Prepared-data checksum mismatch: {artifact}")
    return manifest


def snapshot_ready(session):
    """Preflight without unpickling or constructing a datamodule."""
    return check_prepared_data(session) is not None


def save_prepared_data(session, dm):
    """Save once, before factor construction, without copying large frames."""
    _check_setup(dm)
    expected_config = _fresh_config(session)
    if type(dm.cfg) is not type(expected_config) or _input_config(dm.cfg) != _input_config(expected_config):
        raise RuntimeError("Prepared datamodule does not match the source input configuration")
    directory = cache_directory(session)
    directory.mkdir(parents=True, exist_ok=True)
    with session.base.control.file_lock(directory / "build.lock"):
        existing = check_prepared_data(session)
        if existing is not None:
            return directory / existing["state"]["file"]
        started = time.perf_counter()
        identity = snapshot_identity(session)
        _warm_tensor_arrays(dm)
        generation = uuid.uuid4().hex
        state_path = directory / (generation + ".pkl")
        buffer_path = directory / (generation + ".buffers")
        records = []
        committed = False
        try:
            with state_path.open("xb") as state_stream, buffer_path.open("xb") as buffer_stream:
                state_writer, buffer_writer = _DigestWriter(state_stream), _DigestWriter(buffer_stream)

                def buffer_callback(buffer):
                    view = buffer.raw()
                    padding = (-buffer_writer.size) % 64
                    if padding:
                        buffer_writer.write(bytes(padding))
                    records.append([buffer_writer.size, view.nbytes])
                    buffer_writer.write(view)
                    return False

                _StreamingPickler(state_writer, protocol=5, buffer_callback=buffer_callback).dump(
                    {"schema": SCHEMA_VERSION, "dm": _snapshot_view(dm)})
                for stream in (state_stream, buffer_stream):
                    stream.flush()
                    os.fsync(stream.fileno())
            if snapshot_identity(session) != identity:
                raise RuntimeError("Prepared-data inputs/code changed while saving; rerun CPU preparation")
            manifest = {"status": "completed", "identity": identity,
                "state": _artifact_record(state_path, state_writer.digest.hexdigest()),
                "buffers": _artifact_record(buffer_path, buffer_writer.digest.hexdigest()),
                "buffer_ranges": records, "datamodule_class": type(dm).__module__ + "." + type(dm).__qualname__,
                "input_config": _input_config(dm.cfg), "elapsed_sec": time.perf_counter() - started}
            # Generation files are private until this atomic commit. A reader
            # can never combine metadata from one save with another payload.
            from from_sft_runtime import write_json
            write_json(directory / "manifest.json", manifest)
            committed = True
            print(f"[prepared data] saved {state_path.stat().st_size + buffer_path.stat().st_size:,} bytes "
                  f"in {manifest['elapsed_sec']:.1f}s", flush=True)
            return state_path
        finally:
            if not committed:
                state_path.unlink(missing_ok=True)
                buffer_path.unlink(missing_ok=True)


def load_prepared_data(session):
    """Load the committed snapshot; never call DataModule.setup()."""
    manifest = check_prepared_data(session)
    if manifest is None:
        return None
    directory = cache_directory(session)
    state_path = _artifact_path(directory, manifest["state"])
    buffer_path = _artifact_path(directory, manifest["buffers"])
    fresh_cfg = _fresh_config(session)
    if manifest["input_config"] != _input_config(fresh_cfg):
        raise RuntimeError("Prepared datamodule input configuration differs from this stage")
    started = time.perf_counter()
    mapped = None
    if manifest["buffers"]["bytes"]:
        with buffer_path.open("rb") as stream:
            mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    buffers = []
    for offset, size in manifest["buffer_ranges"]:
        if offset < 0 or size < 0 or offset + size > manifest["buffers"]["bytes"]:
            raise RuntimeError("Prepared-data buffer range exceeds its committed artifact")
        buffers.append(memoryview(mapped)[offset:offset + size] if mapped is not None else memoryview(b""))
    # This is a private generated cache, not an uploaded or external pickle.
    # Its schema, inputs, code, versions and artifact records are checked first.
    with state_path.open("rb") as stream:
        payload = pickle.Unpickler(stream, buffers=buffers).load()
    dm = payload["dm"]
    if payload.get("schema") != SCHEMA_VERSION or type(dm).__module__ + "." + type(dm).__qualname__ != manifest["datamodule_class"]:
        raise RuntimeError("Prepared-data payload has an incompatible datamodule/schema")
    _check_setup(dm)
    if type(dm.cfg) is not type(fresh_cfg) or _input_config(dm.cfg) != manifest["input_config"]:
        raise RuntimeError("Prepared-data payload config differs from its manifest")
    # SequenceDataset owns no cfg, and both datamodules create collators from
    # self.cfg on demand. Updating these operational fields therefore updates
    # every newly created loader without changing cached examples or IDs.
    names = {field.name for field in fields(fresh_cfg)}
    dm.cfg = replace(dm.cfg, **{key: getattr(fresh_cfg, key) for key in LOADER_FIELDS & names})
    dm._exp8_prepared_mapping = mapped  # Keep the external-buffer mapping alive.
    dm._exp8_prepared_manifest = str(directory / "manifest.json")
    print(f"[prepared data] loaded {session.source['source_id']} in {time.perf_counter() - started:.1f}s", flush=True)
    return dm
