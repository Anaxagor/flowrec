"""Stage immutable training inputs on scratch before opening a memory map."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil


def stage_mmap_file(source, scratch_dir=None, *, expected_sha256=None):
    """Return the canonical path or a complete, generation-specific scratch copy.

    Callers keep using the canonical path for their manifests and run identity.
    A lock serializes parallel jobs/worker processes filling the same cache.
    """
    source = Path(source)
    if scratch_dir is None:
        return source

    import fcntl

    scratch_dir = Path(scratch_dir)
    if not scratch_dir.is_absolute():
        raise ValueError(f"FLOWREC_CE_SCRATCH_DIR must be an absolute path: {scratch_dir}")
    source = source.resolve(strict=True)
    before = source.stat()
    generation = (str(source), before.st_size, before.st_mtime_ns, expected_sha256)
    key = hashlib.sha256(json.dumps(generation).encode()).hexdigest()
    directory = scratch_dir / "ce-mmap-v1" / key
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size == before.st_size:
            return destination
        # Reuse this name under the lock so an interrupted copy does not leave
        # a growing set of multi-GB temporary files on subsequent retries.
        temp = directory / ".copying"
        temp.unlink(missing_ok=True)
        available = shutil.disk_usage(directory).free
        if before.st_size > available:
            raise OSError(f"Insufficient scratch space at {directory}: {before.st_size} bytes "
                          f"needed to stage {source}, {available} bytes free")
        print(f"[CE scratch] copying {before.st_size:,} bytes: {source} -> {destination}", flush=True)
        digest = hashlib.sha256() if expected_sha256 is not None else None
        try:
            with source.open("rb") as src, temp.open("xb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
                    if digest is not None:
                        digest.update(chunk)
            after = source.stat()
            if (after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns
                    or after.st_ino != before.st_ino or temp.stat().st_size != before.st_size):
                raise RuntimeError(f"Source changed or short copy while staging {source}")
            if digest is not None and digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"Prepared input checksum mismatch while staging {source}")
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
    return destination
