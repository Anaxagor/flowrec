"""Accept a verified checkpoint relocation without relaxing CE run identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# Exact protocol before the two publication destinations were reorganized.
PREVIOUS_PROTOCOL_SHA256 = "47d2477b17c0dcf333f69b1f2dd9555f067ca4d6e51a2faaa24f63af932d2a07"
ARCHIVE = Path("artifacts/checkpoint_archive/2026-10-30")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_relative_to(root):
        return path.relative_to(root)
    if path.is_absolute() and "/flowrec/" in path.as_posix():
        return Path(path.as_posix().split("/flowrec/", 1)[1])
    return path


def _logical(value: str, root: Path) -> Path:
    relative = _relative(value, root)
    if relative.is_relative_to(ARCHIVE):
        relative = Path("checkpoints") / relative.relative_to(ARCHIVE)
    return relative


def _flat_initial_name(value: str, root: Path) -> Path | None:
    relative = _relative(value, root)
    for prefix in ("checkpoints/exp2/loo_split/initial", "checkpoints/exp2/initial",
                   "checkpoints/exp2-tuned_params/initial"):
        if not relative.is_relative_to(prefix):
            continue
        parts = list(relative.relative_to(prefix).parts)
        if parts[:2] == ["tiger", "yambda_50m"]:
            parts[1] = "yambda_50m_listens"
            if parts[2:3] == ["checkpoints"]:
                del parts[2]
        if len(parts) == 3 and parts[0] == "gpt2" and parts[1].startswith("yambda_"):
            regime = "full_history" if prefix == "checkpoints/exp2-tuned_params/initial" else "last_128"
            parts.insert(2, regime)
        if len(parts) in (3, 4):
            parts[-1] = {"best_model.pt": "best.pt", "best_model.pt.json": "best.pt.json"}.get(parts[-1], parts[-1])
            return Path(*parts)
    return None


def _protocol_path_only_change(saved_hash: str, current_hash: str, protocol_path: Path) -> bool:
    if saved_hash != PREVIOUS_PROTOCOL_SHA256 or _sha256(protocol_path) != current_hash:
        return False
    restored = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        arm = restored["arms"]["tiger_sft_only"]
        arm["promote_root"] = "checkpoints/exp2-tuned_params/finetuned_matched/tiger"
        arm["dataset_roots"]["yambda_500m_listens"]["promote_root"] = "checkpoints/exp2-tuned_params/finetuned_matched_r15/tiger"
    except KeyError:
        return False
    encoded = (json.dumps(restored, indent=2) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest() == saved_hash


def same_after_checkpoint_relocation(saved: Any, current: Any, *, root: Path, protocol_path: Path) -> bool:
    """Keep every scientific value and asset byte/mtime record exact.

    A changed checkpoint path must lead to a selected file whose provenance
    and actual bytes prove its identity. Historical configs are never edited.
    """
    checked: dict[tuple[str, str], bool] = {}

    def path_alias(old: str, new: str) -> bool:
        key = (old, new)
        if key in checked:
            return checked[key]
        checked[key] = False
        relative = _relative(new, root)
        if not relative.is_relative_to("checkpoints/exp2/loo_split/initial"):
            return False
        destination = root / relative
        provenance_path = destination.parent / "provenance.json"
        if not destination.is_file() or not provenance_path.is_file():
            return False
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        source = (provenance.get("source_checkpoint") if destination.name == "best.pt"
                  else provenance.get("metadata", {}).get(destination.name))
        if not isinstance(source, str):
            return False
        old_flat, new_flat = _flat_initial_name(old, root), _flat_initial_name(new, root)
        if _logical(old, root) != _logical(source, root) and not (
                old_flat is not None and old_flat == new_flat):
            return False
        actual = _sha256(destination)
        if destination.name == "best.pt":
            valid = actual == provenance.get("sha256")
        else:
            valid = actual == provenance.get("metadata_sha256", {}).get(destination.name)
        checked[key] = valid
        return valid

    def equal(old: Any, new: Any, key: str = "") -> bool:
        if old == new:
            return True
        if key == "protocol_sha256" and isinstance(old, str) and isinstance(new, str):
            return _protocol_path_only_change(old, new, protocol_path)
        if isinstance(old, str) and isinstance(new, str):
            return path_alias(old, new)
        if isinstance(old, dict) and isinstance(new, dict) and len(old) == len(new):
            remaining = set(new)
            for old_key, value in old.items():
                matches = ([old_key] if old_key in remaining else
                           [candidate for candidate in remaining if path_alias(old_key, candidate)])
                if len(matches) != 1 or not equal(value, new[matches[0]], old_key):
                    return False
                remaining.remove(matches[0])
            return not remaining
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            return all(equal(a, b) for a, b in zip(old, new))
        return False

    return equal(saved, current)
