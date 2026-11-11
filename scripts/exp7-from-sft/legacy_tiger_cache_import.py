"""Import validated exp4 factors without changing the running experiment.

The old cache has no historical asset hashes. Compatibility is established
by the validator, and that limitation is retained in the imported manifest.
Only the new exp8 snapshot and metadata are generated. Source baseline
metrics are evaluated by the ordinary GPU phase, which reuses all factors.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
import shutil
import tempfile

import from_sft_runtime as runtime


def legacy_cache_root(session):
    return (runtime.ROOT / "artifacts/exp4-reward_ablation/gflow_tiger_from_sft"
            / session.source["dataset"] / "reward_factors")


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _check_scope(session):
    if (session.source["source_id"] not in ("tiger_beauty", "tiger_instruments")
            or session.source["model"] != "tiger" or session.args.smoke):
        raise ValueError("Legacy cache reuse supports only full TIGER Beauty/Instruments experiments")


def _import_component(session, dm, key, identity):
    from legacy_tiger_cache_validation import validate_component

    destination = session.cache_dirs()[key]
    origin = legacy_cache_root(session) / key
    if destination.resolve() == origin.resolve():
        raise ValueError("Legacy factors must remain separate from the exp8 cache directory")
    count = len(session.indices(dm))
    # Legacy training only reads these factors. The lock also prevents a
    # concurrent legacy --force-cache builder from replacing them mid-import.
    with ExitStack() as locks:
        locks.enter_context(session.base.control.file_lock(destination.parent / f"{key}.build.lock"))
        if (destination / "manifest.json").is_file():
            runtime.check_factor(session, key, count)
            print(f"Reusing existing exp8 factors: {destination}", flush=True)
            return
        if destination.exists():
            raise RuntimeError(f"Incomplete cache at {destination}; use a fresh --cache-root")
        if not (origin / "manifest.json").is_file() or not (origin / "factors.pt").is_file():
            raise FileNotFoundError(f"Missing legacy factors at {origin}; reuse never rebuilds reward factors")
        locks.enter_context(session.base.control.file_lock(origin.parent / f"{key}.build.lock"))
        manifest_before = (origin / "manifest.json").read_bytes()
        payload_hash = _digest(origin / "factors.pt")
        manifest = validate_component(session, dm, key, origin)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.import-", dir=destination.parent))
        try:
            # A private copy prevents a legacy in-place write from mutating
            # the imported factors; directory symlinks/hardlinks are avoided.
            shutil.copy2(origin / "factors.pt", temporary / "factors.pt")
            if (_digest(temporary / "factors.pt") != payload_hash
                    or (origin / "manifest.json").read_bytes() != manifest_before):
                raise RuntimeError(f"Legacy factors changed during import: {origin}")
            if runtime.input_identity(session) != identity:
                raise RuntimeError("Data/reward assets changed during legacy cache validation")
            config_path = session.cache_root / f"cache_config_{key}.json"
            runtime.write_json(config_path, session.base.cache_config(session.config, key, session.args))
            manifest = {**manifest, "config_path": str(config_path), "output_dir": str(destination),
                        "factors_path": "factors.pt", "imported_from": {
                            "directory": str(origin), "factor_sha256": payload_hash,
                            "manifest_sha256": hashlib.sha256(manifest_before).hexdigest(),
                            "historical_asset_hashes_available": False,
                        }}
            runtime.write_json(temporary / "manifest.json", manifest)
            runtime.write_json(temporary / "exp8_input_identity.json", identity)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        runtime.check_factor(session, key, count)
        print(f"Imported {key} factors without recomputation: {origin}", flush=True)


def _prepare_cpu(session):
    from prepared_data import save_prepared_data, snapshot_ready

    identity = runtime.input_identity(session)
    count = session.source["expected_counts"]["num_finetune_examples"]
    marker = session.cache_root / "prepared_cpu.json"
    with session.base.control.file_lock(session.cache_root / "legacy_import.lock"):
        if marker.is_file():
            previous = runtime.read_json(marker)
            if (previous.get("identity") != identity or previous.get("status") != "completed"
                    or previous.get("teacher_translation") is not None):
                raise RuntimeError(f"Preparation identity changed: {marker}")
            all_factors_present = True
            for key in runtime.COMPONENTS:
                if (session.cache_dirs()[key] / "manifest.json").is_file():
                    runtime.check_factor(session, key, count)
                else:
                    all_factors_present = False
            if all_factors_present and snapshot_ready(session):
                print(f"Already prepared with reusable factors: {marker}", flush=True)
                return
        dm = session.setup_data(allow_build=True)
        audit = session.audit_data(dm)
        audit["supervised_equivalence"] = runtime.compare_tiger_source(session, dm)
        for key in runtime.COMPONENTS:
            _import_component(session, dm, key, identity)
        save_prepared_data(session, dm)
        if runtime.input_identity(session) != identity:
            raise RuntimeError("Data/reward assets changed during legacy cache import")
        runtime.write_json(session.config_path, session.config)
        runtime.write_json(marker, {"identity": identity, "audit": audit, "status": "completed",
                                   "teacher_translation": None,
                                   "reward_factors": "validated reuse; no factor recomputation"})
        print(f"Prepared CPU snapshot and reused all rewards: {session.source['source_id']}", flush=True)


def prepare_reusing_legacy(session, kind):
    _check_scope(session)
    if kind not in ("cpu", "gpu", "all"):
        raise ValueError(f"Unknown preparation kind: {kind}")
    if kind in ("cpu", "all"):
        previous_device = session.args.device
        session.args.device = "cpu"
        try:
            _prepare_cpu(session)
        finally:
            session.args.device = previous_device
    if kind in ("gpu", "all"):
        # Refuse missing/corrupt imports BEFORE the ordinary preparer can
        # enter its factor builder. GPU work here is baseline evaluation only.
        runtime.require_prepared(session, kinds=("cpu",))
        for key in runtime.COMPONENTS:
            runtime.check_factor(session, key, session.source["expected_counts"]["num_finetune_examples"])
        def require_existing_factor(bound, dm, key):
            runtime.check_factor(bound, key, len(bound.indices(dm)))
        with runtime.bind(runtime, build_factor=require_existing_factor):
            runtime.prepare(session, "gpu")
