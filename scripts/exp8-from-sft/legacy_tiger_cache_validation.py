"""Read-only validation before importing legacy TIGER reward factors.

The legacy format has no historical hashes of its raw data or reward-model
assets. We preserve that limitation in the returned provenance: full row and
catalogue checks plus current-model probes establish compatibility, but cannot
prove that every historical input byte was unchanged. No cache is rebuilt,
no datamodule is set up, and no file is written by this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "scripts", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import precompute_reward_factors as factors
from flowrec.data.dense_prefix import dense_prefix_signature
from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache
from prepared_data import LOADER_FIELDS
from sft_source_protocol import relocate_path


def require(condition, message):
    if not condition:
        raise ValueError("Legacy factor import: " + message)


def normalized(value):
    """Resolve recorded repository paths, preserving every semantic setting."""
    value = json.loads(json.dumps(value, default=str))
    if isinstance(value, dict):
        return {key: (str(Path(relocate_path(item, ROOT)).resolve())
                      if key.endswith("_path") and isinstance(item, str) and item else normalized(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def data_settings(cfg):
    return normalized({key: value for key, value in cfg.items() if key not in LOADER_FIELDS})


def file_record(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def equal_tensor(actual, expected, label):
    require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
            and actual.dtype == expected.dtype and torch.equal(actual.cpu(), expected.cpu()),
            f"{label} differs from the current source data/catalogue")


def close_tensor(actual, expected, label, *, atol, rtol):
    require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
            and bool(torch.isfinite(actual).all())
            and torch.allclose(actual.cpu().float(), expected.cpu().float(), atol=atol, rtol=rtol),
            f"{label} differs from the current reward model")


def full_row_check(session, dm, cache):
    frame = dm.train_dataset.examples
    mapping = session.base.tiger_runner.item_id_to_semantic_row(dm).cpu().long()
    item_tokens = torch.tensor([dm.semantic_ids[key] for key in sorted(dm.semantic_ids)], dtype=torch.long)
    target = torch.from_numpy(frame["target_item_id"].to_numpy(dtype="int64"))
    require(bool(((target >= 0) & (target < mapping.numel())).all()), "invalid target item IDs")
    rows = mapping[target]
    require(bool(((rows >= 0) & (rows < len(item_tokens))).all()), "targets missing from semantic catalogue")
    expected = {
        "user_ids": torch.from_numpy(frame["user_id"].to_numpy(dtype="int64")),
        "target_item_ids": target,
        "target_timestamps": torch.from_numpy(frame["target_timestamp"].to_numpy(dtype="int64")),
        "target_semantic_row_ids": rows,
        "target_token_ids": item_tokens[rows],
    }
    for key, values in expected.items():
        equal_tensor(cache.factors.get(key), values, f"all {key} rows")
    require(cache.num_items == len(item_tokens) and cache.manifest.get("sid_len") == dm.sid_len,
            "catalogue size or semantic-ID length differs")
    return item_tokens


def validate_component(session, dm, name, directory):
    """Return a manifest copy with ``legacy_validation`` provenance.

    The caller provides the already initialized source datamodule. Supported
    inputs are full, non-smoke Beauty/Instruments TIGER factors. Verification
    uses CPU only; three distributed examples probe reward/history semantics.
    """
    require(session.source["model"] == "tiger" and session.source["dataset"] in ("beauty", "instruments"),
            "only TIGER Beauty/Instruments imports are supported")
    require(name in ("sasrec", "interaction", "similarity"), f"unsupported component {name!r}")
    require(not session.args.smoke, "smoke factors cannot be imported as full factors")
    require(getattr(dm, "_is_setup", False) and dm.train_dataset is not None, "datamodule must already be set up")
    require(len(dm.train_dataset) > 0, "source training set is empty")
    require(dm.cfg.train_example_layout == "prefix_target", "expected full prefix-target layout")
    require(not any(getattr(session.args, key, None) for key in
                    ("train_last_n", "train_random_n", "train_pool_last_n")), "source uses a training subset")
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("factors_path") == "factors.pt", "unexpected factor payload path")
    payload_path = directory / "factors.pt"
    require(payload_path.is_file() and payload_path.stat().st_size == manifest.get("factor_bytes"),
            "factor payload size differs from committed manifest")
    require(manifest.get("kind") == "factorized" and manifest.get("split") == "train", "expected factorized train cache")
    require(manifest.get("num_examples") == len(dm.train_dataset), "factor row count differs from full source training set")
    for key in ("per_user_last_n", "per_user_random_n", "per_user_random_seed", "per_user_pool_last_n"):
        require(manifest.get(key) is None, f"cache was built with subset selection {key}")
    component = manifest.get("component", {})
    require(component.get("name") == name, "cache component differs")
    require(component.get("mode") == ("hierarchical" if name == "similarity" else None), "cache component mode differs")
    components = manifest.get("components", [])
    require(len(components) == 1 and components[0].get("name") == name
            and components[0].get("weight") == 1.0
            and components[0].get("normalizer") in ("none", "identity"),
            "cache must contain one raw component with unit weight and no normalization")

    expected_config = session.base.cache_config(session.config, name, session.args)
    expected_cfg = session.base.tiger_runner.build_data_config(session.config, config_path=session.config_path)
    wanted_data = data_settings(asdict(expected_cfg))
    require(data_settings(asdict(dm.cfg)) == wanted_data, "provided datamodule configuration differs from session")
    require(data_settings(manifest.get("data_config", {})) == wanted_data, "cache data configuration differs")
    require(manifest.get("training_context") == dense_prefix_signature(session.config.get("dataset", {})),
            "cache training context differs")
    config_path = Path(relocate_path(manifest["config_path"], ROOT)).resolve()
    legacy_config = json.loads(config_path.read_text())
    legacy_cfg = session.base.tiger_runner.build_data_config(legacy_config, config_path=config_path)
    require(data_settings(asdict(legacy_cfg)) == wanted_data, "recorded cache config has different data semantics")
    require(normalized(legacy_config.get("method")) == normalized(expected_config.get("method")),
            "recorded cache reward/teacher configuration differs")
    require(manifest.get("reward_epsilon") == expected_config["method"]["reward_epsilon"], "reward floor differs")
    initial_records = {key: file_record(path) for key, path in
                       (("manifest", manifest_path), ("factors", payload_path), ("config", config_path))}
    cache = FactorizedRewardCache(directory)
    item_tokens = full_row_check(session, dm, cache)
    sample_indices = sorted({0, len(dm.train_dataset) // 2, len(dm.train_dataset) - 1})
    sample = torch.tensor(sample_indices, dtype=torch.long)
    # Avoid SequenceDataset.__getitem__, which materializes every Python row.
    batch = dm._make_collator()(dm.train_dataset.examples.iloc[sample_indices].to_dict(orient="records"))
    checks = ["semantic data configuration", "raw component and reward floor",
              "all user/target/timestamp/token rows", "catalogue row mapping"]
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        specs, epsilon = factors.build_reward_components(expected_config, dm, None, session.config_path, torch.device("cpu"))
        require(len(specs) == 1 and epsilon == manifest["reward_epsilon"], "current reward builder differs")
        model = specs[0]["model"]
        require(normalized(asdict(model.cfg)) == normalized(component.get("config", {})),
                "expanded reward component configuration differs")
        if name == "sasrec":
            expected_dtype = {"float16": torch.float16, "float32": torch.float32}.get(manifest.get("hidden_dtype"))
            require(cache.factors["hidden"].dtype == expected_dtype
                    and bool(torch.isfinite(cache.factors["hidden"]).all()),
                    "SASRec hidden factors have an invalid dtype or nonfinite values")
            raw, wrapper = factors._unwrap_sasrec(model.sasrec_model)
            row_to_item = session.base.tiger_runner.item_id_by_semantic_row(dm)
            table, valid = factors.sasrec_item_table(raw, wrapper, row_to_item)
            equal_tensor(cache.factors.get("item_table"), table, "SASRec item table")
            equal_tensor(cache.factors.get("row_valid"), valid, "SASRec valid catalogue rows")
            require(manifest.get("teacher_id_translation") == (wrapper is not None), "SASRec ID translation differs")
            history = batch["history_item_ids"]
            if wrapper is not None:
                history = wrapper._translate(history)
            hidden = factors.sasrec_final_hidden(raw, history, batch["history_mask"])
            close_tensor(cache.factors["hidden"][sample], hidden, "sampled SASRec hidden states", atol=2e-3, rtol=2e-3)
            checks += ["exact current SASRec item table", "three distributed hidden-state probes (atol=rtol=0.002)"]
        elif name == "similarity":
            equal_tensor(cache.factors.get("item_tokens"), item_tokens, "similarity catalogue tokens")
            close_tensor(cache.factors.get("code_dist"), model.code_dist, "RQ-VAE codebook distances", atol=1e-5, rtol=1e-5)
            checks += ["exact catalogue semantic tokens", "current RQ-VAE distances (atol=rtol=1e-5)"]
        else:
            pointer, cols, values = (cache.factors[key] for key in ("indptr", "cols", "vals"))
            require(pointer.dtype == cols.dtype == torch.long and values.dtype == torch.float32
                    and pointer.ndim == cols.ndim == values.ndim == 1,
                    "invalid sparse interaction tensor types")
            require(int(pointer[0]) == 0 and int(pointer[-1]) == cols.numel() == values.numel()
                    and bool((pointer[1:] >= pointer[:-1]).all()), "invalid sparse interaction row pointers")
            require(bool(((cols >= 0) & (cols < cache.num_items)).all()) and bool(torch.isfinite(values).all()),
                    "invalid sparse interaction columns or values")
            if cols.numel() > 1:
                within_row = torch.ones(cols.numel() - 1, dtype=torch.bool)
                boundaries = pointer[1:-1]
                boundaries = boundaries[(boundaries > 0) & (boundaries < cols.numel())]
                within_row[boundaries - 1] = False
                require(bool(((cols[1:] > cols[:-1]) | ~within_row).all()),
                        "sparse interaction columns must be sorted and unique within each row")
            row_to_item = session.base.tiger_runner.item_id_by_semantic_row(dm)
            item_to_row = session.base.tiger_runner.item_id_to_semantic_row(dm)
            expected = factors.compute_component_rewards(specs[0], batch, item_tokens, row_to_item, item_to_row, epsilon)
            close_tensor(cache.component_rows(sample, torch.device("cpu")), expected,
                         "sampled interaction rewards", atol=1e-6, rtol=1e-6)
            checks += ["sparse interaction structure", "three distributed full-catalogue reward probes"]
    for record in initial_records.values():
        stat = Path(record["path"]).stat()
        require(stat.st_size == record["bytes"] and stat.st_mtime_ns == record["mtime_ns"],
                "legacy artifact changed during validation")
    result = deepcopy(manifest)
    result["legacy_validation"] = {
        "source_id": session.source["source_id"], "all_rows_checked": len(dm.train_dataset),
        "probe_indices": sample_indices, "checks": checks, "artifacts": initial_records,
        "historical_input_hashes_available": False,
        "limitation": "Legacy manifests do not hash historical raw-data or teacher assets; current compatibility is checked, not historical byte identity.",
    }
    return result
