"""Build the final global training window in the original student's item space."""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import gflow_finetune_yambda as global_run
from from_sft_runtime import json_value, read_json, write_json


def pair_root(dataset, model, workspace):
    return Path(workspace).resolve() / "global_train_with_val" / model / dataset


def _profile(dataset, model, workspace, *, stage=False):
    import beauty2023_global_inputs
    import ce_distill_inputs
    import gpt2_gflow_inputs as amazon
    import yambda_gflow_inputs

    inputs = Path(workspace).resolve() / "global_train_with_val" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if dataset in (global_run.Y50, global_run.Y500, beauty2023_global_inputs.DATASET):
        source = beauty2023_global_inputs if dataset == beauty2023_global_inputs.DATASET else yambda_gflow_inputs
        if stage:
            source.stage_inputs(model, dataset, inputs)
        return source.load_profile(model, dataset, inputs)
    if dataset not in amazon.DATASETS or model not in ("tiger", "gpt2"):
        raise ValueError(f"Unknown global initialization {model}/{dataset}")
    # Legacy staging uses named paths. Each student has its own staging lock.
    inputs = inputs / model / dataset
    if stage:
        ce_distill_inputs.stage_inputs(model, dataset, inputs)
    return ce_distill_inputs.load_profile(model, dataset, inputs)


def _config(profile, root):
    config = profile.run_config(SimpleNamespace(batch_size=128, val_batch_size=128, num_workers=0))
    config["method"] = global_run.reward_method(profile, profile.model)
    config["method"]["sasrec"]["item_id_translation_path"] = str(root / "teacher_translation.json")
    identity = {"schema": 1, "protocol": "global_train_with_val", "dataset": profile.dataset,
                "model": profile.model, "source": profile.source_identity, "layout": config["dataset"],
                "reward": config["method"], "seed": config["seed"],
                "catalog": "original_student", "test": "original_test_examples"}
    if profile.dataset == global_run.Y500:
        identity["subset"] = {"last_n": 128, "sample": 50, "seed": global_run.Y500_SUBSET_SEED}
    return json_value(config), json_value(identity)


def _warm_prefixes(rows, min_history_len):
    """Remove cold targets and cold history entries after fixing the source window boundaries."""
    rows = rows.loc[rows.target_item_id.ne(0)].copy()
    histories = ("history_raw_items", "history_item_ids", "history_ratings", "history_verified_purchase")
    for idx, items in rows.history_item_ids.items():
        if 0 in items:
            keep = [i for i, item in enumerate(items) if item]
            for column in histories:
                values = rows.at[idx, column]
                rows.at[idx, column] = [values[i] for i in keep]
    return rows.loc[rows.history_item_ids.str.len().ge(min_history_len)]


def _warm_dense_windows(rows, block, min_history_len):
    """Keep the original dense windows and active warm labels after filtering cold items."""
    affected = rows.target_item_id.eq(0) | rows.history_item_ids.map(lambda items: 0 in items)
    if not affected.any():
        return rows
    starts = dict(zip(block.loc[block.position.eq(0), "user_id"], block.index[block.position.eq(0)]))
    dropped = []
    for idx in rows.index[affected]:
        row = rows.loc[idx]
        ids = row.history_item_ids + [row.target_item_id]
        warm = [pos for pos, item in enumerate(ids) if item]
        active = [original > 0 and warm_pos >= min_history_len and row.label_mask[original - 1]
                  for warm_pos, original in enumerate(warm)]
        if not any(active):
            dropped.append(idx)
            continue
        last = max(i for i, flag in enumerate(active) if flag)
        warm = warm[:last + 1]
        target_pos = row.target_position - len(row.history_item_ids) + warm[-1]
        target = block.iloc[starts[row.user_id] + target_pos]
        for name in ("history_item_ids", "history_raw_items", "history_ratings", "history_verified_purchase"):
            rows.at[idx, name] = [row[name][i] for i in warm[:-1]]
        raw = row.history_raw_items + [row.target_raw_item]
        rows.at[idx, "label_item_ids"] = [ids[i] for i in warm[1:]]
        rows.at[idx, "label_raw_items"] = [raw[i] for i in warm[1:]]
        rows.at[idx, "label_mask"] = active[1:last + 1]
        for dest, source in (("target_item_id", "item_id"), ("target_raw_item", "raw_item_id"),
                             ("target_rating", "rating"), ("target_verified_purchase", "verified_purchase"),
                             ("target_timestamp", "timestamp"), ("target_position", "position")):
            rows.at[idx, dest] = target[source]
    return rows.drop(index=dropped)


def _validation_dense_labels(rows, block, cutoff, min_history_len):
    """Count actual active validation-period targets, including windows crossing the old boundary."""
    first_val = block.loc[block.timestamp.lt(cutoff)].groupby("user_id", sort=False).size().to_dict()
    count = 0
    columns = ["user_id", "target_position", "history_item_ids", "label_item_ids", "label_mask"]
    for user, end, history, labels, mask in rows.loc[rows.target_timestamp.ge(cutoff), columns].itertuples(
            index=False, name=None):
        offset = max(0, first_val.get(user, 0) - (end - len(mask) + 1))
        if 0 not in history and labels[-1]:
            count += sum(mask[offset:])
        else:
            warm = 0
            for i, (item, label, active) in enumerate(zip(history, labels, mask, strict=True)):
                warm += int(item != 0)
                count += int(i >= offset and active and bool(label) and warm >= min_history_len)
    return int(count)


def rebuild_combined_training(dm, *, block_size=50_000, native_layout=False):
    """Replay the source positive-stream layout at the exclusive test boundary.

    Keep test histories (including the original rolling/window-start policy), item IDs
    and semantic IDs fixed. A validation example selected for scoring can represent
    many positive events, so build from the underlying chronological event stream.
    """
    import numpy as np
    import pandas as pd
    from flowrec.data.datasets import SequenceDataset, history_length_stats
    from flowrec.data.dense_prefix import dense_reference_config, example_digest, prefixes_from_dense_windows
    from flowrec.data.popularity import training_item_counts_by_raw_id
    from flowrec.data.sequence_builder import SequenceSplitConfig, _prepare_events, build_global_time_examples

    original_cutoff, test_cutoff = dm.global_val_cutoff, dm.global_test_cutoff
    if (dm.cfg.split_strategy != "global_time" or original_cutoff is None or test_cutoff is None
            or not original_cutoff < test_cutoff):
        raise ValueError("Expected distinct global validation and test cutoffs")
    if dm.cfg.train_target_policy not in ("all", "last_n"):
        raise ValueError(f"Unknown source target policy {dm.cfg.train_target_policy}")
    seq_cfg = SequenceSplitConfig(**{f.name: getattr(dm.cfg, f.name) for f in fields(SequenceSplitConfig)
                                     if hasattr(dm.cfg, f.name)})
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    from_dense = dm.cfg.train_prefix_from_dense
    if from_dense:
        seq_cfg = dense_reference_config(seq_cfg)
    audit = {"original_val_cutoff": int(original_cutoff), "exclusive_training_cutoff": int(test_cutoff),
             "original_train_examples": len(dm.train_dataset), "original_val_examples": len(dm.val_dataset),
             "original_test_examples": len(dm.test_dataset),
             "test_sha256": example_digest(dm.test_dataset.examples), "catalog_items": dm.num_items}
    # Release the original Python histories before rebuilding the full training pool.
    dm.train_dataset = None
    dm.val_dataset = SequenceDataset(dm.val_dataset.examples.iloc[:0], split="val")
    blocks = []
    validation_targets = 0
    events = dm.interactions
    user_ids = events.user_id.to_numpy()
    start = 0
    while start < len(events):
        # Each user belongs to exactly one block, preserving last_n and dense window anchors.
        end = int(np.searchsorted(user_ids, user_ids[min(start + block_size, len(events)) - 1], side="right"))
        block = events.iloc[start:end]
        start = end
        block = block.loc[block.timestamp.lt(test_cutoff)]
        if block.empty:
            continue
        block = _prepare_events(block)
        train, _, _ = build_global_time_examples(block, cfg=seq_cfg,
                                                 val_cutoff=test_cutoff, test_cutoff=test_cutoff)
        if from_dense:
            train = prefixes_from_dense_windows(train, block, last_n=(
                seq_cfg.train_last_n if seq_cfg.train_target_policy == "last_n" else None))
        if "label_mask" in train:
            validation_targets += _validation_dense_labels(train, block, original_cutoff, seq_cfg.min_history_len)
        if block.item_id.eq(0).any():
            if "label_mask" in train:
                train = _warm_dense_windows(train, block, seq_cfg.min_history_len)
            else:
                train = _warm_prefixes(train, seq_cfg.min_history_len)
        if "label_mask" not in train:
            validation_targets += int(train.target_timestamp.ge(original_cutoff).sum())
        if not native_layout:
            train = train.drop(columns=["raw_user_id", "target_raw_item", "history_raw_items", "split",
                                       "label_raw_items"], errors="ignore")
        blocks.append(train)
    if not blocks:
        raise ValueError("No pre-test interactions")
    train = pd.concat(blocks, ignore_index=True)
    if train.empty or train.target_timestamp.ge(test_cutoff).any():
        raise ValueError("Invalid combined training window")
    dm.train_dataset = SequenceDataset(train, split="train")
    dm.global_val_cutoff = test_cutoff  # This is the waterline used by interaction rewards.
    dm.train_item_counts_by_raw_id = training_item_counts_by_raw_id(
        events, split_strategy="global_time", global_val_cutoff=test_cutoff, item2id=dm.item2id)
    dm.stats = replace(dm.stats, num_train_examples=len(train), num_val_examples=0,
                       **history_length_stats(train, dm.val_dataset.examples, dm.test_dataset.examples))
    audit.update(pool_train_examples=len(train), pool_validation_window_targets=validation_targets,
        max_target_timestamp=int(train.target_timestamp.max()),
        train_target_policy=seq_cfg.train_target_policy, train_last_n=seq_cfg.train_last_n,
        max_history_len=seq_cfg.max_history_len, dense_window_prefixes=from_dense)
    return audit


def _native_cache_config(cfg):
    import ce_yambda_prepared_data as compact

    assets = {}
    for name in ("reviews_path", "metadata_path", "interactions_path", "embeddings_path", "semantic_id_path"):
        value = getattr(cfg, name, None)
        if value:
            path = Path(value).resolve(strict=True)
            stat = path.stat()
            assets[name] = [str(path), stat.st_size, stat.st_mtime_ns]
    identity = {"format": "native_global_refit_v1", "data": compact._input_config(cfg), "assets": assets}
    # In matched GPT2 training each prefix contributes exactly one active final item.
    # Store the prefix only, avoiding millions of duplicate dense histories/labels.
    if getattr(cfg, "train_dense_match_prefix", False):
        from flowrec.data.sequence_builder import validate_dense_prefix_match_config
        validate_dense_prefix_match_config(cfg)
        return replace(cfg, train_example_layout="prefix_target", train_dense_match_prefix=False), identity
    return cfg, identity


def prepare_native_refit(directory, data_cls, cfg):
    """Prepare all training examples and the frozen test for an initial-model refit on CPU."""
    import ce_yambda_prepared_data as compact
    from bounded_preparation import bounded_history_build

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    storage_cfg, identity = _native_cache_config(cfg)
    with compact._lock(directory / "native-prepare.lock"):
        if compact.check(directory, identity):
            return
        dm = data_cls(storage_cfg)
        with bounded_history_build(50_000):
            dm.setup()
        audit = rebuild_combined_training(dm)
        audit.update(train_layout=cfg.train_example_layout,
                     train_dense_match_prefix=getattr(cfg, "train_dense_match_prefix", False))
        source_summary = dm.summary()
        source_summary.update(train_example_layout=cfg.train_example_layout)
        if getattr(cfg, "train_dense_match_prefix", False):
            source_summary["train_dense_match_prefix"] = True
        dm.interactions = dm.interaction_store = None
        compact.prepare(directory, dm, None, identity, audit=audit, source_summary=source_summary)


def load_native_refit(directory, cfg):
    """Load a CPU-prepared native datamodule without rebuilding interactions on the GPU node."""
    import ce_yambda_prepared_data as compact

    storage_cfg, identity = _native_cache_config(cfg)
    dm = compact.load(directory, storage_cfg, identity)
    if dm is None:
        raise FileNotFoundError(f"Missing CPU-prepared global refit data in {directory}")
    if getattr(cfg, "train_dense_match_prefix", False):
        from scripts.gpt2_final_data import MatchedDenseRows
        dm.train_dataset = MatchedDenseRows(dm.train_dataset._tensorized_arrays_cache,
                                            id2item=dm.id2item, id2user=dm.id2user)
    dm.cfg = cfg
    manifest = read_json(dm._ce_prepared_manifest)
    audit = {**dm._ce_prepared_audit, "tensors_sha256": manifest["artifacts"]["tensors"]["sha256"]}
    return dm, audit


def prepare(dataset, model, workspace):
    import ce_yambda_prepared_data as compact
    import run_tiger
    from bounded_preparation import bounded_history_build
    from precompute_reward_cache import per_user_random_indices
    from yambda_gflow_teacher import prepare_teacher_translation

    root = pair_root(dataset, model, workspace)
    root.mkdir(parents=True, exist_ok=True)
    with compact._lock(root / "prepare.lock"):
        marker = root / "prepared.json"
        profile = _profile(dataset, model, workspace, stage=not marker.is_file())
        config, identity = _config(profile, root)
        if marker.is_file():
            _require_prepared(root, identity)
            compact.check(root / "prefixes", identity)
            return
        config_path = root / "data_reward_config.json"
        dm = run_tiger.build_data_module(run_tiger.build_data_config(config, config_path=config_path))
        with bounded_history_build(50_000):
            dm.setup()
        native = profile.verify_layout(dm)
        translation = Path(config["method"]["sasrec"]["item_id_translation_path"])
        if hasattr(profile, "verify_teacher_catalog"):
            profile.verify_teacher_catalog(dm, translation)
        else:
            prepare_teacher_translation(dataset, dm, profile.teacher_checkpoint, translation)
        audit = rebuild_combined_training(dm)
        audit.update(source_identity=profile.source_identity, original_layout=native,
                     teacher_translation_sha256=hashlib.sha256(translation.read_bytes()).hexdigest())
        global_run.store_reward_events(dm, root)
        summary = dm.summary()
        dm.interactions = dm.interaction_store = None
        indices = (per_user_random_indices(dm.train_dataset, global_run.Y500_TARGETS_PER_USER,
                                           global_run.Y500_SUBSET_SEED) if dataset == global_run.Y500 else None)
        times = dm.train_dataset.examples.target_timestamp
        if indices is not None:
            times = times.iloc[indices]
        audit.update(selected_train_examples=len(times),
                     selected_validation_window_targets=int(times.ge(audit["original_val_cutoff"]).sum()),
                     target_subset=identity.get("subset"))
        snap = compact.prepare(root / "prefixes", dm, indices, identity, audit=audit, source_summary=summary)
        config["refit_audit"] = {**audit, "source_indices_sha256": snap["source_indices_sha256"]}
        write_json(config_path, config)
        write_json(marker, identity)


def _require_prepared(root, identity):
    if read_json(root / "prepared.json") != identity:
        raise RuntimeError(f"Inconsistent train+validation preparation: {root}")
    for filename in ("prefixes/manifest.json", "positive.parquet", "auxiliary.parquet",
                     "teacher_translation.json", "data_reward_config.json"):
        if not (root / filename).is_file():
            raise FileNotFoundError(root / filename)


def _factor_identity(dm, identity):
    return {"inputs": identity,
            "prefix_tensor_sha256": read_json(dm._ce_prepared_manifest)["artifacts"]["tensors"]["sha256"]}


def _check_factor(directory, name, dm, expected, *, ready=True):
    if ready and read_json(directory / "ready.json") != expected:
        raise RuntimeError(f"Different training rows at {directory}")
    record = read_json(directory / "manifest.json")
    if (record["num_examples"] != len(dm.train_dataset) or record["num_catalog_items"] != len(dm.semantic_ids)
            or record["component"]["name"] != name or not record.get("verification", {}).get("examples")
            or not (directory / "factors.pt").is_file()):
        raise RuntimeError(f"Incomplete or incompatible reward factors at {directory}")
    if name == "sasrec" and not record.get("teacher_id_translation"):
        raise RuntimeError(f"Unverified teacher translation at {directory}")


def load_refit_inputs(workspace, model, dataset, *, need_factors, num_workers, batch_size, eval_batch_size):
    import ce_yambda_prepared_data as compact
    import run_tiger

    root = pair_root(dataset, model, workspace)
    profile = _profile(dataset, model, workspace)
    config_path = root / "data_reward_config.json"
    config, identity = _config(profile, root)
    _require_prepared(root, identity)
    cfg = replace(run_tiger.build_data_config(config, config_path=config_path), batch_size=batch_size,
                  eval_batch_size=eval_batch_size, test_batch_size=eval_batch_size,
                  num_workers=num_workers, persistent_workers=num_workers > 0)
    dm = compact.load(root / "prefixes", cfg, identity)
    snapshot = read_json(dm._ce_prepared_manifest)
    audit = {**dm._ce_prepared_audit, "source_indices_sha256": snapshot["source_indices_sha256"]}
    translation = Path(config["method"]["sasrec"]["item_id_translation_path"])
    if hashlib.sha256(translation.read_bytes()).hexdigest() != audit["teacher_translation_sha256"]:
        raise RuntimeError(f"SASRec item translation changed at {translation}")
    config["refit_audit"] = json_value(audit)
    if read_json(config_path) != config:
        raise RuntimeError(f"Configuration changed after train+validation preparation at {config_path}")
    directories = {name: root / "factors" / name for name in global_run.COMPONENTS}
    if need_factors:
        expected = _factor_identity(dm, identity)
        for name, directory in directories.items():
            _check_factor(directory, name, dm, expected)
    return dm, config, profile.checkpoint, directories


def build_factor(dataset, model, workspace, index, device=None):
    import ce_yambda_prepared_data as compact
    import precompute_reward_factors

    if index not in range(len(global_run.COMPONENTS)):
        raise ValueError(f"Unknown reward index {index}")
    dm, config, _, directories = load_refit_inputs(workspace, model, dataset, need_factors=False,
                                                  num_workers=0, batch_size=128, eval_batch_size=128)
    root = pair_root(dataset, model, workspace)
    name = global_run.COMPONENTS[index]
    directory = directories[name]
    directory.mkdir(parents=True, exist_ok=True)
    expected = _factor_identity(dm, read_json(root / "prepared.json"))
    with compact._lock(directory / "build.lock"):
        if (directory / "ready.json").is_file():
            _check_factor(directory, name, dm, expected)
            return
        if name == "interaction":
            global_run.restore_reward_events(dm, root)
        method = config["method"]
        spec = method["reward"]["components"][index]
        config_path = root / f"factor_{name}.json"
        write_json(config_path, {**config, "method": {"name": "gflow", "reward_model": name,
                                                      "reward": spec["reward"], "sasrec": method["sasrec"]}})
        args = ["--config", str(config_path), "--output-dir", str(directory), "--device",
                device or ("cuda" if index == 0 else "cpu"), "--num-workers", "0",
                "--batch-size", "128", "--verify-batches", "1", "--verify-examples-per-batch", "2"]
        if name == "interaction" and dataset in (global_run.Y500, "beauty_2023"):
            args.append("--spool-interaction")
        precompute_reward_factors.main(args, prepared_data=dm)
        _check_factor(directory, name, dm, expected, ready=False)
        write_json(directory / "ready.json", expected)
