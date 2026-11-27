"""Initial students and their held-out data protocols for GRPO and SDPO backfills."""
from __future__ import annotations

import json
from importlib import import_module
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BEAUTY = "beauty_2023"
Y50 = "yambda_50m_listens"
Y500 = "yambda_500m_listens"


@dataclass
class BackfillInputs:
    model: Any
    data: Any
    checkpoint: Path
    config: dict


def _read(path):
    return json.loads(path.read_text())


def _base_profile(split: str, model: str, dataset: str):
    """Read the matching base's native layout, architecture and held-out population."""
    if split not in {"loo", "global"} or model not in {"gpt2", "tiger"} or dataset not in {BEAUTY, Y50, Y500}:
        raise ValueError(f"Unknown RL backfill profile: {split}/{model}/{dataset}")
    directory = Path("checkpoints/exp2") / f"{split}_split" / "initial" / model / dataset
    if split == "global":
        if model == "tiger" and dataset == BEAUTY:
            # Trial 31 used the same pinned reviews/SIDs, cold-item policy and
            # held-out examples as GPT-2, with rolling prefixes in training.
            source = ROOT / directory.parents[1] / "gpt2" / dataset
            data = _read(source / "study_config.json")["data_config"]
            data.update(train_example_layout="prefix_target", history_padding="left")
            return (directory / "trial_31.pt", data, {"d_kv": 64, "user_ids_count": 1},
                    _read(source / "data_summary.json"))
        if model == "gpt2" and dataset == Y500:
            # Trial 17 used stride-one terminal labels on the
            # TIGER last-128 population. Dropout is omitted from the .pt metadata.
            source = ROOT / directory.parents[1] / "tiger" / dataset
            data = _read(source / "resolved_config.json")["data"]
            data.update(train_example_layout="dense_causal", train_dense_match_prefix=True,
                        train_window_len=data["max_history_len"], train_window_stride=1)
            return (directory / "last_128/trial_17.pt", data,
                    {"dropout": 0.2605724812002705, "user_ids_count": 1},
                    _read(source / "data_summary.json"))
        if model == "gpt2" and dataset == Y50:
            directory /= "full_history"
        summary = _read(ROOT / directory / "data_summary.json")
        if model == "tiger":
            record = _read(ROOT / directory / "resolved_config.json")
            return directory / "best.pt", record["data"], record["model"], summary
        study = _read(ROOT / directory / "study_config.json")
        architecture = _read(ROOT / directory / "best_trial.json")["model"]
        return directory / "best.pt", study["data_config"], architecture, summary

    import yaml

    config_dir = ROOT / "configs/exp2-tuned-params"
    filename = "dataset_amazon_2023.yaml" if dataset == BEAUTY else "dataset_yambda.yaml"
    yaml_cfg = yaml.safe_load((config_dir / filename).read_text())
    data = {k: v for k, v in yaml_cfg.items() if k not in {"datasets", "gpt2", "tiger", "sasrec"}}
    data.update(yaml_cfg["datasets"][dataset])
    data.update(semantic_id_path=f"datasets/semantic_ids/exp2-tuned-params/{dataset}/semantic_ids.json",
                train_example_layout="dense_causal" if model == "gpt2" else "prefix_target",
                train_all_prefixes=True, seed=42)
    if dataset != BEAUTY:
        data.update(train_target_policy="last_n", train_last_n=128)
        if model == "gpt2":
            # The LOO RL configs use the original full-history GPT-2 weights,
            # then sample a fixed 15 targets per user from the last 128.
            directory /= "full_history"
    if model == "gpt2":
        architecture = _read(ROOT / directory / "best_trial.json")["model"]
    else:
        name = ("tiger_beauty_2023_sft_tuned" if dataset == BEAUTY else f"tiger_{dataset}_sft")
        architecture = _read(config_dir / f"{name}.json")["model"]
    if dataset == BEAUTY:
        summary = {"num_test_examples": 662540, "num_val_examples": 662540}
    elif model == "gpt2":
        summary = _read(ROOT / directory / "training_summary.json")["runs"][0]["dataset"]
    else:
        summary = {}
    return directory / "best.pt", data, architecture, summary


def input_config(split: str, model: str, dataset: str, *, input_root: Path | None = None,
                 batch_size: int, eval_batch_size: int, num_workers: int):
    root = Path(input_root or ROOT).expanduser().resolve()
    checkpoint, native, architecture, summary = _base_profile(split, model, dataset)
    strategy = {"global": "global_time", "loo": "leave_one_out"}[split]
    if native.get("split_strategy") != strategy:
        raise ValueError(f"{checkpoint}: expected {strategy}, found {native.get('split_strategy')}")
    source = "amazon" if dataset == BEAUTY else "yambda"
    keys = ("reviews_path", "metadata_path", "semantic_id_path", "interactions_path", "embeddings_path")
    paths = {k: str(root / native[k]) for k in keys if native.get(k)}
    values = {k: v for k, v in native.items() if k not in keys}
    # Stride-one terminal-label pretraining has exactly the rolling prefixes.
    # Disjoint GPT-2 windows must be expanded after filtering the dense catalog.
    values.update(train_prefix_from_dense=(native["train_example_layout"] == "dense_causal"
                                           and not native.get("train_dense_match_prefix", False)),
                  train_example_layout="prefix_target", train_dense_match_prefix=False,
                  history_padding="left", metadata_ids_only=source == "amazon")
    config = {
        "data": {"source": source, **paths},
        "dataset": values,
        "dataloader": {"train_batch_size": batch_size, "validation_batch_size": eval_batch_size,
                       "num_workers": num_workers, "persistent_workers": num_workers > 0,
                       "include_raw_in_batch": False, "train_drop_last": False},
        "model": dict(architecture),
        "seed": native.get("seed", 42),
    }
    if split == "loo" and model == "gpt2" and dataset != BEAUTY:
        config["finetune_subset"] = {"per_user_random_n": 15, "per_user_random_seed": 42}
    return root / checkpoint, config, summary


def _open_raw(cfg, expected):
    from scripts import run_tiger

    data = run_tiger.build_data_module(cfg)
    bounded = import_module("scripts.exp8-from-sft.bounded_preparation")
    with bounded.bounded_history_build(50_000):
        data.setup()
    _check_population(data, expected)
    return data


def _check_population(data, expected):
    observed = {"num_val_examples": len(data.val_dataset), "num_test_examples": len(data.test_dataset),
                "item_vocab_size": data.item_vocab_size, "global_val_cutoff": data.global_val_cutoff,
                "global_test_cutoff": data.global_test_cutoff}
    for key, value in observed.items():
        if key in expected and expected[key] != value:
            raise ValueError(f"{key}={value}, expected {expected[key]} for this initial checkpoint")


def _selected_rows(data, config):
    if "finetune_subset" in config:
        from scripts.precompute_reward_cache import subset_indices

        return subset_indices(data.train_dataset, **config["finetune_subset"])
    return None


def _cache_location(cache_root, split, model, dataset, checkpoint, config):
    from hashlib import sha256

    sources = [checkpoint, *(Path(path) for key, path in config["data"].items() if key.endswith("_path"))]
    identity = {"schema": 1, "dataset": config["dataset"], "model": model, "split": split,
                "seed": config["seed"], "subset": config.get("finetune_subset"),
                "sources": {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in sources}}
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_root) / split / model / dataset / key, identity


def _save_events(data, directory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # The SASRec translation also needs the original IDs/positions of held-out
    # events. The interaction reward applies its own time cutoff on consumption.
    positive = data.interactions
    columns = [k for k in ("user_id", "item_id", "timestamp", "item_id_prefilter", "position") if k in positive]
    auxiliary = data.interaction_store.events
    for name, frame, keep in (("positive", positive, columns),
                               ("auxiliary", auxiliary, list(auxiliary.columns))):
        schema = pa.Schema.from_pandas(frame if name == "auxiliary" else frame.iloc[:1][keep],
                                       preserve_index=False)
        temporary = directory / f"{name}.partial"
        with pq.ParquetWriter(temporary, schema) as writer:
            for start in range(0, len(frame), 250_000):
                chunk = frame.iloc[start:start + 250_000][keep]
                writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
        temporary.replace(directory / f"{name}.parquet")


def _restore_events(data, directory, reward):
    import pandas as pd
    from flowrec.data.interaction_store import InteractionLogStore

    if reward and ("S" in reward or "I" in reward):
        data.interactions = pd.read_parquet(directory / "positive.parquet")
    if reward and "I" in reward:
        events = pd.read_parquet(directory / "auxiliary.parquet")
        data.interaction_store = InteractionLogStore(events, num_users=data.num_users,
                                                    item_vocab_size=data.item_vocab_size)


def prepare_inputs_cache(split: str, model: str, dataset: str, *, input_root: Path | None = None,
                         cache_root: Path) -> Path:
    """Prepare this base once on CPU. The manifest publishes all splits and reward events."""
    import fcntl
    from scripts import run_tiger

    compact = import_module("scripts.exp7-ce-distillation.ce_yambda_prepared_data")
    checkpoint, config, expected = input_config(split, model, dataset, input_root=input_root,
                                                batch_size=1, eval_batch_size=1, num_workers=0)
    directory, identity = _cache_location(cache_root, split, model, dataset, checkpoint, config)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "prefixes/manifest.json"
    with (directory / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = run_tiger.build_data_config(config, config_path=checkpoint)
        if dataset == BEAUTY and not hasattr(compact, "PreparedAmazonDataModule"):
            # Older checkouts only have a compact serializer for Yambda. The
            # native Amazon snapshot includes its reward stores and all splits.
            cfg = replace(cfg, processed_cache_dir=directory / "amazon-native")
            marker = run_tiger.build_data_module(cfg)._processed_cache_path()
            if not marker.is_file():
                _open_raw(cfg, expected)
            return marker
        if compact.check(directory / "prefixes", identity) is not None:
            for name in ("positive", "auxiliary"):
                if not (directory / f"{name}.parquet").is_file():
                    raise FileNotFoundError(f"Missing prepared {name} events in {directory}")
            return marker
        data = _open_raw(cfg, expected)
        source_summary = data.summary()
        _save_events(data, directory)
        # All columns needed by S and I are now on disk. Release their largest
        # source frames before flattening the histories into mmap-backed tensors.
        data.interactions = data.interaction_store = None
        rows = _selected_rows(data, config)
        compact.prepare(directory / "prefixes", data, rows, identity, source_summary=source_summary)
    return marker


def load_inputs(split: str, model: str, dataset: str, *, batch_size: int, eval_batch_size: int,
                num_workers: int, seed: int = 42, input_root: Path | None = None,
                cache_root: Path | None = None, reward: str | None = None) -> BackfillInputs:
    import torch
    from flowrec.data.datasets import SequenceDataset
    from scripts import run_gpt2, run_tiger

    checkpoint, config, expected = input_config(split, model, dataset, input_root=input_root,
                                                batch_size=batch_size, eval_batch_size=eval_batch_size,
                                                num_workers=num_workers)
    cfg = run_tiger.build_data_config(config, config_path=checkpoint)
    if cache_root is None:
        data = _open_raw(cfg, expected)
        selected = _selected_rows(data, config)
        if selected is not None:
            data.train_dataset = SequenceDataset(data.train_dataset.examples.iloc[selected], split="train")
    else:
        directory, identity = _cache_location(cache_root, split, model, dataset, checkpoint, config)
        compact = import_module("scripts.exp7-ce-distillation.ce_yambda_prepared_data")
        if dataset == BEAUTY and not hasattr(compact, "PreparedAmazonDataModule"):
            data = run_tiger.build_data_module(replace(cfg, processed_cache_dir=directory / "amazon-native"))
            if not data._load_processed_cache():
                raise FileNotFoundError(f"Missing prepared Amazon inputs in {directory}")
            if not reward or ("S" not in reward and "I" not in reward):
                data.interactions = data.interactions.iloc[:0].copy()
            if not reward or "I" not in reward:
                data.interaction_store = None
        else:
            data = compact.load(directory / "prefixes", cfg, identity)
            if data is None:
                raise FileNotFoundError(f"Missing prepared RL inputs for {split}/{model}/{dataset} in {directory}")
            _restore_events(data, directory, reward)
        _check_population(data, expected)
    payload = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=False)
    if payload.get("model_type") != model:
        raise ValueError(f"{checkpoint}: expected {model} initial weights")
    # Checkpoint metadata omits TIGER d_kv and GPT-2 dropout. Retain these
    # values from the selected trial while restoring dimensions from the weights.
    config["model"].update(payload["model_config"])
    config["model"].pop("checkpoint_path", None)
    student = (run_gpt2.build_model(config, data) if model == "gpt2"
               else run_tiger.build_model(config, data, config_path=checkpoint))
    state = payload["model_state_dict"]
    if not torch.equal(student.item_semantic_ids, state["item_semantic_ids"]):
        raise ValueError(f"Prepared semantic item catalog disagrees with {checkpoint}")
    student.load_state_dict(state, strict=True)
    config["checkpoint"] = str(checkpoint)
    config["finetune_seed"] = seed
    config["split_counts"] = {f"num_{s}_examples": len(getattr(data, s + "_dataset"))
                              for s in ("train", "val", "test")}
    return BackfillInputs(student, data, checkpoint, config)
