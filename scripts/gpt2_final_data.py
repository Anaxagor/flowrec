"""Store the complete last-128 Yambda final fit as compact, memory-mapped rows."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import sys

from flowrec.data.yambda import YambdaDataModule
from scripts.gpt2_tuning_resume import AtomicTrialTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/exp7-ce-distillation"))
import ce_yambda_prepared_data as compact


def data_identity(cfg):
    if not (cfg.train_dense_match_prefix and cfg.train_example_layout == "dense_causal"
            and cfg.train_window_stride == 1 and cfg.train_window_len == cfg.max_history_len
            and cfg.train_target_policy == "last_n" and cfg.split_strategy == "global_time"
            and cfg.cold_item_train_source == "observed"):
        raise ValueError("Compact final data requires the matched global last-N dense protocol")
    assets = {}
    for name in ("interactions_path", "embeddings_path", "semantic_id_path"):
        path = getattr(cfg, name)
        if path:
            path = Path(path)
            stat = path.stat()
            assets[name] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    return {"format": "gpt2_matched_dense_v1", "assets": assets, "data": compact._input_config(cfg)}


def storage_config(cfg):
    # Compact storage keeps the prefix and its target. The full dense stream
    # equals history + target, shifted by one item, with one active final label.
    return replace(cfg, train_example_layout="prefix_target", train_dense_match_prefix=False)


def check_counts(counts, expected):
    if expected is not None and counts != expected:
        raise ValueError(f"Full matched dataset required: observed {counts}, expected {expected}")


def prepare(directory, cfg, *, expected_counts=None):
    identity = data_identity(cfg)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with compact._lock(directory / "cpu-prepare.lock"):
        manifest = compact.check(directory, identity)
        if manifest:
            check_counts(manifest["counts"], expected_counts)
            return
        data = YambdaDataModule(cfg)
        data.setup()
        splits = ("train", "val", "test")
        counts = {s: len(getattr(data, s + "_dataset")) for s in splits}
        check_counts(counts, expected_counts)
        summary = data.summary()
        frame = data.train_dataset.examples
        # Validate lossless encoding against the actual dense source, including
        # every masked label. Preserve row order and all three original splits.
        for row in frame[["history_item_ids", "target_item_id", "label_item_ids", "label_mask"]].itertuples(index=False):
            history, target, labels, mask = row
            if (not history or len(mask) != len(history) or not mask[-1] or any(mask[:-1])
                    or labels != history[1:] + [target]):
                raise ValueError("Training example cannot be represented as a matched dense row")
        frame.drop(columns=["label_item_ids", "label_mask", "label_raw_items"], inplace=True)
        data.interactions = data.interaction_store = None
        data.cfg = storage_config(cfg)
        compact.prepare(directory, data, None, identity, source_summary=summary)
        print(f"Prepared full last-128 splits: {counts}", flush=True)


class MatchedDenseRows(compact.CompactSequenceDataset):
    def __getitem__(self, index):
        example = super().__getitem__(index)
        history = example["history_item_ids"]
        example["label_item_ids"] = history[1:] + [example["target_item_id"]]
        example["label_raw_items"] = example["history_raw_items"][1:] + [example["target_raw_item"]]
        example["label_mask"] = [False] * (len(history) - 1) + [True]
        return example


def load(directory, cfg, *, expected_counts=None):
    data = compact.load(directory, storage_config(cfg), data_identity(cfg))
    if data is None:
        raise FileNotFoundError(f"Missing prepared final data in {directory}. Run CPU preparation first.")
    counts = {s: len(getattr(data, s + "_dataset")) for s in ("train", "val", "test")}
    check_counts(counts, expected_counts)
    if counts["train"] != data._ce_prepared_full_train_count:
        raise ValueError("A sampled training pool cannot be used for the final fit")
    data.cfg = cfg
    data.train_dataset = MatchedDenseRows(data.train_dataset._tensorized_arrays_cache,
                                          id2item=data.id2item, id2user=data.id2user)
    return data


def winner_counts(export):
    record = json.loads((ROOT / "scripts/global-time/gpt2_yambda500m_last128_checkpoint.json").read_text())
    digest = record["sources"]["gpt2_model_yambda_500m_listens.yaml"]
    if hashlib.sha256(Path(export).read_bytes()).hexdigest() != digest:
        raise ValueError("Final data requires the original winning trial 17 YAML")
    return {s: record["data_summary"][f"num_{s}_examples"] for s in ("train", "val", "test")}


@contextmanager
def training_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".final-training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A final training worker already owns {directory}") from exc
        yield


def final_fit(trainer, train_loader, val_loader, resolved, inputs):
    """Resume only this exact final protocol, including the selected best epoch."""
    from flowrec.training.trainer import FitResult
    from scripts.gpt2_tuning_resume import inspect_resume_checkpoint, restore_trial_checkpoint
    from resumable_trainer import restore_rng
    import torch

    directory = Path(trainer.cfg.checkpoint_dir)
    identity = {key: resolved[key] for key in ("seed", "model", "method", "trainer", "batch_plan", "ranking_ks")}
    identity.update(data=inputs, hparams=resolved["optimizer"])
    identity = json.loads(json.dumps(identity, default=str))
    manifest = directory / "resolved_config.json"
    record = {"resume_identity": identity, "model_config": resolved["model"], "trainer": resolved["trainer"]}
    if manifest.exists():
        if json.loads(manifest.read_text()) != record:
            raise ValueError(f"Final training identity changed at {manifest}")
    elif (directory / "last.pt").exists() or (directory / "best.pt").exists():
        raise ValueError(f"Existing checkpoint has no final training identity at {manifest}")
    else:
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True))
        temporary.replace(manifest)

    source = inspect_resume_checkpoint(directory, identity)
    trainer.shuffle_generator = train_loader.sampler.generator
    finished = False
    if source:
        finished = restore_trial_checkpoint(trainer, source, in_place=True)
        print(f"Restored final fit after {source.completed_epochs} complete epochs", flush=True)
        if not finished:
            rng = torch.load(directory / f"rng_epoch_{source.completed_epochs - 1}.pt", map_location="cpu", weights_only=True)
            trainer.shuffle_generator.set_state(rng["shuffle"])
            if train_loader.num_workers and train_loader.persistent_workers:
                # The uninterrupted loader reuses its workers. Give re-created
                # workers a separate generator so their one extra base-seed
                # draw cannot advance the original sampler ordering.
                train_loader.generator = torch.Generator().set_state(rng["shuffle"])
            restore_rng(rng["model"])
    fit = FitResult(state=trainer.state) if finished else trainer.fit(train_loader, val_loader)
    # fit() only returns a best path when a new record was set in this allocation.
    for attr, name in (("best_checkpoint_path", "best.pt"), ("last_checkpoint_path", "last.pt")):
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"Missing final checkpoint: {path}")
        setattr(fit, attr, str(path))
    return fit


class FinalTrainer(AtomicTrialTrainer):
    def load_checkpoint(self, path, **kwargs):
        payload = super().load_checkpoint(path, **kwargs)
        if Path(path).name == "best.pt":
            # A kill between best.pt and last.pt may have left a newer JSON
            # beside the recovered weights. Publish metadata of the evaluated file.
            keys = ("trainer_config", "trainer_state", "model_type", "model_class", "model_config",
                    "method_name", "method_class")
            sidecar = Path(str(path) + ".json")
            temporary = sidecar.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({key: payload[key] for key in keys}, indent=2, sort_keys=True))
            temporary.replace(sidecar)
        return payload

    def save_checkpoint(self, filename):
        # Publish the small RNG file before the atomic last.pt commit. After a
        # kill the previous last.pt still has its own RNG and best snapshot.
        if Path(filename).name == "last.pt":
            import torch
            from resumable_trainer import capture_rng
            rng = Path(self.cfg.checkpoint_dir) / f"rng_epoch_{self.state.epoch}.pt"
            temporary = rng.with_suffix(".tmp")
            torch.save({"model": capture_rng(), "shuffle": self.shuffle_generator.get_state()}, temporary)
            temporary.replace(rng)
        return super().save_checkpoint(filename)
