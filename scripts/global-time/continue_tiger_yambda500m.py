"""Continue global TIGER dropout-0 for 60 epochs, scoring validation every ten additional epochs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exp7-ce-distillation"))
from flowrec.ft_methods.sft import SFTMethod
from flowrec.training import TrainerConfig
from flowrec.training.loggers import JsonlExperimentLogger
from scripts.global_time_tiger_tuning import atomic_json, runner, tiger
from scripts.gpt2_final_data import training_lock
from resumable_trainer import EpochSeededTrainer

DATASET = "yambda_500m_listens"
INITIAL = ROOT / "checkpoints/exp2/global_split/initial/tiger" / DATASET
OUTPUT = INITIAL / "last_128/dropout-0_plus60"
TRIAL = Path("exp2-tuned_params/tiger") / DATASET / "trials/dropout-0/best.pt"
SOURCES = (INITIAL / "best.pt", ROOT / "checkpoints/global-time" / TRIAL,
           ROOT / "experiments/global-time/checkpoints" / TRIAL)
# The curated and original copies differ only in the metadata paths inside the checkpoint.
SOURCE_DIGESTS = {
    "31c0780d10bfbc53d994b6b811f0ad93abcffb88bb1526feedf1ac2fc2798cca",
    "f368ca93ad6b5f17b0c69e41692319e5d3d42c319cbcc2e6f5088d1db2540685",
}
SOURCE_EPOCHS = 7
EXTRA_EPOCHS = 60
LR = 0.0018511868508356536
VALIDATE_EVERY = 10
MONITOR = "val/sft/ndcg@10"


def source_files(explicit=None):
    candidates = (Path(explicit),) if explicit else SOURCES
    found = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        path = candidate.resolve()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in SOURCE_DIGESTS:
            break
        found[str(path)] = digest
    else:
        if found:
            raise ValueError(f"No matching seven-epoch global dropout-0 checkpoint; found {found}")
        raise FileNotFoundError(f"Missing global dropout-0 checkpoint: {candidates}")
    # A staged best.pt can still use the metadata from the trial or its curated copy.
    configs = (path.with_name("resolved_config.json"), *(p.with_name("resolved_config.json") for p in SOURCES))
    cfg_path = next((p for p in configs if p.is_file()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"Missing dropout-0 resolved_config.json beside {path} or in {INITIAL}")
    config = json.loads(cfg_path.read_text())
    trainer, method = config["trainer"], config["method"]
    if (config["optimizer"]["lr"] != LR or config["model"]["d_kv"] != 32
            or config["model"]["dropout"] != 0 or config["seed"] != 42 or method["name"] != "sft"
            or method["generation_num_beams"] != 10 or trainer["monitor"] != MONITOR
            or (trainer["val_sample_fraction"], trainer["val_sample_seed"], trainer["val_sample_resample"])
            != (0.25, 0, False)):
        raise ValueError(f"Inconsistent dropout-0 training recipe at {cfg_path}")
    return path, digest, config


def publish_epoch(directory):
    # last.pt is committed atomically before selection. Repeat this copy on resume if interrupted.
    for name in ("best.pt", "best.pt.json"):
        destination = directory / name
        temporary = destination.with_name(name + ".tmp")
        shutil.copyfile(directory / name.replace("best", "last"), temporary)
        os.replace(temporary, destination)


class ContinuationTrainer(EpochSeededTrainer):
    def train_one_epoch(self, train_loader, *, epoch, val_loader=None):
        metrics, validation = super().train_one_epoch(train_loader, epoch=epoch)
        additional = epoch + 1 - SOURCE_EPOCHS
        message = f"Epoch {epoch+1}, continuation {additional}/{EXTRA_EPOCHS}, train/loss={metrics['train/loss']:.6f}"
        if additional % VALIDATE_EVERY == 0:
            # The native epoch schedule counts from epoch 1. This continuation counts from epoch 8.
            validation = self.evaluate(val_loader, stage="val")
            score = validation[MONITOR]
            if not math.isfinite(score):
                raise FloatingPointError(f"Non-finite validation NDCG@10 at epoch {epoch + 1}")
            self._update_best_and_maybe_stop(validation, step_scheduler=False)
            message += f", {MONITOR}={score:.8f}"
        print(message, flush=True)
        return metrics, validation

    def save_checkpoint(self, filename):
        path = Path(super().save_checkpoint(filename))
        if path.name == "last.pt" and self.state.best_epoch == self.state.epoch:
            publish_epoch(path.parent)
        return str(path)


def resume_continuation(trainer, path, is_source):
    payload = trainer.load_checkpoint(path, map_location="cpu")
    if not payload["optimizer_state_dict"]["state"] or payload.get("scheduler_state_dict") is not None:
        raise ValueError("Expected a trained AdamW optimizer with constant learning rate")
    if any(group["lr"] != LR for group in trainer.optimizer.param_groups):
        raise ValueError("Checkpoint learning rate disagrees with dropout-0")
    if is_source:
        if trainer.state.best_epoch != SOURCE_EPOCHS - 1 or trainer.state.epoch != trainer.state.best_epoch:
            raise ValueError("Expected the validation-selected seventh epoch of dropout-0")
        # A new selection starts here. The original Adam moments, AMP scaler and step counters continue.
        trainer.state.best_epoch = trainer.state.best_metric = trainer.state.last_monitor_metric = None
        trainer.state.bad_epochs = 0
        trainer.state.should_stop = False
        trainer.state.history.clear()
    else:
        if not payload["epoch_progress"]["epoch_complete"]:
            raise ValueError("Expected an epoch-boundary continuation checkpoint")
        if (trainer.state.epoch + 1 - SOURCE_EPOCHS) % VALIDATE_EVERY == 0:
            # Trainer writes the checkpoint immediately before appending its epoch-end history.
            trainer.state.history.append({"epoch": trainer.state.epoch,
                                         "val": {MONITOR: trainer.state.last_monitor_metric}})
        # A kill between the atomic .pt write and its diagnostic sidecar or best promotion
        # leaves all the information needed for both files in last.pt.
        keys = ("trainer_config", "trainer_state", "model_type", "model_class", "model_config",
                "method_name", "method_class")
        sidecar = {k: payload[k] for k in keys}
        sidecar["epoch_progress"] = {k: v for k, v in payload["epoch_progress"].items() if k != "rng_state"}
        atomic_json(path.with_suffix(".pt.json"), sidecar)
        if trainer.state.best_epoch == trainer.state.epoch:
            publish_epoch(path.parent)
        elif trainer.state.best_epoch is not None and not (path.parent / "best.pt").is_file():
            raise FileNotFoundError(f"Missing earlier selected checkpoint in {path.parent}")
    trainer.restore_progress(payload)
    return payload


def train(data, source, digest, config, output, *, device="cuda"):
    output = Path(output).resolve()
    if output == source.parent or output in (p.parent.resolve() for p in SOURCES):
        raise ValueError("Choose a separate continuation directory to preserve dropout-0")
    if runner.resolve_device(device) != device:
        raise RuntimeError(f"Requested device {device} unavailable")
    last, result_path = output / "last.pt", output / "result.json"
    total = SOURCE_EPOCHS + EXTRA_EPOCHS
    manifest = json.loads(Path(data._ce_prepared_manifest).read_text())
    identity = {"trial": "dropout-0", "dataset": DATASET, "training_split": "train",
                "source_epochs": SOURCE_EPOCHS, "additional_epochs": EXTRA_EPOCHS, "seed": config["seed"],
                "data_sha256": manifest["artifacts"]["tensors"]["sha256"], "optimizer": config["optimizer"],
                "batch_size": data.cfg.batch_size, "model": config["model"], "method": config["method"],
                "monitor": MONITOR, "validate_every_additional_epochs": VALIDATE_EVERY,
                "original_trainer": {k: v for k, v in config["trainer"].items() if k != "checkpoint_dir"}}
    with training_lock(output):
        recipe = output / "resolved_config.json"
        if recipe.is_file():
            if json.loads(recipe.read_text()) != identity:
                raise ValueError(f"Existing continuation uses different data or training settings: {output}")
        elif last.exists() or (output / "best.pt").exists() or result_path.exists():
            raise ValueError(f"Missing continuation metadata beside existing weights in {output}")
        else:
            atomic_json(recipe, identity)
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if (result["total_epochs_completed"] != total or not last.is_file()
                    or not (output / "best.pt").is_file()):
                raise ValueError(f"Inconsistent completed continuation at {output}")
            return result
        runner.seed_everything(config["seed"])
        model = tiger.build_model(config["model"], data)
        expected_ids = model.item_semantic_ids.detach().cpu().clone()
        method = SFTMethod.from_dict(config["method"])
        cfg = replace(TrainerConfig.from_dict(config["trainer"]), checkpoint_dir=output,
                      max_epochs=total, max_steps=None, max_train_batches=None, max_eval_batches=None,
                      monitor=MONITOR, monitor_mode="max", early_stopping_patience=None,
                      run_validation_at_start=False, validate_every_epochs=None, validate_every_steps=None,
                      save_best=False, save_last=True, save_every_epochs=None)
        trainer = ContinuationTrainer.from_optimizer_config(
            model=model, method=method, cfg=cfg, device=device, seed=config["seed"],
            checkpoint_every_updates=0, logger=JsonlExperimentLogger(output / "training_log.jsonl"),
            **config["optimizer"])
        resume = last.is_file()
        payload = resume_continuation(trainer, last if resume else source, is_source=not resume)
        if not torch.equal(expected_ids, payload["model_state_dict"]["item_semantic_ids"]):
            raise ValueError("The source semantic-ID catalog differs from the prepared data")
        del payload, expected_ids
        loader = data.train_dataloader()
        if trainer.state.global_step != trainer.state.epoch * len(loader):
            raise ValueError("The completed checkpoint epochs disagree with the original training loader length")
        print(f"Starting at {trainer.state.epoch} completed epochs, target {total}; "
              f"AdamW lr={LR}, {len(loader)} batches/epoch", flush=True)
        state = trainer.fit(loader, data.val_dataloader()).state
        if state.epoch + 1 != total or state.global_step != total * len(loader):
            raise RuntimeError("TIGER did not complete the full continuation")
        checks = [{"epoch_1based": row["epoch"] + 1, "ndcg_at_10": row["val"][MONITOR]}
                  for row in state.history if MONITOR in row.get("val", {})]
        expected_checks = list(range(SOURCE_EPOCHS + VALIDATE_EVERY, total + 1, VALIDATE_EVERY))
        if [row["epoch_1based"] for row in checks] != expected_checks:
            raise RuntimeError("Incomplete validation history for the continuation")
        result = {"status": "completed", "dataset": DATASET, "trial": "dropout-0", "seed": config["seed"],
                  "source": str(source), "source_sha256": digest, "source_epochs": SOURCE_EPOCHS,
                  "additional_epochs_completed": EXTRA_EPOCHS, "total_epochs_completed": total,
                  "monitor": MONITOR, "selected_epoch_1based": state.best_epoch + 1,
                  "best_val_ndcg_at_10": state.best_metric, "validation": checks,
                  "best_checkpoint": str(output / "best.pt"), "last_checkpoint": str(last),
                  "learning_rate": LR, "global_steps": state.global_step}
        atomic_json(result_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Curated or original seven-epoch dropout-0 best.pt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    import tiger_yambda500m_sft_data as inputs
    source, digest, config = source_files(args.source)
    if args.stage == "prepare":
        inputs.prepare(args.workspace, config["data"], num_workers=0)
    else:
        data = inputs.load(args.workspace, config["data"], num_workers=args.num_workers)
        result = train(data, source, digest, config, args.output_dir, device=args.device)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
