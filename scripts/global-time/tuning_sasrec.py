"""Shared-study, full-cross-entropy SASRec tuning on the global time split.

One hundred trials per dataset, including pruned/failed trials. Array workers
share a journal and allocate trial slots under a filesystem lock. Interrupted
RUNNING trials can be restarted under the same trial number after their owner
exits; active workers retain a per-trial lock. All workers must share the same
filesystem with working flock support.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import importlib.util
import io
import json
import math
import os
import shutil
import signal
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import optuna
import torch
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.training.trainer import Trainer, TrainerConfig
from scripts import sasrec_config_runner as runner

_SPEC = importlib.util.spec_from_file_location(
    "_global_time_full_ce_base", PROJECT_ROOT / "scripts/exp2-tuned-params/tuning_sasrec.py"
)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

DATASETS = base.DATASETS_TO_TUNE
TUNING_SPECS = {name: replace(spec, n_trials=100) for name, spec in base.TUNING_SPECS.items()}
MONITOR = base.MONITOR
CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp2-tuned-params"
CHECKPOINT_ROOT = PROJECT_ROOT / "experiments/global-time/checkpoints/exp2-tuned_params/sasrec"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/global-time/exp2-tuned-params/sasrec"


@dataclass(frozen=True)
class Runtime:
    config_dir: Path = CONFIG_DIR
    checkpoint_root: Path = CHECKPOINT_ROOT
    artifacts_dir: Path = ARTIFACTS_DIR
    tuned_config_dir: Path = CONFIG_DIR
    n_trials: int = 100
    device: str = base.DEVICE
    use_amp: bool = base.USE_AMP
    amp_dtype: str = base.AMP_DTYPE
    progress_bar: bool = True
    max_epochs: int | None = None
    num_workers: int | None = None
    train_batches_per_epoch: int | None = None
    max_eval_batches: int | None = None


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        # Fail if locking is unavailable; never allocate/promote without it.
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(runner.to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def resolve_data_config(dataset_name: str, rt: Runtime):
    filename = base.TUNED_CONFIG_DATA_YAML[dataset_name]
    group = next(g for g in base.run_all.GROUPS if g.data_config_name == filename)
    config = runner.load_yaml(rt.config_dir / filename)
    cfg = group.build_data_config(dataset_name, config, seed=base.SEED)
    if (
        cfg.split_strategy != "global_time"
        or not cfg.filter_cold_items_after_split
        or cfg.cold_item_train_source != "observed"
        or cfg.global_val_timestamp is None
        or cfg.global_test_timestamp is None
        or cfg.global_val_timestamp >= cfg.global_test_timestamp
    ):
        raise ValueError(
            f"{dataset_name}: requires global_time, ordered pinned cutoffs, "
            "filter_cold_items_after_split=true and cold_item_train_source=observed"
        )
    if cfg.train_example_layout != base.REQUIRED_TRAIN_LAYOUT:
        raise ValueError(f"{dataset_name}: requires dense_causal training")
    if cfg.semantic_id_path is not None:
        raise ValueError("Full-CE SASRec tuning must not filter the catalog through semantic IDs")
    if rt.num_workers is not None:
        cfg = replace(cfg, num_workers=rt.num_workers, persistent_workers=False)
    return group, cfg


def build_datamodule(group, cfg):
    data = group.data_module(cfg)
    data.setup()
    if not data.num_items or not len(data.train_dataset) or not len(data.val_dataset):
        raise ValueError("No usable training catalog, training rows or validation targets after splitting")
    print(json.dumps(data.summary(), indent=2, default=str), flush=True)
    return data


def effective_spec(dataset_name: str, rt: Runtime):
    spec = replace(TUNING_SPECS[dataset_name], n_trials=rt.n_trials)
    return replace(spec, max_epochs=rt.max_epochs) if rt.max_epochs is not None else spec


def open_study(dataset_name: str, rt: Runtime, *, worker_id: int, data_cfg):
    dataset_dir = rt.checkpoint_root / dataset_name
    spec = effective_spec(dataset_name, rt)
    protocol = runner.to_jsonable({
        "version": 1, "dataset": dataset_name, "item_loss": "cross_entropy",
        "data_config": asdict(data_cfg), "spec": asdict(spec),
        "seed": base.SEED, "monitor": MONITOR,
        "max_train_batches": rt.train_batches_per_epoch, "max_eval_batches": rt.max_eval_batches,
        # GPU precision can differ across unrestricted node types. Record it
        # on each trial, not as a requirement for joining the shared study.
        "use_amp": rt.use_amp,
    })
    with file_lock(dataset_dir / ".study.lock"):
        study = optuna.create_study(
            study_name=f"sasrec_{dataset_name}_val_ndcg10_exp2_global_time",
            storage=JournalStorage(JournalFileBackend(str(dataset_dir / "optuna_journal.log"))),
            direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=base.SEED + 9973 * worker_id, n_startup_trials=spec.n_startup_trials,
                multivariate=True, constant_liar=True,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=spec.n_startup_trials,
                n_warmup_steps=spec.pruner_warmup_epochs, interval_steps=1,
            ),
        )
        previous = study.user_attrs.get("protocol")
        if previous is not None and previous != protocol:
            raise ValueError("Existing study uses a different split/training protocol; use a new checkpoint root")
        study.set_user_attr("protocol", protocol)
        if not study.trials:
            # Any first worker seeds the study, including arrays without worker 0.
            study.enqueue_trial(dict(base.EXP1_BASELINE_PARAMS))
        atomic_json(dataset_dir / "study_config.json", protocol)
    return study


@dataclass
class TrialLease:
    trial: optuna.Trial
    handle: Any

    def close(self):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def _try_trial_lock(dataset_dir: Path, number: int):
    handle = (dataset_dir / f".trial_{number}.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def claim_trial(study, dataset_dir: Path, spec, *, worker_id: int) -> TrialLease | None:
    with file_lock(dataset_dir / ".study.lock"):
        trials = study.get_trials(deepcopy=False)
        trial, handle = None, None
        for frozen in trials:
            if frozen.state == TrialState.RUNNING:
                handle = _try_trial_lock(dataset_dir, frozen.number)
                if handle is not None:
                    # Reconstruct the same trial; saved suggestions and its seed
                    # survive interruption without consuming another budget slot.
                    trial = optuna.Trial(study, frozen._trial_id)
                    trial.set_user_attr("restarts", frozen.user_attrs.get("restarts", 0) + 1)
                    break
        if trial is None:
            allocated = sum(t.state != TrialState.WAITING for t in trials)
            if allocated >= spec.n_trials:
                return None
            trial = study.ask()
            handle = _try_trial_lock(dataset_dir, trial.number)
            if handle is None:
                raise RuntimeError(f"New trial {trial.number} already has an owner")
        try:
            base.suggest_hparams(trial, spec)
            trial.set_user_attr("worker_id", worker_id)
            trial.set_user_attr("seed", base.SEED + trial.number)
            return TrialLease(trial, handle)
        except BaseException:
            handle.close()
            raise


class EpochReporter(base.EpochValReporter):
    def __call__(self, payload):
        if payload.get("type") == "val_end" and MONITOR in payload:
            with file_lock(self.log_path.with_suffix(".lock")):
                super().__call__(payload)


def train_once(*, data, hparams, spec, rt: Runtime, trial_dir: Path, trial, dataset_name: str):
    runner.seed_everything(base.SEED + trial.number)
    model = base.build_model(data, hparams)
    method = base.build_method(data)  # SFTConfig.item_loss == cross_entropy.
    cfg = TrainerConfig(
        max_epochs=spec.max_epochs, gradient_accumulation_steps=spec.grad_accum,
        max_grad_norm=base.MAX_GRAD_NORM, use_amp=rt.use_amp, amp_dtype=rt.amp_dtype,
        log_every_steps=base.LOG_EVERY_STEPS, progress_bar=rt.progress_bar,
        validate_every_epochs=1, checkpoint_dir=trial_dir, save_best=True, save_last=False,
        monitor=MONITOR, monitor_mode="max", early_stopping_patience=spec.patience,
        min_delta=0.0, early_stopping_mode="best", prefix_method_metrics_with_stage=True,
        max_train_batches=rt.train_batches_per_epoch, max_eval_batches=rt.max_eval_batches,
    )
    trainer = Trainer.from_optimizer_config(
        model=model, method=method, cfg=cfg, device=rt.device,
        lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]),
        betas=base.BETAS, eps=base.EPS,
        logger=EpochReporter(trial=trial, dataset_name=dataset_name,
                            log_path=trial_dir.parents[1] / base.EPOCH_LOG_FILENAME),
    )
    result = trainer.fit(data.train_dataloader(), data.val_dataloader())
    value = trainer.state.best_metric
    if value is None or not math.isfinite(float(value)):
        raise FloatingPointError(f"No finite {MONITOR} value")
    checkpoint = Path(result.best_checkpoint_path) if result.best_checkpoint_path else None
    if checkpoint is None or not checkpoint.is_file():
        raise RuntimeError("Completed trial has no best-epoch checkpoint")
    return {
        "value": float(value), "best_epoch": trainer.state.best_epoch,
        "epochs_completed": trainer.state.epoch + 1, "checkpoint": str(checkpoint),
    }


def tuned_config(best, spec) -> dict:
    return {
        **{k: best.params[k] for k in ("d_model", "nhead", "num_layers", "dropout", "pos_drop")},
        "initializer_range": 0.02, "layer_norm_eps": 1e-8, "pad_item_id": 0,
        "exclude_history_in_generate": False,
        "lr": best.params["lr"], "weight_decay": best.params["weight_decay"],
        "betas": list(base.BETAS), "eps": base.EPS, "max_grad_norm": base.MAX_GRAD_NORM,
        "gradient_accumulation_steps": spec.grad_accum,
        "sft": {"item_loss": "cross_entropy", "compute_train_metrics": False,
                "compute_eval_metrics": True, "compute_item_metrics": True,
                "compute_token_metrics": False, "evaluate_generation": False,
                "item_topk": [1, 5, 10, 20], "pad_item_id": 0},
    }


def write_outputs(study, dataset_name: str, rt: Runtime) -> dict:
    """Caller holds .study.lock; every snapshot reads current journal state."""
    dataset_dir = rt.checkpoint_root / dataset_name
    trials = study.get_trials()
    rows = [{"number": t.number, "state": str(t.state), "value": t.value,
             "params": t.params, "user_attrs": t.user_attrs} for t in trials]
    atomic_json(dataset_dir / "all_trials.json", rows)
    counts = {state.name: sum(t.state == state for t in trials) for state in TrialState}
    summary = {"dataset": dataset_name, "study_name": study.study_name, "metric": MONITOR,
               "item_loss": "cross_entropy", "trials_planned": rt.n_trials,
               "trials_completed": sum(t.state.is_finished() for t in trials), "trial_states": counts}
    table = io.StringIO()
    columns = ["dataset", "trial", "state", "val_ndcg10", "best_epoch", "epochs_completed",
               "pruned_reason", *base.EXP1_BASELINE_PARAMS]
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()
    for trial in trials:
        writer.writerow({"dataset": dataset_name, "trial": trial.number,
                         "state": trial.state.name, "val_ndcg10": trial.value,
                         **{key: trial.user_attrs.get(key) for key in columns[4:7]},
                         **trial.params})
    atomic_text(rt.artifacts_dir / f"sasrec_{dataset_name}_trials.csv", table.getvalue())
    complete = [t for t in trials if t.state == TrialState.COMPLETE and t.value is not None]
    if complete:
        best = max(complete, key=lambda t: (t.value, -t.number))
        best_json = dataset_dir / "best_trial.json"
        previous = json.loads(best_json.read_text()) if best_json.exists() else {}
        checkpoint = dataset_dir / "best_model.pt"
        if previous.get("best_trial_number") != best.number or not checkpoint.exists():
            source = Path(best.user_attrs["checkpoint"])
            temporary = dataset_dir / f".best_model.{os.getpid()}.tmp"
            shutil.copy2(source, temporary)
            temporary.replace(checkpoint)
        config_path = rt.tuned_config_dir / f"sasrec_model_{dataset_name}.yaml"
        header = (
            f"# Full-CE SASRec, global time split: {dataset_name}\n"
            "# Generated by scripts/global-time/tuning_sasrec.py; preprocessing is not tuned.\n"
            f"# Dataset config: {rt.config_dir / base.TUNED_CONFIG_DATA_YAML[dataset_name]}\n"
            f"# Study: {study.study_name}; best trial: {best.number}; {MONITOR}: {best.value}\n"
            f"# Finished trials: {summary['trials_completed']}/{rt.n_trials}\n"
            f"# Checkpoint: {checkpoint}\n"
        )
        atomic_text(config_path, header + yaml.safe_dump(tuned_config(best, effective_spec(dataset_name, rt)), sort_keys=False))
        summary.update(best_value_val_ndcg10=best.value, best_trial_number=best.number,
                       best_epoch=best.user_attrs.get("best_epoch"), best_params=best.params,
                       checkpoint=str(checkpoint), tuned_config=str(config_path))
        atomic_json(best_json, summary)
        atomic_json(rt.artifacts_dir / f"sasrec_{dataset_name}_best_trial.json", summary)
    atomic_json(dataset_dir / "tuning_summary.json", summary)
    atomic_json(rt.artifacts_dir / f"sasrec_{dataset_name}_tuning_summary.json", summary)
    return summary


def run_trial(lease: TrialLease, *, study, dataset_name: str, data, rt: Runtime):
    trial = lease.trial
    dataset_dir = rt.checkpoint_root / dataset_name
    trial_dir = dataset_dir / "trials" / f"trial_{trial.number}"
    state, value, error = TrialState.COMPLETE, None, None
    try:
        try:
            trial.set_user_attr("amp_dtype", rt.amp_dtype if rt.use_amp else None)
            result = train_once(data=data, hparams=trial.params, spec=effective_spec(dataset_name, rt),
                                rt=rt, trial_dir=trial_dir, trial=trial, dataset_name=dataset_name)
            value = result.pop("value")
            for key, item in result.items():
                trial.set_user_attr(key, item)
        except optuna.TrialPruned as exc:
            trial.set_user_attr("pruned_reason", str(exc))
            state = TrialState.PRUNED
        except (FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, FloatingPointError) or "out of memory" in str(exc).lower():
                trial.set_user_attr("pruned_reason", str(exc))
                state = TrialState.PRUNED
            else:
                state, error = TrialState.FAIL, exc
        except Exception as exc:
            state, error = TrialState.FAIL, exc
        # KeyboardInterrupt/SIGTERM leave RUNNING and release the lease, so a
        # new worker restarts this same trial. SIGKILL releases flock in the OS.
        with file_lock(dataset_dir / ".study.lock"):
            if error is not None:
                trial.set_user_attr("error", str(error))
            study.tell(trial, values=value if state == TrialState.COMPLETE else None, state=state)
            write_outputs(study, dataset_name, rt)
        # Preserve the best model centrally, as in the earlier full-CE tuner.
        # If promotion fails, retain trial files for the next worker to recover.
        shutil.rmtree(trial_dir, ignore_errors=True)
        if error is not None:
            raise error
    finally:
        lease.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def tune_one_dataset(dataset_name: str, rt: Runtime, *, worker_id: int = 0):
    group, cfg = resolve_data_config(dataset_name, rt)
    study = open_study(dataset_name, rt, worker_id=worker_id, data_cfg=cfg)
    dataset_dir = rt.checkpoint_root / dataset_name
    # Do not rebuild large datamodules for a study that has already finished.
    with file_lock(dataset_dir / ".study.lock"):
        if sum(t.state.is_finished() for t in study.trials) >= rt.n_trials:
            return write_outputs(study, dataset_name, rt)
    data = build_datamodule(group, cfg)
    with file_lock(dataset_dir / ".study.lock"):
        atomic_json(dataset_dir / "data_summary.json", data.summary())
    while (lease := claim_trial(study, dataset_dir, effective_spec(dataset_name, rt), worker_id=worker_id)) is not None:
        print(f"[{dataset_name}] worker {worker_id}, trial {lease.trial.number}: {lease.trial.params}", flush=True)
        run_trial(lease, study=study, dataset_name=dataset_name, data=data, rt=rt)
    with file_lock(dataset_dir / ".study.lock"):
        return write_outputs(study, dataset_name, rt)


def array_assignment(task_id: int, workers: int, datasets=DATASETS):
    if workers <= 0 or task_id < 0 or task_id >= len(datasets) * workers:
        raise ValueError("Array task must lie in 0 .. datasets * workers - 1")
    return datasets[task_id // workers], task_id % workers


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--tuned-config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--n-trials", type=int, default=100, help="Shared per-dataset budget, not per worker")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--workers-per-dataset", type=int, default=8)
    parser.add_argument("--device", default=base.DEVICE)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-batches-per-epoch", type=int, default=None, help="Debug override")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Debug override")
    args = parser.parse_args(argv)
    for key in ("n_trials", "workers_per_dataset", "max_epochs", "train_batches_per_epoch", "max_eval_batches"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.worker_id < 0 or (args.num_workers is not None and args.num_workers < 0):
        parser.error("Worker IDs and dataloader worker counts must be nonnegative")
    if args.array_task_id is not None:
        try:
            dataset, args.worker_id = array_assignment(args.array_task_id, args.workers_per_dataset, args.datasets)
        except ValueError as exc:
            parser.error(str(exc))
        args.datasets = [dataset]
    return args


def main(argv=None):
    args = parse_args(argv)
    base.validate_specs()
    rt = Runtime(
        **{key: runner.resolve_path(getattr(args, key), base=PROJECT_ROOT)
           for key in ("config_dir", "checkpoint_root", "artifacts_dir", "tuned_config_dir")},
        n_trials=args.n_trials, device=args.device,
        use_amp=args.device.startswith("cuda") if args.use_amp is None else args.use_amp,
        amp_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        progress_bar=args.progress_bar, max_epochs=args.max_epochs, num_workers=args.num_workers,
        train_batches_per_epoch=args.train_batches_per_epoch, max_eval_batches=args.max_eval_batches,
    )
    summaries = {name: tune_one_dataset(name, rt, worker_id=args.worker_id) for name in args.datasets}
    # Array workers write per-dataset outputs, avoiding races on a combined file.
    if len(args.datasets) > 1:
        with file_lock(rt.checkpoint_root / ".summary.lock"):
            atomic_json(rt.checkpoint_root / "tuning_summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; the current trial can be restarted")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_termination)
    main()
