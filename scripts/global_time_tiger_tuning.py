"""Shared training and protocol handling for the two global-time TIGER tuners."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_global_time_tiger_grid_base",
    PROJECT_ROOT / "scripts/exp2-tuned-params/tuning_tiger_yambda_500m.py",
)
assert _SPEC is not None and _SPEC.loader is not None
grid = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = grid
_SPEC.loader.exec_module(grid)
runner = grid.runner
tiger = grid.tiger_all
Trainer = grid.Trainer

DATASETS = ("beauty", "instruments", "beauty_2023", "yambda_50m_listens")
LARGE_DATASET = "yambda_500m_listens"
CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp2-tuned-params"
EXP1_CONFIG_DIR = PROJECT_ROOT / "configs/global-time/exp1-standard_params"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/global-time/exp2-tuned_params/tiger"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/global-time/exp2-tuned-params/tiger"
MONITOR = "val/sft/ndcg@10"


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--stage", choices=("all", "tune", "train-best", "report"), default="all")
    p.add_argument("--list-plan", action="store_true", help="Show resolved protocol without loading data")
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    p.add_argument("--model-config", type=Path, default=EXP1_CONFIG_DIR / "tiger_model.yaml")
    p.add_argument("--trainer-config", type=Path, default=EXP1_CONFIG_DIR / "trainer_tiger.yaml")
    p.add_argument("--semantic-ids-dir", type=Path, default=grid.DEFAULT_SEMANTIC_IDS_DIR)
    p.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--tuned-config-dir", type=Path, default=CONFIG_DIR)
    p.add_argument("--seeds", type=int, nargs="+", help="Final training seeds; default: trainer YAML")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--time-budget-hours", type=float, default=23.5,
                   help="Per invocation, including setup; negative disables the budget")
    p.add_argument("--val-sample-fraction", type=float,
                   help="Search only: default 1 for small Amazon, 0.25 for Beauty 2023/Yambda")
    p.add_argument("--epochs-per-trial", type=int,
                   help="Search budget; 500M defaults to 8, others to the trainer epoch cap")
    p.add_argument("--max-epochs", type=int, help="Debug override for final training only")
    p.add_argument("--train-batches-per-epoch", type=int, help="Debug cap, search and final training")
    p.add_argument("--max-eval-batches", type=int, help="Debug cap, including final evaluation")
    return p


def validate_args(p, args):
    for key in ("epochs_per_trial", "max_epochs", "train_batches_per_epoch", "max_eval_batches"):
        value = getattr(args, key)
        if value is not None and value < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        p.error("--num-workers must be nonnegative")
    if args.val_sample_fraction is not None and not 0 < args.val_sample_fraction <= 1:
        p.error("--val-sample-fraction must be in (0, 1]")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must be distinct")
    if args.time_budget_hours < 0:
        args.time_budget_hours = None
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, runner.resolve_path(value, base=PROJECT_ROOT))
    return args


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(runner.to_jsonable(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def normalize_protocol(protocol):
    """Accept only known legacy defaults that preserve TIGER's data semantics."""
    normalized = deepcopy(protocol)
    data = normalized.get("data", {})
    if normalized.get("version") == 1 and data.get("train_example_layout") == "prefix_target":
        # Added to YambdaDataConfig for opt-in dense GPT-2 training. Older
        # manifests omitted it; False keeps exactly the existing prefix path.
        # Do not fill arbitrary new dataclass fields from their current defaults.
        data.setdefault("train_dense_match_prefix", False)
    return normalized


def protocol_differences(saved, current, path=""):
    """Describe mismatches without weakening the comparison of experiment settings."""
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(saved.keys() | current.keys()):
            key_path = f"{path}.{key}" if path else key
            if key not in saved:
                differences.append(f"{key_path}: saved=<missing>, current={current[key]!r}")
            elif key not in current:
                differences.append(f"{key_path}: saved={saved[key]!r}, current=<missing>")
            else:
                differences.extend(protocol_differences(saved[key], current[key], key_path))
        return differences
    if isinstance(saved, list) and isinstance(current, list) and len(saved) == len(current):
        return [difference for index, (old, new) in enumerate(zip(saved, current))
                for difference in protocol_differences(old, new, f"{path}[{index}]")]
    return [] if saved == current else [f"{path}: saved={saved!r}, current={current!r}"]


@contextmanager
def dataset_lock(root):
    """One worker per dataset; the OS releases this lock even after SIGKILL."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another TIGER worker owns {root}; use one job per dataset") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_data_config(cfg):
    required = {
        "split_strategy": "global_time", "global_history_policy": "rolling",
        "global_eval_target_mode": "last_per_user", "filter_cold_items_after_split": True,
        "cold_item_train_source": "observed", "drop_items_missing_semantic_id": False,
        "train_example_layout": "prefix_target",
    }
    for key, value in required.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"Global-time TIGER requires {key}={value!r}")
    if cfg.global_val_timestamp is None or cfg.global_test_timestamp is None:
        raise ValueError("Global-time TIGER requires pinned timestamps")
    if not cfg.global_val_timestamp < cfg.global_test_timestamp:
        raise ValueError("Global-time TIGER requires ordered timestamps")
    if getattr(cfg, "use_hf", False):
        raise ValueError("Global-time TIGER requires local data (use_hf=false)")


def load_incumbent(dataset, args):
    model = runner.load_yaml(args.model_config)
    trainer = tiger.apply_trainer_defaults(runner.load_yaml(args.trainer_config))
    for group in tiger.GROUPS:
        path = args.config_dir / group.data_config_name
        if not path.is_file():
            continue
        config = runner.load_yaml(path)
        if dataset not in config.get("datasets", {}):
            continue
        cfg, test_batch = tiger.build_data_config(
            group, dataset, config, seed=42, semantic_ids_dir=args.semantic_ids_dir,
            user_hash_mod=int(model.get("user_ids_count", 1)),
        )
        validate_data_config(cfg)
        if args.num_workers is not None:
            cfg = replace(cfg, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        return grid.Incumbent(
            grid.TrialConfig(model, trainer, int(cfg.batch_size),
                             int(trainer.get("gradient_accumulation_steps", 1))),
            cfg, test_batch, group, config,
            {"model_config": str(args.model_config), "trainer_config": str(args.trainer_config),
             "dataset_config": str(path), "semantic_id_path": str(cfg.semantic_id_path)},
        )
    raise ValueError(f"Dataset {dataset!r} is missing from {args.config_dir}")


class TimeBudgetExceeded(RuntimeError):
    pass


class Run:
    def __init__(self, dataset, args, *, search, started_at=None):
        self.dataset, self.args = dataset, args
        self.incumbent = load_incumbent(dataset, args)
        self.root = args.checkpoint_root / dataset
        self.started_at = time.time() if started_at is None else started_at
        self.data = None
        self.device = runner.resolve_device(args.device)
        self.search = search
        self.seeds = args.seeds or list(self.incumbent.config.trainer["seeds"])
        self.fraction = args.val_sample_fraction
        if self.fraction is None:
            self.fraction = 1.0 if dataset in ("beauty", "instruments") else 0.25

    def trainer_config(self, cfg, directory, *, final):
        cli = argparse.Namespace(max_epochs=self.args.max_epochs if final else None,
                                 max_steps=None, max_eval_batches=self.args.max_eval_batches)
        result = tiger.build_trainer_config(
            cfg.trainer, directory, cli,
            use_amp=bool(cfg.trainer.get("use_amp", False)) and self.device.startswith("cuda"),
        )
        result = replace(result, progress_bar=self.args.progress_bar,
                         max_train_batches=self.args.train_batches_per_epoch)
        if final:
            return result
        return replace(
            result, max_epochs=self.args.epochs_per_trial or (
                8 if self.dataset == LARGE_DATASET else result.max_epochs),
            monitor=MONITOR, monitor_mode="max", validate_every_epochs=1,
            early_stopping_patience=None if self.dataset == LARGE_DATASET else result.early_stopping_patience,
            val_sample_fraction=self.fraction if self.fraction < 1 else None,
            val_sample_seed=0, val_sample_resample=False,
        )

    def method(self, cfg, *, final):
        if final:
            return tiger.build_method(cfg.model)
        return grid.build_method(cfg.model, argparse.Namespace(val_topk=[1, 5, 10], val_num_beams=10))

    def protocol(self):
        cfg = self.incumbent.config
        data = asdict(self.incumbent.data_cfg)
        for key in ("num_workers", "persistent_workers", "pin_memory"):
            data.pop(key, None)
        trainers = {}
        for stage in ("search", "final"):
            values = self.trainer_config(cfg, Path("unused"), final=stage == "final").to_dict()
            for key in ("checkpoint_dir", "progress_bar", "use_amp"):
                values.pop(key, None)
            trainers[stage] = values
        return runner.to_jsonable({
            "version": 1, "dataset": self.dataset, "data": data,
            "model": cfg.model, "optimizer": runner.optimizer_config(cfg.trainer),
            "trainer": trainers, "search": self.search, "seed": 42,
        })

    def check_protocol(self):
        protocol = self.protocol()
        path = self.root / "study_config.json"
        if path.exists():
            saved = json.loads(path.read_text())
            differences = protocol_differences(normalize_protocol(saved), normalize_protocol(protocol))
            if differences:
                raise ValueError(
                    f"Existing TIGER run uses a different split/training/search protocol in {path}:\n  "
                    + "\n  ".join(differences)
                    + "\nRestore the original settings to resume, or use a new checkpoint root for a new experiment."
                )
            if saved == protocol:
                return
            backup = path.with_name("study_config.before_compatibility_update.json")
            if not backup.exists():
                atomic_json(backup, saved)
            print(f"[{self.dataset}] accepting legacy protocol with omitted "
                  f"data.train_dense_match_prefix=false; original saved to {backup}", flush=True)
        atomic_json(path, protocol)

    def check_time(self):
        hours = self.args.time_budget_hours
        if hours is not None and time.time() - self.started_at >= hours * 3600:
            raise TimeBudgetExceeded("Worker time budget exhausted; rerun the same command to resume")

    def get_data(self):
        if self.data is None:
            self.check_time()
            for path in runner.required_input_files(self.incumbent.data_cfg):
                if not path.is_file():
                    raise FileNotFoundError(path)
            print(f"[{self.dataset}] loading data once for search and final training", flush=True)
            self.data = self.incumbent.group.data_module(self.incumbent.data_cfg)
            self.data.setup()
            if self.data.semantic_ids is None or self.data.sid_len is None:
                raise ValueError(f"{self.dataset}: missing semantic IDs")
            missing = set(self.data.id2item[1:]) - set(self.data.semantic_ids)
            if missing:
                raise ValueError(f"{self.dataset}: {len(missing)} training items have no semantic ID")
            atomic_json(self.root / "data_summary.json", self.data.summary())
        return self.data

    def train(self, cfg, directory, *, seed=42, final=False, report_epoch=None):
        """Resume only this run's last.pt; final runs start fresh, never from short trials."""
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        self.check_time()
        data = self.get_data()
        self.check_time()
        runner.seed_everything(seed)
        val_batch = (self.incumbent.data_cfg.eval_batch_size if final else
                     max(1, self.incumbent.test_eval_batch_size * int(cfg.model.get("eval_num_beams", 20)) // 10))
        data.cfg = replace(data.cfg, batch_size=cfg.batch_size, eval_batch_size=val_batch, seed=seed)
        train_cfg = self.trainer_config(cfg, directory, final=final)
        method = self.method(cfg, final=final)
        epoch_log = directory / "epoch_log.jsonl"
        trainer = model = None

        def logger(payload):
            if payload.get("type") != "epoch_end":
                return
            with epoch_log.open("a") as handle:
                handle.write(json.dumps(runner.to_jsonable(payload)) + "\n")
            epoch = int(payload["epoch"])
            print(f"[{self.dataset}/{directory.name}] epoch {epoch + 1}/{train_cfg.max_epochs}: "
                  f"{train_cfg.monitor}={trainer.state.best_metric}", flush=True)
            if report_epoch is not None:
                report_epoch(payload)
            # Checkpoints have already been saved by Trainer at epoch_end.
            hours = self.args.time_budget_hours
            if (hours is not None and not trainer.state.should_stop and epoch + 1 < train_cfg.max_epochs
                    and time.time() - self.started_at + float(payload.get("epoch_time_sec", 0)) > hours * 3600):
                raise TimeBudgetExceeded("Next epoch would exceed the time budget; last.pt is resumable")

        try:
            model = tiger.build_model(cfg.model, data)
            trainer = Trainer.from_optimizer_config(
                model=model, method=method, cfg=train_cfg, device=self.device,
                logger=logger, **runner.optimizer_config(cfg.trainer),
            )
            atomic_json(directory / "resolved_config.json", {
                "data": asdict(data.cfg), "model": cfg.model, "trainer": train_cfg.to_dict(),
                "method": method.cfg.to_dict(), "optimizer": runner.optimizer_config(cfg.trainer), "seed": seed,
            })
            last = directory / "last.pt"
            completed = 0
            if last.exists():
                trainer.load_checkpoint(last, load_optimizer=True, load_scheduler=False)
                completed = int(trainer.state.epoch) + 1
                trainer.state.epoch = completed
            if not trainer.state.should_stop and completed < train_cfg.max_epochs:
                trainer.fit(data.train_dataloader(), data.val_dataloader())
                completed = int(trainer.state.epoch) + 1
            value = trainer.state.best_metric
            if value is None:
                raise RuntimeError(f"Training produced no {train_cfg.monitor}")
            if not math.isfinite(value):
                raise FloatingPointError(f"Training produced non-finite {train_cfg.monitor}: {value}")
            result = {
                "status": "complete", "best_value": float(value), "monitor": train_cfg.monitor,
                "best_epoch": trainer.state.best_epoch, "epochs_completed": completed,
                "epochs_planned": train_cfg.max_epochs, "seed": seed, "hparams": cfg.hparams(),
                "best_checkpoint": str(directory / "best.pt"), "last_checkpoint": str(last),
            }
            if final:
                self.check_time()
                trainer.load_checkpoint(directory / "best.pt", load_optimizer=False, load_scheduler=False)
                data.cfg = replace(data.cfg, eval_batch_size=self.incumbent.test_eval_batch_size)
                kwargs = dict(
                    device=self.device, lookup=tiger.SemanticIdItemLookup(data.semantic_ids, data.item2id),
                    item_popularity=runner.training_item_popularity(data), num_catalog_items=int(data.num_items),
                    ks=tiger.resolve_ranking_ks(cfg.model), num_beams=int(cfg.model.get("eval_num_beams", 20)),
                    constrain_valid_sequences=bool(cfg.model.get("constrain_valid_sequences", True)),
                    max_batches=self.args.max_eval_batches,
                )
                result["validation_metrics"] = tiger.evaluate_generation_ranking_metrics(model, data.val_dataloader(), **kwargs)
                result["test_metrics"] = tiger.evaluate_generation_ranking_metrics(model, data.test_dataloader(), **kwargs)
            atomic_json(result_path, result)
            return result
        finally:
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def publish(self, winner, results, *, complete):
        summary = {"dataset": self.dataset, "search_complete": complete,
                   "winner": winner, "trials": results, "protocol": self.protocol()}
        atomic_json(self.root / "tuning_summary.json", summary)
        atomic_json(self.args.artifacts_dir / f"tiger_{self.dataset}_tuning_summary.json", summary)
        if winner is not None:
            atomic_json(self.root / "best_trial.json", winner)
        if winner is not None and complete:
            cfg = grid.Trial(winner["overrides"]).config(self.incumbent.config)
            tuned = {**cfg.model, **runner.optimizer_config(cfg.trainer),
                     "max_grad_norm": cfg.trainer["max_grad_norm"],
                     "gradient_accumulation_steps": cfg.grad_accum, "batch_size": cfg.batch_size}
            path = self.args.tuned_config_dir / f"tiger_model_{self.dataset}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".yaml.tmp")
            tmp.write_text("# Global-time TIGER winner; final trainer schedule comes from trainer_tiger.yaml.\n"
                           + yaml.safe_dump(tuned, sort_keys=False))
            os.replace(tmp, path)
        return summary

    def train_best(self, summary):
        if not summary["search_complete"]:
            raise RuntimeError("Finish the entire search before --stage train-best")
        if summary["winner"] is None:
            raise RuntimeError("Search has no successful trial; cannot train a winner")
        winner = summary["winner"]
        cfg = grid.Trial(winner["overrides"]).config(self.incumbent.config)
        # Freeze winner identity independently of seeds, so extending a search
        # cannot accidentally resume the previous winner's final checkpoint.
        path = self.root / "final_config.json"
        identity = {"winner": winner["trial"], "hparams": cfg.hparams()}
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Final training already belongs to another winner; use a new checkpoint root")
        atomic_json(path, identity)
        for seed in self.seeds:
            result = self.train(cfg, self.root / f"seed_{seed}", seed=seed, final=True)
            atomic_json(self.args.artifacts_dir / f"tiger_{self.dataset}_seed_{seed}_summary.json", result)
