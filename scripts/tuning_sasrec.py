from __future__ import annotations

import gc
import json
import random
import shutil
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import torch


# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------

def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src").exists():
            return parent
    return here.parents[1]


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flowrec.data.datamodule import AmazonReviewDataConfig, AmazonReviewDataModule
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.training.trainer import Trainer, TrainerConfig

try:
    from flowrec.models.sasrec import SASRecModel as SASRec
except ImportError:
    from flowrec.models.sasrec import SASRec

from flowrec.models.sasrec import SASRecConfig


# ---------------------------------------------------------------------
# Dataset registry: all dataset information is read from here
# ---------------------------------------------------------------------

AMAZON_DATASETS = {
    "beauty": {
        "reviews_path": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_beauty_2014.parquet",
        "item_key": "asin",
        "metadata_item_key": "asin",
    },
    "instruments": {
        "reviews_path": "data/raw/dataframe_format/reviews_instruments_2018.parquet",
        "metadata_path": "data/raw/dataframe_format/meta_instruments_2018.parquet",
        "item_key": "asin",
        "metadata_item_key": "asin",
    },
}


# ---------------------------------------------------------------------
# Run configuration: edit here, not in the terminal
# ---------------------------------------------------------------------

DATASETS_TO_TUNE = tuple(AMAZON_DATASETS.keys())

SEED = 42

# Training-example layout. "dense_causal" (new default) supervises every
# position of a user window in one forward pass via the causal attention mask;
# "prefix_target" restores the old one-example-per-prefix layout. NOTE: trials
# tuned under one layout are not comparable to trials tuned under the other
# (optimizer-step counts differ), so keep one layout per study.
TRAIN_EXAMPLE_LAYOUT = "dense_causal"
TRAIN_WINDOW_LEN: int | None = None  # None -> max_history_len
TRAIN_WINDOW_STRIDE: int | None = None  # None -> window length (tiling)

# A dense row carries ~history_len supervised targets, so the old prefix batch
# of 512 would collapse an epoch into very few optimizer steps; 128 keeps the
# step count in the regime the lr search space was designed for.
BATCH_SIZE = 128 if TRAIN_EXAMPLE_LAYOUT == "dense_causal" else 512
EVAL_BATCH_SIZE = 512
NUM_WORKERS = 4

N_TRIALS = 50
N_STARTUP_TRIALS = 10
TIMEOUT = None

MAX_EPOCHS = 500
PATIENCE = 10
MIN_DELTA = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1
LOG_EVERY_STEPS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()
PROGRESS_BAR = True

BASE_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "optuna"

SAVE_BEST_CHECKPOINTS = True
SAVE_LAST_CHECKPOINT = False

COPY_EACH_TRIAL_BEST_CHECKPOINT = True
KEEP_ALL_TRIAL_CHECKPOINT_DIRS = True

KS = (10,)
MONITOR = "val/sft/ndcg@10"

PRINT_EPOCH_VAL_NDCG = True
EPOCH_VAL_NDCG_LOG_FILENAME = "epoch_val_ndcg10.jsonl"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def path_from_repo(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def amp_dtype_for_gpu() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


AMP_DTYPE = amp_dtype_for_gpu()


def make_config(cls, kwargs: dict[str, Any], *, required: tuple[str, ...] = ()):
    if not is_dataclass(cls):
        return cls(**kwargs)

    supported = {f.name for f in fields(cls)}
    missing_required = [name for name in required if name not in supported]

    if missing_required:
        raise RuntimeError(
            f"{cls.__name__} does not support required fields: {missing_required}."
        )

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    ignored = sorted(set(kwargs) - supported)

    if ignored:
        print(f"[Config warning] Ignoring unsupported fields for {cls.__name__}: {ignored}")

    return cls(**filtered)


def json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def scalar_to_python(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def get_item_vocab_size(dm: AmazonReviewDataModule) -> int:
    for attr in [
        "item_vocab_size",
        "vocab_item_size",
        "num_items",
        "n_items",
        "item_count",
        "num_item_ids",
    ]:
        if hasattr(dm, attr):
            return int(getattr(dm, attr))

    if hasattr(dm, "item2id"):
        return max(dm.item2id.values()) + 1

    raise AttributeError("Could not infer item vocabulary size.")


def to_jsonable(metrics: Mapping[str, Any] | None) -> dict[str, float | int | str]:
    if metrics is None:
        return {}

    out: dict[str, float | int | str] = {}

    for key, value in metrics.items():
        value = scalar_to_python(value)

        if isinstance(value, (float, int, str)):
            out[str(key)] = value
        elif value is None:
            continue
        else:
            try:
                out[str(key)] = float(value)
            except Exception:
                pass

    return out


def extract_val_ndcg10(metrics: Mapping[str, Any], metric_key: str = MONITOR) -> float:
    metrics = to_jsonable(metrics)

    if metric_key in metrics:
        return float(metrics[metric_key])

    fallback_keys = [
        "val/sft/ndcg@10",
        "val/ndcg@10",
        "sft/ndcg@10",
        "ndcg@10",
        "val/sft/NDCG@10",
        "val/NDCG@10",
    ]

    for key in fallback_keys:
        if key in metrics:
            return float(metrics[key])

    suffix_matches = [
        key for key in metrics
        if key.lower().endswith("ndcg@10")
    ]

    if len(suffix_matches) == 1:
        return float(metrics[suffix_matches[0]])

    raise KeyError(
        f"Could not find validation NDCG@10 in metrics. "
        f"Expected {metric_key!r}. Available keys: {sorted(metrics.keys())}"
    )


def extract_best_metric_from_fit_result(
    fit_result: Any,
    metric_key: str = MONITOR,
) -> float | None:
    for attr in [
        "best_metric",
        "best_score",
        "best_monitor_value",
        "best_val_metric",
        "best_val_score",
    ]:
        if hasattr(fit_result, attr):
            value = getattr(fit_result, attr)
            if value is not None:
                return float(scalar_to_python(value))

    return None


def extract_best_epoch_from_fit_result(fit_result: Any) -> int | float | str | None:
    for attr in [
        "best_epoch",
        "best_step_epoch",
        "best_monitor_epoch",
        "best_val_epoch",
    ]:
        if hasattr(fit_result, attr):
            value = getattr(fit_result, attr)
            if value is not None:
                return scalar_to_python(value)

    return None


def path_or_none(value: Any) -> Path | None:
    if value is None:
        return None

    text = str(value)

    if text == "" or text.lower() == "none":
        return None

    path = Path(text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


# ---------------------------------------------------------------------
# Per-epoch validation logging
# ---------------------------------------------------------------------

def get_nested_attr(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj

    for name in path:
        if cur is None or not hasattr(cur, name):
            return None
        cur = getattr(cur, name)

    return scalar_to_python(cur)


def get_trainer_epoch(trainer: Trainer) -> int | float | str | None:
    for path in [
        ("current_epoch",),
        ("epoch",),
        ("global_epoch",),
        ("_current_epoch",),
        ("_epoch",),
        ("state", "current_epoch"),
        ("state", "epoch"),
        ("state", "global_epoch"),
    ]:
        value = get_nested_attr(trainer, path)

        if isinstance(value, (int, float, str)):
            return value

    return None


def get_trainer_step(trainer: Trainer) -> int | float | str | None:
    for path in [
        ("global_step",),
        ("step",),
        ("current_step",),
        ("_global_step",),
        ("state", "global_step"),
        ("state", "step"),
        ("state", "current_step"),
    ]:
        value = get_nested_attr(trainer, path)

        if isinstance(value, (int, float, str)):
            return value

    return None


def iter_metric_dicts_from_object(obj: Any, *, depth: int = 0):
    if obj is None or depth > 2:
        return

    if isinstance(obj, Mapping):
        yield obj
        return

    for attr in [
        "metrics",
        "val_metrics",
        "validation_metrics",
        "logged_metrics",
        "callback_metrics",
        "logs",
        "log_dict",
    ]:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, Mapping):
                yield value

    if isinstance(obj, (tuple, list)):
        for item in obj:
            yield from iter_metric_dicts_from_object(item, depth=depth + 1)


def iter_metric_dicts_from_call(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    include_trainer_state: bool,
    trainer: Trainer,
    metric_key: str,
):
    if len(args) >= 2 and isinstance(args[0], str):
        yield {args[0]: args[1]}

    if "name" in kwargs and "value" in kwargs and isinstance(kwargs["name"], str):
        yield {kwargs["name"]: kwargs["value"]}

    for key in [
        "metrics",
        "metric_dict",
        "logs",
        "log_dict",
        "val_metrics",
        "validation_metrics",
    ]:
        value = kwargs.get(key)
        if isinstance(value, Mapping):
            yield value

    if metric_key in kwargs:
        yield kwargs

    for obj in [*args, *kwargs.values(), result]:
        yield from iter_metric_dicts_from_object(obj)

    if include_trainer_state:
        yield from iter_metric_dicts_from_object(trainer)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def attach_epoch_val_ndcg_printer(
    *,
    trainer: Trainer,
    dataset_name: str,
    trial_number: int,
    output_dir: Path,
    metric_key: str = MONITOR,
) -> None:
    if not PRINT_EPOCH_VAL_NDCG:
        return

    state: dict[str, Any] = {
        "printed_keys": set(),
        "fallback_epoch": 0,
    }

    log_path = output_dir / EPOCH_VAL_NDCG_LOG_FILENAME

    def maybe_print(
        *,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> None:
        include_trainer_state = any(
            token in method_name.lower()
            for token in ["val", "valid", "eval"]
        )

        val_ndcg10: float | None = None

        for metrics in iter_metric_dicts_from_call(
            args=args,
            kwargs=kwargs,
            result=result,
            include_trainer_state=include_trainer_state,
            trainer=trainer,
            metric_key=metric_key,
        ):
            try:
                val_ndcg10 = extract_val_ndcg10(metrics, metric_key)
                break
            except Exception:
                continue

        if val_ndcg10 is None:
            return

        raw_epoch = get_trainer_epoch(trainer)
        raw_step = get_trainer_step(trainer)

        dedup_epoch = raw_epoch if raw_epoch is not None else "no_epoch"
        dedup_step = raw_step if raw_step is not None else "no_step"
        dedup_key = (str(dedup_epoch), str(dedup_step), round(float(val_ndcg10), 12))

        if raw_epoch is None and raw_step is None:
            dedup_key = ("no_context", round(float(val_ndcg10), 12))

        if dedup_key in state["printed_keys"]:
            return

        state["printed_keys"].add(dedup_key)

        if raw_epoch is None:
            state["fallback_epoch"] += 1
            epoch = state["fallback_epoch"]
        else:
            epoch = raw_epoch

        step = raw_step

        payload = {
            "dataset": dataset_name,
            "trial": trial_number,
            "epoch": epoch,
            "step": step,
            "metric": metric_key,
            "val_ndcg10": float(val_ndcg10),
        }

        if step is None:
            print(
                f"[dataset={dataset_name} | trial={trial_number} | epoch={epoch}] "
                f"{metric_key}={val_ndcg10:.6f}",
                flush=True,
            )
        else:
            print(
                f"[dataset={dataset_name} | trial={trial_number} | epoch={epoch} | step={step}] "
                f"{metric_key}={val_ndcg10:.6f}",
                flush=True,
            )

        append_jsonl(log_path, payload)

    candidate_method_names = [
        "_log_metrics",
        "log_metrics",
        "log_dict",
        "log",
        "_validate_epoch",
        "validate_epoch",
        "_run_validation",
        "run_validation",
        "_validate",
        "validate",
        "_evaluate",
        "evaluate",
    ]

    wrapped_any = False

    for method_name in candidate_method_names:
        original = getattr(trainer, method_name, None)

        if not callable(original):
            continue

        def wrapped(*args, __original=original, __method_name=method_name, **kwargs):
            result = __original(*args, **kwargs)
            maybe_print(
                method_name=__method_name,
                args=args,
                kwargs=kwargs,
                result=result,
            )
            return result

        setattr(trainer, method_name, wrapped)
        wrapped_any = True

    if not wrapped_any:
        print(
            "[Val logger] No Trainer validation/logging hook found. "
            "The final and best validation NDCG@10 will still be printed after fit().",
            flush=True,
        )


def read_epoch_val_records(
    *,
    log_path: Path,
    dataset_name: str,
    trial_number: int,
) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    records: list[dict[str, Any]] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            if record.get("dataset") != dataset_name:
                continue

            if int(record.get("trial", -1)) != int(trial_number):
                continue

            if "val_ndcg10" not in record:
                continue

            records.append(record)

    return records


def best_logged_epoch_record(
    *,
    output_dir: Path,
    dataset_name: str,
    trial_number: int,
) -> dict[str, Any] | None:
    records = read_epoch_val_records(
        log_path=output_dir / EPOCH_VAL_NDCG_LOG_FILENAME,
        dataset_name=dataset_name,
        trial_number=trial_number,
    )

    if not records:
        return None

    return max(records, key=lambda row: float(row["val_ndcg10"]))


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------

def trial_checkpoint_dir(output_dir: Path, trial_number: int) -> Path:
    return output_dir / "trial_checkpoints" / f"trial_{trial_number}"


def find_best_checkpoint_file(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    patterns = [
        "*best*.pt",
        "*best*.pth",
        "*best*.ckpt",
        "*.pt",
        "*.pth",
        "*.ckpt",
    ]

    candidates: list[Path] = []

    for pattern in patterns:
        candidates.extend(checkpoint_dir.glob(pattern))

    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_best_checkpoint_path_from_fit_result(
    *,
    fit_result: Any,
    checkpoint_dir: Path,
) -> Path | None:
    best_checkpoint_path = path_or_none(getattr(fit_result, "best_checkpoint_path", None))

    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        return best_checkpoint_path

    return find_best_checkpoint_file(checkpoint_dir)


def load_best_checkpoint_into_model(
    *,
    trainer: Trainer,
    model: torch.nn.Module,
    best_checkpoint_path: Path | None,
    dataset_name: str,
    trial_number: int,
) -> Path | None:
    if best_checkpoint_path is None:
        print(
            f"[{dataset_name} | Trial {trial_number}] No best checkpoint found. "
            "Using in-memory model state from the end of fit().",
            flush=True,
        )
        return None

    print(
        f"[{dataset_name} | Trial {trial_number}] Loading best epoch checkpoint: "
        f"{best_checkpoint_path}",
        flush=True,
    )

    if hasattr(trainer, "load_checkpoint") and callable(getattr(trainer, "load_checkpoint")):
        trainer.load_checkpoint(
            best_checkpoint_path,
            load_optimizer=False,
            load_scheduler=False,
            strict=True,
        )
        return best_checkpoint_path

    checkpoint = torch.load(best_checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    return best_checkpoint_path


def copy_trial_best_checkpoint(
    *,
    best_checkpoint_path: Path | None,
    output_dir: Path,
    trial_number: int,
) -> Path | None:
    if best_checkpoint_path is None or not best_checkpoint_path.exists():
        return None

    if not COPY_EACH_TRIAL_BEST_CHECKPOINT:
        return best_checkpoint_path

    dst_dir = output_dir / "trial_best_checkpoints"
    dst_dir.mkdir(parents=True, exist_ok=True)

    suffix = best_checkpoint_path.suffix or ".pt"
    dst_path = dst_dir / f"trial_{trial_number}_best_epoch{suffix}"

    if best_checkpoint_path.resolve() != dst_path.resolve():
        shutil.copy2(best_checkpoint_path, dst_path)

    return dst_path


def copy_study_best_checkpoint(
    *,
    study: optuna.Study,
    output_dir: Path,
) -> Path | None:
    source_text = (
        study.best_trial.user_attrs.get("trial_best_epoch_checkpoint_path")
        or study.best_trial.user_attrs.get("loaded_best_checkpoint_path")
        or study.best_trial.user_attrs.get("best_checkpoint_path")
    )

    source_path = path_or_none(source_text)

    if source_path is None or not source_path.exists():
        return None

    suffix = source_path.suffix or ".pt"
    dst_path = output_dir / f"best_trial_best_epoch_checkpoint{suffix}"
    shutil.copy2(source_path, dst_path)

    return dst_path


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def build_datamodule(
    *,
    dataset_name: str,
    dataset_info: dict[str, str],
    seed: int,
) -> AmazonReviewDataModule:
    data_cfg = make_config(
        AmazonReviewDataConfig,
        {
            "reviews_path": path_from_repo(dataset_info["reviews_path"]),
            "metadata_path": path_from_repo(dataset_info["metadata_path"]),
            "item_key": dataset_info["item_key"],
            "metadata_item_key": dataset_info.get("metadata_item_key", "parent_asin"),
            "split_strategy": "leave_one_out",
            "train_example_layout": TRAIN_EXAMPLE_LAYOUT,
            "train_window_len": TRAIN_WINDOW_LEN,
            "train_window_stride": TRAIN_WINDOW_STRIDE,
            "history_padding": "right",
            "train_all_prefixes": True,
            "min_user_interactions": 5,
            "min_item_interactions": 5,
            "max_history_len": 20,
            "deduplicate_consecutive": False,
            "filter_cold_items_after_split": False,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": NUM_WORKERS > 0,
            "offset_semantic_ids_by_one": True,
            "seed": seed,
        },
    )

    dm = AmazonReviewDataModule(data_cfg)
    dm.setup()

    print(f"\nDataset: {dataset_name}")
    print("Dataset info:")
    print(json.dumps(dataset_info, indent=2))
    print("Dataset summary:")
    print(json.dumps(dm.summary(), indent=2, default=json_default))

    return dm


# ---------------------------------------------------------------------
# Optuna search space
# ---------------------------------------------------------------------

def suggest_sasrec_hparams(trial: optuna.Trial) -> dict[str, Any]:
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead_candidates = [h for h in [1, 2, 4, 8] if d_model % h == 0]

    return {
        "d_model": d_model,
        "nhead": trial.suggest_categorical("nhead", nhead_candidates),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.05, 0.50),
        "pos_drop": trial.suggest_float("pos_drop", 0.05, 0.50),
        "initializer_range": trial.suggest_categorical("initializer_range", [0.01, 0.02, 0.05]),
        "layer_norm_eps": trial.suggest_categorical("layer_norm_eps", [1e-8, 1e-6, 1e-5]),
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0, 5.0]),
    }


# ---------------------------------------------------------------------
# Model / method / trainer
# ---------------------------------------------------------------------

def build_model(
    *,
    dm: AmazonReviewDataModule,
    hparams: dict[str, Any],
) -> SASRec:
    cfg = make_config(
        SASRecConfig,
        {
            "item_vocab_size": get_item_vocab_size(dm),
            # Positional embeddings must cover dense training windows too.
            "max_history_len": max(
                int(dm.cfg.max_history_len), int(dm.cfg.train_window_len or 0)
            ),
            "d_model": hparams["d_model"],
            "nhead": hparams["nhead"],
            "num_layers": hparams["num_layers"],
            "dropout": hparams["dropout"],
            "pos_drop": hparams["pos_drop"],
            "initializer_range": hparams["initializer_range"],
            "layer_norm_eps": hparams["layer_norm_eps"],
            "exclude_history_in_generate": False,
        },
    )

    model = SASRec(cfg)

    layout = dm.cfg.train_example_layout
    if layout not in model.supported_train_layouts:
        raise ValueError(
            f"{model.__class__.__name__} does not support train_example_layout="
            f"{layout!r}; supported: {model.supported_train_layouts}."
        )

    print(model)
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    return model


def build_method() -> SFTMethod:
    cfg = make_config(
        SFTConfig,
        {
            "loss_weight": 1.0,
            "metric_prefix": "sft",
            "compute_train_metrics": False,
            "compute_eval_metrics": True,
            "compute_item_metrics": True,
            "item_topk": KS,
            "compute_token_metrics": False,
            "evaluate_generation": False,
            "pad_item_id": 0,
        },
    )

    return SFTMethod(cfg)


def build_trainer(
    *,
    model: SASRec,
    method: SFTMethod,
    trial: optuna.Trial,
    hparams: dict[str, Any],
    output_dir: Path,
    device: str,
) -> Trainer:
    checkpoint_dir = trial_checkpoint_dir(output_dir, trial.number)

    cfg = make_config(
        TrainerConfig,
        {
            "max_epochs": MAX_EPOCHS,
            "validate_every_epochs": 1,
            "validate_every_steps": None,
            "monitor": MONITOR,
            "monitor_mode": "max",
            "early_stopping_patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_grad_norm": hparams["max_grad_norm"],
            "use_amp": USE_AMP,
            "amp_dtype": AMP_DTYPE,
            "checkpoint_dir": checkpoint_dir,

            # Important: the trial result model is the best epoch, not the last epoch.
            "save_best": SAVE_BEST_CHECKPOINTS,
            "save_last": SAVE_LAST_CHECKPOINT,

            "log_every_steps": LOG_EVERY_STEPS,
            "progress_bar": PROGRESS_BAR,
            "prefix_method_metrics_with_stage": True,
        },
    )

    return Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=cfg,
        device=device,
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------

def make_objective(
    *,
    dataset_name: str,
    dm: AmazonReviewDataModule,
    output_dir: Path,
    device: str,
):
    def objective(trial: optuna.Trial) -> float:
        trial_seed = SEED + trial.number
        set_seed(trial_seed)

        hparams = suggest_sasrec_hparams(trial)

        model = None
        trainer = None

        try:
            model = build_model(dm=dm, hparams=hparams)
            method = build_method()

            trainer = build_trainer(
                model=model,
                method=method,
                trial=trial,
                hparams=hparams,
                output_dir=output_dir,
                device=device,
            )

            attach_epoch_val_ndcg_printer(
                trainer=trainer,
                dataset_name=dataset_name,
                trial_number=trial.number,
                output_dir=output_dir,
                metric_key=MONITOR,
            )

            fit_result = trainer.fit(
                train_loader=dm.train_dataloader(),
                val_loader=dm.val_dataloader(),
            )

            checkpoint_dir = trial_checkpoint_dir(output_dir, trial.number)

            best_checkpoint_path = get_best_checkpoint_path_from_fit_result(
                fit_result=fit_result,
                checkpoint_dir=checkpoint_dir,
            )

            loaded_best_checkpoint_path = load_best_checkpoint_into_model(
                trainer=trainer,
                model=model,
                best_checkpoint_path=best_checkpoint_path,
                dataset_name=dataset_name,
                trial_number=trial.number,
            )

            trial_best_checkpoint_path = copy_trial_best_checkpoint(
                best_checkpoint_path=loaded_best_checkpoint_path,
                output_dir=output_dir,
                trial_number=trial.number,
            )

            val_metrics = to_jsonable(getattr(fit_result, "val_metrics", None))
            logged_best = best_logged_epoch_record(
                output_dir=output_dir,
                dataset_name=dataset_name,
                trial_number=trial.number,
            )

            fit_best_metric = extract_best_metric_from_fit_result(fit_result, MONITOR)
            fit_best_epoch = extract_best_epoch_from_fit_result(fit_result)

            if fit_best_metric is not None:
                val_ndcg10 = float(fit_best_metric)
            elif logged_best is not None:
                val_ndcg10 = float(logged_best["val_ndcg10"])
            else:
                val_ndcg10 = extract_val_ndcg10(val_metrics, MONITOR)

            if fit_best_epoch is not None:
                best_epoch = fit_best_epoch
            elif logged_best is not None:
                best_epoch = logged_best.get("epoch")
            else:
                best_epoch = None

            trial.set_user_attr("dataset", dataset_name)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_val_ndcg10", float(val_ndcg10))
            trial.set_user_attr("val_metrics", val_metrics)

            trial.set_user_attr(
                "best_checkpoint_path",
                str(best_checkpoint_path) if best_checkpoint_path is not None else None,
            )
            trial.set_user_attr(
                "loaded_best_checkpoint_path",
                str(loaded_best_checkpoint_path) if loaded_best_checkpoint_path is not None else None,
            )
            trial.set_user_attr(
                "trial_best_epoch_checkpoint_path",
                str(trial_best_checkpoint_path) if trial_best_checkpoint_path is not None else None,
            )
            trial.set_user_attr(
                "last_checkpoint_path",
                str(getattr(fit_result, "last_checkpoint_path", None)),
            )
            trial.set_user_attr(
                "logged_best_epoch_record",
                logged_best,
            )

            report_step = int(best_epoch) if isinstance(best_epoch, int) else 0
            trial.report(val_ndcg10, step=report_step)

            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned with best_val_ndcg10={val_ndcg10:.6f}")

            print(
                f"[{dataset_name} | Trial {trial.number}] "
                f"best_epoch={best_epoch} best_val_ndcg@10={val_ndcg10:.6f}",
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "hparams": hparams,
                        "best_epoch": best_epoch,
                        "best_val_ndcg10": float(val_ndcg10),
                        "best_checkpoint_path": str(best_checkpoint_path)
                        if best_checkpoint_path is not None
                        else None,
                        "trial_best_epoch_checkpoint_path": str(trial_best_checkpoint_path)
                        if trial_best_checkpoint_path is not None
                        else None,
                        "val_metrics": val_metrics,
                    },
                    indent=2,
                    default=json_default,
                )
            )

            return float(val_ndcg10)

        except RuntimeError as exc:
            message = str(exc).lower()

            if "out of memory" in message or ("cuda" in message and "memory" in message):
                print(f"[{dataset_name} | Trial {trial.number}] CUDA OOM; pruning trial.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned("CUDA out of memory") from exc

            raise

        finally:
            del trainer
            del model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not KEEP_ALL_TRIAL_CHECKPOINT_DIRS:
                trial_dir = trial_checkpoint_dir(output_dir, trial.number)
                if trial_dir.exists():
                    shutil.rmtree(trial_dir, ignore_errors=True)

    return objective


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_study_outputs(
    *,
    study: optuna.Study,
    dataset_name: str,
    dataset_info: dict[str, str],
    output_dir: Path,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)

    study_best_checkpoint_path = copy_study_best_checkpoint(
        study=study,
        output_dir=output_dir,
    )

    best_payload = {
        "dataset": dataset_name,
        "dataset_info": dataset_info,
        "study_name": study.study_name,
        "direction": "maximize",
        "metric": MONITOR,
        "best_trial_number": study.best_trial.number,
        "best_epoch": study.best_trial.user_attrs.get("best_epoch"),
        "best_value_val_ndcg10": study.best_value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "best_trial_best_epoch_checkpoint": str(study_best_checkpoint_path)
        if study_best_checkpoint_path is not None
        else None,
        "fixed_data_config": {
            "split_strategy": "leave_one_out",
            "train_example_layout": TRAIN_EXAMPLE_LAYOUT,
            "train_window_len": TRAIN_WINDOW_LEN,
            "train_window_stride": TRAIN_WINDOW_STRIDE,
            "history_padding": "right",
            "train_all_prefixes": True,
            "min_user_interactions": 5,
            "min_item_interactions": 5,
            "max_history_len": 20,
            "deduplicate_consecutive": False,
            "filter_cold_items_after_split": False,
            "offset_semantic_ids_by_one": True,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "seed": SEED,
        },
    }

    with (output_dir / "best_sasrec_optuna.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, sort_keys=True, default=json_default)

    trials_payload = []

    for t in study.trials:
        trials_payload.append(
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
        )

    with (output_dir / "all_sasrec_optuna_trials.json").open("w", encoding="utf-8") as f:
        json.dump(trials_payload, f, indent=2, sort_keys=True, default=json_default)

    print(f"\n[{dataset_name}] Saved best result to: {output_dir / 'best_sasrec_optuna.json'}")
    print(f"[{dataset_name}] Saved all trials to:  {output_dir / 'all_sasrec_optuna_trials.json'}")
    print(f"[{dataset_name}] Saved epoch val logs to: {output_dir / EPOCH_VAL_NDCG_LOG_FILENAME}")

    if study_best_checkpoint_path is not None:
        print(f"[{dataset_name}] Saved best trial best-epoch checkpoint to: {study_best_checkpoint_path}")

    return study_best_checkpoint_path


# ---------------------------------------------------------------------
# Per-dataset tuning
# ---------------------------------------------------------------------

def tune_one_dataset(dataset_name: str, dataset_info: dict[str, str]) -> tuple[optuna.Study, Path | None]:
    if dataset_name not in AMAZON_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available datasets: {sorted(AMAZON_DATASETS)}"
        )

    output_dir = BASE_OUTPUT_DIR / f"sasrec_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"sasrec_{dataset_name}_val_ndcg10"
    storage = f"sqlite:///{output_dir / 'optuna.db'}"

    print("\n" + "=" * 100)
    print(f"Tuning SASRec on dataset: {dataset_name}")
    print("=" * 100)
    print(f"Project root:    {PROJECT_ROOT}")
    print(f"Output dir:      {output_dir}")
    print(f"Study name:      {study_name}")
    print(f"Optuna storage:  {storage}")
    print(f"Device:          {DEVICE}")
    print(f"Use AMP:         {USE_AMP}")
    print(f"AMP dtype:       {AMP_DTYPE}")
    print(f"N trials:        {N_TRIALS}")
    print(f"Max epochs:      {MAX_EPOCHS}")
    print(f"Patience:        {PATIENCE}")
    print(f"Save best ckpt:  {SAVE_BEST_CHECKPOINTS}")
    print(f"Save last ckpt:  {SAVE_LAST_CHECKPOINT}")

    dm = build_datamodule(
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        seed=SEED,
    )

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=True,
    )

    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(
        dataset_name=dataset_name,
        dm=dm,
        output_dir=output_dir,
        device=DEVICE,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        gc_after_trial=True,
        show_progress_bar=PROGRESS_BAR,
    )

    print(f"\n[{dataset_name}] Best validation NDCG@10: {study.best_value}")
    print(f"[{dataset_name}] Best trial: {study.best_trial.number}")
    print(f"[{dataset_name}] Best epoch: {study.best_trial.user_attrs.get('best_epoch')}")
    print(f"[{dataset_name}] Best params:")
    print(json.dumps(study.best_trial.params, indent=2, sort_keys=True))

    study_best_checkpoint_path = save_study_outputs(
        study=study,
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        output_dir=output_dir,
    )

    del dm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return study, study_best_checkpoint_path


# ---------------------------------------------------------------------
# Main: no terminal arguments needed
# ---------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    completed: dict[str, dict[str, Any]] = {}

    for dataset_name in DATASETS_TO_TUNE:
        dataset_info = AMAZON_DATASETS[dataset_name]
        study, study_best_checkpoint_path = tune_one_dataset(dataset_name, dataset_info)

        completed[dataset_name] = {
            "best_value_val_ndcg10": study.best_value,
            "best_trial_number": study.best_trial.number,
            "best_epoch": study.best_trial.user_attrs.get("best_epoch"),
            "best_params": study.best_trial.params,
            "best_trial_best_epoch_checkpoint": str(study_best_checkpoint_path)
            if study_best_checkpoint_path is not None
            else None,
            "trial_best_epoch_checkpoint_path": study.best_trial.user_attrs.get(
                "trial_best_epoch_checkpoint_path"
            ),
            "output_dir": str(BASE_OUTPUT_DIR / f"sasrec_{dataset_name}"),
            "epoch_val_ndcg10_log": str(
                BASE_OUTPUT_DIR / f"sasrec_{dataset_name}" / EPOCH_VAL_NDCG_LOG_FILENAME
            ),
        }

    summary_path = BASE_OUTPUT_DIR / "sasrec_optuna_all_datasets_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(completed, f, indent=2, sort_keys=True, default=json_default)

    print("\n" + "=" * 100)
    print("Finished tuning all datasets from AMAZON_DATASETS.")
    print(json.dumps(completed, indent=2, sort_keys=True))
    print(f"Saved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()