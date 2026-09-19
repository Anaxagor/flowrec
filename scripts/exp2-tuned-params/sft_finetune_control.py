#!/usr/bin/env python
"""SFT-only fine-tuning control on the exp2 tuned-params TIGER checkpoints.

Hypothesis under test: does continuing plain supervised fine-tuning (token
cross-entropy only) at a LOWER learning rate already deliver the quality we
attribute to GFlow / RL fine-tuning? If yes, the reward-driven objectives are
unnecessary; if the SFT-only runs plateau at the init checkpoint's quality
while GFlow/RL improve, the gain is attributable to the reward signal.

This is the no-reward control for
``scripts/exp2-tuned-params/gflow_finetune_ablation.py``: identical init
checkpoint, data, optimizer, effective batch, schedule, validation monitor and
test protocol -- the ONLY difference is the loss (SFT only: no GFlow term, no
candidate sampler, no reward model).

Per run ({beauty, instruments} x --lr values x --seeds, default 42/43/44):
  - TIGER init: checkpoints/exp2-tuned_params/initial/tiger/<ds>/best.pt
    (tuned SFT checkpoints; trained at lr ~1.2e-3 (beauty) / ~1.7e-3
    (instruments), see configs/exp2-tuned-params/tiger_<ds>_sft.json).
    Every seed starts from the SAME weights; the seed controls the training
    shuffle order, dropout masks and AMP/RNG state.
  - Data: the checkpoint's own training config (same catalog / SID table /
    split), only the parquet paths are re-pointed to data/raw/dataframe_format.
  - Loss: SFT token cross-entropy only.
  - Optimizer: AdamW, lr = --lr (default 3e-5, ~40-55x below the tuned lr and
    identical to the GFlow ablation), weight decay 1e-2, grad clip 0.5,
    effective batch 256 x 8 accumulation = 2048 (= the tuned training batch).
  - Validation: constrained beam-5 NDCG@5 (``val/sft/ndcg@5``) every
    --validate-every epochs, <= --max-epochs (500), early stopping after
    --patience validations without improvement.
  - Final evaluation of best.pt: constrained beam-20 HR / NDCG / Coverage /
    ARP @ {5, 10, 20} on BOTH the validation and the test split (the exp1/exp2
    baseline protocol, so rows are comparable with the SASRec/TIGER tables in
    artifacts/). The SAME numbers are computed once per dataset for the
    untouched init checkpoint (baseline).

Outputs:
  experiments/exp2-sft-finetune/<ds>/init_baseline.json
  experiments/exp2-sft-finetune/<ds>/sft_lr<lr>/seed_<N>/   run dir: checkpoints/,
      run_config.json (replayable with scripts/run_tiger.py), resolved_config.json,
      training_log.jsonl, training_history.json, metrics.json
  checkpoints/exp2-tuned_params/finetuned/tiger/<ds>/sft_only_lr<lr>/seed_<N>/
      best.pt + best.pt.json + metrics.json + run_config.json   (promoted)
  checkpoints/exp2-tuned_params/finetuned/tiger/<ds>/sft_only_lr<lr>/training_summary.json
      per-seed + mean/std over seeds + init baseline
  artifacts/exp2-tuned-params/tiger_sft_finetune_<ds>_metrics_summary.csv
      one row per (variant, lr, seed, split); variant=init rows carry the
      untouched checkpoint (upserted, so restarts / extra seeds never lose rows)
  artifacts/exp2-tuned-params/tiger_sft_finetune_<ds>_summary.json
      per-lr aggregate (mean/std over seeds) + init baseline + protocol

Concurrency: (dataset, seed) pairs may run as separate Slurm array tasks
(scripts/slurm/sft_finetune_control_array.sbatch). Every task writes only its
own run dir; the CSV / JSON summaries and the promotion training_summary.json
are always REBUILT from the metrics.json files found on disk (under a file
lock), and the init baseline is computed once under a lock and reused, so
whichever task finishes last leaves complete summaries. ``--aggregate-only``
rebuilds all summaries from disk without touching data or GPUs.

Examples:
  python scripts/exp2-tuned-params/sft_finetune_control.py --device cuda
  python scripts/exp2-tuned-params/sft_finetune_control.py --device cuda \
      --lr 1e-4 3e-5 1e-5 --seeds 42 43 44        # LR sweep, three seeds each
  python scripts/exp2-tuned-params/sft_finetune_control.py \
      --datasets beauty --seeds 42 --smoke --num-workers 0 --device cpu   # wiring check
  python scripts/exp2-tuned-params/sft_finetune_control.py --aggregate-only  # rebuild summaries
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402
import sasrec_config_runner as sasrec_runner  # noqa: E402  (training_item_popularity for ARP)
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.ft_methods.sft.method import SFTMethod  # noqa: E402
from flowrec.models.tiger import TigerModel  # noqa: E402
from flowrec.training import Trainer, TrainerConfig  # noqa: E402
from flowrec.training.loggers import CompositeExperimentLogger, JsonlExperimentLogger  # noqa: E402


# --------------------------------------------------------------------------- #
# Assets (PROJECT_ROOT-relative; made absolute in the run config)
# --------------------------------------------------------------------------- #
# Spec keys: sft_config (run_tiger-style JSON with the checkpoint's own
# training config), tiger_checkpoint (init weights), data_overrides (config
# "data" entries re-pointed to local paths), and optional per-dataset defaults
# val_batch_size / val_sample_fraction / val_sample_resample (used when the CLI
# flag is not given). The yambda counterpart (sft_finetune_control_yambda.py)
# and the base-matched rerun (sft_finetune_control_matched.py) swap this dict.
DATASETS: dict[str, dict[str, Any]] = {
    "beauty": {
        "sft_config": "configs/exp2-tuned-params/tiger_beauty_sft.json",
        "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt",
        "data_overrides": {
            "reviews_path": "data/raw/dataframe_format/reviews_beauty_2014.parquet",
            "metadata_path": "data/raw/dataframe_format/meta_beauty_2014.parquet",
        },
    },
    "instruments": {
        "sft_config": "configs/exp2-tuned-params/tiger_instruments_sft.json",
        "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/instruments/best.pt",
        "data_overrides": {
            "reviews_path": "data/raw/dataframe_format/reviews_instruments_2018.parquet",
            "metadata_path": "data/raw/dataframe_format/meta_instruments_2018.parquet",
        },
    },
}

DEFAULT_OUTPUT_ROOT = Path("experiments/exp2-sft-finetune")
DEFAULT_VAL_BATCH_SIZE = 64

MONITOR = "val/sft/ndcg@5"
DEFAULT_SEEDS = (42, 43, 44)
# The init checkpoints were trained with seed 42 (cluster runs *_seed42).
INIT_SEED = 42
VARIANT_INIT = "init"
VARIANT_FT = "sft_finetune"
SPLIT_LOADERS = ("validation", "test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "SFT-only fine-tuning control (exp2): continue the tuned TIGER SFT "
        "checkpoints with token cross-entropy only at a lower learning rate."
    )
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="Training seeds; one run per seed from the same init weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer (defaults mirror gflow_finetune_ablation.py).
    p.add_argument("--lr", type=float, nargs="+", default=[3e-5],
                   help="Learning rate(s); one run group per value (default 3e-5, ~40-55x below the tuned lr)")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch-size x grad-accum)")
    p.add_argument("--val-batch-size", type=int, default=None,
                   help=f"Eval-loader batch size (default: dataset spec, else {DEFAULT_VAL_BATCH_SIZE})")
    p.add_argument("--num-workers", type=int, default=8)

    # Schedule / monitoring.
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10,
                   help=f"Early-stopping patience in validations of {MONITOR}; <= 0 disables")
    p.add_argument("--min-epochs", type=int, default=0,
                   help="Early-stopping warm-up: validations of the first N epochs never count "
                        "towards --patience (earliest stop = N + patience validations). Matches "
                        "the gflow arm when its negative-sampling curriculum lasts N epochs")
    p.add_argument("--val-num-beams", type=int, default=5,
                   help="Constrained beams for the validation NDCG@5 monitor")
    p.add_argument("--val-sample-fraction", type=float, default=None,
                   help="Validate on a random fraction of the val split during training "
                        "(default: dataset spec, else full validation)")
    p.add_argument("--train-random-n", type=int, default=None,
                   help="Fine-tune on N train examples per user drawn uniformly at random from the "
                        "checkpoint's layout (default: dataset spec, else the whole train set). "
                        "The gflow arm's --train-random-n: both arms then use identical examples")
    p.add_argument("--train-subset-seed", type=int, default=None,
                   help="Seed of that draw (default: dataset spec, else 0). Not the run seed: "
                        "every seed of the control fine-tunes on the same examples")
    p.add_argument("--val-sample-resample", dest="val_sample_resample",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Re-draw the validation sample at every validation (default: dataset "
                        "spec, else the trainer default = re-draw). --no-val-sample-resample "
                        "draws it once (seed 0): the rule of the gflow arm, whose re-drawn "
                        "monitor let the draw pick the best epoch")
    p.add_argument("--validate-at-start", dest="validate_at_start",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Score the init checkpoint before the first epoch: it seeds the best "
                        "value / best.pt and is the reference for early-stopping patience, so a "
                        "run that never improves stops after `patience` validations with the "
                        "init as its best")

    # Final evaluation (validation + test splits of best.pt and of the init).
    p.add_argument("--test-ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--eval-num-beams", type=int, default=20)
    p.add_argument("--skip-init-eval", action="store_true",
                   help="Do not evaluate the untouched init checkpoint (baseline)")

    # Output.
    p.add_argument("--output-root", type=Path, default=None,
                   help=f"Default: {DEFAULT_OUTPUT_ROOT}")
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts/exp2-tuned-params"),
                   help="Metrics CSV/JSON summaries are written here; pass '' to disable")
    p.add_argument("--promote-root", type=Path,
                   default=Path("checkpoints/exp2-tuned_params/finetuned/tiger"),
                   help="Best checkpoints are copied to <root>/<ds>/sft_only_lr<lr>/seed_<N>/; "
                        "pass '' to disable")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run runs whose metrics.json already exists (default: reuse them)")
    p.add_argument("--resume", action="store_true",
                   help="Resume incomplete runs (no metrics.json yet) from their "
                        "checkpoints/last.pt - optimizer, AMP and early-stopping state "
                        "included - instead of restarting from the init checkpoint. "
                        "The last recorded epoch is repeated once.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Only rebuild the artifacts CSV/JSON, promotion summaries and the console "
                        "summary from the metrics.json files on disk (no data, no GPU)")

    # Safety / debug.
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--sanity-nll-threshold", type=float, default=6.0,
                   help="Abort when the init checkpoint's teacher-forced val NLL exceeds this")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring run: 1 epoch, 8 train batches, 2 eval batches; all outputs "
                        "(runs, artifacts, promoted checkpoints) go under <output-root>_smoke/")
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error(f"--seeds must be distinct; got {args.seeds}")
    return args


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def abspath(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def lr_tag(lr: float) -> str:
    return f"{lr:g}"


def run_dir_for(output_root: Path, dataset: str, lr: float, seed: int) -> Path:
    return output_root / dataset / f"sft_lr{lr_tag(lr)}" / f"seed_{seed}"


def promote_group_dir(promote_root: Path, dataset: str, lr: float) -> Path:
    return promote_root / dataset / f"sft_only_lr{lr_tag(lr)}"


def load_tiger_checkpoint(model: TigerModel, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload
    if isinstance(payload, Mapping):
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
    # Cluster runs may carry a DB/SubTB flow head the local model doesn't have.
    state = {k: v for k, v in state.items() if not k.startswith("flow_head.")}
    model.load_state_dict(state, strict=True)


def finetune_subset_indices(dm, dataset: str, args: argparse.Namespace) -> list[int] | None:
    """Train-dataset indices of the fine-tuning subset, or None for the whole
    train set (the default). --train-random-n / the dataset spec's
    ``train_random_n``: N examples per user drawn uniformly at random from the
    checkpoint's layout with the subset seed - the SAME draw the gflow arm
    makes (precompute_reward_cache.per_user_random_indices), so both arms
    fine-tune on identical examples whatever the run seed."""
    spec = DATASETS[dataset]
    random_n = args.train_random_n if args.train_random_n is not None else spec.get("train_random_n", 0)
    if not random_n or int(random_n) <= 0:
        return None
    from precompute_reward_cache import per_user_random_indices

    seed = args.train_subset_seed if args.train_subset_seed is not None else spec.get("train_subset_seed", 0)
    return per_user_random_indices(dm.train_dataset, int(random_n), int(seed))


def seeded_train_loader(dm, seed: int, indices: list[int] | None = None):
    """``dm.train_dataloader()`` seeds its shuffle generator from ``dm.cfg.seed``.

    The datamodule is shared by all seeds (the leave-one-out split with
    train_target_policy=all is deterministic, and the data MUST equal the
    checkpoint's), so only the shuffle seed is swapped per run -- the same
    ``replace(dm.cfg, ...)`` pattern exp1's runner uses for eval batch sizes.
    With ``indices`` the loader is the same one over that subset of the train
    set (finetune_subset_indices); the datamodule, and with it the catalog,
    keep the checkpoint's full layout.
    """
    dm.cfg = replace(dm.cfg, seed=int(seed))
    if indices is None:
        return dm.train_dataloader()
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    base, collate_fn = dm._loader_parts(dm.train_dataset, dm._make_train_collator())
    return DataLoader(
        Subset(base, [int(i) for i in indices]),
        batch_size=dm.cfg.batch_size, shuffle=True, num_workers=dm.cfg.num_workers,
        pin_memory=dm.cfg.pin_memory,
        persistent_workers=dm.cfg.persistent_workers and dm.cfg.num_workers > 0,
        drop_last=dm.cfg.train_drop_last, collate_fn=collate_fn, generator=generator,
    )


def mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)  # sample std, like pandas
    return mean, math.sqrt(var)


@dataclass(frozen=True)
class Roots:
    output: Path
    artifacts: Path | None
    promote: Path | None


def resolve_roots(args: argparse.Namespace) -> Roots:
    output = abspath(args.output_root if args.output_root is not None else DEFAULT_OUTPUT_ROOT)
    artifacts = abspath(args.artifacts_root) if str(args.artifacts_root) else None
    promote = abspath(args.promote_root) if str(args.promote_root) else None
    if args.smoke:
        # Never let a wiring check pollute the real experiment / artifacts /
        # checkpoint trees: everything lands in one disposable folder.
        output = output.with_name(output.name + "_smoke")
        artifacts = output / "artifacts" if artifacts is not None else None
        promote = output / "checkpoints" if promote is not None else None
    return Roots(output=output, artifacts=artifacts, promote=promote)


@contextmanager
def file_lock(path: Path):
    """Advisory exclusive lock (fcntl.flock) so concurrent array tasks never
    interleave read-modify-write of a shared summary file. No-op where fcntl
    is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------------- #
# Config assembly: one run_tiger-compatible config per (dataset, lr, seed)
# --------------------------------------------------------------------------- #
def build_run_config(dataset: str, lr: float, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    """Start from the checkpoint's own training config, override only the
    fine-tuning knobs. The result is a complete run_tiger.py config."""
    spec = DATASETS[dataset]
    config = copy.deepcopy(load_json(abspath(spec["sft_config"])))

    config["experiment_name"] = f"sft_finetune_{dataset}_lr{lr_tag(lr)}_seed-{seed}"
    config["seed"] = int(seed)
    config["comet"] = {"enabled": False}
    data = dict(config.get("data", {}))
    data.update(dict(spec.get("data_overrides", {})))
    for key in ("reviews_path", "metadata_path", "interactions_path", "embeddings_path", "semantic_id_path"):
        if data.get(key):
            data[key] = str(abspath(data[key]))
    config["data"] = data

    dataloader = dict(config.get("dataloader", {}))
    dataloader["train_batch_size"] = int(args.batch_size)
    dataloader["batch_size"] = int(args.batch_size)
    val_batch_size = (
        args.val_batch_size
        if args.val_batch_size is not None
        else spec.get("val_batch_size", DEFAULT_VAL_BATCH_SIZE)
    )
    dataloader["validation_batch_size"] = int(val_batch_size)
    dataloader["num_workers"] = int(args.num_workers)
    dataloader["persistent_workers"] = int(args.num_workers) > 0
    config["dataloader"] = dataloader

    config["method"] = {"name": "sft"}

    smoke = bool(args.smoke)
    training = dict(config.get("training", {}))
    training.update(
        {
            "init_checkpoint_path": str(abspath(spec["tiger_checkpoint"])),
            "init_checkpoint_strict": True,
            "max_epochs": 1 if smoke else int(args.max_epochs),
            "max_steps": None,
            "validate_every_epochs": 1 if smoke else int(args.validate_every),
            "validate_every_steps": None,
            "gradient_accumulation_steps": int(args.grad_accum),
            "max_grad_norm": float(args.max_grad_norm),
            "use_amp": True,  # the trainer enables AMP on CUDA only
            "amp_dtype": "float16",
            "log_every_steps": 200,
            "progress_bar": False,
            "run_validation_at_start": bool(args.validate_at_start),
            "save_best": True,
            "save_last": True,
            "save_every_epochs": None,
            "monitor": MONITOR,
            "monitor_mode": "max",
            "early_stopping_patience": int(args.patience) if int(args.patience) > 0 else None,
            "early_stopping_mode": "best",
            "early_stopping_min_epochs": int(args.min_epochs),
            "min_delta": 0,
            "scheduler_step_on": "none",
            "evaluate_generation": True,
            "generation_constrain_valid_sequences": True,
            "generation_num_beams": int(args.val_num_beams),
            "generation_num_return_sequences": 5,
            "generation_topk": [5],
            "test_evaluate_generation": True,
            "test_generation_num_beams": int(args.eval_num_beams),
            "test_generation_num_return_sequences": max(int(k) for k in args.test_ks),
            "test_generation_topk": [int(k) for k in args.test_ks],
            "max_train_batches": 8 if smoke else args.max_train_batches,
            "max_eval_batches": 2 if smoke else args.max_eval_batches,
            "val_sample_fraction": (
                args.val_sample_fraction
                if args.val_sample_fraction is not None
                else spec.get("val_sample_fraction")
            ),
        }
    )
    val_sample_resample = (
        args.val_sample_resample
        if args.val_sample_resample is not None
        else spec.get("val_sample_resample")
    )
    if val_sample_resample is not None:  # unset: keep the trainer default
        training["val_sample_resample"] = bool(val_sample_resample)
    config["training"] = training

    optimizer = dict(config.get("optimizer", {}))
    optimizer["tuned_lr"] = optimizer.get("lr")  # keep the original for the record
    optimizer["lr"] = float(lr)
    optimizer["weight_decay"] = float(args.weight_decay)
    optimizer.setdefault("betas", [0.9, 0.999])
    optimizer.setdefault("eps", 1e-8)
    config["optimizer"] = optimizer
    config["scheduler"] = None
    return config


def build_sft_method(config: Mapping[str, Any], model: TigerModel, dm, config_path: Path) -> SFTMethod:
    method = tiger_runner.build_method(
        dict(config), model, data=dm, train_loader=None, config_path=config_path
    )
    if not isinstance(method, SFTMethod):
        raise TypeError(f"Expected an SFTMethod, got {type(method).__name__}")
    return method


def protocol_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "loss": "sft_only (token cross-entropy)",
        "optimizer": "AdamW",
        "learning_rates": [float(x) for x in args.lr],
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accum),
        "seeds": [int(s) for s in args.seeds],
        "max_epochs": int(args.max_epochs),
        "validate_every_epochs": int(args.validate_every),
        "early_stopping_patience": int(args.patience) if int(args.patience) > 0 else None,
        "early_stopping_min_epochs": int(args.min_epochs),
        "monitor": MONITOR,
        "val_num_beams": int(args.val_num_beams),
        "eval_num_beams": int(args.eval_num_beams),
        "eval_ks": [int(k) for k in args.test_ks],
        "constrain_valid_sequences": True,
        "smoke": bool(args.smoke),
    }


# --------------------------------------------------------------------------- #
# Checkpoint sanity + evaluation (HR / NDCG / Coverage / ARP)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sanity_check_checkpoint(model: TigerModel, dm, device: torch.device, threshold: float) -> float:
    """Teacher-forced val NLL of the init; ~2-4 is healthy, ~15 is chance.

    Guards against silent data/checkpoint mismatches (catalog, SID table,
    ordering, runtime), which this repo has been bitten by before.
    """
    probe = SFTMethod(SFTConfig(compute_eval_metrics=False, compute_token_metrics=False,
                                evaluate_generation=False))
    model.eval()
    batch = move_batch(next(iter(dm.val_dataloader())), device)
    nll = float(probe.compute_loss(model, batch, stage="val").loss.item())
    if nll > threshold:
        raise RuntimeError(
            f"Init checkpoint sanity check FAILED: teacher-forced val NLL {nll:.3f} "
            f"> {threshold:.1f}. The checkpoint does not match the datamodule / runtime "
            "(catalog / semantic IDs / ordering / transformers version); refusing to "
            "fine-tune noise. Override with --skip-sanity-check or --sanity-nll-threshold."
        )
    return nll


class CatalogIndex:
    """Generated semantic-ID tuple -> catalog row -> item id / train popularity.

    Rows follow ``sorted(dm.semantic_ids)``, the order TigerModel uses for its
    ``item_semantic_ids`` buffer and ``run_tiger.item_id_by_semantic_row``.
    Popularity is the exp1/exp2 ``training_item_popularity`` (train-partition
    interaction counts), so ARP matches the SASRec/TIGER baseline tables.
    """

    def __init__(self, model: TigerModel, dm) -> None:
        sids = model.item_semantic_ids.detach().cpu().to(torch.int64)  # [rows, L]
        if sids.numel() == 0:
            raise ValueError("Model has no item_semantic_ids buffer; cannot evaluate.")
        self.sid_len = int(sids.size(1))
        self.base = int(sids.max().item()) + 1
        if self.base ** self.sid_len >= 2 ** 63:
            raise ValueError("Cannot pack semantic IDs into int64 keys.")
        keys = self._pack(sids)
        order = keys.argsort()
        self._keys = keys[order]
        self._rows = order
        self.row_to_item = tiger_runner.item_id_by_semantic_row(dm)
        if self.row_to_item is None or self.row_to_item.numel() != sids.size(0):
            raise ValueError("Semantic-row -> item mapping does not match the model catalog.")
        self.popularity = sasrec_runner.training_item_popularity(dm).to(torch.float64)
        self.num_catalog_items = int(dm.num_items)
        self.pad_item_id = int(getattr(dm.cfg, "pad_item_id", 0))

    def _pack(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1, self.sid_len)
        packed = torch.zeros(flat.size(0), dtype=torch.int64)
        for level in range(self.sid_len):
            packed = packed * self.base + flat[:, level].clamp(0, self.base - 1)
        return packed

    def rows(self, generated: torch.Tensor) -> torch.Tensor:
        """Catalog row per generated tuple ([...] -> int64), -1 when unknown."""
        flat = generated.detach().cpu().to(torch.int64).reshape(-1, generated.size(-1))
        in_range = (flat >= 0).all(dim=1) & (flat < self.base).all(dim=1)
        packed = self._pack(flat)
        pos = torch.searchsorted(self._keys, packed).clamp(max=self._keys.numel() - 1)
        found = in_range & self._keys[pos].eq(packed)
        rows = torch.where(found, self._rows[pos], torch.full_like(pos, -1))
        return rows.view(generated.shape[:-1])

    def items(self, rows: torch.Tensor) -> torch.Tensor:
        items = torch.full_like(rows, -1)
        known = rows.ge(0)
        items[known] = self.row_to_item[rows[known]]
        return items


@torch.no_grad()
def evaluate_split(
    model: TigerModel, loader, index: CatalogIndex, *,
    ks: tuple[int, ...], num_beams: int, max_batches: int | None = None,
) -> dict[str, float]:
    """Constrained beam-search HR / NDCG / Coverage / ARP at each k.

    HR/NDCG compare generated tuples against the target tuple (identical to
    the trainer's generation metrics); Coverage counts distinct recommended
    catalog items over ``dm.num_items``; ARP averages train-interaction counts
    of the recommended items that decode to catalog entries (exp1 definition).
    """
    model.eval()
    device = next(model.parameters()).device
    max_k = max(ks)

    n = 0
    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    covered = {k: torch.zeros(index.popularity.numel(), dtype=torch.bool) for k in ks}
    arp_sum = {k: 0.0 for k in ks}
    arp_cnt = {k: 0 for k in ks}

    for batch_idx, raw_batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        generated = model.generate(
            batch, num_beams=num_beams, num_return_sequences=max_k, constrain_valid_sequences=True,
        )
        target = batch["target_token_ids"].long()
        mask = batch.get("target_token_mask", torch.ones_like(target, dtype=torch.bool)).bool()
        matches = ((generated == target.unsqueeze(1)) | ~mask.unsqueeze(1)).all(dim=-1).cpu()
        n += int(target.size(0))
        rec_items = index.items(index.rows(generated))  # [B, max_k], -1 unknown

        for k in ks:
            m = matches[:, :k]
            hit = m.any(dim=1)
            rank = m.float().argmax(dim=1)
            hits[k] += float(hit.float().sum().item())
            if hit.any():
                ndcgs[k] += float((1.0 / torch.log2(rank[hit].float() + 2.0)).sum().item())
            topk = rec_items[:, :k]
            valid = topk.ge(0) & topk.ne(index.pad_item_id)
            item_ids = topk[valid]
            if item_ids.numel():
                covered[k][item_ids.unique()] = True
                arp_sum[k] += float(index.popularity[item_ids].sum().item())
                arp_cnt[k] += int(item_ids.numel())

    out: dict[str, float] = {"num_examples": float(n)}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / max(n, 1)
        out[f"NDCG@{k}"] = ndcgs[k] / max(n, 1)
        out[f"Coverage@{k}"] = float(covered[k].sum().item()) / max(index.num_catalog_items, 1)
        out[f"ARP@{k}"] = arp_sum[k] / max(arp_cnt[k], 1)
    return out


def evaluate_splits(
    model: TigerModel, dm, index: CatalogIndex, args: argparse.Namespace, max_batches: int | None,
) -> dict[str, dict[str, float]]:
    loaders = {"validation": dm.val_dataloader, "test": dm.test_dataloader}
    return {
        split: evaluate_split(
            model, loaders[split](), index,
            ks=tuple(int(k) for k in args.test_ks),
            num_beams=int(args.eval_num_beams),
            max_batches=max_batches,
        )
        for split in SPLIT_LOADERS
    }


def evaluate_init_baseline(
    *, dataset: str, config: Mapping[str, Any], config_path: Path, dm, index: CatalogIndex,
    device: torch.device, args: argparse.Namespace,
) -> dict[str, Any]:
    """The untouched init checkpoint under the exact run protocol: the beam-5
    val monitor value plus beam-20 HR/NDCG/Coverage/ARP on validation + test."""
    model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    load_tiger_checkpoint(model, abspath(DATASETS[dataset]["tiger_checkpoint"]))
    method = build_sft_method(config, model, dm, config_path)
    trainer_cfg = replace(
        tiger_runner.build_trainer_config(config, config_path.parent),
        checkpoint_dir=None, save_best=False, save_last=False,
    )
    trainer = Trainer.from_optimizer_config(
        model=model, method=method, cfg=trainer_cfg, device=device, lr=0.0, logger=None,
    )
    monitor_metrics = trainer.evaluate(dm.val_dataloader(), stage="val")
    splits = evaluate_splits(model, dm, index, args, trainer_cfg.max_eval_batches)
    baseline = {
        "dataset": dataset,
        "variant": VARIANT_INIT,
        "seed": INIT_SEED,
        "lr": config["optimizer"].get("tuned_lr"),
        "checkpoint": str(abspath(DATASETS[dataset]["tiger_checkpoint"])),
        "monitor_val": {k: float(v) for k, v in monitor_metrics.items()},
        "val_ndcg5": float(monitor_metrics.get(MONITOR, float("nan"))),
        **splits,
    }
    del model, method, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return baseline


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def make_status_logger(tag: str):
    keys = (
        "train/loss", "train/sft/token_accuracy", "val/loss", "val/sft/loss",
        MONITOR, "val/sft/recall@5",
    )

    def logger(payload: dict[str, Any]) -> None:
        if payload.get("type") not in {"epoch_end", "val_end", "test_end", "non_finite_loss_skipped"}:
            return
        parts = [f"{k}={payload[k]:.5g}" for k in keys if isinstance(payload.get(k), (int, float))]
        for block_key in ("train", "val"):
            block = payload.get(block_key, {})
            if isinstance(block, Mapping):
                parts += [f"{k}={block[k]:.5g}" for k in keys if isinstance(block.get(k), (int, float))]
        epoch = payload.get("epoch")
        header = f"[{tag}] {payload.get('type')}" + (
            f" epoch={int(epoch) + 1}" if isinstance(epoch, int) else ""
        )
        if payload.get("type") == "epoch_end" and isinstance(payload.get("epoch_time_sec"), (int, float)):
            parts.append(f"epoch_time={payload['epoch_time_sec']:.1f}s")
        print(header + (" | " + " | ".join(parts) if parts else ""), flush=True)

    return logger


# --------------------------------------------------------------------------- #
# One run (dataset, lr, seed)
# --------------------------------------------------------------------------- #
def run_single(
    *,
    dataset: str,
    lr: float,
    seed: int,
    dm,
    index: CatalogIndex,
    baseline: Mapping[str, Any] | None,
    run_dir: Path,
    roots: Roots,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tag = f"{dataset}/sft_lr{lr_tag(lr)}/seed_{seed}"
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_run_config(dataset, lr, seed, args)
    config_path = run_dir / "run_config.json"
    allow_resume = bool(args.resume)
    train_indices = finetune_subset_indices(dm, dataset, args)
    if train_indices is not None:
        print(f"[{tag}] fine-tuning subset: {len(train_indices)} of {len(dm.train_dataset)} train examples "
              f"(random per user)", flush=True)
    if config_path.exists():
        old_init = load_json(config_path).get("training", {}).get("init_checkpoint_path")
        new_init = config["training"]["init_checkpoint_path"]
        if old_init and old_init != new_init:
            if not args.overwrite:
                raise RuntimeError(
                    f"[{tag}] run dir {run_dir} holds a run started from a different "
                    f"init checkpoint:\n  old: {old_init}\n  new: {new_init}\n"
                    "Archive (mv) the run dir to keep the old arm, or pass --overwrite "
                    "to redo it from the new init."
                )
            print(
                f"[{tag}] --overwrite: discarding run started from {old_init}; "
                f"training fresh from {new_init} (resume disabled for this run)",
                flush=True,
            )
            allow_resume = False
    write_json(config_path, config)

    # Same init weights for every seed; the seed drives shuffle / dropout / RNG.
    tiger_runner.seed_everything(int(seed))
    model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    load_tiger_checkpoint(model, abspath(DATASETS[dataset]["tiger_checkpoint"]))
    method = build_sft_method(config, model, dm, config_path)
    trainer_cfg = tiger_runner.build_trainer_config(config, run_dir)

    optimizer_cfg = config["optimizer"]
    logger = CompositeExperimentLogger(
        [JsonlExperimentLogger(run_dir / "training_log.jsonl"), make_status_logger(tag)]
    )
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=trainer_cfg,
        device=device,
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
        logger=logger,
    )

    resolved = {
        "dataset": dataset,
        "lr": float(lr),
        "tuned_lr": optimizer_cfg.get("tuned_lr"),
        "seed": int(seed),
        "loss": "sft_only (token cross-entropy)",
        "method": method.cfg.to_dict(),
        "optimizer": {
            "type": "AdamW",
            "lr": float(optimizer_cfg["lr"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
            "betas": list(optimizer_cfg["betas"]),
            "eps": float(optimizer_cfg["eps"]),
            "effective_batch_size": int(args.batch_size) * int(args.grad_accum),
        },
        "trainer": trainer.cfg.to_dict(),
        "run_config_path": str(config_path),
        "init_checkpoint": str(abspath(DATASETS[dataset]["tiger_checkpoint"])),
        "finetune_subset": ({"per_user_random_n": int(args.train_random_n if args.train_random_n is not None
                                                      else DATASETS[dataset].get("train_random_n", 0)),
                             "seed": int(args.train_subset_seed if args.train_subset_seed is not None
                                         else DATASETS[dataset].get("train_subset_seed", 0)),
                             "num_examples": len(train_indices)}
                            if train_indices is not None else "full train set"),
        "num_train_examples": len(dm.train_dataset),
        "item_vocab_size": int(dm.item_vocab_size),
        "num_train_examples": len(dm.train_dataset),
        "init_baseline": baseline,
    }
    write_json(run_dir / "resolved_config.json", resolved)

    resumed_from_epoch = None
    last_ckpt = run_dir / "checkpoints" / "last.pt"
    if allow_resume and last_ckpt.exists():
        trainer.load_checkpoint(last_ckpt, load_optimizer=True, load_scheduler=True)
        resumed_from_epoch = int(trainer.state.epoch)
        print(
            f"[{tag}] resuming from {last_ckpt}: epoch {resumed_from_epoch + 1}, "
            f"best {trainer.state.best_metric} (epoch "
            f"{trainer.state.best_epoch + 1 if trainer.state.best_epoch is not None else '?'}), "
            f"bad_epochs {trainer.state.bad_epochs}/{trainer_cfg.early_stopping_patience}",
            flush=True,
        )

    print(
        f"[{tag}] fine-tuning: lr={lr:g} (tuned lr {optimizer_cfg.get('tuned_lr')}), seed={seed}, "
        f"epochs<={trainer_cfg.max_epochs}, monitor={MONITOR}, device={device}",
        flush=True,
    )
    started = time.time()
    fit = trainer.fit(seeded_train_loader(dm, seed, train_indices), dm.val_dataloader())
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    best_on_disk = run_dir / "checkpoints" / "best.pt"
    checkpoint_for_eval = (
        fit.best_checkpoint_path
        or (str(best_on_disk) if best_on_disk.exists() else None)
        or fit.last_checkpoint_path
    )
    if checkpoint_for_eval:
        trainer.load_checkpoint(checkpoint_for_eval, load_optimizer=False, load_scheduler=False)

    splits = evaluate_splits(model, dm, index, args, trainer_cfg.max_eval_batches)
    print(f"[{tag}] validation: {json.dumps(splits['validation'], sort_keys=True)}", flush=True)
    print(f"[{tag}] test: {json.dumps(splits['test'], sort_keys=True)}", flush=True)

    promoted: dict[str, str] | None = None
    if roots.promote is not None and fit.best_checkpoint_path:
        promote_dir = promote_group_dir(roots.promote, dataset, lr) / f"seed_{seed}"
        promote_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fit.best_checkpoint_path, promote_dir / "best.pt")
        sidecar = Path(str(fit.best_checkpoint_path) + ".json")
        if sidecar.exists():
            shutil.copy2(sidecar, promote_dir / "best.pt.json")
        shutil.copy2(config_path, promote_dir / "run_config.json")
        promoted = {"best_checkpoint": str(promote_dir / "best.pt"), "dir": str(promote_dir)}

    result = {
        "dataset": dataset,
        "variant": VARIANT_FT,
        "lr": float(lr),
        "seed": int(seed),
        "status": "completed",
        "best_epoch": fit.state.best_epoch,
        "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1),
        "steps_completed": int(fit.state.global_step),
        "early_stopped": bool(fit.state.should_stop),
        "resumed_from_epoch": resumed_from_epoch,
        "elapsed_sec": time.time() - started,
        "run_dir": str(run_dir),
        "evaluated_checkpoint": checkpoint_for_eval,
        "best_checkpoint": (promoted or {}).get("best_checkpoint", fit.best_checkpoint_path),
        "promoted_dir": (promoted or {}).get("dir"),
        "protocol": {k: v for k, v in protocol_dict(args).items() if k not in ("learning_rates", "seeds")},
        **splits,
    }
    write_json(run_dir / "metrics.json", result)
    if promoted is not None:
        write_json(Path(promoted["dir"]) / "metrics.json", result)

    logger.close()
    del model, method, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_completed_result(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    result = load_json(metrics_path)
    result["status"] = "completed (cached)"
    return result


def collect_results_from_disk(output_root: Path, dataset: str) -> dict[float, list[dict[str, Any]]]:
    """All completed (lr, seed) runs of a dataset: ``<output_root>/<ds>/sft_lr*/seed_*/metrics.json``.

    This is the single source of truth for every summary, so concurrent array
    tasks (one per seed) converge on the same complete picture."""
    groups: dict[float, list[dict[str, Any]]] = {}
    dataset_dir = output_root / dataset
    if not dataset_dir.exists():
        return groups
    for metrics_path in sorted(dataset_dir.glob("sft_lr*/seed_*/metrics.json")):
        try:
            result = load_json(metrics_path)
        except (OSError, ValueError):
            continue
        if not isinstance(result.get("test"), Mapping):
            continue
        result.setdefault("status", "completed")
        groups.setdefault(float(result["lr"]), []).append(result)
    for results in groups.values():
        results.sort(key=lambda r: int(r.get("seed", 0)))
    return dict(sorted(groups.items()))


def load_baseline_from_disk(output_root: Path, dataset: str) -> dict[str, Any] | None:
    path = output_root / dataset / "init_baseline.json"
    return load_json(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Aggregation + artifacts
# --------------------------------------------------------------------------- #
def aggregate_group(results: list[Mapping[str, Any]], ks: Iterable[int]) -> dict[str, Any]:
    """Mean / sample-std over seeds of the split metrics and training stats."""
    metric_keys = [f"{m}@{k}" for k in ks for m in ("HR", "NDCG", "Coverage", "ARP")]
    agg: dict[str, Any] = {"num_seeds": len(results), "seeds": [int(r["seed"]) for r in results]}
    for split in SPLIT_LOADERS:
        agg[split] = {}
        for key in metric_keys:
            mean, std = mean_std(r.get(split, {}).get(key) for r in results)
            agg[split][key] = {"mean": mean, "std": std}
    for key in ("best_val_ndcg5", "best_epoch", "epochs_completed"):
        mean, std = mean_std(r.get(key) for r in results)
        agg[key] = {"mean": mean, "std": std}
    return agg


def csv_rows_for(
    row_source: Mapping[str, Any], *, ks: Iterable[int], experiment_dir: str | None,
) -> list[dict[str, Any]]:
    """Two CSV rows (validation, test) in the artifacts/ metrics-summary schema."""
    rows: list[dict[str, Any]] = []
    for split in SPLIT_LOADERS:
        metrics = row_source.get(split)
        if not isinstance(metrics, Mapping):
            continue
        row: dict[str, Any] = {
            "dataset": row_source["dataset"],
            "variant": row_source.get("variant", VARIANT_FT),
            "lr": row_source.get("lr"),
            "seed": row_source.get("seed"),
            "split": split,
            "num_examples": metrics.get("num_examples"),
        }
        for k in ks:
            for m in ("HR", "NDCG", "Coverage", "ARP"):
                row[f"{m}@{k}"] = metrics.get(f"{m}@{k}")
        row["best_epoch"] = (
            row_source["best_epoch"] + 1 if isinstance(row_source.get("best_epoch"), int) else None
        )
        row["best_val_ndcg5"] = row_source.get("best_val_ndcg5")
        row["best_checkpoint"] = row_source.get("best_checkpoint", row_source.get("checkpoint"))
        row["experiment_dir"] = experiment_dir
        rows.append(row)
    return rows


CSV_KEY = ("dataset", "variant", "lr", "seed", "split")


def upsert_csv(path: Path, new_rows: list[dict[str, Any]]) -> None:
    """Merge rows into the CSV by (dataset, variant, lr, seed, split), so
    restarts, extra seeds or a new lr never drop earlier rows."""
    if not new_rows:
        return
    existing: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for row in new_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    def key_of(row: Mapping[str, Any]) -> tuple[str, ...]:
        out = []
        for k in CSV_KEY:
            v = row.get(k)
            if k == "lr" and v not in (None, ""):
                v = f"{float(v):g}"
            elif k == "seed" and v not in (None, ""):
                v = str(int(float(v)))
            out.append("" if v is None else str(v))
        return tuple(out)

    merged: dict[tuple[str, ...], dict[str, Any]] = {key_of(r): r for r in existing}
    for row in new_rows:
        merged[key_of(row)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged.values():
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def write_dataset_artifacts(
    *, dataset: str, artifacts_root: Path, baseline: Mapping[str, Any] | None,
    groups: Mapping[float, list[Mapping[str, Any]]], args: argparse.Namespace,
) -> tuple[Path, Path]:
    ks = [int(k) for k in args.test_ks]
    csv_path = artifacts_root / f"tiger_sft_finetune_{dataset}_metrics_summary.csv"
    json_path = artifacts_root / f"tiger_sft_finetune_{dataset}_summary.json"

    rows: list[dict[str, Any]] = []
    if baseline is not None:
        rows += csv_rows_for(baseline, ks=ks, experiment_dir=None)
    for results in groups.values():
        for r in results:
            rows += csv_rows_for(r, ks=ks, experiment_dir=r.get("run_dir"))
    upsert_csv(csv_path, rows)

    summary = {
        "dataset": dataset,
        "protocol": protocol_dict(args),
        "init_checkpoint": str(abspath(DATASETS[dataset]["tiger_checkpoint"])),
        "init_baseline": baseline,
        "runs": {
            lr_tag(lr): {
                "lr": float(lr),
                "aggregate": aggregate_group(results, ks),
                "seeds": list(results),
            }
            for lr, results in groups.items()
        },
        "metrics_csv": str(csv_path),
    }
    write_json(json_path, summary)
    return csv_path, json_path


def write_promotion_summary(
    *, dataset: str, lr: float, results: list[Mapping[str, Any]], baseline: Mapping[str, Any] | None,
    promote_root: Path, args: argparse.Namespace,
) -> None:
    group_dir = promote_group_dir(promote_root, dataset, lr)
    if not group_dir.exists():
        return
    write_json(
        group_dir / "training_summary.json",
        {
            "dataset": dataset,
            "variant": VARIANT_FT,
            "lr": float(lr),
            "init_checkpoint": str(abspath(DATASETS[dataset]["tiger_checkpoint"])),
            "protocol": protocol_dict(args),
            "aggregate": aggregate_group(results, [int(k) for k in args.test_ks]),
            "seeds": {
                f"seed_{r['seed']}": {
                    k: r.get(k) for k in (
                        "best_epoch", "best_val_ndcg5", "epochs_completed", "steps_completed",
                        "early_stopped", "elapsed_sec", "best_checkpoint", "run_dir",
                        "validation", "test",
                    )
                }
                for r in results
            },
            "init_baseline": baseline,
        },
    )


def refresh_dataset_summaries(
    dataset: str, *, roots: Roots, args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[float, list[dict[str, Any]]]]:
    """Rebuild artifacts CSV/JSON + promotion training_summary.json for a
    dataset from what is on disk (baseline + every completed run), under a lock."""
    baseline = load_baseline_from_disk(roots.output, dataset)
    groups = collect_results_from_disk(roots.output, dataset)
    with file_lock(roots.output / dataset / ".summaries.lock"):
        if roots.artifacts is not None:
            write_dataset_artifacts(
                dataset=dataset, artifacts_root=roots.artifacts, baseline=baseline,
                groups=groups, args=args,
            )
        if roots.promote is not None:
            for lr, results in groups.items():
                write_promotion_summary(
                    dataset=dataset, lr=lr, results=results, baseline=baseline,
                    promote_root=roots.promote, args=args,
                )
    return baseline, groups


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    roots = resolve_roots(args)
    if args.smoke:
        print(f"[smoke] all outputs redirected under {roots.output}", flush=True)
    if args.aggregate_only:
        aggregate_and_report(args, roots, failures=[])
        return

    tiger_runner.seed_everything(int(args.seeds[0]))
    device = torch.device(args.device)
    roots.output.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"\n=== dataset: {dataset} ===", flush=True)
        # One datamodule per dataset, shared by all (lr, seed) runs and the
        # baseline: the data config is identical across runs by construction.
        data_config = build_run_config(dataset, float(args.lr[0]), int(args.seeds[0]), args)
        data_config["experiment_name"] = f"sft_finetune_{dataset}"
        data_config_path = roots.output / dataset / "data_config.json"
        write_json(data_config_path, data_config)

        data_cfg = tiger_runner.build_data_config(dict(data_config), config_path=data_config_path)
        dm = tiger_runner.build_data_module(data_cfg)
        dm.setup()
        print(
            f"[{dataset}] catalog={dm.item_vocab_size} items, "
            f"train={len(dm.train_dataset)} examples, "
            f"val={len(dm.val_dataset)}, test={len(dm.test_dataset)}",
            flush=True,
        )

        # Probe model: sanity gate + catalog index (row -> item / popularity).
        probe = tiger_runner.build_model(dict(data_config), dm, config_path=data_config_path)
        load_tiger_checkpoint(probe, abspath(DATASETS[dataset]["tiger_checkpoint"]))
        index = CatalogIndex(probe, dm)
        if not args.skip_sanity_check:
            probe.to(device)
            nll = sanity_check_checkpoint(probe, dm, device, float(args.sanity_nll_threshold))
            print(f"[{dataset}] init checkpoint sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Baseline: the untouched init checkpoint under the exact same
        # protocol. Computed once; concurrent tasks wait on the lock and reuse.
        baseline: dict[str, Any] | None = None
        baseline_path = roots.output / dataset / "init_baseline.json"
        if not args.skip_init_eval:
            with file_lock(baseline_path.with_suffix(".lock")):
                cached = load_json(baseline_path) if baseline_path.exists() else None
                expected_ckpt = str(abspath(DATASETS[dataset]["tiger_checkpoint"]))
                if cached is not None and cached.get("checkpoint") != expected_ckpt:
                    print(
                        f"[{dataset}] cached init baseline is stale (evaluated "
                        f"{cached.get('checkpoint')}, init is now {expected_ckpt}); recomputing",
                        flush=True,
                    )
                    cached = None
                if cached is not None and not args.overwrite:
                    baseline = cached
                    print(
                        f"[{dataset}] init baseline (cached): val ndcg@5={baseline.get('val_ndcg5'):.5f} "
                        f"test NDCG@5={baseline.get('test', {}).get('NDCG@5'):.5f}",
                        flush=True,
                    )
                else:
                    started = time.time()
                    baseline = evaluate_init_baseline(
                        dataset=dataset, config=data_config, config_path=data_config_path,
                        dm=dm, index=index, device=device, args=args,
                    )
                    baseline["elapsed_sec"] = time.time() - started
                    write_json(baseline_path, baseline)
                    print(
                        f"[{dataset}] init baseline: val ndcg@5={baseline['val_ndcg5']:.5f} | "
                        f"validation {json.dumps(baseline['validation'], sort_keys=True)} | "
                        f"test {json.dumps(baseline['test'], sort_keys=True)}",
                        flush=True,
                    )

        for lr in args.lr:
            lr = float(lr)
            for seed in args.seeds:
                seed = int(seed)
                run_dir = run_dir_for(roots.output, dataset, lr, seed)
                tag = f"{dataset}/sft_lr{lr_tag(lr)}/seed_{seed}"
                cached = None if args.overwrite else load_completed_result(run_dir)
                if cached is not None:
                    print(f"[{tag}] already completed, reusing metrics.json (use --overwrite to re-run)",
                          flush=True)
                    continue
                try:
                    run_single(
                        dataset=dataset, lr=lr, seed=seed, dm=dm, index=index,
                        baseline=baseline, run_dir=run_dir, roots=roots, args=args,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the grid going
                    print(f"[{tag}] FAILED: {exc}", flush=True)
                    traceback.print_exc()
                    failures.append({
                        "dataset": dataset, "variant": VARIANT_FT, "lr": lr, "seed": seed,
                        "status": "failed", "error": str(exc), "run_dir": str(run_dir),
                    })
                # Restart-safe and concurrency-safe: summaries always reflect
                # every completed run on disk, not just this process.
                refresh_dataset_summaries(dataset, roots=roots, args=args)

    aggregate_and_report(args, roots, failures=failures)


def aggregate_and_report(args: argparse.Namespace, roots: Roots, *, failures: list[dict[str, Any]]) -> None:
    ks = [int(k) for k in args.test_ks]
    report: dict[str, Any] = {"output_root": str(roots.output), "datasets": {}}
    baselines: dict[str, Mapping[str, Any] | None] = {}
    groups_by_dataset: dict[str, dict[float, list[dict[str, Any]]]] = {}
    for dataset in args.datasets:
        baseline, groups = refresh_dataset_summaries(dataset, roots=roots, args=args)
        baselines[dataset] = baseline
        groups_by_dataset[dataset] = groups
        report["datasets"][dataset] = {
            "init_baseline": baseline,
            "runs": {
                lr_tag(lr): {"lr": lr, "aggregate": aggregate_group(results, ks), "seeds": results}
                for lr, results in groups.items()
            },
        }
        if roots.artifacts is not None:
            print(
                f"[{dataset}] artifacts: {roots.artifacts / f'tiger_sft_finetune_{dataset}_metrics_summary.csv'} , "
                f"{roots.artifacts / f'tiger_sft_finetune_{dataset}_summary.json'}",
                flush=True,
            )
    report["failed_runs"] = failures
    roots.output.mkdir(parents=True, exist_ok=True)
    write_json(roots.output / "summary.json", report)
    print_summary(groups_by_dataset, baselines, failures, ks=ks, output_root=roots.output)


def print_summary(
    groups_by_dataset: Mapping[str, Mapping[float, list[Mapping[str, Any]]]],
    baselines: Mapping[str, Mapping[str, Any] | None],
    failures: list[Mapping[str, Any]],
    *, ks: list[int], output_root: Path,
) -> None:
    fmt = lambda v: f"{v:.5f}" if isinstance(v, (int, float)) and not math.isnan(v) else "-"  # noqa: E731
    print("\n=== summary: SFT-only fine-tuning vs. init checkpoint "
          "(val ndcg@5 = beam-5 monitor; NDCG@5 = beam-20 constrained; all completed runs on disk) ===",
          flush=True)
    print(
        f"{'dataset':<12} {'lr':<9} {'seed':<9} {'status':<20} {'best_ep':>7} {'val ndcg@5':>11} "
        f"{'val NDCG@5':>11} {'test NDCG@5':>12} {'d test@5':>10}",
        flush=True,
    )
    for dataset, groups in groups_by_dataset.items():
        base = baselines.get(dataset)
        base_test5 = base.get("test", {}).get("NDCG@5") if base else None
        if base is not None:
            print(
                f"{dataset:<12} {'init':<9} {base.get('seed', '-')!s:<9} {'baseline':<20} {'-':>7} "
                f"{fmt(base.get('val_ndcg5')):>11} {fmt(base.get('validation', {}).get('NDCG@5')):>11} "
                f"{fmt(base_test5):>12} {'-':>10}",
                flush=True,
            )
        for lr, rows in groups.items():
            for r in rows:
                test5 = r.get("test", {}).get("NDCG@5")
                delta = (
                    f"{test5 - base_test5:+.5f}"
                    if isinstance(test5, (int, float)) and isinstance(base_test5, (int, float)) else "-"
                )
                best_ep = r.get("best_epoch")
                print(
                    f"{dataset:<12} {lr:<9g} {r.get('seed', '-')!s:<9} {str(r.get('status', 'completed')):<20}"
                    f" {(best_ep + 1) if isinstance(best_ep, int) else '-':>7} "
                    f"{fmt(r.get('best_val_ndcg5')):>11} "
                    f"{fmt(r.get('validation', {}).get('NDCG@5')):>11} {fmt(test5):>12} {delta:>10}",
                    flush=True,
                )
            if len(rows) > 1:
                agg = aggregate_group(rows, ks)
                t_mean, t_std = agg["test"]["NDCG@5"]["mean"], agg["test"]["NDCG@5"]["std"]
                v_mean = agg["validation"]["NDCG@5"]["mean"]
                m_mean = agg["best_val_ndcg5"]["mean"]
                delta = (
                    f"{t_mean - base_test5:+.5f}"
                    if isinstance(base_test5, (int, float)) and t_mean is not None else "-"
                )
                print(
                    f"{dataset:<12} {lr:<9g} {'mean+-std':<9} {f'{len(rows)} seeds':<20} {'-':>7} "
                    f"{fmt(m_mean):>11} {fmt(v_mean):>11} "
                    f"{(fmt(t_mean) + '+-' + fmt(t_std)) if t_mean is not None else '-':>12} {delta:>10}",
                    flush=True,
                )
        if not groups:
            print(f"{dataset:<12} (no completed runs on disk)", flush=True)
    for r in failures:
        print(f"{r['dataset']:<12} {r['lr']:<9g} {r.get('seed', '-')!s:<9} {'FAILED':<20}  error={r.get('error')}",
              flush=True)
    print(f"summary written to {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
