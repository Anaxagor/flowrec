#!/usr/bin/env python
"""Dense-layout SFT-only control for the GPT-2 full-history fine-tuning experiment.

A DIAGNOSTIC arm, not a new protocol. The GPT-2 base SFT checkpoints were
trained on the dense-causal layout (one row per 20-item user window, a label
at every position); both fine-tuning arms of the full-history experiment
(gflow_finetune_gpt2_amazon_fullhist.py: the GFlow grid and its SFT-only
control) use the prefix-target layout, because the GFlow term needs one target
per example. This script fine-tunes the SAME init checkpoint with L_SFT only on
the base run's dense layout. The prefix control now expands those same dense
windows, so both layouts use identical contexts and targets (except the
explicit full-history Yambda 50M diagnostic, while prefix fine-tuning uses
last-128 targets). This control measures the effect of multiple targets in
one forward pass and their batching during optimization.

Everything except the training layout comes from the full-history wrapper,
which reads configs/exp2-tuned-params/finetune_base_matched.json:

  init      the full-history experiment's init checkpoint, resolved through the
            same base.tuned_checkpoint_paths (checkpoints/exp2-tuned_params/
            initial/gpt2/<ds>/best_model.pt + best_trial.json,
            exp2 semantic-ID tables). When the prefix control's
            resolved_config.json is on disk, its init_checkpoint and target
            count must match or the run refuses to start.
  data      Beauty / Instruments / Yambda 50M: all eligible dense targets,
            as in the GPT-2 base SFT run. TIGER's last-128 policy does not
            apply by default to GPT-2 dense training. --train-last-n 128
            selects the additional Yambda 50M tail control, masking older
            labels while retaining original windows, catalog and held-out
            examples; default outputs gain a last128 suffix. Yambda 500M
            samples 15 targets from the last 128. Catalog and
            held-out splits are built before selecting any targets.
  schedule  lr, weight decay, clipping, AMP, validation after every epoch,
            patience 5 after the 10 warm-up epochs, init scored first, beam-5
            val/sft/ndcg@5 monitor, beam-20 final metrics, seeds 42 / 43 / 44,
            validation and final-eval batch sizes, --resume.
  batch     a dense row carries several targets (beauty ~5.6), so "32 x 64
            rows" would be ~5.6x more supervision per optimizer step than the
            prefix control. The micro-batch stays at the protocol's 32 rows and
            the accumulation is chosen so one optimizer step sees as close as
            possible to the prefix control's 2048 TARGETS (beauty: 32 x 11 rows
            ~ 1,960 targets, 68 steps per epoch vs 65). --targets-per-step /
            --dense-grad-accum change it. (The base run itself used 256 rows x 1
            ~ 1,426 targets: --targets-per-step 1426 gives exactly 32 x 8.)

Outputs (apart from the full-history experiment's):
  experiments/exp4-gflow-gpt2-amazon-fullhist-dense-control/<ds>/sft_only_dense/seed_<s>/
      resolved_config.json training_history.json final_metrics.json result.json checkpoints/
  artifacts/exp4-reward_ablation/gflow_finetune_gpt2_amazon_fullhist_dense_control_metrics_summary.csv
      same columns as the full-history CSV (variant = sft_only_dense, plus the
      init rows), so the two files concatenate.

Usage:
  python scripts/exp4-reward-ablations/sft_control_gpt2_amazon_dense.py --datasets beauty --seed 42 --device cuda
  python scripts/exp4-reward-ablations/sft_control_gpt2_amazon_dense.py --smoke --device cpu --num-workers 0
  submit-worker --array=0-5 scripts/scheduler/sft_control_gpt2_amazon_dense.submit-worker      # dataset x seed
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _path in (PROJECT_ROOT, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "exp1-standard-params",
              SCRIPT_DIR.parent / "exp2-tuned-params", SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gflow_finetune_gpt2_amazon as base  # noqa: E402
import gflow_finetune_gpt2_amazon_fullhist as fullhist  # noqa: E402
from flowrec.data.datasets import SequenceDataset, history_length_stats  # noqa: E402

ARM = "gpt2_sft_only_dense_layout"
PREFIX_ARM = fullhist.ARM
VARIANT = "sft_only_dense"
TRAIN_LAYOUT = "dense_causal"

_PROTOCOL = fullhist.load_protocol()
OUTPUT_ROOT = str(_PROTOCOL["arms"][ARM]["output_root"])
METRICS_CSV = str(_PROTOCOL["arms"][ARM]["metrics_csv"])
PREFIX_OUTPUT_ROOT = str(_PROTOCOL["arms"][PREFIX_ARM]["output_root"])
CONTROL_SEEDS = fullhist.CONTROL_SEEDS
TARGETS_PER_STEP = int(_PROTOCOL["shared_schedule"]["effective_batch"])


# --------------------------------------------------------------------------- #
# Arguments: the full-history wrapper's, plus the dense-batch knobs
# --------------------------------------------------------------------------- #
def dense_argv(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """(this script's own options, the argv handed to the base parser)."""
    # allow_abbrev=False: '--seed N' must reach the base parser, not match '--seeds'.
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument("--targets-per-step", type=int, default=TARGETS_PER_STEP,
                     help="Supervised targets one optimizer step should see (default: the "
                          "prefix control's effective batch). The accumulation over 32-row "
                          "micro-batches is chosen to come closest to it")
    own.add_argument("--dense-grad-accum", type=int, default=None,
                     help="Set the accumulation directly instead of deriving it")
    own.add_argument("--seeds", type=int, nargs="+", default=None,
                     help=f"Run several seeds in one process (default: --seed; the protocol's "
                          f"control seeds are {list(CONTROL_SEEDS)})")
    own.add_argument("--prefix-output-root", type=Path, default=Path(PREFIX_OUTPUT_ROOT),
                     help="The full-history experiment's output root: its init baseline is "
                          "imported and its control's init checkpoint / target count checked")
    ns, rest = own.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    # Defaults of the full-history experiment, then THIS arm's locations, then the user's argv.
    forwarded = [*fullhist.FULL_HISTORY_DEFAULTS, "--sft-only",
                 "--output-root", OUTPUT_ROOT, "--metrics-csv", METRICS_CSV, *rest]
    return ns, forwarded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    own, forwarded = dense_argv(raw_argv)
    if any(flag in forwarded for flag in ("-h", "--help")):
        print(__doc__)
        print("Options of this script: --targets-per-step N, --dense-grad-accum N, --seeds S [S ...], "
              "--prefix-output-root DIR; every other option is the base script's:\n")
    args = base.parse_args(forwarded)
    for key, value in vars(own).items():
        setattr(args, key, value)
    if "--prefix-from-dense" in raw_argv:
        raise ValueError("This control uses dense_causal batches; omit --prefix-from-dense")
    args.prefix_from_dense = False
    # Reuse the existing target selector, but give tail controls independent
    # checkpoints, completion/resume state and CSVs by default. Explicit paths
    # remain supported for custom experiments.
    supplied = {arg.split("=", 1)[0] for arg in raw_argv}
    if args.train_last_n > 0:
        tail = f"last{int(args.train_last_n)}"
        if "--output-root" not in supplied:
            args.output_root = Path(f"{args.output_root}-{tail}")
        if "--metrics-csv" not in supplied:
            csv = Path(args.metrics_csv)
            args.metrics_csv = csv.with_name(f"{csv.stem}_{tail}{csv.suffix}")
    return args


# --------------------------------------------------------------------------- #
# Dense data
# --------------------------------------------------------------------------- #
def dense_run_config(dataset: str, args: argparse.Namespace, user_ids_count: int) -> dict[str, Any]:
    """The full-history run config with ONLY the training layout changed (and
    the reward / teacher block dropped: this arm has neither)."""
    config = base.build_run_config(dataset, args, user_ids_count)
    config["experiment_name"] = f"sft_dense_gpt2_{dataset}"
    config["dataset"] = {**config["dataset"], "train_example_layout": TRAIN_LAYOUT}
    config["method"] = {"name": "sft"}
    return config


def count_dense_targets(train_dataset) -> int:
    """Number of active supervised labels over all dense rows."""
    frame = train_dataset.examples
    if "label_mask" in frame.columns:
        return int(sum(int(np.asarray(mask).astype(bool).sum()) for mask in frame["label_mask"]))
    return int(sum(int((np.asarray(labels) > 0).sum()) for labels in frame["label_item_ids"]))


def select_dense_targets(dm, args: argparse.Namespace) -> None:
    """Apply the prefix experiment's target selection to dense LABELS, not rows.

    Run after setup so the checkpoint's full catalog and held-out examples
    remain intact. Window inputs and positions are never cropped or shifted.
    A small scalar table reproduces the shared tail/random selection without
    materializing the corresponding histories as prefix examples.
    """
    pool = getattr(args, "train_pool_last_n", None)
    if pool is None and not args.train_last_n and not args.train_random_n:
        return
    frame = dm.train_dataset.examples
    last = frame.groupby("user_id", sort=False)["target_position"].max().to_dict()
    candidates = []
    for window in frame.itertuples():
        lower = last[window.user_id] - int(pool) + 1 if pool else 0
        if window.target_position < lower:
            continue
        first = window.target_position - len(window.label_mask) + 1
        for offset, active in enumerate(window.label_mask):
            position = first + offset
            if active and position >= lower:
                candidates.append((window.user_id, position, window.Index, offset))
    metadata = pd.DataFrame(candidates, columns=["user_id", "target_position", "row_index", "offset"])
    metadata = metadata.sort_values(["user_id", "target_position"], kind="stable").reset_index(drop=True)
    indices = base.finetune_indices(SequenceDataset(metadata, split="train"), args)
    chosen = metadata.iloc[indices]
    masks = {int(index): [False] * len(frame.at[index, "label_mask"])
             for index in chosen.row_index.unique()}
    for row in chosen.itertuples():
        masks[row.row_index][row.offset] = True
    selected = frame.loc[sorted(masks)].copy()
    selected["label_mask"] = [masks[index] for index in selected.index]
    dm.train_dataset = SequenceDataset(selected, split="train")
    dm.stats.num_train_examples = len(selected)
    for key, value in history_length_stats(selected, dm.val_dataset.examples, dm.test_dataset.examples).items():
        setattr(dm.stats, key, value)


def dense_batch_settings(*, num_rows: int, num_targets: int, micro_batch: int,
                         targets_per_step: int, grad_accum: int | None = None) -> dict[str, Any]:
    """Accumulation over ``micro_batch``-row micro-batches whose expected
    targets per optimizer step is closest to ``targets_per_step``."""
    if num_rows <= 0 or num_targets <= 0:
        raise ValueError("the dense training set is empty")
    targets_per_row = num_targets / num_rows
    if grad_accum is None:
        ideal = targets_per_step / (targets_per_row * micro_batch)
        candidates = {max(1, int(np.floor(ideal))), max(1, int(np.ceil(ideal)))}
        grad_accum = min(candidates, key=lambda a: (abs(a * micro_batch * targets_per_row - targets_per_step), a))
    grad_accum = max(1, int(grad_accum))
    rows_per_step = micro_batch * grad_accum
    return {
        "micro_batch_rows": int(micro_batch), "grad_accum": grad_accum, "rows_per_step": rows_per_step,
        "targets_per_row": targets_per_row,
        "expected_targets_per_step": rows_per_step * targets_per_row,
        "requested_targets_per_step": int(targets_per_step),
        "steps_per_epoch": int(np.ceil(num_rows / rows_per_step)),
    }


def dense_train_loader(dm, *, micro_batch: int, seed: int, shuffle: bool) -> DataLoader:
    """The datamodule's own dense train dataset and collator (the dataset
    yields whatever that collator expects), reshuffled with the RUN seed."""
    reference = dm.train_dataloader()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    workers = int(dm.cfg.num_workers)
    return DataLoader(
        reference.dataset, batch_size=int(micro_batch), shuffle=shuffle,
        num_workers=workers, pin_memory=dm.cfg.pin_memory,
        persistent_workers=bool(dm.cfg.persistent_workers) and workers > 0,
        drop_last=False, collate_fn=reference.collate_fn,
        generator=generator if shuffle else None,
    )


# --------------------------------------------------------------------------- #
# Consistency with the prefix control (when it is on disk)
# --------------------------------------------------------------------------- #
def prefix_control_record(dataset: str, args: argparse.Namespace) -> dict[str, Any] | None:
    root = base.abspath(args.prefix_output_root)
    for seed in (int(args.seed), *CONTROL_SEEDS):
        path = base.sft_control_run_dir(root, dataset, seed) / "resolved_config.json"
        if path.exists():
            try:
                return base.load_json(path)
            except Exception:  # noqa: BLE001 - unreadable: try the next seed
                continue
    return None


def same_repo_path(recorded: Any, ours: Path) -> bool:
    """Same file relative to the repository root (the prefix control may have
    run under another checkout prefix, e.g. on the cluster)."""
    ours = Path(ours)
    rel = ours.relative_to(PROJECT_ROOT).as_posix() if ours.is_relative_to(PROJECT_ROOT) else ours.as_posix()
    recorded = Path(str(recorded)).as_posix()
    return recorded == rel or recorded.endswith("/" + rel)


def check_against_prefix_control(dataset: str, *, num_targets: int, init_checkpoint: Path,
                                 args: argparse.Namespace) -> dict[str, Any]:
    """Same init checkpoint and same supervised targets as the prefix control,
    or refuse: otherwise the two controls would not differ in the layout only."""
    record = prefix_control_record(dataset, args)
    if record is None:
        print(f"[{dataset}] no prefix control under {args.prefix_output_root}: init checkpoint and "
              "target count are NOT cross-checked", flush=True)
        return {"checked": False}
    expected_targets = record.get("data", {}).get("num_finetune_examples")
    recorded_init = record.get("init_checkpoint")
    problems = []
    # The existing Yambda 50M prefix experiment selects last-128. Only the
    # dense tail control shares its target count; the full control has more.
    compare_targets = dataset != "yambda_50m_listens" or (
        args.train_last_n == 128 and not args.train_random_n)
    if compare_targets and expected_targets is not None and int(expected_targets) != int(num_targets):
        problems.append(f"supervised targets {num_targets} != the prefix control's {expected_targets}")
    if recorded_init and not same_repo_path(recorded_init, init_checkpoint):
        problems.append(f"init checkpoint {init_checkpoint} != the prefix control's {recorded_init}")
    if problems and not args.smoke:
        raise RuntimeError(f"[{dataset}] not the prefix control's setup: " + "; ".join(problems))
    print(f"[{dataset}] prefix control check: same init {init_checkpoint}; "
          f"{num_targets} dense supervised targets (target-count comparison: {compare_targets})", flush=True)
    return {"checked": True, "prefix_num_finetune_examples": expected_targets,
            "prefix_init_checkpoint": recorded_init, "target_count_checked": compare_targets}


def ensure_init_baseline(dataset: str, dm, *, device: torch.device, args: argparse.Namespace,
                         output_root: Path, csv_path: Path) -> None:
    """Import the full-history experiment's evaluation of the same init
    checkpoint (metrics do not depend on the training layout), else evaluate."""
    target = output_root / dataset / "baseline" / "init_metrics.json"
    source = base.abspath(args.prefix_output_root) / dataset / "baseline" / "init_metrics.json"
    if not target.exists() and source.exists() and not args.smoke and not args.skip_init_baseline:
        payload = base.load_json(source)
        ours = str(base.tuned_checkpoint_paths(dataset)["checkpoint"])
        if payload.get("checkpoint") == ours and all(k in payload for k in ("validation", "test")):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            identity = {**base.sft_control_identity(dataset, int(args.seed)), "variant": "init"}
            base.upsert_metrics_csv(csv_path, base.gflow_ablation.metric_rows(
                identity, {s: payload[s] for s in ("validation", "test")}, {"status": "completed"}))
            print(f"[{dataset}] init baseline imported from {source}", flush=True)
    base.ensure_init_baseline(dataset=dataset, dm=dm, device=device, args=args,
                              output_root=output_root, csv_path=csv_path)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_dir_for(output_root: Path, dataset: str, seed: int) -> Path:
    return output_root / dataset / VARIANT / f"seed_{seed}"


def run_dense_control(dataset: str, dm, run_dir: Path, *, batch: dict[str, Any], num_targets: int,
                      consistency: dict[str, Any], args: argparse.Namespace, csv_path: Path) -> dict[str, Any]:
    tag = f"{dataset}/{VARIANT}"
    device = torch.device(args.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base.runner.seed_everything(int(args.seed))
    model, profile = base.build_gpt2_from_tuned_checkpoint(dataset, dm)
    model.to(device)
    method = base.gflow_ablation.AliasedSFTMethod(base.light_validation_sft_cfg())
    trainer = base.build_trainer(model, method, tag=tag, run_dir=run_dir,
                                 run_accum=int(batch["grad_accum"]), args=args, device=device)
    identity = base.sft_control_identity(dataset, int(args.seed))
    num_rows = len(dm.train_dataset)
    last_ckpt = base.resumable_checkpoint(run_dir, identity, num_rows, args)
    resumed_from_epoch = base.resume_trainer(trainer, last_ckpt, tag=tag) if last_ckpt is not None else None
    base.write_json(run_dir / "resolved_config.json", {
        **identity, "variant": VARIANT, "model": model.config_dict(), "model_profile": profile,
        "init_checkpoint": str(base.tuned_checkpoint_paths(dataset)["checkpoint"]),
        "loss": "L_SFT only on the base run's dense-causal layout (layout diagnostic of the "
                "full-history SFT-only control)",
        "batching": batch,
        "data": {**base.runner.data_config_paths(dm.cfg), "train_example_layout": TRAIN_LAYOUT,
                 "finetune_subset": base.describe_finetune_subset(args), "num_train_rows": num_rows,
                 "finetune_per_user_pool_last_n": args.train_pool_last_n,
                 "finetune_per_user_last_n": base.cache_last_n(args),
                 "finetune_per_user_random_n": args.train_random_n,
                 "finetune_subset_seed": args.train_subset_seed,
                 "num_finetune_examples": num_rows, "num_supervised_targets": int(num_targets),
                 **dm.summary()},
        "prefix_control_check": consistency,
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(),
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })
    print(f"[{tag}] dense SFT-only fine-tuning: epochs<={trainer.cfg.max_epochs}, "
          f"{batch['micro_batch_rows']} rows x accum {batch['grad_accum']} = {batch['rows_per_step']} rows "
          f"~ {batch['expected_targets_per_step']:.0f} targets per step "
          f"({batch['steps_per_epoch']} steps per epoch)", flush=True)
    loader = dense_train_loader(dm, micro_batch=int(batch["micro_batch_rows"]), seed=int(args.seed),
                                shuffle=not args.smoke)
    fit, splits = base.fit_and_evaluate(trainer, model, dm, [], run_batch=int(batch["micro_batch_rows"]),
                                        run_dir=run_dir, args=args, device=device,
                                        resumed_from_epoch=resumed_from_epoch, train_loader=loader)
    result = {
        **identity, "variant": VARIANT, "status": "completed", "train_example_layout": TRAIN_LAYOUT,
        "resumed_from_epoch": resumed_from_epoch,
        "best_epoch": fit.state.best_epoch, "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1), "elapsed_sec": time.time() - started,
        "best_checkpoint": fit.best_checkpoint_path, "run_dir": str(run_dir), "splits": splits,
        "batching": batch,
    }
    base.write_json(run_dir / base.RESULT_FILENAME, result)
    base.upsert_metrics_csv(csv_path, base.gflow_ablation.metric_rows(
        {**identity, "variant": VARIANT}, splits,
        {"status": "completed", "best_epoch": result["best_epoch"],
         "best_val_ndcg5": result["best_val_ndcg5"], "epochs_completed": result["epochs_completed"],
         "run_dir": str(run_dir), "micro_batch": batch["micro_batch_rows"], "grad_accum": batch["grad_accum"],
         "num_examples_train_rows": num_rows, "targets_per_step": round(batch["expected_targets_per_step"], 1),
         "train_example_layout": TRAIN_LAYOUT, "elapsed_sec": round(result["elapsed_sec"], 1)}))
    del model, method, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def dense_dataset_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    args = base.dataset_args(args, dataset)
    if dataset == "yambda_50m_listens":
        # Dense exp2 GPT-2 tiles the entire eligible training history; last-128
        # is TIGER's prefix policy. Explicit --train-last-n/--train-random-n
        # still allow a separate subset experiment.
        args.train_pool_last_n = None
    if dataset.startswith("yambda_") and "batch_size" not in args.dataset_overrides:
        # Dense rows carry up to 50 labels; the prefix protocol's 256 rows
        # would exceed the requested targets/update before accumulation.
        args.batch_size = min(int(args.batch_size), 16)
    return args


def check_existing_target_selection(run_dir: Path, args: argparse.Namespace) -> None:
    """Do not skip or resume an old last-128 run as an all-target control."""
    config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        return
    previous = base.load_json(config_path).get("data", {})
    expected = {
        "finetune_per_user_pool_last_n": args.train_pool_last_n,
        "finetune_per_user_last_n": base.cache_last_n(args),
        "finetune_per_user_random_n": args.train_random_n,
    }
    for key, value in expected.items():
        if previous.get(key, 0 if key.endswith("random_n") else None) != value:
            raise RuntimeError(f"{run_dir}: saved target selection differs ({key}); "
                               "choose a separate --output-root for this data setting")


def run_dataset(dataset: str, seeds: Sequence[int], *, args: argparse.Namespace, device: torch.device,
                output_root: Path, csv_path: Path) -> list[dict[str, Any]]:
    args = dense_dataset_args(args, dataset)
    for seed in seeds:
        check_existing_target_selection(run_dir_for(output_root, dataset, int(seed)), args)
    pending = [s for s in seeds
               if args.force or not base.run_completed(run_dir_for(output_root, dataset, int(s)))]
    for seed in seeds:
        if seed not in pending:
            print(f"[{dataset}/{VARIANT}] seed {seed} already completed - skipping (--force to redo)", flush=True)
    if not pending:
        return []

    base.validate_inputs(dataset, args, need_teacher=False, components=[])
    model_yaml = base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML)
    profile, _ = base.tuned_profile(dataset)
    run_config = dense_run_config(dataset, args, base.gpt2_runner.user_ids_count(model_yaml, profile))
    config_path = output_root / dataset / "data_config.json"
    base.write_json(config_path, run_config)
    dm = base.tiger_runner.build_data_module(
        base.tiger_runner.build_data_config(dict(run_config), config_path=config_path))
    dm.setup()
    select_dense_targets(dm, args)
    if dm.semantic_ids is None or dm.sid_len is None:
        raise ValueError(f"[{dataset}] datamodule produced no semantic IDs.")
    num_rows, num_targets = len(dm.train_dataset), count_dense_targets(dm.train_dataset)
    print(f"[{dataset}] catalog={int(dm.item_vocab_size)} ids, dense train rows={num_rows} carrying "
          f"{num_targets} supervised targets ({num_targets / num_rows:.2f} per row)", flush=True)

    consistency = check_against_prefix_control(
        dataset, num_targets=num_targets,
        init_checkpoint=base.tuned_checkpoint_paths(dataset)["checkpoint"], args=args)
    probe, _ = base.build_gpt2_from_tuned_checkpoint(dataset, dm)
    probe.to(device)
    if not args.skip_sanity_check:
        nll = base.gpt2_beauty.sanity_check_init_nll(probe, dm, device,
                                                    threshold=float(args.sanity_nll_threshold))
        print(f"[{dataset}] init sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ensure_init_baseline(dataset, dm, device=device, args=args, output_root=output_root, csv_path=csv_path)

    batch = dense_batch_settings(num_rows=num_rows, num_targets=num_targets,
                                 micro_batch=int(args.batch_size),
                                 targets_per_step=int(args.targets_per_step),
                                 grad_accum=args.dense_grad_accum)
    results: list[dict[str, Any]] = []
    for seed in pending:
        args.seed = int(seed)
        run_dir = run_dir_for(output_root, dataset, int(seed))
        try:
            results.append(run_dense_control(dataset, dm, run_dir, batch=batch, num_targets=num_targets,
                                             consistency=consistency, args=args, csv_path=csv_path))
        except Exception as exc:  # noqa: BLE001 - record and continue with the next seed
            print(f"[{dataset}/{VARIANT}] seed {seed} FAILED: {exc}", flush=True)
            traceback.print_exc()
            identity = base.sft_control_identity(dataset, int(seed))
            result = {**identity, "variant": VARIANT, "status": "failed", "error": str(exc),
                      "run_dir": str(run_dir)}
            base.write_json(run_dir / base.RESULT_FILENAME, result)
            base.upsert_metrics_csv(csv_path, [
                {**identity, "variant": VARIANT, "split": split, "status": "failed",
                 "error": str(exc)[:300]} for split in ("validation", "test")])
            results.append(result)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    output_root = base.abspath(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = base.metrics_csv_path(args)
    seeds = [int(s) for s in (args.seeds or [args.seed])]
    print(f"dense-layout SFT-only control: datasets {list(args.datasets)}, seeds {seeds}, "
          f"init = the full-history experiment's checkpoint", flush=True)
    results: list[dict[str, Any]] = []
    for dataset in args.datasets:
        results.extend(run_dataset(dataset, seeds, args=args, device=device,
                                   output_root=output_root, csv_path=csv_path))
    for row in results:
        line = f"{row['dataset']:<12} {VARIANT} seed {row['seed']} {row['status']:<10}"
        if row.get("status") == "completed":
            line += (f" best_val_ndcg@5={row.get('best_val_ndcg5')} "
                     f"test_NDCG@10={row['splits']['test'].get('NDCG@10'):.5f}")
        print(line, flush=True)
    print(f"metrics CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
