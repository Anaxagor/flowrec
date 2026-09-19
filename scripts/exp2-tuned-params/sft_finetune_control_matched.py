#!/usr/bin/env python
"""Base-matched SFT-only fine-tuning control (TIGER, all five datasets:
beauty, instruments, beauty_2023, yambda_50m_listens, yambda_500m_listens).

Same machinery as sft_finetune_control.py / sft_finetune_control_yambda.py
(this script only fixes their defaults per dataset and calls their main), under
the protocol of configs/exp2-tuned-params/finetune_base_matched.json: the
SFT-only arm, every GFlow arm and the base SFT run see the SAME training
examples, and every arm of every model on a dataset uses the SAME batch sizes,
so the comparison isolates the fine-tuning loss and the wall-clock times are
comparable.

  data      the checkpoint's own data config: every prefix of the full user
            history on the Amazon datasets, the last-128-targets-per-user
            layout on yambda_50m. The GFlow arms are the ones that move onto
            this data (gflow_finetune_tiger_matched.py,
            gflow_finetune_gpt2_amazon_fullhist.py).
            yambda_500m (cost decision): 15 targets per user drawn uniformly
            at random, once (seed 42), from the last-128 layout - ~1.3M of
            ~9.9M examples, the SAME examples as the GFlow arm
            (--train-random-n; the subset seed is not the run seed). Its runs,
            checkpoints and metrics live under *-matched-r15 / *_matched_r15.
  batches   beauty / instruments: 32 x accum 64, validation and final-eval
            batch 64 - the setting of the GPT-2 arms on these datasets, which
            cannot score more than 512 candidate sequences per micro-batch
            (32 x (K+1) = 352 at the study-wide K=10). The first control runs
            used 256 x 8.
            beauty_2023 / yambda_50m / yambda_500m (TIGER-only datasets):
            256 x accum 8, validation and final-eval batch 256 (the TIGER
            GFlow arm runs the same at K=10). beauty_2023 is new here: the
            first sweep had no control for it.
            Effective batch 2048 everywhere.
  schedule  validation after EVERY epoch (it costs seconds); fine-tuning
            stops once 5 consecutive epochs bring no new best (the running
            best, seeded by the init checkpoint's score). The counter starts
            after the first 10 epochs - the GFlow arms' curriculum length,
            applied here too so both arms stop by the same rule (earliest
            stop: epoch 15). The first sweep validated every 5 epochs with
            patience 10 (= 50 epochs).
  models    the GPT-2 control on beauty / instruments
            (gflow_finetune_gpt2_amazon_fullhist.py --sft-only) reads the same
            protocol file: same data rule, batches, schedule and seeds.
  monitor   the init checkpoint is scored first (every finished run of the
            first sweep recorded run_validation_at_start false), and the
            monitor sample of the large validation splits (yambda_500m 25%,
            beauty_2023 10%) is drawn ONCE (seed 0) instead of
            re-drawn at every validation - the rule of the GFlow arm, so both
            arms validate on the same examples (with re-drawing all four
            first-sweep runs picked epoch 55, i.e. the draw chose the epoch).
  --resume  an interrupted run continues from checkpoints/last.pt.

Everything lives apart from the first sweep:

  runs       experiments/exp2-sft-finetune-matched/<ds>/sft_lr<lr>/seed_<N>/
  promoted   checkpoints/exp2-tuned_params/finetuned_matched/tiger/<ds>/sft_only_lr<lr>/seed_<N>/
  results    artifacts/exp2-tuned-params/finetune_matched/tiger_sft_finetune_<ds>_metrics_summary.csv
             artifacts/exp2-tuned-params/finetune_matched/tiger_sft_finetune_<ds>_summary.json

Batch sizes differ per dataset, so the datasets run one after another, each
through the base main with its own defaults. Every default is an ordinary flag
of the base script placed BEFORE the command line, so any of them can still be
overridden (argparse keeps the last occurrence):

  python scripts/exp2-tuned-params/sft_finetune_control_matched.py --device cuda
  python scripts/exp2-tuned-params/sft_finetune_control_matched.py \
      --datasets beauty --seeds 42 --smoke --num-workers 0 --device cpu   # wiring check
  python scripts/exp2-tuned-params/sft_finetune_control_matched.py --aggregate-only

Cluster: scripts/scheduler/sft_finetune_control_matched_array.submit-worker
(one (dataset, seed) pair per array task).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sft_finetune_control as base  # noqa: E402
import sft_finetune_control_yambda as yambda  # noqa: E402

PROTOCOL_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"

# Datasets the first sweeps did not cover (same spec shape as base.DATASETS).
EXTRA_DATASETS: dict[str, dict[str, Any]] = {
    "beauty_2023": {
        "sft_config": "configs/exp2-tuned-params/tiger_beauty_2023_sft_tuned.json",
        "tiger_checkpoint": "checkpoints/exp2-tuned_params/initial/tiger/beauty_2023/best.pt",
        "data_overrides": {
            "reviews_path": "data/raw/dataframe_format/reviews_beauty_2023.parquet",
            "metadata_path": "data/raw/dataframe_format/meta_beauty_2023.parquet",
        },
    },
}


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))


def matched_datasets(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The first sweep's dataset specs (same sft_config = same data, same init
    checkpoint) with the matched monitor rule as their per-dataset defaults."""
    specs = {**copy.deepcopy(base.DATASETS), **copy.deepcopy(EXTRA_DATASETS),
             **copy.deepcopy(yambda.YAMBDA_DATASETS)}
    arm = protocol["arms"]["tiger_sft_only"]
    missing = sorted(set(arm["datasets"]) - set(specs))
    if missing:
        raise KeyError(f"{PROTOCOL_CONFIG.name} lists datasets without a control spec: {missing}")
    matched: dict[str, dict[str, Any]] = {}
    for name in arm["datasets"]:
        block = protocol["datasets"][name]
        matched[name] = {
            **specs[name],
            "val_batch_size": int(block["val_batch_size"]),
            "val_sample_fraction": block["val_sample_fraction"],
            "val_sample_resample": bool(block["val_sample_resample"]),
            # the gflow arm's fine-tuning subset (0 = the whole train set)
            "train_random_n": int(block.get("train_random_n", 0)),
            "train_subset_seed": int(block.get("train_subset_seed", 0)),
        }
    return matched


def matched_defaults(dataset: str, protocol: dict[str, Any]) -> list[str]:
    """Base-script flags of one dataset (placed before the user's argv)."""
    block = protocol["datasets"][dataset]
    shared = protocol["shared_schedule"]
    arm = {**protocol["arms"]["tiger_sft_only"],
           **protocol["arms"]["tiger_sft_only"].get("dataset_roots", {}).get(dataset, {})}
    if int(block["batch_size"]) * int(block["grad_accum"]) != int(shared["effective_batch"]):
        raise ValueError(f"{dataset}: batch_size x grad_accum != effective_batch in {PROTOCOL_CONFIG.name}")
    if int(block["test_batch_size"]) != int(block["val_batch_size"]):
        # The control evaluates best.pt through the datamodule's eval loaders,
        # whose batch size is the validation batch.
        raise ValueError(f"{dataset}: the TIGER control needs test_batch_size == val_batch_size")
    return [
        "--batch-size", str(int(block["batch_size"])),
        "--grad-accum", str(int(block["grad_accum"])),
        "--val-batch-size", str(int(block["val_batch_size"])),
        "--validate-every", str(int(shared["validate_every"])),
        "--patience", str(int(shared["patience"])),
        "--min-epochs", str(int(shared["early_stopping_min_epochs"])),
        "--validate-at-start" if shared["run_validation_at_start"] else "--no-validate-at-start",
        "--resume",
        "--output-root", str(arm["output_root"]),
        "--artifacts-root", str(arm["artifacts_root"]),
        "--promote-root", str(arm["promote_root"]),
    ]


def split_datasets(argv: Sequence[str], choices: Sequence[str]) -> tuple[list[str], list[str]]:
    """(--datasets values, the remaining argv). Default: every matched dataset."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--datasets", nargs="+", choices=list(choices), default=list(choices))
    known, rest = pre.parse_known_args(list(argv))
    return list(known.datasets), list(rest)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    protocol = load_protocol()
    # The shared machinery resolves DATASETS / defaults at call time, so
    # swapping the module attributes retargets everything.
    base.DATASETS = matched_datasets(protocol)
    base.DEFAULT_OUTPUT_ROOT = Path(protocol["arms"]["tiger_sft_only"]["output_root"])
    if any(flag in argv for flag in ("-h", "--help")):
        base.main(argv)
        return
    datasets, rest = split_datasets(argv, list(base.DATASETS))
    for dataset in datasets:
        base.main([*matched_defaults(dataset, protocol), "--datasets", dataset, *rest])


if __name__ == "__main__":
    main()
