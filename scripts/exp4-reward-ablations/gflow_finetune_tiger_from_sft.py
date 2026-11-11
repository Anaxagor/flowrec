#!/usr/bin/env python
"""GFlow fine-tuning AFTER the existing TIGER SFT-only control.

Beauty / Instruments, 96 cells each: {sasrec, interaction, similarity,
sasrec_x_interaction_x_similarity} x powers {0.25, 1, 4, 16} x rho {1, 0.1}
x {tb, db, subtb}. The triple reward is a PRODUCT, raised to the power once.
rho is the target GFlow/SFT gradient-norm ratio, calibrated at the SFT-only
weights, not the literal loss coefficient.

Source: checkpoints/exp2/loo_split/sft-finetuned/tiger/<dataset>/best.pt when its
provenance matches the source protocol. --sft-seed aliases resolve to the
primary checkpoint's actual seed (42), recorded in output paths and results.
The GFlow --seed remains independent. Missing primary files fail explicitly.
There is no fallback to initial/.
The source control used 256 x 8, validation every 5 epochs, patience 10.
This NEW stage uses the current base-matched GFlow protocol: all prefixes,
32 x 64, AdamW at constant LR 3e-5, K=10 teacher curriculum over 10 epochs,
validation every epoch, patience 5 counted after epoch 10, <=500 epochs.
Initial validation seeds best.pt; final full validation/test use beam 20.

Outputs, baseline and calibration are separate from GFlow-from-initial and
from each other source seed. Reward factors are shared across source seeds
(same fixed teacher, tokenizer and examples), under a new experiment root.
--smoke isolates both outputs and factors from the full experiment.
--resume is enabled by default; completed cells are skipped.

Examples:
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_from_sft.py --dataset beauty --list
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_from_sft.py --dataset beauty --prepare
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_from_sft.py --dataset beauty --index 0 --require-prepared
  python scripts/exp4-reward-ablations/gflow_finetune_tiger_from_sft.py --dataset instruments --sft-seed 43 --seed 43 --index 0

--prepare builds the three factor caches and evaluates the SFT-only baseline
before training tasks are allocated GPUs. It performs no fine-tuning.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gflow_finetune_tiger_matched as matched  # noqa: E402

base = matched.base
PROJECT_ROOT = base.PROJECT_ROOT
SOURCE_PROTOCOL_DIR = PROJECT_ROOT / "scripts/exp7-from-sft"
if str(SOURCE_PROTOCOL_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_PROTOCOL_DIR))
import sft_source_protocol as sft_sources  # noqa: E402

DATASETS = ("beauty", "instruments")
COMBOS = ("sasrec", "interaction", "similarity", "sasrec_x_interaction_x_similarity")
SOURCE_ROOT = Path("checkpoints/exp2/loo_split/sft-finetuned/tiger")
SOURCE_VARIANT = "sft_only_lr3e-05"
# The established Amazon GFlow positive/auxiliary split stores these fields
# in the interaction store. TIGER/SFT consume item/token sequences, not these
# rating/verified-purchase annotations. Every other example column must match.
AUXILIARY_COLUMNS = ("target_rating", "target_verified_purchase",
                     "history_ratings", "history_verified_purchase")


def source_checkpoint(dataset: str, sft_seed: int) -> Path:
    return sft_sources.source_paths(f"tiger_{dataset}", sft_seed, PROJECT_ROOT)["checkpoint"]


def wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--sft-seed", type=int, default=42)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--require-prepared", action="store_true")
    options, remaining = p.parse_known_args(list(argv))
    options.sft_seed = sft_sources.primary_source_seed(f"tiger_{options.dataset}", options.sft_seed, PROJECT_ROOT)
    return options, ["--dataset", options.dataset, *remaining]


def experiment_argv(options: argparse.Namespace, remaining: Sequence[str]) -> list[str]:
    dataset = options.dataset
    tag = f"{SOURCE_VARIANT}/sft_seed_{options.sft_seed}"
    output = f"experiments/exp4-gflow-tiger-from-sft/{dataset}/{tag}"
    cache = f"artifacts/exp4-reward_ablation/gflow_tiger_from_sft/{dataset}/reward_factors"
    csv = f"gflow_tiger_from_sft_{dataset}_{SOURCE_VARIANT}_sft_seed_{options.sft_seed}"
    argv = [
        *matched.matched_defaults(dataset, matched.load_protocol()),
        "--combos", *COMBOS,
        "--powers", "0.25", "1", "4", "16",
        "--objectives", "tb", "db", "subtb",
        "--rho-targets", "1", "0.1",
        "--gflow-weight-mode", "fixed_grad_ratio",
        "--output-root", output,
        "--cache-root", cache,
        "--metrics-csv-name", csv + ".csv",
        # Path('') is a directory: the base importer ignores it. Never import
        # the SFT control's init_baseline.json, which measures initial/ weights.
        "--control-init-baseline", "",
        *remaining,
    ]
    if "--smoke" in remaining:
        # Apply isolation AFTER explicit path overrides too: debugging must
        # never reuse a full-run baseline or replace its factor caches.
        paths = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        paths.add_argument("--output-root", type=Path)
        paths.add_argument("--cache-root", type=Path)
        paths.add_argument("--metrics-csv-name", type=Path)
        selected, _ = paths.parse_known_args(argv)
        argv += ["--output-root", str(selected.output_root / "smoke"),
                 "--cache-root", str(selected.cache_root / "smoke"),
                 "--metrics-csv-name", str(selected.metrics_csv_name.with_stem(
                     selected.metrics_csv_name.stem + "_smoke"))]
    return argv


@contextmanager
def sft_source(dataset: str, sft_seed: int) -> Iterator[None]:
    """Retarget every base-script checkpoint load, including repeated parsing.

    Restore the module state so importing this wrapper does not alter the
    original GFlow-from-initial scripts or other experiments in one process.
    """
    old_spec = base.DATASETS[dataset]
    active = {key: getattr(base, key) for key in old_spec}
    active["DATASET"] = base.DATASET
    base.DATASETS[dataset] = {**old_spec, "TIGER_CHECKPOINT": str(source_checkpoint(dataset, sft_seed))}
    base.activate_dataset(dataset)
    try:
        yield
    finally:
        base.DATASETS[dataset] = old_spec
        for key, value in active.items():
            setattr(base, key, value)


def validate_source(options: argparse.Namespace) -> dict[str, Any]:
    source_id = f"tiger_{options.dataset}"
    source_seed = sft_sources.primary_source_seed(source_id, options.sft_seed, PROJECT_ROOT)
    paths = sft_sources.source_paths(source_id, source_seed, PROJECT_ROOT)
    checkpoint, config_path = paths["checkpoint"], paths["config"]
    for path in (checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"SFT-only source is missing: {path}; no fallback to initial weights")
    if checkpoint.is_relative_to(PROJECT_ROOT / SOURCE_ROOT):
        # The flat copy must match the exact seed/protocol and original weight
        # hash before it can replace that source in a new training run.
        sft_sources.resolve_source(source_id, source_seed, PROJECT_ROOT)
    source = base.load_json(config_path)
    canonical = base.load_json(base.abspath(base.SFT_CONFIG))
    if source.get("method", {}).get("name") != "sft":
        raise ValueError(f"{config_path} is not an SFT-only run")
    if source.get("seed") != source_seed:
        raise ValueError(f"{config_path}: source seed does not match --sft-seed")
    # Canonical configs avoid source-host absolute paths while preserving the
    # exact model/catalog/prefix construction captured by the SFT-only control.
    for section in ("model", "dataset"):
        if source.get(section) != canonical.get(section):
            raise ValueError(f"{config_path}: {section} differs from {base.SFT_CONFIG}; "
                             "cannot reuse this experiment's data/reward protocol")
    source_sid = str(source.get("data", {}).get("semantic_id_path", ""))
    canonical_sid = str(canonical.get("data", {}).get("semantic_id_path", ""))
    if not source_sid or not base.checkpoint_matches(source_sid, canonical_sid):
        raise ValueError(f"{config_path}: semantic ID table differs from {base.SFT_CONFIG}")
    for key, expected in base.DATA_OVERRIDES.items():
        if not base.checkpoint_matches(source.get("data", {}).get(key), expected):
            raise ValueError(f"{config_path}: {key} differs from the Amazon GFlow data source")
    return {"checkpoint": str(checkpoint), "run_config": str(config_path),
            "sft_seed": source_seed, "source_variant": SOURCE_VARIANT,
            "source_optimizer": source.get("optimizer"), "source_scheduler": source.get("scheduler"),
            "source_training": source.get("training")}


def check_full_data_protocol(args: argparse.Namespace) -> None:
    if args.train_last_n != 0 or args.train_random_n != 0:
        raise ValueError("This comparison requires all base TIGER train prefixes; "
                         "--train-last-n and --train-random-n must both be 0")
    if args.val_sample_fraction not in (None, 1.0):
        raise ValueError("This comparison requires the full base TIGER validation split")
    if not args.smoke and (args.max_train_batches is not None or args.max_eval_batches is not None):
        raise ValueError("Partial train/evaluation is only allowed with --smoke")


def verify_base_examples(config: dict[str, Any], config_path: Path, dm) -> dict[str, Any]:
    """Check every example against the original TIGER SFT preprocessing.

    Amazon GFlow expresses verified purchases as positive_criterion so the
    interaction reward can retain auxiliary events. That must not change
    the supervised examples, item/user IDs, semantic IDs or held-out splits.
    """
    import pandas as pd

    reference = copy.deepcopy(config)
    reference["dataset"] = base.load_json(base.abspath(base.SFT_CONFIG))["dataset"]
    original = base.tiger_runner.build_data_module(
        base.tiger_runner.build_data_config(reference, config_path=config_path))
    original.setup()
    for name in ("item2id", "user2id", "semantic_ids"):
        if getattr(dm, name) != getattr(original, name):
            raise ValueError(f"GFlow preprocessing changed {name} relative to base TIGER")
    counts = {}
    for split, attribute in (("train", "train_dataset"), ("validation", "val_dataset"),
                             ("test", "test_dataset")):
        actual = getattr(dm, attribute).examples
        expected = getattr(original, attribute).examples
        try:
            pd.testing.assert_frame_equal(actual.drop(columns=list(AUXILIARY_COLUMNS), errors="ignore"),
                                          expected.drop(columns=list(AUXILIARY_COLUMNS), errors="ignore"))
        except AssertionError as exc:
            raise ValueError(f"GFlow {split} examples differ from base TIGER: {exc}") from exc
        counts[split] = len(actual)
    report = {"base_sft_config": str(base.abspath(base.SFT_CONFIG)),
              "gflow_reference": "scripts/exp4-reward-ablations/gflow_finetune_tiger_matched.py",
              "data": config["data"], "dataset": config["dataset"],
              "items": len(dm.item2id), "users": len(dm.user2id), "examples": counts,
              "catalog_and_supervised_examples_equal": True,
              "auxiliary_columns_not_used_by_tiger": list(AUXILIARY_COLUMNS)}
    print(f"[data] exact match to base TIGER: {report['items']} items; {counts}", flush=True)
    return report


def prepare_baseline(args: argparse.Namespace) -> None:
    """Evaluate the actual SFT-only weights, once, before the training array."""
    import torch

    base.tiger_runner.seed_everything(int(args.seed))
    output = base.abspath(args.output_root)
    config = base.build_run_config(args)
    config_path = output / "data_reward_config.json"
    dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=config_path))
    dm.setup()
    base.write_json(output / "data_protocol.json", verify_base_examples(config, config_path, dm))
    index = base.control.CatalogIndex(base.tiger_runner.build_model(config, dm, config_path=config_path), dm)
    base.ensure_init_baseline(config=config, config_path=config_path, dm=dm, index=index,
                             device=torch.device(args.device), args=args, output_root=output,
                             csv_path=base.metrics_csv_path(args))


def require_prepared(args: argparse.Namespace) -> None:
    for name in base.components_for([tuple(e["combo"]) for e in base.select_grid(args)]):
        directory = base.component_cache_dir(name, args)
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {manifest}; run --prepare first")
        base.check_component_manifest(base.load_json(manifest), directory, name, None, args)
    baseline = base.abspath(args.output_root) / "baseline/init_metrics.json"
    if not baseline.is_file():
        raise FileNotFoundError(f"Missing {baseline}; run --prepare first")
    payload = base.load_json(baseline)
    if not base.checkpoint_matches(payload.get("checkpoint")) or not all(
        split in payload for split in ("validation", "test")
    ):
        raise ValueError(f"{baseline} is not a complete baseline for {base.TIGER_CHECKPOINT}")
    audit_path = base.abspath(args.output_root) / "data_protocol.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing {audit_path}; run --prepare to verify base TIGER data equivalence")
    audit = base.load_json(audit_path)
    config = base.build_run_config(args)
    if (not audit.get("catalog_and_supervised_examples_equal")
            or audit.get("data") != config["data"] or audit.get("dataset") != config["dataset"]):
        raise ValueError(f"{audit_path}: data protocol changed; run --prepare again")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    options, remaining = wrapper_args(sys.argv[1:] if argv is None else argv)
    with sft_source(options.dataset, options.sft_seed):
        args = base.parse_args(experiment_argv(options, remaining))
    args.sft_seed = options.sft_seed
    return args


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("Additional source/launch flags: --sft-seed INT, --prepare, --require-prepared.\n")
        base.parse_args(["--dataset", "beauty", "--help"])
        return
    options, remaining = wrapper_args(argv)
    with sft_source(options.dataset, options.sft_seed):
        full_argv = experiment_argv(options, remaining)
        args = base.parse_args(full_argv)
        matched.check_uniform_micro_batch(args)
        check_full_data_protocol(args)
        if args.list:
            print(f"SFT-only source: {base.TIGER_CHECKPOINT}")
            base.main(full_argv)
            return
        if options.prepare and (args.index is not None or args.cache_only or args.skip_init_baseline):
            raise ValueError("--prepare builds all selected caches and the baseline; omit "
                             "--index, --cache-only and --skip-init-baseline")
        if options.require_prepared and args.force_cache:
            raise ValueError("Rebuild caches with --prepare --force-cache before the training array")
        source = validate_source(options)
        source["gflow_seed"] = args.seed
        output = base.abspath(args.output_root)
        output.mkdir(parents=True, exist_ok=True)
        # Each task has its own provenance file; no shared file is rewritten
        # by simultaneous array cells or different GFlow seeds.
        provenance = output / "sources" / f"seed_{args.seed}_index_{args.index or 'all'}.json"
        base.write_json(provenance, source)
        if options.prepare:
            base.main([*full_argv, "--cache-only"])
            prepare_baseline(args)
            require_prepared(args)
            return
        if options.require_prepared:
            require_prepared(args)
        # Record exceptions even if --force left an older completed result
        # behind: the legacy main catches these and otherwise exits zero.
        failed_calls: list[str] = []
        original_run_single = base.run_single
        original_write_json = base.write_json
        original_finetune_indices = base.finetune_indices

        def checked_run_single(**kwargs):
            try:
                return original_run_single(**kwargs)
            except Exception:
                failed_calls.append(base.entry_tag(kwargs["entry"]))
                raise

        def task_write_json(path, payload):
            if path == output / "summary.json":
                path = output / "summaries" / f"seed_{args.seed}_index_{args.index or 'all'}.json"
            original_write_json(path, payload)

        def training_indices(dataset, parsed_args):
            indices = original_finetune_indices(dataset, parsed_args)
            # Prefetch can request a batch before the trainer checks its step
            # limit. Keep a smoke loader inside its deliberately tiny cache.
            return indices[:base._smoke_examples(parsed_args)] if parsed_args.smoke else indices

        base.run_single = checked_run_single
        base.write_json = task_write_json
        base.finetune_indices = training_indices
        try:
            base.main(full_argv)
        finally:
            base.run_single = original_run_single
            base.write_json = original_write_json
            base.finetune_indices = original_finetune_indices
        if not args.cache_only:
            # The legacy runner logs exceptions and continues. Make failures
            # visible to the caller, including a missing/incomplete result file.
            failed = list(failed_calls)
            for entry in base.select_grid(args):
                path = base.run_dir_for(output, entry, args.seed) / "result.json"
                if not path.is_file() or base.load_json(path).get("status") != "completed":
                    failed.append(base.entry_tag(entry))
            if failed:
                raise RuntimeError(f"GFlow cells did not complete: {', '.join(dict.fromkeys(failed))}")


if __name__ == "__main__":
    main()
