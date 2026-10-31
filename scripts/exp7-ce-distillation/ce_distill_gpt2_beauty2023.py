#!/usr/bin/env python3
"""Reward CE for the same Amazon Beauty 2023 GPT-2 checkpoint as GFlow.

The default grid has 24 cells: each reward alone and their product, powers
0.25/1/16, rho 1/0.1, seed 42. Three draws per prefix use a smooth low-to-high
reward-half proposal with raw q/proposal importance correction. Training
prefixes preserve the original dense-causal windows, catalog and held-out rows.
--count / --list / --aggregate-only need only the Python standard library.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import ce_distill_gpt2 as gpt2

shared = gpt2.shared
ROOT = shared.ROOT
PROTOCOL = shared.PROTOCOL
DATASETS = ("beauty_2023",)
COMPONENTS = shared.COMPONENTS
DEFAULT_COMBO = shared.DEFAULT_COMBO
COMBOS = (*COMPONENTS, DEFAULT_COMBO)
DEFAULT_OUTPUT_ROOT = "experiments/exp7-ce-distillation-gpt2-beauty2023"
DEFAULT_CACHE_ROOT = "artifacts/exp7-ce-distillation-gpt2-beauty2023"
SOURCE_PROFILE = "exp2_gpt2_beauty_2023_dense_causal"

absolute = shared.absolute
read_json = shared.read_json
write_json = shared.write_json
build_grid = shared.build_grid
run_dir_for = shared.run_dir_for
combo_key = shared.combo_key
components_for = shared.components_for
curriculum_steps_for = shared.curriculum_steps_for
aggregate = shared.aggregate
load_runtime = gpt2.load_runtime
build_run_config = gpt2.build_run_config
component_cache_dir = gpt2.component_cache_dir
check_component_manifest = gpt2.check_component_manifest
verify_caches = gpt2.verify_caches
reward_config = gpt2.reward_config
asset_paths_for = gpt2.asset_paths_for
make_sft = gpt2.make_sft
evaluate_splits = gpt2.evaluate_splits


def parse_args(argv=None):
    args = shared.parse_args(
        argv, description=__doc__, dataset_choices=DATASETS,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        cache_root_help="Beauty 2023 GPT-2 factors: <root>/beauty_2023/reward_factors/<component>",
        parser_defaults={"combos": list(COMBOS), "powers": [.25, 1., 16.],
                         "rho_targets": [1., .1], "num_samples": 3,
                         "sampling": "curriculum_smooth_is", "curriculum_epochs": 5.,
                         "checkpoint_every_updates": 100},
    )
    if args.estimator != "sampled" or args.sampling != "curriculum_smooth_is":
        raise SystemExit("Beauty 2023 CE requires --estimator sampled --sampling curriculum_smooth_is")
    if any(combo not in COMBOS for combo in args.combos):
        raise SystemExit("Beauty 2023 CE supports single rewards and sasrec_x_interaction_x_similarity only")
    return args


def matched_args(args, dataset, seed, matched):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Beauty 2023 CE dataset: {dataset!r}")
    argv = ["--datasets", dataset, "--seed", str(seed), "--device", args.device,
            "--num-workers", str(args.num_workers), "--cache-format", "factorized",
            "--cache-batch-size", str(args.cache_batch_size),
            "--cache-verify-batches", str(args.cache_verify_batches),
            "--num-candidates", str(args.num_samples), "--prefix-from-dense",
            "--combos", DEFAULT_COMBO, "--similarity-modes", "hierarchical"]
    for key in ("batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.smoke:
        argv.append("--smoke")
    b = matched.base.dataset_args(matched.parse_args(argv), dataset)
    if not b.prefix_from_dense or b.train_last_n or b.train_random_n:
        raise RuntimeError("Beauty 2023 CE must use every dense-derived prefix, without a target subset")
    b.output_root = args.output_root
    b.cache_root = (args.output_root / "cache" if args.smoke else
                    absolute(args.cache_root or DEFAULT_CACHE_ROOT))
    return b


def trainer_overrides(args, b):
    # GPT-2's Beauty 2023 override uses full validation, unlike TIGER's 10%.
    return {"amp_dtype": b.amp_dtype, "val_sample_fraction": b.val_sample_fraction,
            "val_sample_resample": b.val_sample_resample}


def build_identity(args, entry, config, trainer_config, assets, curriculum_steps):
    identity = gpt2.build_identity(args, entry, config, trainer_config, assets, curriculum_steps)
    identity["source_profile"] = SOURCE_PROFILE
    return identity


def result_metadata(args, entry, b, config):
    return {"source_profile": SOURCE_PROFILE, "target_selection": "all_dense_window_labels"}


def run_entry(args, entry, matched, base):
    # Pass only the changed hooks; the tested GPT-2 CE loop retains ownership
    # of importance sampling, calibration, resume, and final full evaluation.
    profile = SimpleNamespace(matched_args=matched_args, trainer_overrides=trainer_overrides,
                              build_identity=build_identity, result_metadata=result_metadata)
    return gpt2.run_entry(args, entry, matched, base, profile=profile)


def build_caches(args, matched, base):
    for dataset in args.datasets:
        b = matched_args(args, dataset, 42, matched)
        needed = ([args.cache_component] if args.cache_component else
                  [name for name in COMPONENTS if any(name in c.split("_x_") for c in args.combos)])
        base.validate_inputs(dataset, b, need_teacher="sasrec" in needed,
                             components=[gpt2.INTERNAL_COMPONENTS[name] for name in needed])
        config = build_run_config(base, dataset, b)
        # Independent CPU/GPU array tasks never share a mutable config path.
        destination = args.output_root / dataset / "cache_build"
        preparation = destination / "_and_".join(needed)
        config_path = preparation / "data_reward_config.json"
        write_json(config_path, config)
        started = time.perf_counter()
        base.runner.seed_everything(42)
        dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=config_path))
        dm.setup()
        preparation_sec = time.perf_counter() - started
        write_json(preparation / "dense_prefix_audit.json", dm.dense_prefix_audit)
        count = len(base.finetune_indices(dm.train_dataset, b))
        if args.smoke:
            count = min(count, base._smoke_examples(b))
        for name in needed:
            cache_dir = component_cache_dir(base, dataset, name, b)
            path = cache_dir / "manifest.json"
            existed = path.exists()
            if existed:
                check_component_manifest(base, config, path, name, count, b)
            started = time.perf_counter()
            base.ensure_component_cache(dataset, gpt2.INTERNAL_COMPONENTS[name], config,
                                        destination / name, count, b, prepared_data=dm)
            manifest = check_component_manifest(base, config, path, name, count, b)
            write_json(destination / name / "timing.json", {
                "model": "gpt2", "dataset": dataset, "source_profile": SOURCE_PROFILE,
                "component": name, "reused": existed, "elapsed_sec": time.perf_counter() - started,
                "data_setup_sec": preparation_sec, "num_examples": manifest["num_examples"],
                "cache_path": str(cache_dir), "device": args.device,
            })


def main(argv=None):
    args = parse_args(argv)
    grid = build_grid(args)
    if args.count:
        print(len(grid))
        return
    if args.list:
        for i, entry in enumerate(grid):
            print(f"{i:4d} {entry['dataset']} combo={combo_key(entry)} power={entry['power']:g} "
                  f"rho={entry['rho']:g} seed={entry['seed']} sampling={args.sampling} n={args.num_samples}")
        print(f"total: {len(grid)} runs")
        return
    if args.aggregate_only:
        aggregate(args)
        return
    if args.index is not None:
        if not 0 <= args.index < len(grid):
            raise SystemExit(f"--index must lie in 0..{len(grid) - 1}")
        grid = [grid[args.index]]
    matched, base = load_runtime()
    if args.cache_only:
        build_caches(args, matched, base)
        return
    for entry in grid:
        with base.control.file_lock(run_dir_for(args, entry) / "run.lock"):
            run_entry(args, entry, matched, base)


if __name__ == "__main__":
    main()
