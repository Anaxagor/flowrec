#!/usr/bin/env python
"""Same-reward SFT + CE distillation (§5.4), starting from exp2 TIGER.

Grid defaults and reward semantics come from the base-matched exp4 protocol.
--list / --count / --aggregate-only need only the Python standard library.
See README.md in this directory for the objective and cluster commands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"
COMPONENTS = ("sasrec", "interaction", "similarity")
COMBOS = tuple("_x_".join(combo) for size in range(1, len(COMPONENTS) + 1)
               for combo in itertools.combinations(COMPONENTS, size))
DEFAULT_COMBO = "_x_".join(COMPONENTS)
SAMPLING_ALGORITHMS = {
    "reward": "reward",
    "curriculum": "reward_halves",
    "curriculum_smooth": "reward_halves_linear_mixture",
    "curriculum_smooth_is": "reward_halves_linear_mixture_with_q_importance_sampling",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Publish complete records atomically, including under Slurm arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def tag(value):
    value = float(value)
    return (str(int(value)) if value.is_integer() else repr(value)).replace(".", "p")


def parse_args(argv=None, *, description=None,
               default_output_root="experiments/exp7-ce-distillation", cache_root_help=None,
               dataset_choices=("beauty", "instruments"), default_datasets=None, parser_defaults=None,
               configure_parser=None):
    protocol = read_json(PROTOCOL)
    shared, grid = protocol["shared_schedule"], protocol["gflow_grid"]
    p = argparse.ArgumentParser(description=description or __doc__, allow_abbrev=False)
    p.add_argument("--datasets", nargs="+", choices=list(dataset_choices),
                   default=list(default_datasets or dataset_choices))
    p.add_argument("--combos", nargs="+", choices=COMBOS, default=[DEFAULT_COMBO],
                   help="Reward components multiplied before the common power")
    p.add_argument("--powers", nargs="+", type=float, default=grid["powers"])
    p.add_argument("--rho-targets", nargs="+", type=float, default=grid["rho_targets"])
    p.add_argument("--seeds", nargs="+", type=int, default=[shared["seed_gflow"]])
    p.add_argument("--estimator", choices=["sampled", "exact"], default="sampled")
    p.add_argument("--sampling", choices=list(SAMPLING_ALGORITHMS), default="reward",
                   help="Full q, hard low-to-high half switch, or a linear mixture transition between reward halves")
    p.add_argument("--curriculum-epochs", type=float, default=shared["gflow_curriculum_epochs"],
                   help="Hard curriculum: low-half duration; smooth curriculum: transition duration from all low to all high; 0 starts high")
    p.add_argument("--proposal-q-mix", type=float, default=0.1,
                   help="curriculum_smooth_is: full-q mixture mass, 0 < epsilon <= 1; raw q/proposal importance weights")
    p.add_argument("--num-samples", type=int, default=grid["num_negatives"], help="IID draws WITH replacement from full q, an active half, or a mixture of reward halves")
    p.add_argument("--catalog-chunk-size", type=int, default=16, help="Candidates per history per decoder chunk")
    p.add_argument("--index", type=int, help="One dataset/combo/power/rho/seed cell, as printed by --list")
    p.add_argument("--list", action="store_true")
    p.add_argument("--count", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--cache-component", choices=COMPONENTS)
    p.add_argument("--output-root", type=Path, default=Path(default_output_root))
    p.add_argument("--cache-root", type=Path, help=cache_root_help or "Optional parent: <root>/<dataset>/<component>; default reuses matched exp4 factors")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-batch-size", type=int, default=256)
    p.add_argument("--cache-verify-batches", type=int, default=1)
    p.add_argument("--batch-size", type=int, help="Default: matched dataset protocol")
    p.add_argument("--grad-accum", type=int, help="Default: matched dataset protocol")
    p.add_argument("--lr", type=float, default=shared["lr"])
    p.add_argument("--weight-decay", type=float, default=shared["weight_decay"])
    p.add_argument("--max-epochs", type=int, default=shared["max_epochs"])
    p.add_argument("--patience", type=int, default=shared["patience"])
    p.add_argument("--min-epochs", type=int, default=shared["early_stopping_min_epochs"])
    p.add_argument("--calibration-batches", type=int, default=5)
    p.add_argument("--calibration-seed", type=int, default=1234)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-eval-batches", type=int)
    checkpoint_default = (parser_defaults or {}).get("checkpoint_every_updates")
    p.add_argument("--checkpoint-every-updates", type=int,
                   help="Atomic mid-epoch checkpoint interval; default " + (
                       str(checkpoint_default) if checkpoint_default is not None else "1 for exact, 0 for sampled"))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="Isolated caches/outputs, 8 training and 2 evaluation batches, one epoch")
    if configure_parser is not None:
        configure_parser(p)
    p.set_defaults(**(parser_defaults or {}))
    args = p.parse_args(argv)
    for key in ("powers", "rho_targets"):
        if any(not math.isfinite(x) or x <= 0 for x in getattr(args, key)):
            p.error(f"--{key.replace('_', '-')} must contain finite positive numbers")
    for key in ("datasets", "combos", "powers", "rho_targets", "seeds"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            p.error(f"--{key.replace('_', '-')} contains duplicate grid entries")
    for key in ("num_samples", "catalog_chunk_size", "cache_batch_size", "calibration_batches", "max_epochs", "patience", "batch_size", "grad_accum", "max_train_batches", "max_eval_batches"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.min_epochs < 0 or args.cache_verify_batches < 0:
        p.error("worker count, warm-up epochs, and verification batches must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("invalid learning rate or weight decay")
    if not math.isfinite(args.curriculum_epochs) or args.curriculum_epochs < 0:
        p.error("--curriculum-epochs must be finite and nonnegative")
    if not math.isfinite(args.proposal_q_mix) or not 0 < args.proposal_q_mix <= 1:
        p.error("--proposal-q-mix must be finite and in (0, 1]")
    # Keep metadata-only/Slurm preflight free of torch imports, while matching
    # the core's float32 bound on raw importance weights (at most 1/epsilon).
    if (args.sampling == "curriculum_smooth_is"
            and args.proposal_q_mix < 1.0 / float.fromhex("0x1.fffffep+127")):
        p.error("--proposal-q-mix is too small: importance weights must be representable in float32")
    if args.estimator == "exact" and args.sampling != "reward":
        p.error("--estimator exact computes the full sum and requires --sampling reward")
    if args.checkpoint_every_updates is None:
        args.checkpoint_every_updates = 1 if args.estimator == "exact" else 0
    elif args.checkpoint_every_updates < 0:
        p.error("--checkpoint-every-updates must be nonnegative")
    if args.cache_component and not args.cache_only:
        p.error("--cache-component requires --cache-only")
    if args.cache_only and args.index is not None:
        p.error("--cache-only selects datasets/components, not grid indices")
    args.output_root = absolute(args.output_root)
    if args.smoke:
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
    return args


def build_grid(args):
    # The original triple retains its entry representation as well as its
    # path, so existing identities and completed results remain compatible.
    return [dict(dataset=ds, power=float(power), rho=float(rho), seed=int(seed),
                 **({"combo": combo} if combo != DEFAULT_COMBO else {}))
            for ds, combo, power, rho, seed in itertools.product(
                args.datasets, args.combos, args.powers, args.rho_targets, args.seeds)]


def combo_key(entry):
    return entry.get("combo", DEFAULT_COMBO)


def components_for(entry):
    return tuple(combo_key(entry).split("_x_"))


def estimator_tag(args):
    if args.estimator == "exact":
        return "exact"
    if args.sampling == "curriculum_smooth_is":
        return (f"{args.sampling}_n{args.num_samples}_epochs{tag(args.curriculum_epochs)}"
                f"_qmix{tag(args.proposal_q_mix)}")
    if args.sampling != "reward":
        return f"{args.sampling}_n{args.num_samples}_epochs{tag(args.curriculum_epochs)}"
    return f"sampled_n{args.num_samples}"


def dataset_output_root(args, dataset):
    root = args.output_root / dataset
    namespace = getattr(args, "output_namespaces", {}).get(dataset)
    return root / namespace if namespace else root


def run_dir_for(args, entry):
    root = dataset_output_root(args, entry["dataset"])
    if combo_key(entry) != DEFAULT_COMBO:
        root = root / combo_key(entry)
    return (root / estimator_tag(args) / f"pow_{tag(entry['power'])}"
            / f"rho_{tag(entry['rho'])}" / f"seed_{entry['seed']}")


def curriculum_steps_for(args, num_examples, batch_size):
    if args.sampling == "reward":
        return 0
    batches = math.ceil(num_examples / batch_size)
    limit = 8 if args.smoke else args.max_train_batches
    if limit is not None:
        batches = min(batches, limit)
    return math.ceil(args.curriculum_epochs * batches)


def reward_config(config, entry):
    """Restrict new ablations without changing the original triple config."""
    if combo_key(entry) == DEFAULT_COMBO:
        return config
    config = json.loads(json.dumps(config))
    components = components_for(entry)
    config["method"]["reward"]["components"] = [
        spec for spec in config["method"]["reward"]["components"] if spec["name"] in components]
    if "sasrec" not in components:
        config["method"].pop("sasrec", None)
    return config


def asset_paths_for(base, config, components, component_paths):
    paths = [base.abspath(base.TIGER_CHECKPOINT)]
    if "sasrec" in components:
        paths.append(base.abspath(base.SASREC_CHECKPOINT))
    if "similarity" in components:
        paths.append(base.abspath(base.RQVAE_MODEL))
    return [*paths, *[Path(v) for k, v in config["data"].items()
                      if v and k in ("reviews_path", "metadata_path", "semantic_id_path", "interactions_path", "embeddings_path")],
            *component_paths, *[p.parent / "factors.pt" for p in component_paths]]


def build_identity(args, entry, config, trainer_config, assets, curriculum_steps):
    identity = {
        "version": 1, "entry": entry, "estimator": args.estimator, "num_samples": args.num_samples,
        "rng_policy": "seed + epoch for model and training loader",
        "catalog_chunk_size": args.catalog_chunk_size, "normalization": "valid_actions",
        "reward": "exp4 floored product sasrec * interaction * hierarchical similarity, then total power",
        "config": config, "trainer": trainer_config, "lr": args.lr, "weight_decay": args.weight_decay,
        "calibration_batches": 1 if args.smoke else args.calibration_batches,
        "calibration_seed": args.calibration_seed, "assets": assets,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    }
    if combo_key(entry) != DEFAULT_COMBO or args.sampling != "reward":
        identity.update(version=2, combo=combo_key(entry), sampling=args.sampling,
                        reward="exp4 floored product " + " * ".join(components_for(entry)) + ", then total power")
        if args.sampling != "reward":
            identity["curriculum"] = {"epochs": args.curriculum_epochs, "steps": curriculum_steps,
                "algorithm": "reward_halves", "phases": ["low reward half", "high reward half"],
                "probabilities": "q normalized within the active half", "replacement": True,
                "importance_correction": False}
            if args.sampling in {"curriculum_smooth", "curriculum_smooth_is"}:
                identity["curriculum"].update(
                    algorithm=SAMPLING_ALGORITHMS[args.sampling],
                    phases=["low reward half", "mixture of reward halves", "high reward half"],
                    probabilities="(1-alpha) * q_low_normalized + alpha * q_high_normalized",
                    high_half_probability="alpha = clip(global_microstep / steps, 0, 1); steps=0 gives alpha=1",
                )
            if args.sampling == "curriculum_smooth_is":
                identity.update(version=3, ce_target="q", calibration_sampling="reward")
                identity["curriculum"].update(
                    proposal_q_mix=args.proposal_q_mix, importance_correction=True,
                    probabilities="(1-epsilon)*((1-alpha)*q_low_normalized+alpha*q_high_normalized)+epsilon*q",
                    importance_weights="q/proposal; no clipping or sample self-normalization",
                    full_q_support=True,
                )
    if args.estimator == "exact":
        identity["exact_algorithm"] = "branching_prefix"
    return identity


def sampling_metadata(args):
    corrected = args.sampling == "curriculum_smooth_is"
    return {
        "ce_target": "q" if args.sampling == "reward" or corrected else "curriculum_proposal",
        "importance_correction": corrected,
        "proposal_q_mix": args.proposal_q_mix if corrected else None,
        "calibration_sampling": "reward" if corrected else args.sampling,
    }


def calibrate_distillation_weight(calibrate_fn, model, method, loader, **kwargs):
    """Use the same full-q calibration in both unbiased sampling arms.

    Keep the old calibration unchanged for existing modes. The temporary
    method config is restored even when calibration fails; its sampler never
    leaks into training or checkpoint metadata.
    """
    from dataclasses import replace
    original = method.cfg
    corrected = original.sampling == "curriculum_smooth_is"
    if corrected:
        method.cfg = replace(original, sampling="reward")
    try:
        calibration = calibrate_fn(model, method, loader, **kwargs)
    finally:
        method.cfg = original
    if corrected:
        calibration.update(sampling="reward", ce_target="q")
    return calibration


def load_runtime():
    for folder in (ROOT, ROOT / "scripts", Path(__file__).resolve().parent,
                   ROOT / "scripts/exp2-tuned-params", ROOT / "scripts/exp4-reward-ablations"):
        sys.path.insert(0, str(folder))
    import gflow_finetune_tiger_matched as matched
    return matched, matched.base


def matched_args(args, dataset, seed, matched):
    argv = ["--dataset", dataset, "--seed", str(seed), "--device", args.device,
            "--num-workers", str(args.num_workers), "--cache-batch-size", str(args.cache_batch_size),
            "--cache-verify-batches", str(args.cache_verify_batches), "--combos", "sasrec_x_interaction_x_similarity"]
    for key in ("batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.smoke:
        argv += ["--smoke"]
    base_args = matched.parse_args(argv)
    # No writes to exp4 run directories. Production caches can be shared;
    # smoke factors are always isolated and never replace complete factors.
    if args.smoke:
        base_args.cache_root = args.output_root / "reward_factors" / dataset
    elif args.cache_root is not None:
        base_args.cache_root = absolute(args.cache_root) / dataset
    base_args.output_root = args.output_root / dataset
    return base_args


def fingerprint(paths):
    """Small records pin assets without hashing multi-GB raw data every run."""
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in map(Path, paths)}


def check_identity(path, identity):
    if path.exists():
        saved = read_json(path)
        if saved == identity:
            return
        from checkpoint_identity import same_after_checkpoint_relocation
        if not same_after_checkpoint_relocation(saved, identity, root=ROOT, protocol_path=PROTOCOL):
            raise RuntimeError(f"Protocol/assets changed at {path.parent}; choose a new --output-root. Existing results were retained.")


def verify_caches(base, dm, indices, caches, model):
    import torch
    base.verify_cache_alignment(dm, indices, caches)
    reference_name = "similarity" if "similarity" in caches else next(iter(caches))
    reference = caches[reference_name]
    catalog = model.item_semantic_ids.detach().cpu()
    num_examples = len(dm.train_dataset) if indices is None else len(indices)
    for name, cache in caches.items():
        if cache.num_examples != num_examples:
            raise RuntimeError(f"{name}: cache has {cache.num_examples} rows, expected {num_examples}")
        if cache.num_items != len(catalog):
            raise RuntimeError(f"{name}: cache and TIGER catalog sizes differ")
        if "item_tokens" in cache.factors and not torch.equal(cache.factors["item_tokens"].cpu(), catalog):
            raise RuntimeError(f"{name}: cache and TIGER catalog order differ")
        for key in ("user_ids", "target_item_ids", "target_token_ids", "target_timestamps", "target_semantic_row_ids"):
            if key in reference.factors and key in cache.factors and not torch.equal(reference.factors[key], cache.factors[key]):
                raise RuntimeError(f"{name}: cache {key} disagrees with {reference_name} factors")
        if "target_token_ids" in cache.factors and "target_semantic_row_ids" in cache.factors:
            rows = cache.factors["target_semantic_row_ids"].long().cpu()
            if bool(rows.lt(0).any() | rows.ge(len(catalog)).any()) or not torch.equal(
                catalog[rows], cache.factors["target_token_ids"].cpu()
            ):
                raise RuntimeError(f"{name}: cached target tokens disagree with TIGER catalog order")


def make_sft(base):
    from flowrec.ft_methods.sft.config import SFTConfig
    return base.AliasedSFTMethod(SFTConfig(
        compute_train_metrics=False, compute_eval_metrics=True, compute_token_metrics=True,
        compute_item_metrics=False, evaluate_generation=True, generation_topk=(5,),
        generation_num_beams=5, generation_num_return_sequences=5,
        generation_constrain_valid_sequences=True,
    ))


def build_trainer_config(args, b, run_dir, device):
    from flowrec.training import TrainerConfig
    shared = read_json(PROTOCOL)["shared_schedule"]
    return TrainerConfig(
        max_epochs=1 if args.smoke else args.max_epochs,
        gradient_accumulation_steps=b.grad_accum, max_grad_norm=shared["max_grad_norm"],
        use_amp=device.type == "cuda", amp_dtype=shared["amp_dtype"],
        validate_every_epochs=shared["validate_every"], monitor=shared["monitor"], monitor_mode="max",
        early_stopping_patience=args.patience, early_stopping_min_epochs=0 if args.smoke else args.min_epochs,
        run_validation_at_start=True, val_sample_resample=False, save_best=True, save_last=True,
        val_sample_fraction=getattr(b, "val_sample_fraction", None), val_sample_seed=0,
        checkpoint_dir=run_dir / "checkpoints", log_every_steps=200, progress_bar=False,
        max_train_batches=8 if args.smoke else args.max_train_batches,
        max_eval_batches=2 if args.smoke else args.max_eval_batches,
    )


def run_entry(args, entry, matched, base, *, matched_args_fn=None, result_metadata_fn=None):
    import torch
    from flowrec.ft_methods.calibration import calibrate_component_weight
    from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache
    from flowrec.ft_methods.reward_distillation import RewardDistillationMethod
    from flowrec.training.loggers import CompositeExperimentLogger, JsonlExperimentLogger
    from resumable_trainer import EpochSeededTrainer

    started = time.perf_counter()
    dataset, seed = entry["dataset"], entry["seed"]

    b = (matched_args_fn or matched_args)(args, dataset, seed, matched)
    components = components_for(entry)
    config = reward_config(base.build_run_config(b), entry)
    run_dir = run_dir_for(args, entry)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "data_reward_config.json"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a local smoke test")
    trainer_cfg = build_trainer_config(args, b, run_dir, device)
    component_paths = [base.component_cache_dir(name, b) / "manifest.json" for name in components]
    missing = [str(p) for p in component_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Build reward factors first with --cache-only (or the Slurm submit helper): " + ", ".join(missing))
    asset_paths = asset_paths_for(base, config, components, component_paths)
    cached_examples = int(read_json(component_paths[0])["num_examples"])
    curriculum_steps = curriculum_steps_for(args, cached_examples, b.batch_size)
    identity = build_identity(args, entry, config, trainer_cfg.to_dict(), fingerprint(asset_paths), curriculum_steps)
    check_identity(run_dir / "identity.json", identity)
    if (run_dir / "result.json").exists() and read_json(run_dir / "result.json").get("status") == "completed":
        print(f"Already completed: {run_dir}", flush=True)
        return
    if (run_dir / "checkpoints/last.pt").exists() and not args.resume:
        raise RuntimeError(f"Unfinished run at {run_dir}; use --resume or a new --output-root")
    write_json(run_dir / "identity.json", identity)
    write_json(config_path, config)
    base.tiger_runner.seed_everything(seed)
    dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=config_path))
    dm.setup()
    indices = base.finetune_indices(dm.train_dataset, b)
    num_examples = len(dm.train_dataset) if indices is None else len(indices)
    if args.smoke:
        indices = (list(range(min(num_examples, base._smoke_examples(b)))) if indices is None
                   else indices[:base._smoke_examples(b)])
        num_examples = len(indices)
    print(f"[{dataset}] {num_examples} training examples; preparing {args.estimator} CE", flush=True)
    model = base.tiger_runner.build_model(config, dm, config_path=config_path)
    base.control.load_tiger_checkpoint(model, base.abspath(base.TIGER_CHECKPOINT))
    caches = {}
    for name, path in zip(components, component_paths):
        base.check_component_manifest(read_json(path), path.parent, name, num_examples, b)
        caches[name] = FactorizedRewardCache(path.parent, power=1.0)
    verify_caches(base, dm, indices, caches, model)
    reward_entry = {"combo": components, "power": entry["power"], "sasrec_transform": "sigmoid",
                    "sasrec_temperature": 1.0, "interaction_exponent": 1.0}
    cache = base.composed_cache(reward_entry, caches)
    model.to(device)
    nll = base.control.sanity_check_checkpoint(model, dm, device, 6.0)
    print(f"[{dataset}] init NLL={nll:.4f}; catalog={cache.num_items}", flush=True)
    method = RewardDistillationMethod(cache, model.item_semantic_ids, make_sft(base),
        estimator=args.estimator, num_samples=args.num_samples, catalog_chunk_size=args.catalog_chunk_size,
        sampling=args.sampling, curriculum_steps=curriculum_steps, proposal_q_mix=args.proposal_q_mix)
    method.to(device)
    cal_start = time.perf_counter()
    calibration_path = run_dir / "calibration.json"
    last_checkpoint = run_dir / "checkpoints/last.pt"
    if last_checkpoint.exists() and not calibration_path.exists():
        raise RuntimeError("Cannot resume without calibration.json")
    if calibration_path.exists():
        calibration = read_json(calibration_path)
    else:
        loader = base.indexed_train_loader(dm, indices, shuffle=not args.smoke, batch_size=b.batch_size,
                                           seed=args.calibration_seed, num_workers=0)
        calibration = calibrate_distillation_weight(calibrate_component_weight, model, method, loader, component="distill",
            target_ratio=entry["rho"], num_batches=1 if args.smoke else args.calibration_batches,
            seed=args.calibration_seed, device=device)
        calibration["elapsed_sec"] = time.perf_counter() - cal_start
        write_json(calibration_path, calibration)
    method.distill_weight = calibration["weight"]
    print(f"[{dataset}] power={entry['power']:g}, rho={entry['rho']:g}, lambda_CE={method.distill_weight:.6g}", flush=True)
    logger = CompositeExperimentLogger([JsonlExperimentLogger(run_dir / "training_log.jsonl"),
                                       base.base.make_status_logger(str(run_dir.relative_to(args.output_root)))])
    trainer = EpochSeededTrainer.from_optimizer_config(model=model, method=method, cfg=trainer_cfg, device=device,
        lr=args.lr, weight_decay=args.weight_decay, logger=logger,
        seed=seed, checkpoint_every_updates=args.checkpoint_every_updates)
    index = base.control.CatalogIndex(model, dm)
    baseline_path = args.output_root / dataset / "baseline" / ("smoke.json" if args.smoke else "init_metrics.json")
    baseline_identity = {"checkpoint": identity["assets"][str(base.abspath(base.TIGER_CHECKPOINT))],
                         "data": config["data"], "dataset": config["dataset"], "eval_beams": b.eval_num_beams,
                         "data_assets": {str(config["data"][key]): identity["assets"][str(config["data"][key])]
                                         for key in ("reviews_path", "metadata_path", "semantic_id_path", "interactions_path", "embeddings_path")
                                         if config["data"].get(key)},
                         "ks": b.test_ks, "max_batches": trainer_cfg.max_eval_batches}
    with base.control.file_lock(baseline_path.with_suffix(".lock")):
        if baseline_path.exists():
            if read_json(baseline_path)["identity"] != baseline_identity:
                raise RuntimeError("Baseline protocol changed; choose a new --output-root")
        else:
            write_json(baseline_path, {"identity": baseline_identity, "dataset": dataset, "variant": "init",
                "splits": base.control.evaluate_splits(model, dm, index, b, trainer_cfg.max_eval_batches)})
    base.tiger_runner.seed_everything(seed)
    resumed = None
    if args.resume and last_checkpoint.exists():
        payload = trainer.load_checkpoint(last_checkpoint)
        resumed = trainer.state.epoch
        trainer.restore_progress(payload)
    write_json(run_dir / "resolved_config.json", {"identity": identity, "method": method.cfg.to_dict(),
        "reward": cache.describe(), "calibration": calibration, "num_examples": num_examples,
        "catalog_size": cache.num_items, "baseline": str(baseline_path),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_every_updates": args.checkpoint_every_updates,
        "cache_manifests": {name: read_json(path) for name, path in zip(components, component_paths)}})
    setup_sec = time.perf_counter() - started
    training_start = time.perf_counter()
    loader = base.indexed_train_loader(dm, indices, shuffle=not args.smoke, batch_size=b.batch_size,
                                      seed=seed, reward_cache=cache, prefetch_factor=2)
    fit = trainer.fit(loader, dm.val_dataloader())
    training_sec = time.perf_counter() - training_start
    write_json(run_dir / "training_history.json", {"history": fit.state.history})
    best = run_dir / "checkpoints/best.pt"
    if not best.exists():
        raise RuntimeError("Training did not produce the validation-selected checkpoint")
    trainer.load_checkpoint(best, load_optimizer=False, load_scheduler=False)
    eval_start = time.perf_counter()
    splits = base.control.evaluate_splits(model, dm, index, b, trainer_cfg.max_eval_batches)
    # fit.state remains the final training state when trainer loads best.pt.
    result = {**entry, "combo": combo_key(entry), "sampling": args.sampling,
        **sampling_metadata(args),
        "sampling_algorithm": SAMPLING_ALGORITHMS[args.sampling],
        "curriculum_steps": curriculum_steps, "curriculum_epochs": args.curriculum_epochs if args.sampling != "reward" else None,
        "status": "completed", "variant": "sft_plus_ce", "estimator": args.estimator,
        "num_samples": args.num_samples if args.estimator == "sampled" else None,
        "lambda_distill": method.distill_weight, "best_epoch": fit.state.best_epoch,
        "best_val_ndcg5": fit.state.best_metric, "epochs_completed": fit.state.epoch + 1,
        "micro_steps": fit.state.global_step, "optimizer_steps": fit.state.optimizer_step,
        "early_stopped": fit.state.should_stop, "resumed_from_epoch": resumed,
        "setup_sec_this_invocation": setup_sec, "fit_sec_this_invocation": training_sec,
        "evaluation_sec": time.perf_counter() - eval_start,
        "evaluation_scope": "limited" if trainer_cfg.max_eval_batches is not None else "full",
        "num_examples": num_examples, "catalog_size": cache.num_items,
        "run_dir": str(run_dir), "checkpoint": str(best), "baseline": str(baseline_path), "splits": splits}
    if result_metadata_fn is not None:
        result.update(result_metadata_fn(args, entry, b, config))
    write_json(run_dir / "result.json", result)
    logger.close()
    print(json.dumps(result, indent=2), flush=True)


def summary_dir_for(args):
    root = args.output_root
    if getattr(args, "summary_namespace", None):
        root = root / args.summary_namespace
    if args.combos != [DEFAULT_COMBO]:
        root = root / "ablations"
        if len(args.combos) == 1:
            root = root / args.combos[0]
    return root / estimator_tag(args)


def aggregate(args):
    """Explicit post-array aggregation; never select hyperparameters on test."""
    results, missing = [], []
    for entry in build_grid(args):
        path = run_dir_for(args, entry) / "result.json"
        result = read_json(path) if path.exists() else {}
        if result.get("status") == "completed":
            result = {**result, "combo": combo_key(result), "sampling": result.get("sampling", "reward")}
            results.append(result)
        else:
            missing.append(entry)
    rows = []
    for result in results:
        for split, metrics in result["splits"].items():
            rows.append({**{k: result[k] for k in ("dataset", "combo", "sampling", "power", "rho", "seed", "estimator", "num_samples",
                "lambda_distill", "best_epoch", "best_val_ndcg5", "optimizer_steps", "fit_sec_this_invocation", "run_dir")},
                "curriculum_epochs": result.get("curriculum_epochs"), "curriculum_steps": result.get("curriculum_steps", 0),
                "ce_target": result.get("ce_target"), "importance_correction": result.get("importance_correction"),
                "proposal_q_mix": result.get("proposal_q_mix"),
                "split": split, **metrics})
    output = summary_dir_for(args)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else [
        "dataset", "combo", "sampling", "power", "rho", "seed", "estimator", "split", "run_dir"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Choose power/rho within each reward ablation separately. The reward
    # components are the experiment being compared, not a selection dimension.
    # Incomplete seed groups are excluded; test never enters selection.
    selected = {}
    for dataset in args.datasets:
        by_combo = {}
        for combo in args.combos:
            candidates = []
            for power, rho in itertools.product(args.powers, args.rho_targets):
                group = [r for r in results if r["dataset"] == dataset and r["combo"] == combo
                         and r["sampling"] == args.sampling and r["power"] == power and r["rho"] == rho]
                if len(group) == len(args.seeds) and {r["seed"] for r in group} == set(args.seeds):
                    score = sum(r["splits"]["validation"]["NDCG@5"] for r in group) / len(group)
                    candidates.append({"power": power, "rho": rho, "mean_validation_ndcg5": score,
                                       "runs": [r["run_dir"] for r in group]})
            if candidates:
                by_combo[combo] = max(candidates, key=lambda c: c["mean_validation_ndcg5"])
        if by_combo:
            selected[dataset] = by_combo[DEFAULT_COMBO] if args.combos == [DEFAULT_COMBO] else by_combo
    summary = {"completed": len(results), "expected": len(build_grid(args)), "missing": missing,
               **sampling_metadata(args),
               "selection_metric": "mean validation NDCG@5 (beam 20)",
               "evaluation_scope": "limited" if args.smoke or args.max_eval_batches is not None else "full",
               "combos": args.combos, "sampling": args.sampling,
               "sampling_algorithm": SAMPLING_ALGORITHMS[args.sampling],
               "curriculum_epochs": args.curriculum_epochs if args.sampling != "reward" else None,
               "selected": selected}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    args = parse_args(argv)
    grid = build_grid(args)
    if args.count:
        print(len(grid))
        return
    if args.list:
        for i, entry in enumerate(grid):
            print(f"{i:4d}  {entry['dataset']:12s} combo={combo_key(entry)} power={entry['power']:g} "
                  f"rho={entry['rho']:g} seed={entry['seed']} {args.estimator} sampling={args.sampling}")
        print(f"total: {len(grid)} runs")
        return
    if args.aggregate_only:
        aggregate(args)
        return
    if args.index is not None:
        if not 0 <= args.index < len(grid):
            raise SystemExit(f"--index must lie in 0..{len(grid)-1}")
        grid = [grid[args.index]]
    matched, base = load_runtime()
    if args.cache_only:
        for dataset in args.datasets:
            b = matched_args(args, dataset, 42, matched)
            config = base.build_run_config(b)
            needed = [name for name in COMPONENTS if any(name in combo.split("_x_") for combo in args.combos)]
            for component in ([args.cache_component] if args.cache_component else needed):
                destination = args.output_root / dataset / "cache_build" / component
                destination.mkdir(parents=True, exist_ok=True)
                cache_start = time.perf_counter()
                existed = (base.component_cache_dir(component, b) / "manifest.json").exists()
                manifest = base.ensure_component_cache(component, config, destination, b, None)
                write_json(destination / "timing.json", {"component": component, "dataset": dataset,
                    "reused": existed, "elapsed_sec": time.perf_counter() - cache_start,
                    "num_examples": manifest["num_examples"], "device": args.device})
        return
    for entry in grid:
        # Serial invocations also acquire a per-run lock; duplicate array
        # submissions wait and then observe the completed result.
        with base.control.file_lock(run_dir_for(args, entry) / "run.lock"):
            run_entry(args, entry, matched, base)


if __name__ == "__main__":
    main()
