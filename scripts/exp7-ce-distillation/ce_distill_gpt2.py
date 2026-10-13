#!/usr/bin/env python
"""SFT + reward CE distillation of the exp2 GPT-2 checkpoints.

Training prefixes expand the active labels of the original dense-causal
windows, retaining their context boundaries, catalog, and held-out examples.
Grid, objectives, reward halves, and evaluation match the exp7 TIGER arm.
--list / --count / --aggregate-only require only the standard library.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import ce_distill_tiger as shared

ROOT = shared.ROOT
PROTOCOL = shared.PROTOCOL
COMPONENTS = shared.COMPONENTS
COMBOS = shared.COMBOS
DEFAULT_COMBO = shared.DEFAULT_COMBO
SAMPLING_ALGORITHMS = shared.SAMPLING_ALGORITHMS
DEFAULT_OUTPUT_ROOT = "experiments/exp7-ce-distillation-gpt2"
DEFAULT_CACHE_ROOT = "artifacts/exp7-ce-distillation-gpt2"
INTERNAL_COMPONENTS = {"sasrec": "sasrec", "interaction": "interaction", "similarity": "sim_hier"}

read_json = shared.read_json
write_json = shared.write_json
absolute = shared.absolute
build_grid = shared.build_grid
combo_key = shared.combo_key
components_for = shared.components_for
run_dir_for = shared.run_dir_for
estimator_tag = shared.estimator_tag
curriculum_steps_for = shared.curriculum_steps_for
fingerprint = shared.fingerprint
check_identity = shared.check_identity
aggregate = shared.aggregate
summary_dir_for = shared.summary_dir_for


def parse_args(argv=None):
    return shared.parse_args(argv, description=__doc__, default_output_root=DEFAULT_OUTPUT_ROOT,
        cache_root_help="GPT-2 factors: <root>/<dataset>/reward_factors/<component>; default artifacts/exp7-ce-distillation-gpt2")


def load_runtime():
    for folder in (ROOT, ROOT / "scripts", SCRIPT_DIR,
                   ROOT / "scripts/exp2-tuned-params", ROOT / "scripts/exp4-reward-ablations"):
        sys.path.insert(0, str(folder))
    import gflow_finetune_gpt2_amazon_fullhist as matched
    return matched, matched.base


def matched_args(args, dataset, seed, matched):
    argv = ["--datasets", dataset, "--seed", str(seed), "--device", args.device,
            "--num-workers", str(args.num_workers), "--cache-batch-size", str(args.cache_batch_size),
            "--cache-verify-batches", str(args.cache_verify_batches), "--cache-format", "factorized",
            "--combos", DEFAULT_COMBO, "--similarity-modes", "hierarchical"]
    for key in ("batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.smoke:
        argv += ["--smoke"]
    b = matched.base.dataset_args(matched.parse_args(argv), dataset)
    # Expand the original causal windows, not rolling prefixes of each user.
    b.prefix_from_dense = True
    b.cache_root = (args.output_root / "cache" if args.smoke
                    else absolute(args.cache_root or DEFAULT_CACHE_ROOT))
    b.output_root = args.output_root
    return b


def build_run_config(base, dataset, b):
    profile, _ = base.tuned_profile(dataset)
    model_yaml = base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML)
    user_count = base.gpt2_runner.user_ids_count(model_yaml, profile)
    config = base.build_run_config(dataset, b, user_count)
    config["experiment_name"] = f"ce_distill_gpt2_{dataset}"
    # The datamodule only needs user-token settings in `model`. Pin the full
    # checkpoint architecture separately for human review and resume identity.
    config["pretrained_model"] = profile["model"]
    if not config["dataset"].get("train_prefix_from_dense"):
        raise RuntimeError("GPT-2 CE requires prefixes expanded from dense-causal windows")
    return config


def reward_config(config, entry):
    config = json.loads(json.dumps(config))
    keep = {INTERNAL_COMPONENTS[name] for name in components_for(entry)}
    method = config["method"]
    selected = [(key, spec) for key, spec in zip(method["component_keys"], method["reward"]["components"])
                if key in keep]
    method["component_keys"] = [key for key, _ in selected]
    method["reward"]["components"] = [spec for _, spec in selected]
    if "sasrec" not in keep:
        method.pop("sasrec", None)
    return config


def component_cache_dir(base, dataset, name, b):
    return base.component_cache_dir(dataset, INTERNAL_COMPONENTS[name], b)


def check_component_manifest(base, config, path, name, num_examples, b):
    from flowrec.data.dense_prefix import dense_prefix_signature
    manifest = read_json(path)
    if manifest.get("training_context") != dense_prefix_signature(config["dataset"]):
        raise RuntimeError(f"{path}: reward factors have a different training_context; use GPT-2 dense-window-prefix factors")
    for key, expected in base.finetune_subset_spec(b).items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"{path}: cache subset {key}={manifest.get(key)!r}, expected {expected!r}")
    return base.factor_runner.check_component_manifest(manifest, path.parent, name, num_examples, b,
        similarity_mode="hierarchical" if name == "similarity" else None)


def verify_caches(base, dm, indices, caches, model):
    import torch
    base.factor_runner.verify_cache_alignment(dm, indices, caches)
    reference = caches.get("similarity", next(iter(caches.values())))
    catalog = model.item_semantic_ids.detach().cpu()
    for name, cache in caches.items():
        if cache.num_examples != len(indices):
            raise RuntimeError(f"{name}: cache has {cache.num_examples} rows, expected {len(indices)}")
        if cache.num_items != len(catalog):
            raise RuntimeError(f"{name}: cache and GPT-2 catalog sizes differ")
        if "item_tokens" in cache.factors and not torch.equal(cache.factors["item_tokens"].cpu(), catalog):
            raise RuntimeError(f"{name}: cache and GPT-2 catalog order differ")
        for key in ("user_ids", "target_item_ids", "target_token_ids", "target_timestamps", "target_semantic_row_ids"):
            if key in reference.factors and key in cache.factors and not torch.equal(reference.factors[key], cache.factors[key]):
                raise RuntimeError(f"{name}: cache {key} disagrees with other factors")
        if "target_token_ids" in cache.factors and "target_semantic_row_ids" in cache.factors:
            rows = cache.factors["target_semantic_row_ids"].long().cpu()
            if bool(rows.lt(0).any() | rows.ge(len(catalog)).any()) or not torch.equal(catalog[rows], cache.factors["target_token_ids"].cpu()):
                raise RuntimeError(f"{name}: cached target tokens disagree with GPT-2 catalog order")


def asset_paths_for(base, dataset, config, components, component_paths):
    checkpoint_paths = base.tuned_checkpoint_paths(dataset)
    paths = [checkpoint_paths["checkpoint"], checkpoint_paths["best_trial"],
             base.EXP1_GPT2_MODEL_YAML, base.dataset_yaml_path(dataset)]
    if checkpoint_paths["sidecar"].exists():
        paths.append(checkpoint_paths["sidecar"])
    if "sasrec" in components:
        paths.append(base.abspath(base.DATASETS[dataset]["sasrec_checkpoint"]))
        paths.append(base.abspath(base.DATASETS[dataset]["sasrec_trial"]))
    if "similarity" in components:
        paths.append(base.rqvae_model_path(dataset))
    return [*map(Path, paths), *[Path(v) for k, v in config["data"].items()
        if k in ("reviews_path", "metadata_path", "semantic_id_path")],
        *component_paths, *[p.parent / "factors.pt" for p in component_paths]]


def build_identity(args, entry, config, trainer_config, assets, curriculum_steps):
    identity = shared.build_identity(args, entry, config, trainer_config, assets, curriculum_steps)
    identity.update(model="gpt2", data_protocol="active dense-causal labels expanded to prefixes within their original windows",
                    reward_storage="factorized float32 reconstruction", cache_context_version=1)
    return identity


def make_sft(base):
    return base.gflow_ablation.AliasedSFTMethod(base.light_validation_sft_cfg())


def evaluate_splits(base, model, dm, b, device, max_batches):
    return base.gpt2_beauty.evaluate_splits(model, dm, b, device, max_batches,
                                          splits=("validation", "test"))


def run_entry(args, entry, matched, base):
    import torch
    from flowrec.ft_methods.calibration import calibrate_component_weight
    from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache, ComposedFactorizedRewardCache
    from flowrec.ft_methods.reward_distillation import RewardDistillationMethod
    from flowrec.training import TrainerConfig
    from flowrec.training.loggers import CompositeExperimentLogger, JsonlExperimentLogger
    from resumable_trainer import EpochSeededTrainer

    started = time.perf_counter()
    dataset, seed = entry["dataset"], entry["seed"]
    b = matched_args(args, dataset, seed, matched)
    components = components_for(entry)
    config = reward_config(build_run_config(base, dataset, b), entry)
    run_dir = run_dir_for(args, entry)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "data_reward_config.json"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a local smoke test")
    schedule = read_json(PROTOCOL)["shared_schedule"]
    trainer_cfg = TrainerConfig(
        max_epochs=1 if args.smoke else args.max_epochs,
        gradient_accumulation_steps=b.grad_accum, max_grad_norm=schedule["max_grad_norm"],
        use_amp=device.type == "cuda", amp_dtype=schedule["amp_dtype"],
        validate_every_epochs=schedule["validate_every"], monitor=schedule["monitor"], monitor_mode="max",
        early_stopping_patience=args.patience, early_stopping_min_epochs=0 if args.smoke else args.min_epochs,
        run_validation_at_start=True, val_sample_resample=False, save_best=True, save_last=True,
        checkpoint_dir=run_dir / "checkpoints", log_every_steps=200, progress_bar=False,
        max_train_batches=8 if args.smoke else args.max_train_batches,
        max_eval_batches=2 if args.smoke else args.max_eval_batches,
    )
    component_paths = [component_cache_dir(base, dataset, name, b) / "manifest.json" for name in components]
    missing = [str(path) for path in component_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Build GPT-2 reward factors first with --cache-only: " + ", ".join(missing))
    assets = fingerprint(asset_paths_for(base, dataset, config, components, component_paths))
    cached_examples = int(read_json(component_paths[0])["num_examples"])
    curriculum_steps = curriculum_steps_for(args, cached_examples, b.batch_size)
    identity = build_identity(args, entry, config, trainer_cfg.to_dict(), assets, curriculum_steps)
    check_identity(run_dir / "identity.json", identity)
    if (run_dir / "result.json").exists() and read_json(run_dir / "result.json").get("status") == "completed":
        print(f"Already completed: {run_dir}", flush=True)
        return
    last_checkpoint = run_dir / "checkpoints/last.pt"
    if last_checkpoint.exists() and not args.resume:
        raise RuntimeError(f"Unfinished run at {run_dir}; use --resume or a new --output-root")
    write_json(run_dir / "identity.json", identity)
    write_json(config_path, config)
    base.runner.seed_everything(seed)
    dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=config_path))
    dm.setup()
    write_json(run_dir / "dense_prefix_audit.json", dm.dense_prefix_audit)
    indices = base.finetune_indices(dm.train_dataset, b)
    if args.smoke:
        indices = indices[:base._smoke_examples(b)]
    print(f"[{dataset}] {len(indices)} dense-window training prefixes; preparing {args.estimator} CE", flush=True)
    model, _ = base.build_gpt2_from_tuned_checkpoint(dataset, dm)
    caches = {}
    for name, path in zip(components, component_paths):
        check_component_manifest(base, config, path, name, len(indices), b)
        caches[name] = FactorizedRewardCache(path.parent, power=1.0)
    verify_caches(base, dm, indices, caches, model)
    cache = ComposedFactorizedRewardCache([(name, caches[name], 1.0) for name in components],
                                          composition="multiply", power=entry["power"])
    model.to(device)
    nll = base.gpt2_beauty.sanity_check_init_nll(model, dm, device, threshold=6.0)
    print(f"[{dataset}] init NLL={nll:.4f}; catalog={cache.num_items}", flush=True)
    method = RewardDistillationMethod(cache, model.item_semantic_ids, make_sft(base),
        estimator=args.estimator, num_samples=args.num_samples, catalog_chunk_size=args.catalog_chunk_size,
        sampling=args.sampling, curriculum_steps=curriculum_steps)
    method.to(device)
    calibration_path = run_dir / "calibration.json"
    if last_checkpoint.exists() and not calibration_path.exists():
        raise RuntimeError("Cannot resume without calibration.json")
    if calibration_path.exists():
        calibration = read_json(calibration_path)
    else:
        cal_start = time.perf_counter()
        loader = base.gflow_ablation.indexed_train_loader(dm, indices, shuffle=not args.smoke,
            batch_size=b.batch_size, seed=args.calibration_seed, num_workers=0)
        calibration = calibrate_component_weight(model, method, loader, component="distill",
            target_ratio=entry["rho"], num_batches=1 if args.smoke else args.calibration_batches,
            seed=args.calibration_seed, device=device)
        calibration["elapsed_sec"] = time.perf_counter() - cal_start
        write_json(calibration_path, calibration)
    method.distill_weight = calibration["weight"]
    print(f"[{dataset}] power={entry['power']:g}, rho={entry['rho']:g}, lambda_CE={method.distill_weight:.6g}", flush=True)
    logger = CompositeExperimentLogger([JsonlExperimentLogger(run_dir / "training_log.jsonl"),
        base.gflow_ablation.make_status_logger(str(run_dir.relative_to(args.output_root)))])
    trainer = EpochSeededTrainer.from_optimizer_config(model=model, method=method, cfg=trainer_cfg, device=device,
        lr=args.lr, weight_decay=args.weight_decay, logger=logger,
        seed=seed, checkpoint_every_updates=args.checkpoint_every_updates)
    checkpoint_path = base.tuned_checkpoint_paths(dataset)["checkpoint"]
    baseline_path = args.output_root / dataset / "baseline" / ("smoke.json" if args.smoke else "init_metrics.json")
    baseline_identity = {"model": "gpt2", "checkpoint": assets[str(checkpoint_path)],
        "pretrained_model": config["pretrained_model"], "data": config["data"], "dataset": config["dataset"],
        "data_assets": {str(value): assets[str(value)] for key, value in config["data"].items()
                        if key in ("reviews_path", "metadata_path", "semantic_id_path")},
        "eval_beams": b.eval_num_beams, "ks": b.test_ks, "max_batches": trainer_cfg.max_eval_batches}
    with base.control.file_lock(baseline_path.with_suffix(".lock")):
        if baseline_path.exists():
            if read_json(baseline_path)["identity"] != baseline_identity:
                raise RuntimeError("Baseline protocol changed; choose a new --output-root")
        else:
            write_json(baseline_path, {"identity": baseline_identity, "dataset": dataset, "variant": "init",
                "splits": evaluate_splits(base, model, dm, b, device, trainer_cfg.max_eval_batches)})
    base.runner.seed_everything(seed)
    resumed = None
    if args.resume and last_checkpoint.exists():
        payload = trainer.load_checkpoint(last_checkpoint)
        resumed = trainer.state.epoch
        trainer.restore_progress(payload)
    write_json(run_dir / "resolved_config.json", {"identity": identity, "method": method.cfg.to_dict(),
        "reward": cache.describe(), "calibration": calibration, "num_examples": len(indices),
        "catalog_size": cache.num_items, "baseline": str(baseline_path),
        "dense_prefix_audit": dm.dense_prefix_audit,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_every_updates": args.checkpoint_every_updates,
        "cache_manifests": {name: read_json(path) for name, path in zip(components, component_paths)}})
    setup_sec = time.perf_counter() - started
    training_start = time.perf_counter()
    loader = base.gflow_ablation.indexed_train_loader(dm, indices, shuffle=not args.smoke,
        batch_size=b.batch_size, seed=seed, reward_cache=cache, prefetch_factor=2)
    fit = trainer.fit(loader, dm.val_dataloader())
    training_sec = time.perf_counter() - training_start
    write_json(run_dir / "training_history.json", {"history": fit.state.history})
    best = run_dir / "checkpoints/best.pt"
    if not best.exists():
        raise RuntimeError("Training did not produce the validation-selected checkpoint")
    trainer.load_checkpoint(best, load_optimizer=False, load_scheduler=False)
    eval_start = time.perf_counter()
    splits = evaluate_splits(base, model, dm, b, device, trainer_cfg.max_eval_batches)
    result = {**entry, "model": "gpt2", "combo": combo_key(entry), "sampling": args.sampling,
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
        "num_examples": len(indices), "catalog_size": cache.num_items,
        "run_dir": str(run_dir), "checkpoint": str(best), "baseline": str(baseline_path), "splits": splits}
    write_json(run_dir / "result.json", result)
    logger.close()
    print(json.dumps(result, indent=2), flush=True)


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
            config = build_run_config(base, dataset, b)
            needed = [name for name in COMPONENTS if any(name in combo.split("_x_") for combo in args.combos)]
            for name in ([args.cache_component] if args.cache_component else needed):
                destination = args.output_root / dataset / "cache_build" / name
                destination.mkdir(parents=True, exist_ok=True)
                cache_start = time.perf_counter()
                existed = (component_cache_dir(base, dataset, name, b) / "manifest.json").exists()
                manifest = base.ensure_component_cache(dataset, INTERNAL_COMPONENTS[name], config,
                                                       destination, None, b)
                write_json(destination / "timing.json", {"component": name, "dataset": dataset,
                    "model": "gpt2", "reused": existed, "elapsed_sec": time.perf_counter() - cache_start,
                    "num_examples": manifest["num_examples"], "device": args.device})
        return
    for entry in grid:
        with base.control.file_lock(run_dir_for(args, entry) / "run.lock"):
            run_entry(args, entry, matched, base)


if __name__ == "__main__":
    main()
