"""Fixed-epoch train+validation refits of the previously selected global experiments."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import gflow_finetune_yambda as global_runs
import ce_distill_gpt2 as ce
import ce_distill_gpt2_yambda as ce_cache
import gflow_finetune_gpt2_amazon as gpt
import gflow_finetune_yambda500m as tiger
from from_sft_runtime import write_json
from refit_sources import verify_source
from resumable_trainer import EpochSeededTrainer
from flowrec.ft_methods.calibration import calibrate_component_weight
from flowrec.ft_methods.gflow.factorized_cache import ComposedFactorizedRewardCache, FactorizedRewardCache
from flowrec.ft_methods.reward_distillation import RewardDistillationMethod
from flowrec.training import TrainerConfig
from flowrec.training.loggers import JsonlExperimentLogger
import torch


def _settings(task, resolved, run_dir, device):
    """Use the actual completed run's batch/accumulation, including OOM retries."""
    identity = resolved.get("identity", {})
    trainer = TrainerConfig.from_dict(identity["trainer"] if task["arm"] == "ce" else resolved["trainer"])
    trainer = replace(trainer, max_epochs=int(task["epochs"]), max_steps=None, max_train_batches=None,
                      max_eval_batches=None, checkpoint_dir=run_dir / "checkpoints",
                      early_stopping_patience=None, early_stopping_min_epochs=0, save_best=False,
                      save_last=True, save_every_epochs=None, run_validation_at_start=False,
                      validate_every_epochs=None, validate_every_steps=None, val_sample_fraction=None,
                      val_sample_resample=False, use_amp=trainer.use_amp and device.type == "cuda")
    config = identity.get("config", resolved.get("config", {}))
    dataloader = config.get("dataloader", {})
    batching = resolved.get("batching", {})
    batch = batching.get("batch_size") or dataloader["train_batch_size"]
    if batching.get("gradient_accumulation", trainer.gradient_accumulation_steps) != trainer.gradient_accumulation_steps:
        raise ValueError("Conflicting batch accumulation in source resolved_config.json")
    optimizer = identity if task["arm"] == "ce" else resolved["optimizer"]
    protocol = json.loads(global_runs.PROTOCOL.read_text())
    block = protocol["datasets"][task["dataset"]]
    schedule = protocol["shared_schedule"]
    cli = resolved.get("cli_args", {})
    sampler = resolved.get("sampler", {})
    if not isinstance(sampler, dict):
        sampler = {}
    pipeline = resolved.get("input_pipeline", {})
    calibration = resolved.get("calibration", {})
    return trainer, SimpleNamespace(
        lr=float(optimizer["lr"]), weight_decay=float(optimizer["weight_decay"]),
        batch_size=int(batch), seed=int(task["seed"]), smoke=False,
        test_batch_size=int(batching.get("test_batch_size", block["test_batch_size"])),
        test_ks=cli.get("test_ks", schedule["test_ks"]),
        eval_num_beams=int(cli.get("eval_num_beams", schedule["eval_num_beams"])),
        curriculum_epochs=sampler.get("curriculum_epochs"), curriculum_steps=int(sampler.get("curriculum_steps", 0)),
        sampler_temperature=float(sampler.get("temperature", 1.)), strict_negatives=sampler.get("strict_negatives", True),
        policy_extra_beams=sampler.get("policy_extra_beams"), reward_prefetch=pipeline.get("reward_prefetch", True),
        tensorized_train_loader=pipeline.get("tensorized_train_loader", True),
        loader_prefetch_factor=int(pipeline.get("prefetch_factor", 2)),
        gflow_calibration_seed=int(identity.get("calibration_seed", cli.get("gflow_calibration_seed", 1234))),
        gflow_calibration_batches=int(identity.get("calibration_batches", cli.get("gflow_calibration_batches",
                                                                                  calibration.get("num_batches", 5)))),
        gflow_calibration_eps=float(cli.get("gflow_calibration_eps", 1e-12)),
        catalog_chunk_size=int(identity.get("catalog_chunk_size", 16)),
    )


def _load_student(family, config, dm, checkpoint, config_path):
    if family == "gpt2":
        defaults = gpt.runner.load_yaml(global_runs.ROOT / "configs/global-time/exp1-standard_params/gpt2_model.yaml")
        model = gpt.gpt2_runner.build_model(defaults, {"model": config["model"]}, dm)
        gpt.gpt2_beauty.load_gpt2_checkpoint(model, checkpoint)
    elif family == "tiger":
        model = gpt.tiger_runner.build_model(config, dm, config_path=config_path)
        gpt.control.load_tiger_checkpoint(model, checkpoint)
    else:
        raise ValueError(f"Unsupported refit student {family}")
    gpt.gpt2_beauty.check_semantic_row_table(model, dm)
    return model


def _calibration(run_dir, calibrate):
    path = run_dir / "calibration.json"
    if path.is_file():
        return json.loads(path.read_text())
    if (run_dir / "checkpoints/last.pt").exists():
        raise RuntimeError(f"Refit calibration missing for existing optimizer state at {run_dir}")
    record = calibrate()
    write_json(path, record)
    return record


def _method(task, args, resolved, config, model, dm, cache_dirs, run_dir, device):
    is_gpt = task["model"] == "gpt2"
    sft = ce.make_sft(gpt) if is_gpt else ce.shared.make_sft(gpt.gflow_ablation)
    if task["arm"] == "sft":
        return sft, None, {"name": "sft_only"}
    caches = {k: FactorizedRewardCache(cache_dirs[k], power=1.) for k in global_runs.COMPONENTS}
    ce_cache.verify_caches(gpt, dm, None, caches, model)
    reward = ComposedFactorizedRewardCache([(k, cache, 1.) for k, cache in caches.items()],
                                           composition="multiply", power=float(task["power"]))
    if task["arm"] == "ce":
        args.estimator, args.sampling = task["estimator"], task["sampling"]
        args.num_samples, args.proposal_q_mix = int(task["num_samples"]), 0.1
        if (args.estimator, args.sampling, task["importance_correction"]) != ("sampled", "reward_negatives", False):
            raise ValueError("Only the selected negative-only, uncorrected CE is part of this experiment")
        method = RewardDistillationMethod(reward, model.item_semantic_ids, sft,
            **ce.shared.distillation_sampling_args(args, dm, gpt.tiger_runner, curriculum_steps=0)).to(device)

        def calibrate_ce():
            loader = gpt.gflow_ablation.indexed_train_loader(dm, None, shuffle=True, batch_size=args.batch_size,
                seed=args.gflow_calibration_seed, num_workers=0, tensorized=args.tensorized_train_loader)
            return ce.shared.calibrate_distillation_weight(calibrate_component_weight, model, method, loader,
                component="distill", target_ratio=task["rho"], num_batches=args.gflow_calibration_batches,
                seed=args.gflow_calibration_seed, device=device)

        calibration = _calibration(run_dir, calibrate_ce)
        method.distill_weight = calibration["weight"]
        return method, reward, {**method.cfg.to_dict(), **ce.shared.sampling_metadata(args),
                                "rho": task["rho"], "power": task["power"], "calibration": calibration}

    if task["arm"] != "gflow":
        raise ValueError(f"Unknown refit arm {task['arm']}")
    teacher = None
    if task["sampler"] == "teacher":
        teacher = gpt.tiger_runner.build_sasrec_teacher_model(dm, config["method"]["sasrec"],
                                                              config_path=run_dir / "resolved_config.json")
        teacher.to(device).eval().requires_grad_(False)
    steps = gpt.curriculum_steps_for_run(args, args.batch_size, len(dm.train_dataset))
    sampler = gpt.build_sampler(task["sampler"], model=model, dm=dm, sasrec=teacher, num_negatives=task["num_negatives"],
                                curriculum_steps=steps, args=args)
    gflow_cfg = gpt.GFlowConfig.from_dict({**resolved["gflow"], "gflow_weight": 1.,
        "reward_cache_path": str(cache_dirs["sasrec"]), "easy_negative_steps": sampler.cfg.curriculum_steps})
    expected_sampler = "sasrec" if task["sampler"] == "teacher" else task["sampler"]
    if (gflow_cfg.objective, gflow_cfg.candidate_sampler, gflow_cfg.num_negative_samples) != (
            task["objective"], expected_sampler, task["num_negatives"]):
        raise ValueError("Source GFlow trainer and completed result disagree on the objective or sampled candidates")
    base = gpt if is_gpt else tiger

    def calibrate_gflow():
        return base.calibrate_gflow_weight(model=model, base_gflow_cfg=gflow_cfg, sampler=sampler, sft_cfg=sft.cfg,
                                           cache=reward, dm=dm, subset_indices=None, device=device, args=args,
                                           tag=task["run_key"], target_ratio=task["rho"], batch_size=args.batch_size)

    calibration = _calibration(run_dir, calibrate_gflow)
    gflow_cfg = replace(gflow_cfg, gflow_weight=float(calibration["fixed_gflow_weight"]))
    if is_gpt:
        method = gpt.make_method(gflow_cfg, sampler, sft.cfg, reward)
    else:
        method = tiger.GFlowMethod(gflow_cfg, candidate_sampler=sampler, sft_method=sft)
        method._reward_cache = reward
    return method, reward, {**gflow_cfg.to_dict(), "sampler": task["sampler"],
                            "curriculum_epochs": args.curriculum_epochs, "rho": task["rho"],
                            "power": task["power"], "calibration": calibration}


def run(task, workspace, run_dir, *, device="cuda", num_workers=2):
    """Persist the final weights and full test scores of one (source checkpoint, refit seed)."""
    from refit_inputs import load_refit_inputs

    run_dir = Path(run_dir).resolve()
    device = torch.device(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "checkpoints/last.pt"
    resolved = verify_source(task)
    identity = {"version": 1, "run_key": task["run_key"], "source_id": task["source_id"],
                "source_signature": task["source_signature"],
                "epochs": task["epochs"], "seed": task["seed"]}
    with gpt.control.file_lock(run_dir / ".refit-training.lock"):
        identity_path = run_dir / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError(f"Refit source identity or schedule changed at {run_dir}")
        if not identity_path.exists():
            if last.exists() or (run_dir / "result.json").exists() or (run_dir / "calibration.json").exists():
                raise RuntimeError(f"Missing refit identity for existing work at {run_dir}")
            write_json(identity_path, identity)
        result_path = run_dir / "result.json"
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if (record.get("status") != "completed" or record.get("run_key") != task["run_key"] or
                    record.get("epochs_completed") != task["epochs"] or not last.is_file()):
                raise ValueError(f"Inconsistent completed refit at {run_dir}")
            return record
        trainer_cfg, args = _settings(task, resolved, run_dir, device)
        dm, config, init_checkpoint, directories = load_refit_inputs(workspace, task["model"], task["dataset"],
            need_factors=task["arm"] != "sft", num_workers=num_workers, batch_size=args.batch_size,
            eval_batch_size=args.test_batch_size)
        if trainer_cfg.max_epochs < 1 or len(dm.train_dataset) < 1 or len(dm.test_dataset) < 1:
            raise ValueError("Final refit requires a positive epoch count and nonempty train/test splits")
        snapshot = json.loads(Path(dm._ce_prepared_manifest).read_text())
        inputs = {"data_audit": config["refit_audit"], "architecture": config["model"],
                  "training_tensors_sha256": snapshot["artifacts"]["tensors"]["sha256"]}
        input_path = run_dir / "input_identity.json"
        if input_path.is_file():
            if json.loads(input_path.read_text()) != inputs:
                raise RuntimeError(f"Refit data or initialization changed at {run_dir}")
        else:
            if last.exists() or (run_dir / "calibration.json").exists():
                raise RuntimeError(f"Missing input identity for existing refit at {run_dir}")
            write_json(input_path, inputs)
        gpt.runner.seed_everything(args.seed)
        model = _load_student(task["model"], config, dm, init_checkpoint, run_dir / "resolved_config.json").to(device)
        method, cache, method_record = _method(task, args, resolved, config, model, dm, directories, run_dir, device)
        gpt.runner.seed_everything(args.seed)
        loader = gpt.gflow_ablation.indexed_train_loader(dm, None, shuffle=True, batch_size=args.batch_size,
            seed=args.seed, reward_cache=cache if args.reward_prefetch else None, prefetch_factor=args.loader_prefetch_factor,
            tensorized=args.tensorized_train_loader)
        options = {"trainer": trainer_cfg.to_dict(), "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
                   "batch_size": args.batch_size, "test_batch_size": args.test_batch_size,
                   "eval_num_beams": args.eval_num_beams, "test_ks": args.test_ks,
                   "method": method_record, "training_tensors_sha256": inputs["training_tensors_sha256"],
                   "source_checkpoint": task["source_checkpoint"],
                   "initial_checkpoint": str(init_checkpoint), "refit_seed": args.seed}
        write_json(run_dir / "resolved_config.json", {**options, "config": config})
        logger = JsonlExperimentLogger(run_dir / "training_log.jsonl")
        trainer = EpochSeededTrainer.from_optimizer_config(model=model, method=method, cfg=trainer_cfg,
            device=device, lr=args.lr, weight_decay=args.weight_decay, logger=logger, seed=args.seed,
            checkpoint_every_updates=int(resolved.get("checkpoint_every_updates", 100)))
        if last.is_file():
            payload = trainer.load_checkpoint(last)
            trainer.restore_progress(payload)
            del payload
        fit = trainer.fit(loader)
        expected_steps = trainer_cfg.max_epochs * len(loader)
        if fit.state.global_step != expected_steps or fit.state.epoch + 1 != trainer_cfg.max_epochs or not last.is_file():
            raise RuntimeError(f"Incomplete fixed-epoch refit at {run_dir}: {fit.state.global_step}/{expected_steps} batches")
        # Beam search only uses the student. Release the optimizer, reward teacher, and
        # persistent training workers before allocating the full test beam batch.
        del trainer, method, cache, loader
        if task["model"] == "gpt2":
            test = gpt.gpt2_beauty.evaluate_splits(model, dm, args, device, None, splits=("test",))["test"]
        else:
            index = gpt.control.CatalogIndex(model, dm)
            test = gpt.control.evaluate_split(model, dm.test_dataloader(), index, ks=tuple(args.test_ks),
                                              num_beams=args.eval_num_beams)
        if int(test["num_examples"]) != len(dm.test_dataset) or not all(math.isfinite(float(v)) for v in test.values()):
            raise RuntimeError(f"Incomplete or non-finite full-test evaluation at {run_dir}")
        result = {**task, "status": "completed", "test_metrics": test, "epochs_completed": trainer_cfg.max_epochs,
                  "expected_test_examples": len(dm.test_dataset), "train_examples": len(dm.train_dataset),
                  "checkpoint": str(last), "initial_checkpoint": str(init_checkpoint), "data_audit": config["refit_audit"],
                  "training_info": options, "optimizer_steps": fit.state.optimizer_step, "micro_steps": fit.state.global_step}
        write_json(result_path, result)
        return result
