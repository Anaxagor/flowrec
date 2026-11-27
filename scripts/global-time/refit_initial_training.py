"""Train a fresh global initial model for the HPO winner's fixed epoch count."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exp7-ce-distillation"))

from flowrec.mmap_scratch import stage_mmap_file
from flowrec.training import TrainerConfig
from flowrec.training.loggers import JsonlExperimentLogger
from scripts import sasrec_config_runner as runner
from scripts.gpt2_final_data import training_lock
from scripts.global_time_tiger_tuning import atomic_json
from resumable_trainer import EpochSeededTrainer


@dataclass
class Recipe:
    """Use the tuner's own builders for the training loss, architecture and dataset."""
    data_class: type
    data_config: object
    make_model: object
    make_method: object
    optimizer: dict
    trainer: TrainerConfig
    generation: object = None


def _gpt2(spec, num_workers, device):
    tune = import_module("scripts.global-time.tuning_gpt2")
    dataset, params = spec["dataset"], spec["params"]
    flags = ["--datasets", dataset, "--device", device, "--num-workers", str(num_workers)]
    if dataset in tune.base.MATCHED_DATASETS:
        flags.append("--match-tiger")
    ctx = tune.build_context(tune.parse_args(flags), tune.base.load_regime())
    group, cfg = tune.base.resolve_data_config(dataset, ctx)
    profile = ctx.spec(dataset)
    regime = ctx.regime
    # test_eval_batch_size is a runner-only setting, distinct from the validation batch.
    yaml = runner.load_yaml(ctx.config_dir / group.data_config_name)
    batching = {**yaml.get("gpt2", {}), **yaml["datasets"][dataset].get("gpt2", {})}
    cfg = replace(cfg, test_batch_size=int(batching.get("test_eval_batch_size", tune.base.run_all.DEFAULT_TEST_EVAL_BATCH_SIZE)))
    return Recipe(group.data_module, cfg,
                  lambda dm: tune.base.build_model(ctx, dm, params, profile, dataset),
                  lambda dm: tune.base.build_method(ctx),
                  dict(lr=params["lr"], weight_decay=params["weight_decay"], betas=regime.betas, eps=regime.eps),
                  TrainerConfig(gradient_accumulation_steps=regime.grad_accum, max_grad_norm=regime.max_grad_norm,
                                use_amp=regime.use_amp, amp_dtype=regime.amp_dtype),
                  tune.base.run_all)


def _tiger(spec, num_workers, device):
    from scripts import global_time_tiger_tuning as tune
    args = tune.build_parser(__doc__).parse_args(["--device", device, "--num-workers", str(num_workers)])
    run = tune.Run(spec["dataset"], args, search={})
    incumbent = run.incumbent
    params = spec["params"]
    # The compact 500M grid exports expanded model dimensions and accumulation.
    # Optuna exports axes (e.g. num_layers, which sets encoder AND decoder depth).
    if "num_encoder_layers" in params:
        cfg = tune.grid.clone_config(incumbent.config)
        expected = set(cfg.hparams())
        if expected != set(params):
            raise ValueError(f"Unexpected expanded TIGER parameters: {set(params) ^ expected}")
        cfg.model.update({k: params[k] for k in expected if k in cfg.model})
        cfg.trainer.update({k: params[k] for k in expected if k in cfg.trainer})
        cfg.batch_size = int(params["batch_size"])
        cfg.grad_accum = int(params["gradient_accumulation_steps"])
    else:
        cfg = tune.grid.Trial(params).config(incumbent.config)
    train_cfg = replace(run.trainer_config(cfg, Path("unused"), final=False),
                        gradient_accumulation_steps=cfg.grad_accum)
    data_cfg = replace(incumbent.data_cfg, batch_size=cfg.batch_size, test_batch_size=incumbent.test_eval_batch_size)
    return Recipe(incumbent.group.data_module, data_cfg, partial(tune.tiger.build_model, cfg.model),
                  lambda dm: run.method(cfg, final=False), runner.optimizer_config(cfg.trainer), train_cfg, tune.tiger)


def _sasrec(spec, num_workers, device):
    dataset, params = spec["dataset"], spec["params"]
    tune = import_module("scripts.global-time.tuning_sasrec")
    rt = tune.Runtime(num_workers=num_workers, device=device)
    base = tune.base
    if spec["model"] == "sasrec_sce":
        sce = import_module("scripts.global-time.tuning_sasrec_sce").base
        group, config = sce.find_group(dataset, config_dir=tune.CONFIG_DIR)
        cfg = replace(group.build_data_config(dataset, config, seed=sce.SEED),
                      num_workers=num_workers, persistent_workers=False)
        sce.validate_data_config(dataset, cfg, replace(sce.Runtime(), required_split_strategy="global_time"))
        accum = sce.TUNING_SPECS[dataset].grad_accum
        def method(dm):
            return sce.build_method(dm, sce.resolve_sce_config(params, bucket_base=sce.sce_bucket_base(dm)))
    else:
        group, cfg = tune.resolve_data_config(dataset, rt)
        method = base.build_method
        accum = tune.TUNING_SPECS[dataset].grad_accum
    return Recipe(group.data_module, cfg, lambda dm: base.build_model(dm, params), method,
                  dict(lr=params["lr"], weight_decay=params["weight_decay"], betas=base.BETAS, eps=base.EPS),
                  TrainerConfig(gradient_accumulation_steps=accum, max_grad_norm=base.MAX_GRAD_NORM,
                                use_amp=rt.use_amp and device.startswith("cuda"), amp_dtype=rt.amp_dtype))


def _recipe(spec, workspace, num_workers, device):
    build = {"gpt2": _gpt2, "tiger": _tiger, "sasrec": _sasrec, "sasrec_sce": _sasrec}[spec["model"]]
    recipe = build(spec, num_workers, str(device))
    cfg = recipe.data_config
    if (cfg.split_strategy != "global_time" or cfg.global_val_timestamp is None
            or cfg.global_test_timestamp is None or not cfg.global_val_timestamp < cfg.global_test_timestamp):
        raise ValueError("Expected pinned, ordered global validation and test timestamps")
    if spec["model"] in ("gpt2", "tiger") and spec["dataset"].startswith("yambda_"):
        if cfg.train_target_policy != "last_n" or cfg.train_last_n != 128:
            raise ValueError("Global Yambda generative refits require the last 128 training targets")
        if spec["model"] == "gpt2" and not cfg.train_dense_match_prefix:
            raise ValueError("Global GPT2 Yambda requires TIGER-matched dense training")
    directory = Path(workspace).resolve() / "global_init_refit"
    fields = ("reviews_path", "metadata_path", "interactions_path", "embeddings_path", "semantic_id_path")
    paths = {name: stage_mmap_file(getattr(cfg, name), directory / "inputs")
             for name in fields if getattr(cfg, name, None) is not None}
    # Only the item membership of metadata affects the split. Avoid caching descriptions.
    if hasattr(cfg, "metadata_ids_only"):
        paths["metadata_ids_only"] = True
    recipe.data_config = replace(cfg, **paths)
    return recipe, directory / spec["model"] / spec["dataset"]


def prepare_initial(spec, workspace, *, num_workers=0):
    from refit_inputs import prepare_native_refit
    recipe, directory = _recipe(spec, workspace, num_workers, "cpu")
    return prepare_native_refit(directory, recipe.data_class, recipe.data_config)


def _test(recipe, model, dm, device):
    kwargs = dict(device=device, item_popularity=runner.training_item_popularity(dm),
                  num_catalog_items=dm.num_items, ks=runner.DEFAULT_RANKING_KS, pad_item_id=dm.cfg.pad_item_id)
    loader = dm.test_dataloader()
    if recipe.generation is None:
        return runner.evaluate_ranking_metrics(model, loader, **kwargs, exclude_history=model.cfg.exclude_history_in_generate)
    decoder = recipe.generation
    return decoder.evaluate_generation_ranking_metrics(
        model, loader, **kwargs, lookup=decoder.SemanticIdItemLookup(dm.semantic_ids, dm.item2id),
        num_beams=20, constrain_valid_sequences=True,
    )


def run_initial(spec, workspace, run_dir, *, device="cuda", num_workers=2):
    """Recover only this refit, including when the previous allocation ended during testing."""
    from refit_inputs import load_native_refit

    device = str(device)
    if runner.resolve_device(device) != device:
        raise RuntimeError(f"Requested device {device} unavailable")
    epochs, seed = int(spec["epochs"]), int(spec["seed"])
    if epochs < 1 or epochs != int(spec["best_epoch"]) + 1:
        raise ValueError("Epoch count must equal the one-based best validation epoch")
    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "result.json"
    last = run_dir / "checkpoints/last.pt"
    # The tuner exports the same score to JSON at full precision and to its YAML
    # header at six decimals. Resume on the parameters, identity and schedule.
    selection = {k: spec[k] for k in ("model", "dataset", "trial", "params", "best_epoch", "epochs", "seed")}
    if "study_name" in spec:
        selection["study_name"] = spec["study_name"]
    with training_lock(run_dir):
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if (any(result.get(k) != v for k, v in selection.items()) or result.get("status") != "completed"
                    or result.get("epochs_completed") != epochs or not last.is_file()):
                raise ValueError(f"Existing result disagrees with the requested refit at {run_dir}")
            return result
        recipe, directory = _recipe(spec, workspace, num_workers, device)
        dm, audit = load_native_refit(directory, recipe.data_config)
        runner.seed_everything(seed)
        model, method = recipe.make_model(dm), recipe.make_method(dm)
        cfg = replace(recipe.trainer, max_epochs=epochs, max_steps=None, max_train_batches=None,
                      max_eval_batches=None, checkpoint_dir=last.parent, progress_bar=False,
                      early_stopping_patience=None, early_stopping_min_epochs=0,
                      run_validation_at_start=False, validate_every_epochs=None, validate_every_steps=None,
                      val_sample_fraction=None, val_sample_resample=False, save_best=False, save_last=True,
                      save_every_epochs=None)
        identity = runner.to_jsonable({
            "protocol": "global_initial_train_validation_v1", **selection, "trainer": cfg.to_dict(),
            "model_config": model.config_dict(), "method": method.cfg.to_dict(), "optimizer": recipe.optimizer,
            "batch_size": dm.cfg.batch_size, "test_batch_size": dm.cfg.test_batch_size or dm.cfg.eval_batch_size,
            "data_audit": audit,
        })
        path = run_dir / "resolved_config.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError(f"Source, data or training settings changed at {run_dir}")
        elif last.exists():
            raise RuntimeError(f"Missing identity for existing refit weights at {run_dir}")
        else:
            atomic_json(path, identity)
        trainer = EpochSeededTrainer.from_optimizer_config(
            model=model, method=method, cfg=cfg, device=device, seed=seed,
            checkpoint_every_updates=100, logger=JsonlExperimentLogger(run_dir / "training_log.jsonl"),
            **recipe.optimizer)
        if last.is_file():
            payload = trainer.load_checkpoint(last, map_location="cpu")
            trainer.restore_progress(payload)
            del payload
        train_loader = dm.train_dataloader()
        fit = trainer.fit(train_loader)
        if (fit.state.epoch + 1 != epochs or fit.state.global_step != epochs * len(train_loader)
                or not last.is_file()):
            raise RuntimeError("Fixed-epoch refit stopped before completing all requested training batches")
        del trainer, train_loader, method
        with torch.inference_mode():
            metrics = _test(recipe, model.eval(), dm, device)
        expected = len(dm.test_dataset)
        if (expected < 1 or metrics["num_examples"] != expected
                or not all(math.isfinite(value) for value in metrics.values())):
            raise RuntimeError(f"Invalid full-test result on {expected} targets")
        result = {**spec, "status": "completed", "epochs_completed": epochs,
                  "split_strategy": "global_time", "training_split": "train+validation",
                  "train_examples": len(dm.train_dataset), "test_metrics": metrics, "checkpoint": str(last),
                  "expected_test_examples": expected, "optimizer_steps": fit.state.optimizer_step,
                  "micro_steps": fit.state.global_step, "data_audit": audit}
        atomic_json(result_path, result)
        return result
