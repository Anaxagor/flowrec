"""Run SFT/GFlow cells from an already bound and audited SFT-only source."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ce_from_sft as ce


def gflow_entry(cell):
    combo = tuple(cell["combo"]) if not isinstance(cell["combo"], str) else tuple(cell["combo"].split("_x_"))
    if combo not in ((*((name,) for name in ce.COMPONENTS), ce.COMPONENTS)):
        raise ValueError(f"Unsupported exp8 reward: {combo}")
    if cell["objective"] not in ("tb", "db", "subtb"):
        raise ValueError(f"Unsupported GFlow objective: {cell['objective']}")
    if int(cell.get("num_negatives", 3)) != 3:
        raise ValueError("Exp8 GFlow requires three negatives plus the observed target")
    entry = {"dataset": cell["dataset"], "combo": combo, "power": float(cell["power"]),
             "objective": cell["objective"], "rho": float(cell["rho"]),
             "num_negatives": 3, "sampler": "teacher"}
    if cell["model"] == "gpt2":
        entry.update(components=tuple("sim_hier" if name == "similarity" else name for name in combo),
                     weights=(1.0,) * len(combo), composition="multiply",
                     similarity_mode="hierarchical" if "similarity" in combo else None)
    elif cell["model"] != "tiger":
        raise ValueError(f"Unsupported model: {cell['model']}")
    return entry


def trainer_class(args):
    """Adapt legacy calls to atomic checkpoints and correct mid-epoch resume."""
    from resumable_trainer import EpochSeededTrainer

    class SourceStageTrainer(EpochSeededTrainer):
        @classmethod
        def from_optimizer_config(cls, **kwargs):
            kwargs.setdefault("seed", int(args.seed))
            interval = getattr(args, "checkpoint_every_updates", None)
            kwargs.setdefault("checkpoint_every_updates", 100 if interval is None else int(interval))
            return super().from_optimizer_config(**kwargs)

        def load_checkpoint(self, path, *, load_optimizer=True, **kwargs):
            payload = super().load_checkpoint(path, load_optimizer=load_optimizer, **kwargs)
            if load_optimizer:
                self.restore_progress(payload)
            return payload

    return SourceStageTrainer


def resume_gpt2(trainer, last_checkpoint, *, tag):
    """The adapted trainer restores the exact batch offset, without +1 twice."""
    payload = trainer.load_checkpoint(last_checkpoint, load_optimizer=True, load_scheduler=True)
    previous_epoch = int(payload["trainer_state"]["epoch"])
    print(f"[{tag}] resumed source-stage checkpoint at epoch={trainer.state.epoch}, "
          f"microstep={trainer.state.global_step}, updates={trainer.state.optimizer_step}", flush=True)
    return previous_epoch


def checked_tiger_load(load, model, checkpoint):
    """A same-shaped checkpoint must not replace the rebuilt catalog order."""
    import torch
    expected = model.item_semantic_ids.detach().clone()
    result = load(model, checkpoint)
    if not torch.equal(expected, model.item_semantic_ids):
        raise RuntimeError("SFT-only TIGER checkpoint semantic-ID table differs from the source datamodule")
    return result


@contextmanager
def resumable_trainers(session):
    with ExitStack() as stack:
        bound = trainer_class(session.args)
        stack.enter_context(ce.temporary_attributes(session.base, Trainer=bound))
        if session.source["model"] == "gpt2":
            stack.enter_context(ce.temporary_attributes(session.base, resume_trainer=resume_gpt2))
        else:
            original_load = session.base.control.load_tiger_checkpoint
            stack.enter_context(ce.temporary_attributes(session.base.control,
                load_tiger_checkpoint=lambda model, path: checked_tiger_load(original_load, model, path)))
        yield bound


def load_teacher(session, dm, device):
    teacher = session.base.tiger_runner.build_sasrec_teacher_model(
        dm, dict(session.config["method"]["sasrec"]), config_path=session.config_path)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _tiger_sft(session, cell, dm, indices, trainer_type):
    import torch
    from flowrec.training.loggers import CompositeExperimentLogger, JsonlExperimentLogger

    base, b, run_dir = session.base, session.args, Path(session.run_dir)
    started = time.perf_counter()
    device = torch.device(b.device)
    base.tiger_runner.seed_everything(int(b.seed))
    model = base.tiger_runner.build_model(session.config, dm, config_path=session.config_path)
    base.control.load_tiger_checkpoint(model, Path(session.checkpoint))
    model.to(device)
    method = ce.tiger.make_sft(base)
    cfg = replace(ce.tiger.build_trainer_config(ce.ce_args(session), b, run_dir, device),
                  **ce.trainer_overrides(session))
    logger = CompositeExperimentLogger([JsonlExperimentLogger(run_dir / "training_log.jsonl"),
                                       base.base.make_status_logger(f"{cell['dataset']}/sft_from_sft")])
    trainer = trainer_type.from_optimizer_config(model=model, method=method, cfg=cfg, device=device,
        lr=float(b.lr), weight_decay=float(b.weight_decay), logger=logger)
    last_checkpoint = run_dir / "checkpoints/last.pt"
    resumed = None
    if last_checkpoint.exists():
        if not b.resume:
            raise RuntimeError(f"Unfinished SFT continuation at {run_dir}; resume is required")
        payload = trainer.load_checkpoint(last_checkpoint)
        resumed = int(payload["trainer_state"]["epoch"])
    ce.tiger.write_json(run_dir / "resolved_config.json", {
        **cell, "variant": "sft_only", "initial_checkpoint": str(session.checkpoint),
        "from_sft_source": session.source_identity, "config": session.config,
        "trainer": cfg.to_dict(), "optimizer": {"lr": b.lr, "weight_decay": b.weight_decay},
        "batching": {"batch_size": b.batch_size, "gradient_accumulation": b.grad_accum,
                     "effective_batch": b.batch_size * b.grad_accum},
        "num_finetune_examples": len(indices), "resumed_from_epoch": resumed,
        "checkpoint_every_updates": trainer.checkpoint_every_updates,
    })
    loader = base.indexed_train_loader(dm, indices, shuffle=not b.smoke,
        batch_size=int(b.batch_size), seed=int(b.seed),
        prefetch_factor=int(getattr(b, "loader_prefetch_factor", 2)),
        tensorized=bool(getattr(b, "tensorized_train_loader", True)))
    try:
        fit = trainer.fit(loader, dm.val_dataloader())
        ce.tiger.write_json(run_dir / "training_history.json", {"history": fit.state.history})
        best = run_dir / "checkpoints/best.pt"
        if not best.exists():
            raise RuntimeError("SFT continuation did not produce a validation-selected checkpoint")
        trainer.load_checkpoint(best, load_optimizer=False, load_scheduler=False)
        index = base.control.CatalogIndex(model, dm)
        splits = base.control.evaluate_splits(model, dm, index, b, cfg.max_eval_batches)
        result = {**cell, "variant": "sft_only", "status": "completed",
            "from_sft_source": session.source_identity, "initial_checkpoint": str(session.checkpoint),
            "best_epoch": fit.state.best_epoch, "best_val_ndcg5": fit.state.best_metric,
            "epochs_completed": fit.state.epoch + 1, "resumed_from_epoch": resumed,
            "micro_steps": fit.state.global_step, "optimizer_steps": fit.state.optimizer_step,
            "elapsed_sec": time.perf_counter() - started, "best_checkpoint": str(best),
            "run_dir": str(run_dir), "splits": splits}
        ce.tiger.write_json(run_dir / "final_metrics.json", splits)
        ce.tiger.write_json(run_dir / "result.json", result)
        return result
    finally:
        logger.close()


def run_training(session, cell, dm, indices):
    """Run one source-validated cell; errors always propagate to Slurm."""
    import torch

    base, b = session.base, session.args
    if (cell["model"], cell["dataset"], int(cell["seed"])) != (
        session.source["model"], session.source["dataset"], int(b.seed)
    ):
        raise ValueError("Training cell differs from the bound SFT source/runtime")
    if cell["arm"] not in ("sft", "gflow"):
        raise ValueError("run_training only accepts SFT and GFlow arms")
    indices = list(range(len(dm.train_dataset))) if indices is None else indices
    if b.smoke:
        indices = indices[:base._smoke_examples(b)]
    if not len(indices):
        raise ValueError("The selected source's fine-tuning subset is empty")
    run_dir = Path(session.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (cell["arm"] == "gflow" and (run_dir / "checkpoints/last.pt").exists()
            and not (run_dir / "calibration.json").is_file()):
        raise RuntimeError("Cannot resume GFlow without its original calibration.json")
    csv_path = run_dir / "metrics.csv"
    with resumable_trainers(session) as trainer_type:
        if cell["arm"] == "sft":
            if cell["model"] == "tiger":
                return _tiger_sft(session, cell, dm, indices, trainer_type)
            context = base.DatasetContext(dataset=cell["dataset"], run_config=session.config,
                dm=dm, subset_indices=indices, sasrec=None, cache_dirs={})
            return base.run_sft_control(cell["dataset"], context, run_dir, args=b, csv_path=csv_path)
        entry = gflow_entry(cell)
        teacher = load_teacher(session, dm, torch.device(b.device))
        directories = session.cache_dirs()
        if cell["model"] == "gpt2":
            from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache
            # Direct run_single bypasses the legacy build_dataset_context,
            # which normally checks these cache row identities before fit.
            for key in entry["components"]:
                cache = FactorizedRewardCache(directories[key], power=1.0)
                base.factor_runner.verify_cache_alignment(dm, indices, {key: cache},
                    tensorized=bool(b.tensorized_train_loader))
                del cache
            return base.run_single(entry=entry, run_config=session.config, dm=dm,
                subset_indices=indices, sasrec=teacher,
                cache_dirs={key: directories[key] for key in entry["components"]},
                run_dir=run_dir, csv_path=csv_path, args=b)
        from flowrec.ft_methods.gflow.factorized_cache import FactorizedRewardCache
        caches = {name: FactorizedRewardCache(directories[name], power=1.0) for name in entry["combo"]}
        base.verify_cache_alignment(dm, indices, caches, tensorized=bool(b.tensorized_train_loader))
        catalog_model = base.tiger_runner.build_model(session.config, dm, config_path=session.config_path)
        index = base.control.CatalogIndex(catalog_model, dm)
        del catalog_model
        return base.run_single(entry=entry, config=session.config, config_path=session.config_path,
            dm=dm, subset_indices=indices, sasrec=teacher, index=index, caches=caches,
            run_dir=run_dir, csv_path=csv_path, args=b, init_baseline_path=session.baseline_path)
