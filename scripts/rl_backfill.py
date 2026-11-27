"""Training and held-out evaluation shared by the GRPO and SDPO experiment grids."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cell:
    split: str
    model: str
    dataset: str
    reward: str | None = None


def parser_for(algorithm):
    parser = argparse.ArgumentParser(
        description=f"Fill missing {algorithm.upper()} metrics from base checkpoints, one task per cell and seed.",
        epilog="Use --list to view the grid and --index to run one task.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", help="print the task grid without loading assets")
    modes.add_argument("--prepare", action="store_true", help="prepare shared data for the selected model and split on CPU")
    parser.add_argument("--index", type=int, help="select one array element")
    parser.add_argument("--exclude", nargs=3, metavar=("SPLIT", "MODEL", "DATASET"),
                        help="skip this split/model/dataset for every reward and seed, keeping the original indices")
    if algorithm == "grpo":
        parser.add_argument("--all-rewards", action="store_true",
                            help="repeat all four rewards on each incomplete model/dataset/split")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--input-root", type=Path, default=ROOT,
                        help="root containing the project's data, datasets and checkpoints")
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments" / "rl_backfill")
    parser.add_argument("--cache-root", type=Path, help="shared directory for CPU-prepared inputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=10,
                        help="number of initial epochs excluded from early-stopping patience (default: 10)")
    parser.add_argument("--patience", type=int, default=5, help="validations without an improvement in NDCG@5")
    parser.add_argument("--lr", type=float, help="default 3e-5 for Beauty 2023, 1e-5 for Yambda")
    parser.add_argument("--beta", type=float, help="default 1 for Beauty 2023, 0.1 for Yambda")
    parser.add_argument("--device", default="cuda")
    if Path(__file__).with_name("rl_backfill_submission.py").is_file():
        from scripts.rl_backfill_submission import configure_execution
        configure_execution(parser, modes)
    return parser


def main(algorithm, grid, argv=None):
    parser = parser_for(algorithm)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct integers between 0 and 2^32-1")
    if min(args.batch_size, args.eval_batch_size, args.grad_accum, args.epochs, args.patience) < 1 or args.num_workers < 0:
        parser.error("batch sizes, grad-accum, epochs and patience must be positive; num-workers must be nonnegative")
    if args.min_epochs < 0:
        parser.error("--min-epochs must be nonnegative")
    if any(x is not None and (not math.isfinite(x) or x <= 0) for x in (args.beta, args.lr)):
        parser.error("lr and beta must be finite and positive")
    cells = grid(args.all_rewards) if algorithm == "grpo" else grid()
    tasks = [(cell, seed) for cell in cells for seed in args.seeds]
    excluded = tuple(args.exclude or ())
    selected = {i: (cell, seed) for i, (cell, seed) in enumerate(tasks)
                if (cell.split, cell.model, cell.dataset) != excluded}
    if hasattr(args, "handle_execution") and args.handle_execution(parser, algorithm, selected, args):
        return
    index = args.index
    if args.prepare and (index is None or args.cache_root is None):
        parser.error("--prepare needs --cache-root and a task index")
    if args.list or index is None:
        print("index\tsplit\tmodel\tdataset\treward\tseed")
        for i, (cell, seed) in selected.items():
            print(f"{i}\t{cell.split}\t{cell.model}\t{cell.dataset}\t{cell.reward or '-'}\t{seed}")
        return
    if not 0 <= index < len(tasks):
        parser.error(f"index must be in 0..{len(tasks) - 1}")
    if index not in selected:
        parser.error(f"index {index} is excluded by --exclude")
    cell, seed = selected[index]
    if args.prepare:
        from scripts.rl_backfill_inputs import prepare_inputs_cache

        manifest = prepare_inputs_cache(cell.split, cell.model, cell.dataset, input_root=args.input_root.resolve(),
                                        cache_root=args.cache_root.resolve())
        print(manifest, flush=True)
        return
    run(algorithm, cell, seed, args)


def write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=str, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_weights(path, payload):
    import torch

    tmp = path.with_suffix(".partial")
    torch.save(payload, tmp)
    tmp.replace(path)


def run(algorithm, cell, seed, args):
    beauty = cell.dataset == "beauty_2023"
    settings = {
        "algorithm": algorithm, **asdict(cell), "seed": seed, "initialization": "base",
        "validation_baseline": "initial_checkpoint",
        "negative_sampler": "on_policy", "num_negatives": 16, "sft_weight": 0.0,
        "frozen_reference": "initial_checkpoint",
        "lr": args.lr if args.lr is not None else (3e-5 if beauty else 1e-5),
        "beta": args.beta if args.beta is not None else (1.0 if beauty else 0.1),
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.grad_accum, "max_epochs": args.epochs, "patience": args.patience,
        "early_stopping_min_epochs": args.min_epochs,
        "input_root": str(args.input_root.resolve()), "validation_metric": "NDCG@5", "test_beams": 20,
    }
    if cell.reward:
        settings["reward_power"] = 1.0
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    directory = args.output_root / algorithm / cell.split / cell.model / cell.dataset
    if cell.reward:
        directory /= cell.reward.replace("×", "x")
    directory = directory / f"seed_{seed}" / digest
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker is running {directory}") from exc
        if not (directory / "final_summary.json").is_file():
            train_and_evaluate(algorithm, cell, seed, args, settings, directory)
        print(directory / "final_summary.json", flush=True)


def train_and_evaluate(algorithm, cell, seed, args, settings, directory):
    import torch

    from flowrec.ft_methods.rl_backfill import OnPolicyBackfillMethod
    from flowrec.training.trainer import TrainerConfig
    from scripts.rl_backfill_inputs import load_inputs
    from scripts.run_tiger import item_id_to_semantic_row
    from scripts.sasrec_config_runner import seed_everything, training_item_popularity

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select --device cpu for a CPU test.")
    seed_everything(seed)
    inputs = load_inputs(cell.split, cell.model, cell.dataset, batch_size=args.batch_size,
                         eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
                         seed=seed, input_root=args.input_root.resolve(),
                         cache_root=args.cache_root, reward=cell.reward)
    model, data = inputs.model, inputs.data
    reward = None
    if algorithm == "grpo":
        from scripts.rl_backfill_rewards import build_backfill_reward

        reward = build_backfill_reward(cell.reward, dataset=cell.dataset, split=cell.split,
                                       data=data, input_root=args.input_root.resolve())
    method = OnPolicyBackfillMethod(
        model, algorithm=algorithm, item_id_to_semantic_row=item_id_to_semantic_row(data),
        reward_model=reward, beta=settings["beta"],
    )
    resumable = import_module("scripts.exp7-ce-distillation.resumable_trainer")
    trainer = resumable.EpochSeededTrainer.from_optimizer_config(
        model=model, method=method, device=device, lr=settings["lr"], weight_decay=0.01,
        seed=seed, checkpoint_every_updates=1024,
        cfg=TrainerConfig(gradient_accumulation_steps=args.grad_accum, max_grad_norm=0.5,
                          use_amp=device.type == "cuda", progress_bar=False,
                          amp_dtype="bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float16"),
        logger=lambda record: print(json.dumps(record, default=str), flush=True),
    )
    evaluation = import_module(f"scripts.exp1-standard-params.run_{cell.model}_sft_all")
    eval_kwargs = dict(device=device, lookup=evaluation.SemanticIdItemLookup(data.semantic_ids, data.item2id),
                       item_popularity=training_item_popularity(data), num_catalog_items=int(data.num_items),
                       constrain_valid_sequences=True, pad_item_id=data.cfg.pad_item_id)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    if not len(train_loader):
        raise ValueError("Empty training loader")
    source = Path(inputs.checkpoint)
    manifest = {**settings, "checkpoint": str(source), "checkpoint_bytes": source.stat().st_size,
                "checkpoint_mtime_ns": source.stat().st_mtime_ns, "source_config": inputs.config}
    manifest_path = directory / "resolved_config.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest, default=str)):
            raise ValueError(f"Run inputs have changed since {manifest_path} was written")
    else:
        write_json(manifest_path, manifest)

    def validate(stage):
        metrics = evaluation.evaluate_generation_ranking_metrics(
            model, val_loader, ks=(5,), num_beams=5, **eval_kwargs)
        if not math.isfinite(metrics["NDCG@5"]):
            raise ValueError(f"Non-finite validation NDCG@5 for {stage}")
        return metrics

    last_path = directory / "last.pt"

    def save_progress():
        save_weights(last_path, {"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                                 "scaler": trainer._scaler.state_dict(), "step": trainer.state.global_step,
                                 "optimizer_step": trainer.state.optimizer_step, "best": best, "history": history,
                                 "initial_validation": initial_validation})

    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer._scaler.load_state_dict(saved["scaler"])
        trainer.state.global_step = saved["step"]
        trainer.state.optimizer_step = saved["optimizer_step"]
        best, history = saved["best"], saved["history"]
        initial_validation = saved["initial_validation"]
        del saved
    else:
        initial_validation = validate("init")
        best = {"epoch": -1, "score": initial_validation["NDCG@5"], "file": "best_init.pt"}
        history = []
        save_weights(directory / best["file"], model.state_dict())
        save_progress()
        print(f"init, validation NDCG@5 {best['score']:.6f}", flush=True)

    for epoch in range(len(history), args.epochs):
        # Both terms count completed epochs, including on checkpoint resume.
        patience_start = max(args.min_epochs, best["epoch"] + 1)
        if len(history) - patience_start >= args.patience:
            break
        # EpochSeededTrainer restores on-policy RNG and shuffle position at the
        # last saved optimizer step. Commit the epoch's validation result and
        # best checkpoint before discarding this intra-epoch recovery point.
        progress_dir = directory / f"epoch_{epoch}_progress"
        partial = progress_dir / "last.pt"
        trainer.cfg = replace(trainer.cfg, checkpoint_dir=progress_dir)
        if partial.is_file():
            saved = trainer.load_checkpoint(partial, map_location="cpu")
            trainer.restore_progress(saved)
            del saved
        train_metrics, _ = trainer.train_one_epoch(train_loader, epoch=epoch)
        validation = validate(f"epoch {epoch + 1}")
        score = validation["NDCG@5"]
        previous_best = best
        if score > best["score"]:
            best = {"epoch": epoch, "score": score, "file": f"best_epoch_{epoch}.pt"}
            save_weights(directory / best["file"], model.state_dict())
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        save_progress()
        if previous_best != best:
            (directory / previous_best["file"]).unlink(missing_ok=True)
        if partial.exists():
            partial.unlink()
            (progress_dir / "last.pt.json").unlink(missing_ok=True)
            if not any(progress_dir.iterdir()):
                progress_dir.rmdir()
        print(f"epoch {epoch + 1}, validation NDCG@5 {score:.6f}", flush=True)
    best_path = directory / best["file"]
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    test = evaluation.evaluate_generation_ranking_metrics(
        model, data.test_dataloader(), ks=(5, 10, 20), num_beams=20, **eval_kwargs)
    for k in (5, 10, 20):
        test[f"Recall@{k}"] = test[f"HR@{k}"]
    write_json(directory / "final_summary.json", {**settings, "initial_checkpoint": str(source),
               "initial_validation": initial_validation,
               "best_checkpoint": str(best_path), "best_validation": best,
               "epochs_completed": len(history), "history": history, "test_metrics": test})
