from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowrec.training.loggers import to_jsonable
from flowrec.training.trainer import Trainer
from run_tiger import (
    build_data_config,
    build_data_module,
    build_method,
    build_model,
    build_trainer_config,
    data_config_paths,
    load_json,
    # override_generation_num_beams,
    resolve_config_path,
    resolve_path,
    # restore_generation_config,
    seed_everything,
    write_json,
)

LOGGER = logging.getLogger("flowrec.evaluate_tiger_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TIGER checkpoint on the test split.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the original TIGER JSON config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a TIGER checkpoint, e.g. best.pt.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing experiment dir. Uses checkpoints/best.pt and cli_args.json when omitted.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write test_summary.json/md. Defaults to --run-dir when provided.",
    )
    parser.add_argument(
        "--test-generation-num-beams",
        type=int,
        default=None,
        help="Override training.test_generation_num_beams for this evaluation.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit for the test dataloader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_optional_existing_path(args.run_dir)
    config_path = resolve_config_arg(args.config, run_dir=run_dir)
    checkpoint_path = resolve_checkpoint_arg(args.checkpoint, run_dir=run_dir)

    config = load_json(config_path)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_eval_batches is not None:
        config.setdefault("training", {})["max_eval_batches"] = int(args.max_eval_batches)

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    setup_logging()

    LOGGER.info("Config: %s", config_path)
    LOGGER.info("Checkpoint: %s", checkpoint_path)

    data_cfg = build_data_config(config, config_path=config_path)
    data = build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("TIGER evaluation requires data.semantic_id_path.")

    train_loader = data.train_dataloader()
    test_loader = data.test_dataloader()
    model = build_model(config, data)
    method = build_method(config, model)

    eval_dir = resolve_output_dir(args.output_dir, run_dir=run_dir, checkpoint_path=checkpoint_path)
    trainer_cfg = build_trainer_config(config, eval_dir)
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=trainer_cfg,
        device=args.device,
        lr=float(config.get("optimizer", {}).get("lr", 3e-4)),
        weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.0)),
        betas=tuple(config.get("optimizer", {}).get("betas", (0.9, 0.999))),
        eps=float(config.get("optimizer", {}).get("eps", 1e-8)),
        scheduler=None,
        logger=None,
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False, load_scheduler=False)

    test_metrics = trainer.test(test_loader)

    summary = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset": {**data_config_paths(data_cfg), **data.summary()},
        "model": model.config_dict(),
        "trainer_state": {
            "epoch": int(trainer.state.epoch),
            "global_step": int(trainer.state.global_step),
            "best_metric": trainer.state.best_metric,
            "best_epoch": trainer.state.best_epoch,
        },
        "test_metrics": test_metrics,
    }

    output_dir = args.output_dir or run_dir
    if output_dir is not None:
        output_dir = resolve_path(output_dir, base=PROJECT_ROOT)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "test_summary.json", summary)
        write_text_summary(output_dir / "test_summary.md", summary)
        LOGGER.info("Wrote %s", output_dir / "test_summary.json")

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def resolve_optional_existing_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    path = resolve_path(value, base=PROJECT_ROOT)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_config_arg(value: Path | None, *, run_dir: Path | None) -> Path:
    if value is not None:
        return resolve_config_path(value)
    if run_dir is None:
        raise ValueError("Provide either --config or --run-dir.")

    cli_args_path = run_dir / "cli_args.json"
    if cli_args_path.exists():
        cli_args = load_json(cli_args_path)
        config = cli_args.get("config")
        if config is not None:
            return resolve_config_path(config)

    resolved_path = run_dir / "resolved_config.json"
    if resolved_path.exists():
        resolved = load_json(resolved_path)
        config = resolved.get("config_path")
        if config is not None:
            return resolve_config_path(config)

    raise ValueError(f"Could not infer config from {run_dir}; pass --config explicitly.")


def resolve_checkpoint_arg(value: Path | None, *, run_dir: Path | None) -> Path:
    if value is not None:
        return resolve_path(value, base=PROJECT_ROOT)
    if run_dir is None:
        raise ValueError("Provide either --checkpoint or --run-dir.")
    for name in ("best.pt", "last.pt"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No best.pt or last.pt found under {run_dir / checkpoints}")


def resolve_output_dir(value: Path | None, *, run_dir: Path | None, checkpoint_path: Path) -> Path:
    if value is not None:
        return resolve_path(value, base=PROJECT_ROOT)
    if run_dir is not None:
        return run_dir
    parent = checkpoint_path.parent.parent
    return parent if parent.exists() else PROJECT_ROOT / "experiments" / "tiger_eval"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def write_text_summary(path: Path, summary: Mapping[str, Any]) -> None:
    metrics = json.dumps(to_jsonable(summary.get("test_metrics", {})), indent=2, sort_keys=True)
    dataset = summary.get("dataset", {})
    lines = [
        "# TIGER Test Summary",
        "",
        "- Config: {}".format(summary.get("config_path")),
        "- Checkpoint: {}".format(summary.get("checkpoint_path")),
        "- Dataset: {}".format(dataset.get("reviews_path") or dataset.get("interactions_path")),
        "- Semantic IDs: {}".format(dataset.get("semantic_id_path")),
        "",
        "## Test Metrics",
        metrics,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
