#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tiger as tiger_runner
from flowrec.ft_methods.sft.config import SFTConfig
from flowrec.ft_methods.sft.method import SFTMethod
from flowrec.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a trained TIGER checkpoint on the test split.")
    parser.add_argument("--config", type=Path, required=True, help="Training config used for the run.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write metrics JSON.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path, *, config_path: Path) -> Path:
    if path.is_absolute():
        return path
    return tiger_runner.resolve_experiment_path(path, config_path=config_path)


def build_eval_method(config: dict[str, Any]) -> SFTMethod:
    training = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))
    top_k = int(model_cfg.get("top_k", model_cfg.get("num_return_sequences", 20)))
    generation_topk = tuple(int(k) for k in training.get("generation_topk", [5, 10, min(20, top_k)]))
    num_beams = training.get(
        "test_generation_num_beams",
        training.get("generation_num_beams", model_cfg.get("num_beams", 100)),
    )
    return SFTMethod(
        SFTConfig(
            compute_train_metrics=False,
            compute_eval_metrics=True,
            compute_token_metrics=True,
            compute_item_metrics=False,
            evaluate_generation=True,
            generation_topk=generation_topk,
            generation_num_beams=int(num_beams),
            generation_num_return_sequences=top_k,
            generation_constrain_valid_sequences=bool(
                training.get("generation_constrain_valid_sequences", False)
            ),
            generation_exclude_seen_items=bool(training.get("generation_exclude_seen_items", False)),
        )
    )


def main() -> None:
    args = parse_args()
    config_path = tiger_runner.resolve_config_path(args.config)
    config = tiger_runner.load_json(config_path)

    if args.eval_batch_size is not None:
        config.setdefault("dataloader", {})["validation_batch_size"] = int(args.eval_batch_size)
    if args.num_workers is not None:
        config.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
        config.setdefault("dataloader", {})["persistent_workers"] = int(args.num_workers) > 0
    if args.num_beams is not None:
        config.setdefault("training", {})["test_generation_num_beams"] = int(args.num_beams)

    seed = int(config.get("seed", 42))
    tiger_runner.seed_everything(seed)

    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("TIGER test evaluation requires data.semantic_id_path.")

    model = tiger_runner.build_model(config, data)
    checkpoint_path = resolve_path(args.checkpoint_path, config_path=config_path)
    tiger_runner.load_model_state_dict(model, checkpoint_path, strict=True)

    method = build_eval_method(config)
    trainer = Trainer.from_optimizer_config(
        model=model,
        method=method,
        cfg=TrainerConfig(max_eval_batches=args.max_eval_batches),
        device=args.device,
        lr=0.0,
        weight_decay=0.0,
    )
    metrics = trainer.test(data.test_dataloader())
    result = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "data_summary": data.summary(),
        "metrics": metrics,
    }

    output_json = resolve_path(args.output_json, config_path=config_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
