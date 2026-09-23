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
EXP1_SCRIPT_DIR = SCRIPT_DIR / "exp1-standard-params"
for path in (PROJECT_ROOT, SCRIPT_DIR, EXP1_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_tiger as tiger_runner
import run_tiger_sft_all as tiger_sft_runner
from scripts import sasrec_config_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate a trained TIGER checkpoint on the test split."
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Optional compact JSON with evaluation settings.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config used to build the dataset and model.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument(
        "--constrain-valid-sequences",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    eval_data: dict[str, Any] = {}
    if args.eval_config is not None:
        eval_config_path = tiger_runner.resolve_config_path(args.eval_config)
        eval_data = tiger_runner.load_json(eval_config_path)
        args.eval_config = eval_config_path

    aliases = {
        "config": "experiment_config",
        "checkpoint_path": "checkpoint_path",
        "output_json": "output_json",
        "device": "device",
        "eval_batch_size": "eval_batch_size",
        "num_workers": "num_workers",
        "num_beams": "num_beams",
        "ks": "ks",
        "constrain_valid_sequences": "constrain_valid_sequences",
        "max_eval_batches": "max_eval_batches",
    }
    for arg_name, config_name in aliases.items():
        if getattr(args, arg_name) is None and eval_data.get(config_name) is not None:
            value = eval_data[config_name]
            if arg_name in {"config", "checkpoint_path", "output_json"}:
                value = Path(value)
            setattr(args, arg_name, value)

    if args.config is None:
        parser.error("--config is required, directly or as experiment_config in --eval-config")
    if args.checkpoint_path is None:
        parser.error("--checkpoint-path is required")
    if args.ks is None:
        args.ks = [5, 10, 20]
    if not args.ks or any(int(k) <= 0 for k in args.ks):
        parser.error("--ks must contain positive integers")
    args.ks = sorted({int(k) for k in args.ks})
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def resolve_path(path: Path, *, config_path: Path) -> Path:
    if path.is_absolute():
        return path
    return tiger_runner.resolve_experiment_path(path, config_path=config_path)


def main() -> None:
    args = parse_args()
    config_path = tiger_runner.resolve_config_path(args.config)
    config = tiger_runner.load_json(config_path)

    if args.eval_batch_size is not None:
        config.setdefault("dataloader", {})["validation_batch_size"] = int(
            args.eval_batch_size
        )
    if args.num_workers is not None:
        config.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
        config.setdefault("dataloader", {})["persistent_workers"] = int(args.num_workers) > 0

    training = dict(config.get("training", {}))
    num_beams = int(
        args.num_beams
        if args.num_beams is not None
        else training.get("test_generation_num_beams", max(args.ks))
    )
    if num_beams < max(args.ks):
        raise ValueError(
            f"num_beams={num_beams} must be at least max(ks)={max(args.ks)}."
        )
    constrain_valid_sequences = bool(
        args.constrain_valid_sequences
        if args.constrain_valid_sequences is not None
        else training.get("generation_constrain_valid_sequences", True)
    )

    seed = int(config.get("seed", 42))
    tiger_runner.seed_everything(seed)

    data_cfg = tiger_runner.build_data_config(config, config_path=config_path)
    data = tiger_runner.build_data_module(data_cfg)
    data.setup()
    if data.semantic_ids is None or data.sid_len is None:
        raise ValueError("TIGER test evaluation requires data.semantic_id_path.")

    model = tiger_runner.build_model(config, data, config_path=config_path)
    checkpoint_path = resolve_path(args.checkpoint_path, config_path=config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    tiger_runner.load_model_state_dict(model, checkpoint_path, strict=True)
    model.to(torch.device(args.device))

    lookup = tiger_sft_runner.SemanticIdItemLookup(data.semantic_ids, data.item2id)
    metrics = tiger_sft_runner.evaluate_generation_ranking_metrics(
        model,
        data.test_dataloader(),
        device=args.device,
        lookup=lookup,
        item_popularity=sasrec_config_runner.training_item_popularity(data),
        num_catalog_items=int(data.num_items),
        ks=tuple(args.ks),
        num_beams=num_beams,
        constrain_valid_sequences=constrain_valid_sequences,
        max_batches=args.max_eval_batches,
    )
    result = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "split": "test",
        "ks": args.ks,
        "num_beams": num_beams,
        "constrain_valid_sequences": constrain_valid_sequences,
        "data_summary": data.summary(),
        "metrics": metrics,
    }

    output_json = (
        resolve_path(args.output_json, config_path=config_path)
        if args.output_json is not None
        else checkpoint_path.parent.parent / "test_metrics.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"Saved metrics to {output_json}", flush=True)


if __name__ == "__main__":
    main()
