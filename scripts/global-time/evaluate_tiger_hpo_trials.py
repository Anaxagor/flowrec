"""Refit the selected global TIGER winner, or inspect its historical HPO weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/exp1-standard-params"))
import run_tiger_sft_all as tiger
from scripts import sasrec_config_runner as runner


@dataclass(frozen=True)
class HistoricalTrial:
    name: str
    group: str
    val_ndcg10: float
    best_epoch: int  # Zero-based in the saved Trainer state
    d_kv: int
    cutoffs: tuple[int, int]
    test_examples: int


# The HPO summaries identify the validation winners and their best epochs.
TRIALS = {
    "beauty": HistoricalTrial("trial_0094", "amazon", 0.019851863614892166, 58,
                              64, (1394150400, 1399939200), 4732),
    "yambda_50m_listens": HistoricalTrial("trial_0011", "yambda", 0.10430165861004745, 143,
                                         32, (25016630, 25520600), 6868),
}
OUTPUT = ROOT / "artifacts/global-time/tiger_hpo_trial_evaluation"
# The October studies used this root before the tuner's default moved to experiments/.
HISTORICAL_HPO_ROOT = ROOT / "checkpoints/global-time/exp2-tuned_params/tiger"


def checkpoint_path(dataset, trial, explicit):
    path = explicit if explicit is not None else HISTORICAL_HPO_ROOT / dataset / "trials" / trial.name / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TIGER checkpoint for {dataset}/{trial.name}: {path}")
    return path.resolve()


def verify_checkpoint(payload, dataset, trial):
    trainer = payload["trainer_config"]
    state = payload["trainer_state"]
    expected_dir = Path("exp2-tuned_params/tiger") / dataset / "trials" / trial.name
    if (payload.get("model_type") != "tiger" or
            Path(trainer["checkpoint_dir"]).parts[-len(expected_dir.parts):] != expected_dir.parts or
            "global-time" not in Path(trainer["checkpoint_dir"]).parts or
            trainer["monitor"] != "val/sft/ndcg@10" or
            state["best_metric"] != trial.val_ndcg10 or
            state["epoch"] != trial.best_epoch or state["best_epoch"] != trial.best_epoch):
        raise ValueError(f"Expected the best HPO snapshot for {expected_dir} "
                         f"(epoch {trial.best_epoch}, val NDCG@10 {trial.val_ndcg10}). "
                         f"Found {trainer['checkpoint_dir']}, epoch {state['epoch']}, "
                         f"best metric {state['best_metric']}.")


def make_data(dataset, trial, workers):
    group = next(g for g in tiger.GROUPS if g.key == trial.group)
    config = runner.load_yaml(ROOT / "configs/global-time/exp2-tuned-params" / group.data_config_name)
    cfg, batch_size = tiger.build_data_config(group, dataset, config, seed=42,
                                               semantic_ids_dir=ROOT / "datasets/semantic_ids/exp2-tuned-params",
                                               user_hash_mod=1)
    if ((cfg.global_val_timestamp, cfg.global_test_timestamp) != trial.cutoffs or
            cfg.split_strategy != "global_time" or cfg.global_history_policy != "rolling" or
            cfg.global_eval_target_mode != "last_per_user" or not cfg.filter_cold_items_after_split or
            cfg.cold_item_train_source != "observed" or cfg.train_example_layout != "prefix_target" or
            cfg.drop_items_missing_semantic_id or cfg.min_history_len != 1 or
            (cfg.min_user_interactions, cfg.min_item_interactions) != (5, 5)):
        raise ValueError(f"Unexpected global-time data configuration for {dataset}")
    if dataset == "beauty" and (cfg.train_target_policy, cfg.max_history_len, cfg.positive_criterion) != (
            "all", 20, "verified_purchase == True"):
        raise ValueError("Beauty requires all verified-purchase prefixes and a 20-event history.")
    if dataset == "yambda_50m_listens" and (cfg.train_target_policy, cfg.train_last_n, cfg.max_history_len) != (
            "last_n", 128, 50):
        raise ValueError("Yambda 50M requires last_128 training targets and a 50-event history.")
    data = group.data_module(replace(cfg, eval_batch_size=batch_size, num_workers=workers,
                                     persistent_workers=workers > 0))
    data.setup()
    if len(data.test_dataset) != trial.test_examples:
        raise ValueError(f"Expected {trial.test_examples} test targets, got {len(data.test_dataset)}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=TRIALS)
    parser.add_argument("--checkpoint", type=Path, help="Path to the original HPO best.pt")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workspace", type=Path, help="Scratch directory for train+validation data")
    parser.add_argument("--original-weights", action="store_true", help="Reproduce the historical HPO snapshot evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true", help="Verify checkpoint identity and exit")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if not (args.check_only or args.original_weights or args.checkpoint):
        import evaluate_global_checkpoints as initial
        from refit_initial_training import prepare_initial

        if args.workspace is None:
            parser.error("Pass --workspace to prepare the data and refit the selected model")
        spec = initial.refit_sources.select_initial(ROOT, "tiger", args.dataset)
        directory = initial.result_directory(args.output_root or initial.OUTPUT, spec)
        prepare_initial(spec, args.workspace)
        initial.fit(spec, args.workspace, directory, device=args.device, num_workers=args.num_workers)
        return
    trial = TRIALS[args.dataset]
    path = checkpoint_path(args.dataset, trial, args.checkpoint)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    verify_checkpoint(payload, args.dataset, trial)
    print(f"Verified {path} ({trial.name}, SHA-256 {digest})", flush=True)
    if args.check_only:
        return
    if runner.resolve_device(args.device) != args.device:
        raise RuntimeError(f"Requested device {args.device} is unavailable.")
    device = torch.device(args.device)
    data = make_data(args.dataset, trial, args.num_workers)
    if set(data.id2item[1:]) - set(data.semantic_ids):
        raise ValueError("Some test catalog items have no semantic ID.")
    model = tiger.build_model({**payload["model_config"], "d_kv": trial.d_kv}, data)
    state = payload["model_state_dict"]
    if not torch.equal(model.item_semantic_ids, state["item_semantic_ids"]):
        raise ValueError("The checkpoint's semantic-ID table differs from the evaluation catalog.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        metrics = tiger.evaluate_generation_ranking_metrics(
            model, data.test_dataloader(), device=device, ks=(5, 10, 20), num_beams=20,
            constrain_valid_sequences=True, lookup=tiger.SemanticIdItemLookup(data.semantic_ids, data.item2id),
            item_popularity=runner.training_item_popularity(data), num_catalog_items=data.num_items,
        )
    if metrics["num_examples"] != trial.test_examples:
        raise ValueError(f"Evaluated {metrics['num_examples']} of {trial.test_examples} targets.")
    result = dict(dataset=args.dataset, trial=trial.name, seed=42, checkpoint=str(path),
                  checkpoint_sha256=digest, hpo_validation_ndcg10=trial.val_ndcg10,
                  split_strategy="global_time", eval_num_beams=20, test_metrics=metrics)
    output = (args.output_root or OUTPUT) / args.dataset / trial.name / "test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as tmp:
        json.dump(result, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
    os.replace(tmp.name, output)
    print(f"Saved full-test metrics to {output}", flush=True)


if __name__ == "__main__":
    main()
