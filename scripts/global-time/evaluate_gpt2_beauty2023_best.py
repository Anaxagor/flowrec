"""Score the validation-selected GPT-2 Beauty 2023 global checkpoint on its full test split."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowrec.mmap_scratch import stage_mmap_file
from scripts import sasrec_config_runner as runner

gpt2 = import_module("scripts.exp1-standard-params.run_gpt2_sft_all")
data = import_module("scripts.global-time.gpt2_beauty2023_final_data")
SOURCE = ROOT / "checkpoints/exp2/global_split/initial/gpt2/beauty_2023"
KS = (5, 10, 20)
MONITOR = "val/sft/generation_ndcg@10"


def load_source(directory, scratch_dir):
    """Read the selected checkpoint and its original data and model recipes."""
    record = json.loads((directory / "provenance.json").read_text())
    expected = dict(model="gpt2", dataset="beauty_2023", stage="initial", split="global_split",
                    checkpoint=str((SOURCE / "best.pt").relative_to(ROOT)))
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected Beauty 2023 checkpoint provenance in {directory}")
    metadata = {}
    for name in ("best_trial.json", "study_config.json", "data_summary.json"):
        content = (directory / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["metadata_sha256"][name]:
            raise ValueError(f"Checkpoint metadata checksum differs for {name}")
        metadata[name] = json.loads(content)
    trial = metadata["best_trial.json"]
    study = metadata["study_config.json"]
    if (trial["dataset"] != record["dataset"] or trial["best_trial_number"] != record["trial"]
            or trial["metric"] != MONITOR or study["monitor"] != MONITOR):
        raise ValueError("Selected GPT-2 trial disagrees with the archived study or provenance")
    path = stage_mmap_file(directory / "best.pt", scratch_dir.resolve(), expected_sha256=record["sha256"])
    payload = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
    state = payload["trainer_state"]
    if (payload["model_type"] != "gpt2" or payload["method_name"] != "sft"
            or payload["trainer_config"]["monitor"] != MONITOR
            or state["epoch"] != trial["best_epoch"] or state["best_epoch"] != trial["best_epoch"]
            or state["best_metric"] != trial["best_value_val_generation_ndcg10"]):
        raise ValueError("best.pt is not the recorded best validation epoch")
    return record, trial, study, metadata["data_summary.json"], payload


@torch.inference_mode()
def evaluate(dm, source, device):
    record, trial, study, summary, payload = source
    expected = summary["num_test_examples"]
    if (dm.cfg.split_strategy != "global_time" or summary["split_strategy"] != "global_time"
            or dm.cfg.global_val_timestamp != summary["global_val_cutoff"]
            or dm.cfg.global_test_timestamp != summary["global_test_cutoff"]
            or len(dm.test_dataset) != expected or expected <= 0
            or dm.num_items != summary["num_items"]):
        raise ValueError("Prepared test split or catalog differs from the archived trial")
    model = gpt2.build_model(study["regime"]["model_config"], {"model": trial["model"]}, dm)
    weights = payload["model_state_dict"]
    if model.config_dict() != payload["model_config"] or not torch.equal(
            model.item_semantic_ids, weights["item_semantic_ids"]):
        raise ValueError("GPT-2 architecture or semantic catalog differs from best.pt")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    metrics = gpt2.evaluate_generation_ranking_metrics(
        model, dm.test_dataloader(), device=device, ks=KS, num_beams=20, constrain_valid_sequences=True,
        lookup=gpt2.SemanticIdItemLookup(dm.semantic_ids, dm.item2id),
        item_popularity=runner.training_item_popularity(dm), num_catalog_items=dm.num_items,
        pad_item_id=dm.cfg.pad_item_id,
    )
    if metrics["num_examples"] != expected or not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError(f"Invalid test metrics or incomplete evaluation of {expected} examples")
    return dict(model=record["model"], dataset=record["dataset"], trial=record["trial"], seed=record["seed"],
                checkpoint=record["checkpoint"], checkpoint_sha256=record["sha256"],
                best_epoch=trial["best_epoch"], best_epoch_one_based=trial["best_epoch"] + 1,
                hpo_validation_ndcg10=trial["best_value_val_generation_ndcg10"],
                split_strategy="global_time", global_val_cutoff=summary["global_val_cutoff"],
                global_test_cutoff=summary["global_test_cutoff"], expected_test_examples=expected,
                eval_num_beams=20, ranking_ks=list(KS), test_metrics=metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "evaluate"):
        cmd = stages.add_parser(name)
        cmd.add_argument("--source-dir", type=Path, default=SOURCE)
        cmd.add_argument("--prepared-dir", type=Path, required=True)
        cmd.add_argument("--scratch-dir", type=Path, required=True)
    cmd = stages.choices["evaluate"]
    cmd.add_argument("--device", default="cuda")
    cmd.add_argument("--num-workers", type=int, default=0)
    cmd.add_argument("--test-batch-size", type=int, default=128)
    cmd.add_argument("--output", type=Path, help="Defaults to test_metrics.json beside the source checkpoint")
    args = parser.parse_args(argv)
    source = load_source(args.source_dir, args.scratch_dir)
    if args.stage == "prepare":
        data.prepare(args.source_dir, args.prepared_dir, args.scratch_dir)
        return
    if args.num_workers < 0 or args.test_batch_size < 1:
        parser.error("Expected nonnegative loader workers and positive test batch size")
    if runner.resolve_device(args.device) != args.device:
        raise RuntimeError(f"Requested device {args.device} is unavailable")
    output = args.output or args.source_dir / "test_metrics.json"
    protected = {"provenance.json", *source[0]["metadata_sha256"]}
    if output.suffix != ".json" or output.resolve() in {(args.source_dir / n).resolve() for n in protected}:
        parser.error("--output must name a JSON results file separate from the source metadata")
    dm = data.load(args.source_dir, args.prepared_dir, args.scratch_dir,
                   num_workers=args.num_workers, test_batch_size=args.test_batch_size)
    result = evaluate(dm, source, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=output.parent, prefix=output.name + ".", delete=False) as tmp:
        json.dump(result, tmp, indent=2, sort_keys=True, allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(output)
    print(f"Saved {output}\n" + json.dumps(result["test_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
