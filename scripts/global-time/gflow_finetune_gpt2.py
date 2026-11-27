"""Matched global-time GPT-2 fine-tuning on Beauty 2014 and Instruments 2018."""
from __future__ import annotations

import argparse
import filecmp
from pathlib import Path
import shutil
import sys

import torch

from gpt2_gflow_inputs import SAMPLERS, configure_inputs, input_files, verify_sources

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/exp4-reward-ablations"))
import gflow_finetune_gpt2_amazon_fullhist as matched

base = matched.base
PROTOCOL = ROOT / "configs/global-time/exp2-tuned-params/finetune_base_matched.json"
EXPERIMENT = "gpt2-gflow-k3"
COMPONENTS = ("sasrec", "interaction", "sim_hier")


def stage_inputs(dataset: str, destination: Path) -> None:
    """Publish complete inputs once. Keep existing factors paired with those inputs."""
    for relative in input_files(dataset):
        source, target = ROOT / relative, destination / relative
        if target.is_file():
            if not filecmp.cmp(source, target, shallow=False):
                raise RuntimeError(f"{source} changed after staging. Prepare a fresh workspace.")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        shutil.copy2(source, partial)
        partial.replace(target)


def settings(cli: argparse.Namespace, scratch: Path, output: Path) -> argparse.Namespace:
    protocol = base.load_json(PROTOCOL)
    shared = protocol["shared_schedule"]
    seed = int(shared["seed_gflow"])
    if cli.stage == "sft":
        seeds = shared["seeds_sft_only"]
        if not 0 <= cli.index < len(seeds):
            raise ValueError(f"SFT-only index must be in 0..{len(seeds) - 1}")
        seed = int(seeds[cli.index])
    flags = [
        *matched.full_history_defaults(protocol),
        "--datasets", cli.dataset,
        "--component-combos", "_x_".join(COMPONENTS),
        "--samplers", *SAMPLERS, "--strict-negatives",
        "--composition", "multiply", "--weight-levels", "1",
        "--powers", "0.25", "1", "16", "--rho-targets", "1", "0.1",
        "--objectives", "tb", "db", "subtb", "--num-candidates", "3",
        "--gflow-weight-mode", "fixed_grad_ratio", "--cache-format", "factorized",
        "--seed", str(seed), "--amp-dtype", "float16", "--device", cli.device,
        "--num-workers", str(cli.num_workers), "--cache-root", str(scratch / "rewards"),
        "--output-root", str(output), "--metrics-csv", str(output / "metrics.csv"),
    ]
    if cli.stage == "prepare":
        flags.extend(["--cache-only", "--cache-batch-size", "128"])
    else:
        flags.append("--require-existing-caches")
    if cli.stage == "sft":
        flags.append("--sft-only")
    if cli.smoke:
        flags.append("--smoke")
    return base.parse_args(flags)


def prepare(cli, args, scratch: Path, output: Path) -> None:
    config_dir = output / cli.dataset / "preparation"
    with base.control.file_lock(scratch / ".prepare.lock"):
        stage_inputs(cli.dataset, scratch / "inputs")
        checksums = verify_sources(cli.dataset, scratch / "inputs")
        base.build_dataset_context(
            cli.dataset, COMPONENTS, args=args, device=torch.device(args.device), output_root=output,
            csv_path=base.metrics_csv_path(args), config_dir=config_dir,
        )
        base.write_json(config_dir / "source_checksums.json", checksums)


def train(cli, args, output: Path, entries) -> None:
    device = torch.device(args.device)
    csv_path = base.metrics_csv_path(args)
    entry = None
    if cli.stage == "gflow":
        if not 0 <= cli.index < len(entries):
            raise ValueError(f"GFlow index must be in 0..{len(entries) - 1}")
        entry = entries[cli.index]
        run_dir = base.run_dir_for(output, entry, args.seed)
    else:
        run_dir = base.sft_control_run_dir(output, cli.dataset, args.seed)

    # A resubmitted array can overlap a previous submission of the same slot.
    with base.control.file_lock(run_dir / ".run.lock"):
        if base.run_completed(run_dir):
            print(f"Already completed: {run_dir}", flush=True)
            return
        ctx = base.build_dataset_context(
            cli.dataset, COMPONENTS if entry is not None else (),
            args=args, device=device, output_root=output, csv_path=csv_path,
            need_teacher=entry is not None and entry["sampler"] == "teacher",
            config_dir=run_dir,
        )
        if entry is None:
            result = base.execute_sft_control(cli.dataset, ctx, run_dir, args=args, csv_path=csv_path)
        else:
            result = base.execute_entry(entry, ctx, run_dir, args=args, csv_path=csv_path)
        if result["status"] != "completed":
            raise RuntimeError(f"{run_dir}: {result.get('error', result['status'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "prepare", "sft", "gflow"))
    parser.add_argument("--dataset", required=True, choices=("beauty", "instruments"))
    parser.add_argument("--workspace", type=Path, help="Allocated /scratch workspace shared by the tasks")
    parser.add_argument("--index", type=int, help="Index within this stage and dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/global-time" / EXPERIMENT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Use a separate smoke directory for short training and evaluation")
    cli = parser.parse_args(argv)
    if cli.stage in ("sft", "gflow") and cli.index is None:
        parser.error("Training requires --index")
    if cli.stage != "list" and (cli.workspace is None or not cli.workspace.is_absolute()):
        parser.error("--workspace must be an absolute workspace path")
    if cli.device is None:
        cli.device = "cpu" if cli.stage in ("list", "prepare") else "cuda"
    scratch = (cli.workspace or ROOT) / EXPERIMENT / cli.dataset
    output = cli.output_root.resolve()
    if cli.smoke:
        scratch, output = scratch / "smoke", output / "smoke"
    base.MATCHED_PROTOCOL = PROTOCOL
    configure_inputs(base, scratch / "inputs")
    args = settings(cli, scratch, output)
    entries = base.build_grid(args)
    if cli.stage == "list":
        for i, entry in enumerate(entries):
            print(f"gflow {i:2d}  {base.entry_tag(entry)} seed={args.seed}")
        for i, seed in enumerate(base.load_json(PROTOCOL)["shared_schedule"]["seeds_sft_only"]):
            print(f"sft   {i:2d}  {cli.dataset}/sft_only seed={seed}")
        return

    base.runner.seed_everything(args.seed)
    if cli.stage == "prepare":
        prepare(cli, args, scratch, output)
    else:
        train(cli, args, output, entries)


if __name__ == "__main__":
    main()
