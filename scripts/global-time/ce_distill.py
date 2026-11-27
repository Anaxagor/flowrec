"""Global-time SFT + reward cross-entropy on Amazon Beauty and Instruments."""
from __future__ import annotations

import argparse
from dataclasses import replace
import faulthandler
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/exp7-ce-distillation"))
import ce_distill_gpt2 as ce
import ce_distill_tiger as shared

EXPERIMENT = "ce-distill-q-k10"
PROTOCOL = ROOT / "configs/global-time/exp2-tuned-params/finetune_base_matched.json"


class GlobalProfile:
    """Connect the existing CE trainer to this student's audited global-time data."""
    DATA_AUDIT_FILENAME = "training_layout_audit.json"
    data_audit_key = "training_layout_audit"

    def __init__(self, model_name, dataset, scratch):
        import ce_distill_inputs as inputs
        self.model_name = model_name
        self.dataset = dataset
        self.scratch = scratch
        self.input_root = scratch / "inputs"
        self.inputs = inputs

    def matched_args(self, args, dataset, seed, matched):
        flags = [
            *matched.full_history_defaults(shared.read_json(PROTOCOL)),
            "--datasets", dataset, "--seed", str(seed), "--device", args.device,
            "--num-workers", str(args.num_workers), "--num-candidates", str(args.num_samples),
            "--cache-format", "factorized", "--cache-batch-size", str(args.cache_batch_size),
            "--cache-root", str(self.scratch / "rewards"), "--output-root", str(args.output_root),
            "--component-combos", "sasrec_x_interaction_x_sim_hier", "--composition", "multiply",
            "--weight-levels", "1",
        ]
        if args.smoke:
            flags.append("--smoke")
        b = matched.base.parse_args(flags)
        b.prefix_from_dense = self.model_name == "gpt2"
        return b

    def build_run_config(self, base, dataset, b):
        return self.inputs.build_run_config(base, self.model_name, dataset, b)

    def audit_data(self, base, dm, b):
        return self.inputs.verify_layout(self.model_name, self.dataset, self.input_root, dm)

    def asset_paths_for(self, base, dataset, config, components, component_paths):
        paths = [self.input_root / rel for rel in self.inputs.sources.input_files(dataset, student=self.model_name)]
        paths.append(base.dataset_yaml_path(dataset))
        if self.model_name == "gpt2":
            paths.append(base.EXP1_GPT2_MODEL_YAML)
        return [*paths, *component_paths, *[p.parent / "factors.pt" for p in component_paths]]

    def load_initial_model(self, base, dataset, dm, config, config_path):
        checkpoint = self.inputs.checkpoint_path(self.model_name, dataset, self.input_root)
        if self.model_name == "gpt2":
            model, _ = base.build_gpt2_from_tuned_checkpoint(dataset, dm)
        else:
            model = base.tiger_runner.build_model(config, dm, config_path=config_path)
            base.control.load_tiger_checkpoint(model, checkpoint)
            base.gpt2_beauty.check_semantic_row_table(model, dm)
        return model, checkpoint

    def make_sft(self, base):
        return ce.make_sft(base) if self.model_name == "gpt2" else shared.make_sft(base.gflow_ablation)

    def evaluate_splits(self, base, model, dm, b, device, max_batches):
        if self.model_name == "gpt2":
            return ce.evaluate_splits(base, model, dm, b, device, max_batches)
        previous = dm.cfg
        try:
            dm.cfg = replace(previous, eval_batch_size=b.test_batch_size)
            index = base.control.CatalogIndex(model, dm)
            return base.control.evaluate_splits(model, dm, index, b, max_batches)
        finally:
            dm.cfg = previous

    def build_identity(self, args, entry, config, trainer_config, assets, curriculum_steps):
        identity = shared.build_identity(args, entry, config, trainer_config, assets, curriculum_steps)
        identity.update(model=self.model_name, split_strategy="global_time",
                        init_checkpoint=str(self.inputs.checkpoint_path(self.model_name, self.dataset, self.input_root)))
        return identity

    def result_metadata(self, args, entry, b, config):
        return {"split_strategy": "global_time", "target_selection": "all",
                "training_context": "dense_window" if self.model_name == "gpt2" else "rolling",
                "init_checkpoint": str(self.inputs.checkpoint_path(self.model_name, self.dataset, self.input_root))}


def prepare(profile, args, matched, base):
    dataset = profile.dataset
    destination = args.output_root / dataset / "preparation"
    print(f"[prepare] Acquiring lock in {profile.scratch}", flush=True)
    with base.control.file_lock(profile.scratch / ".prepare.lock"):
        print(f"[prepare] Staging and verifying inputs in {profile.input_root}", flush=True)
        checksums = profile.inputs.stage_inputs(profile.model_name, dataset, profile.input_root)
        b = profile.matched_args(args, dataset, args.seeds[0], matched)
        config = profile.build_run_config(base, dataset, b)
        path = destination / "data_reward_config.json"
        shared.write_json(path, config)
        base.runner.seed_everything(args.seeds[0])
        print("[prepare] Loading the dataset and checking the training layout", flush=True)
        dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=path))
        dm.setup()
        audit = profile.audit_data(base, dm, b)
        shared.write_json(destination / profile.DATA_AUDIT_FILENAME, audit)
        for key in ce.INTERNAL_COMPONENTS.values():
            print(f"[prepare] Checking or building reward cache: {key}", flush=True)
            base.ensure_component_cache(dataset, key, config, destination, len(dm.train_dataset), b,
                                        prepared_data=dm)
        shared.write_json(destination / "source_checksums.json", checksums)
        print(f"[prepare] Finished {profile.model_name}/{dataset}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "missing", "prepare", "train"))
    parser.add_argument("--model", required=True, choices=("gpt2", "tiger"))
    parser.add_argument("--dataset", required=True, choices=("beauty", "instruments"))
    parser.add_argument("--index", type=int, help="Power/rho cell, as printed by list")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/global-time" / EXPERIMENT)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args(argv)
    if cli.stage in ("prepare", "train") and (cli.workspace is None or not cli.workspace.is_absolute()):
        parser.error("prepare/train requires --workspace with an absolute scratch path")
    output = cli.output_root.resolve()
    scratch = (cli.workspace or ROOT) / EXPERIMENT
    if cli.smoke:
        output, scratch = output / "smoke", scratch / "smoke"
    flags = [
        "--datasets", cli.dataset, "--combos", shared.DEFAULT_COMBO,
        "--powers", ".25", "1", "16", "--rho-targets", "1", ".1",
        "--estimator", "sampled", "--sampling", "reward_negatives", "--num-samples", "10",
        "--device", cli.device or ("cuda" if cli.stage == "train" else "cpu"),
        "--num-workers", str(cli.num_workers), "--cache-batch-size", "128",
    ]
    if cli.smoke:
        flags.append("--smoke")
    args = shared.parse_args(flags, protocol_path=PROTOCOL)
    args.output_root = output / cli.model
    grid = shared.build_grid(args)
    if cli.stage == "missing":
        indices = []
        for i, entry in enumerate(grid):
            result = ce.run_dir_for(args, entry) / "result.json"
            if not result.is_file() or shared.read_json(result).get("status") != "completed":
                indices.append(str(i))
        print(",".join(indices))
        return
    if cli.stage == "list":
        for i, entry in enumerate(grid):
            print(f"{i}: {cli.model} {cli.dataset} {shared.combo_key(entry)} "
                  f"power={entry['power']:g} rho={entry['rho']:g} seed={entry['seed']} "
                  f"sampling={args.sampling} draws={args.num_samples} importance_correction=false")
        return
    if cli.stage == "train" and (cli.index is None or not 0 <= cli.index < len(grid)):
        parser.error(f"train requires --index 0..{len(grid)-1}")
    preparing = cli.stage == "prepare"
    if preparing:
        print("[prepare] Loading the training libraries", flush=True)
        faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
    try:
        matched, base = ce.load_runtime()
        if preparing:
            print("[prepare] Libraries loaded. Loading input helpers", flush=True)
        profile = GlobalProfile(cli.model, cli.dataset, scratch / cli.model / cli.dataset)
        if preparing:
            print(f"[prepare] Resolving scratch input path {profile.input_root}", flush=True)
        profile.inputs.configure_inputs(base, cli.model, cli.dataset, profile.input_root)
    finally:
        if preparing:
            faulthandler.cancel_dump_traceback_later()
    if preparing:
        prepare(profile, args, matched, base)
    else:
        entry = grid[cli.index]
        with base.control.file_lock(ce.run_dir_for(args, entry) / "run.lock"):
            ce.run_entry(args, entry, matched, base, profile=profile)


if __name__ == "__main__":
    main()
