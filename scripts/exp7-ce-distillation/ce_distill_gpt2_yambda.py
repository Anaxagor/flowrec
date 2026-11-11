#!/usr/bin/env python
"""CE distillation of matched GPT-2 Yambda trial 36 (50M) and trial 4 (500M).

50M retains every last-128 training target, as in TIGER CE. 500M samples
--train-random-n targets per user from that pool once (default 15), with
subset seed 42. Sampled50 is available with --train-random-n 50. Both preserve
the selected checkpoints' complete prefix histories, catalog and held-out
examples. --list / --count / --aggregate-only use only the standard library.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import ce_distill_gpt2 as gpt2
import ce_distill_tiger_yambda as yambda
import ce_yambda_data_runtime as prepared_runtime
import ce_yambda_checkpoint_paths as checkpoint_paths

shared = gpt2.shared
ROOT = shared.ROOT
PROTOCOL = shared.PROTOCOL
COMPONENTS = shared.COMPONENTS
DEFAULT_COMBO = shared.DEFAULT_COMBO
COMBOS = ("sasrec", "interaction", "similarity", DEFAULT_COMBO)
DATASETS = ("yambda_50m_listens", "yambda_500m_listens")
DEFAULT_OUTPUT_ROOT = "experiments/exp7-ce-distillation-gpt2-yambda"
DEFAULT_CACHE_ROOT = "artifacts/exp7-ce-distillation-gpt2-yambda"
DATA_AUDIT_FILENAME = "matched_prefix_audit.json"
SOURCE_TRIALS = {DATASETS[0]: 36, DATASETS[1]: 4}
read_json = shared.read_json
write_json = shared.write_json
absolute = shared.absolute
build_grid = shared.build_grid
run_dir_for = shared.run_dir_for
combo_key = shared.combo_key
components_for = shared.components_for
aggregate = shared.aggregate
component_cache_dir = gpt2.component_cache_dir
reward_config = gpt2.reward_config
curriculum_steps_for = shared.curriculum_steps_for


def configure_parser(parser):
    prepared_runtime.configure_parser(parser)
    parser.add_argument("--train-random-n", type=int, default=15,
                        help="500M only: fixed uniform targets/user from the last-128 pool (1..127)")
    parser.add_argument("--train-subset-seed", type=int, default=42,
                        help="500M only: fixed target-subset seed, independent of model seeds")


def parse_args(argv=None):
    args = shared.parse_args(argv, description=__doc__, dataset_choices=DATASETS,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        cache_root_help="Matched GPT-2 factors; nondefault 500M subsets use <root>/sampledN_seedS/<dataset>/reward_factors",
        configure_parser=configure_parser,
        parser_defaults={"combos": list(COMBOS), "powers": [1., .25, 16.],
                         "rho_targets": [1., .1], "num_samples": 3,
                         "sampling": "reward", "curriculum_epochs": 5.,
                         "checkpoint_every_updates": 10})
    if args.estimator != "sampled":
        raise SystemExit("Yambda GPT-2 CE supports --estimator sampled; exact CE is reserved for small catalogs")
    if not 1 <= args.train_random_n < 128:
        raise SystemExit("--train-random-n must be in 1..127; full last-128 fine-tuning is forbidden for Yambda 500M")
    if args.train_subset_seed < 0:
        raise SystemExit("--train-subset-seed must be nonnegative")
    prepared_runtime.validate_args(args)
    args.output_namespaces = {}
    if (args.train_random_n, args.train_subset_seed) != (15, 42):
        namespace = f"sampled{args.train_random_n}_seed{args.train_subset_seed}"
        args.output_namespaces[DATASETS[1]] = namespace
        if DATASETS[1] in args.datasets:
            args.summary_namespace = "yambda500m_" + namespace
    return args


def load_runtime():
    for folder in (ROOT, ROOT / "scripts", ROOT / "scripts/exp2-tuned-params",
                   ROOT / "scripts/exp4-reward-ablations"):
        sys.path.insert(0, str(folder))
    import gflow_finetune_gpt2_yambda as matched
    checkpoint_paths.configure_gpt2(matched.base)
    return matched, matched.base


def matched_args(args, dataset, seed, matched):
    if dataset not in SOURCE_TRIALS:
        raise ValueError(f"Unsupported GPT-2 Yambda dataset {dataset!r}")
    source = (["--yambda50-base", "trial36"] if SOURCE_TRIALS[dataset] == 36 else
              ["--yambda500-base", "trial4", "--yambda500-targets", "last128"])
    argv = ["--datasets", dataset, *source, "--seed", str(seed), "--device", args.device,
            "--num-workers", str(args.num_workers), "--cache-format", "factorized",
            "--cache-batch-size", str(args.cache_batch_size),
            "--cache-verify-batches", str(args.cache_verify_batches),
            "--num-candidates", str(args.num_samples), "--combos", DEFAULT_COMBO,
            "--similarity-modes", "hierarchical"]
    for key in ("batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.smoke:
        argv += ["--smoke"]
    b = matched.parse_args(argv)
    # Override only AFTER the selected-source parser resolves its GFlow
    # last128/sample50 settings. Do not call dataset_args again afterward.
    b.train_last_n = 0
    b.train_random_n = args.train_random_n if dataset == DATASETS[1] else 0
    b.train_subset_seed = args.train_subset_seed if dataset == DATASETS[1] else 42
    b.train_pool_last_n = 128 if dataset == DATASETS[1] else None
    b.ce_target_selection = f"sampled{b.train_random_n}" if dataset == DATASETS[1] else "last128"
    b.prefix_from_dense = False
    b.output_root = args.output_root
    b.cache_root = (args.output_root / "cache" if args.smoke else
                    absolute(args.cache_root or DEFAULT_CACHE_ROOT))
    namespace = args.output_namespaces.get(dataset)
    if namespace:
        b.cache_root = b.cache_root / namespace
    check_subset(b)
    return b


def check_subset(b):
    dataset = b.datasets[0]
    if b.train_last_n != 0:
        raise RuntimeError("GPT-2 Yambda CE does not support an additional per-user tail subset")
    if dataset == DATASETS[1]:
        if not 1 <= b.train_random_n < 128 or b.train_subset_seed < 0 or b.train_pool_last_n != 128:
            raise RuntimeError("GPT-2 Yambda 500M CE requires 1..127 sampled targets/user from the last-128 pool and a nonnegative subset seed")
        if b.ce_target_selection != f"sampled{b.train_random_n}":
            raise RuntimeError("GPT-2 Yambda 500M target selection disagrees with the sampled subset")
    elif b.train_random_n != 0 or b.train_pool_last_n is not None or b.ce_target_selection != "last128":
        raise RuntimeError("GPT-2 Yambda 50M CE retains the complete last-128 layout")


class YambdaRuntime:
    """Bind the selected source to model helpers whose legacy default is exp2."""
    def __init__(self, base, args):
        self._base, self._args = base, args
        self._profiles = {}

    def __getattr__(self, name):
        return getattr(self._base, name)

    def tuned_profile(self, dataset):
        if dataset not in self._profiles:
            self._profiles[dataset] = self._base.tuned_profile(dataset, self._args)
        return self._profiles[dataset]

    def tuned_checkpoint_paths(self, dataset):
        return self._base.tuned_checkpoint_paths(dataset, self._args)

    def build_gpt2_from_tuned_checkpoint(self, dataset, dm):
        return self._base.build_gpt2_from_tuned_checkpoint(dataset, dm, self._args)

    def finetune_indices(self, train_dataset, b):
        check_subset(b)
        if getattr(train_dataset, "_ce_prepared", False):
            return None
        if b.ce_target_selection == "last128":
            # The audited datamodule already contains exactly this pool.
            return None
        return self._base.finetune_indices(train_dataset, b)


def runtime_for(base, b):
    return YambdaRuntime(base, b)


def teacher_translation_path(dataset, b):
    return absolute(b.cache_root) / dataset / "source_assets/teacher_id_translation.json"


def build_run_config(base, dataset, b):
    check_subset(b)
    if dataset != b.datasets[0]:
        raise RuntimeError("Selected GPT-2 dataset differs from its bound target-subset protocol")
    if not isinstance(base, YambdaRuntime):
        base = runtime_for(base, b)
    profile, _ = base.tuned_profile(dataset)
    model_yaml = base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML)
    config = base.build_run_config(dataset, b, base.gpt2_runner.user_ids_count(model_yaml, profile))
    expected = {"train_example_layout": "prefix_target", "train_prefix_from_dense": False,
                "train_dense_match_prefix": False, "train_target_policy": "last_n",
                "train_last_n": 128, "max_history_len": 50}
    for key, value in expected.items():
        if config["dataset"].get(key) != value:
            raise RuntimeError(f"Selected GPT-2 trial requires dataset.{key}={value!r}")
    if not config["dataset"].get("matched_tuning_data_audit"):
        raise RuntimeError("Selected GPT-2 trial requires the frozen training/context/catalog data audit")
    if config.get("initial_model", {}).get("trial") != SOURCE_TRIALS[dataset]:
        raise RuntimeError("CE must initialize from the explicitly selected matched GPT-2 trial")
    config["experiment_name"] = f"ce_distill_gpt2_{dataset}_trial{SOURCE_TRIALS[dataset]}"
    config["pretrained_model"] = profile["model"]
    config["method"]["sasrec"]["item_id_translation_path"] = str(teacher_translation_path(dataset, b))
    config["ce_data_protocol"] = {
        "model": "gpt2", "source_trial": SOURCE_TRIALS[dataset],
        "training_layout": "prefix_target matched to recorded dense_causal_match_prefix",
        "per_user_base_last_n": 128, "max_history_len": 50,
        "target_selection": b.ce_target_selection,
        "subset": base.finetune_subset_spec(b),
        "auxiliary_events": "retained in interaction store",
    }
    return config


def audit_data(base, dm, b):
    if getattr(dm, "_ce_prepared", False):
        return dm._ce_prepared_audit
    dataset = b.datasets[0]
    audit = base.matched_source.audit_prefix_data(
        dm, base.matched_source.load_recorded(ROOT, dataset))
    return {**audit, "target_selection": b.ce_target_selection,
            "subset": base.finetune_subset_spec(b)}


def load_data(args, base, b, config, config_path):
    return prepared_runtime.load_or_build(args, "gpt2", base, b, config, config_path, b.datasets[0])


def trainer_overrides(args, b):
    return {"amp_dtype": "bfloat16", "val_sample_fraction": b.val_sample_fraction,
            "val_sample_seed": 0, "val_sample_resample": False}


def result_metadata(args, entry, b, config):
    return {"source_trial": SOURCE_TRIALS[entry["dataset"]],
            "target_selection": b.ce_target_selection,
            "finetune_subset": config["ce_data_protocol"]["subset"]}


def build_identity(args, entry, config, trainer_config, assets, curriculum_steps):
    identity = shared.build_identity(args, entry, config, trainer_config, assets, curriculum_steps)
    identity.update(model="gpt2", data_protocol="matched dense-causal checkpoint prefixes with frozen target subset",
                    source_trial=SOURCE_TRIALS[entry["dataset"]],
                    ce_data_protocol=config["ce_data_protocol"],
                    reward_storage="factorized float32 reconstruction", cache_context_version=2)
    return identity


def asset_paths_for(base, dataset, config, components, component_paths):
    paths = base.tuned_checkpoint_paths(dataset)
    assets = [paths["checkpoint"], paths["resolved_config"], base.EXP1_GPT2_MODEL_YAML,
              base.dataset_yaml_path(dataset)]
    assets.extend(paths[key] for key in ("sidecar", "data_audit") if paths[key].is_file())
    if "sasrec" in components:
        teacher = base.abspath(base.DATASETS[dataset]["sasrec_checkpoint"])
        assets.extend([teacher, base.abspath(base.DATASETS[dataset]["sasrec_trial"]),
                       teacher.parent / "training_summary.json",
                       Path(config["method"]["sasrec"]["item_id_translation_path"])])
    if "similarity" in components:
        assets.append(base.rqvae_model_path(dataset))
    assets.extend(Path(value) for key, value in config["data"].items()
                  if key.endswith("_path") and value)
    return [*map(Path, assets), *component_paths, *[p.parent / "factors.pt" for p in component_paths]]


def check_component_manifest(base, config, path, name, num_examples, b):
    manifest = gpt2.check_component_manifest(base, config, path, name, num_examples, b)
    if not (path.parent / "factors.pt").is_file():
        raise RuntimeError(f"{path.parent}: reward factors are missing")
    if manifest.get("split") != "train":
        raise RuntimeError(f"{path}: CE requires a training reward cache")
    if num_examples is not None and manifest.get("num_examples") != num_examples:
        raise RuntimeError(f"{path}: cache row count differs from the fixed fine-tuning subset")
    if name == "sasrec" and manifest.get("teacher_id_translation") is not True:
        raise RuntimeError(f"{path}: matched GPT-2 SASRec factors require teacher item ID translation")
    source = yambda._existing_source_config(manifest)
    full = json.loads(json.dumps(config))
    full["method"]["sasrec"] = {**base.sasrec_method_cfg(b.datasets[0]),
        "item_id_translation_path": str(teacher_translation_path(b.datasets[0], b))}
    wanted = base.cache_config(full, gpt2.INTERNAL_COMPONENTS[name])
    for section in ("data", "dataset", "method", "initial_model"):
        if not yambda._same_metadata(source.get(section), wanted.get(section)):
            raise RuntimeError(f"{path}: source {section} differs from the selected GPT-2 trial/data/reward protocol")
    return manifest


def verify_caches(base, dm, indices, caches, model):
    """Audit all row identities in bounded chunks, and actual collated targets."""
    import torch
    count = len(dm.train_dataset) if indices is None else len(indices)
    if count == 0:
        raise RuntimeError("Matched GPT-2 fine-tuning subset is empty")
    catalog = model.item_semantic_ids.detach().cpu()
    reference = caches.get("similarity", next(iter(caches.values())))
    examples = dm.train_dataset.examples
    for name, cache in caches.items():
        if cache.num_examples != count or cache.num_items != len(catalog):
            raise RuntimeError(f"{name}: cache rows/catalog differ from matched GPT-2 data")
        if "item_tokens" in cache.factors and not torch.equal(cache.factors["item_tokens"].cpu(), catalog):
            raise RuntimeError(f"{name}: cache catalog order differs from GPT-2")
        for start in range(0, count, 262144):
            end = min(start + 262144, count)
            rows = examples.iloc[start:end] if indices is None else examples.iloc[indices[start:end]]
            for key, column in (("user_ids", "user_id"), ("target_item_ids", "target_item_id"),
                                ("target_timestamps", "target_timestamp")):
                if key not in cache.factors:
                    raise RuntimeError(f"{name}: cached {key} identity is missing")
                actual = cache.factors[key][start:end].reshape(-1).cpu()
                expected = torch.as_tensor(rows[column].to_numpy(copy=False))
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"{name}: cached {key} differs at rows {start}:{end}")
            for key in ("target_semantic_row_ids", "target_token_ids"):
                if key in reference.factors and key in cache.factors and not torch.equal(
                        reference.factors[key][start:end].cpu(), cache.factors[key][start:end].cpu()):
                    raise RuntimeError(f"{name}: cached {key} differs between reward components")
            if "target_token_ids" in cache.factors:
                target_rows = cache.factors["target_semantic_row_ids"][start:end].long().cpu()
                if (bool(target_rows.lt(0).any() | target_rows.ge(len(catalog)).any()) or
                        not torch.equal(catalog[target_rows], cache.factors["target_token_ids"][start:end].cpu())):
                    raise RuntimeError(f"{name}: cached target tokens disagree with the GPT-2 catalog")
    positions = sorted({*range(min(count, 64)), *range(max(0, count // 2 - 32), min(count, count // 2 + 32)),
                        *range(max(0, count - 64), count)})
    selected = positions if indices is None else [indices[pos] for pos in positions]
    batch = next(iter(base.gflow_ablation.indexed_train_loader(
        dm, selected, shuffle=False, batch_size=len(selected), seed=0, num_workers=0, tensorized=True)))
    mapping = base.tiger_runner.item_id_to_semantic_row(dm)
    expected = mapping[batch["target_item_ids"].reshape(-1).long()].cpu()
    for name, cache in caches.items():
        if not torch.equal(cache.target_rows(torch.tensor(positions)).reshape(-1).long().cpu(), expected):
            raise RuntimeError(f"{name}: cached targets differ from collated GPT-2 prefixes")


def build_caches(args, matched, raw_base):
    for dataset in args.datasets:
        b = matched_args(args, dataset, 42, matched)
        base = runtime_for(raw_base, b)
        config = build_run_config(base, dataset, b)
        needed = ([args.cache_component] if args.cache_component else
                  [name for name in COMPONENTS if any(name in c.split("_x_") for c in args.combos)])
        base.validate_inputs(dataset, b, need_teacher="sasrec" in needed,
                             components=[gpt2.INTERNAL_COMPONENTS[name] for name in needed])
        destination = shared.dataset_output_root(args, dataset) / "cache_build"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "data_reward_config.json"
        write_json(path, config)
        base.runner.seed_everything(42)
        dm = base.tiger_runner.build_data_module(base.tiger_runner.build_data_config(config, config_path=path))
        dm.setup()
        write_json(destination / "matched_prefix_audit.json", audit_data(base, dm, b))
        indices = base.finetune_indices(dm.train_dataset, b)
        count = len(dm.train_dataset) if indices is None else len(indices)
        if args.smoke:
            count = min(count, base._smoke_examples(b))
        if "sasrec" in needed:
            translation = base.matched_teacher_translation(dataset, dm, b)
            if translation != teacher_translation_path(dataset, b):
                raise RuntimeError("SASRec translation path differs from the CE cache protocol")
        for name in needed:
            cache_dir = component_cache_dir(base, dataset, name, b)
            manifest_path = cache_dir / "manifest.json"
            existed = manifest_path.exists()
            if existed:
                check_component_manifest(base, config, manifest_path, name, count, b)
            started = time.perf_counter()
            base.ensure_component_cache(dataset, gpt2.INTERNAL_COMPONENTS[name], config,
                                        destination / name, count, b, prepared_data=dm)
            manifest = check_component_manifest(base, config, manifest_path, name, count, b)
            write_json(destination / name / "timing.json", {
                "model": "gpt2", "dataset": dataset, "source_trial": SOURCE_TRIALS[dataset],
                "component": name, "target_selection": b.ce_target_selection,
                "reused": existed, "elapsed_sec": time.perf_counter() - started,
                "num_examples": manifest["num_examples"], "cache_path": str(cache_dir),
                "device": args.device})


def main(argv=None):
    args = parse_args(argv)
    grid = build_grid(args)
    if args.count:
        print(len(grid))
        return
    if args.list:
        for i, entry in enumerate(grid):
            print(f"{i:4d} {entry['dataset']:20s} trial={SOURCE_TRIALS[entry['dataset']]} "
                  f"targets={'last128' if entry['dataset'] == DATASETS[0] else f'sampled{args.train_random_n}'} "
                  f"combo={combo_key(entry)} power={entry['power']:g} rho={entry['rho']:g} "
                  f"seed={entry['seed']} sampling={args.sampling}")
        print(f"total: {len(grid)} runs")
        return
    if args.aggregate_only:
        aggregate(args)
        return
    if args.index is not None:
        if not 0 <= args.index < len(grid):
            raise SystemExit(f"--index must lie in 0..{len(grid) - 1}")
        grid = [grid[args.index]]
    matched, base = load_runtime()
    if args.prepare_data_only:
        b = matched_args(args, DATASETS[1], 42, matched)
        bound = runtime_for(base, b)
        config = build_run_config(bound, DATASETS[1], b)
        path = prepared_runtime.directory(args, "gpt2") / "source_config.json"
        write_json(path, config)
        prepared_runtime.prepare_once(args, "gpt2", bound, b, config, path, audit_fn=audit_data)
        return
    if args.cache_only:
        build_caches(args, matched, base)
        return
    for entry in grid:
        with base.control.file_lock(run_dir_for(args, entry) / "run.lock"):
            gpt2.run_entry(args, entry, matched, base, profile=sys.modules[__name__])


if __name__ == "__main__":
    main()
