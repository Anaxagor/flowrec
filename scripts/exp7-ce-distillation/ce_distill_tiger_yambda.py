#!/usr/bin/env python
"""TIGER CE distillation on exp2 Yambda 50M/500M, all last-128 targets/user.

Default: seven reward products x four powers x two rho values x two datasets,
seed 42, three IID replacement samples, smooth low/high reward-half transition
over five epochs. The original prefix histories have at most 50 items.

--count / --list / --aggregate-only need only the standard library. --smoke
limits training/cache rows but still constructs the complete Yambda dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import ce_distill_tiger as shared

ROOT = shared.ROOT
PROTOCOL = shared.PROTOCOL
COMPONENTS = shared.COMPONENTS
COMBOS = shared.COMBOS
DEFAULT_COMBO = shared.DEFAULT_COMBO
DATASETS = ("yambda_50m_listens", "yambda_500m_listens")
DEFAULT_OUTPUT_ROOT = "experiments/exp7-ce-distillation-tiger-yambda"
DEFAULT_CACHE_ROOT = "artifacts/exp7-ce-distillation-tiger-yambda"
LEGACY_CACHE_ROOTS = {
    "yambda_50m_listens": "artifacts/exp4-reward_ablation/gflow_yambda50m_matched/reward_factors",
    "yambda_500m_listens": "artifacts/exp4-reward_ablation/gflow_yambda500m_last128/reward_factors",
}
POSITIVE_CRITERION = "event_type == 'listen' and played_ratio_pct >= 50"
OPERATIONAL_DATA_FIELDS = frozenset({
    "batch_size", "eval_batch_size", "test_batch_size", "num_workers", "pin_memory",
    "persistent_workers", "include_raw_in_batch", "keep_raw_interactions", "tensorized_batches",
    "seed", "processed_cache_dir", "rebuild_processed_cache",
})

read_json = shared.read_json
write_json = shared.write_json
absolute = shared.absolute
build_grid = shared.build_grid
run_dir_for = shared.run_dir_for
components_for = shared.components_for
combo_key = shared.combo_key
aggregate = shared.aggregate
summary_dir_for = shared.summary_dir_for
curriculum_steps_for = shared.curriculum_steps_for


def parse_args(argv=None):
    args = shared.parse_args(argv, description=__doc__, dataset_choices=DATASETS,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        cache_root_help="Override factors parent: <root>/<dataset>/last128/reward_factors; otherwise reuse verified full-layout exp4 factors or isolate new factors",
        parser_defaults={"combos": list(COMBOS), "sampling": "curriculum_smooth", "num_samples": 3,
                         "curriculum_epochs": 5.0, "checkpoint_every_updates": 10})
    if args.estimator != "sampled":
        raise SystemExit("The Yambda runner supports --estimator sampled; full-catalog exact CE is reserved for the small datasets")
    return args


def assert_last128_config(config):
    dataset = config["dataset"]
    expected = {"train_example_layout": "prefix_target", "train_target_policy": "last_n",
                "train_last_n": 128, "max_history_len": 50,
                "positive_criterion": POSITIVE_CRITERION,
                "allowed_event_types": None, "min_played_ratio_pct": None}
    for key, value in expected.items():
        if dataset.get(key) != value:
            raise RuntimeError(f"Yambda last-128 protocol requires dataset.{key}={value!r}; got {dataset.get(key)!r}")
    if dataset.get("train_prefix_from_dense", False):
        raise RuntimeError("TIGER Yambda must retain its exp2 prefix contexts, not GPT-2 dense-window prefixes")


def build_run_config(base, b):
    if b.train_last_n != 0 or b.train_random_n != 0:
        raise RuntimeError("Yambda CE must use every target of the base last-128 layout without another subset")
    config = base.build_run_config(b)
    assert_last128_config(config)
    config["ce_data_protocol"] = {
        "model": "tiger", "training_layout": "exp2 prefix_target",
        "per_user_base_last_n": 128, "max_history_len": 50,
        "additional_per_user_last_n": None, "additional_per_user_random_n": None,
        "auxiliary_events": "retained in interaction store",
        "cache_root": str(absolute(b.cache_root)),
    }
    return config


def _same_metadata(recorded, expected, key=""):
    """Allow moving the repository between hosts while keeping asset suffixes."""
    if isinstance(expected, dict):
        return (isinstance(recorded, dict) and recorded.keys() == expected.keys()
                and all(_same_metadata(recorded[name], value, name) for name, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(recorded, list) and len(recorded) == len(expected)
                and all(_same_metadata(a, b, key) for a, b in zip(recorded, expected)))
    if key.endswith("_path") and recorded and expected:
        wanted = Path(expected)
        if str(recorded) == str(wanted):
            return True
        if wanted.is_relative_to(ROOT):
            return str(recorded).endswith("/" + str(wanted.relative_to(ROOT)))
    return recorded == expected


def _existing_source_config(manifest):
    source = manifest.get("config_path")
    if not source:
        raise RuntimeError("cache lacks its source config_path")
    path = Path(source)
    if not path.is_file():
        # Old manifests can point at the repository on the build host.
        marker = "/flowrec/"
        if marker in str(path):
            path = ROOT / str(path).rsplit(marker, 1)[1]
    if not path.is_file():
        raise RuntimeError(f"cache source config is unavailable: {source}")
    return read_json(path)


def check_cache_metadata(base, config, name, cache_dir, b, num_examples=None):
    """Refuse last-8/random-15 factors and mismatched data/reward provenance."""
    from dataclasses import asdict
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file() or not (cache_dir / "factors.pt").is_file():
        raise RuntimeError(f"{cache_dir}: complete reward factors are missing")
    manifest = read_json(manifest_path)
    if manifest.get("split") != "train":
        raise RuntimeError(f"{cache_dir}: expected a training cache")
    summary = manifest.get("data", {})
    if manifest.get("num_catalog_items") != summary.get("num_items") or not summary.get("num_items"):
        raise RuntimeError(f"{cache_dir}: catalog count disagrees with the recorded full dataset")
    if not b.smoke and (not summary.get("num_train_examples")
                        or manifest.get("num_examples") != summary["num_train_examples"]):
        raise RuntimeError(f"{cache_dir}: reward rows do not cover the full recorded last-128 training layout")
    base.check_component_manifest(manifest, cache_dir, name, num_examples, b)
    expected = asdict(base.tiger_runner.build_data_config(config, config_path=ROOT / "configs/ce_yambda.json"))
    expected = json.loads(json.dumps(expected, default=str))
    expected = {key: value for key, value in expected.items() if key not in OPERATIONAL_DATA_FIELDS}
    recorded = {key: value for key, value in manifest.get("data_config", {}).items()
                if key not in OPERATIONAL_DATA_FIELDS}
    if not _same_metadata(recorded, expected):
        raise RuntimeError(f"{cache_dir}: data_config differs from the complete exp2 last-128 layout")
    source = _existing_source_config(manifest)
    wanted = base.cache_config(config, name, b)
    for section in ("data", "dataset", "method"):
        if not _same_metadata(source.get(section), wanted.get(section)):
            raise RuntimeError(f"{cache_dir}: source {section} differs from the current data/reward protocol")
    return manifest


def select_cache_root(args, dataset, b, base):
    if args.smoke:
        return args.output_root / "cache" / dataset / "last128/reward_factors"
    if args.cache_root is not None:
        return absolute(args.cache_root) / dataset / "last128/reward_factors"
    candidate = absolute(LEGACY_CACHE_ROOTS[dataset])
    # All three components decide one dataset-wide root. Selecting an ablation
    # or launching one cache array task must not change the eventual root.
    if all((candidate / name / "manifest.json").is_file() for name in COMPONENTS):
        config = build_run_config(base, b)
        try:
            for name in COMPONENTS:
                check_cache_metadata(base, config, name, candidate / name, b)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print(f"[{dataset}] existing full-layout factors are incompatible; using isolated CE caches: {exc}", file=sys.stderr)
        else:
            return candidate
    return absolute(DEFAULT_CACHE_ROOT) / dataset / "last128/reward_factors"


def matched_args(args, dataset, seed, matched):
    argv = ["--dataset", dataset, "--seed", str(seed), "--device", args.device,
        "--num-workers", str(args.num_workers), "--cache-batch-size", str(args.cache_batch_size),
        "--cache-verify-batches", str(args.cache_verify_batches), "--combos", DEFAULT_COMBO,
        "--train-last-n", "0", "--train-random-n", "0", "--num-candidates", str(args.num_samples)]
    for key in ("batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.smoke:
        argv += ["--smoke"]
    b = matched.parse_args(argv)
    b.output_root = args.output_root / dataset
    b.cache_root = select_cache_root(args, dataset, b, matched.base)
    return b


class YambdaRuntime:
    """Retain the shared TIGER trainer while checking the Yambda cache protocol."""
    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def build_run_config(self, b):
        return build_run_config(self._base, b)

    def finetune_indices(self, train_dataset, b):
        if b.train_last_n != 0 or b.train_random_n != 0:
            raise RuntimeError("Yambda CE requires the entire exp2 last-128 layout")
        # None selects all rows without materializing ten million Python ints
        # or another copy of the tensorized training dataset in each worker.
        return None

    def verify_cache_alignment(self, dm, indices, caches):
        import torch
        count = len(dm.train_dataset) if indices is None else len(indices)
        width = min(count, 256)
        positions = sorted(set([*range(width),
            *range(max(0, count // 2 - width // 2), min(count, count // 2 + (width + 1) // 2)),
            *range(max(0, count - width), count)]))
        selected = positions if indices is None else [indices[pos] for pos in positions]
        loader = self._base.indexed_train_loader(dm, selected, shuffle=False, batch_size=len(selected),
                                                seed=0, num_workers=0, tensorized=True)
        batch = next(iter(loader))
        mapping = self._base.tiger_runner.item_id_to_semantic_row(dm)
        expected_rows = mapping[batch["target_item_ids"].reshape(-1).long()]
        selected_rows = torch.tensor(positions, dtype=torch.long)
        examples = getattr(dm.train_dataset, "examples", None)
        if examples is not None and indices is not None:
            examples = examples.iloc[indices]
        for name, cache in caches.items():
            if cache.num_examples != count:
                raise RuntimeError(f"{name}: cache has {cache.num_examples} rows; full last-128 layout has {count}")
            if not torch.equal(cache.target_rows(selected_rows).long().cpu(), expected_rows.cpu()):
                raise RuntimeError(f"{name}: cached targets disagree with the head/middle/tail of the last-128 layout")
            for key in ("user_ids", "target_item_ids", "target_timestamps", "target_token_ids"):
                if key in cache.factors and key in batch:
                    expected = batch[key].cpu()
                    actual = cache.factors[key][selected_rows].cpu()
                    if not torch.equal(actual.reshape(expected.shape), expected):
                        raise RuntimeError(f"{name}: cached {key} disagree with the head/middle/tail of training")
            # Old caches may have been produced with a different Polars thread
            # count. Audit every available target identity, retaining only a
            # bounded tensor view at a time rather than a list of all examples.
            for factor_key, column in (("user_ids", "user_id"), ("target_item_ids", "target_item_id"),
                                       ("target_timestamps", "target_timestamp")):
                if examples is None or column not in examples or factor_key not in cache.factors:
                    continue
                values = examples[column].to_numpy(copy=False)
                for start in range(0, count, 262144):
                    end = min(count, start + 262144)
                    expected = torch.as_tensor(values[start:end])
                    actual = cache.factors[factor_key][start:end].reshape(-1).cpu()
                    if not torch.equal(actual, expected):
                        raise RuntimeError(f"{name}: cached {factor_key} disagree with training rows {start}:{end}")

    def check_component_manifest(self, manifest, cache_dir, name, num_examples, b):
        return check_cache_metadata(self._base, self.build_run_config(b), name, cache_dir, b, num_examples)

    def ensure_component_cache(self, name, config, destination, b, num_examples):
        cache_dir = self._base.component_cache_dir(name, b)
        if (cache_dir / "manifest.json").exists():
            check_cache_metadata(self._base, config, name, cache_dir, b, num_examples)
        self._base.ensure_component_cache(name, config, destination, b, num_examples)
        return check_cache_metadata(self._base, config, name, cache_dir, b, num_examples)


def load_runtime():
    matched, base = shared.load_runtime()
    return matched, YambdaRuntime(base)


def main(argv=None):
    args = parse_args(argv)
    grid = build_grid(args)
    if args.count:
        print(len(grid))
        return
    if args.list:
        for i, entry in enumerate(grid):
            print(f"{i:4d} {entry['dataset']:20s} combo={combo_key(entry)} "
                  f"power={entry['power']:g} rho={entry['rho']:g} seed={entry['seed']}")
        print(f"total: {len(grid)} runs")
        return
    if args.aggregate_only:
        aggregate(args)
        return
    if args.index is not None:
        if not 0 <= args.index < len(grid):
            raise SystemExit(f"--index must lie in 0..{len(grid)-1}")
        grid = [grid[args.index]]
    if args.smoke:
        print("Yambda --smoke limits training and factor rows, but still constructs the COMPLETE dataset and catalog. "
              "Use the synthetic protocol tests for a small local check.", file=sys.stderr, flush=True)
    matched, base = load_runtime()
    if args.cache_only:
        for dataset in args.datasets:
            b = matched_args(args, dataset, 42, matched)
            config = base.build_run_config(b)
            needed = [name for name in COMPONENTS if any(name in combo.split("_x_") for combo in args.combos)]
            for name in ([args.cache_component] if args.cache_component else needed):
                destination = args.output_root / dataset / "cache_build" / name
                destination.mkdir(parents=True, exist_ok=True)
                path = base.component_cache_dir(name, b)
                existed = (path / "manifest.json").exists()
                started = time.perf_counter()
                manifest = base.ensure_component_cache(name, config, destination, b, None)
                write_json(destination / "timing.json", {"component": name, "dataset": dataset,
                    "reused": existed, "elapsed_sec": time.perf_counter() - started,
                    "num_examples": manifest["num_examples"], "cache_path": str(path), "device": args.device})
        return
    for entry in grid:
        with base.control.file_lock(run_dir_for(args, entry) / "run.lock"):
            shared.run_entry(args, entry, matched, base, matched_args_fn=matched_args)


if __name__ == "__main__":
    main()
