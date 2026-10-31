"""Source-bound runtime for exp8. Training modules are imported only on demand.

The source manifest, rather than a current experiment's defaults, determines
the data, catalog, context construction, subset, architecture and dropout.
Scoped bindings keep the older experiment entry points unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "scripts", ROOT / "scripts/exp4-reward-ablations",
                  ROOT / "scripts/exp7-ce-distillation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

COMPONENTS = ("sasrec", "interaction", "similarity")
AUXILIARY_COLUMNS = ("target_rating", "target_verified_purchase", "history_ratings", "history_verified_purchase")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomic metadata writes; readers never see a partially written marker."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


@contextmanager
def bind(module, **values):
    missing = object()
    previous = {key: getattr(module, key, missing) for key in values}
    for key, value in values.items():
        setattr(module, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                delattr(module, key)
            else:
                setattr(module, key, value)


def stage_args(source, options, base, seed):
    """Use existing parser for mechanical flags, then replay source settings."""
    if source["model"] == "gpt2":
        args = base.parse_args(["--datasets", source["dataset"], "--device", options.device])
        # Only replay flags that describe inputs, not old output/grid selections.
        for key in ("dataset_profile", "yambda50_base", "yambda500_base", "yambda500_targets",
                    "prefix_from_dense"):
            if key in source["recorded_cli"]:
                setattr(args, key, source["recorded_cli"][key])
        args.cache_format = "factorized"
        args.similarity_modes = ["hierarchical"]
        args.component_combos = None
        args.composition = "multiply"
        args.weight_levels = [1.0]
        args.uniform_micro_batch = True
        args.require_existing_caches = True
    else:
        args = base.parse_args(["--dataset", source["dataset"], "--device", options.device])
        args.control_init_baseline = ""
        args.promote_root = None
    for key, value in source["subset"].items():
        setattr(args, key, value)
    batching, trainer = source["batching"], source["trainer"]
    args.batch_size = int(batching["batch_size"])
    args.grad_accum = int(batching["gradient_accumulation"])
    if options.batch_size is not None:
        effective = args.batch_size * args.grad_accum
        if options.grad_accum is None and effective % options.batch_size:
            raise ValueError("--batch-size must divide the source effective batch, or specify --grad-accum")
        args.batch_size = options.batch_size
        args.grad_accum = effective // options.batch_size
    if options.grad_accum is not None:
        args.grad_accum = options.grad_accum
    args.val_batch_size = int(batching["val_batch_size"])
    args.test_batch_size = int(batching["test_batch_size"])
    for key, field, default in (
        ("max_epochs", "max_epochs", 500), ("patience", "early_stopping_patience", 5),
        ("min_epochs", "early_stopping_min_epochs", 10), ("validate_every", "validate_every_epochs", 1),
        ("max_grad_norm", "max_grad_norm", 0.5), ("amp_dtype", "amp_dtype", "float16"),
        ("val_sample_fraction", "val_sample_fraction", None), ("val_sample_seed", "val_sample_seed", 0),
    ):
        setattr(args, key, trainer.get(field, default))
    # An unchanged source must remain selectable as best in this new stage.
    args.validate_at_start = True
    args.val_sample_resample = False
    args.monitor = "val/sft/ndcg@5"
    if options.max_epochs is not None:
        args.max_epochs = options.max_epochs
    args.lr = 3e-5
    args.weight_decay = float(source["optimizer"]["weight_decay"])
    args.seed, args.device, args.num_workers = seed, options.device, options.num_workers
    args.resume, args.force, args.force_cache, args.smoke = True, False, False, options.smoke
    args.skip_init_baseline, args.skip_sanity_check = False, False
    args.max_train_batches = 8 if options.smoke else None
    args.max_eval_batches = 2 if options.smoke else None
    args.eval_num_beams, args.test_ks = 20, [5, 10, 20]
    args.num_candidates, args.samplers = [3], ["teacher"]
    args.combos = ["sasrec", "interaction", "similarity", "sasrec_x_interaction_x_similarity"]
    args.powers, args.rho_targets, args.objectives = [.25, 1., 16.], [.1, 1.], ["tb", "db", "subtb"]
    args.max_candidate_batch = args.batch_size * 4
    args.curriculum_epochs = 10.0
    args.sasrec_transform, args.sasrec_temperature = "sigmoid", 1.0
    args.sasrec_temperatures, args.interaction_exponents = None, None
    args.gflow_weight_mode = "fixed_grad_ratio"
    args.gflow_calibration_batches = 1 if options.smoke else 5
    args.gflow_calibration_seed = 1234
    args.cache_batch_size = options.cache_batch_size
    args.cache_verify_batches = options.cache_verify_batches
    args.checkpoint_every_updates = options.checkpoint_every_updates
    args.hidden_dtype = args.cache_dtype = "float16"
    args.tensorized_train_loader, args.reward_prefetch = True, True
    if options.smoke:
        args.max_epochs, args.min_epochs, args.validate_every = 1, 0, 1
        args.val_batch_size = args.test_batch_size = min(2, args.batch_size)
    return args


def build_config(source, args, base):
    config = deepcopy(source["data_config"])
    config["experiment_name"] = "exp8_from_sft_" + source["source_id"]
    config.setdefault("data", {}).setdefault("source", "yambda" if source["dataset"].startswith("yambda") else "amazon")
    if source["model"] == "tiger":
        values = config["dataset"]
        if source["dataset"] in ("beauty", "instruments", "beauty_2023"):
            values["use_verified_only"] = False
            values["positive_criterion"] = "verified_purchase == True"
        elif values.get("allowed_event_types") is not None or values.get("min_played_ratio_pct") is not None:
            # Retain auxiliary events for interaction rewards, with exactly the
            # same positive stream. CPU preparation compares ALL SFT examples.
            if values.get("allowed_event_types") != ["listen"] or values.get("min_played_ratio_pct") != 50:
                raise ValueError("Unknown TIGER positive filtering; cannot safely retain auxiliary events")
            if values.get("positive_criterion") is not None:
                raise ValueError("Unexpected additional TIGER positive criterion")
            values.update(allowed_event_types=None, min_played_ratio_pct=None,
                          positive_criterion="event_type == 'listen' and played_ratio_pct >= 50")
        components = [base.component_spec(name, args) for name in COMPONENTS]
        teacher = base.sasrec_method_cfg()
    else:
        components = [base.component_spec(name, source["dataset"], args)
                      for name in ("sasrec", "interaction", "sim_hier")]
        teacher = base.sasrec_method_cfg(source["dataset"])
    config["method"] = {"name": "gflow", "reward_model": "multiply",
                        "reward_epsilon": base.GFLOW_REWARD_EPSILON,
                        "component_keys": ["sasrec", "interaction", "sim_hier"] if source["model"] == "gpt2" else list(COMPONENTS),
                        "reward": {"components": components}, "sasrec": teacher}
    config.setdefault("dataloader", {}).update(
        train_batch_size=args.batch_size, batch_size=args.batch_size,
        validation_batch_size=args.val_batch_size, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, train_drop_last=False,
    )
    return config


@dataclass
class SourceSession:
    source: dict
    options: argparse.Namespace
    base: object
    args: argparse.Namespace
    config: dict
    output_root: Path
    cache_root: Path
    checkpoint: Path
    source_identity: dict
    source_assets: list

    @property
    def config_path(self):
        return self.output_root / "data_reward_config.json"

    @property
    def baseline_path(self):
        return self.output_root / "baseline/init_metrics.json"

    def cache_dirs(self):
        if self.source["model"] == "gpt2":
            return {name: self.base.component_cache_dir(self.source["dataset"], name, self.args)
                    for name in ("sasrec", "interaction", "sim_hier")}
        return {name: self.base.component_cache_dir(name, self.args) for name in COMPONENTS}

    def indices(self, dm, *, full=False):
        values = self._original_indices(dm.train_dataset, self.args)
        if values is None:
            values = list(range(len(dm.train_dataset)))
        return values if full or not self.args.smoke else values[:self.base._smoke_examples(self.args)]

    def audit_data(self, dm):
        actual = dm.summary()
        actual["num_finetune_examples"] = len(self.indices(dm, full=True))
        for key, expected in self.source["expected_counts"].items():
            if actual.get(key) != expected:
                raise RuntimeError(f"{self.source['source_id']}: source data {key}={expected}, rebuilt={actual.get(key)}")
        audit = {"source": self.source_identity, "counts": {key: actual[key] for key in self.source["expected_counts"]},
                 "subset": self.source["subset"], "training_context": actual.get("training_context"),
                 "preprocessing_origin": self.source["preprocessing_origin"]}
        if self.source["model"] == "gpt2":
            if self.source["protocol"] == "dense_window_v1":
                if actual.get("training_context") != self.source["input_audit"]:
                    raise RuntimeError("GPT-2 source dense-window context changed")
                audit["dense_prefix_audit"] = dm.dense_prefix_audit
            else:
                audit["matched_prefix_audit"] = self.base.matched_source.audit_prefix_data(dm, {
                    "dataset": self.source["dataset"], "matched_data_audit": self.source["input_audit"]})
        return audit

    def setup_data(self, *, allow_build=False):
        from prepared_data import load_prepared_data
        dm = load_prepared_data(self)
        if dm is None:
            if not allow_build:
                raise RuntimeError("Prepared data are missing; run CPU preparation before GPU tasks")
            dm = self.base.tiger_runner.build_data_module(self.base.tiger_runner.build_data_config(
                deepcopy(self.config), config_path=self.config_path))
            dm.setup()
        self.audit_data(dm)
        return dm


def assert_source_manifest(session):
    """Never silently reuse a baseline or factors after source replacement."""
    for root in (session.output_root, session.cache_root):
        path = root / "source_identity.json"
        with session.base.control.file_lock(path.with_suffix(".lock")):
            if path.exists() and read_json(path) != session.source_identity:
                raise RuntimeError(f"Source changed at {root}; select a new output/cache root")
            write_json(path, session.source_identity)


@contextmanager
def open_source(source, options, seed):
    import torch
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for local smoke checks")
    if source["fingerprint_kind"] != "checkpoint_and_metadata":
        raise ValueError("Training/preparation requires a checkpoint content hash")
    if source["model"] == "gpt2":
        import gflow_finetune_gpt2_amazon as base
    else:
        import gflow_finetune_yambda500m as base
    with ExitStack() as stack:
        if source["model"] == "tiger":
            active = {key: getattr(base, key) for key in base.DATASETS[source["dataset"]]}
            active["DATASET"] = base.DATASET
            stack.enter_context(bind(base, **active))
        args = stage_args(source, options, base, seed)
        output = options.output_root / source["source_id"] / f"sft_seed_{source['source_seed']}"
        cache = options.cache_root / source["source_id"] / f"sft_seed_{source['source_seed']}"
        args.output_root = output
        args.cache_root = cache if source["model"] == "gpt2" else cache / "reward_factors"
        identity = {key: source[key] for key in ("source_id", "model", "dataset", "source_seed",
                    "source_fingerprint", "checkpoint_sha256", "protocol")}
        config = build_config(source, args, base)
        config["from_sft_source"] = identity
        assets = list(dict.fromkeys(Path(row["path"]) for row in source["metadata_files"]))
        session = SourceSession(source, options, base, args, config, output, cache,
                                Path(source["checkpoint"]), identity, assets)
        session._original_indices = base.finetune_indices
        def bounded_indices(dataset, b):
            values = session._original_indices(dataset, b)
            if values is None:
                values = list(range(len(dataset)))
            return values[:base._smoke_examples(b)] if b.smoke else values
        stack.enter_context(bind(base, finetune_indices=bounded_indices))
        if source["model"] == "gpt2":
            paths = {"checkpoint": session.checkpoint, "sidecar": Path(str(session.checkpoint) + ".json"),
                     "best_trial": Path(source["run_dir"]) / "resolved_config.json"}
            stack.enter_context(bind(base,
                tuned_checkpoint_paths=lambda *_a, **_k: dict(paths),
                tuned_profile=lambda *_a, **_k: (deepcopy(source["model_profile"]), deepcopy(source["model_config"])),
                build_run_config=lambda *_a, **_k: deepcopy(session.config)))
        else:
            stack.enter_context(bind(base, TIGER_CHECKPOINT=str(session.checkpoint),
                                     build_run_config=lambda *_a, **_k: deepcopy(session.config)))
        assert_source_manifest(session)
        # The translation is built once during GPU preparation. All later
        # stages consume the same explicit raw-item mapping.
        translation = cache / source["dataset"] / "source_assets/teacher_id_translation.json"
        if source["model"] == "gpt2" and source["dataset"].startswith("yambda") and translation.is_file():
            session.config["method"]["sasrec"]["item_id_translation_path"] = str(translation)
        yield session


def input_identity(session):
    """Inputs and data-affecting settings used by both preparation phases."""
    from ce_distill_tiger import fingerprint
    paths = [Path(v) for k, v in session.config["data"].items() if k.endswith("_path") and v]
    # Include reward model assets; exclude the generated teacher translation,
    # whose content is tracked by the sasrec factor manifest.
    paths += [Path(session.config["method"]["sasrec"]["checkpoint_path"])]
    for spec in session.config["method"]["reward"]["components"]:
        paths += [Path(v) for k, v in spec["reward"].items() if k.endswith("_path") and v]
    method = deepcopy(session.config["method"])
    method["sasrec"].pop("item_id_translation_path", None)
    return json_value({"source": session.source_identity, "data": session.config["data"],
                       "dataset": session.config["dataset"], "model": session.config["model"],
                       "reward_method": method,
                       "subset": session.source["subset"], "assets": fingerprint(paths),
                       "smoke_examples": session.base._smoke_examples(session.args) if session.args.smoke else None})


def compare_tiger_source(session, dm):
    """Verify every supervised row when retaining auxiliary reward events."""
    reference = deepcopy(session.source["data_config"])
    # Equal rows alone do not guarantee equal model inputs: padding, history
    # truncation and user tokens are applied later by the collator. Only the
    # positive-event partition may differ from the recorded source protocol.
    for key in ("data", "model", "seed"):
        expected, actual = deepcopy(reference.get(key)), deepcopy(session.config.get(key))
        if key == "data":
            default_source = "yambda" if session.source["dataset"].startswith("yambda") else "amazon"
            expected.setdefault("source", default_source)
            actual.setdefault("source", default_source)
        if expected != actual:
            raise RuntimeError(f"TIGER source {key} configuration changed")
    filter_keys = {"use_verified_only", "positive_criterion", "allowed_event_types", "min_played_ratio_pct"}
    expected = {key: value for key, value in reference["dataset"].items() if key not in filter_keys}
    actual = {key: value for key, value in session.config["dataset"].items() if key not in filter_keys}
    if expected != actual:
        raise RuntimeError("TIGER source dataset/collator configuration changed")
    if reference["dataset"] == session.config["dataset"]:
        return {"comparison": "identical source dataset configuration"}
    import pandas as pd
    original = session.base.tiger_runner.build_data_module(session.base.tiger_runner.build_data_config(
        reference, config_path=session.config_path))
    original.setup()
    for name in ("item2id", "user2id", "semantic_ids"):
        if getattr(original, name) != getattr(dm, name):
            raise RuntimeError(f"Auxiliary-event retention changed source {name}")
    for name in ("train_dataset", "val_dataset", "test_dataset"):
        pd.testing.assert_frame_equal(
            getattr(original, name).examples.drop(columns=list(AUXILIARY_COLUMNS), errors="ignore"),
            getattr(dm, name).examples.drop(columns=list(AUXILIARY_COLUMNS), errors="ignore"),
            check_exact=True)
    return {"comparison": "all supervised rows, histories, IDs and semantic IDs equal",
            "auxiliary_columns_not_consumed_by_tiger": list(AUXILIARY_COLUMNS)}


def build_factor(session, dm, key):
    """Use the existing factor builder in-process, avoiding a second full DM."""
    import precompute_reward_factors as factors
    base, args = session.base, session.args
    directory = session.cache_dirs()[key]
    config = (base.cache_config(session.config, key) if session.source["model"] == "gpt2"
              else base.cache_config(session.config, key, args))
    config_path = session.cache_root / f"cache_config_{key}.json"
    manifest_path = directory / "manifest.json"
    with base.control.file_lock(directory.parent / f"{key}.build.lock"):
        if manifest_path.exists():
            check_factor(session, key, len(session.indices(dm)))
            return
        write_json(config_path, config)
        argv = ["--config", str(config_path), "--output-dir", str(directory), "--split", "train",
                "--batch-size", str(args.cache_batch_size), "--num-workers", str(args.num_workers),
                "--device", args.device, "--hidden-dtype", args.hidden_dtype,
                "--verify-batches", str(args.cache_verify_batches)]
        for field, option in (("train_last_n", "--per-user-last-n"),
                              ("train_random_n", "--per-user-random-n"),
                              ("train_pool_last_n", "--per-user-pool-last-n")):
            value = getattr(args, field, None)
            if value:
                argv.extend([option, str(value)])
        argv.extend(["--per-user-random-seed", str(args.train_subset_seed)])
        if args.smoke:
            argv.extend(["--max-examples", str(base._smoke_examples(args))])
        write_json(directory / "exp8_input_identity.json", input_identity(session))
        factors.main(argv, prepared_data=dm)
        check_factor(session, key, len(session.indices(dm)))


def check_factor(session, key, expected):
    directory = session.cache_dirs()[key]
    manifest = read_json(directory / "manifest.json")
    if not (directory / "factors.pt").is_file():
        raise FileNotFoundError(directory / "factors.pt")
    if read_json(directory / "exp8_input_identity.json") != input_identity(session):
        raise RuntimeError(f"Source/data/reward assets changed at {directory}")
    name = "similarity" if key == "sim_hier" else key
    if session.source["model"] == "gpt2":
        session.base.factor_runner.check_component_manifest(manifest, directory, name, expected,
                                                            session.args, similarity_mode="hierarchical" if name == "similarity" else None)
    else:
        session.base.check_component_manifest(manifest, directory, key, expected, session.args)
    if manifest["num_examples"] != expected:
        raise RuntimeError(f"{directory}: expected {expected} factor rows, found {manifest['num_examples']}")


def teacher_translation_identity(session):
    if session.source["model"] != "gpt2" or not session.source["dataset"].startswith("yambda"):
        return None
    path = session.cache_root / session.source["dataset"] / "source_assets/teacher_id_translation.json"
    if not path.is_file():
        raise RuntimeError("Missing SASRec ID translation; run CPU preparation in a fresh cache root")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(session, kind):
    from prepared_data import save_prepared_data, snapshot_ready
    if kind == "all":
        # Only the CPU factor stage switches devices. No GPU computation is
        # needed for interaction/similarity, including on a local GPU host.
        device = session.args.device
        session.args.device = "cpu"
        try:
            prepare(session, "cpu")
        finally:
            session.args.device = device
        prepare(session, "gpu")
        return
    identity = input_identity(session)
    marker = session.cache_root / f"prepared_{kind}.json"
    if kind == "gpu":
        require_prepared(session, kinds=("cpu",))
    if marker.exists():
        previous = read_json(marker)
        if previous.get("identity") != identity or previous.get("status") != "completed":
            raise RuntimeError(f"Preparation identity changed: {marker}")
        if previous.get("teacher_translation") != teacher_translation_identity(session):
            raise RuntimeError(f"Prepared SASRec ID translation changed: {marker}")
        names = ("interaction", "sim_hier" if session.source["model"] == "gpt2" else "similarity") if kind == "cpu" else ("sasrec",)
        count = session.base._smoke_examples(session.args) if session.args.smoke else session.source["expected_counts"]["num_finetune_examples"]
        for key in names:
            check_factor(session, key, count)
        if snapshot_ready(session) and (kind == "cpu" or baseline_ready(session)):
            print(f"Already prepared: {marker}", flush=True)
            return
    dm = session.setup_data(allow_build=kind == "cpu")
    audit = session.audit_data(dm)
    if kind == "cpu":
        if session.source["model"] == "tiger":
            audit["supervised_equivalence"] = compare_tiger_source(session, dm)
        save_prepared_data(session, dm)
        if session.source["model"] == "gpt2" and session.source["dataset"].startswith("yambda"):
            # Building the teacher's own raw-ID catalog is CPU preprocessing,
            # too. Never trigger its first build on an allocated GPU.
            path = session.base.matched_teacher_translation(session.source["dataset"], dm, session.args)
            session.config["method"]["sasrec"]["item_id_translation_path"] = str(path)
        keys = ("interaction", "sim_hier" if session.source["model"] == "gpt2" else "similarity")
    else:
        keys = ("sasrec",)
    for key in keys:
        build_factor(session, dm, key)
    if kind == "gpu":
        from ce_from_sft import prepare_ce
        baseline = prepare_ce(session, dm=dm)
        write_json(session.baseline_path, {"checkpoint": str(session.checkpoint),
                   "from_sft_source": session.source_identity, **baseline["splits"]})
        write_json(session.config_path, session.config)
    write_json(marker, {"identity": identity, "audit": audit, "status": "completed",
                        "teacher_translation": teacher_translation_identity(session)})
    print(f"Prepared {kind}: {session.source['source_id']}", flush=True)


def require_prepared(session, kinds=("cpu", "gpu")):
    from prepared_data import snapshot_ready
    wanted = input_identity(session)
    for kind in kinds:
        marker = session.cache_root / f"prepared_{kind}.json"
        record = read_json(marker) if marker.is_file() else {}
        if record.get("identity") != wanted or record.get("status") != "completed":
            raise RuntimeError(f"Missing or incompatible {kind} preparation: {marker}; run preparation first")
        if record.get("teacher_translation") != teacher_translation_identity(session):
            raise RuntimeError(f"SASRec ID translation changed since CPU preparation: {marker}")
    if not snapshot_ready(session):
        raise RuntimeError("Missing prepared data snapshot; run CPU preparation first")
    if "gpu" in kinds:
        if not baseline_ready(session):
            raise RuntimeError("Missing SFT-only baseline for this exact source")


def baseline_ready(session):
    from ce_from_sft import baseline_identity, ce_args, source_config
    path = session.output_root / "ce" / session.source["dataset"] / "baseline" / (
        "smoke.json" if session.args.smoke else "init_metrics.json")
    if not session.baseline_path.is_file() or not path.is_file():
        return False
    record, ce_record = read_json(session.baseline_path), read_json(path)
    return (record.get("from_sft_source") == session.source_identity
            and all(isinstance(record.get(split), dict) and record[split] for split in ("validation", "test"))
            and ce_record.get("identity") == baseline_identity(session, source_config(session), ce_args(session))
            and all(ce_record.get("splits", {}).get(split) == record[split] for split in ("validation", "test")))


def training_identity(session, cell):
    # Slurm array size, worker count and absolute cache/output paths are not
    # optimization settings. Filtering a resubmission must preserve identity.
    fields = ("lr", "weight_decay", "batch_size", "grad_accum", "max_epochs", "patience",
              "min_epochs", "validate_every", "validate_at_start", "max_grad_norm", "amp_dtype",
              "val_sample_fraction", "val_sample_seed", "val_sample_resample", "monitor",
              "val_batch_size", "test_batch_size", "eval_num_beams", "test_ks",
              "max_train_batches", "max_eval_batches", "smoke", "curriculum_epochs", "curriculum_steps",
              "sampler_temperature", "sasrec_transform", "sasrec_temperature", "max_candidate_batch",
              "gflow_weight_mode", "gflow_calibration_batches", "gflow_calibration_seed", "gflow_calibration_eps")
    return json_value({"cell": cell, "source": session.source_identity,
                       "inputs": input_identity(session),
                       "stage_args": {key: getattr(session.args, key) for key in fields}})


def run_cell(session, cell):
    from finetune_from_sft import cell_key
    require_prepared(session)
    key = cell_key(cell)
    session.run_dir = session.output_root / cell["arm"] / key / f"seed_{cell['seed']}"
    job_dir = session.output_root / "jobs" / key
    identity = training_identity(session, cell)
    with session.base.control.file_lock(job_dir / "run.lock"):
        path = job_dir / "identity.json"
        if path.exists() and read_json(path) != identity:
            raise RuntimeError(f"Stage settings changed for {job_dir}; use a new output root")
        write_json(path, identity)
        completed = job_dir / "result.json"
        if completed.exists() and read_json(completed).get("status") == "completed":
            print(f"Already completed: {completed}", flush=True)
            return read_json(completed)
        count = session.base._smoke_examples(session.args) if session.args.smoke else session.source["expected_counts"]["num_finetune_examples"]
        if cell["arm"] != "sft":
            for component in cell["combo"]:
                native = "sim_hier" if component == "similarity" and cell["model"] == "gpt2" else component
                check_factor(session, native, count)
        if cell["arm"] == "ce":
            from ce_from_sft import run_ce
            result = run_ce(session, cell)
        else:
            from stage_training import run_training
            dm = session.setup_data()
            write_json(job_dir / "source_data_audit.json", session.audit_data(dm))
            result = run_training(session, cell, dm, session.indices(dm))
        if result.get("status") != "completed":
            raise RuntimeError(f"Experiment did not complete: {cell}")
        result = {**result, "from_sft_source": session.source_identity, "experiment_cell": cell}
        write_json(completed, result)
        return result
