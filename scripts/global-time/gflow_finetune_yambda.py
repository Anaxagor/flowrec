"""Global-time GFlow and SFT-only from the original Yambda TIGER and GPT-2 checkpoints."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/global-time/exp2-tuned-params/finetune_base_matched.json"
EXPERIMENT = "yambda-gflow-k3"
Y50 = "yambda_50m_listens"
Y500 = "yambda_500m_listens"
Y500_TARGETS_PER_USER = 50
Y500_SUBSET_SEED = 42
COMPONENTS = ("sasrec", "interaction", "similarity")
for directory in (ROOT / "scripts/exp8-from-sft", ROOT / "scripts/exp7-ce-distillation",
                  ROOT / "scripts/exp4-reward-ablations", ROOT / "scripts", ROOT):
    sys.path.insert(0, str(directory))


def cells(model, dataset):
    common = dict(model=model, dataset=dataset)
    schedule = json.loads(PROTOCOL.read_text())["shared_schedule"]
    return {
        "sft": [dict(common, arm="sft", seed=seed) for seed in schedule["seeds_sft_only"]],
        "gflow": [dict(common, arm="gflow", seed=schedule["seed_gflow"], combo="_x_".join(COMPONENTS),
                       sampler="teacher", num_negatives=3, power=power, rho=rho, objective=loss)
                  for power, rho, loss in product((.25, 1., 16.), (1., .1), ("tb", "db", "subtb"))],
    }


def training_args(base, cli, seed, scratch, output):
    schedule = json.loads(PROTOCOL.read_text())
    block, shared = schedule["datasets"][cli.dataset], schedule["shared_schedule"]
    # The matched GPT-2 K=3 runs fit 128 x four trajectories on an 80 GB
    # card. Keep the same effective batch for its SFT-only control and TIGER.
    batch = 2 if cli.smoke else (128 if cli.model == "gpt2" else block["batch_size"])
    accum = 1 if cli.smoke else shared["effective_batch"] // batch
    flag = "--dataset" if cli.model == "tiger" else "--datasets"
    sampling = (["--train-random-n", str(Y500_TARGETS_PER_USER), "--train-subset-seed", str(Y500_SUBSET_SEED)]
                if cli.dataset == Y500 else ["--train-random-n", "0"])
    args = base.parse_args([flag, cli.dataset, "--device", cli.device, "--num-workers", str(cli.num_workers),
                            "--train-last-n", "0", *sampling, "--batch-size", str(batch),
                            "--grad-accum", str(accum), "--num-candidates", "3", "--seed", str(seed)])
    args.output_root, args.cache_root = output, scratch / "factors"
    args.smoke, args.resume = cli.smoke, True
    args.max_epochs = 1 if cli.smoke else shared["max_epochs"]
    args.val_batch_size = 2 if cli.smoke else block["val_batch_size"]
    args.test_batch_size = 2 if cli.smoke else block["test_batch_size"]
    args.max_train_batches, args.max_eval_batches = (8, 2) if cli.smoke else (None, None)
    args.lr, args.weight_decay, args.max_grad_norm = shared["lr"], shared["weight_decay"], shared["max_grad_norm"]
    args.val_sample_fraction = None if cli.smoke else block["val_sample_fraction"]
    args.val_sample_resample, args.validate_at_start = False, True
    args.validate_every, args.patience = shared["validate_every"], shared["patience"]
    args.min_epochs = 0 if cli.smoke else shared["early_stopping_min_epochs"]
    args.amp_dtype = shared["amp_dtype"]
    args.eval_num_beams, args.test_ks = shared["eval_num_beams"], shared["test_ks"]
    args.curriculum_epochs = shared["gflow_curriculum_epochs"]
    args.gflow_weight_mode = "fixed_grad_ratio"
    args.gflow_calibration_batches = 1 if cli.smoke else 5
    args.strict_negatives, args.reward_prefetch, args.tensorized_train_loader = True, True, True
    args.teacher_sampler_label = "sasrec_curriculum" if cli.dataset == Y50 else "sasrec_sce_curriculum"
    args.max_candidate_batch = batch * 4
    args.promote_root = None
    if cli.model == "gpt2":
        args.cache_format, args.uniform_micro_batch, args.train_pool_last_n = "factorized", True, None
        # The checkpoint supervises one full-context target per dense row.
        args.prefix_from_dense = False
    return args


def reward_method(profile, model):
    keys = ("sasrec", "interaction", "sim_hier") if model == "gpt2" else COMPONENTS
    teacher_params = json.loads(profile.teacher_trial.read_text())
    teacher_params = teacher_params.get("best_params", teacher_params)
    teacher = {"checkpoint_path": str(profile.teacher_checkpoint),
               "model": {k: teacher_params[k] for k in ("d_model", "nhead", "num_layers", "dropout", "pos_drop")
                         if k in teacher_params}}
    rewards = [{"name": name, "weight": 1., "reward": reward} for name, reward in zip(COMPONENTS, (
        {"transform": "sigmoid", "temperature": 1.}, {"training_window_only": True},
        {"mode": "hierarchical", "level_stride": 256, "rqvae_model_path": str(profile.codebook)},
    ))]
    return {"name": "gflow", "reward_model": "multiply", "component_keys": list(keys),
            "reward": {"components": rewards}, "sasrec": teacher}


def store_reward_events(dm, directory):
    """Only the interaction-factor worker needs the two positive/auxiliary event streams."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    cutoff = dm.global_val_cutoff
    if cutoff is None or dm.interaction_store is None or dm.interactions is None:
        raise ValueError("Global-time interaction factors require both event streams and the training cutoff")
    for name, frame in (("positive", dm.interactions.loc[:, ["user_id", "item_id", "timestamp"]]),
                        ("auxiliary", dm.interaction_store.events)):
        schema = pa.Schema.from_pandas(frame, preserve_index=False)
        final = directory / (name + ".parquet")
        partial = final.with_suffix(".partial")
        with pq.ParquetWriter(partial, schema) as writer:
            for start in range(0, len(frame), 250_000):
                chunk = frame.iloc[start:start + 250_000]
                chunk = chunk.loc[chunk["timestamp"] < cutoff]
                writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
        partial.replace(final)


def restore_reward_events(dm, directory):
    import pandas as pd
    from flowrec.data.interaction_store import InteractionLogStore
    dm.interactions = pd.read_parquet(directory / "positive.parquet")
    dm.interaction_store = InteractionLogStore(pd.read_parquet(directory / "auxiliary.parquet"),
                                               num_users=dm.num_users, item_vocab_size=dm.item_vocab_size)


def record_train_subset(session, snapshot):
    if "finetune_subset" in session.source_identity:
        session.config["finetune_subset"] = {
            **session.source_identity["finetune_subset"],
            "num_source_examples": snapshot["full_train_count"],
            "num_selected_examples": snapshot["counts"]["train"],
            "source_indices_sha256": snapshot["source_indices_sha256"],
        }


def prepare(session, profile):
    import ce_yambda_prepared_data as compact
    from bounded_preparation import bounded_history_build
    from from_sft_runtime import write_json
    from yambda_gflow_teacher import prepare_teacher_translation
    base, scratch = session.base, session.cache_root
    cfg = base.tiger_runner.build_data_config(session.config, config_path=session.config_path)
    marker = scratch / "prepared.json"
    if marker.exists():
        require_prepared(session)
        return
    print(f"Building {session.source['dataset']} {session.source['model']} prefixes", flush=True)
    dm = base.tiger_runner.build_data_module(cfg)
    with bounded_history_build(50_000):
        dm.setup()
    audit = profile.verify_layout(dm)
    prepare_teacher_translation(session.source["dataset"], dm, profile.teacher_checkpoint,
                                Path(session.config["method"]["sasrec"]["item_id_translation_path"]))
    store_reward_events(dm, scratch)
    # The interaction worker reopens the full training window. Only the
    # target rows are sampled; the catalog and their histories are already fixed.
    summary = dm.summary()
    dm.interactions = dm.interaction_store = None
    indices = base.finetune_indices(dm.train_dataset, session.args) if session.args.train_random_n else None
    snapshot = compact.prepare(scratch / "prefixes", dm, indices, session.source_identity,
                               audit=audit, source_summary=summary)
    record_train_subset(session, snapshot)
    if "finetune_subset" in session.config:
        audit = {**audit, "finetune_subset": session.config["finetune_subset"]}
    write_json(session.config_path, session.config)
    write_json(session.output_root / "preparation/layout_audit.json", audit)
    write_json(marker, session.source_identity)


def require_prepared(session):
    import ce_yambda_prepared_data as compact
    scratch = session.cache_root
    expected = session.source_identity
    marker = scratch / "prepared.json"
    if not marker.is_file() or json.loads(marker.read_text()) != expected:
        raise RuntimeError(f"Missing or mismatched CPU preparation at {scratch}")
    for path in (scratch / "positive.parquet", scratch / "auxiliary.parquet",
                 Path(session.config["method"]["sasrec"]["item_id_translation_path"])):
        if not path.is_file():
            raise FileNotFoundError(path)
    snapshot = compact.check(scratch / "prefixes", expected)
    if snapshot is None:
        raise RuntimeError(f"Missing compact prefix snapshot at {scratch}")
    record_train_subset(session, snapshot)


def factor(session, dm, index):
    import precompute_reward_factors
    from from_sft_runtime import write_json
    name = session.config["method"]["component_keys"][index]
    method = session.config["method"]
    spec = method["reward"]["components"][index]
    directory = session.cache_dirs()[name]
    config = {**session.config, "method": {"name": "gflow", "reward_model": spec["name"],
              "reward": spec["reward"], "sasrec": method["sasrec"]}}
    config_path = session.cache_root / f"factor_{name}.json"
    with session.base.control.file_lock(directory / "build.lock"):
        if (directory / "ready.json").exists():
            check_factor(session, dm, name)
            return
        if spec["name"] == "interaction":
            restore_reward_events(dm, session.cache_root)
        write_json(config_path, config)
        flags = ["--config", str(config_path), "--output-dir", str(directory), "--device", session.args.device,
                 "--num-workers", str(session.args.num_workers), "--batch-size", "128",
                 "--verify-batches", "1", "--verify-examples-per-batch", "2"]
        if spec["name"] == "interaction" and session.source["dataset"] == Y500:
            flags.append("--spool-interaction")
        if session.args.smoke:
            flags.extend(["--max-examples", str(session.base._smoke_examples(session.args))])
        # The compact train rows already follow the fixed draw in source order.
        # Passing sampling flags to the factor loader would draw a second time.
        precompute_reward_factors.main(flags, prepared_data=dm)
        if "finetune_subset" in session.config:
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(session.base.subset_spec(session.args), finetune_subset=session.config["finetune_subset"])
            write_json(manifest_path, manifest)
        check_factor(session, dm, name, require_ready=False)
        write_json(directory / "ready.json", session.source_identity)


def check_factor(session, dm, name, *, require_ready=True):
    directory = session.cache_dirs()[name]
    if require_ready and json.loads((directory / "ready.json").read_text()) != session.source_identity:
        raise RuntimeError(f"Reward source changed at {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("finetune_subset") != session.config.get("finetune_subset"):
        raise RuntimeError(f"Factor cache selected different source rows at {directory}")
    expected = len(dm.train_dataset)
    if session.args.smoke:
        expected = min(expected, session.base._smoke_examples(session.args))
    validator = session.base.factor_runner if session.source["model"] == "gpt2" else session.base
    validator.check_component_manifest(manifest, directory, "similarity" if name == "sim_hier" else name,
                                       expected, session.args)
    if manifest["num_examples"] != expected or not (directory / "factors.pt").is_file():
        raise RuntimeError(f"Incomplete factor cache: {directory}")
    if manifest["num_catalog_items"] != len(dm.semantic_ids) or not manifest.get("verification", {}).get("examples"):
        raise RuntimeError(f"Factor catalog or numerical verification missing at {directory}")
    if name == "sasrec" and not manifest.get("teacher_id_translation"):
        raise RuntimeError("SASRec cache was built without the global catalog translation")


def run_cell(session, dm, cell):
    from from_sft_runtime import write_json
    from stage_training import run_training
    if cell["arm"] == "sft":
        relative = Path("sft_only")
    else:
        for name in session.cache_dirs():
            check_factor(session, dm, name)
        relative = Path("gflow") / f"power_{cell['power']:g}" / f"rho_{cell['rho']:g}" / cell["objective"]
    session.run_dir = session.output_root / relative / f"seed_{cell['seed']}"
    directory = session.run_dir
    with session.base.control.file_lock(directory / ".run.lock"):
        identity = {"cell": cell, "global_init": session.source_identity}
        if "finetune_subset" in session.config:
            identity["finetune_subset"] = session.config["finetune_subset"]
        manifest = directory / "identity.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise RuntimeError(f"Initialization or protocol changed at {directory}")
        write_json(manifest, identity)
        result_file = directory / "result.json"
        if result_file.exists() and json.loads(result_file.read_text()).get("status") == "completed":
            print(f"Already complete: {directory}", flush=True)
            return
        result = run_training(session, cell, dm, range(len(dm.train_dataset)))
        if result["status"] != "completed":
            raise RuntimeError(f"Training failed at {directory}")
        print(f"Completed {directory}", flush=True)


def execute(cli, cell):
    from from_sft_runtime import SourceSession, bind, json_value, write_json
    from yambda_gflow_inputs import load_profile, stage_inputs
    import ce_yambda_prepared_data as compact
    import gflow_finetune_gpt2_amazon as gpt2
    import gflow_finetune_yambda500m as tiger
    base = gpt2 if cli.model == "gpt2" else tiger
    shared = cli.workspace / EXPERIMENT
    inputs = shared / "inputs"
    scratch = shared / cli.model / cli.dataset
    output = cli.output_root.resolve() / cli.model / cli.dataset
    if cli.dataset == Y500:
        scratch, output = scratch / "sampled50", output / "sampled50"
    if cli.smoke:
        scratch, output = scratch / "smoke", output / "smoke"
    if cli.stage == "prepare":
        stage_inputs(cli.model, cli.dataset, inputs)
    profile = load_profile(cli.model, cli.dataset, inputs)
    if cli.model == "tiger":
        base.activate_dataset(cli.dataset)
    args = training_args(base, cli, cell["seed"] if cell else 42, scratch, output)
    config = profile.run_config(args)
    config["method"] = reward_method(profile, cli.model)
    config["method"]["sasrec"]["item_id_translation_path"] = str(scratch / "teacher_translation.json")
    source = {"model": cli.model, "dataset": cli.dataset}
    identity = json_value({"source": profile.source_identity, "model": cli.model, "dataset": cli.dataset,
                           "split": "global_time", "last_n": 128, "reward": config["method"],
                           "inputs": config["data"], "layout": config["dataset"]})
    if cli.dataset == Y500:
        identity["finetune_subset"] = {"sampling": "uniform_without_replacement", "pool_last_n": 128,
                                      **base.subset_spec(args)}
    session = SourceSession(source, cli, base, args, config, output, scratch,
                            profile.checkpoint, identity, [])
    session.source_metadata_key = "global_init"
    with ExitStack() as stack:
        stack.enter_context(bind(base, write_json=write_json))
        if cli.model == "gpt2":
            model_profile = {"model": profile.model_config}
            stack.enter_context(bind(base, tuned_checkpoint_paths=lambda *_: {"checkpoint": profile.checkpoint},
                                     tuned_profile=lambda *_: (model_profile, profile.model_config),
                                     EXP1_GPT2_MODEL_YAML=ROOT / "configs/global-time/exp1-standard_params/gpt2_model.yaml"))
        else:
            specs = {spec["name"]: spec for spec in config["method"]["reward"]["components"]}
            stack.enter_context(bind(base, TIGER_CHECKPOINT=str(profile.checkpoint),
                                     SASREC_CHECKPOINT=str(profile.teacher_checkpoint),
                                     RQVAE_MODEL=str(profile.codebook), component_spec=lambda name, _: specs[name],
                                     POSITIVE_CRITERION=config["dataset"].get("positive_criterion")))
        if cli.stage == "prepare":
            with base.control.file_lock(scratch / "prepare.lock"):
                prepare(session, profile)
            return
        require_prepared(session)
        cfg = base.tiger_runner.build_data_config(config, config_path=session.config_path)
        dm = compact.load(scratch / "prefixes", cfg, identity)
        if cli.stage == "factors":
            factor(session, dm, cli.index)
        else:
            run_cell(session, dm, cell)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "prepare", "factors", "sft", "gflow"))
    parser.add_argument("--dataset", required=True, choices=(Y50, Y500))
    parser.add_argument("--model", required=True, choices=("tiger", "gpt2"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/global-time" / EXPERIMENT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args(argv)
    if cli.model == "gpt2" and cli.dataset != Y50:
        parser.error("GPT-2/last_128 is available only for Yambda 50M in this experiment")
    grid = cells(cli.model, cli.dataset)
    if cli.stage == "list":
        for arm, entries in grid.items():
            for index, entry in enumerate(entries):
                print(arm, index, json.dumps(entry, sort_keys=True))
        return
    if cli.workspace is None or not cli.workspace.is_absolute():
        parser.error("--workspace must be an absolute scratch path")
    if cli.stage != "prepare":
        size = len(COMPONENTS) if cli.stage == "factors" else len(grid[cli.stage])
        if cli.index is None or not 0 <= cli.index < size:
            parser.error(f"{cli.stage} requires --index between 0 and {size - 1}")
    if cli.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if cli.device is None:
        cli.device = ("cpu" if cli.stage == "prepare" or (cli.stage == "factors" and cli.index != 0)
                      else "cuda")
    execute(cli, grid[cli.stage][cli.index] if cli.stage in grid else None)


if __name__ == "__main__":
    main()
