"""Optuna-tune TIGER on Beauty, Instruments, Beauty 2023 and Yambda 50M.

Uses pinned global-time splits. Defaults to tune then train the winner from
scratch for the original trainer schedule and seeds. Parallel workers can share
a dataset with --stage tune --worker-id N. Rerun to resume interrupted work.
Yambda 500M has its own compact-grid entry point, tuning_tiger_yambda_500m.py.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import global_time_tiger_tuning as shared
from scripts.checkpoint_runs import publish_primary_checkpoint
from scripts.optuna_trial_lease import allocation_lock, claim_trial, journal_storage, worker_mode_lock

DATASETS = shared.DATASETS
TRIAL_BUDGETS = dict(zip(DATASETS, (100, 100, 60, 40)))
SEARCH_SPACE = {
    "lr": {"low": 1e-4, "high": 5e-3, "log": True},
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "embedding_dim": [128, 256], "num_layers": [2, 4],
    "weight_decay": [0.0, 0.01, 0.1], "d_kv": [32, 64],
}


def parse_args(argv=None):
    p = shared.build_parser(__doc__)
    p.add_argument("--datasets", nargs="+", choices=(*DATASETS, "all"))
    p.add_argument("--n-trials", type=int, help="Default budgets: 100/100/60/40")
    p.add_argument("--worker-id", type=int, help="Join a parallel tuning study with this worker index")
    p.add_argument("--time-budget-exit-code", type=int, default=0,
                   help="Exit status on a resumable budget stop (e.g. 75 for a batch supervisor)")
    p.add_argument("--publish-base", action="store_true",
                   help="Publish the evaluated final seed-42 Beauty 2023 winner to the global initial checkpoint tree")
    args = shared.validate_args(p, p.parse_args(argv))
    if args.n_trials is not None and args.n_trials < 1:
        p.error("--n-trials must be positive")
    if args.worker_id is not None and (args.worker_id < 0 or args.stage != "tune"):
        p.error("--worker-id requires a nonnegative index and --stage tune")
    if not 0 <= args.time_budget_exit_code <= 255:
        p.error("--time-budget-exit-code must fit in 0..255")
    args.datasets = args.datasets or (["beauty_2023"] if args.publish_base else ["all"])
    args.datasets = list(DATASETS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    if args.publish_base:
        if args.datasets != ["beauty_2023"] or args.stage != "train-best" or args.list_plan:
            p.error("--publish-base requires --stage train-best for Beauty 2023")
        args.seeds = args.seeds or [42]
        if 42 not in args.seeds:
            p.error("--publish-base requires final training seed 42")
        if (args.n_trials not in (None, TRIAL_BUDGETS["beauty_2023"]) or args.val_sample_fraction not in (None, 0.25)
                or any(value is not None for value in (args.max_epochs, args.epochs_per_trial,
                                                       args.train_batches_per_epoch, args.max_eval_batches))):
            p.error("--publish-base requires the full study and the uncapped final training/evaluation protocol")
        if (args.config_dir != shared.CONFIG_DIR or args.semantic_ids_dir != shared.grid.DEFAULT_SEMANTIC_IDS_DIR
                or args.model_config != shared.EXP1_CONFIG_DIR / "tiger_model.yaml"
                or args.trainer_config != shared.EXP1_CONFIG_DIR / "trainer_tiger.yaml"):
            p.error("--publish-base requires the standard global dataset, semantic IDs, model and trainer inputs")
        working_root = args.checkpoint_root.resolve()
        if working_root.is_relative_to(PROJECT_ROOT / "artifacts/checkpoint_archive") or (
                working_root.is_relative_to(PROJECT_ROOT / "checkpoints")):
            p.error("--publish-base needs a working study directory outside the archive and curated checkpoints")
    return args


def search_spec(dataset, args, incumbent):
    # Always include the configured baseline in every categorical axis.
    space = {}
    for name, values in SEARCH_SPACE.items():
        if isinstance(values, dict):
            baseline = shared.grid.GRID_PARAMS[name].read(incumbent.config)
            space[name] = {**values, "low": min(values["low"], baseline), "high": max(values["high"], baseline)}
        else:
            baseline = shared.grid.GRID_PARAMS[name].read(incumbent.config)
            space[name] = list(dict.fromkeys([baseline, *values]))
    return {"kind": "optuna", "n_trials": args.n_trials or TRIAL_BUDGETS[dataset],
            "space": space, "sampler": "TPE", "startup_trials": 10, "pruner_warmup_epochs": 10}


def suggest(trial, space):
    return {name: trial.suggest_float(name, **values) if isinstance(values, dict)
            else trial.suggest_categorical(name, values) for name, values in space.items()}


def summarize(run, study):
    with allocation_lock(run.root):
        return publish_snapshot(run, study.get_trials(deepcopy=False))


def publish_snapshot(run, trials):
    results = [{"trial": f"trial_{t.number:04d}", "status": t.state.name.lower(),
                "best_value": t.value, "overrides": t.params,
                "pruned_reason": t.user_attrs.get("pruned_reason"),
                "error": t.user_attrs.get("error")} for t in trials]
    finished = [t for t in trials if t.state.is_finished()]
    complete = len(finished) >= run.search["n_trials"] and all(t.state.is_finished() for t in trials)
    candidates = [r for r in results if r["status"] == "complete" and r["best_value"] is not None]
    winner = max(candidates, key=lambda r: r["best_value"]) if candidates else None
    return run.publish(winner, results, complete=complete)


def open_study(run):
    study = optuna.create_study(
        study_name=f"tiger_{run.dataset}_global_time_ndcg10", direction="maximize", load_if_exists=True,
        storage=journal_storage(run.root / "optuna_journal.log", resumable=run.args.worker_id is not None),
        sampler=optuna.samplers.TPESampler(seed=42 + 9973 * (run.args.worker_id or 0), n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )
    if not study.trials:
        study.enqueue_trial({name: shared.grid.GRID_PARAMS[name].read(run.incumbent.config)
                             for name in run.search["space"]})
    return study


def tune(run):
    with allocation_lock(run.root):
        run.check_protocol()
        study = open_study(run)
    if run.args.stage in ("report", "train-best"):
        return summarize(run, study)
    while True:
        with claim_trial(study, run.root, run.search["n_trials"], count_failures=True) as trial:
            if trial is None:
                return summarize(run, study)
            run_trial(run, study, trial)
        summarize(run, study)


def run_trial(run, study, trial):
    overrides = suggest(trial, run.search["space"])
    cfg = shared.grid.Trial(overrides).config(run.incumbent.config)

    def report(payload):
        value = payload.get("val", {}).get(shared.MONITOR)
        step = int(payload["epoch"])
        if value is not None and step not in trial._get_latest_trial().intermediate_values:
            trial.report(float(value), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    try:
        result = run.train(cfg, run.root / "trials" / f"trial_{trial.number:04d}", report_epoch=report)
        trial.set_user_attr("best_checkpoint", result["best_checkpoint"])
        study.tell(trial, result["best_value"])
    except shared.TimeBudgetExceeded:
        summarize(run, study)
        raise
    except optuna.TrialPruned:
        study.tell(trial, state=TrialState.PRUNED)
    except FloatingPointError as exc:
        trial.set_user_attr("pruned_reason", "non_finite_loss")
        trial.set_user_attr("error", str(exc)[:2000])
        study.tell(trial, state=TrialState.PRUNED)
        print(f"[{run.dataset}/trial_{trial.number:04d}] pruned for non-finite loss: {exc}", flush=True)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            trial.set_user_attr("error", str(exc)[:2000])
            study.tell(trial, state=TrialState.PRUNED)
        else:
            study.tell(trial, state=TrialState.FAIL)
            summarize(run, study)
            raise


def publish_beauty_base(run, summary):
    """Publish only the evaluated seed-42 fit of this study's selected configuration."""
    if not summary["search_complete"] or run.dataset != "beauty_2023" or summary["dataset"] != run.dataset:
        raise ValueError("Publication requires the completed global-time Beauty 2023 search")
    winner = summary["winner"]
    cfg = shared.grid.Trial(winner["overrides"]).config(run.incumbent.config)
    seed_dir = run.root / "seed_42"
    checkpoint = seed_dir / "best.pt"
    result = json.loads((seed_dir / "result.json").read_text())
    resolved = json.loads((seed_dir / "resolved_config.json").read_text())
    trainer = run.trainer_config(cfg, seed_dir, final=True).to_dict()
    if (result.get("status") != "complete" or result.get("seed") != 42 or resolved.get("seed") != 42
            or result.get("hparams") != cfg.hparams() or resolved.get("model") != cfg.model
            or resolved.get("optimizer") != shared.runner.to_jsonable(shared.runner.optimizer_config(cfg.trainer))
            or result.get("monitor") != trainer["monitor"] or result.get("epochs_planned") != trainer["max_epochs"]
            or not 0 <= result["best_epoch"] < result["epochs_completed"] <= trainer["max_epochs"]
            or Path(result.get("best_checkpoint", "")).resolve() != checkpoint.resolve()):
        raise ValueError("Publication requires the completed final seed-42 winner and its own best.pt")
    # Loader workers and progress display may change across allocations.
    data = {key: value for key, value in resolved["data"].items()
            if key not in ("num_workers", "persistent_workers", "pin_memory")}
    if (data != summary["protocol"]["data"]
            or any(resolved["trainer"].get(key) != value for key, value in trainer.items()
                   if key not in ("progress_bar", "checkpoint_dir"))):
        raise ValueError("The saved seed-42 data or training protocol differs from the canonical Beauty 2023 run")
    sizes = json.loads((run.root / "data_summary.json").read_text())
    for split, count in (("validation", "num_val_examples"), ("test", "num_test_examples")):
        metrics = result.get(f"{split}_metrics", {})
        if (not sizes[count] or metrics.get("num_examples") != sizes[count]
                or not math.isfinite(metrics.get("NDCG@10", float("nan")))):
            raise ValueError(f"Publication requires complete {split} metrics for the final seed-42 checkpoint")
    payload = shared.torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = payload["trainer_state"]
    if (payload.get("model_type") != "tiger" or not payload.get("model_state_dict")
            or payload.get("trainer_config") != resolved["trainer"]
            or state["best_epoch"] != result["best_epoch"] or state["best_metric"] != result["best_value"]):
        raise ValueError("The saved best.pt does not match the evaluated final TIGER checkpoint")
    study_name = f"tiger_{run.dataset}_global_time_ndcg10"
    destination = PROJECT_ROOT / "checkpoints/exp2/global_split/initial/tiger"
    with allocation_lock(destination / run.dataset):
        publish_primary_checkpoint(
            PROJECT_ROOT, destination, run.dataset, 42, checkpoint, run_dir=seed_dir,
            parameters=cfg.hparams(), summary={**result, "study_name": study_name, "winner": winner["trial"]},
        )
        path = destination / run.dataset / "best_trial.json"
        record = json.loads(path.read_text())
        shared.atomic_json(path, {**record, "study_name": study_name,
                                  "best_trial_number": int(winner["trial"].removeprefix("trial_"))})
    print(f"Published {destination / run.dataset / 'best.pt'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    for dataset in args.datasets:
        incumbent = shared.load_incumbent(dataset, args)
        run = shared.Run(dataset, args, search=search_spec(dataset, args, incumbent), started_at=started)
        if args.list_plan:
            print(json.dumps(run.protocol(), indent=2))
            continue
        if (args.worker_id is not None or args.publish_base) and args.device.startswith("cuda") and run.device == "cpu":
            raise RuntimeError("TIGER requested CUDA but no GPU is available")
        with worker_mode_lock(run.root, parallel=args.worker_id is not None):
            try:
                summary = tune(run)
                if args.stage in ("all", "train-best"):
                    run.train_best(summary)
                    if args.publish_base:
                        publish_beauty_base(run, summary)
            except shared.TimeBudgetExceeded as exc:
                print(str(exc), flush=True)
                return args.time_budget_exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
