"""Optuna-tune TIGER on Beauty, Instruments, Beauty 2023 and Yambda 50M.

Uses pinned global-time splits. Defaults to tune then train the winner from
scratch for the original trainer schedule and seeds. One worker per dataset;
rerun the same command to resume interrupted trials or final seed runs.
Yambda 500M has its own compact-grid entry point, tuning_tiger_yambda_500m.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import global_time_tiger_tuning as shared

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
    p.add_argument("--datasets", nargs="+", choices=(*DATASETS, "all"), default=["all"])
    p.add_argument("--n-trials", type=int, help="Default budgets: 100/100/60/40")
    args = shared.validate_args(p, p.parse_args(argv))
    if args.n_trials is not None and args.n_trials < 1:
        p.error("--n-trials must be positive")
    args.datasets = list(DATASETS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
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
    results = [{"trial": f"trial_{t.number:04d}", "status": t.state.name.lower(),
                "best_value": t.value, "overrides": t.params,
                "pruned_reason": t.user_attrs.get("pruned_reason"),
                "error": t.user_attrs.get("error")} for t in study.trials]
    finished = [t for t in study.trials if t.state.is_finished()]
    complete = len(finished) >= run.search["n_trials"] and all(t.state.is_finished() for t in study.trials)
    candidates = [r for r in results if r["status"] == "complete" and r["best_value"] is not None]
    winner = max(candidates, key=lambda r: r["best_value"]) if candidates else None
    return run.publish(winner, results, complete=complete)


def tune(run):
    study = optuna.create_study(
        study_name=f"tiger_{run.dataset}_global_time_ndcg10", direction="maximize", load_if_exists=True,
        storage=JournalStorage(JournalFileBackend(str(run.root / "optuna_journal.log"))),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )
    if not study.trials:
        study.enqueue_trial({name: shared.grid.GRID_PARAMS[name].read(run.incumbent.config)
                             for name in run.search["space"]})
    if run.args.stage in ("report", "train-best"):
        return summarize(run, study)
    while True:
        # Dataset lock guarantees that any RUNNING trial here is interrupted.
        running = [t for t in study.trials if t.state == TrialState.RUNNING]
        if not running and sum(t.state.is_finished() for t in study.trials) >= run.search["n_trials"]:
            break
        run.check_time()
        trial = optuna.Trial(study, running[0]._trial_id) if running else study.ask()
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
        summarize(run, study)
    return summarize(run, study)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    for dataset in args.datasets:
        incumbent = shared.load_incumbent(dataset, args)
        run = shared.Run(dataset, args, search=search_spec(dataset, args, incumbent), started_at=started)
        if args.list_plan:
            print(json.dumps(run.protocol(), indent=2))
            continue
        with shared.dataset_lock(run.root):
            run.check_protocol()
            try:
                summary = tune(run)
                if args.stage in ("all", "train-best"):
                    run.train_best(summary)
            except shared.TimeBudgetExceeded as exc:
                print(str(exc), flush=True)
                return


if __name__ == "__main__":
    main()
