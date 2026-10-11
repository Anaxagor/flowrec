"""Compact global-time TIGER grid on Yambda 500M, then full winner training.

Defaults: baseline plus nine single-axis alternatives, eight epochs each,
fixed 25% validation sample, constrained 10-beam NDCG@10. After the grid,
train the winner from scratch with the original Tiger trainer schedule
(500-epoch cap, patience 10, full 1-beam validation NDCG@1; seeds 42/43/44).
Final evaluation uses full validation/test splits and 20 beams.

One worker loads data once and resumes trials/final runs from last.pt on
resubmission. --stage tune/train-best/report separates the phases if needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import global_time_tiger_tuning as shared

DATASET = shared.LARGE_DATASET


def parse_args(argv=None):
    p = shared.build_parser(__doc__)
    p.set_defaults(epochs_per_trial=8)
    p.add_argument("--design", choices=("ofat", "full"), default="ofat")
    p.add_argument("--params", nargs="+", choices=list(shared.grid.GRID_PARAMS),
                   default=list(shared.grid.DEFAULT_PARAMS))
    p.add_argument("--grid", action="append", type=shared.grid.parse_grid_spec, default=[],
                   metavar="NAME=V1,V2", help="Replace an axis's alternatives; baseline is always included")
    args = shared.validate_args(p, p.parse_args(argv))
    args.params = list(dict.fromkeys(args.params))
    for name, _ in args.grid:
        if name not in args.params:
            p.error(f"--grid {name} requires {name} in --params")
    return args


def build_plan(args, incumbent):
    grids = shared.grid.resolve_grids(args.params, incumbent.config, dict(args.grid))
    plan = shared.grid.build_plan(design=args.design, params=args.params, grids=grids,
                                  incumbent=incumbent.config, extra_trials=[])
    search = {"kind": "grid", "design": args.design, "grids": grids,
              "trials": [{"trial": t.name, "overrides": t.overrides} for t in plan]}
    return plan, search


def summarize(run, plan):
    results = []
    for trial in plan:
        path = run.root / "trials" / trial.name / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
        results.append({**result, "trial": trial.name, "overrides": trial.overrides})
    candidates = [r for r in results if r["status"] == "complete"]
    winner = max(candidates, key=lambda r: r["best_value"]) if candidates else None
    complete = all(r["status"] in ("complete", "oom", "non_finite_loss") for r in results)
    return run.publish(winner, results, complete=complete)


def tune(run, plan):
    if run.args.stage not in ("report", "train-best"):
        for trial in plan:
            directory = run.root / "trials" / trial.name
            try:
                run.train(trial.config(run.incumbent.config), directory)
            except shared.TimeBudgetExceeded:
                summarize(run, plan)
                raise
            except FloatingPointError as exc:
                # Numerical divergence disqualifies this grid point, including
                # its earlier finite checkpoints; the remaining points still run.
                shared.atomic_json(directory / "result.json", {
                    "status": "non_finite_loss", "error": str(exc)[:2000],
                    "best_value": None, "hparams": trial.config(run.incumbent.config).hparams(),
                })
                print(f"[{run.dataset}/{trial.name}] non-finite loss; excluding this trial "
                      f"and continuing the grid: {exc}", flush=True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                shared.atomic_json(directory / "result.json", {"status": "oom", "error": str(exc)[:2000]})
            summarize(run, plan)
    return summarize(run, plan)


def main(argv=None):
    args = parse_args(argv)
    incumbent = shared.load_incumbent(DATASET, args)
    plan, search = build_plan(args, incumbent)
    run = shared.Run(DATASET, args, search=search)
    if args.list_plan:
        print(json.dumps(run.protocol(), indent=2))
        return
    with shared.dataset_lock(run.root):
        run.check_protocol()
        try:
            summary = tune(run, plan)
            if args.stage in ("all", "train-best"):
                run.train_best(summary)
        except shared.TimeBudgetExceeded as exc:
            print(str(exc), flush=True)


if __name__ == "__main__":
    main()
