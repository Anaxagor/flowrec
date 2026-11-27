"""Select global HPO winners, refit fresh initial models, and score the held-out test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import refit_sources

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/global-time/initial_train_with_validation"


def result_directory(output, spec):
    # A copied export or a higher precision printout of the same score does not change the fit.
    identity = {k: spec[k] for k in ("model", "dataset", "study_name", "trial", "params", "epochs", "seed")}
    digest = hashlib.sha256(encode(identity)).hexdigest()[:16]
    return output / spec["model"] / spec["dataset"] / f"{spec['trial']}-{digest}"


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def complete(directory, spec):
    path = directory / "result.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    if (result.get("status") != "completed" or result.get("epochs_completed") != spec["epochs"]
            or any(result.get(k) != spec[k] for k in ("model", "dataset", "trial", "params", "seed"))
            or refit_sources._metric_error(result.get("test_metrics"), result.get("expected_test_examples"))
            or not (directory / "checkpoints/last.pt").is_file()):
        raise ValueError(f"Inconsistent completed refit at {directory}")
    return True


def select(args):
    pairs = [(m, d) for m, d in refit_sources.supported_initial_pairs()
             if (not args.model or args.model == m) and (not args.dataset or args.dataset == d)]
    if not pairs:
        raise SystemExit("No global HPO study for the requested model/dataset.")
    tasks = []
    unavailable = 0
    print("MODEL  DATASET  TRIAL  VAL_NDCG@10  REFIT_EPOCHS  STATUS", flush=True)
    for model, dataset in pairs:
        try:
            spec = refit_sources.select_initial(ROOT, model, dataset)
        except refit_sources.InitialSelectionUnavailable as exc:
            print(f"{model}  {dataset}  unavailable: {exc}", flush=True)
            unavailable += 1
            continue
        directory = result_directory(args.output_root, spec)
        done = complete(directory, spec)
        print(f"{model}  {dataset}  {spec['trial']}  {spec['validation_ndcg10']:.8f}  "
              f"{spec['epochs']}  {'completed' if done else 'ready'}", flush=True)
        if not done:
            tasks.append(dict(spec=spec, output=str(directory)))
    if unavailable:
        print(f"Unavailable pairs: {unavailable} of {len(pairs)}. See the evidence missing above.", flush=True)
        if unavailable == len(pairs):
            raise SystemExit(1)
    return tasks


def fit(spec, workspace, directory, *, device, num_workers):
    import refit_initial_training as training

    if not complete(directory, spec):
        training.run_initial(spec, workspace, directory, device=device, num_workers=num_workers)
    print("Test result:", directory / "result.json", flush=True)


def selection_args(cmd):
    cmd.add_argument("--model", choices=("gpt2", "tiger", "sasrec", "sasrec_sce"))
    cmd.add_argument("--dataset", choices=("beauty", "instruments", "beauty_2023",
                                            "yambda_50m_listens", "yambda_500m_listens"))
    cmd.add_argument("--output-root", type=Path, default=OUTPUT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection_args(sub.add_parser("plan"))
    if Path(__file__).with_name("initial_refit_submission.py").is_file():
        import initial_refit_submission
        initial_refit_submission.add_command(sub, selection_args)
    worker = sub.add_parser("work")
    worker.add_argument("stage", choices=("prepare", "train"))
    worker.add_argument("manifest", type=Path)
    worker.add_argument("index", type=int)
    args = parser.parse_args(argv)
    if args.command == "work":
        import refit_initial_training as training

        manifest = json.loads(args.manifest.read_text())
        if not 0 <= args.index < len(manifest["tasks"]):
            parser.error("Task index is outside the manifest")
        task, workspace = manifest["tasks"][args.index], Path(manifest["workspace"])
        if args.stage == "prepare":
            training.prepare_initial(task["spec"], workspace, num_workers=0)
        else:
            fit(task["spec"], workspace, Path(task["output"]), device="cuda", num_workers=2)
    else:
        args.output_root = args.output_root.resolve()
        tasks = select(args)
        if hasattr(args, "dispatch"):
            args.dispatch(args, tasks)


if __name__ == "__main__":
    main()
