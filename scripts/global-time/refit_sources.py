"""Select initial global HPO winners and inventory the downstream refit experiments.

Published exports can be read on a login node with the standard library. Live
Optuna journals use the project environment. Discovery reads metadata and never
loads model tensors, including for a repository mounted at a different path.
"""
from __future__ import annotations

from itertools import product
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile

import gflow_finetune_yambda as global_grid

REFIT_SEEDS = (42, 43, 44)
LEGACY_DATASETS = ("beauty", "instruments")
COMPACT_DATASETS = (global_grid.Y50, global_grid.Y500, "beauty_2023")
COMBO = "_x_".join(global_grid.COMPONENTS)
METRIC_KEYS = tuple(f"{metric}@{k}" for metric in ("HR", "NDCG", "Coverage", "ARP") for k in (5, 10, 20))
DEFAULT_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments/global-time"


class InitialSelectionUnavailable(ValueError):
    """No completed global HPO winner or no reliable epoch for that winner."""


def supported_initial_pairs():
    """The model/dataset combinations supported by the original global tuners."""
    for dataset in (*LEGACY_DATASETS, "beauty_2023", global_grid.Y50, global_grid.Y500):
        for model in ("gpt2", "tiger", "sasrec_sce" if dataset == global_grid.Y500 else "sasrec"):
            yield model, dataset


def _study_name(model, dataset):
    if model == "tiger":
        return f"tiger_{dataset}_global_time_ndcg10"
    metric = "generation_ndcg10" if model == "gpt2" else "ndcg10"
    suffix = "_last_128" if model == "gpt2" and dataset.startswith("yambda_") else ""
    return f"{model}_{dataset}_val_{metric}_exp2{suffix}_global_time"


def _initial_locations(root, model, dataset):
    """Locate the published checkpoint, tuner archives and individual study runs."""
    matched = model == "gpt2" and dataset.startswith("yambda_")
    subdir = "gpt2_last_128" if matched else model
    experiments = root / "experiments/global-time"
    canonical = root / "checkpoints/exp2/global_split/initial" / model / dataset
    canonical = canonical / "last_128" if matched else canonical
    archive = root / "artifacts/checkpoint_archive"
    dirs = [canonical, *(base / subdir / dataset for base in [
        root / "checkpoints/global-time/exp2-tuned_params",
        experiments / "checkpoints/exp2-tuned_params",
        *sorted(archive.glob("*/global-time/exp2-tuned_params")),
    ])]
    slug = dataset.replace("_", "").removesuffix("listens")
    jobs_root = experiments / "hpo" / f"{model}_{slug}"
    if matched:
        jobs_root /= "last_128"
    jobs = sorted(p for p in jobs_root.glob("*") if p.is_dir() and p.name.isdigit())
    dirs.extend(job / "checkpoints" / dataset for job in jobs)
    return dirs, jobs


def _journal_winner(path, study_name):
    # Use Optuna's reader for live cluster studies, including trials completed
    # before the winner manifest was published. No SQLite/Journal conversion.
    try:
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
    except ImportError as exc:
        raise InitialSelectionUnavailable(f"Activate the project's Optuna environment to read {path}") from exc
    storage = JournalStorage(JournalFileBackend(str(path)))
    if study_name not in optuna.study.get_all_study_names(storage):
        return None  # A new job may have created its journal before registering the study.
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        return None
    best = study.best_trial
    epoch = best.user_attrs.get("best_epoch")
    if epoch is None:
        # The TIGER tuner reports its validation metric at each zero-based epoch.
        epoch = min((e for e, v in best.intermediate_values.items() if v == best.value), default=None)
    return dict(trial=f"trial_{best.number:04d}", params=best.params, score=best.value,
                epoch=epoch, seed=best.user_attrs.get("seed"))


def _eligible_protocol(protocol, model, dataset):
    """Exclude deliberately shortened debug evaluations and incompatible data regimes."""
    if not protocol:
        return True
    cfg = protocol.get("data") or protocol.get("data_config")
    if cfg and (cfg.get("split_strategy") != "global_time" or (model in ("gpt2", "tiger")
            and dataset.startswith("yambda_") and (cfg.get("train_target_policy"), cfg.get("train_last_n"))
            != ("last_n", 128))):
        return False
    trainer = protocol.get("trainer", {}).get("search", protocol)
    return all(trainer.get(k) is None for k in ("debug_max_batches", "max_train_batches", "max_eval_batches"))


def _published_winner(path, model, dataset, study_name):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        # Optuna versions have exported the state as "1", "COMPLETE", or "TrialState.COMPLETE".
        rows = [r for r in data if str(r["state"]).split(".")[-1] in ("1", "COMPLETE")
                and r.get("value") is not None]
        if not rows:
            return None
        row = max(rows, key=lambda r: (r["value"], -r["number"]))
        attrs = row.get("user_attrs", {})
        return dict(trial=f"trial_{row['number']:04d}", params=row["params"], score=row["value"],
                    epoch=attrs.get("best_epoch"), seed=attrs.get("seed"))
    if data.get("dataset", dataset) != dataset or data.get("study_name", study_name) != study_name:
        raise InitialSelectionUnavailable(f"Unexpected dataset or study name in {path}")
    if model == "tiger":
        protocol = data.get("protocol", {})
        if not _eligible_protocol(protocol, model, dataset):
            return None
        if "trials" in data:
            completed = [t for t in data["trials"] if t.get("status") == "complete"]
            if not completed:
                return None
            data = max(completed, key=lambda t: t["best_value"])
        if data.get("status") != "complete" or "trial" not in data:
            return None  # May be a publication for a separately trained final seed.
        params = data.get("hparams") if dataset == global_grid.Y500 else data.get("overrides")
        return dict(trial=data["trial"], params=params, score=data["best_value"],
                    epoch=data.get("best_epoch"), seed=data.get("seed"))
    score_key = "best_value_val_generation_ndcg10" if model == "gpt2" else "best_value_val_ndcg10"
    selection = str(data.get("selection", "")).lower()
    if "best_trial_number" not in data or any(tag in selection for tag in ("rerank", "re-rank", "over seeds")):
        return None
    if data.get("metric") != ("val/sft/generation_ndcg@10" if model == "gpt2" else "val/sft/ndcg@10"):
        raise InitialSelectionUnavailable(f"Unexpected HPO validation metric in {path}")
    return dict(trial=f"trial_{data['best_trial_number']:04d}", params=data["best_params"],
                score=data[score_key], epoch=data.get("best_epoch"), seed=None)


def _exported_gpt2_winner(path, study_name):
    """Read the selected trial from a compact export containing its tuned YAML."""
    content = path.read_text()
    if f"# study: {study_name}\n" not in content or "# selection: best single-seed trial" not in content:
        return None
    trial = re.search(r"^# best trial: (\d+), best epoch: (\d+)$", content, re.M)
    score = re.search(r"^# best val/sft/generation_ndcg@10: ([\d.]+)$", content, re.M)
    params = re.search(r"^# search-space point: (.+)$", content, re.M)
    if not (trial and score and params):
        raise InitialSelectionUnavailable(f"Missing HPO provenance in {path}")
    number, epoch = int(trial[1]), int(trial[2])
    row = dict(trial=f"trial_{number:04d}", params=json.loads(params[1]), score=float(score[1]),
               epoch=epoch, seed=None, rounded=True)
    dataset = path.stem.removeprefix("gpt2_model_")
    sidecar = path.parents[1] / "checkpoints" / dataset / "trials" / f"trial_{number}" / "best.pt.json"
    if sidecar.is_file():
        snapshot = json.loads(sidecar.read_text())
        state, cfg = snapshot["trainer_state"], snapshot["trainer_config"]
        if (snapshot.get("model_type") != "gpt2" or cfg.get("monitor") != "val/sft/generation_ndcg@10"
            or Path(cfg["checkpoint_dir"]).parts[-3:] != (dataset, "trials", f"trial_{number}")
            or state.get("epoch") != epoch or state.get("best_epoch") != epoch
            or f"{state['best_metric']:.6f}" != score[1]):
            raise InitialSelectionUnavailable(f"{sidecar} does not match the HPO winner in {path}")
        row.update(score=state["best_metric"], rounded=False, sidecar=sidecar)
    return row


def _tiger_winner_epoch(root, dirs, jobs, dataset, winner):
    trial, score = winner["trial"], winner["score"]
    for directory in dirs:
        for path in (directory / "trials" / trial / "result.json", directory / "result.json"):
            if path.is_file():
                data = json.loads(path.read_text())
                checkpoint = Path(data.get("best_checkpoint", ""))
                if (data.get("status") == "complete" and checkpoint.parts[-3:] == ("trials", trial, "best.pt")
                    and data.get("best_value") == score and data.get("monitor") == "val/sft/ndcg@10"):
                    return data["best_epoch"], path
    if jobs and Path(__file__).with_name("refit_global.py").is_file():
        from refit_global import epoch_from_run_logs
        found = epoch_from_run_logs(root, jobs, dataset, winner)
        if found is not None:
            return found
    raise InitialSelectionUnavailable(f"Missing best_epoch of winning {dataset}/{trial} (validation={score})")


def select_initial(root, model, dataset):
    """Select the highest *completed* single-trial global validation NDCG@10.

    Return native HPO parameters and a one-based training duration. A newer
    study can be partial. A weaker complete study or a final retraining run
    never supplies a missing epoch for the winner.
    """
    root = Path(root).resolve()
    if (model, dataset) not in supported_initial_pairs():
        raise InitialSelectionUnavailable(f"No global HPO protocol for {model}/{dataset}")
    study_name = _study_name(model, dataset)
    dirs, jobs = _initial_locations(root, model, dataset)
    candidates = []

    def record(row, path):
        if row is not None:
            if (not isinstance(row["params"], dict) or not row["params"]
                    or not isinstance(row["score"], (float, int)) or not 0 <= row["score"] <= 1):
                raise InitialSelectionUnavailable(f"Invalid HPO winner in {path}")
            candidates.append({**row, "evidence": [path, row["sidecar"]] if "sidecar" in row else [path]})

    for directory in dirs:
        config = directory / "study_config.json"
        if config.is_file() and not _eligible_protocol(json.loads(config.read_text()), model, dataset):
            continue
        journal = directory / "optuna_journal.log"
        # Archived exports include the published results even on login nodes
        # without Optuna. Live journals are authoritative over provisional
        # best_trial.json written by the objective before Optuna marks COMPLETE.
        if journal.is_file() and ((root / "artifacts") not in directory.parents or importlib.util.find_spec("optuna")):
            record(_journal_winner(journal, study_name), journal)
            continue
        for name in ("all_trials.json", "tuning_summary.json", "best_trial.json"):
            path = directory / name
            if path.is_file():
                record(_published_winner(path, model, dataset, study_name), path)
        if model == "tiger" and dataset == global_grid.Y500:
            # The compact grid has no Optuna journal. An individual result can
            # be fully committed before the aggregate winner is republished.
            for path in sorted((directory / "trials").glob("*/result.json")):
                data = json.loads(path.read_text())
                if data.get("status") == "complete" and data.get("monitor") == "val/sft/ndcg@10":
                    record(dict(trial=path.parent.name, params=data["hparams"], score=data["best_value"],
                                epoch=data.get("best_epoch"), seed=data.get("seed")), path)
    # Read per-job exports only when they supply the missing journal for an eligible study.
    # A pending trial can already have published a provisional winner in the same job.
    export_jobs = []
    for job in jobs:
        directory = job / "checkpoints" / dataset
        if (directory / "optuna_journal.log").is_file():
            continue
        config = directory / "study_config.json"
        if not config.is_file() or _eligible_protocol(json.loads(config.read_text()), model, dataset):
            export_jobs.append(job)
    export = root / "artifacts/global-time/exp2-tuned-params" / model
    for parent in (export, *(job / "artifacts" for job in export_jobs)):
        for suffix in ("tuning_summary", "best_trial"):
            path = parent / f"{model}_{dataset}_{suffix}.json"
            if path.is_file():
                record(_published_winner(path, model, dataset, study_name), path)
    if model == "gpt2":
        for job in export_jobs:
            path = job / "configs" / f"gpt2_model_{dataset}.yaml"
            if path.is_file():
                record(_exported_gpt2_winner(path, study_name), path)
    if not candidates:
        raise InitialSelectionUnavailable(f"No completed {study_name} trial found")
    # The YAML header rounds scores to six decimals; prefer exact records of
    # the same trial. Do not guess an ordering across indistinguishable exports.
    exact = [c for c in candidates if not c.get("rounded")]
    candidates = [c for c in candidates if not c.get("rounded") or not any(
        x["trial"] == c["trial"] and x["params"] == c["params"] and x["epoch"] == c["epoch"]
        and round(x["score"], 6) == c["score"] for x in exact)]
    best = max(candidates, key=lambda c: c["score"])
    same = []
    for candidate in candidates:
        equal = candidate["score"] == best["score"]
        close = abs(candidate["score"] - best["score"]) <= 0.0000005
        if close and (candidate.get("rounded") or best.get("rounded")) and candidate["params"] != best["params"]:
            raise InitialSelectionUnavailable(f"Indistinguishable rounded scores in {study_name}; copy the HPO JSONs")
        if equal and candidate["trial"] == best["trial"] and candidate["params"] == best["params"]:
            same.append(candidate)
    best["evidence"] = sorted({p for c in same for p in c["evidence"]})
    epochs = {c["epoch"] for c in same if c["epoch"] is not None}
    if len(epochs) > 1:
        raise InitialSelectionUnavailable(f"Conflicting best_epoch for {study_name}/{best['trial']}: {epochs}")
    epoch = next(iter(epochs), None)
    if epoch is None and model == "tiger":
        epoch, path = _tiger_winner_epoch(root, dirs, jobs, dataset, best)
        best["evidence"].append(path)
    if type(epoch) is not int or epoch < 0:
        raise InitialSelectionUnavailable(f"No valid best_epoch for {study_name}/{best['trial']}")
    seed = next((c["seed"] for c in same if c.get("seed") is not None), None)
    if seed is None:
        seed = 42 if model == "tiger" else 42 + int(best["trial"].removeprefix("trial_"))
    return dict(model=model, dataset=dataset, trial=best["trial"], params=best["params"], seed=seed,
                validation_ndcg10=best["score"], best_epoch=epoch, epochs=epoch + 1, study_name=study_name,
                sources=[str(path.relative_to(root)) for path in best["evidence"]])


def _tag(value):
    return f"{value:g}".replace(".", "p")


def catalog():
    """Yield the source identity and its exact directory for the full original experiment."""
    for dataset in (*LEGACY_DATASETS, *COMPACT_DATASETS):
        legacy = dataset in LEGACY_DATASETS
        beauty23 = dataset == "beauty_2023"
        for model in ("gpt2", "tiger"):
            grid = global_grid.cells(model, dataset)
            model_pair = Path(model) / dataset
            if dataset == global_grid.Y500:
                model_pair /= "sampled50"
            if beauty23:
                parent = Path("beauty2023-global-k3/gflow-sft") / model_pair
                ce_parent = Path("beauty2023-global-k3/ce") / model_pair
            elif legacy:
                parent = Path("gpt2-gflow-k3") / dataset
                ce_parent = Path("ce-distill-q-k10") / model_pair
            else:
                parent = Path("yambda-gflow-k3") / model_pair
                ce_parent = Path("yambda-ce-q-k3") / model_pair

            if not legacy or model == "gpt2":
                for cell in grid["sft"]:
                    path = parent / "sft_only" / f"seed_{cell['seed']}"
                    yield _source(path, cell)
                samplers = ("random", "policy", "teacher", "q") if legacy else ("teacher",)
                for original, sampler in product(grid["gflow"], samplers):
                    cell = {**original, "sampler": sampler}
                    power, rho, loss = cell["power"], cell["rho"], cell["objective"]
                    if legacy:
                        combo_dir = "sasrec_x_interaction_x_sim_hier"
                        if sampler != "teacher":
                            combo_dir += f"_neg-{sampler}"
                        variant = f"{loss}_k3" + (f"_rho{_tag(rho)}" if rho != 1 else "")
                        path = parent / combo_dir / "w1-1-1" / f"pow_{_tag(power)}" / variant
                    else:
                        path = parent / "gflow" / f"power_{power:g}" / f"rho_{rho:g}" / loss
                    yield _source(path / f"seed_{cell['seed']}", cell)

            ce_k = 10 if legacy else 3
            for power, rho in product(global_grid.REWARD_POWERS, global_grid.RHO_TARGETS):
                seed = grid["gflow"][0]["seed"]
                cell = dict(dataset=dataset, model=model, arm="ce", seed=seed, combo=COMBO,
                            power=power, rho=rho, num_samples=ce_k, num_negatives=ce_k,
                            estimator="sampled", sampling="reward_negatives", importance_correction=False)
                path = ce_parent / f"reward_negatives_n{ce_k}" / f"pow_{_tag(power)}" / f"rho_{_tag(rho)}"
                yield _source(path / f"seed_{seed}", cell)


def _source(path, cell):
    identity = dict(cell)
    identity["source_seed"] = identity.pop("seed")
    return {**identity, "experiment": path.parts[0], "source_id": path.as_posix()}


def _metric_error(metrics, expected_count=None):
    if not isinstance(metrics, dict):
        return "test metrics must be an object"
    for key in ("num_examples", *METRIC_KEYS):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            return f"missing or nonfinite test metric {key}"
    n = metrics["num_examples"]
    if n <= 0 or int(n) != n or (expected_count is not None and n != expected_count):
        return f"incomplete test split: num_examples={n}, expected={expected_count}"
    return None


def _ledger(path):
    """Read atomically published JSONL. Incomplete entries do not suppress a retry."""
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = row["run_key"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid refit metrics entry") from exc
            if key in rows:
                raise ValueError(f"{path}:{line_no}: duplicate refit metrics key {key}")
            rows[key] = row
    return rows


def _complete(row):
    if not row or row.get("status") != "completed":
        return False
    expected = (row.get("details") or {}).get("expected_test_examples")
    return _metric_error(row.get("test_metrics"), expected) is None


def has_metrics(metrics_path, run_key):
    return _complete(_ledger(Path(metrics_path)).get(run_key))


def _expected_result(source, result):
    """Different generations recorded different model/variant tags and ratio names."""
    if result.get("dataset") != source["dataset"] or result.get("seed") != source["source_seed"]:
        raise ValueError("source dataset or seed differs from its experiment directory")
    if result.get("model", source["model"]) != source["model"]:
        raise ValueError("source model differs from its experiment directory")
    checks = []
    if source["arm"] == "sft":
        if result.get("variant") != "sft_only":
            raise ValueError("expected an SFT-only result")
    else:
        combo = str(result.get("combo", "")).replace("sim_hier", "similarity")
        if combo != COMBO or result.get("power") != source["power"]:
            raise ValueError("source reward product or power differs from the requested grid")
        if result.get("rho", result.get("target_grad_ratio")) != source["rho"]:
            raise ValueError("source reward weight differs from the requested grid")
        if source["arm"] == "ce":
            checks = ["num_samples", "estimator", "sampling", "importance_correction"]
        else:
            checks = ["objective", "sampler", "num_negatives"]
    for key in checks:
        if result.get(key) != source[key]:
            raise ValueError(f"source {key}={result.get(key)!r}, expected {source[key]!r}")
    epochs = result.get("epochs_completed")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"invalid source epochs_completed={epochs!r}")
    return epochs


def _stat_checkpoint(path):
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise FileNotFoundError(f"missing or empty checkpoint: {path}")
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_json_with_hash(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _run_key(source_id, seed):
    return f"{source_id}/refit_seed_{seed}"


def plan_missing(experiments_root, metrics_path, seeds=REFIT_SEEDS):
    """Look up unscored source/seed combinations and return available tasks and inventory counts."""
    root = Path(experiments_root).resolve()
    ledger = _ledger(Path(metrics_path))
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected distinct refit seeds")
    stats = dict(total_sources=0, total_runs=0, already_scored=0, available_sources=0,
                 missing_results=0, unfinished_sources=0, missing_checkpoints=0, invalid_sources=[], planned_runs=0)
    tasks = []
    for source in catalog():
        stats["total_sources"] += 1
        stats["total_runs"] += len(seeds)
        key = source["source_id"]
        pending = [seed for seed in seeds if not _complete(ledger.get(_run_key(key, seed)))]
        stats["already_scored"] += len(seeds) - len(pending)
        if not pending:
            continue
        directory = root / key
        result_path = directory / "result.json"
        resolved_path = directory / "resolved_config.json"
        # All original families keep the validation-selected checkpoint here. Result files
        # copied from a cluster contain absolute paths under the original user's home.
        checkpoint = directory / "checkpoints/best.pt"
        if not result_path.is_file():
            stats["missing_results"] += 1
            continue
        try:
            result, result_sha = _read_json_with_hash(result_path)
            if result.get("status") != "completed":
                stats["unfinished_sources"] += 1
                continue
            epochs = _expected_result(source, result)
            try:
                checkpoint_stat = _stat_checkpoint(checkpoint)
            except FileNotFoundError:
                stats["missing_checkpoints"] += 1
                continue
            config, config_sha = _read_json_with_hash(resolved_path)
            if not isinstance(config, dict) or not config:
                raise ValueError("empty source training configuration")
            signature = dict(result_sha256=result_sha, config_sha256=config_sha, checkpoint=checkpoint_stat)
            for seed in seeds:
                row = ledger.get(_run_key(key, seed))
                if _complete(row) and row.get("source_signature") != signature:
                    raise ValueError("source artifacts changed since a sibling refit was recorded")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            stats["invalid_sources"].append({"source_id": key, "reason": str(exc)})
            continue
        stats["available_sources"] += 1
        for seed in pending:
            tasks.append({**source, "run_key": _run_key(key, seed), "seed": seed, "epochs": epochs,
                          "source_best_epoch": result.get("best_epoch"), "source_dir": str(directory),
                          "source_result": str(result_path), "source_resolved_config": str(resolved_path),
                          "source_checkpoint": str(checkpoint), "source_signature": signature})
    stats["planned_runs"] = len(tasks)
    return tasks, stats


def verify_source(task):
    """Catch replacement of an original run between planning and GPU execution."""
    result, result_sha = _read_json_with_hash(Path(task["source_result"]))
    config, config_sha = _read_json_with_hash(Path(task["source_resolved_config"]))
    signature = dict(result_sha256=result_sha, config_sha256=config_sha,
                     checkpoint=_stat_checkpoint(Path(task["source_checkpoint"])))
    if result.get("status") != "completed" or signature != task["source_signature"]:
        raise ValueError(f"Original checkpoint or configuration has changed: {task['source_id']}")
    return config


def append_metrics(metrics_path, task, test_metrics, *, details=None):
    """Commit one complete test evaluation, atomically across concurrent workers."""
    path = Path(metrics_path)
    details = dict(details or {})
    problem = _metric_error(test_metrics, details.get("expected_test_examples"))
    if problem:
        raise ValueError(problem)
    if task["run_key"] != _run_key(task["source_id"], task["seed"]):
        raise ValueError("refit run key differs from its source/seed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _ledger(path)
        if _complete(rows.get(task["run_key"])):
            return False
        config = verify_source(task)
        rows[task["run_key"]] = {**task, "status": "completed", "training_split": "train+validation",
                                 "source_finetune_config": config, "test_metrics": test_metrics, "details": details}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                             delete=False, encoding="utf-8") as output:
                temporary = Path(output.name)
                for row in rows.values():
                    output.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return True
