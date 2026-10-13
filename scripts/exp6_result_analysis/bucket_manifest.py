"""Fixed checkpoint selection from GFlow+Recsys updated (5).xlsx.

No checkpoint discovery or alternative experiment families: missing files remain
in the manifest so the same manifest can be used on the cluster. Additional
seeds request SFT controls only; the workbook selected GFlow runs at seed 42.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


PROVENANCE_PATH = Path(__file__).with_name("bucket_selection_provenance.json")
DATASETS = ("beauty", "instruments", "yambda_50m_listens", "yambda_500m_listens")
GPT_ROOT = "experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix"
METRICS_ROOT = "artifacts/exp4-reward_ablation"
# label, workbook row, CSV row (header is row 1), combo, power, loss, rho, relative run path
SELECTED = {
    "beauty": (
        ("best_ndcg", 28, 111, "sasrec", 4., "db", 1., "sasrec/w1/pow_4/db_k10"),
        ("best_coverage", 29, 138, "sasrec_x_interaction", 16., "tb", .1,
         "sasrec_x_interaction/w1-1/pow_16/tb_k10_rho0p1"),
        ("pareto", 30, 111, "sasrec", 4., "db", 1., "sasrec/w1/pow_4/db_k10"),
    ),
    "instruments": (
        ("best_ndcg", 12, 309, "interaction_x_sim_hier", 1., "db", 1.,
         "interaction_x_sim_hier/w1-1/pow_1/db_k10"),
        ("best_coverage", 13, 328, "interaction_x_sim_hier", 16., "tb", .1,
         "interaction_x_sim_hier/w1-1/pow_16/tb_k10_rho0p1"),
        ("pareto", 14, 557, "sim_hier", 16., "subtb", 1., "sim_hier/w1/pow_16/subtb_k10"),
    ),
    "yambda_50m_listens": (
        ("best_ndcg", 47, 17, "sasrec", 1., "db", 1., "sasrec/pow_1/db_k10"),
        ("best_coverage", 48, 81, "sasrec_x_interaction_x_similarity", 16., "subtb", 1.,
         "sasrec_x_interaction_x_similarity/pow_16/subtb_k10"),
        ("pareto", 49, 9, "sasrec", .25, "subtb", 1., "sasrec/pow_0p25/subtb_k10"),
    ),
    "yambda_500m_listens": (
        ("best_ndcg", 56, 92, "sasrec", 4., "db", 1., "sasrec/pow_4/db_k10"),
        ("best_coverage", 57, 126, "sasrec_x_interaction", 16., "subtb", 1.,
         "sasrec_x_interaction/pow_16/subtb_k10"),
        ("pareto", 58, 163, "sasrec_x_similarity", .25, "db", 1.,
         "sasrec_x_similarity/pow_0p25/db_k10"),
    ),
}
BASELINE_ROWS = {"beauty": (26, 27), "instruments": (10, 11),
                 "yambda_50m_listens": (41, 43), "yambda_500m_listens": (53, 55)}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _portable(value: str | Path) -> str:
    """Strip a recorded cluster checkout prefix, preserving the experiment."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    for marker in ("checkpoints", "experiments", "datasets", "configs", "artifacts", "data"):
        if marker in path.parts:
            return Path(*path.parts[path.parts.index(marker):]).as_posix()
    return str(path)


def _metrics(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()
            if key.startswith(("HR@", "NDCG@", "Coverage@", "ARP@")) and value not in (None, "")}


def _validate_evidence(spec: dict[str, Any], resolved: dict[str, Any], path: Path) -> None:
    if not resolved:
        return
    for field in ("init_checkpoint", "sft_checkpoint"):
        if resolved.get(field) and _portable(resolved[field]) != spec["init_checkpoint"]:
            raise ValueError(f"{path}: unexpected initial checkpoint {resolved[field]}")
    for field in ("dataset", "seed"):
        if field in resolved and resolved[field] != spec[field]:
            raise ValueError(f"{path}: unexpected {field}: {resolved[field]}")
    if spec["model"] == "gpt2":
        if not resolved.get("cli_args", {}).get("prefix_from_dense", False):
            raise ValueError(f"{path}: expected full-history dense-prefix GPT-2 protocol")
    if spec["stage"] != "gflow":
        return
    expected = spec["selection_parameters"]
    for field, keys in (("combo", ("combo",)), ("objective", ("objective",)),
                        ("power", ("power", "reward_power")),
                        ("rho", ("rho_target", "target_grad_ratio")),
                        ("num_negatives", ("num_negatives",))):
        actual = next((resolved[k] for k in keys if k in resolved), None)
        if actual is not None and actual != expected[field]:
            raise ValueError(f"{path}: selected {field}={expected[field]}, recorded {actual}")
    if spec["model"] == "tiger":
        data = resolved.get("data", {})
        random_n = 15 if spec["dataset"] == "yambda_500m_listens" else None
        if data.get("datamodule_train_last_n") != 128 or data.get("finetune_per_user_random_n") != random_n:
            raise ValueError(f"{path}: expected matched last-128 / random-{random_n} protocol")
        if random_n and data.get("finetune_subset_seed") != 42:
            raise ValueError(f"{path}: expected fixed random-target subset seed 42")


def _attach_evidence(spec: dict[str, Any], repo_root: Path) -> None:
    resolved_path = repo_root / spec["resolved_config"] if spec.get("resolved_config") else None
    resolved = _json(resolved_path) if resolved_path else {}
    _validate_evidence(spec, resolved, resolved_path)
    result = _json(repo_root / spec["result_path"]) if spec.get("result_path") else {}
    if result.get("status") not in (None, "completed"):
        raise ValueError(f"Incomplete selected run: {spec['result_path']}")
    recorded = _metrics(result.get("splits", {}).get("test", result.get("test", {})))
    source = repo_root / spec["source_metrics_csv"]
    matching_rows = []
    if source.is_file():
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("dataset") != spec["dataset"] or row.get("split") != "test":
                    continue
                if spec["stage"] == "init":
                    match = row.get("variant") == "init"
                else:
                    match = bool(row.get("run_dir")) and _portable(row["run_dir"]) == spec["run_dir"]
                if match:
                    matching_rows.append(row)
        if len(matching_rows) > 1:
            raise ValueError(f"Ambiguous selected test metrics in {source}: {spec['run_id']}")
        if matching_rows:
            if spec["stage"] == "gflow":
                row = matching_rows[0]
                for field, column in (("combo", "combo"), ("objective", "objective"),
                                      ("power", "power"), ("rho", "target_grad_ratio"),
                                      ("num_negatives", "num_negatives")):
                    expected = spec["selection_parameters"][field]
                    actual = row.get(column)
                    if actual not in (None, ""):
                        actual = float(actual) if isinstance(expected, (int, float)) else actual
                        if actual != expected:
                            raise ValueError(f"{source}: selected {field}={expected}, recorded {actual}")
            recorded = recorded or _metrics(matching_rows[0])
    spec["recorded_test_metrics"] = recorded
    spec["available"] = (repo_root / spec["checkpoint"]).is_file()
    spec["evidence_available"] = bool(resolved or result or matching_rows)


def build_manifest(repo_root: Path, seeds: Sequence[int] = (42,)) -> list[dict[str, Any]]:
    """Return 19 specs by default: four init, four SFT, eleven unique GFlow.

    ``seeds`` selects SFT controls; seed 42 is mandatory for paired comparisons
    against the table's fixed GFlow selection. Missing weights are retained.
    All saved paths are relative to the caller's repository root.
    """
    if not seeds or 42 not in seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique and include 42 (the workbook GFlow seed)")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    repo_root = Path(repo_root)
    provenance = _json(PROVENANCE_PATH)
    output = []
    for dataset in DATASETS:
        model = "gpt2" if dataset in ("beauty", "instruments") else "tiger"
        is500 = dataset == "yambda_500m_listens"
        gflow_root = (f"{GPT_ROOT}/{dataset}" if model == "gpt2" else
                      f"experiments/exp4-gflow-yambda{'500' if is500 else '50'}m-matched" + ("-r15" if is500 else ""))
        sft_root = (gflow_root if model == "gpt2" else
                    "experiments/exp2-sft-finetune-matched" + ("-r15" if is500 else "") + f"/{dataset}")
        init = (f"checkpoints/exp2-tuned_params/initial/gpt2/{dataset}/best_model.pt" if model == "gpt2" else
                "checkpoints/exp2-tuned_params/initial/tiger/" +
                ("yambda_500m_listens/best.pt" if is500 else "yambda_50m/checkpoints/best.pt"))
        data_config = (f"{gflow_root}/data_reward_config.json" if model == "gpt2" else
                       f"{sft_root}/sft_lr3e-05/seed_42/run_config.json")
        source_csv = f"{METRICS_ROOT}/" + (
            "gflow_finetune_gpt2_amazon_fullhist_dense_prefix_metrics_summary.csv" if model == "gpt2" else
            f"gflow_finetune_yambda_{'500' if is500 else '50'}m_matched" + ("_r15" if is500 else "") + "_metrics_summary.csv")
        baseline = f"{model}_{dataset}_init"
        common = {"model": model, "dataset": dataset, "split": "test",
                  "baseline_run_id": baseline, "shared_initial_weights": True,
                  "init_checkpoint": init, "data_config": data_config,
                  "best_trial": f"checkpoints/exp2-tuned_params/initial/gpt2/{dataset}/best_trial.json" if model == "gpt2" else None,
                  "source_metrics_csv": source_csv, "source_family": gflow_root,
                  "source_workbook": provenance["workbook_name"],
                  "source_workbook_sha256": provenance["sha256"],
                  "selection_sheet": provenance["sheet"],
                  "selection_provenance": "scripts/exp6_result_analysis/bucket_selection_provenance.json"}
        init_row, sft_row = BASELINE_ROWS[dataset]
        initial = {**common, "run_id": baseline, "stage": "init", "seed": None,
                   "sft_run_id": None, "checkpoint": init, "run_dir": None,
                   "resolved_config": None, "result_path": f"{gflow_root}/baseline/init_metrics.json",
                   "selection_labels": ["init"], "selection_cells": [f"B{init_row}"],
                   "protocol_note": "One shared tuned checkpoint; workbook init aggregates must not replace this checkpoint's own metrics."}
        _attach_evidence(initial, repo_root)
        output.append(initial)
        for seed in seeds:
            run_dir = f"{sft_root}/{'sft_only' if model == 'gpt2' else 'sft_lr3e-05'}/seed_{seed}"
            sft_id = f"{model}_{dataset}_sft_seed{seed}"
            spec = {**common, "run_id": sft_id, "stage": "sft", "seed": seed,
                    "sft_run_id": sft_id, "checkpoint": f"{run_dir}/checkpoints/best.pt",
                    "run_dir": run_dir, "resolved_config": f"{run_dir}/resolved_config.json",
                    "result_path": f"{run_dir}/{'result' if model == 'gpt2' else 'metrics'}.json",
                    "selection_labels": ["sft_only"], "selection_cells": [f"B{sft_row}"],
                    "protocol_note": "Matched SFT-only control; same shared initial weights and held-out examples as GFlow."}
            if model == "tiger":
                spec["data_config"] = f"{run_dir}/run_config.json"
                # Both copies belong to this exact matched run. Choose the
                # promoted copy only if its run-directory copy is absent.
                promoted = ("checkpoints/exp2-tuned_params/finetuned_matched" + ("_r15" if is500 else "") +
                            f"/tiger/{dataset}/sft_only_lr3e-05/seed_{seed}/best.pt")
                spec["checkpoint_alternatives"] = [promoted]
                if not (repo_root / spec["checkpoint"]).is_file() and (repo_root / promoted).is_file():
                    spec["checkpoint"] = promoted
            _attach_evidence(spec, repo_root)
            output.append(spec)
        by_path = {}
        for label, row, csv_row, combo, power, loss, rho, suffix in SELECTED[dataset]:
            run_dir = f"{gflow_root}/{suffix}/seed_42"
            if run_dir in by_path:
                by_path[run_dir]["selection_labels"].append(label)
                by_path[run_dir]["selection_cells"].append(f"B{row}")
                continue
            spec = {**common, "run_id": f"{model}_{dataset}_gflow_{label}_seed42",
                    "stage": "gflow", "seed": 42,
                    "sft_run_id": f"{model}_{dataset}_sft_seed42",
                    "checkpoint": f"{run_dir}/checkpoints/best.pt", "run_dir": run_dir,
                    "resolved_config": f"{run_dir}/resolved_config.json", "result_path": f"{run_dir}/result.json",
                    "selection_labels": [label], "selection_cells": [f"B{row}"],
                    "source_csv_row_at_selection": csv_row,
                    "selection_parameters": {"combo": combo, "power": power, "objective": loss,
                                             "rho": rho, "num_negatives": 10},
                    "protocol_note": "Fixed workbook GFlow selection, seed 42; selected by test metrics, so bucket analysis is exploratory."}
            _attach_evidence(spec, repo_root)
            if not spec["recorded_test_metrics"]:
                cells = provenance["rows"][str(row)]["cells"]
                columns = ("G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R")
                metrics = ("HR@5", "NDCG@5", "Coverage@5", "HR@10", "NDCG@10", "Coverage@10",
                           "HR@20", "NDCG@20", "Coverage@20", "ARP@5", "ARP@10", "ARP@20")
                spec["recorded_test_metrics"] = {name: float(cells[f"{column}{row}"])
                                                  for column, name in zip(columns, metrics)}
                spec["recorded_metrics_source"] = "workbook fixed seed-42 row"
            by_path[run_dir] = spec
            output.append(spec)
    return output
