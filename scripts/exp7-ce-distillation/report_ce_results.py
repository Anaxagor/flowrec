#!/usr/bin/env python3
"""Export all completed CE test metrics and measured times to the best epoch.

Read-only for experiment files. Produces CSV, JSON, XLSX and a standalone HTML
table under --output-dir. ML packages and checkpoint weights are not loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    ("experiments/exp7-ce-distillation", "TIGER", ("beauty", "instruments")),
    ("experiments/exp7-ce-distillation-gpt2", "GPT-2", ("beauty", "instruments")),
    ("experiments/exp7-ce-distillation-tiger-yambda", "TIGER", ("yambda_50m_listens", "yambda_500m_listens")),
)
DEFAULT_COMBO = "sasrec_x_interaction_x_similarity"
METRICS = [f"{metric}@{k}" for k in (5, 10, 20) for metric in ("HR", "NDCG", "Coverage", "ARP")]
NOTES = [
    "Включены все завершённые CE-конфигурации с полным test, без выбора лучших по test. Исходные модели добавлены отдельными baseline-строками.",
    "Время до best — сумма epoch_time_sec до выбранной эпохи включительно. Эпоха выбирается по validation NDCG@5, а не по test; это практическая оценка сходимости, не доказательство математической сходимости.",
    "Эпохи в таблице пронумерованы с 1. Best=0 и статус init_kept означают, что лучшим остался начальный checkpoint: улучшения по validation не было. Нулевое время до него не означает, что обучение заняло 0 минут.",
    "Время до best включает обучение, validation в конце эпох и сохранение checkpoints; исключает подготовку данных, калибровку, начальную validation, финальную оценку и ожидание Slurm.",
    "Fit — время trainer.fit вместе с созданием train loader и начальной validation, до фактической остановки. Setup и финальная оценка указаны отдельно. Построение reward caches и очередь Slurm не входят в эти времена.",
    "Для возобновлённых или неполных историй оценка времени до best не выдаётся; время fit относится только к последнему вызову. Исключение — исходный checkpoint, для которого время до best равно 0 по определению.",
    "Режим exact определяется по estimator=exact: семплирование в нём не используется, даже если исходный result.json содержит sampling=reward. В таблице и сводке времени exact выделен отдельно.",
    "curriculum_smooth_is оценивает CE по полному q: proposal содержит примесь epsilon*q, а loss использует сырые веса q/proposal. Старые curriculum и curriculum_smooth оценивают CE по текущему proposal без importance correction.",
    "HR, NDCG и Coverage в CSV/JSON/XLSX сохранены как доли. HTML и Excel отображают Coverage в процентах; ARP — средняя популярность рекомендованных items, не процент.",
    "Для сопоставления времени учитывайте GPU, batch, accumulation и длину истории. Поля конфигурации и ссылки на исходные файлы сохранены для каждой строки; время сходимости не выбиралось по test.",
    "Файлы — локальная копия результатов. Отсутствие result.json не определяет текущее состояние задачи в Slurm.",
]


def column(key, label, group="Конфигурация", fmt="text"):
    return {"key": key, "label": label, "group": group, "format": fmt}


COLUMNS = [
    column("dataset", "Датасет"), column("model", "Модель"),
    column("sampling", "Режим CE"), column("combo", "Награда"),
    column("power", "Power", fmt="decimal"), column("rho", "Rho", fmt="decimal"),
    column("seed", "Seed", fmt="integer"), column("num_samples", "N выборок", fmt="integer"),
    column("curriculum_epochs", "Переход, эпох", fmt="decimal"),
    column("proposal_q_mix", "Примесь q (epsilon)", fmt="decimal"),
    column("lambda_distill", "Вес CE", fmt="decimal"),
    *[column(key, key, "Test @" + key.split("@")[1], "percent" if key.startswith("Coverage") else "decimal")
      for key in METRICS],
    column("test_num_examples", "Примеров test", "Test", "integer"),
    column("best_epoch", "Лучшая эпоха (0 = init)", "Время", "integer"),
    column("best_time_min", "До best, мин", "Время", "minutes"),
    column("epochs_completed", "Эпох выполнено", "Время", "integer"),
    column("fit_time_min", "Fit до остановки, мин", "Время", "minutes"),
    column("setup_time_min", "Подготовка запуска, мин", "Время", "minutes"),
    column("evaluation_time_min", "Финальная оценка, мин", "Время", "minutes"),
    column("run_time_min", "Всего в вызове, мин", "Время", "minutes"),
    column("best_status", "Статус best", "Время"),
    column("timing_status", "Полнота времени", "Время"),
    column("estimator", "Оценка CE", "Протокол"),
    column("ce_target", "Цель CE", "Протокол"),
    column("importance_correction", "Поправка q/proposal", "Протокол"),
    column("calibration_sampling", "Семплирование калибровки", "Протокол"),
    column("evaluation_scope", "Объём оценки", "Протокол"),
    column("gpu", "GPU", "Протокол"),
    column("batch_size", "Batch", "Протокол", "integer"),
    column("grad_accum", "Accumulation", "Протокол", "integer"),
    column("early_stopped", "Early stopping", "Протокол"),
    column("resumed_from_epoch", "Resume с эпохи (0-based)", "Протокол", "integer"),
    column("row_type", "Тип строки", "Источник"),
    column("source_result", "Файл результата", "Источник"),
    column("source_history", "Файл истории", "Источник"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional_json(path):
    return read_json(path) if path.is_file() else {}


def relative(path):
    return os.path.relpath(path, ROOT)


def check_test_metrics(result, path):
    metrics = result.get("splits", {}).get("test", {})
    for key in [*METRICS, "num_examples"]:
        if key not in metrics or not isinstance(metrics[key], (int, float)) or not math.isfinite(metrics[key]):
            raise ValueError(f"Missing/nonfinite test {key}: {path}")
    return metrics


def time_to_best(result, history):
    best = int(result["best_epoch"])
    if best == -1:
        return 0.0, "init_kept", "initial_checkpoint"
    if result.get("resumed_from_epoch") is not None:
        return None, "validation_best", "unavailable_after_resume"
    epochs = [row for row in history if row.get("epoch", -1) >= 0]
    expected = list(range(int(result["epochs_completed"])))
    if [row["epoch"] for row in epochs] != expected:
        return None, "validation_best", "unavailable_incomplete_history"
    if best not in expected or any(not isinstance(row.get("epoch_time_sec"), (int, float))
        or not math.isfinite(row["epoch_time_sec"]) or row["epoch_time_sec"] <= 0 for row in epochs):
        return None, "validation_best", "unavailable_epoch_times"
    return sum(row["epoch_time_sec"] for row in epochs if row["epoch"] <= best) / 60, "validation_best", "complete_epoch_times"


def run_row(path, result, model, output_dir):
    test = check_test_metrics(result, path)
    history_path = path.parent / "training_history.json"
    history = optional_json(history_path).get("history", [])
    best_time, status, timing_status = time_to_best(result, history)
    resolved = optional_json(path.parent / "resolved_config.json")
    identity = optional_json(path.parent / "identity.json") or resolved.get("identity", {})
    config = identity.get("config", {})
    loader = config.get("dataloader", {})
    trainer = identity.get("trainer", {})
    durations = [result.get(key) for key in ("setup_sec_this_invocation", "fit_sec_this_invocation", "evaluation_sec")]
    duration_minutes = [v / 60 if v is not None else None for v in durations]
    return {
        "row_type": "run", "dataset": result["dataset"], "model": model,
        "sampling": "exact" if result["estimator"] == "exact" else result.get("sampling", "reward"),
        "combo": result.get("combo", DEFAULT_COMBO),
        "power": result["power"], "rho": result["rho"], "seed": result["seed"],
        "num_samples": result.get("num_samples"), "curriculum_epochs": result.get("curriculum_epochs"),
        "proposal_q_mix": result.get("proposal_q_mix"),
        "ce_target": result.get("ce_target", "q" if result.get("sampling", "reward") in {"reward", "curriculum_smooth_is"} else "curriculum_proposal"),
        "importance_correction": result.get("importance_correction", result.get("sampling") == "curriculum_smooth_is"),
        "calibration_sampling": result.get("calibration_sampling", "reward" if result.get("sampling") == "curriculum_smooth_is" else result.get("sampling", "reward")),
        "lambda_distill": result.get("lambda_distill"), "estimator": result["estimator"],
        **{key: test[key] for key in METRICS}, "test_num_examples": int(test["num_examples"]),
        "best_epoch": int(result["best_epoch"]) + 1, "best_time_min": best_time,
        "epochs_completed": result["epochs_completed"], "best_status": status, "timing_status": timing_status,
        "setup_time_min": duration_minutes[0], "fit_time_min": duration_minutes[1],
        "evaluation_time_min": duration_minutes[2],
        "run_time_min": sum(duration_minutes) if all(v is not None for v in duration_minutes) else None,
        "early_stopped": result.get("early_stopped"), "resumed_from_epoch": result.get("resumed_from_epoch"),
        "evaluation_scope": result.get("evaluation_scope", "unknown"), "gpu": resolved.get("gpu"),
        "batch_size": loader.get("train_batch_size", loader.get("batch_size")),
        "grad_accum": trainer.get("gradient_accumulation_steps"),
        "source_result": relative(path), "source_history": relative(history_path),
        "source_link": os.path.relpath(path, output_dir),
    }


def baseline_row(path, model, output_dir):
    result = read_json(path)
    test = check_test_metrics(result, path)
    return {"row_type": "baseline", "dataset": result["dataset"], "model": model,
            "sampling": "baseline", "combo": "initial_exp2", "estimator": "baseline",
            **{key: test[key] for key in METRICS}, "test_num_examples": int(test["num_examples"]),
            "best_status": "reference", "timing_status": "not_applicable",
            "evaluation_scope": "full" if result.get("identity", {}).get("max_batches") is None else "limited",
            "source_result": relative(path), "source_history": None,
            "source_link": os.path.relpath(path, output_dir)}


def collect(output_dir):
    rows, availability, skipped = [], [], []
    for location, model, known_datasets in DEFAULT_ROOTS:
        root = ROOT / location
        completed_paths = set()
        for path in sorted(root.rglob("result.json")) if root.is_dir() else []:
            result = read_json(path)
            if result.get("status") != "completed" or result.get("evaluation_scope") != "full":
                skipped.append(relative(path))
                continue
            rows.append(run_row(path, result, model, output_dir))
            completed_paths.add(path.parent)
        for dataset in known_datasets:
            directory = root / dataset
            baseline = directory / "baseline/init_metrics.json"
            if baseline.is_file():
                rows.append(baseline_row(baseline, model, output_dir))
            initialized = [p.parent for p in directory.rglob("identity.json")] if directory.is_dir() else []
            current = [row for row in rows if row["model"] == model and row["dataset"] == dataset and row["row_type"] == "run"]
            unfinished = [path for path in initialized if path not in completed_paths]
            availability.append({"model": model, "dataset": dataset, "completed": len(current),
                                 "initialized_without_full_result": len(unfinished),
                                 "root": location, "unfinished_paths": [relative(p) for p in unfinished]})
    order = {"baseline": 0, "reward": 1, "curriculum": 2, "curriculum_smooth": 3, "curriculum_smooth_is": 4, "exact": 5}
    rows.sort(key=lambda row: (row["dataset"], row["model"], order.get(row["sampling"], 9), row["combo"],
                              row.get("power", 0), -row.get("rho", 0), row.get("seed", 0)))
    return rows, availability, skipped


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["row_type"] == "run":
            groups[(row["model"], row["dataset"], row["sampling"])].append(row)
    summaries = []
    for (model, dataset, sampling), group in sorted(groups.items()):
        improved = [row for row in group if row["best_status"] == "validation_best" and row["best_time_min"] is not None]
        times = [row["best_time_min"] for row in improved]
        fit = [row["fit_time_min"] for row in group if row["fit_time_min"] is not None]
        summaries.append({"model": model, "dataset": dataset, "sampling": sampling, "runs": len(group),
            "init_kept": sum(row["best_status"] == "init_kept" for row in group),
            "best_time_min_min": min(times) if times else None, "best_time_min_median": statistics.median(times) if times else None,
            "best_time_min_max": max(times) if times else None,
            "fit_time_min_min": min(fit) if fit else None, "fit_time_min_median": statistics.median(fit) if fit else None,
            "fit_time_min_max": max(fit) if fit else None})
    return summaries


def write_csv(rows, columns, path):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[col["key"] for col in columns], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(rows, metadata, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    table_index = 0

    def data_sheet(name, subset, columns):
        nonlocal table_index
        sheet = workbook.create_sheet(name)
        sheet.append([col["label"] for col in columns])
        for row in subset:
            sheet.append([row.get(col["key"]) for col in columns])
        sheet.freeze_panes = "E2" if len(columns) > 5 else "A2"
        sheet.sheet_view.zoomScale = 85
        sheet.row_dimensions[1].height = 42
        for index, col in enumerate(columns, 1):
            key = col["key"]
            sheet.column_dimensions[get_column_letter(index)].width = (
                32 if key in {"combo", "gpu"} else 27 if key in {"sampling", "best_status", "timing_status"}
                else 55 if key.startswith("source_") else 22 if key == "dataset" else 16)
            header = sheet.cell(1, index)
            header.fill = PatternFill("solid", fgColor="18384D")
            header.font = Font(color="FFFFFF", bold=True)
            header.alignment = Alignment(wrap_text=True, vertical="center")
            for row_index, row in enumerate(subset, 2):
                cell = sheet.cell(row_index, index)
                cell.alignment = Alignment(vertical="center")
                cell.number_format = {"percent": "0.00%", "decimal": "0.000000", "minutes": "0.0", "integer": "0"}.get(col["format"], "General")
                if key in {"power", "rho", "curriculum_epochs"}:
                    cell.number_format = "0.##"
                if row.get("row_type") == "baseline":
                    cell.fill = PatternFill("solid", fgColor="E4EDF4")
                    cell.font = Font(bold=True)
                elif row.get("best_status") == "init_kept" and key in {"best_epoch", "best_time_min", "best_status"}:
                    cell.fill = PatternFill("solid", fgColor="FFF1D4")
                if key in {"source_result", "source_history"} and row.get(key):
                    cell.hyperlink = os.path.relpath(ROOT / row[key], path.parent)
                    cell.style = "Hyperlink"
        if subset:
            table_index += 1
            table = Table(displayName=f"Results{table_index}", ref=f"A1:{get_column_letter(len(columns))}{sheet.max_row}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
            sheet.add_table(table)
        return sheet

    data_sheet("Все метрики", rows, COLUMNS)
    for dataset in sorted({row["dataset"] for row in rows}):
        data_sheet(dataset[:31], [row for row in rows if row["dataset"] == dataset], COLUMNS)
    time_keys = {"dataset", "model", "sampling", "combo", "power", "rho", "seed", "proposal_q_mix", "best_epoch", "best_time_min",
                 "epochs_completed", "fit_time_min", "setup_time_min", "evaluation_time_min", "run_time_min",
                 "best_status", "timing_status", "gpu", "source_result", "source_history"}
    data_sheet("Время до best", [row for row in rows if row["row_type"] == "run"], [c for c in COLUMNS if c["key"] in time_keys])
    data_sheet("Доступность", metadata["availability"], [
        column("model", "Модель"), column("dataset", "Датасет"), column("completed", "Готовых test", fmt="integer"),
        column("initialized_without_full_result", "Начато без полного test", fmt="integer"), column("root", "Корень результатов")])
    notes = workbook.create_sheet("Как читать")
    notes.column_dimensions["A"].width = 115
    notes.append(["CE: все тестовые метрики и время до лучшего validation checkpoint"])
    notes.append(["Дата снимка: " + metadata["generated_at"]])
    notes.append([f"Завершённых конфигураций: {metadata['counts']['completed_runs']}; исходных моделей: {metadata['counts']['baselines']}."])
    for note in metadata["notes"]:
        notes.append([note])
    notes.append(["Поля CSV / Excel"])
    for col in COLUMNS:
        notes.append([f"{col['key']} — {col['label']}"])
    for cell in notes["A"]:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        notes.row_dimensions[cell.row].height = 45 if cell.row <= len(metadata["notes"]) + 3 else 28
    notes["A1"].font = Font(bold=True, size=14)
    workbook.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", default=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"),
                        help="User-facing snapshot date/time in Europe/Moscow")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, availability, skipped = collect(output)
    counts = Counter(row["row_type"] for row in rows)
    protocol_note = "Параметры включённых запусков: " + "; ".join(
        f"{key}=" + ", ".join(map(str, sorted({row[key] for row in rows if row['row_type'] == 'run' and row.get(key) is not None}, key=str)))
        for key in ("seed", "num_samples", "curriculum_epochs", "gpu", "batch_size", "grad_accum")) + "."
    metadata = {"generated_at": args.as_of, "timezone": "Europe/Moscow", "notes": [*NOTES, protocol_note],
                "counts": {"completed_runs": counts["run"], "baselines": counts["baseline"]},
                "availability": availability, "skipped_results": skipped,
                "missing_results": [f"{x['model']} / {x['dataset']}: нет завершённых полных test-результатов"
                                    for x in availability if x["completed"] == 0],
                "timing_summary": summarize(rows)}
    write_csv(rows, COLUMNS, output / "test_metrics.csv")
    write_xlsx(rows, metadata, output / "test_metrics.xlsx")
    (output / "test_metrics.json").write_text(json.dumps({"metadata": metadata, "columns": COLUMNS, "rows": rows},
                                                       ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from report_ce_html import render_html
    render_html(rows, metadata, COLUMNS, output / "test_metrics.html")
    print(json.dumps({"output_dir": str(output), **metadata["counts"], "timing_summary": metadata["timing_summary"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
