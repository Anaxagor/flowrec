#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RUNS = [
    (
        "mlp_logz",
        PROJECT_ROOT / "configs" / "tiger_beauty_global_time_gflow_tb_random_w001_mlp_logz.json",
    ),
    (
        "vector_logz",
        PROJECT_ROOT / "configs" / "tiger_beauty_global_time_gflow_tb_random_w001_vector_logz.json",
    ),
]


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = (
        PROJECT_ROOT
        / "experiments"
        / "tiger_sweeps"
        / f"{stamp}_tiger_beauty_global_time_gflow_logz_ablation_2run"
    )
    sweep_dir.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    for name, config_path in RUNS:
        log_path = sweep_dir / f"{name}.launch.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_tiger.py"),
            "--config",
            str(config_path),
            "--run-name",
            f"tiger_beauty_global_time_gflow_{name}",
            "--seed",
            "42",
            "--experiments-dir",
            str(sweep_dir),
        ]
        record: dict[str, Any] = {
            "name": name,
            "config_path": str(config_path),
            "command": cmd,
            "launch_log": str(log_path),
            "status": "running",
        }
        records.append(record)
        write_summary(sweep_dir, records)

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(cmd) + "\n\n")
            log_file.flush()
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        record["returncode"] = result.returncode
        summary_path = find_latest_summary(sweep_dir, name)
        if summary_path is not None:
            record.update(load_run_record(summary_path))
        record["status"] = "completed" if result.returncode == 0 and summary_path is not None else "failed"
        write_summary(sweep_dir, records)
        if record["status"] == "failed":
            return result.returncode or 1

    write_summary(sweep_dir, records)
    return 0


def find_latest_summary(sweep_dir: Path, name: str) -> Path | None:
    candidates = sorted(
        sweep_dir.glob(f"*{name}*/final_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_run_record(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    return {
        "summary_path": str(summary_path),
        "run_dir": str(summary_path.parent),
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_completed": summary.get("epochs_completed"),
        "steps_completed": summary.get("steps_completed"),
        "validation_metrics": summary.get("validation_metrics", {}),
        "test_metrics": summary.get("test_metrics", {}),
    }


def write_summary(sweep_dir: Path, records: list[dict[str, Any]]) -> None:
    payload = {"runs": records}
    (sweep_dir / "sweep_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Beauty GFlow LogZ Ablation",
        "",
        "| run | status | best val | best epoch | test ndcg@20 | test recall@20 | test coverage@20 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        test = record.get("test_metrics", {})
        lines.append(
            "| {name} | {status} | {best} | {epoch} | {ndcg} | {recall} | {coverage} |".format(
                name=record["name"],
                status=record["status"],
                best=format_value(record.get("best_metric")),
                epoch=format_value(record.get("best_epoch")),
                ndcg=format_value(test.get("test/sft/generation_ndcg@20")),
                recall=format_value(test.get("test/sft/generation_recall@20")),
                coverage=format_value(test.get("test/sft/generation_coverage@20")),
            )
        )
    lines.append("")
    (sweep_dir / "sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
