#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TIGER seed sweep and aggregate metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiger_beauty_global_time_sft.json",
        help="Base JSON config passed to scripts/run_tiger.py.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "tiger_sweeps",
        help="Parent folder for sweep directories.",
    )
    parser.add_argument("--sweep-dir", type=Path, default=None, help="Resume or write into this sweep dir.")
    parser.add_argument("--sweep-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Debug override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug override.")
    parser.add_argument("--force", action="store_true", help="Rerun seeds even if a summary exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep launching remaining seeds if one training run fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_json(config_path)

    sweep_dir = resolve_sweep_dir(args, config)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    run_name_base = str(config.get("experiment_name", config_path.stem))
    launched: list[dict[str, Any]] = []

    for seed in args.seeds:
        existing = None if args.force else find_existing_summary(sweep_dir, seed)
        if existing is not None:
            launched.append(load_run_record(seed, existing, status="skipped_existing"))
            continue

        run_name = f"{run_name_base}_seed-{seed}"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_tiger.py"),
            "--config",
            str(config_path),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--experiments-dir",
            str(sweep_dir),
        ]
        if args.device is not None:
            cmd.extend(["--device", args.device])
        if args.max_epochs is not None:
            cmd.extend(["--max-epochs", str(args.max_epochs)])
        if args.max_steps is not None:
            cmd.extend(["--max-steps", str(args.max_steps)])

        log_path = sweep_dir / f"seed_{seed}.launch.log"
        record: dict[str, Any] = {
            "seed": seed,
            "command": cmd,
            "launch_log": str(log_path),
            "status": "dry_run" if args.dry_run else "running",
        }
        launched.append(record)

        if args.dry_run:
            print(" ".join(cmd))
            continue

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
        summary_path = find_existing_summary(sweep_dir, seed)
        if result.returncode == 0 and summary_path is not None:
            record.update(load_run_record(seed, summary_path, status="completed"))
        else:
            record["status"] = "failed"
            if summary_path is not None:
                record["summary_path"] = str(summary_path)
                record["run_dir"] = str(summary_path.parent)
            write_sweep_outputs(sweep_dir, config_path, args.seeds, launched)
            if not args.continue_on_error:
                return result.returncode or 1

        write_sweep_outputs(sweep_dir, config_path, args.seeds, launched)

    write_sweep_outputs(sweep_dir, config_path, args.seeds, launched)
    failed = [run for run in launched if run.get("status") == "failed"]
    return 1 if failed else 0


def resolve_sweep_dir(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    if args.sweep_dir is not None:
        return resolve_path(args.sweep_dir)

    sweep_name = args.sweep_name or f"{config.get('experiment_name', 'tiger')}_seed_sweep"
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in sweep_name).strip("-")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return resolve_path(args.experiments_root) / f"{stamp}_{safe_name}"


def load_run_record(seed: int, summary_path: Path, *, status: str) -> dict[str, Any]:
    summary = load_json(summary_path)
    return {
        "seed": seed,
        "status": status,
        "summary_path": str(summary_path),
        "run_dir": str(summary_path.parent),
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_completed": summary.get("epochs_completed"),
        "steps_completed": summary.get("steps_completed"),
        "validation_metrics": summary.get("validation_metrics", {}),
        "test_metrics": summary.get("test_metrics", {}),
    }


def find_existing_summary(sweep_dir: Path, seed: int) -> Path | None:
    candidates = sorted(
        sweep_dir.glob(f"*seed-{seed}*/final_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    summaries = sorted(
        sweep_dir.glob("*/final_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summaries:
        cli_args_path = summary_path.parent / "cli_args.json"
        if not cli_args_path.exists():
            continue
        try:
            cli_args = load_json(cli_args_path)
        except (OSError, json.JSONDecodeError):
            continue
        if cli_args.get("seed") == seed:
            return summary_path
    return None


def write_sweep_outputs(
    sweep_dir: Path,
    config_path: Path,
    seeds: Iterable[int],
    runs: list[dict[str, Any]],
) -> None:
    completed = [run for run in runs if run.get("status") in {"completed", "skipped_existing"}]
    payload = {
        "config_path": str(config_path),
        "seeds": list(seeds),
        "num_completed": len(completed),
        "num_runs": len(runs),
        "runs": runs,
        "aggregates": {
            "run": aggregate_run_fields(completed),
            "validation_metrics": aggregate_metric_group(completed, "validation_metrics"),
            "test_metrics": aggregate_metric_group(completed, "test_metrics"),
        },
    }

    write_json(sweep_dir / "sweep_summary.json", payload)
    (sweep_dir / "sweep_summary.md").write_text(render_markdown(payload), encoding="utf-8")


def aggregate_run_fields(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = ("best_metric", "best_epoch", "epochs_completed", "steps_completed")
    out: dict[str, dict[str, Any]] = {}
    for field in fields:
        values = [float(run[field]) for run in runs if is_number(run.get(field))]
        if values:
            out[field] = summarize_values(values)
    return out


def aggregate_metric_group(runs: list[dict[str, Any]], group: str) -> dict[str, dict[str, Any]]:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        metrics = run.get(group, {})
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if is_number(value):
                values_by_key[str(key)].append(float(value))

    return {
        key: summarize_values(values)
        for key, values in sorted(values_by_key.items())
        if values
    }


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TIGER Seed Sweep Summary",
        "",
        f"- Config: `{payload['config_path']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- Completed runs: `{payload['num_completed']} / {payload['num_runs']}`",
        "",
        "## Runs",
        "",
        "| seed | status | best val | best epoch | test ndcg@20 | test recall@20 | test coverage@20 | run |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for run in payload["runs"]:
        test = run.get("test_metrics", {}) if isinstance(run.get("test_metrics"), Mapping) else {}
        lines.append(
            "| {seed} | {status} | {best} | {epoch} | {ndcg} | {recall} | {coverage} | `{run_dir}` |".format(
                seed=run.get("seed"),
                status=run.get("status"),
                best=format_number(run.get("best_metric")),
                epoch=format_number(run.get("best_epoch"), digits=0),
                ndcg=format_number(test.get("test/sft/generation_ndcg@20")),
                recall=format_number(test.get("test/sft/generation_recall@20")),
                coverage=format_number(test.get("test/sft/generation_coverage@20")),
                run_dir=run.get("run_dir", ""),
            )
        )

    lines.extend(["", "## Aggregates", ""])
    selected = [
        ("run", "best_metric"),
        ("test_metrics", "test/sft/generation_ndcg@20"),
        ("test_metrics", "test/sft/generation_recall@20"),
        ("test_metrics", "test/sft/generation_coverage@20"),
        ("validation_metrics", "val/sft/generation_ndcg@20"),
        ("validation_metrics", "val/sft/generation_recall@20"),
        ("validation_metrics", "val/sft/generation_coverage@20"),
    ]
    lines.extend(["| metric | mean | std | values |", "| --- | ---: | ---: | --- |"])
    aggregates = payload.get("aggregates", {})
    for group, key in selected:
        stats = aggregates.get(group, {}).get(key)
        if stats is None:
            continue
        lines.append(
            f"| `{key}` | {format_number(stats.get('mean'))} | "
            f"{format_number(stats.get('std'))} | `{format_values(stats.get('values', []))}` |"
        )

    lines.extend(
        [
            "",
            "Full numeric aggregates are in `sweep_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def format_number(value: Any, *, digits: int = 6) -> str:
    if not is_number(value):
        return ""
    if digits == 0:
        return str(int(round(float(value))))
    return f"{float(value):.{digits}f}"


def format_values(values: Iterable[Any]) -> str:
    return ", ".join(format_number(value) for value in values if is_number(value))


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
