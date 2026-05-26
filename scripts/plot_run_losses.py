from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SFT and GFlow loss curves from FlowRec experiment runs."
    )
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="Experiment run directories, or paths to training_log.jsonl files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to losses.png next to one run, or ./losses.png.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving.")
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window for train curves. Validation curves are not smoothed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt

    runs = [load_run(path, smooth=max(int(args.smooth), 1)) for path in args.runs]
    output = args.output or default_output(args.runs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    plot_component(axes[0], runs, component="sft")
    plot_component(axes[1], runs, component="gflow")

    for ax in axes:
        ax.set_xlabel("global step")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved {output}")

    if args.show:
        plt.show()


def load_run(path: Path, *, smooth: int) -> dict[str, Any]:
    log_path = resolve_log_path(path)
    label = infer_label(log_path)

    series = {
        "label": label,
        "sft": {"train": [], "val": []},
        "gflow": {"train": [], "val": []},
    }

    for row in iter_jsonl(log_path):
        event_type = row.get("type")
        if event_type == "train_step":
            step = as_number(row.get("global_step"))
            add_point(series["sft"]["train"], step, find_metric(row, "train", "sft_loss"))
            add_point(series["gflow"]["train"], step, find_metric(row, "train", "gflow_loss"))
        elif event_type == "epoch_end":
            step = as_number(row.get("global_step"))
            train = row.get("train") if isinstance(row.get("train"), Mapping) else {}
            val = row.get("val") if isinstance(row.get("val"), Mapping) else {}

            add_point(series["sft"]["train"], step, find_metric(train, "train", "sft_loss"))
            add_point(series["gflow"]["train"], step, find_metric(train, "train", "gflow_loss"))
            add_point(series["sft"]["val"], step, find_metric(val, "val", "sft_loss"))
            add_point(series["gflow"]["val"], step, find_metric(val, "val", "gflow_loss"))

    if smooth > 1:
        for component in ("sft", "gflow"):
            series[component]["train"] = moving_average_points(series[component]["train"], smooth)

    return series


def plot_component(ax: Any, runs: Iterable[dict[str, Any]], *, component: str) -> None:
    plotted = False
    for run in runs:
        label = str(run["label"])
        train = run[component]["train"]
        val = run[component]["val"]

        if train:
            steps, values = zip(*train)
            ax.plot(steps, values, linewidth=1.6, label=f"{label} train")
            plotted = True

        if val:
            steps, values = zip(*val)
            ax.plot(steps, values, marker="o", linewidth=1.8, label=f"{label} val")
            plotted = True

    title = "SFT loss" if component == "sft" else "GFlow loss"
    ax.set_title(title)
    ax.set_ylabel("loss")
    if not plotted:
        ax.text(
            0.5,
            0.5,
            f"No {component} loss found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def find_metric(metrics: Mapping[str, Any], stage: str, name: str) -> float | None:
    candidates = (
        f"{stage}/gflow/{name}",
        f"gflow/{name}",
        f"{stage}/{name}",
        name,
    )
    for key in candidates:
        value = as_number(metrics.get(key))
        if value is not None:
            return value
    return None


def add_point(points: list[tuple[float, float]], step: float | None, value: float | None) -> None:
    if step is None or value is None:
        return
    points.append((step, value))


def moving_average_points(
    points: list[tuple[float, float]],
    window: int,
) -> list[tuple[float, float]]:
    if window <= 1 or len(points) < 2:
        return points

    smoothed: list[tuple[float, float]] = []
    values: list[float] = []
    for step, value in points:
        values.append(value)
        recent = values[-window:]
        smoothed.append((step, sum(recent) / len(recent)))
    return smoothed


def resolve_log_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "training_log.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Training log does not exist: {path}")
    return path


def infer_label(log_path: Path) -> str:
    run_dir = log_path.parent
    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
        run_name = config.get("run_dir")
        if isinstance(run_name, str):
            return Path(run_name).name
    return run_dir.name


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def default_output(paths: list[Path]) -> Path:
    if len(paths) == 1 and paths[0].is_dir():
        return paths[0] / "losses.png"
    return Path("losses.png")


if __name__ == "__main__":
    main()
