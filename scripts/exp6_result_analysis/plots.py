"""Self-contained visual report for the Beauty recommendation comparison.

The input tables contain observed recommendation lists, not model probabilities.
Consequently this module describes catalog exposure and list overlap, and never
equates either quantity with policy entropy or claims a causal effect of SFT.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


_COLORS = {"GPT-2": "#2878B5", "TIGER": "#D55E00"}
_BASE_STAGES = {"init", "base", "baseline", "initial"}


def _model(value: Any) -> str:
    normalized = str(value).lower().replace("-", "").replace("_", "")
    return {"gpt2": "GPT-2", "tiger": "TIGER"}.get(normalized, str(value))


def _is_base(value: Any) -> bool:
    return str(value).lower() in _BASE_STAGES


def _stage(value: Any) -> str:
    return "Init" if _is_base(value) else "SFT-only"


def _color(value: Any) -> str:
    return _COLORS.get(_model(value), "#6B7280")


def _has(frame: pd.DataFrame, *columns: str) -> bool:
    return not frame.empty and set(columns).issubset(frame.columns)


def _at_k(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    return frame.loc[frame["k"].eq(k)].copy() if "k" in frame else frame.copy()


def _ordered_models(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["model"].unique(), key=lambda x: (x != "GPT-2", str(x)))


def _clean_axis(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def _save(fig: Any, directory: Path, stem: str) -> tuple[Path, Path]:
    png, svg = directory / f"{stem}.png", directory / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def _seed_label(row: pd.Series) -> str:
    seed = row.get("seed")
    return "" if pd.isna(seed) else f"seed {int(seed)}" if isinstance(seed, (int, float, np.number)) else f"seed {seed}"


def _tradeoff(summary: pd.DataFrame, comparisons: pd.DataFrame, ks: list[int]) -> Any | None:
    if not _has(summary, "ndcg", "coverage", "model", "stage", "run_id"):
        return None
    fig, axes = plt.subplots(1, len(ks), figsize=(6.2 * len(ks), 4.6), squeeze=False, layout="constrained")
    for ax, k in zip(axes.flat, ks):
        data = _at_k(summary, k).drop_duplicates("run_id")
        cmp = _at_k(comparisons, k)
        by_run = data.set_index("run_id")
        for _, row in data.iterrows():
            base = _is_base(row["stage"])
            color = _color(row["model"])
            ax.scatter(row["ndcg"], row["coverage"] * 100, s=100 if base else 48,
                       marker="s" if base else "o", color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)
            if base:
                ax.annotate(f"{row['model']} init", (row["ndcg"], row["coverage"] * 100),
                            xytext=(6, -15), textcoords="offset points", fontsize=9, color=color)
        if _has(cmp, "run_id", "baseline_run_id"):
            for _, pair in cmp.iterrows():
                if pair["run_id"] not in by_run.index or pair["baseline_run_id"] not in by_run.index:
                    continue
                before, after = by_run.loc[pair["baseline_run_id"]], by_run.loc[pair["run_id"]]
                ax.annotate("", xy=(after["ndcg"], after["coverage"] * 100),
                            xytext=(before["ndcg"], before["coverage"] * 100),
                            arrowprops={"arrowstyle": "->", "color": _color(after["model"]),
                                        "alpha": 0.58, "lw": 1.4, "shrinkA": 6, "shrinkB": 6})
        for model in _ordered_models(data):
            ft = data.loc[data["model"].eq(model) & ~data["stage"].map(_is_base)]
            if not ft.empty:
                ax.scatter(ft["ndcg"].mean(), ft["coverage"].mean() * 100,
                           marker="D", s=80, facecolor="white", edgecolor=_color(model), linewidth=1.8, zorder=5)
        ax.set(xlabel=f"NDCG@{k}", ylabel=f"Catalog coverage@{k}, %", title=f"Accuracy and catalog coverage · K = {k}")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.margins(x=0.15, y=0.18)
        _clean_axis(ax)
    axes.flat[0].legend(handles=[
        Line2D([], [], marker="s", color="#555555", linestyle="", label="Init (shared checkpoint)"),
        Line2D([], [], marker="o", color="#555555", linestyle="", label="SFT-only seed"),
        Line2D([], [], marker="D", markerfacecolor="white", color="#555555", linestyle="", label="SFT-only mean"),
    ], fontsize=8, loc="best")
    return fig


def _metric_changes(summary: pd.DataFrame, comparisons: pd.DataFrame, k: int) -> Any | None:
    if not _has(summary, "run_id", "model") or not _has(comparisons, "run_id", "baseline_run_id"):
        return None
    data = _at_k(summary, k).drop_duplicates("run_id").set_index("run_id")
    changes = []
    for _, pair in _at_k(comparisons, k).iterrows():
        if pair["run_id"] not in data.index or pair["baseline_run_id"] not in data.index:
            continue
        before, after = data.loc[pair["baseline_run_id"]], data.loc[pair["run_id"]]
        for metric in ("ndcg", "coverage", "arp"):
            if metric in data and np.isfinite(before[metric]) and before[metric] != 0:
                changes.append({"model": after["model"], "metric": metric,
                                "change": (after[metric] / before[metric] - 1) * 100})
    changes = pd.DataFrame(changes)
    if changes.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout="constrained")
    models = _ordered_models(changes)
    for ax, metric in zip(axes, ("ndcg", "coverage", "arp")):
        for index, model in enumerate(models):
            values = changes.loc[changes["model"].eq(model) & changes["metric"].eq(metric), "change"].to_numpy()
            if not len(values):
                continue
            mean = values.mean()
            ax.bar(index, mean, width=0.5, color=_color(model), alpha=0.28)
            ax.scatter(index + np.linspace(-0.10, 0.10, len(values)), values,
                       color=_color(model), s=35, edgecolor="white", linewidth=0.5, zorder=3)
            ax.annotate(f"{mean:+.2f}%", (index, mean), xytext=(0, 8 if mean >= 0 else -17),
                        textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#6B7280", lw=0.8)
        ax.set(xticks=range(len(models)), xticklabels=models, ylabel="Relative change from init, %",
               title=f"{metric.upper() if metric != 'coverage' else 'Coverage'}@{k}")
        ax.margins(y=0.25)
        _clean_axis(ax)
    return fig


def _accuracy_attribution(attribution: pd.DataFrame, ks: list[int]) -> Any | None:
    if not _has(attribution, "model", "run_id", "target_group", "delta_ndcg_contribution"):
        return None
    fig, axes = plt.subplots(1, len(ks), figsize=(6.1 * len(ks), 4.5), squeeze=False, layout="constrained")
    groups = [("newly covered targets", "Targets newly covered", "#209C80"),
              ("previously covered targets", "Targets already covered", "#697DB1"),
              ("never covered targets", "Targets never covered", "#AAB2BE")]
    for ax, k in zip(axes.flat, ks):
        data = _at_k(attribution, k)
        models = _ordered_models(data)
        for index, model in enumerate(models):
            subset = data.loc[data["model"].eq(model)]
            positive, negative = 0.0, 0.0
            for group, label, color in groups:
                contribution = subset.loc[subset["target_group"].eq(group), "delta_ndcg_contribution"].mean() * 1000
                if not np.isfinite(contribution):
                    continue
                ax.bar(index, contribution, width=0.50, bottom=positive if contribution >= 0 else negative,
                       color=color, alpha=0.85, label=label if index == 0 else None)
                if contribution >= 0:
                    positive += contribution
                else:
                    negative += contribution
            totals = subset.groupby("run_id")["delta_ndcg_contribution"].sum().to_numpy() * 1000
            ax.scatter(index + np.linspace(-0.11, 0.11, len(totals)), totals,
                       color="#172B4D", s=30, edgecolor="white", linewidth=0.6, zorder=4,
                       label="Total per SFT seed" if index == 0 else None)
            ax.plot([index - 0.29, index + 0.29], [totals.mean()] * 2, color="#172B4D", lw=1.8,
                    label="Mean total change" if index == 0 else None)
        ax.axhline(0, color="#758095", lw=0.8)
        ax.set(xticks=range(len(models)), xticklabels=models, ylabel="Contribution to ΔNDCG (×10⁻³)",
               title=f"Where does the accuracy change come from? · K = {k}")
        ax.margins(y=0.23)
        _clean_axis(ax)
    axes.flat[0].legend(fontsize=8, loc="best")
    return fig


def _lorenz(items: pd.DataFrame, k: int, catalog_size: int | None) -> Any | None:
    if not _has(items, "run_id", "model", "stage", "exposures"):
        return None
    data = _at_k(items, k)
    fig, ax = plt.subplots(figsize=(8.5, 5.4), layout="constrained")
    grid = np.linspace(0, 1, 401)
    plotted = False
    for (model, stage), group in data.groupby(["model", "stage"], sort=True):
        curves = []
        for _, run in group.groupby("run_id"):
            values = np.asarray(run["exposures"], dtype=float)
            values = values[np.isfinite(values) & (values >= 0)]
            size = max(len(values), catalog_size or 0)
            if size > len(values):
                values = np.pad(values, (size - len(values), 0))
            if values.sum() <= 0:
                continue
            cumulative = np.r_[0, np.cumsum(np.sort(values)) / values.sum()]
            curves.append(np.interp(grid, np.linspace(0, 1, len(cumulative)), cumulative))
        if not curves:
            continue
        curves = np.asarray(curves)
        ax.plot(grid, curves.mean(axis=0), color=_color(model), ls="--" if _is_base(stage) else "-",
                lw=2.1, label=f"{model} {_stage(stage)}")
        if len(curves) > 1:
            ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=_color(model), alpha=0.10)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color="#9299A3", linestyle=":", linewidth=1.2, label="Uniform exposure")
    ax.set(xlabel="Share of catalog, ordered by increasing exposure", ylabel="Cumulative share of recommendation slots",
           title=f"Concentration of catalog exposure · K = {k}", xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", fontsize=9)
    _clean_axis(ax)
    return fig


def _new_items(items: pd.DataFrame, k: int) -> Any | None:
    if not _has(items, "run_id", "model", "stage", "exposures", "popularity", "status"):
        return None
    data = _at_k(items, k)
    data = data.loc[~data["stage"].map(_is_base)]
    newly = data.loc[data["status"].eq("new") & data["exposures"].gt(0)]
    if newly.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    models = _ordered_models(data)
    bins = np.array([1, 2, 5, 17, 65, np.inf])
    labels = ["1", "2–4", "5–16", "17–64", "65+"]
    width = 0.72 / max(len(models), 1)
    for index, model in enumerate(models):
        shares = []
        for _, run in newly.loc[newly["model"].eq(model)].groupby("run_id"):
            counts, _ = np.histogram(run["exposures"].to_numpy(), bins=bins)
            shares.append(counts / counts.sum())
        if shares:
            axes[0].bar(np.arange(len(labels)) + (index - (len(models) - 1) / 2) * width,
                        np.mean(shares, axis=0), width=width, color=_color(model), alpha=0.85, label=model)
        for status, style, description in (("new", "-", "newly covered"), ("retained", "--", "retained")):
            group = data.loc[data["model"].eq(model) & data["status"].eq(status) & data["exposures"].gt(0)]
            if group.empty:
                continue
            max_pop = max(float(group["popularity"].max()), 1)
            x = np.unique(np.r_[0, np.geomspace(1, max_pop, 300)])
            cdfs = []
            for _, run in group.groupby("run_id"):
                values = np.sort(run["popularity"].dropna().to_numpy())
                if len(values):
                    cdfs.append(np.searchsorted(values, x, side="right") / len(values))
            if cdfs:
                axes[1].plot(x, np.mean(cdfs, axis=0), color=_color(model), ls=style, lw=2,
                             label=f"{model}: {description}")
    axes[0].set(xticks=np.arange(len(labels)), xticklabels=labels,
                xlabel="Recommendation slots per newly covered item", ylabel="Share of newly covered items",
                title=f"How often do the new items appear? · K = {k}")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel="Item popularity in training interactions", ylabel="Share of items at or below this popularity",
                title="Are the new items less popular?", ylim=(0, 1))
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        _clean_axis(ax)
    return fig


def _user_changes(users: pd.DataFrame, k: int) -> Any | None:
    if not _has(users, "run_id", "model", "delta_ndcg", "list_overlap", "new_item_share"):
        return None
    data = _at_k(users, k)
    data["outcome"] = np.select([data["delta_ndcg"].gt(1e-12), data["delta_ndcg"].lt(-1e-12)],
                                ["Improved", "Worsened"], default="Unchanged")
    outcomes = ["Improved", "Unchanged", "Worsened"]
    models = _ordered_models(data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    width = 0.72 / max(len(models), 1)
    for index, model in enumerate(models):
        group = data.loc[data["model"].eq(model)]
        totals = group.groupby("run_id").size()
        for metric_index, metric in enumerate(("share", "list_overlap", "new_item_share")):
            if metric == "share":
                pivot = group.groupby(["run_id", "outcome"]).size().unstack(fill_value=0)
                pivot = pivot.reindex(index=totals.index, columns=outcomes, fill_value=0).div(totals, axis=0)
            else:
                pivot = group.groupby(["run_id", "outcome"])[metric].mean().unstack().reindex(columns=outcomes)
            means = pivot.mean(axis=0)
            x = np.arange(len(outcomes)) + (index - (len(models) - 1) / 2) * width
            axes[metric_index].bar(x, means, width=width, color=_color(model), alpha=0.70, label=model)
            for position, outcome in zip(x, outcomes):
                values = pivot[outcome].dropna().to_numpy()
                axes[metric_index].scatter(position + np.linspace(-0.035, 0.035, len(values)), values,
                                           color=_color(model), s=12, edgecolor="white", linewidth=0.4, zorder=3)
    titles = [f"Users by NDCG@{k} change", "Overlap with the user's init list", "New-to-catalog items in user lists"]
    labels = ["Share of users", f"Mean |init ∩ SFT| / {k}", "Mean share of recommendation slots"]
    for ax, title, label in zip(axes, titles, labels):
        ax.set(xticks=np.arange(len(outcomes)), xticklabels=outcomes, title=title, ylabel=label, ylim=(0, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        _clean_axis(ax)
    axes[0].legend(fontsize=9)
    return fig


def _popularity_shifts(popularity: pd.DataFrame, k: int) -> Any | None:
    if not _has(popularity, "run_id", "model", "stage", "popularity_group", "exposure_share"):
        return None
    data = _at_k(popularity, k)
    preferred = ["Tail (bottom 50%)", "Middle (50–80%)", "Head (top 20%)"]
    present = data["popularity_group"].dropna().unique().tolist()
    groups = [x for x in preferred if x in present] + sorted(set(present) - set(preferred))
    models = _ordered_models(data)
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 4.5), squeeze=False, layout="constrained")
    for ax, model in zip(axes.flat, models):
        subset = data.loc[data["model"].eq(model)]
        for stage, stage_data in subset.groupby("stage", sort=True):
            means = stage_data.groupby("popularity_group")["exposure_share"].mean().reindex(groups)
            base = _is_base(stage)
            x = np.arange(len(groups)) + (-0.17 if base else 0.17)
            ax.bar(x, means, width=0.31, color=_color(model), alpha=0.30 if base else 0.88,
                   label=_stage(stage), edgecolor="white")
            if not base:
                for position, group in zip(x, groups):
                    values = stage_data.loc[stage_data["popularity_group"].eq(group), "exposure_share"].to_numpy()
                    ax.scatter(position + np.linspace(-0.045, 0.045, len(values)), values,
                               color="#222222", s=14, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set(xticks=np.arange(len(groups)), xticklabels=[g.replace(" (", "\n(") for g in groups],
               ylabel="Share of recommendation slots", title=f"{model}: exposure by training popularity · K = {k}")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=9)
        _clean_axis(ax)
    return fig


def _cross_model(cross: pd.DataFrame, summary: pd.DataFrame, k: int) -> Any | None:
    if not _has(cross, "run_a", "run_b", "mean_list_overlap"):
        return None
    data = _at_k(cross, k)
    if data.empty:
        return None
    known = _at_k(summary, k).drop_duplicates("run_id") if "run_id" in summary else pd.DataFrame()
    ids = sorted(set(data["run_a"]) | set(data["run_b"]))
    labels = {run: str(run) for run in ids}
    if not known.empty:
        records = known.set_index("run_id")
        ids.sort(key=lambda run: (_model(records.loc[run, "model"]),
                                  not _is_base(records.loc[run, "stage"]), str(run)) if run in records.index else (str(run), False, str(run)))
        for run in ids:
            if run in records.index:
                row = records.loc[run]
                labels[run] = f"{row['model']} {_stage(row['stage'])}"
                if not _is_base(row["stage"]):
                    labels[run] += f"\n{_seed_label(row)}"
    metrics = [("mean_list_overlap", "Mean overlap of user lists"), ("catalog_jaccard", "Jaccard of covered catalog")]
    metrics = [pair for pair in metrics if pair[0] in data]
    side = max(5, 0.65 * len(ids) + 2)
    fig, axes = plt.subplots(1, len(metrics), figsize=(side * len(metrics), side), squeeze=False, layout="constrained")
    index = {run: i for i, run in enumerate(ids)}
    for ax, (metric, title) in zip(axes.flat, metrics):
        matrix = np.full((len(ids), len(ids)), np.nan)
        np.fill_diagonal(matrix, 1)
        for _, row in data.iterrows():
            a, b = index[row["run_a"]], index[row["run_b"]]
            matrix[a, b] = matrix[b, a] = row[metric]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if value > 0.60 else "#172B4D")
        ax.set(xticks=range(len(ids)), yticks=range(len(ids)),
               xticklabels=[labels[run] for run in ids], yticklabels=[labels[run] for run in ids],
               title=f"{title} · K = {k}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=PercentFormatter(1))
    return fig


def _summary_html(summary: pd.DataFrame) -> str:
    columns = [x for x in ("ndcg", "hr", "coverage", "arp", "exposure_gini", "effective_catalog", "mean_pairwise_overlap") if x in summary]
    if not columns or not _has(summary, "model", "stage", "k"):
        return "<p>Таблица итоговых метрик отсутствует.</p>"
    data = summary.copy()
    data["stage"] = data["stage"].map(_stage)
    if "run_id" in data:
        data = data.drop_duplicates(["run_id", "k"])
    rows = []
    for (model, stage, k), group in data.groupby(["model", "stage", "k"], sort=True):
        row = {"Модель": model, "Этап": stage, "K": int(k), "Чекпойнтов": len(group)}
        for metric in columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = values.mean()
            if metric in {"coverage", "mean_pairwise_overlap"}:
                value = f"{100 * mean:.2f}%"
                deviation = f"{100 * values.std(ddof=1):.2f} п.п."
            else:
                digits = 1 if metric in {"arp", "effective_catalog"} else 5
                value = f"{mean:.{digits}f}"
                deviation = f"{values.std(ddof=1):.{digits}f}"
            row[metric] = "—" if values.empty else value + (f" ± {deviation}" if len(values) > 1 else "")
        rows.append(row)
    return pd.DataFrame(rows).to_html(index=False, border=0, classes="metrics", escape=True)


def _observations(summary: pd.DataFrame, comparisons: pd.DataFrame, k: int) -> str:
    if not _has(summary, "run_id", "model", "ndcg", "coverage") or not _has(comparisons, "run_id", "baseline_run_id"):
        return ""
    rows = _at_k(summary, k).drop_duplicates("run_id").set_index("run_id")
    result = []
    values: dict[str, list[tuple[float, float, float]]] = {}
    for _, pair in _at_k(comparisons, k).iterrows():
        if pair["run_id"] not in rows.index or pair["baseline_run_id"] not in rows.index:
            continue
        before, after = rows.loc[pair["baseline_run_id"]], rows.loc[pair["run_id"]]
        relative_ndcg = (after["ndcg"] / before["ndcg"] - 1) * 100 if before["ndcg"] else np.nan
        relative_coverage = (after["coverage"] / before["coverage"] - 1) * 100 if before["coverage"] else np.nan
        absolute_coverage = (after["coverage"] - before["coverage"]) * 100
        values.setdefault(str(after["model"]), []).append((relative_ndcg, relative_coverage, absolute_coverage))
    for model, measurements in sorted(values.items()):
        ndcg, coverage, points = np.mean(measurements, axis=0)
        result.append(f"<li><strong>{html.escape(model)}</strong>: NDCG@{k} {ndcg:+.2f}%, "
                      f"coverage@{k} {coverage:+.2f}% ({points:+.2f} п.п.). "
                      f"Среднее изменение по {len(measurements)} SFT-чекпойнтам относительно соответствующего init.</li>")
    return "<ul>" + "".join(result) + "</ul>" if result else ""


def _comparisons_html(comparisons: pd.DataFrame, k: int) -> str:
    if not _has(comparisons, "run_id", "model", "delta_ndcg"):
        return ""
    rows = []
    for _, item in _at_k(comparisons, k).iterrows():
        row = {"Модель": item["model"], "Seed": _seed_label(item).replace("seed ", ""),
               "Δ NDCG": f"{item['delta_ndcg']:+.6f}"}
        low, high = item.get("ndcg_ci_low", np.nan), item.get("ndcg_ci_high", np.nan)
        if np.isfinite(low) and np.isfinite(high):
            row["95% CI Δ NDCG"] = f"[{low:+.6f}; {high:+.6f}]"
        for name, label, format_string in (
            ("new_catalog_items", "Новых товаров", ".0f"),
            ("lost_catalog_items", "Ушедших товаров", ".0f"),
            ("new_item_exposure_share", "Доля показов новых товаров", ".2%"),
            ("mean_list_overlap", "Совпадение с init", ".2%"),
        ):
            if name in item:
                row[label] = format(item[name], format_string)
        rows.append(row)
    if not rows:
        return ""
    return (f"<details><summary>Изменения выдачи по отдельным seeds, K = {k}</summary>"
            '<div class="table-wrap">' + pd.DataFrame(rows).to_html(index=False, border=0, escape=True)
            + "</div><p>Если указан 95% CI, это парный bootstrap по пользователям для фиксированной пары чекпойнтов. "
            "Он не оценивает неопределённость по обучающим seeds. Новые и ушедшие товары определяются по совокупному "
            "охвату каталога; число новых минус число ушедших равно изменению числа уникальных товаров.</p></details>")


def render_report(tables: dict[str, pd.DataFrame], output_dir: Path, metadata: dict) -> Path:
    """Write PNG/SVG figures and a portable HTML report; return its path.

    Required table: ``summary`` (run_id, model, stage, seed, k, ndcg, hr,
    coverage, arp). Additional tables are optional; unavailable analyses are
    identified in the report. See the command-line analysis module for the
    definitions of metrics and the provenance of recommendation lists.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for name, table in tables.items():
        prepared[name] = table.copy()
        if "model" in prepared[name]:
            prepared[name]["model"] = prepared[name]["model"].map(_model)
    empty = pd.DataFrame()
    summary = prepared.get("summary", empty)
    comparisons = prepared.get("comparisons", empty)
    items, users = prepared.get("items", empty), prepared.get("users", empty)
    popularity, cross = prepared.get("popularity", empty), prepared.get("cross_model", empty)
    attribution = prepared.get("accuracy_attribution", empty)
    available_ks = sorted(int(x) for x in summary.get("k", pd.Series(dtype=int)).dropna().unique())
    k = int(metadata.get("primary_k", 5 if 5 in available_ks else available_ks[0] if available_ks else 5))
    ks = sorted(set([k] + ([max(available_ks)] if available_ks else [])))
    catalog_size = metadata.get("catalog_size")
    catalog_size = int(catalog_size) if catalog_size is not None else None
    specs = [
        ("01_accuracy_coverage", "Точность и охват каталога", lambda: _tradeoff(summary, comparisons, ks),
         "Квадрат — исходный чекпойнт, кружки — отдельные SFT-seeds, полый ромб — их среднее. "
         "Стрелки связывают каждый SFT-чекпойнт с его исходной моделью. Совместный рост двух метрик "
         "не устанавливает причинную связь: здесь один датасет и две архитектуры, а seeds используют общую исходную модель."),
        ("02_metric_changes", "Размер изменений относительно исходной модели", lambda: _metric_changes(summary, comparisons, k),
         "Столбцы — среднее относительное изменение по SFT-seeds, точки — отдельные seeds. "
         "Coverage считает уникальные рекомендованные товары; ARP — среднюю обучающую популярность с учётом каждого показа. "
         "Проценты изменения coverage здесь относительные, а не процентные пункты."),
        ("03_exposure_lorenz", "Насколько сконцентрированы показы", lambda: _lorenz(items, k, catalog_size),
         "Кривая Лоренца включает весь каталог, в том числе товары без показов. Приближение к диагонали означает "
         "более равномерные показы. Сплошная линия — среднее SFT-кривых, полоса — диапазон между seeds, "
         "пунктир — init. Эта концентрация относится ко всем выдачам вместе, а не к энтропии вероятностей модели."),
        ("04_new_catalog_items", "Какие товары расширяют охват", lambda: _new_items(items, k),
         "Новые товары — не встречавшиеся ни в одной исходной top-K выдаче, но появившиеся после SFT; "
         "это определение применяется отдельно к каждой модели и seed. Слева — как часто показываются такие товары. "
         "Справа — распределение обучающей популярности новых и сохранившихся товаров, каждый товар имеет одинаковый вес. "
         "Кривые и доли усредняются по seeds, где соответствующая группа непуста."),
        ("05_user_changes", "У каких пользователей меняются рекомендации", lambda: _user_changes(users, k),
         "Пользователи разделены по изменению NDCG после SFT. Вторая панель — доля общих товаров в исходной и новой выдаче "
         "одного пользователя; третья — доля позиций, занятых новыми для общего охвата товарами. "
         "«Unchanged» включает случаи, когда правильный ответ не попал в top-K ни до, ни после SFT. "
         "Средние условные на непустую группу; точки показывают seeds. Эти группы описывают изменения, а не их причины."),
        ("06_popularity_exposure", "Как перераспределились показы между популярными и редкими товарами", lambda: _popularity_shifts(popularity, k),
         "Группы каталога фиксируются по популярности в обучающих взаимодействиях и одинаковы для всех моделей. "
         "Tail — нижние 50% по рангу популярности, Middle — 50–80%, Head — верхние 20%; равные частоты получают "
         "средний процентильный ранг, поэтому размеры групп могут отличаться от указанных долей. "
         "Высота столбца — доля показов, а не доля товаров. Для SFT показаны среднее и отдельные seeds."),
        ("07_run_overlap", "Насколько похожи выдачи GPT-2 и TIGER", lambda: _cross_model(cross, summary, k),
         "Слева для каждой пары чекпойнтов усредняется |Aᵤ ∩ Bᵤ| / K по одним и тем же пользователям; "
         "порядок внутри top-K не учитывается. Справа — Jaccard множеств товаров, рекомендованных хотя бы одному пользователю. "
         "Высокое совпадение общего охвата не означает, что модели рекомендуют одинаковые товары каждому пользователю."),
    ]
    if _has(attribution, "model", "run_id", "target_group", "delta_ndcg_contribution"):
        specs[1] = (
            "02_accuracy_attribution", "Связан ли прирост точности с расширением охвата",
            lambda: _accuracy_attribution(attribution, ks),
            "Изменение среднего NDCG разложено по группам правильных ответов пользователей. "
            "Зелёный вклад — ответы, которых не было ни в одной исходной выдаче, но которые появились после SFT. "
            "Синий — ответы, уже встречавшиеся в исходных выдачах, даже если у других пользователей. "
            "Никогда не покрытые ответы дают нулевой вклад. Сумма вкладов точно равна общему изменению NDCG; "
            "отрицательный вклад рисуется ниже нуля. Столбцы усредняются по seeds, точки показывают их полный ΔNDCG. "
            "Это арифметическое разложение наблюдаемого изменения, а не доказательство причинности. "
            "Масштаб оси: значение 1 соответствует изменению NDCG на 0,001.",
        )
    rendered, unavailable = [], []
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 10, "axes.titleweight": "medium", "svg.fonttype": "none",
             "savefig.facecolor": "white", "figure.facecolor": "white"}
    with plt.rc_context(style):
        for stem, title, factory, caption in specs:
            figure = factory()
            if figure is None:
                unavailable.append(title)
                continue
            png, svg = _save(figure, figures_dir, stem)
            rendered.append((title, caption, png, svg))
    warnings = metadata.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    if metadata.get("subsample") or metadata.get("max_users"):
        warnings = list(warnings) + ["Используется подвыборка пользователей. Coverage и число новых товаров "
                                     "зависят от её размера; результаты нельзя напрямую сравнивать с полным тестом."]
    warning_html = "" if not warnings else '<aside class="notice"><strong>Условия интерпретации</strong><ul>' + "".join(
        f"<li>{html.escape(str(warning))}</li>" for warning in warnings) + "</ul></aside>"
    source_tables = []
    for name in prepared:
        for path in (output_dir / f"{name}.csv", output_dir / "tables" / f"{name}.csv",
                     output_dir / f"{name}.csv.gz", output_dir / "tables" / f"{name}.csv.gz"):
            if path.exists():
                source_tables.append(f'<a href="{html.escape(path.relative_to(output_dir).as_posix())}">{html.escape(path.name)}</a>')
                break
    fragments = []
    for title, caption, png, svg in rendered:
        encoded = base64.b64encode(png.read_bytes()).decode("ascii")
        fragments.append(f'<section><h2>{html.escape(title)}</h2><figure>'
                         f'<img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}">'
                         f'<figcaption>{html.escape(caption)}</figcaption></figure>'
                         f'<p class="downloads"><a href="{png.relative_to(output_dir).as_posix()}">PNG</a> · '
                         f'<a href="{svg.relative_to(output_dir).as_posix()}">SVG</a></p></section>')
    missing_html = "" if not unavailable else "<p>Нет исходных данных для разделов: " + html.escape("; ".join(unavailable)) + ".</p>"
    user_count = metadata.get("num_users", "не указано")
    escaped_metadata = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    report = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beauty: GPT-2 и TIGER до и после SFT</title>
<style>
body{margin:0;background:#f4f6f9;color:#172b4d;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1250px;margin:0 auto;padding:36px 24px 70px}h1{font-size:30px;line-height:1.2;margin-bottom:15px}
h2{font-size:21px;line-height:1.3;margin-top:0}p{max-width:1000px}section,.overview{background:white;border:1px solid #dfe5ed;border-radius:12px;padding:24px;margin:24px 0}
figure{margin:0}img{width:100%;height:auto}figcaption{font-size:14px;color:#46556b;margin:14px 0}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}th,td{text-align:right;padding:9px;border-bottom:1px solid #e5eaf0}th{background:#f2f5f9}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
a{color:#176da0}.downloads{font-size:13px;margin-bottom:0}.muted{color:#59697c}.notice{border-left:4px solid #bd7820;background:#fff6e6;padding:15px 20px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#edf1f6;padding:15px}details{margin-top:24px}
@media print{body{background:white}main{padding:0}section{break-inside:avoid;border:0;padding:10px 0}.downloads,details{display:none}}
</style></head><body><main>
<h1>Beauty: GPT-2 и TIGER до и после SFT-only</h1>
<p>Анализ точности, охвата каталога и изменений рекомендаций на одних и тех же пользователях. "Разнообразие" здесь """
    report += ("измеряется через охват и распределение показов; семантическое разнообразие внутри одной выдачи не измеряется.</p>"
               f'<p class="muted">Пользователей: {html.escape(str(user_count))} · Товаров в каталоге: '
               f'{html.escape(str(catalog_size if catalog_size is not None else "не указано"))} · Основной K = {k}</p>'
               + warning_html + '<div class="overview"><h2>Наблюдаемые изменения</h2>'
               + _observations(summary, comparisons, k)
               + "<p>Отчёт описывает различия сохранённых выдач. По двум моделям на одном датасете нельзя оценить "
               "устойчивую корреляцию между приростом NDCG и coverage или установить, что одно изменение вызывает другое. "
               "SFT-seeds с общим init не являются независимыми повторениями исходного обучения.</p>"
               + '<div class="table-wrap">' + _summary_html(summary) + "</div>"
               + '<p class="muted">± — выборочное стандартное отклонение между чекпойнтами, не доверительный интервал. '
               "Init показан один раз на каждый исходный чекпойнт. Coverage и overlap представлены в процентах.</p></div>"
               + _comparisons_html(comparisons, k)
               + "".join(fragments) + missing_html)
    report += ("<section><h2>Как читать метрики</h2><ul>"
               "<li><strong>Coverage</strong> — доля каталога, встречающаяся хотя бы в одной top-K выдаче. "
               "Рост может обеспечиваться товарами, появившимися всего один раз.</li>"
               "<li><strong>ARP</strong> — средняя популярность рекомендованных товаров в обучающих взаимодействиях; "
               "один товар учитывается столько раз, сколько его рекомендовали.</li>"
               "<li><strong>Exposure Gini / effective catalog</strong> — концентрация всех рекомендательных показов. "
               "Effective catalog = exp(H), где H — энтропия нормированных частот показов. "
               "Это не энтропия выходных вероятностей генеративной модели.</li>"
               "<li><strong>Mean pairwise overlap</strong> — средняя доля общих товаров в выдачах разных пользователей "
               "одной модели. Снижение совместимо с более различающимися выдачами, но само по себе не доказывает "
               "более точную персонализацию.</li></ul></section>")
    if source_tables:
        report += "<p>Таблицы для дальнейшего анализа: " + " · ".join(source_tables) + ".</p>"
    validation = prepared.get("metric_validation", empty)
    if not validation.empty:
        columns = [column for column in ("run_id", "k", "metric", "recorded", "recomputed", "difference") if column in validation]
        report += ("<details><summary>Проверка воспроизведения сохранённых метрик</summary>"
                   "<p>Recorded — метрика исходного эксперимента, recomputed — пересчитанная по выдачам этого анализа, "
                   "difference = recomputed − recorded. Проверка выполняется на полном тестовом наборе. "
                   "Различия режима вычислений (например, FP32 и mixed precision), версии библиотек или настроек генерации "
                   "могут менять выдачи. Ненулевые расхождения требуют учитывать, что анализ относится к пересчитанным выдачам, "
                   "а не гарантирует точное воспроизведение опубликованных чисел.</p>"
                   '<div class="table-wrap">'
                   + validation[columns].head(200).to_html(index=False, border=0, float_format=lambda value: f"{value:.7g}", escape=True)
                   + "</div>"
                   + ("<p>Показаны первые 200 строк; полная проверка — в CSV.</p>" if len(validation) > 200 else "")
                   + "</details>")
    report += ("<details><summary>Параметры, источники чекпойнтов и определения метрик</summary>"
               f"<pre>{escaped_metadata}</pre></details></main></body></html>")
    report_path = output_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path
