"""Export plots and a portable HTML report of observed bucket diagnostics."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_report(tables, folder: Path, metadata):
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = {s["run_id"]: s for s in metadata["runs"]}
    labels = {rid: ("init" if s["stage"] == "init" else
                    f"SFT seed{s['seed']}" if s["stage"] == "sft" else
                    "GFlow " + " / ".join(s["selection_labels"])) for rid, s in specs.items()}
    images = []

    def save(fig, name, caption):
        path = figures / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        encoded = base64.b64encode(path.read_bytes()).decode()
        images.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
                      f'<figcaption>{html.escape(caption)}</figcaption></figure>')

    def heatmap(ax, frame, row_key, col_key, value, title, *, diverging=False, row_labels=None):
        if frame.empty:
            ax.set_visible(False)
            return
        # Preserve predefined bucket order, including empty groups.
        rows, cols = list(dict.fromkeys(frame[row_key])), list(dict.fromkeys(frame[col_key]))
        matrix = frame.pivot(index=row_key, columns=col_key, values=value).reindex(index=rows, columns=cols).to_numpy(float)
        finite = matrix[np.isfinite(matrix)]
        limit = max(float(np.abs(finite).max()), 1e-6) if finite.size else 1
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="RdBu" if diverging else "Blues",
                          vmin=-limit if diverging else 0, vmax=limit)
        ax.set_xticks(np.arange(len(cols)), cols, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(rows)), [row_labels.get(x, x) if row_labels else x for x in rows])
        ax.set_title(title)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = matrix[i, j]
                ax.text(j, i, "—" if not np.isfinite(v) else f"{v:+.4f}" if diverging else f"{v:.4f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(v) and abs(v) > .65 * limit else "black")
        ax.figure.colorbar(image, ax=ax, shrink=.75)

    for k in sorted(set(metadata["ks"])):
        cohorts = tables["cohorts"].query("k == @k")
        fig, axes = plt.subplots(1, 2, figsize=(17, max(4, len(specs) * .6)), layout="constrained")
        for ax, kind, col in zip(axes, ("target_popularity", "history_length"), ("target_popularity_bucket", "history_bucket")):
            heatmap(ax, cohorts.loc[cohorts.cohort_type.eq(kind)], "run_id", col, "ndcg",
                    f"NDCG@{k}: {kind}", row_labels=labels)
        save(fig, f"ndcg_buckets_k{k}", "NDCG по популярности тестового таргета и полной длине истории. Пустые группы отмечены прочерком.")

        exposure = tables["popularity_exposure"].query("k == @k and cohort_type == 'overall'")
        fig, axes = plt.subplots(1, 2, figsize=(17, max(4, len(specs) * .6)), layout="constrained")
        for ax, metric in zip(axes, ("exposure_share", "coverage")):
            heatmap(ax, exposure, "run_id", "popularity_bucket", metric, f"{metric}@{k}", row_labels=labels)
        save(fig, f"exposure_buckets_k{k}", "Доли показов и охват внутри каждого бакета каталога; знаменатель coverage — размер этого бакета.")

        comparisons = tables["comparisons"].query("k == @k")
        for reference in ("init", "sft"):
            selected = comparisons.loc[comparisons.reference_stage.eq(reference) & comparisons.run_id.map(lambda r: specs[r]["stage"] == "gflow")]
            fig, axes = plt.subplots(1, 2, figsize=(17, 4.5), layout="constrained")
            for ax, kind, col in zip(axes, ("target_popularity", "history_length"), ("target_popularity_bucket", "history_bucket")):
                heatmap(ax, selected.loc[selected.cohort_type.eq(kind)], "run_id", col, "delta_ndcg",
                        f"Delta NDCG@{k} vs {reference}", diverging=True, row_labels=labels)
            save(fig, f"delta_ndcg_vs_{reference}_k{k}", f"Парное изменение NDCG относительно {reference}. Для SFT используется seed 42.")
            cross = selected.loc[selected.cohort_type.eq("target_popularity_x_history_length")]
            run_ids = list(dict.fromkeys(cross.run_id))
            if run_ids:
                fig, axes = plt.subplots(1, len(run_ids), figsize=(7 * len(run_ids), 4), squeeze=False, layout="constrained")
                for ax, rid in zip(axes.flat, run_ids):
                    heatmap(ax, cross.loc[cross.run_id.eq(rid)], "target_popularity_bucket", "history_bucket", "delta_ndcg",
                            f"{labels[rid]} vs {reference}, K={k}", diverging=True)
                save(fig, f"cross_delta_vs_{reference}_k{k}", "Пересечение групп популярности таргета и длины истории. Размеры групп доступны в cohorts.csv.")

    summary = tables["summary"].copy()
    summary["run"] = summary.run_id.map(labels)
    columns = [c for c in ("run", "k", "num_users", "hr", "ndcg", "coverage", "arp", "exposure_gini", "effective_catalog") if c in summary]
    selections = "".join(f"<li>{html.escape(labels[rid])}: <code>{html.escape(spec['checkpoint'])}</code>"
                         f" (таблица {html.escape(', '.join(spec['selection_cells']))})</li>" for rid, spec in specs.items())
    sample_note = "<p><b>Это подвыборка: её coverage нельзя сравнивать с полным тестом.</b></p>" if metadata["subsample"] else ""
    body = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Finetuning buckets: {html.escape(metadata['dataset'])}</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#182b3a}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}img{{max-width:100%}}figure{{margin:32px 0}}figcaption{{color:#526474}}code{{overflow-wrap:anywhere}}h1,h2{{font-weight:600}}</style>
<h1>{html.escape(metadata['dataset'])}: изменения после файнтюнинга</h1>{sample_note}
<p>Единый test split, constrained beam search {metadata['generation']['num_beams']}, FP32; ранги сохраняются при оценке бакетов.
Популярность — частота тренировочных взаимодействий. Нулевые частоты выделены в unseen;
остальные items разбиты по среднему процентильному рангу: tail ≤50%, middle ≤80%, head &gt;80%.
Равные частоты не разделяются. История — все положительные события до тестового таргета
после исходной предобработки, до отсечения контекста и фильтрации холодных айтемов из истории.</p>
<p>Coverage и энтропия показов описывают разнообразие общего охвата. Семантическое разнообразие
внутри списка этим экспериментом не измеряется. Выбор конфигураций унаследован из таблицы;
это описательный анализ фиксированных моделей. Сравнение checkpoint до/после не отслеживает отдельные эпохи обучения.</p>
<h2>Общие метрики</h2>{summary[columns].to_html(index=False, float_format=lambda x: f'{x:.6g}', escape=True)}
{''.join(images)}<h2>Что использовать для выводов</h2>
<p><code>accuracy_attribution.csv</code> даёт аддитивные вклады бакетов в общий ΔHR/ΔNDCG.
<code>comparisons.csv</code> содержит парные выигрыши, потери и изменения ранга.
<code>item_deltas.csv.gz</code> показывает конкретные items, их число тестовых таргетов,
изменения точности и показов. Для редких таргетов учитывайте малый размер выборки.
<code>metric_validation.csv</code> сравнивает пересчитанные полные метрики с исходными артефактами.</p>
<h2>Выбранные чекпойнты</h2><ul>{selections}</ul></html>"""
    path = folder / "report.html"
    path.write_text(body, encoding="utf-8")
    return path
