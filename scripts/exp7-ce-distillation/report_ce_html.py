"""Dependency-free interactive HTML for the collected CE experiment results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def render_html(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    columns: Sequence[Mapping[str, str]],
    destination: Path,
) -> None:
    """Write a standalone, sortable report; raw metric values remain in CSV.

    JSON is embedded as inert data with HTML-sensitive characters escaped.
    The browser creates table cells with textContent, never interpolated HTML.
    Baselines remain visible when filtering a reward or sampling mode, to make
    comparisons possible; dataset and free-text filters apply to every row.
    """
    payload = json.dumps(
        {"rows": list(rows), "metadata": dict(metadata), "columns": list(columns)},
        ensure_ascii=False, allow_nan=False,
    )
    for character, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                              ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(character, escape)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__REPORT_PAYLOAD__", payload), encoding="utf-8")


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CE distillation — experiment results</title>
<style>
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172b3a; background: #f4f6f8; font-size: 14px; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px; }
main { max-width: 1800px; margin: 0 auto; }
header { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; }
h1 { margin: 0 0 8px; font-size: 27px; letter-spacing: -.5px; font-weight: 650; }
.subtitle, .muted { color: #576777; }
.subtitle { margin: 0; line-height: 1.6; }
.export { background: #174d74; color: white; border-radius: 7px; padding: 11px 16px;
  text-decoration: none; white-space: nowrap; font-weight: 600; }
.export:hover { background: #103a59; }
.exports { display: flex; gap: 8px; flex-wrap: wrap; }
.export.secondary { background: white; color: #174d74; border: 1px solid #b8c7d2; padding: 10px 15px; }
.export.secondary:hover { background: #edf3f7; }
.summary { display: flex; flex-wrap: wrap; gap: 9px; margin: 20px 0 16px; }
.badge { border: 1px solid #d9e2e9; background: white; border-radius: 6px; padding: 7px 11px; }
.notes { margin: 0 0 18px; padding: 13px 16px; border-left: 3px solid #7195ad;
  background: #eaf0f4; border-radius: 0 6px 6px 0; line-height: 1.55; }
.notes ul { margin: 0; padding-left: 18px; }
.notes li + li { margin-top: 4px; }
.notes details { margin-top: 9px; color: inherit; }
.notes details ul { margin-top: 8px; }
.filters { display: flex; align-items: flex-end; flex-wrap: wrap; gap: 12px;
  background: white; border: 1px solid #d9e2e9; border-radius: 8px; padding: 15px; }
label { display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 600; }
input, select, .reset { font: inherit; font-size: 14px; color: #172b3a; background: white;
  border: 1px solid #b8c7d2; border-radius: 5px; padding: 8px 10px; min-height: 36px; }
select { min-width: 170px; max-width: 350px; }
.search { flex: 1; min-width: 210px; }
input { width: 100%; }
.reset { cursor: pointer; }
.reset:hover { background: #edf3f7; }
a:focus-visible, input:focus-visible, select:focus-visible, button:focus-visible {
  outline: 3px solid #81b6de; outline-offset: 2px; }
.table-info { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
  margin: 13px 1px 9px; font-size: 12px; line-height: 1.5; }
.baseline-key { display: inline-block; width: 10px; height: 10px; margin-right: 5px;
  background: #fff3d8; border: 1px solid #ddc28c; border-radius: 2px; }
.table-wrap { overflow: auto; max-height: 72vh; background: white;
  border: 1px solid #ccd8e0; border-radius: 8px; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12px;
  font-variant-numeric: tabular-nums; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e3e9ed; white-space: nowrap; text-align: left; }
thead th { position: sticky; background: #edf2f6; z-index: 3; }
thead .groups th { top: 0; height: 36px; padding: 8px 12px; line-height: 19px;
  background: #dfe8ef; color: #344e61; font-weight: 650; border-right: 1px solid #c9d6df; }
thead .labels th { top: 36px; padding: 0; border-bottom: 1px solid #b8c9d5; }
thead button { width: 100%; padding: 12px; border: 0; background: transparent;
  font: inherit; font-weight: 650; text-align: inherit; color: inherit; cursor: pointer; }
thead button:hover { background: #d7e7f2; }
.sort-mark { display: inline-block; min-width: 12px; margin-left: 5px; color: #738b9c; }
.number { text-align: right; }
tbody tr:nth-child(even) td { background: #f8fafb; }
tbody tr:hover td { background: #eaf3f9; }
tbody tr.baseline td { background: #fff5df; }
tbody tr.baseline:hover td { background: #ffedc3; }
.group-start { border-left: 1px solid #ccd8e0; }
td:first-child { font-weight: 600; }
td a { color: #185d8a; text-underline-offset: 3px; }
.missing { color: #8795a0; }
.empty { text-align: center; padding: 30px; color: #576777; }
details { margin-top: 18px; line-height: 1.5; color: #576777; }
summary { cursor: pointer; }
details ul { overflow-wrap: anywhere; }
footer { margin-top: 18px; color: #687987; font-size: 12px; line-height: 1.5; }
noscript { display: block; padding: 20px; background: #fff5df; }
@media (max-width: 700px) { body { padding: 16px; } header { flex-direction: column; }
  h1 { font-size: 23px; } .filters label { width: 100%; } select { max-width: none; }
  .table-wrap { max-height: 70vh; } }
@media print { body { padding: 0; background: white; } .filters, .export, .reset { display: none; }
  .table-wrap { max-height: none; overflow: visible; } table { font-size: 8px; }
  th, td { padding: 4px; } thead th { position: static; } }
</style>
</head>
<body>
<main>
<header>
  <div><h1>CE distillation: test results</h1>
    <p class="subtitle">Completed configurations and baseline models · <span id="generated"></span></p></div>
  <div class="exports">
    <a class="export" href="test_metrics.csv" download>Download CSV</a>
    <a class="export secondary" href="test_metrics.xlsx" download>Download XLSX</a>
  </div>
</header>
<div class="summary" id="summary"></div>
<aside class="notes" id="notes" aria-label="How to read the results"></aside>
<section class="filters" aria-label="Table filters">
  <label>Dataset<select id="dataset"><option value="">All datasets</option></select></label>
  <label>CE mode<select id="sampling"><option value="">All modes</option></select></label>
  <label>Reward<select id="combo"><option value="">All combinations</option></select></label>
  <label class="search">Search<input id="search" type="search" placeholder="Model, exponent, seed, path…"></label>
  <button type="button" class="reset" id="reset">Reset</button>
</section>
<div class="table-info">
  <span id="visible" aria-live="polite"></span>
  <span><span class="baseline-key"></span>Baseline models remain visible when filtering by mode or reward. Click a heading to sort.</span>
</div>
<div class="table-wrap" tabindex="0" aria-label="Results table; scroll right to see the remaining metrics">
  <table id="results" aria-label="All test metrics for completed CE experiments">
    <thead><tr class="groups" id="groups"></tr><tr class="labels" id="labels"></tr></thead>
    <tbody id="body"></tbody>
  </table>
</div>
<noscript>JavaScript is required to filter and view the HTML table. All values are available in test_metrics.csv.</noscript>
<details id="missing-results" hidden><summary id="missing-title"></summary><ul id="missing-list"></ul></details>
<footer>An empty cell “—” means the value is missing from the source data.
  HTML numbers are rounded for readability; CSV retains the original values.
  The report runs locally and does not access the network.</footer>
</main>
<script type="application/json" id="report-data">__REPORT_PAYLOAD__</script>
<script>
(() => {
  "use strict";
  const { rows, metadata, columns } = JSON.parse(document.getElementById("report-data").textContent);
  const element = id => document.getElementById(id);
  const text = value => value === null || value === undefined ? "" : String(value);
  const groupLabels = { Configuration: "Configuration", "Test@5": "Test · @5",
    "Test@10": "Test · @10", "Test@20": "Test · @20", Time: "Time", Provenance: "Sources" };
  const modeLabels = { reward: "Reward distribution", curriculum: "Two phases",
    curriculum_smooth: "Smooth transition", curriculum_smooth_is: "Smooth transition with q/proposal", exact: "Exact CE" };
  const formats = {
    integer: new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 }),
    decimal: new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 6 }),
    minutes: new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }),
    percent: new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 3 }),
  };
  const isNumber = column => Object.prototype.hasOwnProperty.call(formats, column.format);
  const empty = value => value === null || value === undefined || value === "";
  const display = (value, column) => {
    if (empty(value)) return "—";
    if (isNumber(column)) {
      const number = Number(value);
      return Number.isFinite(number) ? formats[column.format].format(number) : "—";
    }
    return text(value);
  };
  const safeLink = value => {
    const path = text(value).trim();
    // Report links are relative files. Schemes, network paths and controls
    // must never become navigable URLs, including backslash variants.
    return path && !/^[a-z][a-z0-9+.-]*:/i.test(path)
      && !/^[\\/]/.test(path) && !/[\u0000-\u001f\u007f]/.test(path) ? path : null;
  };
  element("generated").textContent = text(metadata.generated_at);
  const counts = metadata.counts || {};
  const summaries = [
    ["Completed runs", counts.completed_runs ?? rows.filter(r => r.row_type !== "baseline").length],
    ["Baseline models", counts.baselines ?? rows.filter(r => r.row_type === "baseline").length],
    ["Datasets", new Set(rows.map(r => r.dataset)).size],
  ];
  for (const [label, count] of summaries) {
    const badge = document.createElement("span"); badge.className = "badge";
    badge.textContent = `${label}: ${count}`; element("summary").append(badge);
  }
  const notes = metadata.notes || [];
  if (notes.length) {
    const list = document.createElement("ul");
    for (const note of notes.slice(0, 2)) {
      const item = document.createElement("li"); item.textContent = text(note); list.append(item);
    }
    element("notes").append(list);
    if (notes.length > 2) {
      const details = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = "Methodology and timing limitations";
      const remaining = document.createElement("ul");
      for (const note of notes.slice(2)) {
        const item = document.createElement("li"); item.textContent = text(note); remaining.append(item);
      }
      details.append(summary, remaining); element("notes").append(details);
    }
  } else element("notes").hidden = true;
  const missing = metadata.missing_results || [];
  if (missing.length) {
    element("missing-results").hidden = false;
    element("missing-title").textContent = `No completed result: ${missing.length}`;
    for (const source of missing) {
      const item = document.createElement("li"); item.textContent = text(source); element("missing-list").append(item);
    }
  }
  for (const key of ["dataset", "sampling", "combo"]) {
    const values = [...new Set(rows.filter(r => key === "dataset" || r.row_type !== "baseline")
      .map(row => text(row[key])).filter(Boolean))].sort((a, b) => a.localeCompare(b, "ru", {numeric: true}));
    for (const value of values) {
      const option = document.createElement("option"); option.value = value;
      option.textContent = key === "sampling" ? (modeLabels[value] || value) : value;
      element(key).append(option);
    }
  }
  const groupStarts = new Set();
  let lastGroup = null, groupCell = null;
  columns.forEach((column, index) => {
    const name = column.group || "";
    if (index === 0 || name !== lastGroup) {
      groupStarts.add(index); groupCell = document.createElement("th"); groupCell.scope = "colgroup";
      groupCell.colSpan = 1; groupCell.textContent = groupLabels[name] || name;
      element("groups").append(groupCell); lastGroup = name;
    } else groupCell.colSpan += 1;
  });
  let sortKey = null, sortDirection = 1;
  const headerCells = [];
  columns.forEach((column, index) => {
    const th = document.createElement("th"); th.scope = "col"; th.setAttribute("aria-sort", "none");
    if (isNumber(column)) th.classList.add("number");
    if (groupStarts.has(index) && index) th.classList.add("group-start");
    const button = document.createElement("button"); button.type = "button";
    button.append(document.createTextNode(column.label || column.key));
    const mark = document.createElement("span"); mark.className = "sort-mark"; mark.textContent = "↕";
    mark.setAttribute("aria-hidden", "true"); button.append(mark);
    button.addEventListener("click", () => {
      sortDirection = sortKey === column.key ? -sortDirection : 1;
      sortKey = column.key; render();
    });
    th.append(button); element("labels").append(th); headerCells.push({th, mark, key: column.key});
  });
  const searchable = rows.map(row => Object.values(row).map(text).join(" ").toLocaleLowerCase("ru"));
  function render() {
    const query = element("search").value.trim().toLocaleLowerCase("ru");
    const dataset = element("dataset").value, sampling = element("sampling").value, combo = element("combo").value;
    let selected = rows.map((row, index) => ({row, index})).filter(({row, index}) =>
      (!dataset || text(row.dataset) === dataset) &&
      (row.row_type === "baseline" || ((!sampling || text(row.sampling) === sampling) && (!combo || text(row.combo) === combo))) &&
      (!query || searchable[index].includes(query)));
    const column = columns.find(candidate => candidate.key === sortKey);
    if (column) selected.sort((a, b) => {
      const left = a.row[sortKey], right = b.row[sortKey];
      const leftMissing = empty(left) || (isNumber(column) && !Number.isFinite(Number(left)));
      const rightMissing = empty(right) || (isNumber(column) && !Number.isFinite(Number(right)));
      if (leftMissing || rightMissing) return leftMissing === rightMissing ? a.index - b.index : leftMissing ? 1 : -1;
      const comparison = isNumber(column) ? Number(left) - Number(right) : text(left).localeCompare(text(right), "ru", {numeric: true});
      return comparison * sortDirection || a.index - b.index;
    });
    for (const header of headerCells) {
      const active = header.key === sortKey;
      header.th.setAttribute("aria-sort", active ? (sortDirection === 1 ? "ascending" : "descending") : "none");
      header.mark.textContent = active ? (sortDirection === 1 ? "↑" : "↓") : "↕";
    }
    const fragment = document.createDocumentFragment();
    for (const {row} of selected) {
      const tr = document.createElement("tr");
      if (row.row_type === "baseline") { tr.className = "baseline"; tr.title = "Baseline model"; }
      columns.forEach((column, index) => {
        const td = document.createElement("td"); const value = row[column.key];
        if (isNumber(column)) td.classList.add("number");
        if (groupStarts.has(index) && index) td.classList.add("group-start");
        if (empty(value)) td.classList.add("missing");
        const link = (column.key === "source_result" || column.key === "source_link") && safeLink(row.source_link);
        if (link) {
          const anchor = document.createElement("a"); anchor.href = link;
          anchor.textContent = "Source result ↗"; anchor.title = text(row.source_result || row.source_link);
          td.append(anchor);
        } else td.textContent = display(value, column);
        if (isNumber(column) && !empty(value)) td.title = text(value);
        tr.append(td);
      });
      fragment.append(tr);
    }
    if (!selected.length) {
      const tr = document.createElement("tr"), td = document.createElement("td");
      td.colSpan = Math.max(columns.length, 1); td.className = "empty";
      td.textContent = "No rows match the filters."; tr.append(td); fragment.append(tr);
    }
    element("body").replaceChildren(fragment);
    element("visible").textContent = `Rows shown: ${selected.length} of ${rows.length}. Scroll right to see more columns.`;
  }
  for (const id of ["dataset", "sampling", "combo", "search"]) element(id).addEventListener("input", render);
  element("reset").addEventListener("click", () => {
    for (const id of ["dataset", "sampling", "combo", "search"]) element(id).value = "";
    sortKey = null; sortDirection = 1; render();
  });
  render();
})();
</script>
</body>
</html>
'''
