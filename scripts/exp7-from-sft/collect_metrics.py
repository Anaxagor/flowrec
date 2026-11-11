#!/usr/bin/env python3
"""Collect completed from-SFT results, keeping historical protocols distinct.

The exp8 jobs directory is the canonical, one-file-per-cell index. Its
source-level baseline is collected separately from continued training. The older
TIGER grid stores one CSV row per evaluation split; its full experimental
identity is used to pair split rows. Incomplete pairs remain in the output.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXP8 = ROOT / "experiments/exp8-from-sft"
LEGACY = ROOT / "artifacts/exp4-reward_ablation"
LEGACY_RESULTS = ROOT / "experiments/exp4-gflow-tiger-from-sft"
OUTPUT = ROOT / "artifacts/exp8-from-sft/all_metrics.csv"
LEGACY_ID = (
    "dataset", "combo", "power", "objective", "num_negatives", "seed",
    "variant", "target_grad_ratio", "sasrec_transform", "sasrec_temperature",
    "interaction_exponent", "sampler",
)
FIRST_COLUMNS = (
    "protocol", "model", "dataset", "source_id", "source_seed", "seed",
    "arm", "variant", "status", "combo", "objective", "sampling", "power",
    "rho", "num_samples", "num_negatives", "lambda_gflow", "lambda_distill",
    "best_epoch", "epochs_completed", "best_val_ndcg5", "completeness", "available_splits", "available_metrics",
    "configuration_id", "source_result",
)


def flatten(value, prefix=""):
    """Flatten mappings without dropping unknown fields; lists remain JSON."""
    if not isinstance(value, dict):
        return {prefix: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value}
    out = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else key
        out.update(flatten(child, name))
    return out


def relative(path):
    return str(path.relative_to(ROOT))


def exp8_rows():
    rows = []
    # The job record contains experiment_cell, including the intended arm.
    # The native run directories duplicate these results and are not scanned.
    for path in sorted(EXP8.glob("*/sft_seed_*/jobs/*/result.json")):
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict):
            raise ValueError(f"Result is not an object: {path}")
        cell = data.get("experiment_cell") or {}
        provenance = data.get("from_sft_source") or {}
        if not isinstance(cell, dict) or not isinstance(provenance, dict):
            raise ValueError(f"Invalid cell or provenance in {path}")
        row = flatten({k: v for k, v in data.items() if k not in {"splits", "experiment_cell", "from_sft_source"}})
        row.update(flatten(cell, "experiment_cell"))
        row.update(flatten(provenance, "from_sft_source"))
        # Canonical configuration takes precedence over variant-specific metadata.
        for key, value in cell.items():
            if isinstance(value, list) and key == "combo":
                value = "_x_".join(map(str, value))
            elif isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            row[key] = value
        for key in ("source_id", "source_seed", "model", "dataset"):
            if row.get(key) is None and key in provenance:
                row[key] = provenance[key]
        splits = data.get("splits") or {}
        if not isinstance(splits, dict):
            raise ValueError(f"Invalid splits in {path}")
        for split, metrics in splits.items():
            if not isinstance(metrics, dict):
                raise ValueError(f"Invalid {split} metrics in {path}")
            row.update(flatten(metrics, split))
        row.update(protocol="exp8_from_sft", configuration_id=path.parent.name,
                   completeness="complete_json" if {"test", "validation"}.issubset(splits) else "partial_json",
                   source_result=relative(path), available_splits=",".join(sorted(splits)))
        rows.append(row)
    return rows


def exp8_baseline_rows():
    """Collect the canonical initial-checkpoint evaluation once per source."""
    rows = []
    # The CE subdirectory contains a duplicate with a different structure; use
    # only the top-level source baseline created by prepare(kind="gpu").
    for path in sorted(EXP8.glob("*/sft_seed_*/baseline/init_metrics.json")):
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict) or not isinstance(data.get("from_sft_source"), dict):
            raise ValueError(f"Missing baseline source identity in {path}")
        provenance = data["from_sft_source"]
        required = ("source_id", "source_seed", "model", "dataset")
        if any(provenance.get(key) is None for key in required):
            raise ValueError(f"Incomplete baseline source identity in {path}")
        if (path.parents[2].name != str(provenance["source_id"])
                or path.parents[1].name != f"sft_seed_{provenance['source_seed']}"):
            raise ValueError(f"Baseline source identity disagrees with its path: {path}")
        splits = {}
        for split in ("validation", "test"):
            if split in data:
                if not isinstance(data[split], dict):
                    raise ValueError(f"Invalid baseline {split} metrics in {path}")
                if data[split]:
                    splits[split] = data[split]
        row = flatten({k: v for k, v in data.items()
                       if k not in {"validation", "test", "from_sft_source"}})
        row.update(flatten(provenance, "from_sft_source"))
        row.update({key: provenance[key] for key in required})
        for split, metrics in splits.items():
            row.update(flatten(metrics, split))
        complete = {"test", "validation"}.issubset(splits)
        row.update(protocol="exp8_from_sft", arm="init", variant="init",
                   status="completed" if complete else "partial",
                   completeness="complete_json" if complete else "partial_json",
                   source_result=relative(path), available_splits=",".join(sorted(splits)))
        rows.append(row)
    return rows


def is_metric(name):
    return "@" in name or name == "num_examples"


def legacy_rows():
    output = []
    for path in sorted(LEGACY.glob("gflow_tiger_from_sft_*_sft_seed_42.csv")):
        grouped = defaultdict(dict)
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                missing = set(LEGACY_ID + ("split",)) - set(row)
                if missing:
                    raise ValueError(f"Missing legacy columns {missing} in {path}")
                split = row["split"]
                if not split:
                    raise ValueError(f"Missing split in {path}")
                key = tuple(row[k] for k in LEGACY_ID)
                if split in grouped[key]:
                    raise ValueError(f"Duplicate legacy split {split} for {key} in {path}")
                grouped[key][split] = row
        for key, splits in sorted(grouped.items()):
            base = dict(zip(LEGACY_ID, key))
            row = dict(base)
            row.update(protocol="legacy_tiger_gflow", model="tiger",
                       source_id=f"tiger_{base['dataset']}", source_seed="42",
                       arm="init" if base["variant"] == "init" else "gflow",
                       rho=base["target_grad_ratio"], source_result=relative(path),
                       completeness="complete_csv" if {"test", "validation"}.issubset(splits) else "partial_csv",
                       available_splits=",".join(sorted(splits)))
            # Keep split-specific evaluation data in separate columns. For any
            # other inconsistent metadata, retain both values explicitly.
            columns = set().union(*(value.keys() for value in splits.values()))
            for column in columns - set(LEGACY_ID) - {"split"}:
                values = {split: data.get(column, "") for split, data in splits.items()}
                nonempty = {v for v in values.values() if v != ""}
                if is_metric(column) or len(nonempty) > 1:
                    for split, value in values.items():
                        row[f"{split}.{column}"] = value
                elif nonempty:
                    row[column] = next(iter(nonempty))
            output.append(row)
    return output


def legacy_cell_key(row):
    """Normalize numeric configuration values for matching CSV to JSON."""
    try:
        return (row["dataset"], row["combo"], Decimal(str(row["power"])),
                row["objective"], int(row["num_negatives"]), int(row["seed"]),
                Decimal(str(row.get("rho", row.get("target_grad_ratio")))))
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError(f"Incomplete legacy configuration: {row}") from exc


def same_value(left, right):
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except InvalidOperation:
        return str(left) == str(right)


def merge_legacy_json(rows):
    """Add full run results, preserving matching CSV-only fields and provenance."""
    indexed = {}
    for row in rows:
        if row.get("protocol") == "legacy_tiger_gflow" and row.get("arm") == "gflow":
            key = legacy_cell_key(row)
            if key in indexed:
                raise ValueError(f"Duplicate legacy CSV configuration: {key}")
            indexed[key] = row
    seen = set()
    for path in sorted(LEGACY_RESULTS.glob("*/sft_only_lr3e-05/sft_seed_*/**/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid legacy result in {path}")
        key = legacy_cell_key(data)
        parts = path.relative_to(LEGACY_RESULTS).parts
        if parts[0] != data["dataset"] or not parts[2].startswith("sft_seed_"):
            raise ValueError(f"Legacy source identity disagrees with path: {path}")
        source_seed = int(parts[2].removeprefix("sft_seed_"))
        if key in seen:
            raise ValueError(f"Duplicate legacy JSON configuration: {key}")
        seen.add(key)
        row = indexed.get(key)
        if row is None:
            row = {"protocol": "legacy_tiger_gflow", "model": "tiger",
                   "source_id": f"tiger_{data['dataset']}", "source_seed": source_seed,
                   "arm": "gflow", "variant": "gflow"}
            rows.append(row)
        else:
            if int(row["source_seed"]) != source_seed:
                raise ValueError(f"Legacy source seed mismatch: {path}")
            row["supplemental_source_result"] = row["source_result"]
        splits = data.get("splits") or {}
        if not isinstance(splits, dict) or any(not isinstance(v, dict) for v in splits.values()):
            raise ValueError(f"Invalid legacy splits in {path}")
        fields = flatten({k: v for k, v in data.items() if k != "splits"})
        for split, metrics in splits.items():
            fields.update(flatten(metrics, split))
        for column, value in fields.items():
            if value is None:
                continue
            old = row.get(column)
            if old not in (None, "", "-") and not same_value(old, value):
                raise ValueError(f"Legacy CSV and JSON disagree on {column}: {path} ({old!r}, {value!r})")
            row[column] = value
        row["rho"] = data["target_grad_ratio"]
        available = set(filter(None, row.get("available_splits", "").split(","))) | set(splits)
        row["available_splits"] = ",".join(sorted(available))
        row["completeness"] = "complete_json" if {"test", "validation"}.issubset(splits) else (
            "complete_csv_and_json" if {"test", "validation"}.issubset(available) else "partial_json")
        row["source_result"] = relative(path)
    return rows


def merge_legacy_baseline_json(rows):
    """Keep original legacy evaluations even when their summary CSV is absent."""
    indexed = {}
    for row in rows:
        if row.get("protocol") != "legacy_tiger_gflow" or row.get("arm") != "init":
            continue
        key = (row["dataset"], int(row["source_seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate legacy baseline: {key}")
        indexed[key] = row
    for path in sorted(LEGACY_RESULTS.glob("*/sft_only_lr3e-05/sft_seed_*/baseline/init_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(LEGACY_RESULTS).parts
        dataset, source_seed = parts[0], int(parts[2].removeprefix("sft_seed_"))
        checkpoint = data.get("checkpoint") if isinstance(data, dict) else None
        expected_suffix = ("tiger", dataset, "sft_only_lr3e-05", f"seed_{source_seed}", "best.pt")
        if (not isinstance(data, dict) or data.get("dataset") != dataset
                or not isinstance(checkpoint, str)
                or Path(checkpoint).parts[-5:] != expected_suffix
                or data.get("source_seed", source_seed) != source_seed):
            raise ValueError(f"Legacy baseline source identity disagrees with path: {path}")
        fields = flatten({k: v for k, v in data.items() if k not in {"validation", "test"}})
        splits = set()
        for split in ("validation", "test"):
            metrics = data.get(split, {})
            if not isinstance(metrics, dict):
                raise ValueError(f"Invalid legacy baseline {split} metrics: {path}")
            if metrics:
                splits.add(split)
                fields.update(flatten(metrics, split))
        fields.update(model="tiger", source_id=f"tiger_{dataset}", source_seed=source_seed)
        row = indexed.get((dataset, source_seed))
        if row is None:
            row = {"protocol": "legacy_tiger_gflow", "arm": "init", "variant": "init"}
            rows.append(row)
        else:
            row["supplemental_source_result"] = row["source_result"]
        for column, value in fields.items():
            old = row.get(column)
            if old not in (None, "", "-") and not same_value(old, value):
                raise ValueError(f"Legacy baseline CSV and JSON disagree on {column}: {path}")
            row[column] = value
        available = set(filter(None, row.get("available_splits", "").split(","))) | splits
        complete = {"validation", "test"}.issubset(available)
        row.update(status="completed" if complete else "partial", source_result=relative(path),
                   available_splits=",".join(sorted(available)),
                   completeness="complete_json" if len(splits) == 2 else (
                       "complete_csv_and_json" if complete else "partial_json"))
    return rows


def write_csv(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    ordered = [key for key in FIRST_COLUMNS if key in columns]
    ordered.extend(sorted(columns - set(ordered)))
    fd, temp_name = tempfile.mkstemp(prefix=".all_metrics.", suffix=".tmp", dir=OUTPUT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, OUTPUT)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    rows = exp8_rows() + exp8_baseline_rows() + merge_legacy_baseline_json(merge_legacy_json(legacy_rows()))
    rows.sort(key=lambda x: tuple(str(x.get(k) or "") for k in (
        "protocol", "model", "dataset", "arm", "combo", "power", "rho", "objective", "sampling", "configuration_id")))
    if not rows:
        raise SystemExit("No metrics found; existing output was not changed.")
    write_csv(rows)
    counts = Counter((x["protocol"], x.get("model"), x.get("dataset"), x.get("arm")) for x in rows)
    print(f"Wrote {len(rows)} rows to {relative(OUTPUT)}")
    for key, count in sorted(counts.items()):
        print(" / ".join(str(x or "") for x in key), count)
    incomplete = sum(not {"test", "validation"}.issubset(set(x["available_splits"].split(","))) for x in rows)
    print(f"Rows missing validation or test: {incomplete}")


if __name__ == "__main__":
    main()
