"""Discover first/second SFT-only and fine-tuning from the first SFT-only.

Only the four requested source families enter discovery. Fine-tuning selection
uses test metrics by default and keeps every seed of a selected config.
Historical paths are preserved; missing weights are reported, never substituted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

DATASETS = ("beauty", "yambda_50m_listens")
MODELS = ("gpt2", "tiger")
STAGES = ("sft", "sft2", "gflow", "ce")
EXCLUDED_GROUPS: set[tuple[str, str, str]] = set()
OLD = "checkpoints/exp2-tuned_params"
BASE = {
    ("gpt2", "beauty"): f"{OLD}/initial/gpt2/beauty/best_model.pt",
    ("gpt2", "yambda_50m_listens"): f"{OLD}/gpt2_tiger_matched/yambda_50m_listens/trials/trial_36/best.pt",
    ("tiger", "beauty"): f"{OLD}/initial/tiger/beauty/best.pt",
    ("tiger", "yambda_50m_listens"): f"{OLD}/initial/tiger/yambda_50m/checkpoints/best.pt",
}
SOURCE_IDS = {
    ("gpt2", "beauty"): "gpt2_beauty",
    ("gpt2", "yambda_50m_listens"): "gpt2_yambda50_trial36",
    ("tiger", "beauty"): "tiger_beauty",
    ("tiger", "yambda_50m_listens"): "tiger_yambda50_matched",
}
GPT2_FAMILIES = {
    "beauty": "experiments/exp4-gflow-gpt2-amazon-fullhist-dense-prefix/beauty",
    "yambda_50m_listens": "experiments/exp4-gflow-gpt2-yambda50-trial36-k3/yambda_50m_listens",
}
TIGER_FAMILIES = {
    "beauty": ("finetuned", "exp2-sft-finetune"),
    "yambda_50m_listens": ("finetuned_matched", "exp2-sft-finetune-matched"),
}
PRIMARY_SFT = {
    ("gpt2", dataset): f"{family}/sft_only/seed_42/checkpoints/best.pt"
    for dataset, family in GPT2_FAMILIES.items()
}
PRIMARY_SFT.update({
    ("tiger", dataset): f"{OLD}/{promoted}/tiger/{dataset}/sft_only_lr3e-05/seed_42/best.pt"
    for dataset, (promoted, _) in TIGER_FAMILIES.items()
})
SOURCE_PROTOCOLS = {
    ("gpt2", "beauty"): "dense_window_v1",
    ("gpt2", "yambda_50m_listens"): "matched_trial36",
    ("tiger", "beauty"): "finetuned",
    ("tiger", "yambda_50m_listens"): "finetuned_matched",
}
EXP8 = "experiments/exp8-from-sft"
LEGACY_TIGER = "experiments/exp4-gflow-tiger-from-sft/beauty/sft_only_lr3e-05"


def portable(value: str | Path) -> str:
    """Remove a recorded checkout prefix without changing a checkpoint path."""
    path = Path(value)
    if path.is_absolute():
        indices = [path.parts.index(k) for k in ("checkpoints", "experiments", "artifacts", "data", "datasets", "configs") if k in path.parts]
        if indices:
            return Path(*path.parts[min(indices):]).as_posix()
    return path.as_posix()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _score(metrics: dict[str, Any]) -> tuple[float, float] | None:
    try:
        result = float(metrics["NDCG@10"]), float(metrics["Coverage@10"])
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(x) for x in result) else None


def select_three(candidates: list[dict[str, Any]], split: str = "test") -> list[dict[str, Any]]:
    """Select maxima and equal-weight normalized Pareto compromise.

    Pareto normalization uses bounds over the non-dominated front. A constant
    coordinate is assigned normalized value one. Ties use NDCG, coverage, then
    the configuration path. Identical selections merge labels, not seeds.
    """
    if split not in {"test", "validation"}:
        raise ValueError("selection split must be test or validation")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if _score(candidate.get("selection_metrics", {})) is not None:
            grouped.setdefault(candidate["configuration_id"], []).append(candidate)
    scores = {key: tuple(mean(_score(v["selection_metrics"])[i] for v in values) for i in (0, 1))
              for key, values in grouped.items()}
    if not scores:
        return []
    front = [key for key, (n, c) in scores.items() if not any(
        n2 >= n and c2 >= c and (n2 > n or c2 > c) for n2, c2 in scores.values())]
    bounds = [(min(scores[key][i] for key in front), max(scores[key][i] for key in front)) for i in (0, 1)]

    def distance(key: str) -> float:
        return sum((0.0 if hi == lo else 1.0 - (scores[key][i] - lo) / (hi - lo)) ** 2
                   for i, (lo, hi) in enumerate(bounds))

    chosen = {
        "best_ndcg": min(scores, key=lambda k: (-scores[k][0], -scores[k][1], k)),
        "best_coverage": min(scores, key=lambda k: (-scores[k][1], -scores[k][0], k)),
        "pareto": min(front, key=lambda k: (distance(k), -scores[k][0], -scores[k][1], k)),
    }
    output = []
    for key in dict.fromkeys(chosen.values()):
        for spec in sorted(grouped[key], key=lambda s: (s.get("seed", 0), s["checkpoint"])):
            output.append({**spec, "selection_labels": [label for label, selected in chosen.items() if selected == key],
                           "selection_split": split, "selection_aggregation": "mean over saved seeds per configuration",
                           "selection_mean_metrics": dict(zip(("NDCG@10", "Coverage@10"), scores[key])),
                           "selection_candidate_count": len(scores), "pareto_front_size": len(front),
                           "pareto_rule": "minimum squared Euclidean distance to (1,1), min-max normalization over Pareto front; constant coordinate=1; ties NDCG, coverage, path"})
    return output



def first_checkpoint(model: str, dataset: str, seed: int = 42) -> str:
    return PRIMARY_SFT[model, dataset].replace("/seed_42/", f"/seed_{seed}/")


def _spec(root: Path, model: str, dataset: str, stage: str, checkpoint: str,
          seed: int, **extra: Any) -> dict[str, Any]:
    checkpoint = portable(checkpoint)
    suffix = hashlib.sha256(checkpoint.encode()).hexdigest()[:10]
    return {"run_id": f"{model}_{dataset}_{stage}_seed{seed}_{suffix}",
            "model": model, "dataset": dataset, "stage": stage, "split": "test", "seed": seed,
            "checkpoint": checkpoint, "checkpoint_kind": "best",
            "preserve_legacy_checkpoint_path": True, "available": (root / checkpoint).is_file(),
            "best_trial": None, "selection_labels": [stage], "source_id": SOURCE_IDS[model, dataset],
            "data_config_selector": [], **extra}


def validate_run_scope(spec: dict[str, Any]) -> None:
    """Validate frozen manifest lineage without loading weights or source files."""
    key = spec.get("model"), spec.get("dataset")
    if key not in SOURCE_IDS or spec.get("stage") not in STAGES:
        raise ValueError(f"Run outside requested model/dataset/stages: {spec.get('run_id')}")
    model, dataset = key
    if spec.get("source_id") != SOURCE_IDS[key]:
        raise ValueError("Wrong first SFT-only source identity")
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid run seed")
    checkpoint = portable(spec.get("checkpoint", ""))
    kind = spec.get("checkpoint_kind", "best")
    if kind not in {"best", "last"} or Path(checkpoint).name != f"{kind}.pt":
        raise ValueError("Checkpoint must be the recorded best.pt or last.pt")
    if spec["stage"] == "sft":
        expected = str(Path(first_checkpoint(model, dataset, seed)).with_name(f"{kind}.pt"))
        if checkpoint != expected:
            raise ValueError("First SFT checkpoint is outside the selected trial/variant/seed family")
        if key == ("gpt2", "yambda_50m_listens") and portable(spec.get("init_checkpoint", "")) != BASE[key]:
            raise ValueError("GPT-2 Yambda50 first SFT requires trial_36/best.pt")
        return
    source_seed = spec.get("source_seed")
    if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
        raise ValueError("Missing first SFT-only source seed")
    if portable(spec.get("init_checkpoint", "")) != first_checkpoint(model, dataset, source_seed):
        raise ValueError("Fine-tuning must start from the selected first SFT-only checkpoint")
    family = f"{EXP8}/{SOURCE_IDS[key]}/sft_seed_{source_seed}"
    arm = "sft" if spec["stage"] == "sft2" else spec["stage"]
    legacy = model == "tiger" and dataset == "beauty" and spec["stage"] == "gflow"
    if not checkpoint.startswith(f"{family}/{arm}/") and not (
            legacy and checkpoint.startswith(f"{LEGACY_TIGER}/sft_seed_{source_seed}/")):
        raise ValueError("Checkpoint is not from the requested first SFT-only experiment")
    if spec["stage"] in {"gflow", "ce"} and kind != "best":
        raise ValueError("Fine-tuning selection metrics refer to best.pt")


def _first_tiger(root: Path, dataset: str) -> list[dict[str, Any]]:
    promoted, experiment = TIGER_FAMILIES[dataset]
    directory = root / OLD / promoted / "tiger" / dataset / "sft_only_lr3e-05"
    paths = sorted(directory.glob("seed_*/metrics.json"))
    paths += sorted((root / "experiments" / experiment / dataset / "sft_lr3e-05").glob("seed_*/metrics.json"))
    output, seen = [], set()
    for path in paths:
        result = _read(path)
        if result.get("status") != "completed":
            continue
        seed = int(result["seed"])
        checkpoint = first_checkpoint("tiger", dataset, seed)
        if portable(result.get("best_checkpoint", "")) != checkpoint:
            raise ValueError(f"SFT metadata points outside selected family: {path}")
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        run_dir = path.parent.relative_to(root).as_posix()
        output.append(_spec(root, "tiger", dataset, "sft", checkpoint, seed,
            init_checkpoint=BASE["tiger", dataset], run_dir=run_dir,
            data_config=f"{run_dir}/run_config.json", result_path=str(path.relative_to(root)),
            recorded_test_metrics=result.get("test", {})))
    return output


def _portable_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _portable_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable_config(v) for v in value]
    return portable(value) if isinstance(value, str) and value.startswith("/") else value


def _gpt2_data_config(root: Path, dataset: str, resolved: dict) -> str:
    """Use a captured exp8 protocol only when bound to this original SFT family."""
    original = f"{GPT2_FAMILIES[dataset]}/data_reward_config.json"
    if (root / original).is_file():
        return original
    directory = root / EXP8 / SOURCE_IDS["gpt2", dataset]
    for path in sorted(directory.glob("sft_seed_*/data_reward_config.json")):
        source_seed = int(path.parent.name.removeprefix("sft_seed_"))
        identity = _read(path.with_name("source_identity.json"))
        config = _read(path)
        _check_source(root, path, config, "gpt2", dataset, source_seed, identity)
        source = root / first_checkpoint("gpt2", dataset, source_seed)
        captured = _read(source.parents[1] / "resolved_config.json")
        if not captured or portable(captured.get("init_checkpoint", "")) != portable(resolved["init_checkpoint"]):
            raise ValueError(f"Cannot bind captured evaluation protocol to first SFT: {path}")
        # Other seeds may share this source's evaluation protocol, but only
        # after checking their original preprocessing and model records agree.
        for key in ("model", "data", "input_pipeline"):
            if _portable_config(captured.get(key)) != _portable_config(resolved.get(key)):
                raise ValueError(f"Captured evaluation protocol differs in {key}: {path}")
        for key, value in config.get("data", {}).items():
            if key in resolved.get("data", {}) and _portable_config(value) != _portable_config(resolved["data"][key]):
                raise ValueError(f"Captured evaluation data.{key} differs: {path}")
        for key in ("split_strategy", "train_example_layout", "positive_criterion", "dataset_size", "dataset_type"):
            if key in resolved.get("data", {}) and config.get("dataset", {}).get(key) != resolved["data"][key]:
                raise ValueError(f"Captured evaluation dataset.{key} differs: {path}")
        return path.relative_to(root).as_posix()
    # Keep missing original config visible to the standard preflight error.
    return original


def _first_gpt2(root: Path, dataset: str) -> list[dict[str, Any]]:
    output = []
    family = GPT2_FAMILIES[dataset]
    for path in sorted((root / family / "sft_only").glob("seed_*/result.json")):
        result, resolved = _read(path), _read(path.with_name("resolved_config.json"))
        if result.get("status") != "completed":
            continue
        seed = int(result["seed"])
        checkpoint = first_checkpoint("gpt2", dataset, seed)
        if portable(result.get("best_checkpoint", "")) != checkpoint:
            raise ValueError(f"SFT metadata points outside selected family: {path}")
        initial = resolved.get("init_checkpoint")
        if not initial:
            raise ValueError(f"Missing first SFT initialization provenance: {path}")
        run_dir = path.parent.relative_to(root).as_posix()
        spec = _spec(root, "gpt2", dataset, "sft", checkpoint, seed,
            init_checkpoint=portable(initial), source_family=family,
            run_dir=run_dir, resolved_config=f"{run_dir}/resolved_config.json",
            data_config=_gpt2_data_config(root, dataset, resolved), result_path=str(path.relative_to(root)),
            recorded_test_metrics=result.get("splits", {}).get("test", {}))
        validate_run_scope(spec)
        output.append(spec)
    return output


def _check_source(root: Path, path: Path, record: dict, model: str, dataset: str,
                  source_seed: int, identity: dict | None = None) -> None:
    """Require explicit lineage; a directory name alone is insufficient."""
    expected = first_checkpoint(model, dataset, source_seed)
    initial = record.get("initial_checkpoint") or record.get("sft_checkpoint") or record.get("init_checkpoint")
    provenance = record.get("from_sft_source")
    if provenance:
        required = {"source_id": SOURCE_IDS[model, dataset], "source_seed": source_seed,
                    "model": model, "dataset": dataset, "protocol": SOURCE_PROTOCOLS[model, dataset]}
        if any(provenance.get(k) != v for k, v in required.items()):
            raise ValueError(f"Wrong first SFT identity: {path}")
        if not identity or any(provenance.get(k) != identity.get(k) for k in
                               (*required, "checkpoint_sha256", "source_fingerprint", "protocol")):
            raise ValueError(f"Source identity differs from frozen first SFT protocol: {path}")
        if not provenance.get("checkpoint_sha256") or not provenance.get("source_fingerprint"):
            raise ValueError(f"Incomplete source provenance: {path}")
    elif not initial:
        raise ValueError(f"Missing first SFT checkpoint provenance: {path}")
    if initial and portable(initial) != expected:
        # Modern records may name the promoted copy. It is accepted only when
        # its provenance binds it to the same source hash and original path.
        source = root / portable(initial)
        promoted = _read(source.with_name("provenance.json"))
        if (not provenance or promoted.get("sha256") != provenance.get("checkpoint_sha256")
                or portable(promoted.get("source_checkpoint", "")) != expected):
            raise ValueError(f"Wrong first SFT checkpoint: {path}: {initial}")


def _exp8_candidates(root: Path, model: str, dataset: str, split: str) -> list[dict[str, Any]]:
    output = []
    source_id = SOURCE_IDS[model, dataset]
    for path in sorted((root / EXP8 / source_id).glob("sft_seed_*/jobs/*/result.json")):
        result = _read(path)
        if result.get("status") != "completed" or result.get("evaluation_scope", "full") != "full":
            continue
        cell = result.get("experiment_cell", {})
        arm = cell.get("arm")
        if arm not in {"sft", "gflow", "ce"}:
            raise ValueError(f"Missing/unknown exp8 arm: {path}")
        source_dir = path.parents[2]
        source_seed = int(source_dir.name.removeprefix("sft_seed_"))
        if any(cell.get(k) != v for k, v in {"source_id": source_id, "source_seed": source_seed,
                                            "model": model, "dataset": dataset}.items()):
            raise ValueError(f"exp8 cell disagrees with source directory: {path}")
        identity = _read(source_dir / "source_identity.json")
        _check_source(root, path, result, model, dataset, source_seed, identity)
        seed = int(cell["seed"])
        if result.get("seed") != seed:
            raise ValueError(f"exp8 run seed mismatch: {path}")
        run_dir = portable(result["run_dir"])
        checkpoint = result.get("best_checkpoint") or result.get("checkpoint") or f"{run_dir}/checkpoints/best.pt"
        config = {k: v for k, v in cell.items() if k not in {"seed", "source_seed"}}
        config_id = f"exp8/{source_id}/" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        family = source_dir.relative_to(root).as_posix()
        spec = _spec(root, model, dataset, "sft2" if arm == "sft" else arm, checkpoint, seed,
            source_seed=source_seed, source_family=family, from_sft_source=result.get("from_sft_source"),
            init_checkpoint=first_checkpoint(model, dataset, source_seed),
            run_dir=run_dir, resolved_config=f"{run_dir}/resolved_config.json",
            data_config=f"{family}/data_reward_config.json", result_path=str(path.relative_to(root)),
            configuration_id=config_id, selection_parameters=cell,
            selection_metrics=result.get("splits", {}).get(split, {}),
            recorded_test_metrics=result.get("splits", {}).get("test", {}))
        validate_run_scope(spec)
        output.append(spec)
    return output


def _legacy_tiger_candidates(root: Path, split: str) -> list[dict[str, Any]]:
    output = []
    for path in sorted((root / LEGACY_TIGER).glob("sft_seed_*/**/result.json")):
        result = _read(path)
        if result.get("status") != "completed" or result.get("evaluation_scope", "full") != "full":
            continue
        if result.get("objective") not in {"tb", "db", "subtb"}:
            continue
        relative = path.relative_to(root / LEGACY_TIGER)
        source_seed = int(relative.parts[0].removeprefix("sft_seed_"))
        _check_source(root, path, result, "tiger", "beauty", source_seed)
        run_dir = path.parent.relative_to(root).as_posix()
        family = f"{LEGACY_TIGER}/sft_seed_{source_seed}"
        checkpoint = result.get("best_checkpoint") or f"{run_dir}/checkpoints/best.pt"
        spec = _spec(root, "tiger", "beauty", "gflow", checkpoint, int(result["seed"]),
            source_seed=source_seed, source_family=family,
            init_checkpoint=first_checkpoint("tiger", "beauty", source_seed),
            run_dir=run_dir, resolved_config=f"{run_dir}/resolved_config.json",
            data_config=f"{family}/data_reward_config.json", result_path=str(path.relative_to(root)),
            configuration_id="legacy_tiger/" + "/".join(relative.parts[1:-2]),
            selection_parameters={k: result.get(k) for k in ("combo", "power", "objective", "target_grad_ratio", "num_negatives")},
            selection_metrics=result.get("splits", {}).get(split, {}),
            recorded_test_metrics=result.get("splits", {}).get("test", {}))
        validate_run_scope(spec)
        output.append(spec)
    return output


def _checkpoint_kinds(root: Path, specs: list[dict[str, Any]], kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        for kind in kinds:
            checkpoint = str(Path(spec["checkpoint"]).with_name(f"{kind}.pt"))
            # best.pt is the completed run's declared model; last.pt is optional
            # and only exported if the inventory or sidecar demonstrates it.
            if kind == "last" and not any((root / (checkpoint + suffix)).is_file() for suffix in ("", ".json")):
                continue
            suffix = hashlib.sha256(checkpoint.encode()).hexdigest()[:10]
            export_spec = {**spec, "checkpoint": checkpoint, "checkpoint_kind": kind,
                "run_id": f"{spec['model']}_{spec['dataset']}_{spec['stage']}_seed{spec['seed']}_{suffix}",
                "available": (root / checkpoint).is_file()}
            if kind == "last":
                export_spec.pop("recorded_test_metrics", None)
            output.append(export_spec)
    return output


def build_manifest(repo_root: Path, selection_split: str = "test", *,
                   checkpoint_kinds: tuple[str, ...] = ("best",)) -> list[dict[str, Any]]:
    if selection_split not in {"test", "validation"}:
        raise ValueError("selection split must be test or validation")
    if not checkpoint_kinds or set(checkpoint_kinds) - {"best", "last"}:
        raise ValueError("checkpoint_kinds must contain best and/or last")
    root = Path(repo_root).resolve()
    first = [spec for dataset in DATASETS for spec in
             [*_first_gpt2(root, dataset), *_first_tiger(root, dataset)]]
    output = _checkpoint_kinds(root, first, tuple(dict.fromkeys(checkpoint_kinds)))
    for model in MODELS:
        for dataset in DATASETS:
            candidates = _exp8_candidates(root, model, dataset, selection_split)
            if (model, dataset) == ("tiger", "beauty"):
                candidates.extend(_legacy_tiger_candidates(root, selection_split))
            output.extend(_checkpoint_kinds(root, [r for r in candidates if r["stage"] == "sft2"],
                                            tuple(dict.fromkeys(checkpoint_kinds))))
            for arm in ("gflow", "ce"):
                output.extend(select_three([r for r in candidates if r["stage"] == arm], selection_split))
    seen = set()
    for spec in output:
        validate_run_scope(spec)
        if spec["checkpoint"] in seen:
            raise ValueError(f"Duplicate discovered checkpoint: {spec['checkpoint']}")
        seen.add(spec["checkpoint"])
    return output


def missing_groups(runs: list[dict[str, Any]]) -> list[str]:
    present = {(r["model"], r["dataset"], r["stage"]) for r in runs}
    return [f"{dataset}/{model}/{stage}" for dataset in DATASETS for model in MODELS for stage in STAGES
            if (model, dataset, stage) not in present]
