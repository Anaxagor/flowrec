"""Paired ranking and exposure diagnostics, independent of model inference."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    if x.sum() == 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum())
                 - (len(x) + 1) / len(x))


def _bootstrap_mean(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Paired user bootstrap; does not represent uncertainty across training seeds."""
    if samples <= 0:
        return float("nan"), float("nan")
    means = np.empty(samples)
    # Bound memory independently of the number of users and bootstrap replicates.
    for start in range(0, samples, 32):
        end = min(start + 32, samples)
        ix = rng.integers(len(values), size=(end - start, len(values)))
        means[start:end] = values[ix].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def _overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Row lists have no duplicates; -1 denotes an unfilled/invalid slot.
    return ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)).any(axis=2).sum(axis=1) / a.shape[1]


def analyze_bundles(
    specs: list[dict[str, Any]],
    bundles: dict[str, dict[str, np.ndarray]],
    catalog: pd.DataFrame,
    ks: tuple[int, ...] = (5, 10, 20),
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Compare runs on exactly the same users and held-out targets.

    Coverage counts catalog items appearing at least once. Popularity is a
    fixed count from training interactions. Neither quantity is a measure of
    semantic diversity within one user's list or model probability entropy.
    ``mean_pairwise_overlap`` is the exact average |A intersect B| / K over
    distinct users, computed from item exposure counts without quadratic work.
    """
    if not specs:
        raise ValueError("No runs supplied")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or min(ks) < 1:
        raise ValueError("ks must contain positive cutoffs")
    run_ids = [s["run_id"] for s in specs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run_id")
    cat = catalog.copy()
    cat["item_id"] = cat["item_id"].astype(str)
    if len(cat) == 0 or cat.item_id.duplicated().any() or cat.item_id.eq("").any():
        raise ValueError("Catalog must contain unique, nonempty item IDs")
    popularity = pd.to_numeric(cat["popularity"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(popularity).all() or (popularity < 0).any():
        raise ValueError("Popularity must be finite and nonnegative")
    cat_ids = cat.item_id.to_numpy(dtype=str)
    index = pd.Index(cat_ids)
    percentile = pd.Series(popularity).rank(method="average", pct=True).to_numpy()
    groups = np.where(percentile > .8, "Head (top 20%)",
                      np.where(percentile > .5, "Middle (50–80%)", "Tail (bottom 50%)"))
    cat["popularity_group"] = groups
    encoded: dict[str, dict[str, np.ndarray]] = {}
    reference = None
    reference_contexts = None
    has_contexts = ["history_fingerprints" in bundles[rid] for rid in run_ids]
    if any(has_contexts) and not all(has_contexts):
        raise ValueError("History fingerprints must be present for every run or none")
    by_id = {s["run_id"]: s for s in specs}
    for spec in specs:
        rid = spec["run_id"]
        b = bundles[rid]
        users = np.asarray(b["user_ids"], dtype=str)
        targets = np.asarray(b["target_item_ids"], dtype=str)
        histories = np.asarray(b["history_lengths"], dtype=int)
        recs = np.asarray(b["recommendations"], dtype=str)
        if (users.ndim != 1 or not len(users) or len(set(users)) != len(users)
                or targets.shape != users.shape or histories.shape != users.shape
                or recs.ndim != 2 or recs.shape[0] != len(users)):
            raise ValueError(f"{rid}: invalid arrays or duplicate users")
        if recs.shape[1] < max(ks):
            raise ValueError(f"{rid}: only {recs.shape[1]} ranks available, requested {max(ks)}")
        order = np.argsort(users)
        users, targets, histories, recs = users[order], targets[order], histories[order], recs[order]
        if all(has_contexts):
            contexts = np.asarray(b["history_fingerprints"], dtype=str)
            if contexts.shape != users.shape:
                raise ValueError(f"{rid}: invalid history fingerprints")
            contexts = contexts[order]
            if reference_contexts is None:
                reference_contexts = contexts
            elif not np.array_equal(contexts, reference_contexts):
                raise ValueError(f"{rid}: mismatched history contents")
        if reference is None:
            reference = (users, targets, histories)
        else:
            for name, arr, ref in zip(("users", "targets", "history lengths"),
                                      (users, targets, histories), reference):
                if not np.array_equal(arr, ref):
                    raise ValueError(f"{rid}: mismatched {name}; comparisons require the same test split")
        target_idx = index.get_indexer(targets)
        rec_idx = index.get_indexer(recs.ravel()).reshape(recs.shape)
        if (target_idx < 0).any() or ((rec_idx < 0) & (recs != "")).any():
            raise ValueError(f"{rid}: unknown item outside the shared catalog")
        for row in rec_idx:
            valid = row[row >= 0]
            if len(valid) != len(np.unique(valid)):
                raise ValueError(f"{rid}: duplicate recommendation within a user list")
        encoded[rid] = dict(users=users, targets=target_idx, histories=histories, recs=rec_idx)
        if spec["stage"] == "sft":
            baseline = spec.get("baseline_run_id")
            if baseline not in by_id or by_id[baseline]["stage"] != "init" or by_id[baseline]["model"] != spec["model"]:
                raise ValueError(f"{rid}: missing or incompatible baseline_run_id")

    summaries, items, popularity_rows = [], [], []
    user_rows, comparisons, cross, attribution = [], [], [], []
    statistics = {}
    rng = np.random.default_rng(seed)
    for spec in specs:
        rid = spec["run_id"]
        b = encoded[rid]
        n = len(b["users"])
        common = {key: spec.get(key) for key in ("run_id", "model", "stage", "seed")}
        for k in ks:
            rec = b["recs"][:, :k]
            valid = rec >= 0
            hit = rec == b["targets"][:, None]
            ndcg = (hit / np.log2(np.arange(k) + 2)).sum(axis=1)
            counts = np.bincount(rec[valid], minlength=len(cat))
            probs = counts[counts > 0] / max(int(counts.sum()), 1)
            entropy = float(-(probs * np.log(probs)).sum())
            per_pop = np.where(valid, popularity[np.maximum(rec, 0)], 0)
            user_arp = np.divide(per_pop.sum(axis=1), valid.sum(axis=1),
                                 out=np.zeros(n), where=valid.sum(axis=1) > 0)
            # ARP matches the project's evaluation: mean over valid recommendation slots.
            arp = float(per_pop.sum() / max(int(valid.sum()), 1))
            stats = dict(rec=rec, ndcg=ndcg, hr=hit.any(axis=1), counts=counts, arp=user_arp)
            statistics[(rid, k)] = stats
            summaries.append({**common, "k": k, "num_users": n, "catalog_size": len(cat),
                              "ndcg": float(ndcg.mean()), "hr": float(hit.any(axis=1).mean()),
                              "coverage": float((counts > 0).mean()), "arp": arp,
                              "unique_items": int((counts > 0).sum()),
                              "valid_slot_rate": float(valid.mean()),
                              "exposure_gini": _gini(counts), "exposure_entropy": entropy,
                              "effective_catalog": float(np.exp(entropy)),
                              "mean_pairwise_overlap": float(np.dot(counts, counts - 1) / (n * (n - 1) * k)) if n > 1 else float("nan")})
            for group in ("Tail (bottom 50%)", "Middle (50–80%)", "Head (top 20%)"):
                mask = groups == group
                popularity_rows.append({**common, "k": k, "popularity_group": group,
                                        "catalog_items": int(mask.sum()),
                                        "exposures": int(counts[mask].sum()),
                                        "exposure_share": float(counts[mask].sum() / max(int(counts.sum()), 1)),
                                        "target_share": float((groups[b["targets"]] == group).mean())})

    summary = pd.DataFrame(summaries)
    for spec in specs:
        rid = spec["run_id"]
        common = {key: spec.get(key) for key in ("run_id", "model", "stage", "seed")}
        b = encoded[rid]
        n = len(b["users"])
        for k in ks:
            stats = statistics[(rid, k)]
            baseline_id = spec.get("baseline_run_id") if spec["stage"] == "sft" else None
            before = statistics[(baseline_id, k)] if baseline_id else stats
            counts, before_counts = stats["counts"], before["counts"]
            new = (counts > 0) & (before_counts == 0)
            lost = (counts == 0) & (before_counts > 0)
            status = np.where(new, "new", np.where(lost, "lost", np.where(counts > 0, "retained", "unseen")))
            frame = cat.copy()
            for key, value in {**common, "k": k}.items():
                frame[key] = value
            frame["exposures"] = counts
            frame["baseline_exposures"] = before_counts
            frame["status"] = status
            items.append(frame)
            if baseline_id is None:
                continue
            delta = stats["ndcg"] - before["ndcg"]
            overlap = _overlap(stats["rec"], before["rec"])
            ci_low, ci_high = _bootstrap_mean(delta, bootstrap_samples, rng)
            after_row = summary[(summary.run_id == rid) & (summary.k == k)].iloc[0]
            before_row = summary[(summary.run_id == baseline_id) & (summary.k == k)].iloc[0]
            comp = {**common, "baseline_run_id": baseline_id, "k": k,
                    "mean_list_overlap": float(overlap.mean()),
                    "new_catalog_items": int(new.sum()), "lost_catalog_items": int(lost.sum()),
                    "net_catalog_items": int(new.sum() - lost.sum()),
                    "new_item_exposure_share": float(counts[new].sum() / max(int(counts.sum()), 1)),
                    "ndcg_ci_low": ci_low, "ndcg_ci_high": ci_high,
                    "users_improved": int((delta > 0).sum()), "users_worsened": int((delta < 0).sum()),
                    "users_unchanged": int((delta == 0).sum())}
            for metric in ("ndcg", "hr", "coverage", "arp", "exposure_gini", "mean_pairwise_overlap"):
                diff = float(after_row[metric] - before_row[metric])
                comp[f"delta_{metric}"] = diff
                comp[f"relative_delta_{metric}"] = diff / float(before_row[metric]) if before_row[metric] else float("nan")
            comparisons.append(comp)
            rec = stats["rec"]
            new_slots = new[np.maximum(rec, 0)] & (rec >= 0)
            outcome = np.where(delta > 0, "improved", np.where(delta < 0, "worsened", "unchanged"))
            user_frame = pd.DataFrame({"user_id": b["users"], "target_item_id": cat_ids[b["targets"]],
                                       "history_length": b["histories"], "target_popularity": popularity[b["targets"]],
                                       "target_popularity_group": groups[b["targets"]],
                                       "ndcg_before": before["ndcg"], "ndcg_after": stats["ndcg"],
                                       "delta_ndcg": delta, "ndcg_outcome": outcome,
                                       "list_overlap": overlap, "new_item_share": new_slots.mean(axis=1),
                                       "arp_before": before["arp"], "arp_after": stats["arp"],
                                       "target_newly_covered": new[b["targets"]],
                                       "hit_before": before["hr"], "hit_after": stats["hr"]})
            for key, value in {**common, "baseline_run_id": baseline_id, "k": k}.items():
                user_frame[key] = value
            user_rows.append(user_frame)
            for label, selected in (("newly covered targets", new[b["targets"]]),
                                    ("previously covered targets", before_counts[b["targets"]] > 0),
                                    ("never covered targets", (counts[b["targets"]] == 0) & (before_counts[b["targets"]] == 0))):
                attribution.append({**common, "k": k, "target_group": label,
                                    "num_users": int(selected.sum()),
                                    "delta_ndcg_contribution": float(delta[selected].sum() / n),
                                    "mean_delta_ndcg": float(delta[selected].mean()) if selected.any() else 0.0})

    for left, right in combinations(specs, 2):
        for k in ks:
            a, b = statistics[(left["run_id"], k)], statistics[(right["run_id"], k)]
            covered_a, covered_b = a["counts"] > 0, b["counts"] > 0
            ra = pd.Series(a["counts"]).rank().to_numpy()
            rb = pd.Series(b["counts"]).rank().to_numpy()
            correlation = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")
            cross.append({"run_a": left["run_id"], "run_b": right["run_id"],
                          "model_a": left["model"], "model_b": right["model"],
                          "stage_a": left["stage"], "stage_b": right["stage"], "k": k,
                          "mean_list_overlap": float(_overlap(a["rec"], b["rec"]).mean()),
                          "catalog_jaccard": float((covered_a & covered_b).sum() / max(int((covered_a | covered_b).sum()), 1)),
                          "exposure_spearman": correlation})
    return {"summary": summary, "comparisons": pd.DataFrame(comparisons),
            "items": pd.concat(items, ignore_index=True),
            "users": pd.concat(user_rows, ignore_index=True) if user_rows else pd.DataFrame(),
            "popularity": pd.DataFrame(popularity_rows), "cross_model": pd.DataFrame(cross),
            "accuracy_attribution": pd.DataFrame(attribution)}
