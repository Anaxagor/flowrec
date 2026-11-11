"""Train-popularity and history cohort diagnostics on unchanged recommendation lists.

Popularity buckets rank positive training counts using average ranks / number of
positive-count items: tail <= .5, middle <= .8, head > .8. Zero counts form a
separate unseen bucket. History bins are [0, first_edge], then (left, right],
then (last_edge, infinity); full pre-truncation lengths take precedence.

All metrics use the original list positions. Cohorting on the held-out target
never filters a list before computing HR/NDCG. Diversity here means catalog
exposure, not semantic intra-list diversity or model probability entropy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


POPULARITY_BUCKETS = ("unseen", "tail", "middle", "head")
_DIVERSITY = ("coverage", "arp", "exposure_gini", "exposure_entropy",
              "effective_catalog", "mean_pairwise_overlap", "valid_slot_rate")


def _integers(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    try:
        numeric = raw.astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain nonnegative integers") from exc
    if (not np.isfinite(numeric).all() or (numeric < 0).any()
            or (numeric != np.floor(numeric)).any()):
        raise ValueError(f"{name} must contain nonnegative integers")
    return numeric.astype(np.int64)


def _catalog_buckets(catalog: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if not {"item_id", "popularity"}.issubset(catalog):
        raise ValueError("Catalog requires item_id and popularity")
    cat = catalog[["item_id", "popularity"]].copy()
    if cat.item_id.isna().any():
        raise ValueError("Catalog must contain unique, nonempty item IDs")
    cat["item_id"] = cat.item_id.astype(str)
    if not len(cat) or cat.item_id.duplicated().any() or cat.item_id.eq("").any():
        raise ValueError("Catalog must contain unique, nonempty item IDs")
    cat["popularity"] = _integers(cat.popularity, "Training popularity")
    positive = cat.popularity.to_numpy() > 0
    percentile = np.full(len(cat), np.nan)
    percentile[positive] = cat.loc[positive, "popularity"].rank(method="average", pct=True)
    codes = np.zeros(len(cat), dtype=np.int8)
    codes[positive] = np.where(percentile[positive] <= .5, 1,
                               np.where(percentile[positive] <= .8, 2, 3))
    cat["positive_popularity_percentile"] = percentile
    cat["popularity_bucket"] = pd.Categorical.from_codes(codes, POPULARITY_BUCKETS, ordered=True)
    return cat, codes


def _validate_and_encode(specs, bundles, cat, ks):
    if not specs:
        raise ValueError("No runs supplied")
    by_id = {s["run_id"]: s for s in specs}
    if len(by_id) != len(specs):
        raise ValueError("Duplicate run_id")
    missing = set(by_id) - set(bundles)
    if missing:
        raise ValueError(f"Missing bundles: {sorted(missing)}")
    full = ["full_history_lengths" in bundles[rid] for rid in by_id]
    if any(full) and not all(full):
        raise ValueError("Full history lengths must be present for every run or none")
    if not all("history_fingerprints" in bundles[rid] for rid in by_id):
        raise ValueError("History fingerprints are required for every run")
    index = pd.Index(cat.item_id)
    encoded, reference = {}, None
    for spec in specs:
        rid, stage = spec["run_id"], spec["stage"]
        if stage not in ("init", "sft", "gflow"):
            raise ValueError(f"{rid}: unsupported stage {stage!r}")
        if stage in ("sft", "gflow"):
            init = by_id.get(spec.get("baseline_run_id"))
            if init is None or init["stage"] != "init" or init["model"] != spec["model"]:
                raise ValueError(f"{rid}: missing or incompatible baseline_run_id")
            if init.get("seed") is not None and init["seed"] != spec.get("seed"):
                raise ValueError(f"{rid}: init seed does not match")
        if stage == "gflow":
            sft = by_id.get(spec.get("sft_run_id"))
            if (sft is None or sft["stage"] != "sft" or sft["model"] != spec["model"]
                    or sft.get("seed") != spec.get("seed")
                    or sft.get("baseline_run_id") != spec.get("baseline_run_id")):
                raise ValueError(f"{rid}: missing or incompatible matched sft_run_id/seed")
        bundle = bundles[rid]
        users = np.asarray(bundle["user_ids"], dtype=str)
        targets = np.asarray(bundle["target_item_ids"], dtype=str)
        clipped = _integers(bundle["history_lengths"], f"{rid}: history lengths")
        histories = _integers(bundle.get("full_history_lengths", clipped), f"{rid}: full history lengths")
        contexts = np.asarray(bundle["history_fingerprints"], dtype=str)
        recs = np.asarray(bundle["recommendations"], dtype=str)
        if (users.ndim != 1 or not len(users) or np.unique(users).size != len(users)
                or (users == "").any() or any(a.shape != users.shape for a in
                   (targets, clipped, histories, contexts))
                or (contexts == "").any() or recs.ndim != 2 or recs.shape[0] != len(users)):
            raise ValueError(f"{rid}: invalid arrays, history fingerprints, or duplicate users")
        if (histories < clipped).any():
            raise ValueError(f"{rid}: full history lengths shorter than model-input histories")
        if recs.shape[1] < max(ks):
            raise ValueError(f"{rid}: only {recs.shape[1]} ranks available, requested {max(ks)}")
        order = np.argsort(users)
        alignment = (users[order], targets[order], clipped[order], histories[order], contexts[order])
        if reference is None:
            reference = alignment
        else:
            for name, actual, expected in zip(("users", "targets", "history lengths",
                                               "full history lengths", "history fingerprints"), alignment, reference):
                if not np.array_equal(actual, expected):
                    raise ValueError(f"{rid}: mismatched {name}; comparisons require identical test contexts")
        target_idx = index.get_indexer(alignment[1]).astype(np.int32)
        if (target_idx < 0).any():
            raise ValueError(f"{rid}: unknown target outside shared catalog")
        # Validate the entire saved list, and retain just the cutoffs needed for analysis.
        encoded_recs = np.empty((len(users), max(ks)), dtype=np.int32)
        for start in range(0, len(users), 50_000):
            rows = recs[order[start:start + 50_000]]
            indices = index.get_indexer(rows.ravel()).reshape(rows.shape)
            if ((indices < 0) & (rows != "")).any():
                raise ValueError(f"{rid}: unknown recommendation outside shared catalog")
            sorted_rows = np.sort(indices, axis=1)
            if ((sorted_rows[:, 1:] == sorted_rows[:, :-1]) & (sorted_rows[:, 1:] >= 0)).any():
                raise ValueError(f"{rid}: duplicate valid recommendation within a user list")
            encoded_recs[start:start + len(rows)] = indices[:, :max(ks)]
        ranks = np.zeros(len(users), dtype=np.int32)
        for rank in range(max(ks)):
            ranks[encoded_recs[:, rank] == target_idx] = rank + 1
        encoded[rid] = {"recs": encoded_recs, "ranks": ranks}
    return encoded, reference, "full_history_lengths" if all(full) else "history_lengths (clipped fallback)"


def _cohorts(target_pop, histories, edges):
    history_names = [f"0-{edges[0]}"] + [f"{a + 1}-{b}" for a, b in zip(edges[:-1], edges[1:])]
    history_names.append(f"{edges[-1] + 1}+")
    hc = np.searchsorted(edges, histories, side="left")
    result = [("overall", "all", "all", np.arange(len(histories)))]
    for p, name in enumerate(POPULARITY_BUCKETS):
        result.append(("target_popularity", name, "all", np.flatnonzero(target_pop == p)))
    for h, name in enumerate(history_names):
        result.append(("history_length", "all", name, np.flatnonzero(hc == h)))
    for p, pname in enumerate(POPULARITY_BUCKETS):
        for h, hname in enumerate(history_names):
            result.append(("target_popularity_x_history_length", pname, hname,
                           np.flatnonzero((target_pop == p) & (hc == h))))
    return result


def _exposure_counts(recs, rows, k, size):
    counts = np.zeros(size, dtype=np.int64)
    # Limit the temporary recommendation matrix even for very large test sets.
    for start in range(0, len(rows), max(1, 1_000_000 // k)):
        chunk = recs[rows[start:start + max(1, 1_000_000 // k)], :k].ravel()
        counts += np.bincount(chunk[chunk >= 0], minlength=size)
    return counts


def _concentration(counts, catalog_size):
    positive = np.asarray(counts)[np.asarray(counts) > 0]
    if not len(positive) or not catalog_size:
        return dict(exposure_gini=np.nan, exposure_entropy=np.nan, effective_catalog=np.nan)
    x = np.sort(positive.astype(np.float64))
    total = x.sum()
    # Zero-exposure catalog members enter the Gini denominator without sorting them.
    positions = np.arange(catalog_size - len(x) + 1, catalog_size + 1, dtype=float)
    gini = 2 * np.dot(positions, x) / (catalog_size * total) - (catalog_size + 1) / catalog_size
    probs = x / total
    entropy = float(-np.dot(probs, np.log(probs)))
    return dict(exposure_gini=float(gini), exposure_entropy=entropy, effective_catalog=float(np.exp(entropy)))


def _diversity(counts, popularity, n, k):
    total, unique = int(counts.sum()), int(np.count_nonzero(counts))
    return dict(catalog_size=len(counts), exposures=total, unique_items=unique,
                coverage=unique / len(counts) if n else np.nan,
                arp=float(np.dot(counts, popularity) / total) if total else np.nan,
                valid_slot_rate=total / (n * k) if n else np.nan,
                mean_pairwise_overlap=float(np.dot(counts.astype(float), counts - 1) / (n * (n - 1) * k))
                if n > 1 else np.nan, **_concentration(counts, len(counts)))


def _list_overlap(a, b, k):
    # O(N*K^2) comparisons, O(N*K) temporary memory, never N*K*K.
    overlap = np.zeros(len(a), dtype=float)
    for start in range(0, len(a), 50_000):
        left, right = a[start:start + 50_000, :k], b[start:start + 50_000, :k]
        counts = np.zeros(len(left), dtype=np.int32)
        for rank in range(k):
            counts += ((left[:, rank, None] == right) & (left[:, rank, None] >= 0)).any(axis=1)
        overlap[start:start + len(left)] = counts / k
    return overlap


def _bootstrap(delta_hr, delta_ndcg, samples, rng):
    """Paired user bootstrap, bounded to ~1M sampled indices, not seed uncertainty."""
    if not samples or not len(delta_hr):
        return {f"{metric}_ci_{bound}": np.nan for metric in ("hr", "ndcg") for bound in ("low", "high")}
    n = len(delta_hr)
    means = np.empty((samples, 2))
    batch = max(1, min(32, 1_000_000 // n))
    for start in range(0, samples, batch):
        stop = min(start + batch, samples)
        totals = np.zeros((stop - start, 2))
        for offset in range(0, n, 1_000_000):
            ix = rng.integers(n, size=(stop - start, min(1_000_000, n - offset)))
            totals[:, 0] += delta_hr[ix].sum(axis=1)
            totals[:, 1] += delta_ndcg[ix].sum(axis=1)
        means[start:stop] = totals / n
    quantiles = np.quantile(means, [.025, .975], axis=0)
    return {f"{metric}_ci_{bound}": float(quantiles[j, i])
            for i, metric in enumerate(("hr", "ndcg")) for j, bound in enumerate(("low", "high"))}


def _mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def _item_frame(cat, indices, common, k):
    frame = cat.iloc[indices].copy().reset_index(drop=True)
    for key, value in {**common, "k": k}.items():
        frame[key] = value
    return frame


def _compact(frames):
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ("run_id", "model", "stage", "reference_run_id", "reference_stage"):
        if col in frame:
            frame[col] = frame[col].astype("category")
    return frame


def analyze_buckets(specs: list[dict[str, Any]], bundles: dict[str, dict[str, np.ndarray]],
                    catalog: pd.DataFrame, ks: tuple[int, ...] = (5, 10, 20),
                    history_edges: tuple[int, ...] = (5, 10, 20, 50, 100, 200),
                    bootstrap_samples: int = 0, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Analyze one dataset with explicitly matched init/SFT/GFlow runs.

    ``popularity`` must be training-only counts supplied by the caller; held-out
    targets never affect bucket assignment. A blank recommendation is padding:
    its position stays in the ranking and valid_slot_rate exposes missing slots.
    Full histories, clipped histories and history fingerprints must align by
    user identity. Fingerprints describe actual model inputs. Full-history
    arrays are optional only when absent for every run (reported as fallback).

    Outputs include complete empty cohort rows and compact item tables omitting
    items with neither held-out targets nor exposure. Contributions divide each
    cohort's paired delta sum by the *whole test population*, so each partition
    sums exactly to the overall delta. Conditional means divide by cohort size.
    Exposure Gini includes zero-exposure members of the relevant catalog.
    """
    cutoffs = _integers(ks, "ks")
    if cutoffs.ndim != 1 or not len(cutoffs) or (cutoffs < 1).any():
        raise ValueError("ks must contain positive cutoffs")
    ks = tuple(sorted(set(cutoffs.tolist())))
    edges = _integers(history_edges, "history_edges")
    if edges.ndim != 1 or not len(edges) or (np.diff(edges) <= 0).any():
        raise ValueError("history_edges must be nonempty and strictly increasing")
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    cat, pop_codes = _catalog_buckets(catalog)
    encoded, reference, history_source = _validate_and_encode(specs, bundles, cat, ks)
    _, targets, clipped, histories, _ = reference
    size, n = len(cat), len(targets)
    target_idx = pd.Index(cat.item_id).get_indexer(targets)
    target_counts = np.bincount(target_idx, minlength=size)
    popularity = cat.popularity.to_numpy(dtype=float)
    cohorts = _cohorts(pop_codes[target_idx], histories, edges)
    pop_indices = [np.flatnonzero(pop_codes == p) for p in range(len(POPULARITY_BUCKETS))]
    by_id = {s["run_id"]: s for s in specs}
    summary, cohort_rows, exposure_rows, comparisons, attribution = [], [], [], [], []
    item_frames, delta_frames = [], []
    rng = np.random.default_rng(seed)
    for k in ks:
        # Keep only one cutoff's statistics; recommendations themselves are never copied per K.
        statistics = {}
        for spec in specs:
            rid = spec["run_id"]
            b = encoded[rid]
            common = {key: spec.get(key) for key in ("run_id", "model", "stage", "seed")}
            ranks = b["ranks"]
            hit = (ranks > 0) & (ranks <= k)
            ndcg = np.zeros(n)
            ndcg[hit] = 1 / np.log2(ranks[hit] + 1)
            stats = {"hr": hit, "ndcg": ndcg, "cohorts": {}}
            for group_no, (kind, pop_name, history_name, rows) in enumerate(cohorts):
                counts = _exposure_counts(b["recs"], rows, k, size)
                group = dict(cohort_type=kind, target_popularity_bucket=pop_name,
                             history_bucket=history_name, history_length_source=history_source)
                metrics = dict(num_users=len(rows), hr=_mean(hit[rows]), ndcg=_mean(ndcg[rows]),
                               mean_history_length=_mean(histories[rows]),
                               mean_input_history_length=_mean(clipped[rows]),
                               **_diversity(counts, popularity, len(rows), k))
                record = {**common, "k": k, **group, **metrics}
                (summary if kind == "overall" else cohort_rows).append(record)
                stats["cohorts"][group_no] = metrics
                if kind == "overall":
                    stats["counts"] = counts
                total = int(counts.sum())
                for pname, ix in zip(POPULARITY_BUCKETS, pop_indices):
                    subset = counts[ix]
                    exposures = int(subset.sum())
                    exposure_rows.append({**common, "k": k, **group, "popularity_bucket": pname,
                        "num_users": len(rows), "catalog_items": len(ix), "exposures": exposures,
                        "unique_items": int(np.count_nonzero(subset)),
                        "coverage": np.count_nonzero(subset) / len(ix) if len(ix) and len(rows) else np.nan,
                        "exposure_share": exposures / total if total else np.nan,
                        "target_share": _mean(pop_codes[target_idx[rows]] == POPULARITY_BUCKETS.index(pname)),
                        **_concentration(subset, len(ix))})
            stats["target_hits"] = np.bincount(target_idx, weights=hit, minlength=size)
            stats["target_ndcg_sum"] = np.bincount(target_idx, weights=ndcg, minlength=size)
            statistics[rid] = stats
            active = np.flatnonzero((target_counts > 0) | (stats["counts"] > 0))
            frame = _item_frame(cat, active, common, k)
            frame["target_count"] = target_counts[active]
            frame["target_hits"] = stats["target_hits"][active].astype(np.int64)
            frame["target_ndcg_sum"] = stats["target_ndcg_sum"][active]
            denominator = np.where(target_counts[active] > 0, target_counts[active], np.nan)
            frame["hr"] = frame.target_hits / denominator
            frame["ndcg"] = frame.target_ndcg_sum / denominator
            frame["exposures"] = stats["counts"][active]
            total = stats["counts"].sum()
            frame["exposure_share"] = frame.exposures / total if total else np.nan
            item_frames.append(frame)
        for spec in specs:
            rid = spec["run_id"]
            references = []
            if spec["stage"] in ("sft", "gflow"):
                references.append(spec["baseline_run_id"])
            if spec["stage"] == "gflow":
                references.append(spec["sft_run_id"])
            after = statistics[rid]
            for reference_id in references:
                before = statistics[reference_id]
                common = {key: spec.get(key) for key in ("run_id", "model", "stage", "seed")}
                common.update(reference_run_id=reference_id, reference_stage=by_id[reference_id]["stage"])
                dh = after["hr"].astype(np.int8) - before["hr"].astype(np.int8)
                dn = after["ndcg"] - before["ndcg"]
                overlap = _list_overlap(encoded[rid]["recs"], encoded[reference_id]["recs"], k)
                both = after["hr"] & before["hr"]
                rank_change = encoded[reference_id]["ranks"] - encoded[rid]["ranks"]
                truncated_change = np.where(before["hr"], encoded[reference_id]["ranks"], k + 1) - np.where(
                    after["hr"], encoded[rid]["ranks"], k + 1)
                for group_no, (kind, pop_name, history_name, rows) in enumerate(cohorts):
                    group = dict(cohort_type=kind, target_popularity_bucket=pop_name,
                                 history_bucket=history_name, history_length_source=history_source)
                    selected_both = rows[both[rows]]
                    row = {**common, "k": k, **group, "num_users": len(rows),
                        "population_share": len(rows) / n, "delta_hr": _mean(dh[rows]),
                        "delta_ndcg": _mean(dn[rows]), "mean_list_overlap": _mean(overlap[rows]),
                        "hit_gains": int((dh[rows] > 0).sum()), "hit_losses": int((dh[rows] < 0).sum()),
                        "users_improved": int((dn[rows] > 0).sum()), "users_worsened": int((dn[rows] < 0).sum()),
                        "users_unchanged": int((dn[rows] == 0).sum()), "users_both_hit": len(selected_both),
                        "rank_improvements_both_hit": int((rank_change[selected_both] > 0).sum()),
                        "rank_regressions_both_hit": int((rank_change[selected_both] < 0).sum()),
                        "mean_rank_improvement_both_hit": _mean(rank_change[selected_both]),
                        "mean_truncated_rank_improvement": _mean(truncated_change[rows]),
                        **_bootstrap(dh[rows], dn[rows], bootstrap_samples, rng)}
                    for metric in _DIVERSITY:
                        row[f"delta_{metric}"] = after["cohorts"][group_no][metric] - before["cohorts"][group_no][metric]
                    comparisons.append(row)
                    attribution.append({**common, "k": k, **group, "num_users": len(rows),
                        "population_share": len(rows) / n, "mean_delta_hr": _mean(dh[rows]),
                        "mean_delta_ndcg": _mean(dn[rows]),
                        "delta_hr_contribution": float(dh[rows].sum() / n),
                        "delta_ndcg_contribution": float(dn[rows].sum() / n)})
                active = np.flatnonzero((target_counts > 0) | (after["counts"] > 0) | (before["counts"] > 0))
                frame = _item_frame(cat, active, common, k)
                frame["target_count"] = target_counts[active]
                denominator = np.where(target_counts[active] > 0, target_counts[active], np.nan)
                for suffix, stats in (("before", before), ("after", after)):
                    frame[f"exposures_{suffix}"] = stats["counts"][active]
                    frame[f"target_hits_{suffix}"] = stats["target_hits"][active].astype(np.int64)
                    frame[f"hr_{suffix}"] = stats["target_hits"][active] / denominator
                    frame[f"ndcg_{suffix}"] = stats["target_ndcg_sum"][active] / denominator
                frame["delta_exposures"] = frame.exposures_after - frame.exposures_before
                frame["delta_hr"] = frame.hr_after - frame.hr_before
                frame["delta_ndcg"] = frame.ndcg_after - frame.ndcg_before
                frame["delta_hr_contribution"] = (after["target_hits"][active] - before["target_hits"][active]) / n
                frame["delta_ndcg_contribution"] = (after["target_ndcg_sum"][active] - before["target_ndcg_sum"][active]) / n
                for name, values in (("hit_gains", dh > 0), ("hit_losses", dh < 0),
                                     ("users_improved", dn > 0), ("users_worsened", dn < 0),
                                     ("rank_improvements_both_hit", both & (rank_change > 0)),
                                     ("rank_regressions_both_hit", both & (rank_change < 0))):
                    frame[name] = np.bincount(target_idx, weights=values, minlength=size)[active].astype(np.int64)
                frame["exposure_status"] = np.where(frame.exposures_before == 0,
                    np.where(frame.exposures_after > 0, "new", "unexposed"),
                    np.where(frame.exposures_after == 0, "lost", "retained"))
                delta_frames.append(frame)
    definitions = [
        ("popularity", "Caller-supplied counts from training interactions only; never test target frequency."),
        ("popularity_buckets", "unseen=count 0; positive counts ranked with average ties / positive catalog size; tail <=0.5; middle (0.5,0.8]; head >0.8. Ties stay together, so sizes can differ from nominal fractions."),
        ("history_buckets", f"Source={history_source}; integer lengths; [0,{edges[0]}], then (left,right], then >{edges[-1]}; edges={edges.tolist()}."),
        ("ranking", "HR and single-target NDCG at the original top-K positions. Cohorting never filters or reranks recommendations; empty slots preserve their rank."),
        ("cohort_diversity", "Coverage and exposure concentration use all recommendations from cohort users and the full catalog; target popularity selects users, not allowed recommendation items."),
        ("popularity_exposure", "Recommended-item bucket coverage divides unique exposed bucket items by all catalog items in that bucket. Exposure share divides its slots by all valid cohort slots; concentration is within that bucket."),
        ("diversity_scope", "ARP is mean training count over valid recommendation slots. Gini includes zero-exposure catalog items. Entropy uses natural logarithms; effective_catalog=exp(entropy). These are not semantic ILD or model-distribution entropy."),
        ("mean_pairwise_overlap", "Exact mean intersection/K across distinct user pairs in a cohort, derived from sum c_i(c_i-1)/(n(n-1)K). Undefined for n<2."),
        ("empty_groups", "Empty groups retain num_users=0, counts=0, metric means=NaN, and additive accuracy contributions=0. No-exposure entropy, Gini and effective count are NaN."),
        ("comparisons", "SFT versus explicit init; each GFlow versus explicit init and same-seed SFT sharing the same init. Positive delta is after minus reference; positive rank improvement means moving upward."),
        ("attribution", "Cohort paired HR/NDCG sums divided by all test users. Sum separately within each cohort_type to recover total delta; do not sum different partitions together."),
        ("rank_improvement", "Both-hit rank improvement=reference rank minus tuned rank. Truncated rank improvement assigns misses K+1; it is not an uncensored catalog rank."),
        ("items", "Compact per-item tables retain any target or exposed item; item_deltas retains exposures on either side. catalog_buckets retains every catalog item. No per-user export."),
        ("bootstrap", f"{bootstrap_samples} paired user resamples, 95% percentile intervals for mean HR/NDCG differences; describes test-user uncertainty, not training-seed uncertainty."),
    ]
    return dict(summary=pd.DataFrame(summary), cohorts=pd.DataFrame(cohort_rows),
                popularity_exposure=pd.DataFrame(exposure_rows), comparisons=pd.DataFrame(comparisons),
                accuracy_attribution=pd.DataFrame(attribution), items=_compact(item_frames),
                item_deltas=_compact(delta_frames), catalog_buckets=cat,
                definitions=pd.DataFrame(definitions, columns=["name", "definition"]))
