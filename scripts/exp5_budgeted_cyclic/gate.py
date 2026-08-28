"""User-adaptive policy gate (spec §25): validation-trained, frozen before test.

Labels come from VALIDATION per-user metrics only (never test targets).
Features are restricted to quantities available at inference under the stated
cache regime; held-out-target-defined slices are diagnostics, never features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GATE_FEATURES = [
    "history_length",
    "tiger_entropy_L1", "tiger_entropy_L2", "tiger_entropy_L3", "tiger_entropy_L4",
    "tiger_margin_L1", "tiger_margin_L2", "tiger_margin_L3", "tiger_margin_L4",
    "raw_valid_mass_L1", "raw_valid_mass_L2", "raw_valid_mass_L3", "raw_valid_mass_L4",
    "sasrec_score_entropy",
    "tiger_sasrec_top20_overlap",
    "tiger_sasrec_rank_corr_top50",
    "js_tiger_sasrec",
    "log_C_max", "log_C_q90",
    "predicted_oracle_cost",
    "suffix_collision_multiplicity",
]


def user_gate_features(
    u: Any, raw_valid_mass: np.ndarray, history_length: int, target_ctx: Any,
) -> dict[str, float]:
    """Inference-available features (offline-cache regime) for one user.

    ``u`` is a policies.UserPolicyContext; ``target_ctx`` the gate's reference
    reward target (its oracle supplies log C and the predicted cost).
    """
    trie = u.trie
    feats: dict[str, float] = {"history_length": float(history_length)}
    ent, margin = u.entropy, u._margin
    reach = np.exp(_log_reach(u))
    for lvl in range(1, trie.sid_len + 1):
        parents = trie.internal_node_ids[trie.level[trie.internal_node_ids] == lvl - 1]
        w = reach[parents]
        w = w / max(w.sum(), 1e-300)
        feats[f"tiger_entropy_L{lvl}"] = float((ent[parents] * w).sum())
        feats[f"tiger_margin_L{lvl}"] = float((margin[parents] * w).sum())
        rv = raw_valid_mass[parents] if len(raw_valid_mass) >= len(parents) else np.ones(len(parents))
        feats[f"raw_valid_mass_L{lvl}"] = float((rv * w).sum())

    p_s = np.exp(u.raw_scores - u.raw_scores.max())
    p_s = p_s / p_s.sum()
    nz = p_s > 0
    feats["sasrec_score_entropy"] = float(-(p_s[nz] * np.log(p_s[nz])).sum())

    top_t = np.argsort(-u.log_p0, kind="stable")[:50]
    top_s = np.argsort(-u.raw_scores, kind="stable")[:50]
    feats["tiger_sasrec_top20_overlap"] = float(
        len(set(top_t[:20]) & set(top_s[:20])) / 20.0)
    from scipy.stats import spearmanr
    cand = np.unique(np.concatenate([top_t, top_s]))
    rc = spearmanr(u.log_p0[cand], u.raw_scores[cand]).statistic
    feats["tiger_sasrec_rank_corr_top50"] = float(rc if np.isfinite(rc) else 0.0)

    p0 = u.p0
    m = 0.5 * (p0 + p_s)
    nz0, nzs = p0 > 0, p_s > 0
    feats["js_tiger_sasrec"] = float(
        0.5 * np.sum(p0[nz0] * (np.log(p0[nz0]) - np.log(m[nz0])))
        + 0.5 * np.sum(p_s[nzs] * (np.log(p_s[nzs]) - np.log(m[nzs]))))

    if target_ctx.oracle is not None:
        lc = target_ctx.oracle.log_C[trie.internal_node_ids]
        feats["log_C_max"] = float(lc.max())
        feats["log_C_q90"] = float(np.quantile(lc, 0.9))
        feats["predicted_oracle_cost"] = float(target_ctx.oracle.E_del_inf)
    else:
        feats["log_C_max"] = feats["log_C_q90"] = feats["predicted_oracle_cost"] = np.nan

    top1 = int(np.argmax(u.log_p0))
    leaf = trie.row_to_terminal_node[top1]
    feats["suffix_collision_multiplicity"] = float(
        trie.child_count[trie.parent[leaf]])
    return feats


def _log_reach(u: Any) -> np.ndarray:
    trie = u.trie
    lr = np.zeros(trie.num_nodes)
    for depth in range(1, trie.sid_len + 1):
        ids = np.arange(trie.level_offsets[depth], trie.level_offsets[depth + 1])
        lr[ids] = lr[trie.parent[ids]] + u.edge_logprob[ids]
    return lr


@dataclass
class GateArtifact:
    model_kind: str
    candidate_policies: list[str]      # canonical config hashes
    feature_names: list[str]
    label_metric: str
    cv_accuracy: float
    model_path: str

    def save_meta(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))


def train_gate(
    features_df: pd.DataFrame, labels: np.ndarray, *, candidate_policies: list[str],
    label_metric: str, out_dir: Path, seed: int = 42, model_kind: str = "logistic",
) -> GateArtifact | None:
    """Cross-validated gate on validation features; frozen (saved) before test."""
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
    except ImportError:
        print("sklearn unavailable; skipping gate training", flush=True)
        return None

    X = features_df[[c for c in GATE_FEATURES if c in features_df.columns]].to_numpy()
    y = np.asarray(labels)
    classes, counts = np.unique(y, return_counts=True)
    min_per_class = 5
    if len(classes) < 2 or counts.min() < min_per_class:
        print(f"gate labels too degenerate for CV (classes {dict(zip(classes.tolist(), counts.tolist()))}, "
              f"need >=2 classes with >={min_per_class} users each); skipping", flush=True)
        return None
    cv = int(min(5, counts.min()))
    models = {
        "logistic": make_pipeline(
            SimpleImputer(), StandardScaler(),
            LogisticRegression(max_iter=2000)),
        "gbt": make_pipeline(
            SimpleImputer(),
            GradientBoostingClassifier(random_state=seed)),
    }
    best_kind, best_score, best_model = None, -1.0, None
    for kind, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, error_score=np.nan)
        except Exception as exc:
            print(f"  gate {kind}: CV failed ({exc}); skipped", flush=True)
            continue
        mean = float(np.nanmean(scores))
        if np.isfinite(mean) and mean > best_score:
            best_kind, best_score, best_model = kind, mean, model
    if best_model is None:
        print("gate: no model survived cross-validation; skipping", flush=True)
        return None
    best_model.fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "gate_model.joblib"
    joblib.dump(best_model, model_path)
    art = GateArtifact(
        model_kind=best_kind, candidate_policies=candidate_policies,
        feature_names=[c for c in GATE_FEATURES if c in features_df.columns],
        label_metric=label_metric, cv_accuracy=best_score,
        model_path=str(model_path))
    art.save_meta(out_dir / "gate_meta.json")
    print(f"gate: {best_kind} cv-accuracy {best_score:.3f} "
          f"over {len(candidate_policies)} policies", flush=True)
    return art


def load_gate(out_dir: Path) -> tuple[Any, GateArtifact] | None:
    meta_path = out_dir / "gate_meta.json"
    if not meta_path.exists():
        return None
    import joblib
    meta = GateArtifact(**json.loads(meta_path.read_text()))
    return joblib.load(meta.model_path), meta
