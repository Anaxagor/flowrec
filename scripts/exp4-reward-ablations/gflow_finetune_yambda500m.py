#!/usr/bin/env python
"""GFlow reward ablation on the Yambda 500m exp2 TIGER checkpoint (key cells).

The yambda_500m counterpart of gflow_finetune_yambda50m.py. It reuses that
script's machinery (protocol config, gradient-norm-calibrated gflow weight,
per-user tail subset, sanity gates, CSV aggregation) and replaces the parts
that do not scale to 87.5k users x 374,283 catalog items:

  reward caches   ONE FACTORIZED cache per reward component (sasrec,
                  interaction, similarity; scripts/precompute_reward_factors.py)
                  instead of one dense cache per combo: the dense rows would be
                  ~0.5 TB per cache here, the factors are a few hundred MB.
                  Combos / powers are composed at lookup time on the GPU
                  (ComposedFactorizedRewardCache, exact logZ), so every combo
                  and power shares the same three caches and no
                  --store-power-sums bookkeeping exists.
  teacher         the SCE-tuned exp2 SASRec (checkpoints/exp2/loo_split/initial/
                  sasrec_sce/yambda_500m_listens/best.pt, val NDCG@10
                  0.124) scores the curriculum negatives AND is the sasrec
                  reward. It lives in the pre-cold-filter id space (vocab
                  434,658 = the SID table): run_tiger.build_sasrec_teacher_model
                  wraps it with id translation and sanity_check_teacher
                  verifies the mapping before anything is cached.
  resume          --resume continues an unfinished run from
                  checkpoints/last.pt (optimizer / AMP / early-stopping state,
                  the init-calibrated lambda from calibration.json), so runs
                  can span the cluster's 3-day walltime by resubmission.
  init baseline   the SFT-only control already evaluated this checkpoint under
                  the same beam-20 protocol; its init_baseline.json is imported
                  instead of spending hours recomputing it.

Reduced key grid (24 runs, seed 42, rho=1.0):
  combos (4)      sasrec, interaction, similarity, sasrec_x_interaction_x_similarity
  powers (2)      1 (raw product) and 16 (the sharpening that helped DB on 50m)
  objectives (3)  tb, db, subtb
  negatives (1)   K=10 (curriculum over teacher scores; K=3 and 10 were
                  indistinguishable on 50m, larger K is ~(K+1)x the compute)
Every dimension stays selectable (--combos/--powers/--objectives/
--num-candidates/--rho-targets) for follow-ups; the three caches serve all.

Protocol (configs/exp2-tuned-params/finetune_gflow.json, shared with the SFT
control): lr 3e-5, effective batch 2048, <=500 epochs, validate every 5
(patience 10 validations), grad clip 0.5, fp16 AMP, beam-5 val monitor,
beam-20 final evaluation on the FULL validation + test splits. The
`yambda_500m` block adds: fine-tune on each user's last 8 train examples
(~0.70M, the 50m rule; the datamodule keeps the checkpoint's last-128 layout
so the catalog matches) and monitor on a 0.25 sample of the 87k-row val
split (the yambda_500m SFT-control rule).

The last-8 subset makes this arm's data differ from the base SFT run and from
the SFT-only control (both train on the full last-128 layout). --train-last-n 0
removes the subset; gflow_finetune_tiger_matched.py runs the grid that way
(configs/exp2-tuned-params/finetune_base_matched.json) with its own outputs,
factor caches, and metrics CSV.

Usage:
  python scripts/exp4-reward-ablations/gflow_finetune_yambda500m.py --list
  python scripts/exp4-reward-ablations/gflow_finetune_yambda500m.py --cache-only
  python scripts/exp4-reward-ablations/gflow_finetune_yambda500m.py --index 0 --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR), str(SCRIPTS_DIR / "exp2-tuned-params"), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_yambda50m as base  # noqa: E402
import run_tiger as tiger_runner  # noqa: E402
import sft_finetune_control as control  # noqa: E402
from precompute_reward_cache import per_user_random_indices, per_user_tail_indices  # noqa: E402
from flowrec.candidates.sampler import (  # noqa: E402
    CandidateSamplerConfig,
    SemanticIDCandidateSampler,
)
from flowrec.ft_methods.calibration import (  # noqa: E402
    calibrate_from_loss_pairs,
    variant_loss_pairs,
)
from flowrec.ft_methods.gflow.config import GFLOW_REWARD_EPSILON, GFlowConfig  # noqa: E402
from flowrec.ft_methods.gflow.factorized_cache import (  # noqa: E402
    CACHE_KIND,
    ComposedFactorizedRewardCache,
    FactorizedRewardCache,
)
from flowrec.ft_methods.gflow.method import GFlowMethod  # noqa: E402
from flowrec.ft_methods.sft.config import SFTConfig  # noqa: E402
from flowrec.training import Trainer, TrainerConfig  # noqa: E402

abspath, load_json, write_json, move_batch = base.abspath, base.load_json, base.write_json, base.move_batch
combo_key, power_tag = base.combo_key, base.power_tag  # entry_tag / run_dir_for are extended below
indexed_train_loader, AliasedSFTMethod = base.indexed_train_loader, base.AliasedSFTMethod
metric_rows = base.metric_rows  # upsert_metrics_csv is extended below (wider row key)

# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
DATASET = "yambda_500m_listens"
SFT_CONFIG = "configs/exp2-tuned-params/tiger_yambda_500m_listens_sft.json"
# The exp2 init checkpoint the SFT-only control fine-tunes from: the OFAT
# tuning winner (trials/dropout-0, an 8-epoch snapshot, monitor 0.1078) trained
# to convergence at the tuned lr and promoted here (best epoch 54, monitor
# 0.1107). Must stay the control's `tiger_checkpoint`: the imported init
# baseline and the gflow runs have to start from the same weights.
TIGER_CHECKPOINT = "checkpoints/exp2/loo_split/initial/tiger/yambda_500m_listens/best.pt"
# SCE-tuned exp2 SASRec: sampler teacher + sasrec reward. Architecture comes
# from the checkpoint's own model_config; best_trial.json is only a fallback.
SASREC_CHECKPOINT = "checkpoints/exp2/loo_split/initial/sasrec_sce/yambda_500m_listens/best.pt"
SASREC_TRIAL = "checkpoints/exp2/loo_split/initial/sasrec_sce/yambda_500m_listens/best_trial.json"
RQVAE_MODEL = "datasets/semantic_ids/exp2-tuned-params/yambda_500m_listens/model.pt"
# The captured 500m config already carries this criterion (no destructive
# scope filters); re-asserted so the run config is explicit.
POSITIVE_CRITERION = "event_type == 'listen' and played_ratio_pct >= 50"
LEVEL_STRIDE = 256  # level-offset SID table, like 50m (plus the JSON+1 offset)
# The SFT-only control's init evaluation of this checkpoint (same protocol).
CONTROL_INIT_BASELINE = "experiments/exp2-sft-finetune-yambda/yambda_500m_listens/init_baseline.json"
# config["data"] entries re-pointed to local files (Amazon: the captured
# configs name the training host's parquet paths).
DATA_OVERRIDES: dict[str, str] = {}
# config["dataset"] entries replaced so that POSITIVE_CRITERION defines the
# positive stream instead of a destructive scope filter: the training examples
# stay those of the checkpoint while the complement lands in the auxiliary
# interaction store the interaction reward reads.
YAMBDA_DATASET_OVERRIDES: dict[str, Any] = {"allowed_event_types": None, "min_played_ratio_pct": None}
DATASET_OVERRIDES: dict[str, Any] = dict(YAMBDA_DATASET_OVERRIDES)
# sanity_check_teacher floor (val NDCG@10 on the first validation batches; a
# misaligned item-id space scores ~0).
TEACHER_NDCG_THRESHOLD = 0.02

COMPONENT_ORDER = ("sasrec", "interaction", "similarity")
# Components a combo may name = factor caches. `similarity` is the hierarchical
# (RQ-VAE codebook) similarity to the target; `sim_embed` is the cosine
# similarity between item content embeddings (its own cache; never part of the
# legacy three-component run config).
CACHE_COMPONENTS = (*COMPONENT_ORDER, "sim_embed")
SIMILARITY_MODE_OF = {"similarity": "hierarchical", "sim_embed": "embedding_cosine"}
# Where the K negatives come from. teacher = the SASRec curriculum (legacy run
# dirs / CSV rows); random = uniform over the catalog; policy = the model being
# fine-tuned samples them itself, always on-policy and constrained to valid
# catalog items. The last two need no teacher model at all unless the combo
# contains the sasrec reward.
SAMPLERS = ("teacher", "random", "policy")
DEFAULT_SAMPLER = "teacher"
DEFAULT_COMBOS = ("sasrec", "interaction", "similarity", "sasrec_x_interaction_x_similarity")
DEFAULT_POWERS = (1.0, 16.0)
DEFAULT_OBJECTIVES = ("tb", "db", "subtb")
DEFAULT_NUM_NEGATIVES = (10,)
DEFAULT_OUTPUT_ROOT = "experiments/exp4-gflow-yambda500m"
DEFAULT_CACHE_ROOT = "artifacts/exp4-reward_ablation/gflow_yambda500m/reward_factors"
METRICS_CSV_NAME = "gflow_finetune_yambda_500m_metrics_summary.csv"
PROTOCOL_BLOCK = "yambda_500m"

# --------------------------------------------------------------------------- #
# Datasets: the factorized-cache path is dataset-agnostic. The module-level
# asset constants above are the ACTIVE dataset's (default yambda_500m_listens);
# --dataset rebinds them before anything reads them.
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = DATASET
AMAZON_POSITIVE_CRITERION = "verified_purchase == True"
_TIGER_INIT = "checkpoints/exp2/loo_split/initial/tiger"
_SASREC_INIT = "checkpoints/exp2/loo_split/initial/sasrec"
_SID_ROOT = "datasets/semantic_ids/exp2-tuned-params"
_RAW_AMAZON = "data/raw/dataframe_format"


# Item content embeddings of the sim_embed reward: Yambda's audio embeddings
# (item_id / normalized_embed parquet), the Amazon *_content_embeddings.pt
# payloads the GPT-2 arm uses (same file per dataset = same reward for both models).
EMBEDDINGS_PATH = "data/raw/yambda/embeddings.parquet"
_EMBEDDINGS_ROOT = "datasets/embeddings"


def _dataset_spec(**spec: Any) -> dict[str, Any]:
    """A COMPLETE spec (activate_dataset rebinds every key, so switching
    datasets inside one process never leaks the previous dataset's values)."""
    defaults = {
        "POSITIVE_CRITERION": POSITIVE_CRITERION, "LEVEL_STRIDE": 256,
        "DATA_OVERRIDES": {}, "DATASET_OVERRIDES": dict(YAMBDA_DATASET_OVERRIDES),
        "TEACHER_NDCG_THRESHOLD": 0.02, "EMBEDDINGS_PATH": EMBEDDINGS_PATH,
    }
    return {**defaults, **spec}


def _amazon_spec(name: str, *, sft_config: str, reviews: str, metadata: str, rqvae_model: str,
                 control_init_baseline: str, teacher_ndcg_threshold: float) -> dict[str, Any]:
    """TIGER on an Amazon dataset (only run through the base-matched wrapper,
    gflow_finetune_tiger_matched.py: full user history, no last-N subset).

    Data = the checkpoint's captured config, exactly as the SFT-only control
    builds it (sft_finetune_control*.py: same sft_config, same parquet
    re-pointing), with the verified-purchase rule expressed as
    positive_criterion instead of use_verified_only. Verified 2026-09-19 on
    all three datasets: catalog and train examples are identical (beauty
    12,101 items / 131,413 examples; instruments 8,556 / 116,989; beauty_2023
    163,050 / 3,493,253), and the unverified reviews stay available to the
    interaction reward (instruments 10,792, beauty_2023 296,645; beauty 2014
    has none). Teacher = the exp2 full-CE tuned SASRec the GPT-2
    arm uses. RQ-VAE = the tokenizer behind the promoted level-offset SID
    table (promoted token = code + level * 256 on every level, checked).
    """
    return _dataset_spec(
        SFT_CONFIG=sft_config, TIGER_CHECKPOINT=f"{_TIGER_INIT}/{name}/best.pt",
        SASREC_CHECKPOINT=f"{_SASREC_INIT}/{name}/best.pt",
        SASREC_TRIAL=f"{_SASREC_INIT}/{name}/best_trial.json",
        RQVAE_MODEL=rqvae_model, CONTROL_INIT_BASELINE=control_init_baseline,
        DEFAULT_OUTPUT_ROOT=f"experiments/exp4-gflow-tiger-{name}",
        DEFAULT_CACHE_ROOT=f"artifacts/exp4-reward_ablation/gflow_tiger_{name}/reward_factors",
        METRICS_CSV_NAME=f"gflow_finetune_tiger_{name}_metrics_summary.csv",
        PROTOCOL_BLOCK=f"tiger_{name}",
        POSITIVE_CRITERION=AMAZON_POSITIVE_CRITERION,
        DATA_OVERRIDES={"reviews_path": f"{_RAW_AMAZON}/{reviews}.parquet",
                        "metadata_path": f"{_RAW_AMAZON}/{metadata}.parquet"},
        DATASET_OVERRIDES={"use_verified_only": False},
        TEACHER_NDCG_THRESHOLD=teacher_ndcg_threshold,
        EMBEDDINGS_PATH=f"{_EMBEDDINGS_ROOT}/{name}_content_embeddings.pt",
    )


DATASETS: dict[str, dict[str, Any]] = {
    "yambda_500m_listens": _dataset_spec(
        SFT_CONFIG=SFT_CONFIG, TIGER_CHECKPOINT=TIGER_CHECKPOINT,
        SASREC_CHECKPOINT=SASREC_CHECKPOINT, SASREC_TRIAL=SASREC_TRIAL,
        RQVAE_MODEL=RQVAE_MODEL, CONTROL_INIT_BASELINE=CONTROL_INIT_BASELINE,
        DEFAULT_OUTPUT_ROOT=DEFAULT_OUTPUT_ROOT, DEFAULT_CACHE_ROOT=DEFAULT_CACHE_ROOT,
        METRICS_CSV_NAME=METRICS_CSV_NAME, PROTOCOL_BLOCK=PROTOCOL_BLOCK,
    ),
    # yambda_50m through the same factorized path: the dense float16 caches of
    # gflow_finetune_yambda50m.py overflow for exp(score) rewards, the factors
    # are rebuilt in float32 at lookup time. Assets = the 50m grid's (full-CE
    # tuned SASRec teacher, val NDCG@10 0.117 vs the TIGER init's 0.084).
    "yambda_50m_listens": _dataset_spec(
        SFT_CONFIG=base.SFT_CONFIG, TIGER_CHECKPOINT=base.TIGER_CHECKPOINT,
        SASREC_CHECKPOINT=base.SASREC_CHECKPOINT, SASREC_TRIAL=base.SASREC_TRIAL,
        RQVAE_MODEL=base.RQVAE_MODEL,
        CONTROL_INIT_BASELINE="experiments/exp2-sft-finetune-yambda/yambda_50m_listens/init_baseline.json",
        DEFAULT_OUTPUT_ROOT="experiments/exp4-gflow-yambda50m-factorized",
        DEFAULT_CACHE_ROOT="artifacts/exp4-reward_ablation/gflow_yambda50m/reward_factors",
        METRICS_CSV_NAME="gflow_finetune_yambda_50m_factorized_metrics_summary.csv",
        PROTOCOL_BLOCK="yambda_50m_factorized",
    ),
    # TIGER on the Amazon datasets: the counterpart of the GPT-2 arm
    # (gflow_finetune_gpt2_amazon_fullhist.py) on beauty / instruments, plus
    # beauty_2023. SASRec val NDCG@10 floors: the GPT-2 script's values
    # (tuning recorded 0.053 beauty / 0.138 instruments).
    "beauty": _amazon_spec(
        "beauty", sft_config="configs/exp2-tuned-params/tiger_beauty_sft.json",
        reviews="reviews_beauty_2014", metadata="meta_beauty_2014",
        rqvae_model=f"{_SID_ROOT}/beauty/rqvae/model.pt",
        control_init_baseline="experiments/exp2-sft-finetune/beauty/init_baseline.json",
        teacher_ndcg_threshold=0.02),
    "instruments": _amazon_spec(
        "instruments", sft_config="configs/exp2-tuned-params/tiger_instruments_sft.json",
        reviews="reviews_instruments_2018", metadata="meta_instruments_2018",
        rqvae_model=f"{_SID_ROOT}/instruments/rqvae/model.pt",
        control_init_baseline="experiments/exp2-sft-finetune/instruments/init_baseline.json",
        teacher_ndcg_threshold=0.04),
    "beauty_2023": _amazon_spec(
        "beauty_2023", sft_config="configs/exp2-tuned-params/tiger_beauty_2023_sft_tuned.json",
        reviews="reviews_beauty_2023", metadata="meta_beauty_2023",
        rqvae_model=f"{_SID_ROOT}/beauty_2023/model.pt",
        control_init_baseline="experiments/exp2-sft-finetune-matched/beauty_2023/init_baseline.json",
        teacher_ndcg_threshold=0.01),
}


def activate_dataset(name: str) -> None:
    """Rebind the module's asset constants to ``name``'s spec."""
    if name not in DATASETS:
        raise SystemExit(f"--dataset must be one of {sorted(DATASETS)}, got {name!r}")
    globals()["DATASET"] = name
    globals().update(DATASETS[name])


# Legacy reward form: runs with it keep their original run dirs / CSV rows.
LEGACY_SASREC_TRANSFORM = "sigmoid"
LEGACY_SASREC_TEMPERATURE = 1.0
CSV_KEY = tuple(base.CSV_KEY) + ("sasrec_transform", "sasrec_temperature", "interaction_exponent", "sampler")


def all_combos() -> list[tuple[str, ...]]:
    """The base script's seven combos first (unchanged order, so existing grid
    indices stay put), then every combo that contains sim_embed."""
    import itertools

    legacy = [tuple(c) for c in base.all_combos()]
    extended = [combo for size in range(1, len(CACHE_COMPONENTS) + 1)
                for combo in itertools.combinations(CACHE_COMPONENTS, size) if "sim_embed" in combo]
    return legacy + extended


def protocol_defaults() -> dict[str, Any]:
    """Shared protocol + the yambda_500m block of finetune_gflow.json."""
    defaults = dict(base.protocol_defaults())
    if base.PROTOCOL_CONFIG.exists():
        payload = json.loads(base.PROTOCOL_CONFIG.read_text(encoding="utf-8"))
        defaults.update({k: v for k, v in payload.get(PROTOCOL_BLOCK, {}).items() if k != "note"})
    return defaults


# --------------------------------------------------------------------------- #
# Grid: base dimensions x sasrec temperature x interaction exponent
# --------------------------------------------------------------------------- #
def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    """base.build_grid expanded by the reward-FORM dimensions.

    ``--sasrec-temperatures`` applies to combos containing sasrec and
    ``--interaction-exponents`` to combos where interaction modulates another
    component (alone, its exponent is just the power); elsewhere the dimension
    collapses to its neutral value so no duplicate runs appear.
    """
    temperatures = [float(t) for t in (args.sasrec_temperatures or [args.sasrec_temperature])]
    exponents = [float(e) for e in (args.interaction_exponents or [1.0])]
    wanted = set(args.combos)
    rhos = args.rho_targets or [args.gflow_target_grad_ratio]
    # sampler-major, then the base script's order (combo-major: consecutive
    # indices share their caches).
    entries = [{"sampler": str(sampler), "combo": combo, "power": float(power), "rho": float(rho),
                "objective": str(objective), "num_negatives": int(k)}
               for sampler in (getattr(args, "samplers", None) or [DEFAULT_SAMPLER])
               for combo in all_combos() if combo_key(combo) in wanted
               for power in args.powers for rho in rhos
               for objective in args.objectives for k in args.num_candidates]
    grid: list[dict[str, Any]] = []
    for entry in entries:
        combo = tuple(entry["combo"])
        has_sasrec = "sasrec" in combo
        modulated = "interaction" in combo and len(combo) > 1
        for temperature in (temperatures if has_sasrec else [LEGACY_SASREC_TEMPERATURE]):
            for exponent in (exponents if modulated else [1.0]):
                grid.append({
                    **entry,
                    "sasrec_transform": str(args.sasrec_transform) if has_sasrec else "-",
                    "sasrec_temperature": float(temperature) if has_sasrec else "-",
                    "interaction_exponent": float(exponent),
                })
    return grid


def variant_suffix(entry: Mapping[str, Any]) -> str:
    """Run-dir / tag suffix of a non-legacy reward form ('' for sigmoid, T=1,
    interaction exponent 1, so existing runs are still found)."""
    parts: list[str] = []
    transform = entry.get("sasrec_transform", "-")
    if transform != "-":
        temperature = float(entry.get("sasrec_temperature", LEGACY_SASREC_TEMPERATURE))
        if transform != LEGACY_SASREC_TRANSFORM or temperature != LEGACY_SASREC_TEMPERATURE:
            parts.append(f"{transform}_T{power_tag(temperature)}")
    exponent = float(entry.get("interaction_exponent", 1.0))
    if exponent != 1.0:
        parts.append(f"ie{power_tag(exponent)}")
    sampler = str(entry.get("sampler", DEFAULT_SAMPLER))
    if sampler != DEFAULT_SAMPLER:
        parts.append(f"neg-{sampler}")
    return "".join(f"_{part}" for part in parts)


def entry_tag(entry: Mapping[str, Any]) -> str:  # noqa: F811 - extends base.entry_tag
    return base.entry_tag(entry) + variant_suffix(entry)


def run_dir_for(output_root: Path, entry: Mapping[str, Any], seed: int) -> Path:  # noqa: F811
    legacy = base.run_dir_for(output_root, entry, seed)
    return legacy.parent.with_name(legacy.parent.name + variant_suffix(entry)) / legacy.name


def csv_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"sasrec_transform": entry.get("sasrec_transform", "-"),
            "sasrec_temperature": entry.get("sasrec_temperature", "-"),
            "interaction_exponent": float(entry.get("interaction_exponent", 1.0)),
            "sampler": str(entry.get("sampler", DEFAULT_SAMPLER))}


def _legacy_csv_defaults(values: dict[str, str]) -> None:
    """Rows written before the reward-form dimensions existed."""
    gflow = values.get("variant") == "gflow"
    has_sasrec = gflow and "sasrec" in values.get("combo", "").split("_x_")
    if not values.get("sasrec_transform"):
        values["sasrec_transform"] = LEGACY_SASREC_TRANSFORM if has_sasrec else "-"
    if not values.get("sasrec_temperature"):
        values["sasrec_temperature"] = str(LEGACY_SASREC_TEMPERATURE) if has_sasrec else "-"
    if not values.get("interaction_exponent"):
        values["interaction_exponent"] = "1.0" if gflow else "-"
    if not values.get("sampler"):
        values["sampler"] = DEFAULT_SAMPLER if gflow else "-"


def upsert_metrics_csv(csv_path: Path, new_rows: list[dict[str, Any]]) -> None:  # noqa: F811
    base.upsert_metrics_csv(csv_path, new_rows, key=CSV_KEY, legacy_defaults=_legacy_csv_defaults)


def component_spec(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "sasrec":
        # The factor cache is always BUILT (and self-checked) with the legacy
        # transform: its factors are transform-agnostic and every run applies
        # its own transform / temperature at lookup time (composed_cache), so
        # one cache serves sigmoid and exp at any temperature.
        return {"name": "sasrec", "weight": 1.0,
                "reward": {"transform": LEGACY_SASREC_TRANSFORM,
                           "temperature": LEGACY_SASREC_TEMPERATURE}}
    if name == "interaction":
        return {"name": "interaction", "weight": 1.0, "reward": {}}
    if name == "similarity":
        return {"name": "similarity", "weight": 1.0,
                "reward": {"mode": "hierarchical", "level_stride": LEVEL_STRIDE,
                           "rqvae_model_path": str(abspath(RQVAE_MODEL))}}
    if name == "sim_embed":
        # The similarity reward model in its content-embedding mode; "sim_embed"
        # is only the cache / combo key.
        return {"name": "similarity", "weight": 1.0,
                "reward": {"mode": "embedding_cosine", "embeddings_path": str(abspath(EMBEDDINGS_PATH))}}
    raise ValueError(f"Unknown reward component {name!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # The dataset decides asset paths AND protocol defaults, so resolve it first.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default=DEFAULT_DATASET)
    activate_dataset(pre.parse_known_args(argv)[0].dataset)

    D = protocol_defaults()
    p = argparse.ArgumentParser(
        "GFlow reward ablation on yambda (factorized per-component reward caches, "
        "calibrated gflow weight): combos x powers x objectives x K x rho x "
        "sasrec temperature x interaction exponent."
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET,
                   help="yambda_500m_listens (default), yambda_50m_listens, or an Amazon dataset "
                        "(beauty, instruments, beauty_2023: run them through "
                        "gflow_finetune_tiger_matched.py); selects the exp2 assets, output/cache "
                        "roots, metrics CSV and the protocol block")
    combo_keys = [combo_key(c) for c in all_combos()]
    p.add_argument("--combos", nargs="+", choices=combo_keys, default=list(DEFAULT_COMBOS))
    p.add_argument("--samplers", nargs="+", choices=list(SAMPLERS), default=[DEFAULT_SAMPLER],
                   help="Grid dimension: source of the K negatives. teacher = SASRec curriculum "
                        "(default, legacy run dirs); random = uniform over the catalog; policy = "
                        "sampled without replacement from the model being fine-tuned, always "
                        "on-policy (stochastic beam search constrained to valid catalog items; no "
                        "curriculum, temperature 1). random / policy load no teacher unless a combo "
                        "contains sasrec")
    p.add_argument("--policy-extra-beams", type=int, default=None,
                   help="policy sampler: beams beyond K (default K, i.e. 2K beams); the target, "
                        "history items and non-catalog rows among the beams are dropped")
    p.add_argument("--powers", type=float, nargs="+", default=list(DEFAULT_POWERS))
    p.add_argument("--objectives", nargs="+", choices=list(DEFAULT_OBJECTIVES),
                   default=list(DEFAULT_OBJECTIVES))
    p.add_argument("--num-candidates", type=int, nargs="+", default=list(DEFAULT_NUM_NEGATIVES))
    p.add_argument("--seed", type=int, default=int(D.get("seed", 42)))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Grid selection by index — same semantics as the 50m script.
    p.add_argument("--list", action="store_true", help="Print the run grid with indices and exit")
    p.add_argument("--index", type=str, default=None,
                   help="One grid entry ('7') or an inclusive range ('4-7') run sequentially")
    p.add_argument("--force", action="store_true",
                   help="Re-run entries whose result.json already reports completed")
    p.add_argument("--resume", action="store_true",
                   help="Continue unfinished runs from checkpoints/last.pt (optimizer, AMP, "
                        "early-stopping state; lambda from calibration.json) instead of "
                        "restarting from the init checkpoint")

    # Data scale.
    p.add_argument("--train-last-n", type=int, default=int(D.get("train_last_n", 8)),
                   help="Fine-tune on the last N train examples per user (subset of the "
                        "checkpoint's last-128 layout; the catalog is untouched). 0 = no "
                        "subset: every example of the checkpoint's own train layout, i.e. the "
                        "data of the base SFT run and of the SFT-only control; the factor "
                        "caches are tied to the subset, so that mode needs its own "
                        "--cache-root (see gflow_finetune_tiger_matched.py)")
    p.add_argument("--train-random-n", type=int, default=int(D.get("train_random_n", 0)),
                   help="Fine-tune on N train examples per user drawn uniformly at RANDOM from the "
                        "checkpoint's layout (all of a user's last-128 targets are equally likely; "
                        "one fixed draw, --train-subset-seed). Cheaper than the full layout without "
                        "the recency bias of --train-last-n; exclusive with it. Needs its own "
                        "--cache-root (cache rows are positions in the subset)")
    p.add_argument("--train-subset-seed", type=int, default=int(D.get("train_subset_seed", 0)),
                   help="Seed of the --train-random-n draw. NOT the run seed: every arm, cell and "
                        "seed of an experiment must fine-tune on the same examples")
    p.add_argument("--val-sample-fraction", type=float, default=D.get("val_sample_fraction"),
                   help="In-training monitor validates on this fraction of the val split "
                        "(yambda_500m SFT-control protocol: 0.25); final eval uses full splits")
    p.add_argument("--val-sample-resample", dest="val_sample_resample",
                   action=argparse.BooleanOptionalAction,
                   default=bool(D.get("val_sample_resample", False)),
                   help="Re-draw the validation sample at every validation (trainer default). "
                        "OFF here: the draw's seed depends only on the step count, so re-drawing "
                        "makes every run pick the epoch with the easiest subset (sd ~0.0017 "
                        "NDCG@5 per draw) and gives each K its own draw sequence. A fixed subset "
                        "keeps validations comparable with each other and with the start value")
    p.add_argument("--validate-at-start", dest="validate_at_start",
                   action=argparse.BooleanOptionalAction,
                   default=bool(D.get("run_validation_at_start", True)),
                   help="Score the init checkpoint before the first epoch: it seeds the best "
                        "value / best.pt and is the reference for early-stopping patience, so a "
                        "run that never improves stops after `patience` validations with the "
                        "init as its best (SFT-control protocol)")

    # Reward pieces.
    p.add_argument("--sasrec-transform", default=str(D.get("sasrec_transform", "sigmoid")),
                   help="Score -> reward transform, applied at LOOKUP time (the sasrec factors "
                        "are transform-agnostic, so no cache rebuild). 'exp' makes the reward the "
                        "teacher's softmax: R = exp(score / T). The default sigmoid saturates: "
                        "on yambda_50m the top-1, 100th and true item all get reward 1.000")
    p.add_argument("--sasrec-temperature", type=float, default=float(D.get("sasrec_temperature", 1.0)))
    p.add_argument("--sasrec-temperatures", type=float, nargs="+", default=None,
                   help="Make the sasrec temperature a grid dimension (combos with sasrec only)")
    p.add_argument("--interaction-exponents", type=float, nargs="+", default=None,
                   help="Grid dimension: exponent gamma of the interaction modulator in "
                        "multi-component combos, R = sasrec * interaction ** gamma * ... "
                        "(composed at lookup time, no cache cost; default 1)")

    # Sampler.
    p.add_argument("--curriculum-steps", type=int, default=int(D.get("curriculum_steps", 3000)),
                   help="Batch-steps of the easy->hard curriculum ramp (protocol constant; "
                        "~1.1 epochs at last-8 / batch 256 here)")
    p.add_argument("--curriculum-epochs", type=float, default=D.get("curriculum_epochs"),
                   help="Define the easy->hard ramp in EPOCHS over the fine-tuning examples "
                        "(overrides --curriculum-steps). In micro-steps the ramp's length in "
                        "passes depends on the data scope, the micro-batch and K (3000 steps: "
                        "~11 epochs of the 50m last-8 subset, <1 epoch of the full layout); in "
                        "epochs it does not")
    p.add_argument("--sampler-temperature", type=float, default=float(D.get("sampler_temperature", 1.0)))

    # Gradient-norm calibration of the gflow weight.
    p.add_argument("--gflow-weight-mode", choices=["fixed_grad_ratio", "constant"],
                   default=str(D.get("gflow_weight_mode", "fixed_grad_ratio")))
    p.add_argument("--gflow-target-grad-ratio", type=float,
                   default=float(D.get("gflow_target_grad_ratio", 1.0)))
    p.add_argument("--rho-targets", type=float, nargs="+", default=None,
                   help="Make rho_target a grid dimension (rho=1.0 keeps legacy run dirs)")
    p.add_argument("--gflow-constant-weight", type=float, default=0.1)
    p.add_argument("--gflow-calibration-batches", type=int,
                   default=int(D.get("gflow_calibration_batches", 5)))
    p.add_argument("--gflow-calibration-eps", type=float,
                   default=float(D.get("gflow_calibration_eps", 1e-12)))
    p.add_argument("--gflow-calibration-seed", type=int,
                   default=int(D.get("gflow_calibration_seed", 1234)))

    # Factorized reward caches (one per component).
    p.add_argument("--cache-batch-size", type=int, default=256,
                   help="Examples per batch when computing the reward factors")
    p.add_argument("--hidden-dtype", choices=["float16", "float32"],
                   default=str(D.get("hidden_dtype", "float16")),
                   help="Storage dtype of the per-example SASRec hidden states")
    p.add_argument("--cache-verify-batches", type=int, default=1,
                   help="precompute_reward_factors --verify-batches (dense-vs-factorized self-check)")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true",
                   help="Build the component caches needed by the selected combos, then exit "
                        "(no datamodule in this process: run one task per component first)")
    p.add_argument("--cache-components", nargs="+", choices=list(CACHE_COMPONENTS), default=None,
                   help="With --cache-only: restrict to these components (independent cache preparation)")

    # Candidate-scoring memory cap (TIGER-sized; K=10 keeps 256 x 8).
    p.add_argument("--max-candidate-batch", type=int,
                   default=int(D.get("max_candidate_batch", 4096)))

    # Training (SFT-control protocol).
    p.add_argument("--max-epochs", type=int, default=int(D.get("max_epochs", 500)))
    p.add_argument("--validate-every", type=int, default=int(D.get("validate_every", 5)))
    p.add_argument("--patience", type=int, default=int(D.get("patience", 10)))
    p.add_argument("--min-epochs", type=int, default=int(D.get("min_epochs", 0)),
                   help="Early-stopping warm-up: validations of the first N epochs never count "
                        "towards --patience, so a run cannot be stopped before its curriculum "
                        "has reached full difficulty (set it to --curriculum-epochs)")
    p.add_argument("--lr", type=float, default=float(D.get("lr", 3e-5)))
    p.add_argument("--weight-decay", type=float, default=float(D.get("weight_decay", 1e-2)))
    p.add_argument("--batch-size", type=int, default=int(D.get("batch_size", 256)))
    p.add_argument("--grad-accum", type=int, default=int(D.get("grad_accum", 8)))
    p.add_argument("--max-grad-norm", type=float, default=float(D.get("max_grad_norm", 0.5)))
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"],
                   default=str(D.get("amp_dtype", "float16")))
    p.add_argument("--val-batch-size", type=int, default=int(D.get("val_batch_size", 256)))
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--loader-prefetch-factor", type=int, default=2,
                   help="Number of CPU minibatches prepared ahead per training worker")
    p.add_argument("--reward-prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="Prepare compact reward factors in training DataLoader workers")
    p.add_argument("--tensorized-train-loader", action=argparse.BooleanOptionalAction, default=True,
                   help="Assemble training batches from compact arrays instead of Python records")

    # Final evaluation.
    p.add_argument("--test-ks", type=int, nargs="+", default=list(D.get("test_ks", [5, 10, 20])))
    p.add_argument("--eval-num-beams", type=int, default=int(D.get("eval_num_beams", 20)))

    # Output.
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE_ROOT))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/exp4-reward_ablation"))
    p.add_argument("--metrics-csv-name", type=str, default=None,
                   help="Metrics CSV file name inside --artifacts-dir (default: the dataset's "
                        "own CSV; a changed protocol keeps its rows in a separate file)")
    p.add_argument("--promote-root", type=str, default="",
                   help="Copy best.pt per run here; empty disables promotion")
    p.add_argument("--control-init-baseline", type=Path, default=Path(CONTROL_INIT_BASELINE),
                   help="SFT-control init_baseline.json to import instead of re-evaluating "
                        "the init checkpoint (pass '' to force re-evaluation)")

    # Safety / debug.
    p.add_argument("--skip-sanity-check", action="store_true")
    p.add_argument("--sanity-nll-threshold", type=float, default=6.0)
    p.add_argument("--teacher-ndcg-threshold", type=float, default=float(TEACHER_NDCG_THRESHOLD),
                   help="sanity_check_teacher floor on the first ~768 validation examples "
                        "(dataset spec; a misaligned teacher id space scores ~0)")
    p.add_argument("--skip-init-baseline", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny wiring run: ~2048-example caches, 1 epoch, unshuffled")
    args = p.parse_args(argv)
    if args.num_workers < 0 or args.loader_prefetch_factor < 1:
        p.error("--num-workers must be nonnegative and --loader-prefetch-factor positive")
    return args


def cache_last_n(args: argparse.Namespace) -> int | None:
    """--per-user-last-n of the factor caches (the value their manifests
    record): None = no tail subset (--train-last-n 0)."""
    last_n = int(args.train_last_n)
    return last_n if last_n > 0 else None


def cache_random_n(args: argparse.Namespace) -> int | None:
    """--per-user-random-n of the factor caches: None = no random subset."""
    random_n = int(getattr(args, "train_random_n", 0) or 0)
    return random_n if random_n > 0 else None


def subset_spec(args: argparse.Namespace) -> dict[str, Any]:
    """The fine-tuning subset in the terms a cache manifest records it."""
    if cache_last_n(args) is not None and cache_random_n(args) is not None:
        raise SystemExit("--train-last-n and --train-random-n are exclusive")
    random_n = cache_random_n(args)
    return {"per_user_last_n": cache_last_n(args), "per_user_random_n": random_n,
            "per_user_random_seed": int(args.train_subset_seed) if random_n is not None else None}


def describe_subset(spec: Mapping[str, Any]) -> str:
    if spec.get("per_user_random_n") is not None:
        return (f"{spec['per_user_random_n']} random examples per user "
                f"(seed {spec.get('per_user_random_seed')})")
    if spec.get("per_user_last_n") is not None:
        return f"each user's last {spec['per_user_last_n']} examples"
    return "the full train layout"


def finetune_indices(train_dataset, args: argparse.Namespace) -> list[int]:
    """Train-dataset indices the run fine-tunes on, in dataset order = the row
    order of the factor caches: every user's last --train-last-n examples,
    --train-random-n random examples per user, or every example of the
    checkpoint's layout when both are 0."""
    spec = subset_spec(args)
    if spec["per_user_random_n"] is not None:
        return per_user_random_indices(train_dataset, spec["per_user_random_n"], spec["per_user_random_seed"])
    if spec["per_user_last_n"] is not None:
        return per_user_tail_indices(train_dataset, spec["per_user_last_n"])
    return list(range(len(train_dataset)))


def metrics_csv_path(args: argparse.Namespace) -> Path:
    return abspath(args.artifacts_dir) / (args.metrics_csv_name or METRICS_CSV_NAME)


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def sasrec_method_cfg() -> dict[str, Any]:
    params: dict[str, Any] = {}
    trial = abspath(SASREC_TRIAL)
    if trial.exists():
        payload = load_json(trial)
        params = payload.get("best_params", payload)
    model_keys = ("d_model", "nhead", "num_layers", "dropout", "pos_drop")
    return {
        "checkpoint_path": str(abspath(SASREC_CHECKPOINT)),
        "model": {k: params[k] for k in model_keys if k in params},
    }


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """The datamodule / teacher config shared by every combo (the reward is
    composed at lookup time, so the config carries no combo)."""
    config = copy.deepcopy(load_json(abspath(SFT_CONFIG)))
    config["experiment_name"] = f"gflow_{DATASET}"
    config["seed"] = int(args.seed)
    config["data"] = {**config.get("data", {}), **DATA_OVERRIDES}
    for key in ("interactions_path", "reviews_path", "metadata_path", "semantic_id_path", "embeddings_path"):
        if config["data"].get(key):
            config["data"][key] = str(abspath(config["data"][key]))

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_cfg.update(DATASET_OVERRIDES)
    dataset_cfg["positive_criterion"] = POSITIVE_CRITERION
    config["dataset"] = dataset_cfg

    dataloader = dict(config.get("dataloader", {}))
    dataloader["train_batch_size"] = int(args.batch_size)
    dataloader["batch_size"] = int(args.batch_size)
    dataloader["validation_batch_size"] = int(args.val_batch_size)
    dataloader["num_workers"] = int(args.num_workers)
    dataloader["persistent_workers"] = int(args.num_workers) > 0
    config["dataloader"] = dataloader

    config["method"] = {
        "name": "gflow",
        "reward_model": "multiply",
        "reward_epsilon": GFLOW_REWARD_EPSILON,
        "reward": {"components": [component_spec(name, args) for name in COMPONENT_ORDER]},
        "sasrec": sasrec_method_cfg(),
    }
    return config


def cache_config(run_config: Mapping[str, Any], name: str, args: argparse.Namespace) -> dict[str, Any]:
    """precompute_reward_factors config for ONE raw component (weight 1)."""
    config = json.loads(json.dumps(run_config))
    spec = component_spec(name, args)
    config["experiment_name"] = f"gflow_{DATASET}_{name}_factors"
    config["method"] = {
        "name": "gflow",
        "reward_model": str(spec["name"]),
        "reward_epsilon": GFLOW_REWARD_EPSILON,
        "reward": dict(spec["reward"]),
        "sasrec": dict(run_config["method"]["sasrec"]),
    }
    return config


def components_for(combos: list[tuple[str, ...]]) -> list[str]:
    needed = {name for combo in combos for name in combo}
    return [name for name in CACHE_COMPONENTS if name in needed]


# --------------------------------------------------------------------------- #
# Factorized reward caches
# --------------------------------------------------------------------------- #
def _smoke_examples(args: argparse.Namespace) -> int:
    return int(args.batch_size) * 8


def component_cache_dir(name: str, args: argparse.Namespace) -> Path:
    return abspath(args.cache_root) / name


def ensure_component_cache(
    name: str, run_config: Mapping[str, Any], config_dir: Path, args: argparse.Namespace,
    num_subset_examples: int | None,
) -> dict[str, Any]:
    cache_dir = component_cache_dir(name, args)
    manifest_path = cache_dir / "manifest.json"
    config_path = config_dir / f"cache_config_{name}.json"
    write_json(config_path, cache_config(run_config, name, args))
    with control.file_lock(cache_dir.parent / f"{cache_dir.name}.build.lock"):
        if manifest_path.exists() and not args.force_cache:
            return check_component_manifest(load_json(manifest_path), cache_dir, name,
                                            num_subset_examples, args)
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "precompute_reward_factors.py"),
            "--config", str(config_path),
            "--output-dir", str(cache_dir),
            "--split", "train",
            "--batch-size", str(int(args.cache_batch_size)),
            "--num-workers", str(int(args.num_workers)),
            "--device", str(args.device),
            "--hidden-dtype", str(args.hidden_dtype),
            "--verify-batches", str(int(args.cache_verify_batches)),
        ]
        if cache_last_n(args) is not None:
            cmd += ["--per-user-last-n", str(cache_last_n(args))]
        if cache_random_n(args) is not None:
            cmd += ["--per-user-random-n", str(cache_random_n(args)),
                    "--per-user-random-seed", str(int(args.train_subset_seed))]
        if args.smoke:
            cmd += ["--max-examples", str(_smoke_examples(args))]
        print(f"[{name}] precomputing reward factors: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        manifest = load_json(manifest_path)
    return check_component_manifest(manifest, cache_dir, name, num_subset_examples, args)


def check_component_manifest(
    manifest: Mapping[str, Any], cache_dir: Path, name: str,
    num_subset_examples: int | None, args: argparse.Namespace,
    *, similarity_mode: str | None = None,
) -> dict[str, Any]:
    if manifest.get("kind") != CACHE_KIND:
        raise RuntimeError(f"{cache_dir} is not a factorized reward cache (kind={manifest.get('kind')!r}).")
    component = manifest.get("component", {})
    mode = similarity_mode or SIMILARITY_MODE_OF.get(name)
    reward_model = "similarity" if mode is not None else name
    if component.get("name") != reward_model:
        raise RuntimeError(
            f"Factor cache at {cache_dir} holds component {component.get('name')!r}, "
            f"expected {reward_model!r}. Rebuild with --force-cache."
        )
    if mode is not None and component.get("mode") != mode:
        raise RuntimeError(
            f"Factor cache at {cache_dir} stores similarity mode {component.get('mode')!r}, "
            f"expected {mode!r}. Rebuild with --force-cache."
        )
    # Cache rows are positions in the fine-tuning subset: caches of the full
    # layout, of a per-user tail and of a per-user random draw (n, seed) are
    # never interchangeable.
    wanted = subset_spec(args)
    stored = {key: (int(manifest[key]) if manifest.get(key) is not None else None) for key in wanted}
    if stored != wanted:
        raise RuntimeError(
            f"Factor cache at {cache_dir} covers {describe_subset(stored)}, but this run "
            f"fine-tunes on {describe_subset(wanted)}. "
            "Use a separate --cache-root or rebuild with --force-cache."
        )
    if num_subset_examples is not None:
        cached = int(manifest.get("num_examples", 0))
        expected = _smoke_examples(args) if args.smoke else num_subset_examples
        if cached < min(expected, num_subset_examples):
            raise RuntimeError(
                f"Factor cache at {cache_dir} covers {cached} examples but the fine-tuning "
                f"subset has {num_subset_examples}. Rebuild with --force-cache (config drift?)."
            )
    verification = manifest.get("verification")
    if verification:
        print(f"[{name}] factor cache self-check: {json.dumps(verification, sort_keys=True)}", flush=True)
    return dict(manifest)


def verify_cache_alignment(dm, subset_indices: list[int], caches: Mapping[str, FactorizedRewardCache],
                           *, tensorized: bool = True) -> None:
    """Spot-check that subset example order matches every cache's rows via the
    stored target semantic rows (catches config drift / tie-order changes)."""
    mapping = tiger_runner.item_id_to_semantic_row(dm)
    n = min(256, min(c.num_examples for c in caches.values()))
    loader = indexed_train_loader(dm, subset_indices, shuffle=False, batch_size=n, seed=0,
                                  num_workers=0, tensorized=tensorized)
    batch = next(iter(loader))
    targets = batch["target_item_ids"].reshape(-1).long()
    ours = mapping[targets.clamp_min(0).clamp_max(mapping.numel() - 1)]
    indices = torch.arange(int(targets.numel()))
    for name, cache in caches.items():
        theirs = cache.target_rows(indices).long()
        agree = float((ours == theirs).float().mean().item())
        if agree < 1.0:
            raise RuntimeError(
                f"Factor cache {name!r} disagrees with the datamodule on "
                f"{(1 - agree) * 100:.1f}% of the first {int(targets.numel())} targets — built "
                "from a different data configuration/order. Rebuild with --force-cache."
            )


def composed_cache(
    entry: Mapping[str, Any], caches: Mapping[str, FactorizedRewardCache]
) -> ComposedFactorizedRewardCache:
    """R = (prod_c component_c ** w_c) ** power for one grid entry.

    The sasrec component is a VIEW with the entry's transform / temperature
    (same factors, no rebuild); the interaction component carries the entry's
    exponent when it modulates other components.
    """
    combo = tuple(entry["combo"])
    components = []
    for name in combo:
        cache, weight = caches[name], 1.0
        if name == "sasrec" and entry.get("sasrec_transform", "-") != "-":
            cache = cache.with_config(transform=str(entry["sasrec_transform"]),
                                      temperature=float(entry["sasrec_temperature"]))
        if name == "interaction" and len(combo) > 1:
            weight = float(entry.get("interaction_exponent", 1.0))
        components.append((name, cache, weight))
    return ComposedFactorizedRewardCache(components, composition="multiply", power=float(entry["power"]))


# --------------------------------------------------------------------------- #
# Calibration with the composed cache injected into both variants
# --------------------------------------------------------------------------- #
def calibrate_gflow_weight(
    *, model, base_gflow_cfg: GFlowConfig, sampler, sft_cfg: SFTConfig, cache,
    dm, subset_indices: list[int], device: torch.device, args: argparse.Namespace,
    tag: str, target_ratio: float, batch_size: int,
) -> dict[str, Any]:
    """Same procedure as the 50m script (two GFlowMethod variants, identical
    seeding, full precision, init checkpoint), with the composed factorized
    cache injected so neither variant opens reward_cache_path."""
    sft_variant = GFlowMethod(replace(base_gflow_cfg, sft_weight=1.0, gflow_weight=0.0),
                              candidate_sampler=sampler, sft_method=AliasedSFTMethod(sft_cfg))
    gflow_variant = GFlowMethod(replace(base_gflow_cfg, sft_weight=0.0, gflow_weight=1.0),
                                candidate_sampler=sampler, sft_method=AliasedSFTMethod(sft_cfg))
    sft_variant._reward_cache = cache
    gflow_variant._reward_cache = cache

    params = [p for p in model.parameters() if p.requires_grad]
    loader = indexed_train_loader(
        dm, subset_indices, shuffle=not args.smoke, batch_size=int(batch_size),
        seed=int(args.gflow_calibration_seed), num_workers=0,
        tensorized=bool(args.tensorized_train_loader),
    )
    num_batches = 1 if args.smoke else int(args.gflow_calibration_batches)
    was_training = model.training
    model.train()
    loss_pairs = variant_loss_pairs(
        model, sft_variant, gflow_variant, loader,
        seed=int(args.gflow_calibration_seed), step=0, num_batches=num_batches, device=device,
    )
    result = calibrate_from_loss_pairs(
        loss_pairs, params, target_ratio=float(target_ratio), eps=float(args.gflow_calibration_eps),
    )
    model.train(was_training)
    tiger_runner.seed_everything(int(args.seed))
    print(
        f"[{tag}] calibration: ||g_SFT||~{result['sft_grad_norm_median']:.4g} "
        f"||g_GFlow||~{result['gflow_grad_norm_median']:.4g} r0={result['raw_grad_ratio_median']:.4g} "
        f"cos={result['grad_cosine_median']:.3f} rho_target={result['target_grad_ratio']} "
        f"-> lambda={result['fixed_gflow_weight']:.6g}",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- #
# Init-checkpoint identity
# --------------------------------------------------------------------------- #
def checkpoint_matches(recorded: Any, expected: str | None = None) -> bool:
    """True when a recorded checkpoint path names this script's init checkpoint.

    Compared by repo-relative suffix so cluster and local prefixes may differ.
    """
    return base.checkpoint_path_matches(PROJECT_ROOT, recorded, expected or TIGER_CHECKPOINT)


def stale_run_reason(run_dir: Path) -> str | None:
    """Why a run directory must be neither skipped as completed nor resumed.

    A run fine-tuned from a different init checkpoint (recorded in its
    resolved_config.json) is stale; None means the directory is current.
    """
    resolved = run_dir / "resolved_config.json"
    if not resolved.exists():
        return None
    try:
        recorded = load_json(resolved).get("sft_checkpoint")
    except Exception:  # noqa: BLE001 - unreadable config: treat as stale
        return "unreadable resolved_config.json"
    if checkpoint_matches(recorded):
        return None
    return f"fine-tuned from {recorded}, not {TIGER_CHECKPOINT}"


def retire_stale_run(run_dir: Path, reason: str, tag: str) -> Path:
    """Move a stale run directory aside (nothing is deleted) and return the new path."""
    target = run_dir.with_name(f"{run_dir.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[{tag}] stale run ({reason}); moving {run_dir} -> {target}", flush=True)
    run_dir.rename(target)
    return target


# --------------------------------------------------------------------------- #
# Init baseline: import the SFT control's evaluation when available
# --------------------------------------------------------------------------- #
def ensure_init_baseline(
    *, config, config_path: Path, dm, index, device: torch.device, args: argparse.Namespace,
    output_root: Path, csv_path: Path,
) -> None:
    baseline_path = output_root / "baseline" / "init_metrics.json"
    if args.skip_init_baseline:
        return

    def current() -> bool:
        if not baseline_path.exists():
            return False
        try:
            recorded = load_json(baseline_path).get("checkpoint")
        except Exception:  # noqa: BLE001 - unreadable: redo
            return False
        if checkpoint_matches(recorded):
            return True
        print(f"[init] {baseline_path} was measured on {recorded}, not {TIGER_CHECKPOINT}: redoing",
              flush=True)
        return False

    if current():
        return
    with control.file_lock(baseline_path.with_suffix(".lock")):
        if current():
            return
        control_path = abspath(args.control_init_baseline) if str(args.control_init_baseline) else None
        if control_path is not None and control_path.is_file():
            imported = load_json(control_path)
            complete = all(k in imported for k in ("validation", "test"))
            if complete and not checkpoint_matches(imported.get("checkpoint")):
                print(f"[init] control baseline {control_path} was measured on "
                      f"{imported.get('checkpoint')}, not {TIGER_CHECKPOINT}: evaluating instead",
                      flush=True)
            elif complete:
                print(f"[init] importing the SFT-control init baseline from {control_path}", flush=True)
                splits = {"validation": dict(imported["validation"]), "test": dict(imported["test"])}
                payload = {"dataset": DATASET, "checkpoint": str(abspath(TIGER_CHECKPOINT)),
                           "imported_from": str(control_path),
                           "control_checkpoint": imported.get("checkpoint"), **splits}
                write_json(baseline_path, payload)
                identity = {"dataset": DATASET, "combo": "-", "power": "-", "objective": "-",
                            "num_negatives": "-", "seed": int(args.seed), "variant": "init"}
                upsert_metrics_csv(csv_path, metric_rows(identity, splits, {"status": "completed"}))
                return
        print("[init] evaluating the untouched init checkpoint (beam-20, full splits) ...", flush=True)
        model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
        control.load_tiger_checkpoint(model, abspath(TIGER_CHECKPOINT))
        model.to(device).eval()
        max_eval = 2 if args.smoke else args.max_eval_batches
        splits = {
            split: control.evaluate_split(model, loader(), index, ks=tuple(args.test_ks),
                                          num_beams=int(args.eval_num_beams), max_batches=max_eval)
            for split, loader in (("validation", dm.val_dataloader), ("test", dm.test_dataloader))
        }
        write_json(baseline_path, {"dataset": DATASET, "checkpoint": str(abspath(TIGER_CHECKPOINT)), **splits})
        identity = {"dataset": DATASET, "combo": "-", "power": "-", "objective": "-",
                    "num_negatives": "-", "seed": int(args.seed), "variant": "init"}
        upsert_metrics_csv(csv_path, metric_rows(identity, splits, {"status": "completed"}))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Curriculum length under per-K micro-batch scaling
# --------------------------------------------------------------------------- #
def curriculum_steps_for_batch(args: argparse.Namespace, run_batch: int,
                               num_examples: int | None = None) -> int:
    """The easy->hard ramp in micro-steps for this run's micro-batch.

    With --curriculum-epochs the ramp is that many passes over the run's
    ``num_examples`` fine-tuning examples (base.curriculum_steps_for_epochs),
    whatever the micro-batch. Otherwise:

    --curriculum-steps is defined at the protocol micro-batch (--batch-size).
    Larger K shrinks the micro-batch (batch * (K+1) <= max_candidate_batch), so
    the same number of micro-steps would cover fewer examples and the ramp
    would silently shorten with K. Scaling by batch_size / run_batch keeps the
    ramp at the same number of EXAMPLES for every K (unchanged for K <= 10).
    """
    if getattr(args, "curriculum_epochs", None) is not None:
        if num_examples is None:
            raise ValueError("--curriculum-epochs needs the number of fine-tuning examples")
        return base.curriculum_steps_for_epochs(args.curriculum_epochs, num_examples, run_batch)
    protocol = int(args.curriculum_steps)
    if protocol <= 0 or int(run_batch) >= int(args.batch_size):
        return protocol
    return int(round(protocol * int(args.batch_size) / int(run_batch)))


# --------------------------------------------------------------------------- #
# Negative samplers
# --------------------------------------------------------------------------- #
def needs_teacher(entries: Sequence[Mapping[str, Any]]) -> bool:
    """The SASRec teacher is loaded only for the teacher sampler or a combo
    with the sasrec reward (its factors are cached; the model itself is only
    sanity-checked then)."""
    return any(str(e.get("sampler", DEFAULT_SAMPLER)) == "teacher" or "sasrec" in tuple(e["combo"])
               for e in entries)


def build_sampler(name: str, *, model, mapping: torch.Tensor, sasrec, num_negatives: int,
                  curriculum_steps: int, args: argparse.Namespace) -> SemanticIDCandidateSampler:
    """K negatives + the positive, never the target or a batch-history item.

    teacher  SASRec-scored easy->hard curriculum over the teacher's vocabulary.
    random   uniform over the catalog (no difficulty notion, so no curriculum).
    policy   the model being fine-tuned samples them, ALWAYS on-policy (its own
             distribution from the first step: no curriculum, temperature 1),
             decoding constrained to valid catalog items at every level.
    """
    common = dict(num_negatives=int(num_negatives), include_positive=True, exclude_history=True)
    table = model.item_semantic_ids.detach()
    if name == "teacher":
        if sasrec is None:
            raise RuntimeError("the teacher sampler needs the SASRec teacher model")
        return SemanticIDCandidateSampler(
            CandidateSamplerConfig(mode="curriculum", curriculum_steps=int(curriculum_steps),
                                   temperature=float(args.sampler_temperature), **common),
            item_semantic_ids=table,
            item_id_to_semantic_row=tiger_runner.restrict_item_id_mapping_to_scorer_vocab(mapping, sasrec),
            scorer=sasrec,
        )
    if name == "random":
        return SemanticIDCandidateSampler(CandidateSamplerConfig(mode="random", **common),
                                          item_semantic_ids=table, item_id_to_semantic_row=mapping)
    if name == "policy":
        return SemanticIDCandidateSampler(
            CandidateSamplerConfig(mode="policy",
                                   policy_extra_beams=getattr(args, "policy_extra_beams", None), **common),
            item_semantic_ids=table, item_id_to_semantic_row=mapping,
        )
    raise ValueError(f"Unknown sampler {name!r}; expected one of {SAMPLERS}")


# --------------------------------------------------------------------------- #
# One run (with resume)
# --------------------------------------------------------------------------- #
def run_single(
    *, entry: Mapping[str, Any], config: Mapping[str, Any], config_path: Path, dm,
    subset_indices: list[int], sasrec, index, caches: Mapping[str, FactorizedRewardCache],
    run_dir: Path, csv_path: Path, args: argparse.Namespace,
    init_baseline_path: Path | None = None,
) -> dict[str, Any]:
    tag = entry_tag(entry)
    device = torch.device(args.device)
    stale = stale_run_reason(run_dir)
    if stale is not None:
        retire_stale_run(run_dir, stale, tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    combo = tuple(entry["combo"])

    tiger_runner.seed_everything(int(args.seed))
    model = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    control.load_tiger_checkpoint(model, abspath(TIGER_CHECKPOINT))
    model.to(device)

    run_batch, run_accum = base.batch_settings_for_k(int(entry["num_negatives"]), args)
    curriculum_steps = curriculum_steps_for_batch(args, run_batch, len(subset_indices))
    if args.curriculum_epochs is not None and str(entry.get("sampler", DEFAULT_SAMPLER)) == "teacher":
        print(f"[{tag}] curriculum: {args.curriculum_epochs:g} epochs of {len(subset_indices)} examples "
              f"= {curriculum_steps} micro-steps at batch {run_batch}", flush=True)
    if run_batch != int(args.batch_size):
        print(f"[{tag}] candidate-memory scaling: batch {args.batch_size} -> {run_batch}, "
              f"grad_accum {args.grad_accum} -> {run_accum}, curriculum "
              f"{args.curriculum_steps} -> {curriculum_steps} micro-steps", flush=True)

    mapping = tiger_runner.item_id_to_semantic_row(dm)
    sampler_name = str(entry.get("sampler", DEFAULT_SAMPLER))
    sampler = build_sampler(sampler_name, model=model, mapping=mapping, sasrec=sasrec,
                            num_negatives=int(entry["num_negatives"]),
                            curriculum_steps=curriculum_steps, args=args)
    sft_cfg = SFTConfig(
        compute_train_metrics=False, compute_eval_metrics=True, compute_token_metrics=True,
        compute_item_metrics=False, evaluate_generation=True, generation_topk=(5,),
        generation_num_beams=5, generation_num_return_sequences=5,
        generation_constrain_valid_sequences=True,
    )
    cache = composed_cache(entry, caches)
    base_gflow_cfg = GFlowConfig(
        objective=str(entry["objective"]), sft_weight=1.0, gflow_weight=1.0,
        # Placeholder path: the composed cache is injected, never opened by path;
        # the power is applied by the composed cache, so the method's stays 1.
        reward_cache_path=str(component_cache_dir(combo[0], args)), reward_cache_power=1.0,
        tb_loss_weighting="uniform",
        candidate_sampler={"teacher": "sasrec"}.get(sampler_name, sampler_name),
        num_negative_samples=int(entry["num_negatives"]),
        easy_negative_steps=sampler.cfg.curriculum_steps,
        candidate_sampling_temperature=float(args.sampler_temperature),
    )

    last_ckpt = run_dir / "checkpoints" / "last.pt"
    calibration_path = run_dir / "calibration.json"
    resuming = bool(args.resume) and last_ckpt.exists() and calibration_path.exists()
    if resuming:
        calibration = load_json(calibration_path)
        lambda_gflow = float(calibration["fixed_gflow_weight"])
        print(f"[{tag}] resuming: lambda={lambda_gflow:.6g} from {calibration_path}", flush=True)
    elif args.gflow_weight_mode == "fixed_grad_ratio":
        calibration = calibrate_gflow_weight(
            model=model, base_gflow_cfg=base_gflow_cfg, sampler=sampler, sft_cfg=sft_cfg,
            cache=cache, dm=dm, subset_indices=subset_indices, device=device, args=args,
            tag=tag, target_ratio=float(entry["rho"]), batch_size=run_batch,
        )
        lambda_gflow = float(calibration["fixed_gflow_weight"])
        write_json(calibration_path, calibration)
    else:
        calibration = {"mode": "constant", "fixed_gflow_weight": float(args.gflow_constant_weight),
                       "target_grad_ratio": None}
        lambda_gflow = float(args.gflow_constant_weight)
        write_json(calibration_path, calibration)

    method = GFlowMethod(replace(base_gflow_cfg, gflow_weight=lambda_gflow),
                         candidate_sampler=sampler, sft_method=AliasedSFTMethod(sft_cfg))
    method._reward_cache = cache

    max_epochs = 1 if args.smoke else int(args.max_epochs)
    max_train_batches = 8 if args.smoke else args.max_train_batches
    max_eval_batches = 2 if args.smoke else args.max_eval_batches
    trainer = Trainer.from_optimizer_config(
        model=model, method=method,
        cfg=TrainerConfig(
            max_epochs=max_epochs, gradient_accumulation_steps=int(run_accum),
            max_grad_norm=float(args.max_grad_norm), use_amp=device.type == "cuda",
            amp_dtype=str(args.amp_dtype), validate_every_epochs=int(args.validate_every),
            log_every_steps=200, monitor="val/sft/ndcg@5", monitor_mode="max",
            early_stopping_patience=int(args.patience),
            early_stopping_min_epochs=int(args.min_epochs), save_best=True, save_last=True,
            run_validation_at_start=bool(args.validate_at_start),
            checkpoint_dir=run_dir / "checkpoints", max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches, val_sample_fraction=args.val_sample_fraction,
            val_sample_resample=bool(args.val_sample_resample),
        ),
        lr=float(args.lr), weight_decay=float(args.weight_decay), device=device,
        logger=base.make_status_logger(tag),
    )
    resumed_from_epoch = None
    if resuming:
        trainer.load_checkpoint(last_ckpt, load_optimizer=True, load_scheduler=True)
        resumed_from_epoch = int(trainer.state.epoch)
        print(f"[{tag}] resumed from {last_ckpt}: epoch {resumed_from_epoch + 1}, "
              f"best {trainer.state.best_metric}, bad_epochs {trainer.state.bad_epochs}/{args.patience}",
              flush=True)

    resolved = {
        "dataset": DATASET, "combo": combo_key(combo), "reward_power": float(entry["power"]),
        "objective": str(entry["objective"]), "num_negatives": int(entry["num_negatives"]),
        "seed": int(args.seed), "gflow": method.cfg.to_dict(),
        "gflow_weight_mode": str(args.gflow_weight_mode), "rho_target": float(entry["rho"]),
        "batching": {"batch_size": run_batch, "gradient_accumulation": run_accum,
                     "effective_batch": run_batch * run_accum},
        "input_pipeline": {"num_workers": int(args.num_workers),
                           "prefetch_factor": int(args.loader_prefetch_factor),
                           "reward_prefetch": bool(args.reward_prefetch),
                           "tensorized_train_loader": bool(args.tensorized_train_loader)},
        "calibration": {k: v for k, v in calibration.items() if k != "per_batch"},
        "sampler": {"type": {"teacher": "sasrec_sce_curriculum", "random": "uniform_catalog",
                             "policy": "model_stochastic_beam_search"}[sampler_name],
                    "name": sampler_name, "mode": sampler.cfg.mode,
                    "num_negatives": int(entry["num_negatives"]), "include_positive": True,
                    "exclude_history": True, "curriculum_steps": sampler.cfg.curriculum_steps,
                    "curriculum_steps_protocol": int(args.curriculum_steps),
                    "curriculum_epochs": args.curriculum_epochs if sampler_name == "teacher" else None,
                    "temperature": float(sampler.cfg.temperature),
                    "on_policy": sampler_name == "policy", "valid_item_constraint": True,
                    "policy_extra_beams": sampler.cfg.policy_extra_beams if sampler_name == "policy" else None,
                    "teacher": str(abspath(SASREC_CHECKPOINT)) if sampler_name == "teacher" else None},
        "reward": {"form": cache.describe()["form"], "cache_kind": CACHE_KIND,
                   "sasrec_transform": entry.get("sasrec_transform", "-"),
                   "sasrec_temperature": entry.get("sasrec_temperature", "-"),
                   "interaction_exponent": float(entry.get("interaction_exponent", 1.0)),
                   "components": [component_spec(n, args) for n in combo],
                   "component_caches": {n: str(component_cache_dir(n, args)) for n in combo}},
        "data": {"positive_criterion": POSITIVE_CRITERION,
                 "datamodule_train_target_policy": config["dataset"].get("train_target_policy", "all"),
                 "datamodule_train_last_n": (int(config["dataset"]["train_last_n"])
                                             if config["dataset"].get("train_target_policy") == "last_n" else None),
                 "finetune_per_user_last_n": cache_last_n(args),
                 "finetune_per_user_random_n": cache_random_n(args),
                 "finetune_subset_seed": subset_spec(args)["per_user_random_seed"],
                 "finetune_subset": ("per_user_random" if cache_random_n(args) is not None
                                     else "full_layout" if cache_last_n(args) is None else "per_user_tail"),
                 "num_datamodule_train_examples": len(dm.train_dataset),
                 "num_finetune_examples": len(subset_indices),
                 "val_sample_fraction": args.val_sample_fraction},
        "optimizer": {"lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "trainer": trainer.cfg.to_dict(), "sft_checkpoint": str(abspath(TIGER_CHECKPOINT)),
        "item_vocab_size": int(dm.item_vocab_size), "resumed_from_epoch": resumed_from_epoch,
    }
    write_json(run_dir / "resolved_config.json", resolved)

    print(f"[{tag}] fine-tuning: lambda={lambda_gflow:.6g}, epochs<={max_epochs}", flush=True)
    print(f"[{tag}] input pipeline: tensorized={args.tensorized_train_loader}, "
          f"reward_prefetch={args.reward_prefetch}, workers={args.num_workers}, "
          f"prefetch_factor={args.loader_prefetch_factor}", flush=True)
    fit = trainer.fit(
        indexed_train_loader(dm, subset_indices, shuffle=not args.smoke,
                             batch_size=int(run_batch), seed=int(args.seed),
                             reward_cache=cache if args.reward_prefetch else None,
                             prefetch_factor=int(args.loader_prefetch_factor),
                             tensorized=bool(args.tensorized_train_loader)),
        dm.val_dataloader(),
    )
    write_json(run_dir / "training_history.json", {"history": fit.state.history})

    best_on_disk = run_dir / "checkpoints" / "best.pt"
    checkpoint_for_eval = (
        fit.best_checkpoint_path
        or (str(best_on_disk) if best_on_disk.exists() else None)
        or fit.last_checkpoint_path
    )
    if checkpoint_for_eval:
        trainer.load_checkpoint(checkpoint_for_eval, load_optimizer=False, load_scheduler=False)

    # best_epoch -1: no epoch beat the init checkpoint, so best.pt is a copy of
    # the init weights. Their beam-20 full-split metrics are the init baseline;
    # reuse it instead of spending hours re-measuring the same weights.
    splits = None
    final_source = "evaluated"
    if fit.state.best_epoch == -1 and init_baseline_path is not None and not args.smoke:
        try:
            baseline = load_json(init_baseline_path)
        except Exception:  # noqa: BLE001 - missing/unreadable: evaluate
            baseline = {}
        if checkpoint_matches(baseline.get("checkpoint")) and all(
            k in baseline for k in ("validation", "test")
        ):
            splits = {"validation": dict(baseline["validation"]), "test": dict(baseline["test"])}
            final_source = "init_baseline"
            print(f"[{tag}] no epoch beat the init checkpoint; final metrics = init baseline "
                  f"({init_baseline_path})", flush=True)
    if splits is None:
        splits = {
            split: control.evaluate_split(model, loader(), index, ks=tuple(args.test_ks),
                                          num_beams=int(args.eval_num_beams),
                                          max_batches=max_eval_batches)
            for split, loader in (("validation", dm.val_dataloader), ("test", dm.test_dataloader))
        }
    write_json(run_dir / "final_metrics.json", {**splits, "source": final_source})

    if args.promote_root and checkpoint_for_eval:
        promote_dir = (abspath(args.promote_root) / DATASET /
                       f"gflow_{combo_key(combo)}{variant_suffix(entry)}_p{power_tag(entry['power'])}_"
                       f"{entry['objective']}_k{entry['num_negatives']}")
        promote_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_for_eval, promote_dir / "best.pt")
        sidecar = Path(str(checkpoint_for_eval) + ".json")
        if sidecar.exists():
            shutil.copy2(sidecar, promote_dir / "best.pt.json")
        write_json(promote_dir / "final_metrics.json", splits)

    result = {
        "dataset": DATASET, "combo": combo_key(combo), "power": float(entry["power"]),
        "objective": str(entry["objective"]), "num_negatives": int(entry["num_negatives"]),
        "sampler": sampler_name,
        "seed": int(args.seed), "status": "completed", "lambda_gflow": lambda_gflow,
        "target_grad_ratio": calibration.get("target_grad_ratio"),
        "raw_grad_ratio_median": calibration.get("raw_grad_ratio_median"),
        "grad_cosine_median": calibration.get("grad_cosine_median"),
        "best_epoch": fit.state.best_epoch, "best_val_ndcg5": fit.state.best_metric,
        "epochs_completed": int(fit.state.epoch + 1), "resumed_from_epoch": resumed_from_epoch,
        "final_metrics_source": final_source, "sft_checkpoint": str(abspath(TIGER_CHECKPOINT)),
        "elapsed_sec": time.time() - started, "run_dir": str(run_dir), "splits": splits,
    }
    write_json(run_dir / "result.json", result)
    result.update(csv_identity(entry))
    identity = {**csv_identity(entry),
                "dataset": DATASET, "combo": result["combo"], "power": result["power"],
                "objective": result["objective"], "num_negatives": result["num_negatives"],
                "seed": result["seed"], "variant": "gflow"}
    extra = {"status": "completed", "lambda_gflow": lambda_gflow,
             "target_grad_ratio": result["target_grad_ratio"],
             "raw_grad_ratio_median": result["raw_grad_ratio_median"],
             "grad_cosine_median": result["grad_cosine_median"],
             "best_epoch": result["best_epoch"], "best_val_ndcg5": result["best_val_ndcg5"],
             "epochs_completed": result["epochs_completed"], "run_dir": str(run_dir)}
    upsert_metrics_csv(csv_path, metric_rows(identity, splits, extra))

    del model, method, trainer, sampler, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def select_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = build_grid(args)
    if args.index is None:
        return grid
    text = str(args.index)
    try:
        start, end = (int(x) for x in text.split("-", 1)) if "-" in text else (int(text), int(text))
    except ValueError:
        raise SystemExit(f"--index must be 'N' or 'A-B', got {text!r}") from None
    if start < 0 or start > end:
        raise SystemExit(f"--index range {text!r} is invalid")
    if start >= len(grid):
        print(f"--index {text}: start beyond the grid ({len(grid)} entries) — nothing to do")
        return []
    end = min(end, len(grid) - 1)
    print(f"--index {text}: running {end - start + 1} grid entries [{start}..{end}]", flush=True)
    return grid[start:end + 1]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list:
        for i, entry in enumerate(build_grid(args)):
            print(f"{i:4d}  {entry_tag(entry)}")
        print(f"total: {len(build_grid(args))} runs")
        return
    grid = select_grid(args)
    if not grid:
        return

    tiger_runner.seed_everything(int(args.seed))
    device = torch.device(args.device)
    output_root = abspath(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_csv_path(args)

    config = build_run_config(args)
    config_path = output_root / "data_reward_config.json"
    write_json(config_path, config)
    combos = [tuple(c) for c in {tuple(e["combo"]) for e in grid}]
    combos.sort(key=lambda c: [all_combos().index(c)])
    needed = components_for(combos)

    if args.cache_only:
        # No datamodule in this process (the precompute builds its own);
        # sizes are re-checked against the subset by the training tasks.
        wanted = [c for c in needed if not args.cache_components or c in args.cache_components]
        for name in wanted:
            manifest = ensure_component_cache(name, config, output_root, args, None)
            print(f"[{name}] factor cache ready: {manifest['num_examples']} examples, "
                  f"{manifest.get('factor_bytes', 0) / 1e9:.2f} GB of factors", flush=True)
        return

    pending: list[dict[str, Any]] = []
    for entry in grid:
        run_dir = run_dir_for(output_root, entry, int(args.seed))
        done = run_dir / "result.json"
        if done.exists() and not args.force and stale_run_reason(run_dir) is None:
            try:
                if load_json(done).get("status") == "completed":
                    print(f"[{entry_tag(entry)}] already completed — skipping (--force to redo)", flush=True)
                    continue
            except Exception:  # noqa: BLE001 - unreadable marker: re-run
                pass
        pending.append(entry)
    if not pending:
        print("nothing to run", flush=True)
        return

    data_cfg = tiger_runner.build_data_config(dict(config), config_path=config_path)
    dm = tiger_runner.build_data_module(data_cfg)
    setup_started = time.perf_counter()
    print(f"[{DATASET}] preparing datamodule on CPU", flush=True)
    dm.setup()
    print(f"[{DATASET}] datamodule ready in {time.perf_counter() - setup_started:.1f}s", flush=True)
    store = dm.interaction_store
    subset_indices = finetune_indices(dm.train_dataset, args)
    print(f"[{DATASET}] catalog={dm.item_vocab_size} items, train={len(dm.train_dataset)} examples "
          f"(fine-tune subset: {len(subset_indices)}), aux store={len(store) if store is not None else 0} events",
          flush=True)

    probe = tiger_runner.build_model(dict(config), dm, config_path=config_path)
    control.load_tiger_checkpoint(probe, abspath(TIGER_CHECKPOINT))
    probe.to(device)
    if not args.skip_sanity_check:
        nll = control.sanity_check_checkpoint(probe, dm, device, float(args.sanity_nll_threshold))
        print(f"[{DATASET}] init sanity OK: teacher-forced val NLL {nll:.3f}", flush=True)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sasrec = None
    if needs_teacher(pending):
        sasrec = tiger_runner.build_sasrec_teacher_model(dm, dict(config["method"]["sasrec"]), config_path=config_path)
        sasrec.to(device).eval()
        for param in sasrec.parameters():
            param.requires_grad_(False)
        if not args.skip_sanity_check:
            # ~768 examples whatever the validation batch (3 batches at 256, 12 at 64).
            teacher_ndcg = base.sanity_check_teacher(
                sasrec, dm, device, batches=max(3, -(-768 // int(args.val_batch_size))),
                threshold=float(args.teacher_ndcg_threshold))
            print(f"[{DATASET}] SASRec teacher sanity OK: val NDCG@10 {teacher_ndcg:.4f}", flush=True)
    else:
        print(f"[{DATASET}] teacher-free grid (samplers "
              f"{sorted({str(e.get('sampler', DEFAULT_SAMPLER)) for e in pending})}, no sasrec reward): "
              "the SASRec teacher is not loaded", flush=True)

    caches: dict[str, FactorizedRewardCache] = {}
    for name in needed:
        manifest = ensure_component_cache(name, config, output_root, args, len(subset_indices))
        caches[name] = FactorizedRewardCache(component_cache_dir(name, args), power=1.0)
        print(f"[{name}] factor cache ready: {manifest['num_examples']} examples, "
              f"{manifest.get('factor_bytes', 0) / 1e9:.2f} GB of factors", flush=True)
    verify_cache_alignment(dm, subset_indices, caches, tensorized=bool(args.tensorized_train_loader))

    index = control.CatalogIndex(tiger_runner.build_model(dict(config), dm, config_path=config_path), dm)
    ensure_init_baseline(config=config, config_path=config_path, dm=dm, index=index, device=device,
                         args=args, output_root=output_root, csv_path=csv_path)

    all_results: list[dict[str, Any]] = []
    for entry in pending:
        run_dir = run_dir_for(output_root, entry, int(args.seed))
        try:
            result = run_single(entry=entry, config=config, config_path=config_path, dm=dm,
                                subset_indices=subset_indices, sasrec=sasrec, index=index,
                                caches=caches, run_dir=run_dir, csv_path=csv_path, args=args,
                                init_baseline_path=output_root / "baseline" / "init_metrics.json")
        except Exception as exc:  # noqa: BLE001 - keep the grid going
            import traceback

            print(f"[{entry_tag(entry)}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            result = {"dataset": DATASET, "combo": combo_key(entry["combo"]),
                      "power": float(entry["power"]), "objective": entry["objective"],
                      "num_negatives": int(entry["num_negatives"]), "seed": int(args.seed),
                      "target_grad_ratio": float(entry["rho"]), "status": "failed",
                      "error": str(exc), "run_dir": str(run_dir)}
            identity = {**csv_identity(entry),
                        **{k: result[k] for k in ("dataset", "combo", "power", "objective",
                                                  "num_negatives", "seed", "target_grad_ratio")}}
            upsert_metrics_csv(csv_path, [
                {**identity, "variant": "gflow", "split": split, "status": "failed",
                 "error": str(exc)[:300]} for split in ("validation", "test")
            ])
        all_results.append(result)
        write_json(output_root / "summary.json", {"runs": all_results})

    print("\n=== summary ===", flush=True)
    for row in all_results:
        line = (f"{row['combo']:<36} p={row['power']:<6g} {row['objective']:<6} "
                f"K={row['num_negatives']:<5} {row['status']:<10}")
        if row.get("status") == "completed":
            line += (f" lambda={row['lambda_gflow']:.4g} best_val_ndcg@5={row['best_val_ndcg5']}"
                     f" test_NDCG@5={row['splits']['test'].get('NDCG@5')}")
        print(line, flush=True)
    print(f"metrics CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
