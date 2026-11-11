"""Resolve the two known checkpoint layouts for CE Yambda entry points.

Prefer the original exp2 asset paths when present: existing CE run identities
and reward-cache configs record them.
On migrated checkouts use the canonical layout. Never search other trials or
substitute the full-history / SFT-finetuned GPT-2 models.

Runtime bindings are local to the CE process; shared GFlow source files and
the compact-data implementation stay unchanged, preserving prepared data.
"""
from pathlib import Path


TRIALS = {"yambda_50m_listens": 36, "yambda_500m_listens": 4}
LEGACY_INITIAL = Path("checkpoints/exp2-tuned_params/initial")


def matched_trial_paths(project_root, dataset="yambda_50m_listens"):
    root = Path(project_root)
    trial = TRIALS[dataset]
    legacy = root / "checkpoints/exp2-tuned_params/gpt2_tiger_matched" / dataset
    directory = legacy / "trials" / f"trial_{trial}"
    if (directory / "best.pt").is_file():
        # Keep the whole bundle in one layout. Missing metadata must fail the
        # existing recorded architecture/data audit, not be borrowed elsewhere.
        audit = legacy / "data_audit.json"
        if not audit.is_file() and (directory / "data_audit.json").is_file():
            audit = directory / "data_audit.json"
    else:
        directory = root / "checkpoints/exp2/loo_split/initial/gpt2" / dataset / "last_128"
        audit = directory / "data_audit.json"
    return {"checkpoint": directory / "best.pt", "sidecar": directory / "best.pt.json",
            "resolved_config": directory / "resolved_config.json", "data_audit": audit}


def _legacy_teacher(dataset):
    family = "sasrec_sce" if dataset == "yambda_500m_listens" else "sasrec"
    directory = LEGACY_INITIAL / family / dataset
    return directory / "best_model.pt", directory / "best_trial.json"


def configure_gpt2(base):
    # All inherited config, model and audit helpers call this same module, so
    # they must agree on the selected trial paths before config construction.
    base.matched_source.trial_paths = matched_trial_paths
    for dataset in TRIALS:
        checkpoint, trial = _legacy_teacher(dataset)
        if (base.PROJECT_ROOT / checkpoint).is_file():
            base.DATASETS[dataset] = {**base.DATASETS[dataset],
                                     "sasrec_checkpoint": str(checkpoint),
                                     "sasrec_trial": str(trial)}


def configure_tiger(base):
    # Call after matched.parse_args(), which rebinds the active dataset's
    # constants. Do not alter the shared dataset registry or other datasets.
    dataset = base.DATASET
    if dataset not in TRIALS:
        raise ValueError(f"Unsupported CE Yambda dataset {dataset!r}")
    checkpoint = LEGACY_INITIAL / "tiger" / (
        "yambda_50m/checkpoints/best.pt" if dataset == "yambda_50m_listens"
        else "yambda_500m_listens/best.pt")
    if (base.PROJECT_ROOT / checkpoint).is_file():
        base.TIGER_CHECKPOINT = str(checkpoint)
    teacher, trial = _legacy_teacher(dataset)
    if (base.PROJECT_ROOT / teacher).is_file():
        base.SASREC_CHECKPOINT = str(teacher)
        base.SASREC_TRIAL = str(trial)
