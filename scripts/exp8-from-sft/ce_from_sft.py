#!/usr/bin/env python3
"""CE adapters for the exp8 runner's explicitly selected SFT-only source.

Both arms estimate CE against q = R / sum(R): ``reward`` draws directly
from q; ``curriculum_smooth_is`` draws from a smooth reward-half mixture
with full-q support and applies raw q/proposal weights. K=3 means three
IID catalog draws WITH replacement, not three guaranteed negative items.
The observed target remains in the separate SFT term and can also be drawn
by CE. Excluding it would change the teacher distribution and objective.

The common source runtime owns checkpoint loading, data, subsets and cache
preparation. This module reuses exp7's tested training, calibration and
optimizer-boundary resume loops without substituting exp7's initial weights
or its dense-prefix / Yambda subset defaults.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CE_DIR = ROOT / "scripts/exp7-ce-distillation"
if str(CE_DIR) not in sys.path:
    sys.path.insert(0, str(CE_DIR))
import ce_distill_tiger as tiger
import ce_distill_gpt2 as gpt2

SAMPLINGS = ("reward", "curriculum_smooth_is")
COMPONENTS = ("sasrec", "interaction", "similarity")


def ce_args(session, sampling="reward"):
    """Stage flags only; source-derived data/batching stay in session.args."""
    if sampling not in SAMPLINGS:
        raise ValueError(f"Unsupported exp8 CE sampler: {sampling}")
    b = session.args
    values = dict(vars(b))
    checkpoint_interval = getattr(b, "checkpoint_every_updates", None)
    values.update(
        output_root=Path(session.output_root) / "ce", estimator="sampled",
        sampling=sampling, num_samples=int(getattr(b, "ce_num_samples", 3)),
        catalog_chunk_size=int(getattr(b, "ce_catalog_chunk_size", 16)),
        curriculum_epochs=float(getattr(b, "ce_curriculum_epochs", 5.0)),
        proposal_q_mix=float(getattr(b, "ce_proposal_q_mix", 0.1)),
        calibration_batches=int(getattr(b, "ce_calibration_batches", 5)),
        calibration_seed=int(getattr(b, "ce_calibration_seed", 1234)),
        checkpoint_every_updates=100 if checkpoint_interval is None else int(checkpoint_interval),
        resume=bool(getattr(b, "resume", True)),
        max_train_batches=getattr(b, "max_train_batches", None),
        max_eval_batches=getattr(b, "max_eval_batches", None),
        smoke=bool(getattr(b, "smoke", False)),
    )
    if values["num_samples"] != 3:
        raise ValueError("The exp8 sampled-CE protocol requires exactly three catalog draws")
    return SimpleNamespace(**values)


def ce_entry(cell):
    combo = cell["combo"]
    if not isinstance(combo, str):
        combo = "_x_".join(combo)
    if combo not in (*COMPONENTS, "_x_".join(COMPONENTS)):
        raise ValueError(f"Unsupported exp8 CE reward: {combo}")
    return {"dataset": cell["dataset"], "seed": int(cell["seed"]),
            "combo": combo, "power": float(cell["power"]), "rho": float(cell["rho"])}


def source_config(session):
    config = deepcopy(session.config)
    config["from_sft_source"] = deepcopy(session.source_identity)
    if session.source["model"] == "gpt2" and "pretrained_model" not in config:
        profile, _ = session.base.tuned_profile(session.source["dataset"])
        config["pretrained_model"] = deepcopy(profile["model"])
    return config


def asset_paths(session, config, components, component_paths):
    """Pin the source snapshot and all actual data/reward inputs, not exp2."""
    paths = [Path(session.checkpoint), *map(Path, session.source_assets)]
    paths.extend(Path(value) for key, value in config["data"].items()
                 if key.endswith("_path") and value)
    method = config.get("method", {})
    if "sasrec" in components:
        paths.extend(Path(value) for key, value in method.get("sasrec", {}).items()
                     if key.endswith("_path") and value)
    for spec in method.get("reward", {}).get("components", []):
        if spec.get("name") in components:
            paths.extend(Path(value) for key, value in spec.get("reward", {}).items()
                         if key.endswith("_path") and value)
    paths.extend(map(Path, component_paths))
    paths.extend(Path(path).parent / "factors.pt" for path in component_paths)
    return list(dict.fromkeys(paths))


def trainer_overrides(session):
    """Honor the stage settings resolved together with the source snapshot."""
    b = session.args
    result = {"val_sample_fraction": getattr(b, "val_sample_fraction", None),
              "val_sample_resample": bool(getattr(b, "val_sample_resample", False)),
              "val_sample_seed": int(getattr(b, "val_sample_seed", 0))}
    for arg_name, config_name in (("amp_dtype", "amp_dtype"),
                                  ("max_grad_norm", "max_grad_norm"),
                                  ("validate_every", "validate_every_epochs"),
                                  ("monitor", "monitor")):
        if getattr(b, arg_name, None) is not None:
            result[config_name] = getattr(b, arg_name)
    return result


def result_metadata(session):
    return {"model": session.source["model"], "from_sft_source": session.source_identity,
            "source_id": session.source.get("source_id"), "source_seed": session.source.get("source_seed"),
            "initial_checkpoint": str(session.checkpoint),
            "ce_draws": "three IID full-catalog draws with replacement; target is not excluded"}


@contextmanager
def temporary_attributes(obj, **values):
    previous = {key: getattr(obj, key) for key in values}
    for key, value in values.items():
        setattr(obj, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(obj, key, value)


@contextmanager
def prepared_datamodules(session):
    """Legacy CE loops receive the CPU snapshot instead of rebuilding data."""
    from prepared_data import load_prepared_data

    def build(_cfg):
        dm = load_prepared_data(session)
        if dm is None:
            raise RuntimeError("CE requires a prepared datamodule snapshot; run CPU preparation first")
        return dm

    with temporary_attributes(session.base.tiger_runner, build_data_module=build):
        yield


def _tiger_run(session, args, entry):
    original_identity = tiger.build_identity
    original_trainer = tiger.build_trainer_config
    original_verify = tiger.verify_caches
    config = source_config(session)

    def identity(*items):
        return {**original_identity(*items), "model": "tiger",
                "from_sft_source": session.source_identity}

    def verify(base, dm, indices, caches, model):
        audit = session.audit_data(dm)
        tiger.write_json(tiger.run_dir_for(args, entry) / "source_data_audit.json", audit)
        return original_verify(base, dm, indices, caches, model)

    with temporary_attributes(session.base, build_run_config=lambda _b: deepcopy(config)):
        with temporary_attributes(
            tiger, build_identity=identity,
            asset_paths_for=lambda _base, cfg, components, paths: asset_paths(session, cfg, components, paths),
            build_trainer_config=lambda *items: replace(original_trainer(*items), **trainer_overrides(session)),
            verify_caches=verify,
        ):
            tiger.run_entry(args, entry, None, session.base,
                matched_args_fn=lambda *_items: session.args,
                result_metadata_fn=lambda *_items: result_metadata(session))


def _gpt2_run(session, args, entry):
    def verify(base, dm, indices, caches, model):
        # The original Amazon helper requires a concrete index sequence;
        # source runtimes may represent their full layout with None.
        if indices is None:
            indices = range(len(dm.train_dataset))
        return gpt2.verify_caches(base, dm, indices, caches, model)

    def identity(*items):
        return {**tiger.build_identity(*items), "model": "gpt2",
                "from_sft_source": session.source_identity,
                "data_protocol": "selected SFT-only source snapshot"}

    profile = SimpleNamespace(
        matched_args=lambda *_items: session.args,
        build_run_config=lambda *_items: source_config(session),
        asset_paths_for=lambda _base, _dataset, cfg, components, paths: asset_paths(session, cfg, components, paths),
        build_identity=identity,
        audit_data=lambda _base, dm, _b: session.audit_data(dm),
        trainer_overrides=lambda *_items: trainer_overrides(session),
        verify_caches=verify,
        result_metadata=lambda *_items: result_metadata(session),
        DATA_AUDIT_FILENAME="source_data_audit.json",
    )
    gpt2.run_entry(args, entry, None, session.base, profile=profile)


def run_ce(session, cell):
    """Execute one already selected cell under the bound source runtime."""
    model, dataset = session.source["model"], session.source["dataset"]
    if cell["model"] != model or cell["dataset"] != dataset:
        raise ValueError("CE cell does not match the bound SFT source")
    if any(key in session.source and cell.get(key) != session.source[key]
           for key in ("source_id", "source_seed")):
        raise ValueError("CE cell does not match the bound SFT source identity")
    if int(cell["seed"]) != int(session.args.seed):
        raise ValueError("CE cell seed does not match the bound runtime")
    if model not in ("tiger", "gpt2"):
        raise ValueError(f"Unsupported CE model: {model}")
    args, entry = ce_args(session, cell["sampling"]), ce_entry(cell)
    run_dir = tiger.run_dir_for(args, entry)
    with session.base.control.file_lock(run_dir / "run.lock"):
        with prepared_datamodules(session):
            (_tiger_run if model == "tiger" else _gpt2_run)(session, args, entry)
        result = tiger.read_json(run_dir / "result.json")
        if result.get("status") != "completed":
            raise RuntimeError(f"CE did not complete: {run_dir}")
        return result


def baseline_identity(session, config, args):
    assets = tiger.fingerprint(asset_paths(session, config, (), []))
    b = session.args
    identity = {"checkpoint": assets[str(Path(session.checkpoint))],
        "data": config["data"], "dataset": config["dataset"],
        "data_assets": {str(value): assets[str(value)] for key, value in config["data"].items()
                        if key in ("reviews_path", "metadata_path", "semantic_id_path",
                                   "interactions_path", "embeddings_path") and value},
        "eval_beams": b.eval_num_beams, "ks": b.test_ks,
        "max_batches": 2 if args.smoke else args.max_eval_batches}
    if session.source["model"] == "gpt2":
        identity.update(model="gpt2", pretrained_model=config["pretrained_model"])
    return identity


def prepare_ce(session, *, dm=None, splits=None):
    """Evaluate SFT-only weights once, before allocating the CE training array.

    The baseline format/identity exactly matches the reused exp7 loops. It
    covers validation and test with the stage's beam/evaluation parameters.
    The common preparer may supply its already evaluated ``splits`` (same
    beam/batch protocol), or its prepared ``dm`` to avoid rebuilding data.
    """
    args = ce_args(session)
    b, base = session.args, session.base
    model_type, dataset = session.source["model"], session.source["dataset"]
    config = source_config(session)
    identity = baseline_identity(session, config, args)
    directory = args.output_root / dataset / "baseline"
    path = directory / ("smoke.json" if args.smoke else "init_metrics.json")
    with base.control.file_lock(path.with_suffix(".lock")):
        if path.exists():
            previous = tiger.read_json(path)
            if previous.get("identity") != identity or not all(
                split in previous.get("splits", {}) for split in ("validation", "test")
            ):
                raise RuntimeError(f"CE baseline identity/content differs at {path}; use a new output root")
            return previous
        if splits is not None:
            if not all(split in splits for split in ("validation", "test")):
                raise ValueError("CE preparation requires both validation and test baseline metrics")
            payload = {"identity": identity, "dataset": dataset, "variant": "init",
                       "from_sft_source": session.source_identity, "splits": deepcopy(splits)}
            tiger.write_json(path, payload)
            return payload
        import torch

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        base.tiger_runner.seed_everything(int(b.seed))
        config_path = directory / "data_reward_config.json"
        tiger.write_json(config_path, config)
        if dm is None:
            from prepared_data import load_prepared_data
            dm = load_prepared_data(session)
            if dm is None:
                raise RuntimeError("CE baseline requires a prepared datamodule snapshot; run CPU preparation first")
        tiger.write_json(directory / "source_data_audit.json", session.audit_data(dm))
        if model_type == "gpt2":
            model, _ = base.build_gpt2_from_tuned_checkpoint(dataset, dm)
        elif model_type == "tiger":
            model = base.tiger_runner.build_model(config, dm, config_path=config_path)
            expected_table = model.item_semantic_ids.detach().cpu().clone()
            base.control.load_tiger_checkpoint(model, Path(session.checkpoint))
            if not torch.equal(expected_table, model.item_semantic_ids.detach().cpu()):
                raise RuntimeError("Source TIGER checkpoint semantic IDs differ from the prepared data catalog")
        else:
            raise ValueError(f"Unsupported CE model: {model_type}")
        model.to(device).eval()
        max_batches = identity["max_batches"]
        if model_type == "gpt2":
            splits = gpt2.evaluate_splits(base, model, dm, b, device, max_batches)
        else:
            index = base.control.CatalogIndex(model, dm)
            splits = base.control.evaluate_splits(model, dm, index, b, max_batches)
        payload = {"identity": identity, "dataset": dataset, "variant": "init",
                   "from_sft_source": session.source_identity, "splits": splits}
        tiger.write_json(path, payload)
        return payload
