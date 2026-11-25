"""Prepare global-time CE inputs with the training layout of each student."""
from __future__ import annotations

import filecmp
import json
from pathlib import Path
import shutil

import gpt2_gflow_inputs as sources


def checkpoint_path(model: str, dataset: str, input_root: Path) -> Path:
    return input_root / sources.CHECKPOINTS / model / dataset / "best.pt"


def stage_inputs(model: str, dataset: str, input_root: Path) -> dict:
    """Called under the preparation lock for this model and dataset."""
    for relative in sources.input_files(dataset, student=model):
        source, target = sources.REPO / relative, input_root / relative
        if target.is_file():
            if not filecmp.cmp(source, target, shallow=False):
                raise RuntimeError(f"Staged input differs from {source}: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    return sources.verify_sources(dataset, input_root, student=model)


def configure_inputs(base, model: str, dataset: str, input_root: Path) -> None:
    sources.configure_inputs(base, input_root)
    base.MATCHED_PROTOCOL = sources.CONFIG_DIR / "exp2-tuned-params/finetune_base_matched.json"
    if model == "tiger":
        base.DATASETS[dataset]["tiger_dir"] = str(checkpoint_path(model, dataset, input_root).parent)


def build_run_config(base, model: str, dataset: str, b) -> dict:
    """Keep the reward specs, target selection and cutoffs identical to global GFlow."""
    b.prefix_from_dense = model == "gpt2"
    if model == "gpt2":
        profile, _ = base.tuned_profile(dataset, b)
        model_yaml = base.runner.load_yaml(base.EXP1_GPT2_MODEL_YAML)
        user_count = base.gpt2_runner.user_ids_count(model_yaml, profile)
        config = base.build_run_config(dataset, b, user_count)
        config["pretrained_model"] = profile["model"]
    elif model == "tiger":
        directory = Path(base.DATASETS[dataset]["tiger_dir"])
        resolved = base.load_json(directory / "resolved_config.json")
        # TIGER's checkpoint sidecar omits d_kv; the winning trial's resolved
        # model records it (64 on Beauty, 32 on Instruments).
        architecture = {**resolved["model"], **base.load_json(directory / "best.pt.json")["model_config"]}
        config = base.build_run_config(dataset, b, int(architecture["user_ids_count"]))
        config["model"] = config["pretrained_model"] = architecture
    else:
        raise ValueError(f"Unknown student {model!r}")
    config["experiment_name"] = f"ce_distill_global_{model}_{dataset}"
    return config


def verify_layout(model: str, dataset: str, input_root: Path, dm) -> dict:
    """Compare the prepared split to the archived full checkpoint data summary."""
    directory = checkpoint_path(model, dataset, input_root).parent
    recorded = json.loads((directory / "data_summary.json").read_text())
    summary = dm.summary()
    fields = ("global_val_cutoff", "global_test_cutoff", "item_vocab_size", "num_users",
              "num_val_examples", "num_test_examples", "mean_val_history_len", "mean_test_history_len")
    expected = {key: recorded[key] for key in fields}
    expected["num_train_examples"] = recorded["num_train_examples"]
    if model == "gpt2":
        audit = dm.dense_prefix_audit
        if audit["dense_train_rows"] != recorded["num_train_examples"]:
            raise ValueError(f"{dataset}: reconstructed dense windows differ from the GPT-2 checkpoint")
        # Each next-item label of the original H/stride-H windows becomes a row.
        expected["num_train_examples"] = round(recorded["num_train_examples"] * recorded["mean_train_history_len"])
    else:
        audit = {"reference_layout": "prefix_target", "training_context": "rolling"}
        expected["mean_train_history_len"] = recorded["mean_train_history_len"]
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"{model}/{dataset}: prepared {key}={summary[key]}, expected {value}")
    return {**audit, **expected}
