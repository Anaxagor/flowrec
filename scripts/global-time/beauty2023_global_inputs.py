"""Original global Beauty 2023 students, teacher, and their target layouts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from yambda_gflow_inputs import stage_asset_files


ROOT = Path(__file__).resolve().parents[2]
DATASET = "beauty_2023"
INITIAL = Path("checkpoints/exp2/global_split/initial")
TOKENIZER = Path("datasets/semantic_ids/exp2-tuned-params") / DATASET
TIGER_RECORD = Path("scripts/global-time/tiger_beauty2023_trial31_checkpoint.json")
TEACHER_DIR = INITIAL / "sasrec" / DATASET


def _read(path):
    return json.loads(path.read_text())


def _assets(model, dataset):
    if dataset != DATASET or model not in ("gpt2", "tiger"):
        raise ValueError(f"Unsupported Beauty 2023 global checkpoint: {model}/{dataset}")
    student_dir = INITIAL / model / dataset
    files = {Path("data/raw/dataframe_format") / f"{kind}_{dataset}.parquet": None
             for kind in ("reviews", "meta")}
    files[TOKENIZER / "semantic_ids.json"] = "5433d3ffe96629079ad06f46511f34e155cb0d0926cc8d19420e0f409ae843c9"
    files[TOKENIZER / "model.pt"] = "d0a08555a8e39551cc29123a40b7d54f66ff7bc439e220b6c7f0067167495719"
    checkpoint = student_dir / ("trial_31.pt" if model == "tiger" else "best.pt")
    if model == "tiger":
        record = _read(ROOT / TIGER_RECORD)
        if record["checkpoint"] != str(checkpoint):
            raise ValueError("Wrong global TIGER Beauty 2023 checkpoint in trial record")
        files[checkpoint] = record["checkpoint_sha256"]
        files[TIGER_RECORD] = None
    for family in (("sasrec", "gpt2") if model == "gpt2" else ("sasrec",)):
        directory = INITIAL / family / dataset
        record = _read(ROOT / directory / "provenance.json")
        expected = dict(model=family, dataset=dataset, split="global_split", stage="initial",
                        checkpoint=str(directory / "best.pt"))
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Wrong initialization provenance at {directory}")
        files[directory / "best.pt"] = record["sha256"]
        for name in ("study_config.json", "best_trial.json", "data_summary.json"):
            files[directory / name] = record["metadata_sha256"][name]
    return checkpoint, files


def stage_inputs(model, dataset, input_root):
    _, files = _assets(model, dataset)
    return stage_asset_files(files, input_root, Path(input_root) / f"{model}-{dataset}.json", root=ROOT)


@dataclass
class Beauty2023Profile:
    model: str
    dataset: str
    checkpoint: Path
    teacher_checkpoint: Path
    teacher_trial: Path
    semantic_ids: Path
    codebook: Path
    model_config: dict
    data_config: dict
    recorded_summary: dict
    source_identity: dict

    def run_config(self, args):
        data = self.data_config
        paths = {k: data[k] for k in ("reviews_path", "metadata_path", "semantic_id_path")}
        # Projecting metadata to its IDs preserves the review membership filter.
        # Disjoint GPT-2 dense windows are expanded only after train-catalog filtering.
        dataset = {**data, "source": "amazon", "metadata_ids_only": True,
                   "train_example_layout": "prefix_target", "train_prefix_from_dense": self.model == "gpt2",
                   "max_sequence_length": data["max_history_len"]}
        loader = dict(train_batch_size=args.batch_size, validation_batch_size=args.val_batch_size,
                      num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                      pin_memory=True, include_raw_in_batch=False, train_drop_last=False)
        return {"data": {"source": "amazon", **paths}, "dataset": dataset,
                "dataloader": loader, "model": dict(self.model_config), "seed": data["seed"]}

    def verify_layout(self, dm):
        """Check the full split and original SID row ordering before building rewards."""
        import torch

        actual, original = dm.summary(), self.recorded_summary
        fields = ("split_strategy", "global_val_cutoff", "global_test_cutoff", "num_interactions",
                  "item_vocab_size", "num_users", "sid_len", "positive_criterion", "post_split_filter",
                  "interaction_store", "num_val_examples", "num_test_examples",
                  "mean_val_history_len", "mean_test_history_len")
        expected = {k: original[k] for k in fields}
        if self.model == "gpt2":
            audit = dm.dense_prefix_audit
            if audit["dense_train_rows"] != original["num_train_examples"]:
                raise ValueError("Beauty 2023 dense windows differ from the GPT-2 initialization")
            expected["num_train_examples"] = round(original["num_train_examples"] * original["mean_train_history_len"])
        else:
            audit = {"reference_layout": "prefix_target"}
            expected.update({k: original[k] for k in ("num_train_examples", "mean_train_history_len")})
        for field, value in expected.items():
            if actual[field] != value:
                raise ValueError(f"{self.model}/{self.dataset}: prepared {field}={actual[field]!r}, expected {value!r}")
        checkpoint = torch.load(self.checkpoint, mmap=True, map_location="cpu", weights_only=False)
        rows = torch.as_tensor([dm.semantic_ids[k] for k in sorted(dm.semantic_ids)])
        if not torch.equal(rows, checkpoint["model_state_dict"]["item_semantic_ids"]):
            raise ValueError(f"Prepared semantic catalog differs from the global {self.model} checkpoint")
        return {**audit, **expected, "train_target_policy": "all", "max_history_len": dm.cfg.max_history_len}

    def verify_teacher_catalog(self, dm, destination):
        """Reconstruct the teacher's dense observed-item numbering from the full event stream."""
        import numpy as np
        import torch
        from yambda_gflow_teacher import dense_teacher_ids

        table, size = dense_teacher_ids(dm.interactions, val_cutoff=dm.global_val_cutoff,
                                        student_vocab_size=dm.item_vocab_size, num_users=dm.num_users)
        ckpt = torch.load(self.teacher_checkpoint, mmap=True, map_location="cpu", weights_only=False)
        if (not np.array_equal(table, np.arange(size)) or size != ckpt["model_config"]["item_vocab_size"]
                or size != ckpt["model_state_dict"]["item_embedding.weight"].shape[0]):
            raise ValueError("Global Beauty 2023 teacher and student item catalogs differ")
        record = {"teacher_checkpoint": str(self.teacher_checkpoint), "teacher_item_vocab_size": size,
                  "target_catalog_sha256": hashlib.sha256(json.dumps(sorted(dm.item2id.items())).encode()).hexdigest(),
                  "item_id_translation": table.tolist()}
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record) + "\n")
        temporary.replace(path)
        return path


def load_profile(model, dataset, input_root):
    checkpoint, files = _assets(model, dataset)
    root = Path(input_root)
    staged = ({str(p): str(ROOT / p) for p in files} if root == ROOT
              else _read(root / f"{model}-{dataset}.json"))
    paths = {p: Path(staged[str(p)]) for p in files}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"Missing Beauty 2023 staged input {p}; run prepare")
    if model == "tiger":
        record = _read(paths[TIGER_RECORD])
        data, architecture, summary = record["data"], record["model"], record["data_summary"]
    else:
        directory = checkpoint.parent
        study = _read(paths[directory / "study_config.json"])
        architecture = {**_read(paths[directory / "best_trial.json"])["model"],
                        "user_ids_count": study["regime"]["user_ids_count"]}
        data = study["data_config"]
        summary = _read(paths[directory / "data_summary.json"])
    teacher = _read(paths[TEACHER_DIR / "study_config.json"])["data_config"]
    # The two streams must use the same raw events, factors, cutoffs and cold-item rule.
    common = ("reviews_path", "metadata_path", "item_key", "metadata_item_key", "filter_reviews_to_metadata",
              "min_user_interactions", "min_item_interactions", "positive_criterion", "use_verified_only",
              "deduplicate_consecutive", "min_history_len", "max_history_len", "global_val_timestamp",
              "global_test_timestamp", "global_history_policy", "global_eval_target_mode",
              "split_strategy", "filter_cold_items_after_split", "cold_item_train_source",
              "drop_items_missing_semantic_id", "train_all_prefixes", "train_target_policy", "seed")
    for k in common:
        if data[k] != teacher[k]:
            raise ValueError(f"{model} and SASRec Beauty 2023 preprocessing differs at {k}")
    if (teacher["train_example_layout"] != "dense_causal" or teacher["train_window_len"] is not None
            or teacher["train_window_stride"] is not None or teacher["semantic_id_path"] is not None
            or data["train_example_layout"] != ("prefix_target" if model == "tiger" else "dense_causal")
            or data["train_target_policy"] != "all" or data["split_strategy"] != "global_time"
            or data["semantic_id_path"] != str(TOKENIZER / "semantic_ids.json")):
        raise ValueError("Unexpected native Beauty 2023 training layout")
    data = {**data, **{k: str(paths[Path(data[k])]) for k in ("reviews_path", "metadata_path", "semantic_id_path")}}
    teacher_checkpoint = TEACHER_DIR / "best.pt"
    identity = {"student": str(checkpoint), "student_sha256": files[checkpoint],
                "teacher": str(teacher_checkpoint), "teacher_sha256": files[teacher_checkpoint], "inputs": staged}
    return Beauty2023Profile(model, dataset, paths[checkpoint], paths[teacher_checkpoint],
                             paths[TEACHER_DIR / "best_trial.json"], paths[TOKENIZER / "semantic_ids.json"],
                             paths[TOKENIZER / "model.pt"], architecture, data, summary, identity)
