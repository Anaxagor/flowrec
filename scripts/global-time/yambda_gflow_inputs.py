"""Inputs of the global-time Yambda last-128 student checkpoints."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
INITIAL = Path("checkpoints/exp2/global_split/initial")
SID_ROOT = Path("datasets/semantic_ids/exp2-tuned-params")
GPT_RECORD = Path("scripts/global-time/gpt2_yambda50m_last128_checkpoint.json")
Y50 = "yambda_50m_listens"
Y500 = "yambda_500m_listens"
PAIRS = (("tiger", Y50), ("gpt2", Y50), ("tiger", Y500))


def _read(path):
    return json.loads(path.read_text())


def _assets(model, dataset):
    if (model, dataset) not in PAIRS:
        raise ValueError(f"Unsupported global-time experiment: {model}/{dataset}")
    teacher = "sasrec" if dataset == Y50 else "sasrec_sce"
    student_dir = INITIAL / model / dataset
    teacher_dir = INITIAL / teacher / dataset
    sid_dir = SID_ROOT / dataset
    files = {sid_dir / "semantic_ids.json": None, sid_dir / "model.pt": None,
             Path("data/raw/yambda/embeddings.parquet"): None,
             Path("data/raw/yambda/flat") / ("50m" if dataset == Y50 else "500m") / "multi_event.parquet": None}
    bundles = [(teacher, teacher_dir, ("best_trial.json",))]
    if model == "gpt2":
        student_dir = student_dir / "last_128"
        files[GPT_RECORD] = None
        files[student_dir / "best.pt"] = _read(ROOT / GPT_RECORD)["checkpoint_sha256"]
    else:
        bundles.append((model, student_dir, ("resolved_config.json", "data_summary.json", "best.pt.json")))
    for family, directory, sidecars in bundles:
        record = _read(ROOT / directory / "provenance.json")
        expected = dict(split="global_split", stage="initial", model=family, dataset=dataset,
                        checkpoint=str(directory / "best.pt"))
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Unexpected checkpoint provenance at {directory}")
        files[directory / "best.pt"] = record["sha256"]
        for name in sidecars:
            files[directory / name] = record["metadata_sha256"][name]
    return student_dir, teacher_dir, files


def _stage_manifest(input_root, model, dataset):
    return Path(input_root) / f"{model}-{dataset}.json"


def stage_inputs(model, dataset, input_root):
    """Copy immutable generations once across all three pairs, including the shared embeddings."""
    from flowrec.mmap_scratch import stage_mmap_file

    _, _, files = _assets(model, dataset)
    target = _stage_manifest(input_root, model, dataset)
    paths = {str(p): str(stage_mmap_file(ROOT / p, input_root, expected_sha256=digest))
             for p, digest in files.items()}
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as stream:
        json.dump(paths, stream, indent=2, sort_keys=True)
        temp = Path(stream.name)
    temp.replace(target)
    return paths


@dataclass
class GlobalYambdaProfile:
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
        flat = dict(self.data_config)
        data = {name: flat[name] for name in ("interactions_path", "semantic_id_path", "embeddings_path",
                                              "use_hf", "hf_dataset_name", "streaming", "max_rows")}
        data["source"] = "yambda"
        dataset = {**flat, "source": "yambda", "max_sequence_length": flat["max_history_len"]}
        workers = args.num_workers
        loader = dict(train_batch_size=args.batch_size, batch_size=args.batch_size,
                      validation_batch_size=args.val_batch_size, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=flat["pin_memory"],
                      train_drop_last=flat["train_drop_last"], include_raw_in_batch=False)
        return {"data": data, "dataset": dataset, "model": dict(self.model_config),
                "dataloader": loader, "seed": flat["seed"]}

    def verify_layout(self, dm):
        """Compare the full pretraining population before any reward-only time filter."""
        actual = dm.summary()
        keys = ("dataset_size", "split_strategy", "global_val_cutoff", "global_test_cutoff",
                "num_train_examples", "num_val_examples", "num_test_examples", "mean_train_history_len",
                "mean_val_history_len", "mean_test_history_len", "item_vocab_size", "num_users", "sid_len",
                "num_interactions", "post_split_filter", "interaction_store")
        for k in keys:
            if actual[k] != self.recorded_summary[k]:
                raise ValueError(f"{self.model}/{self.dataset}: {k} differs from the initial checkpoint "
                                 f"({actual[k]!r} vs {self.recorded_summary[k]!r})")
        audit = {k: actual[k] for k in keys}
        audit.update(train_target_policy="last_n", train_last_n=128, max_history_len=50)
        if self.model == "gpt2":
            audit.update(_verify_dense_rows(dm))
        else:
            audit.update(reference_layout="prefix_target")
        return audit


def load_profile(model, dataset, input_root):
    student_dir, teacher_dir, files = _assets(model, dataset)
    input_root = Path(input_root)
    if input_root == ROOT:  # Inspect the original assets without staging them.
        paths = {str(p): ROOT / p for p in files}
    else:
        paths = {k: Path(v) for k, v in _read(_stage_manifest(input_root, model, dataset)).items()}
    if any(not paths[str(p)].is_file() for p in files):
        raise FileNotFoundError(f"Missing staged input for {model}/{dataset}; run prepare again")
    checkpoint = student_dir / "best.pt"
    teacher = teacher_dir / "best.pt"
    if model == "gpt2":
        record = _read(paths[str(GPT_RECORD)])
        architecture = record["model"]
        summary = record["data_summary"]
    else:
        record = _read(paths[str(student_dir / "resolved_config.json")])
        architecture = {**record["model"], **_read(paths[str(student_dir / "best.pt.json")])["model_config"]}
        summary = _read(paths[str(student_dir / "data_summary.json")])
    flat = dict(record["data"])
    expected = dict(split_strategy="global_time", global_history_policy="rolling",
                    global_eval_target_mode="last_per_user", train_last_n=128,
                    train_all_prefixes=True, train_target_policy="last_n", max_history_len=50,
                    filter_cold_items_after_split=True, cold_item_train_source="observed",
                    drop_items_missing_semantic_id=False, drop_items_missing_embedding=True,
                    train_example_layout="dense_causal" if model == "gpt2" else "prefix_target")
    if model == "gpt2":
        expected.update(train_dense_match_prefix=True, train_window_len=50, train_window_stride=1)
    if any(flat.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected last-128 training layout in {checkpoint}")
    for k in ("interactions_path", "embeddings_path", "semantic_id_path"):
        flat[k] = str(paths[flat[k]])
    flat.update(train_example_layout="prefix_target", train_dense_match_prefix=False,
                train_prefix_from_dense=False, include_raw_in_batch=False)
    sid_dir = SID_ROOT / dataset
    identity = {"student": str(checkpoint), "student_sha256": files[checkpoint], "teacher": str(teacher),
                "teacher_sha256": files[teacher], "inputs": {str(p): str(paths[str(p)]) for p in files}}
    return GlobalYambdaProfile(model, dataset, paths[str(checkpoint)], paths[str(teacher)],
                              paths[str(teacher_dir / "best_trial.json")], paths[str(sid_dir / "semantic_ids.json")],
                              paths[str(sid_dir / "model.pt")], architecture, flat, summary, identity)


def _verify_dense_rows(dm):
    """Replay the original stride-one terminal-label windows in bounded, complete user blocks."""
    import numpy as np
    import pandas as pd
    from flowrec.data.post_split_filtering import _filter_and_remap_examples
    from flowrec.data.sequence_builder import build_sequence_splits
    from flowrec.data.yambda import _sequence_split_config_from_yambda

    cfg = replace(_sequence_split_config_from_yambda(dm.cfg), train_example_layout="dense_causal",
                  train_dense_match_prefix=True, train_window_len=50, train_window_stride=1)
    frames = {name: getattr(dm, name + "_dataset").examples for name in ("train", "val", "test")}
    users = {name: frame["user_id"].to_numpy() for name, frame in frames.items()}
    events = dm.interactions
    event_users = events["user_id"].to_numpy()
    compared = dict.fromkeys(frames, 0)
    observed = set()
    columns = ["raw_user_id", "user_id", "target_position", "target_timestamp", "target_raw_item",
               "target_item_id", "target_rating", "target_verified_purchase", "history_item_ids",
               "history_raw_items", "history_ratings", "history_verified_purchase"]
    for first in range(0, dm.num_users, 64):
        bounds = [first, first + 64]
        lo, hi = np.searchsorted(event_users, bounds)
        # Replay from the original IDs preserved by remap_interactions_to_catalog;
        # cold events retain their pre-filter IDs, timestamps and raw sequence positions.
        block = events.iloc[lo:hi].assign(item_id=events["item_id_prefilter"].iloc[lo:hi])
        reference = build_sequence_splits(block, cfg=cfg)
        for name, frame in frames.items():
            lo, hi = np.searchsorted(users[name], bounds)
            expected = getattr(reference, name)
            if name == "train":
                # The dense objective predicts just the final next-item label in each window.
                for mask, labels, target, history in zip(expected.label_mask, expected.label_item_ids,
                                                         expected.target_item_id, expected.history_raw_items):
                    if not mask[-1] or any(mask[:-1]) or labels[-1] != target:
                        raise ValueError("GPT-2 reference window has a different active label")
                    observed.update(history)
                observed.update(expected.target_raw_item)
                expected = expected.assign(
                    target_item_id=expected.target_raw_item.map(dm.item2id),
                    history_item_ids=[[dm.item2id[item] for item in history] for history in expected.history_raw_items],
                )
            else:
                expected, _ = _filter_and_remap_examples(expected, item2id=dm.item2id,
                                                         min_history_len=dm.cfg.min_history_len)
            try:
                pd.testing.assert_frame_equal(expected[columns].reset_index(drop=True),
                                              frame.iloc[lo:hi][columns].reset_index(drop=True),
                                              check_dtype=False, check_exact=True)
            except AssertionError as exc:
                raise ValueError(f"GPT-2 last-128 dense/prefix mismatch in {name}, users {bounds}") from exc
            compared[name] += len(expected)
    originals = events.drop_duplicates("item_id_prefilter")[["raw_item_id", "item_id_prefilter"]]
    items = originals.loc[originals.raw_item_id.isin(observed)].sort_values("item_id_prefilter")
    if list(items.raw_item_id) != list(dm.id2item[1:]):
        raise ValueError("GPT-2 supervised dense windows and prefix catalog differ")
    return {"reference_layout": "dense_causal_match_prefix", "train_window_len": 50,
            "train_window_stride": 1, "dense_prefix_equal": True, "rows_compared": compared}
