"""The four pointwise GRPO rewards for the missing LOO/global experiments."""
from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flowrec.models.sasrec import SASRec
from flowrec.rewards.mix import MultiplicativeRewardModel
from flowrec.rewards.sasrec import SASRecRewardConfig, SASRecRewardModel
from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel, load_rqvae_codebooks
from scripts import run_tiger

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ("S", "H", "I", "S×H×I")


class _CandidateSASRec(nn.Module):
    """Score only the generated items, in the teacher's own item ID space."""

    def __init__(self, teacher: SASRec, translation: torch.Tensor):
        super().__init__()
        self.teacher = teacher
        self.item_vocab_size = translation.numel()
        self.register_buffer("translation", translation, persistent=False)

    def score_items(self, batch, candidate_item_ids):
        from scripts.precompute_reward_factors import sasrec_final_hidden

        history = self.translation[batch["history_item_ids"].long()]
        hidden = sasrec_final_hidden(self.teacher, history, batch["history_mask"])
        items = self.teacher.item_embedding(self.translation[candidate_item_ids])
        return torch.bmm(items, hidden.unsqueeze(-1)).squeeze(-1)


def _teacher_translation(data, checkpoint, teacher_vocab_size, dataset, split, supplied):
    if supplied is not None:
        record = json.loads(Path(supplied).read_text())
        catalog_hash = hashlib.sha256(json.dumps(sorted(data.item2id.items())).encode()).hexdigest()
        if (record["target_catalog_sha256"] != catalog_hash
                or record["teacher_item_vocab_size"] != teacher_vocab_size):
            raise ValueError("The supplied SASRec ID translation does not match the student and teacher.")
        table = np.asarray(record["item_id_translation"], dtype=np.int64)
    elif split == "global":
        # Dense teacher windows cover training events from users with at least two
        # positives before the global cutoff. The last-128 student can drop items.
        dense_teacher_ids = import_module("scripts.global-time.yambda_gflow_teacher").dense_teacher_ids
        table, dense_size = dense_teacher_ids(
            data.interactions, val_cutoff=data.global_val_cutoff,
            student_vocab_size=data.item_vocab_size, num_users=data.num_users,
        )
        if dense_size != teacher_vocab_size:
            raise ValueError(f"Global training stream gives {dense_size} SASRec IDs, expected {teacher_vocab_size}")
    else:
        # LOO teachers kept every item of the original factorization. Cold-filtered
        # student histories preserve those IDs in the positive event stream.
        summary = json.loads((checkpoint.parent / "training_summary.json").read_text())
        source = next(run["dataset"] for run in summary["runs"] if run["dataset"]["name"] == dataset)
        actual = data.summary()
        for key in ("num_items_before_cold_filter", "num_interactions", "num_users",
                    "positive_criterion", "split_strategy"):
            if actual[key] != source[key]:
                raise ValueError(f"LOO {dataset}: SASRec source stream differs in {key}; supply a raw-ID translation.")
        if source["num_items_before_cold_filter"] + 1 != teacher_vocab_size or source["num_cold_items_removed"]:
            raise ValueError("LOO SASRec did not use the complete pre-filter catalog; supply a raw-ID translation.")
        if "item_id_prefilter" in data.interactions:
            table = np.zeros(data.item_vocab_size, dtype=np.int64)
            table[data.interactions["item_id"].to_numpy()] = data.interactions["item_id_prefilter"].to_numpy()
            table[0] = 0
        else:
            if data.item_vocab_size != teacher_vocab_size:
                raise ValueError("LOO student is missing its pre-filter SASRec ID mapping.")
            table = np.arange(data.item_vocab_size, dtype=np.int64)
    if (table.shape != (data.item_vocab_size,) or table[0] != 0
            or np.any(table[1:] <= 0) or np.any(table >= teacher_vocab_size)
            or np.unique(table).size != table.size):
        raise ValueError("Incomplete or non-unique SASRec ID translation for the student catalog.")
    return torch.from_numpy(table)


def _sasrec_reward(data, dataset, split, root, checkpoint_override, item_id_translation):
    family = "sasrec_sce" if dataset == "yambda_500m_listens" else "sasrec"
    checkpoint = (Path(checkpoint_override) if checkpoint_override is not None else
                  root / "checkpoints" / "exp2" / f"{split}_split" / "initial" / family / dataset / "best.pt")
    provenance = checkpoint.parent / "provenance.json"
    if provenance.is_file():
        source = json.loads(provenance.read_text())
        if any(source.get(k) != v for k, v in {"dataset": dataset, "split": f"{split}_split",
                                              "stage": "initial", "model": family}.items()):
            raise ValueError(f"Wrong SASRec checkpoint for {dataset}/{split}: {checkpoint}")
    payload = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
    teacher = SASRec(payload["model_config"])
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    translation = _teacher_translation(data, checkpoint, teacher.item_vocab_size, dataset, split, item_id_translation)
    reward = SASRecRewardConfig(transform="softplus", temperature=1., allow_invalid_candidates=False)
    return SASRecRewardModel(_CandidateSASRec(teacher, translation), reward,
                             item_id_by_semantic_row=run_tiger.item_id_by_semantic_row(data))


def build_backfill_reward(reward, *, dataset, split, data, input_root=None, sasrec_checkpoint=None,
                          item_id_translation=None, rqvae_checkpoint=None):
    """Construct active components only, with exponent one and no batch normalization."""
    if reward not in REWARDS:
        raise ValueError(f"Unknown GRPO reward {reward!r}, expected one of {REWARDS}")
    split_name = {"loo": "leave_one_out", "global": "global_time"}[split]
    if data.cfg.split_strategy != split_name:
        raise ValueError(f"Requested {split} reward on {data.cfg.split_strategy} data")
    components = []
    for name in reward.split("×"):
        if name == "S":
            root = Path(input_root or ROOT).expanduser().resolve()
            component = _sasrec_reward(data, dataset, split, root, sasrec_checkpoint, item_id_translation)
        elif name == "H":
            # These promoted tokenizers store level offsets in semantic_ids.json.
            sid_dir = Path(data.cfg.semantic_id_path).parent
            checkpoint = Path(rqvae_checkpoint) if rqvae_checkpoint is not None else sid_dir / "model.pt"
            metadata = json.loads((sid_dir / "semantic_id_metadata.json").read_text())
            level_stride = int(metadata["rqvae_config"]["codebook_size"]) if metadata["level_offset_tokens"] else 0
            cfg = SimilarityRewardConfig(mode="hierarchical", sharpness=1., level_stride=level_stride,
                                         token_offset=int(data.cfg.offset_semantic_ids_by_one))
            component = SimilarityRewardModel(cfg, codebooks=load_rqvae_codebooks(checkpoint))
        else:
            component = run_tiger.build_reward_model(
                "interaction", data=data, train_loader=None, config_path=ROOT, seed=42,
                reward_cfg={"training_window_only": split == "global"}, svd_cfg={}, svd_model=None,
                sasrec_model=None, reward_epsilon=1e-6,
            )
        components.append(component)
    if len(components) == 1:
        return components[0]
    return MultiplicativeRewardModel(components, weights=[1., 1., 1.])
