#!/usr/bin/env python
"""Minimal probe: exp2 TIGER SFT predictions on the first N beauty val examples.

  python scripts/predict_beauty_val.py -n 8 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_tiger as tiger_runner  # noqa: E402

CONFIG = ROOT / "configs/exp2-tuned-params/tiger_beauty_sft.json"
CHECKPOINT = ROOT / "checkpoints/exp2-tuned_params/initial/tiger/beauty/best.pt"


def main() -> None:
    ap = argparse.ArgumentParser("exp2 beauty val predictions")
    ap.add_argument("-n", type=int, default=8, help="Number of validation examples")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--num-beams", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    # The config's data paths point at the training host; use the local dump.
    config["data"]["reviews_path"] = str(ROOT / "data/raw/dataframe_format/reviews_beauty_2014.parquet")
    config["data"]["metadata_path"] = str(ROOT / "data/raw/dataframe_format/meta_beauty_2014.parquet")
    config["dataloader"]["num_workers"] = 0

    data_cfg = tiger_runner.build_data_config(config, config_path=CONFIG)
    dm = tiger_runner.build_data_module(data_cfg)
    dm.setup()

    model = tiger_runner.build_model(config, dm, config_path=CONFIG)
    state = torch.load(CHECKPOINT, map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("flow_head.")}, strict=True
    )
    device = torch.device(args.device)
    model.to(device).eval()

    n = min(args.n, len(dm.val_dataset))
    examples = [dm.val_dataset[i] for i in range(n)]
    batch = dm._make_collator()(examples)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        generated = model.generate(
            batch,
            num_beams=args.num_beams,
            num_return_sequences=args.top_k,
            constrain_valid_sequences=True,
        )  # [N, top_k, sid_len]

    # Row i of the model's semantic table corresponds to sorted(semantic_ids)[i].
    raw_by_tokens = {
        tuple(map(int, row)): raw
        for raw, row in zip(sorted(dm.semantic_ids), model.item_semantic_ids.cpu().tolist())
    }

    for i, ex in enumerate(examples):
        target_tokens = tuple(map(int, batch["target_token_ids"][i].cpu().tolist()))
        history = ex.get("history_raw_items", [])
        print(f"\n[{i}] user={ex.get('raw_user_id', ex.get('user_id'))} "
              f"history({len(history)}): ... {' '.join(map(str, history[-3:]))}")
        print(f"    target: {ex.get('target_raw_item', '?')} tokens={list(target_tokens)}")
        for rank, pred in enumerate(generated[i].cpu().tolist(), start=1):
            key = tuple(map(int, pred))
            hit = "  <-- HIT" if key == target_tokens else ""
            print(f"    pred {rank}: {raw_by_tokens.get(key, '<not in catalog>')} tokens={list(key)}{hit}")


if __name__ == "__main__":
    main()
