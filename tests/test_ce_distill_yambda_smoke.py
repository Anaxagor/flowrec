"""Synthetic Yambda pools -> frozen target subsets -> factor caches -> CE.

Uses the production 50M/500M configuration builders, but all input files and
models are tiny and temporary. No full dataset or trained checkpoint is read.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from flowrec.data.interaction_store import build_positive_history_store
from flowrec.ft_methods.base import LOSS_COMPONENTS_KEY
from flowrec.ft_methods.gflow.factorized_cache import CACHE_KIND, FactorizedRewardCache
from flowrec.ft_methods.gflow.prefetch import PREPARED_REWARDS_KEY
from flowrec.ft_methods.reward_distillation import RewardDistillationMethod
from flowrec.ft_methods.sft import SFTConfig, SFTMethod
from flowrec.models.sasrec import SASRec, SASRecConfig
from flowrec.rewards.interaction import InteractionRewardConfig, InteractionRewardModel
from flowrec.rewards.sasrec import SASRecRewardConfig, transform_sasrec_scores
from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "scripts/exp7-ce-distillation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import ce_distill_tiger_yambda as runner  # noqa: E402
from precompute_reward_factors import (  # noqa: E402
    EvidenceIndex, interaction_fill_value, interaction_sparse_batch,
    sasrec_final_hidden, sasrec_item_table,
)


def _write_inputs(root):
    rows = []
    for user in range(2):
        rows.extend(dict(uid=user, item_id=(i + user) % 12, timestamp=1000 + i * 10,
                         event_type="listen", played_ratio_pct=90, is_organic=1,
                         track_length_seconds=200) for i in range(140))
        rows.extend(dict(uid=user, item_id=item, timestamp=timestamp, event_type=event,
                         played_ratio_pct=pct, is_organic=1, track_length_seconds=200)
                    for item, timestamp, event, pct in (
                        (0, 1001, "like", 0), (1, 1003, "dislike", 0),
                        (2, 1005, "listen", 20), (3, 10000, "like", 0)))
    pd.DataFrame(rows).to_parquet(root / "events.parquet")
    pd.DataFrame({"item_id": list(range(12)),
                  "normalized_embed": [[float(i), 1., 2.] for i in range(12)]}
                 ).to_parquet(root / "embeddings.parquet")
    # The promoted table is already level-offset. Only the 500M datamodule
    # adds one; semantic row order is lexicographic ("10" before "2").
    raw_sids = {str(i): [i // 4, 256 + i % 4, 512 + i % 5, 768] for i in range(12)}
    (root / "sids.json").write_text(json.dumps(raw_sids))
    return raw_sids


def _write_caches(root, base, args, config, dm, data_cfg, batch, model, offset):
    """Use the factor-precompute kernels, with an actual tiny SASRec teacher."""
    catalog = model.item_semantic_ids.detach()
    row_to_item = base.tiger_runner.item_id_by_semantic_row(dm)
    item_to_row = base.tiger_runner.item_id_to_semantic_row(dm)
    common = {key: batch[key].clone() for key in
              ("user_ids", "target_item_ids", "target_token_ids", "target_timestamps")}
    common["target_semantic_row_ids"] = item_to_row[batch["target_item_ids"]]
    teacher = SASRec(SASRecConfig(item_vocab_size=dm.item_vocab_size,
        max_history_len=50, d_model=8, nhead=2, num_layers=1, dropout=0., pos_drop=0.)).eval()
    hidden = sasrec_final_hidden(teacher, batch["history_item_ids"], batch["history_mask"]).half()
    table, valid = sasrec_item_table(teacher, None, row_to_item)
    sasrec_cfg = SASRecRewardConfig(transform="sigmoid")

    interaction_cfg = InteractionRewardConfig(dataset="yambda")
    interaction = InteractionRewardModel(dm.interaction_store, interaction_cfg,
        positive_store=build_positive_history_store(dm.interactions,
            num_users=dm.num_users, item_vocab_size=dm.item_vocab_size),
        item_id_by_semantic_row=row_to_item)
    fill = interaction_fill_value(interaction, batch, dm.sid_len, torch.device("cpu"))
    sparse_rows, sparse_values = interaction_sparse_batch(
        interaction, EvidenceIndex(interaction), batch, item_to_row,
        sid_len=dm.sid_len, fill_value=fill, reward_epsilon=1e-6, device=torch.device("cpu"))
    lengths = np.asarray([len(row) for row in sparse_rows])
    similarity_cfg = SimilarityRewardConfig(mode="hierarchical", token_offset=offset,
                                            level_stride=256)
    similarity = SimilarityRewardModel(similarity_cfg, codebooks=torch.randn(3, 256, 3))
    specific = {
        "sasrec": {"hidden": hidden, "item_table": table, "row_valid": valid},
        "interaction": {"indptr": torch.tensor(np.r_[0, lengths.cumsum()]),
                        "cols": torch.tensor(np.concatenate(sparse_rows)),
                        "vals": torch.tensor(np.concatenate(sparse_values))},
        "similarity": {"item_tokens": catalog, "code_dist": similarity.code_dist},
    }
    configs = {"sasrec": sasrec_cfg, "interaction": interaction_cfg, "similarity": similarity_cfg}
    caches = {}
    for name in runner.COMPONENTS:
        directory = root / name
        directory.mkdir(parents=True)
        source = directory / "source.json"
        source.write_text(json.dumps(base.cache_config(config, name, args)))
        torch.save({**common, **specific[name]}, directory / "factors.pt")
        manifest = {
            "kind": CACHE_KIND, "factors_path": "factors.pt", "split": "train",
            "config_path": str(source), "num_examples": len(batch["target_item_ids"]),
            "num_catalog_items": len(catalog), "sid_len": dm.sid_len,
            "reward_epsilon": 1e-6, "data_config": asdict(data_cfg), "data": dm.summary(),
            **base.subset_spec(args),
            "component": {"name": name, "mode": "hierarchical" if name == "similarity" else None,
                          "config": asdict(configs[name])},
            "fill_value": fill,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, default=str))
        runner.check_cache_metadata(base, config, name, directory, args, len(batch["target_item_ids"]))
        caches[name] = FactorizedRewardCache(directory)
    # Quantization matches production float16 hidden factors.
    expected = transform_sasrec_scores(hidden.float() @ table.T, sasrec_cfg).clamp_min(1e-6)
    torch.testing.assert_close(caches["sasrec"].component_rows(torch.arange(len(hidden)),
                               torch.device("cpu")), expected)
    assert bool((caches["interaction"].component_rows(torch.arange(len(hidden)),
                 torch.device("cpu")) != fill).any())
    return caches


@pytest.mark.parametrize("dataset,offset", [("yambda_50m_listens", 0), ("yambda_500m_listens", 1)])
@pytest.mark.parametrize("sampling", ["curriculum_smooth", "curriculum_smooth_is"])
def test_yambda_target_subset_factor_prefetch_smooth_ce_backward(tmp_path, monkeypatch, dataset, offset, sampling):
    torch.manual_seed(42)
    matched, runtime = runner.load_runtime()
    base = runtime._base
    previous_dataset = base.DATASET
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        args = runner.parse_args(["--datasets", dataset, "--device", "cpu", "--batch-size", "2",
                                 "--num-workers", "0", "--cache-root", str(tmp_path / "cache"),
                                 "--output-root", str(tmp_path / "output")])
        b = runner.matched_args(args, dataset, 42, matched)
        original = json.loads(base.abspath(base.SFT_CONFIG).read_text())
        raw_sids = _write_inputs(tmp_path)
        original["data"].update(interactions_path=str(tmp_path / "events.parquet"),
                                semantic_id_path=str(tmp_path / "sids.json"),
                                embeddings_path=str(tmp_path / "embeddings.parquet") if offset else None)
        original["model"].update(embedding_dim=8, num_heads=2, num_encoder_layers=1,
                                 num_decoder_layers=1, dim_feedforward=16, d_kv=4,
                                 dropout=0., num_beams=2, top_k=2)
        original["dataloader"].update(pin_memory=False, include_raw_in_batch=False)
        config_path = tmp_path / "tiny_sft.json"
        config_path.write_text(json.dumps(original))
        with monkeypatch.context() as patch:
            patch.setattr(base, "SFT_CONFIG", str(config_path))
            patch.setattr(base, "sasrec_method_cfg", lambda: {"checkpoint_path": str(tmp_path / "tiny_teacher.pt")})
            config = runtime.build_run_config(b)
            data_cfg = base.tiger_runner.build_data_config(config, config_path=config_path)
            dm = base.tiger_runner.build_data_module(data_cfg)
            dm.setup()
            examples = dm.train_dataset.examples
            assert examples.groupby("user_id").size().tolist() == [128, 128]
            assert examples.groupby("user_id").target_timestamp.min().tolist() == [1100, 1100]
            assert examples.groupby("user_id").target_timestamp.max().tolist() == [2370, 2370]
            assert examples.history_item_ids.map(len).max() == 50
            assert examples.history_item_ids.map(len).min() == 10
            assert len(dm.val_dataset) == len(dm.test_dataset) == 2
            assert dm.interaction_store.counts_by_type() == {"like": 4, "dislike": 2, "listen": 2}
            assert dm.semantic_ids == {item: [token + offset for token in tokens]
                                       for item, tokens in raw_sids.items()}
            selected_indices = runtime.finetune_indices(dm.train_dataset, b)
            selected_count = 100 if offset else 256
            if offset:
                assert len(selected_indices) == selected_count
                assert examples.iloc[selected_indices].groupby("user_id").size().tolist() == [50, 50]
                assert selected_indices == runtime.finetune_indices(dm.train_dataset, b)
                assert selected_indices != list(range(selected_count))
            else:
                assert selected_indices is None
            all_batch = next(iter(base.indexed_train_loader(dm, selected_indices, shuffle=False,
                batch_size=len(examples), seed=0, num_workers=0, tensorized=True)))
            torch.testing.assert_close(all_batch["example_indices"], torch.arange(selected_count))
            model = base.tiger_runner.build_model(config, dm, config_path=config_path)
            caches = _write_caches(tmp_path / "factors", base, b, config, dm, data_cfg,
                                   all_batch, model, offset)
            if offset:
                args.prepared_data_root = tmp_path / "prepared"
                args.require_prepared_data = True
                original_dm, original_indices = dm, selected_indices
                runner.prepared_runtime.prepare_once(args, "tiger", runtime, b,
                                                     config, config_path)

                def no_raw_data(*unused, **kwargs):
                    pytest.fail("Prepared TIGER CE must not rebuild or resample the full dataset")

                patch.setattr(base.tiger_runner, "build_data_module", no_raw_data)
                patch.setattr(base, "finetune_indices", no_raw_data)
                runner.prepared_runtime.prepare_once(args, "tiger", runtime, b,
                                                     config, config_path)
                dm = runner.load_data(args, runtime, b, config, config_path)
                selected_indices = runtime.finetune_indices(dm.train_dataset, b)
                assert selected_indices is None
                assert dm._ce_prepared_source_indices.tolist() == original_indices
                assert len(dm.train_dataset) == selected_count
                assert dm.interactions.empty and dm.interaction_store is None
                assert dm.item2id == original_dm.item2id
                assert dm.train_item_counts_by_raw_id == original_dm.train_item_counts_by_raw_id
                actual = next(iter(base.indexed_train_loader(dm, None, shuffle=False,
                    batch_size=selected_count, seed=0, num_workers=0, tensorized=True)))
                assert actual.keys() == all_batch.keys()
                for key in all_batch:
                    torch.testing.assert_close(actual[key], all_batch[key], rtol=0, atol=0)
                for split in ("val", "test"):
                    expected = next(iter(getattr(original_dm, split + "_dataloader")()))
                    actual = next(iter(getattr(dm, split + "_dataloader")()))
                    assert actual.keys() == expected.keys()
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            runner.shared.verify_caches(runtime, dm, selected_indices, caches, model)
            # A wrong late metadata row must not pass a head-only alignment check.
            old_timestamp = caches["sasrec"].factors["target_timestamps"][-1].clone()
            caches["sasrec"].factors["target_timestamps"][-1] += 1
            with pytest.raises(RuntimeError, match="target_timestamps"):
                runtime.verify_cache_alignment(dm, selected_indices, caches)
            caches["sasrec"].factors["target_timestamps"][-1] = old_timestamp

            for combo in runner.COMBOS:
                names = tuple(combo.split("_x_"))
                entry = {"combo": names, "power": 1., "sasrec_transform": "sigmoid",
                         "sasrec_temperature": 1., "interaction_exponent": 1.}
                cache = base.composed_cache(entry, caches)
                batch = next(iter(base.indexed_train_loader(dm, selected_indices, shuffle=True,
                    batch_size=2, seed=29, num_workers=0, tensorized=True, reward_cache=cache)))
                assert PREPARED_REWARDS_KEY in batch
                indices = batch["example_indices"]
                torch.testing.assert_close(batch["target_token_ids"], all_batch["target_token_ids"][indices])
                empty = torch.empty(2, 0, dtype=torch.long)
                product = torch.ones(2, len(model.item_semantic_ids))
                for name in names:
                    product *= caches[name].component_rows(indices, torch.device("cpu")).clamp_min(1e-6)
                for power in (.25, 1., 4., 16.):
                    powered = base.composed_cache({**entry, "power": power}, caches)
                    _, total, actual = powered.select_candidates(indices, empty, torch.device("cpu"),
                        include_full=True, prepared=batch[PREPARED_REWARDS_KEY])
                    expected = product.clamp_min(1e-6).pow(power).clamp_min(1e-6)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(total.reshape(-1), expected.sum(-1))
                method = RewardDistillationMethod(cache, model.item_semantic_ids,
                    SFTMethod(SFTConfig(compute_train_metrics=False, compute_eval_metrics=False)),
                    sampling=sampling, curriculum_steps=10, num_samples=3)
                for step, alpha in ((0, 0.), (5, .5), (10, 1.)):
                    model.zero_grad(set_to_none=True)
                    result = method.compute_loss(model, batch, step=step)
                    assert torch.isfinite(result.loss)
                    assert result.extra["distill_sampled_item_rows"].shape == (2, 3)
                    assert result.metrics["distill/curriculum_high_probability"] == alpha
                    ce = result.extra[LOSS_COMPONENTS_KEY]["distill"]
                    encoder_grad = torch.autograd.grad(ce, model.model.encoder.block[0].parameters(),
                                                       retain_graph=True, allow_unused=True)
                    assert any(gradient is not None and bool(gradient.abs().sum() > 0)
                               for gradient in encoder_grad)
                    result.loss.backward()
                    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    assert grads and all(bool(torch.isfinite(gradient).all()) for gradient in grads)
                    assert any(bool(parameter.grad.abs().sum() > 0) for parameter in model.model.decoder.parameters()
                               if parameter.grad is not None)
    finally:
        base.activate_dataset(previous_dataset)
        torch.set_num_threads(previous_threads)
