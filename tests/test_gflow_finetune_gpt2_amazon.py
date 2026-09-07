"""exp4 GPT-2 Amazon GFlow ablation: composed reward cache, grid, protocol.

The composed cache (weights / power / composition at lookup time over raw
per-component caches) must reproduce the MultiplicativeRewardModel /
MixedRewardModel semantics the exp4 TIGER scripts bake into their caches at
build time, including the exact full-catalog logZ and the target-row
substitution for negative candidate rows.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_gpt2_amazon as g  # noqa: E402

EPS = 1e-6


def _write_component_cache(root: Path, name: str, rewards: torch.Tensor,
                           target_rows: torch.Tensor, *, shard_size: int = 2,
                           store_power_sums: tuple[float, ...] = ()) -> Path:
    """Shard layout of scripts/precompute_reward_cache.py for one raw component
    (power sums only matter for the base-class comparison test)."""
    cache_dir = root / name
    cache_dir.mkdir(parents=True)
    shards = []
    for shard_idx, start in enumerate(range(0, rewards.size(0), shard_size)):
        block = rewards[start:start + shard_size]
        stored = block.to(torch.float16).float().clamp_min(EPS)
        payload = {
            "example_start": start,
            "num_examples": int(block.size(0)),
            "target_semantic_row_ids": target_rows[start:start + shard_size],
            "full_rewards": block.to(torch.float16),
            "reward_sum": stored.sum(dim=1, keepdim=True),
        }
        for power in store_power_sums:
            slug = format(float(power), ".12g").replace("-", "m").replace(".", "p")
            payload[f"reward_sum_power_{slug}"] = stored.pow(power).clamp_min(EPS).sum(dim=1, keepdim=True)
        torch.save(payload, cache_dir / f"shard_{shard_idx:05d}.pt")
        shards.append({"path": f"shard_{shard_idx:05d}.pt", "example_start": start,
                       "num_examples": int(block.size(0))})
    (cache_dir / "manifest.json").write_text(json.dumps({
        "num_examples": int(rewards.size(0)),
        "per_user_last_n": 8,
        "stored_power_sums": list(store_power_sums),
        "components": [{"name": name, "weight": 1.0, "normalizer": "none"}],
        "shards": shards,
    }))
    return cache_dir


@pytest.fixture()
def component_caches(tmp_path):
    torch.manual_seed(0)
    num_examples, num_items = 5, 7
    # float16-exact values so the reference computation is not dtype-limited
    rewards = {
        "sasrec": (torch.randint(1, 16, (num_examples, num_items)).float() / 16),
        "interaction": torch.where(torch.rand(num_examples, num_items) < 0.3, 2.0, 1.0),
        "similarity": (torch.randint(0, 4, (num_examples, num_items)).float() / 4).clamp_min(0.25),
    }
    target_rows = torch.tensor([3, 0, 6, 1, 4])
    dirs = {name: _write_component_cache(tmp_path, name, r, target_rows, store_power_sums=(3.0,))
            for name, r in rewards.items()}
    # The reference sees exactly what the caches store (float16 -> float32, eps floor).
    stored = {name: r.to(torch.float16).float().clamp_min(EPS) for name, r in rewards.items()}
    return dirs, stored, target_rows


def _reference(stored, names, weights, composition, power):
    full = None
    for name, w in zip(names, weights):
        r = stored[name]
        if composition == "multiply":
            term = r.pow(w)
            full = term if full is None else full * term
        else:
            term = (r / r.amax(dim=1, keepdim=True)).clamp_min(EPS) * w
            full = term if full is None else full + term
    return full.clamp_min(EPS).pow(power).clamp_min(EPS)


class TestComposedRewardCache:
    @pytest.mark.parametrize("composition", ["multiply", "mixed"])
    @pytest.mark.parametrize("power", [0.25, 1.0, 4.0])
    def test_matches_reference_composition(self, component_caches, composition, power):
        dirs, stored, target_rows = component_caches
        names = ("sasrec", "interaction", "similarity")
        weights = (1.0, 2.0, 1.0)
        cache = g.ComposedRewardCache(
            [(n, dirs[n], w) for n, w in zip(names, weights)],
            composition=composition, power=power,
        )
        indices = torch.tensor([4, 1, 2])
        rows = torch.tensor([[-1, 0, 5], [-1, 6, 6], [-1, 2, 3]])
        selected, reward_sum, full = cache.select_candidates(
            indices, rows, device=torch.device("cpu"), include_full=True)

        expected_full = _reference(stored, names, weights, composition, power)[indices]
        resolved_rows = torch.where(rows.ge(0), rows, target_rows[indices].unsqueeze(1))
        expected_selected = expected_full.gather(1, resolved_rows)

        assert torch.allclose(full, expected_full, rtol=1e-5, atol=1e-7)
        assert torch.allclose(selected, expected_selected, rtol=1e-5, atol=1e-7)
        assert torch.allclose(reward_sum, expected_full.sum(dim=1), rtol=1e-5, atol=1e-6)
        assert cache.num_examples == 5 and cache.num_items == 7

    def test_single_component_power_equals_base_cache_power(self, component_caches):
        """One component, weight 1: identical to the base cache with the same power."""
        from flowrec.ft_methods.gflow.method import _PrecomputedRewardCache

        dirs, _, _ = component_caches
        composed = g.ComposedRewardCache([("sasrec", dirs["sasrec"], 1.0)],
                                         composition="multiply", power=3.0)
        rows = torch.tensor([[-1, 1], [2, 3]])
        idx = torch.tensor([0, 3])
        sel_c, _, full_c = composed.select_candidates(idx, rows, torch.device("cpu"), include_full=True)
        base = _PrecomputedRewardCache(dirs["sasrec"], power=3.0)
        sel_b, _, full_b = base.select_candidates(idx, rows, torch.device("cpu"), include_full=True)
        assert torch.allclose(sel_c, sel_b) and torch.allclose(full_c, full_b)

    def test_weight_is_an_exponent_in_multiply_mode(self, component_caches):
        dirs, stored, _ = component_caches
        cache = g.ComposedRewardCache([("sasrec", dirs["sasrec"], 2.0)],
                                      composition="multiply", power=1.0)
        _, _, full = cache.select_candidates(torch.tensor([0]), torch.tensor([[0]]),
                                             torch.device("cpu"), include_full=True)
        assert torch.allclose(full[0], stored["sasrec"][0].square())

    def test_include_full_false_returns_no_matrix(self, component_caches):
        dirs, _, _ = component_caches
        cache = g.ComposedRewardCache([("sasrec", dirs["sasrec"], 1.0),
                                       ("similarity", dirs["similarity"], 1.0)],
                                      composition="multiply", power=1.0)
        _, reward_sum, full = cache.select_candidates(
            torch.tensor([1]), torch.tensor([[-1, 2]]), torch.device("cpu"))
        assert full is None and reward_sum.shape == (1,)

    def test_rejects_mismatched_caches_and_bad_params(self, component_caches, tmp_path):
        dirs, _, _ = component_caches
        short = _write_component_cache(tmp_path / "other", "sasrec", torch.ones(3, 7), torch.zeros(3).long())
        with pytest.raises(ValueError, match="num_examples"):
            g.ComposedRewardCache([("sasrec", short, 1.0), ("similarity", dirs["similarity"], 1.0)],
                                  composition="multiply", power=1.0)
        with pytest.raises(ValueError):
            g.ComposedRewardCache([("sasrec", dirs["sasrec"], 1.0)], composition="sum", power=1.0)
        with pytest.raises(ValueError):
            g.ComposedRewardCache([("sasrec", dirs["sasrec"], 0.0)], composition="multiply", power=1.0)
        with pytest.raises(ValueError):
            g.ComposedRewardCache([("sasrec", dirs["sasrec"], 1.0)], composition="multiply", power=0.0)


class TestGrid:
    def test_weight_assignments_enumerate_ratios_only(self):
        assert g.weight_assignments(1, (1, 2)) == [(1.0,)]
        assert g.weight_assignments(2, (1, 2)) == [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
        assert len(g.weight_assignments(3, (1, 2))) == 7
        assert all(min(a) == 1.0 for a in g.weight_assignments(3, (1, 2)))
        assert g.weight_assignments(3, (1,)) == [(1.0, 1.0, 1.0)]
        assert len(g.weight_assignments(2, (1, 2, 4))) == 5

    def test_default_grid_size_and_order(self):
        args = g.parse_args([])
        grid = g.build_grid(args)
        # 33 weighted combos (5 similarity-free + 2 modes x 14) x 4 powers
        # x 3 objectives x 4 K per dataset, dataset-major.
        per_dataset = 33 * 4 * 3 * 4
        assert len(grid) == 2 * per_dataset == 3168
        assert [e["dataset"] for e in grid[:per_dataset]] == ["beauty"] * per_dataset
        assert grid[per_dataset]["dataset"] == "instruments"
        first = grid[0]
        assert first["combo"] == ("sasrec",) and first["components"] == ("sasrec",)
        assert first["similarity_mode"] is None and first["weights"] == (1.0,)
        assert first["power"] == 0.25 and first["num_negatives"] == 1
        assert first["objective"] == "tb" and first["composition"] == "multiply"
        # Consecutive entries vary K fastest, then the objective.
        assert [e["num_negatives"] for e in grid[:4]] == [1, 3, 10, 100]
        assert [e["objective"] for e in grid[:12:4]] == ["tb", "db", "subtb"]
        # Combos with `similarity` exist once per mode, with concrete keys.
        sim_entries = [e for e in grid if e["combo"] == ("sasrec", "similarity")]
        assert {e["components"] for e in sim_entries} == {("sasrec", "sim_hier"), ("sasrec", "sim_embed")}
        assert {e["similarity_mode"] for e in sim_entries} == {"hierarchical", "embedding_cosine"}
        # Every run dir / tag is unique.
        dirs = {g.run_dir_for(Path("x"), e, 42) for e in grid}
        assert len(dirs) == len(grid)

    def test_run_dir_layout(self):
        args = g.parse_args(["--composition", "mixed", "--rho-targets", "0.1"])
        entry = g.build_grid(args)[0]
        entry = {**entry, "combo": ("sasrec", "similarity"), "components": ("sasrec", "sim_embed"),
                 "similarity_mode": "embedding_cosine", "weights": (1.0, 2.0), "power": 4.0,
                 "num_negatives": 10, "objective": "subtb"}
        path = g.run_dir_for(Path("root"), entry, 42)
        assert path == Path("root/beauty/mixed_sasrec_x_sim_embed/w1-2/pow_4/subtb_k10_rho0p1/seed_42")
        plain = {**entry, "composition": "multiply", "rho": 1.0}
        assert g.run_dir_for(Path("root"), plain, 42) == Path(
            "root/beauty/sasrec_x_sim_embed/w1-2/pow_4/subtb_k10/seed_42")
        identity = g.entry_identity(plain, 42)
        assert identity["combo"] == "sasrec_x_sim_embed"
        assert identity["similarity_mode"] == "embedding_cosine"

    def test_filters_and_components(self):
        args = g.parse_args(["--datasets", "instruments", "--combos", "sasrec_x_interaction",
                             "--powers", "1", "--num-candidates", "3", "--weight-levels", "1",
                             "--objectives", "tb"])
        grid = g.build_grid(args)
        assert len(grid) == 1 and grid[0]["weights"] == (1.0, 1.0)
        assert g.components_for(grid) == ["sasrec", "interaction"]
        args = g.parse_args(["--combos", "similarity", "--similarity-modes", "exact",
                             "--objectives", "db", "--powers", "1", "--num-candidates", "1"])
        grid = g.build_grid(args)
        assert len(grid) == 2 and g.components_for(grid) == ["sim_exact"]
        assert g.component_keys_for(args) == ["sasrec", "interaction", "sim_exact"]
        with pytest.raises(SystemExit):
            g.parse_args(["--weight-levels", "0"])
        with pytest.raises(SystemExit):
            g.parse_args(["--similarity-modes", "cosine"])


class TestProtocol:
    def test_defaults_come_from_finetune_gflow_json(self):
        cfg = json.loads(g.PROTOCOL_CONFIG.read_text())
        shared, gflow = cfg["shared_with_sft_control"], cfg["gflow"]
        a = g.parse_args([])
        assert a.lr == shared["lr"] and a.weight_decay == shared["weight_decay"]
        assert a.batch_size == shared["batch_size"] and a.grad_accum == shared["grad_accum"]
        assert a.max_epochs == shared["max_epochs"] and a.patience == shared["patience"]
        assert a.validate_every == shared["validate_every"]
        assert a.max_grad_norm == shared["max_grad_norm"] and a.amp_dtype == shared["amp_dtype"]
        assert a.eval_num_beams == shared["eval_num_beams"] and a.test_ks == shared["test_ks"]
        assert a.gflow_weight_mode == gflow["gflow_weight_mode"]
        assert a.gflow_target_grad_ratio == gflow["gflow_target_grad_ratio"]
        assert a.curriculum_steps == gflow["curriculum_steps"]
        assert a.train_last_n == gflow["train_last_n"]
        assert a.max_candidate_batch == gflow["max_candidate_batch"]
        assert "scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon.py" in cfg["consumers"]

    def test_cache_config_isolates_one_component(self):
        args = g.parse_args([])
        run_config = g.build_run_config("beauty", args, user_ids_count=1)
        assert run_config["method"]["reward_model"] == "multiply"
        assert run_config["method"]["component_keys"] == ["sasrec", "interaction", "sim_hier", "sim_embed"]
        assert [c["name"] for c in run_config["method"]["reward"]["components"]] == [
            "sasrec", "interaction", "similarity", "similarity"]
        hier = g.cache_config(run_config, "sim_hier")
        assert hier["method"]["reward_model"] == "similarity"
        assert hier["method"]["reward"]["mode"] == "hierarchical"
        assert hier["method"]["reward"]["level_stride"] == g.LEVEL_STRIDE
        assert hier["method"]["reward"]["rqvae_model_path"].endswith("exp1-standard_params/beauty/model.pt")
        embed = g.cache_config(run_config, "sim_embed")
        assert embed["method"]["reward_model"] == "similarity"
        assert embed["method"]["reward"] == {
            "mode": "embedding_cosine",
            "embeddings_path": str(g.EMBEDDINGS_ROOT / "beauty_content_embeddings.pt"),
        }
        assert g.cache_config(run_config, "interaction")["method"]["reward_model"] == "interaction"
        assert embed["data"] == run_config["data"]
        assert run_config["dataset"]["train_example_layout"] == "prefix_target"
        assert run_config["dataset"]["positive_criterion"] == "verified_purchase == True"

    def test_embedding_cosine_cache_fast_path_matches_model(self):
        """precompute_reward_cache's full-catalog embedding_cosine path must
        equal SimilarityRewardModel scoring every catalog row."""
        from precompute_reward_cache import compute_component_rewards
        from flowrec.rewards.similarity import SimilarityRewardConfig, SimilarityRewardModel
        from flowrec.trajectories.semantic import SemanticTrajectoryBatch

        torch.manual_seed(1)
        item_vocab_size, num_rows, dim, sid_len = 9, 7, 5, 3
        table = torch.randn(item_vocab_size, dim)
        table[0] = 0.0  # pad item
        # semantic row -> internal item id (rows cover items 1..7 in shuffled order)
        row_to_item = torch.tensor([3, 1, 7, 2, 6, 4, 5])
        item_to_row = torch.full((item_vocab_size,), -1, dtype=torch.long)
        item_to_row[row_to_item] = torch.arange(num_rows)
        model = SimilarityRewardModel(
            SimilarityRewardConfig(mode="embedding_cosine", sharpness=2.0),
            item_embeddings=table, item_id_by_semantic_row=row_to_item,
        ).eval()
        item_tokens = torch.randint(1, 20, (num_rows, sid_len))
        batch = {"target_item_ids": torch.tensor([3, 5, 7]), "target_token_ids": item_tokens[[0, 6, 2]]}
        fast = compute_component_rewards(
            {"model": model}, batch, item_tokens, row_to_item, item_to_row, 1e-6)

        bsz = batch["target_item_ids"].numel()
        trajectories = SemanticTrajectoryBatch(
            candidate_token_ids=item_tokens.unsqueeze(0).expand(bsz, -1, -1),
            candidate_token_mask=torch.ones(bsz, num_rows, sid_len, dtype=torch.bool),
            is_positive=torch.zeros(bsz, num_rows, dtype=torch.bool),
            candidate_item_row_ids=torch.arange(num_rows).unsqueeze(0).expand(bsz, -1),
        )
        reference = model(batch, trajectories)
        assert fast.shape == (bsz, num_rows)
        assert torch.allclose(fast, reference, atol=1e-6)

    def test_tuned_profile_reads_sidecar_and_dropout(self):
        for dataset in g.DATASETS:
            paths = g.tuned_checkpoint_paths(dataset)
            if not paths["sidecar"].exists():
                pytest.skip("tuned GPT-2 checkpoints not present")
            profile, sidecar = g.tuned_profile(dataset)
            model = profile["model"]
            assert model["embedding_dim"] == sidecar["embedding_dim"]
            assert model["num_layers"] == sidecar["num_layers"]
            assert model["dim_feedforward"] == sidecar["dim_feedforward"]
            assert model["max_position_embeddings"] == sidecar["max_position_embeddings"]
            assert 0.0 < model["dropout"] < 1.0
            assert model["attention_dropout"] == model["dropout"]


class TestCooperativeWorkers:
    def _entries(self):
        args = g.parse_args(["--datasets", "instruments", "--combos", "sasrec", "--powers", "1",
                             "--num-candidates", "1", "3", "--objectives", "tb", "db"])
        return args, g.build_grid(args)

    def test_claim_lifecycle(self, tmp_path):
        args, grid = self._entries()
        root = tmp_path / "exp"
        run_dir = g.run_dir_for(root, grid[0], 42)
        store = g.ClaimStore(root, worker_id=0, stale_claim_hours=1.0)
        other = g.ClaimStore(root, worker_id=1, stale_claim_hours=1.0)
        assert store.status_of(run_dir) == "pending"
        assert store.try_claim(run_dir, retry_failed=False)
        assert store.status_of(run_dir) == "running"
        assert not other.try_claim(run_dir, retry_failed=False)  # live claim
        assert json.loads(store.claim_path(run_dir).read_text())["worker_id"] == 0
        store.release(run_dir)
        assert store.status_of(run_dir) == "pending"
        # completed runs are never claimed
        g.write_json(run_dir / g.RESULT_FILENAME, {"status": "completed"})
        assert store.status_of(run_dir) == "completed"
        assert not other.try_claim(run_dir, retry_failed=True)
        # failed runs only with retry_failed
        g.write_json(run_dir / g.RESULT_FILENAME, {"status": "failed"})
        assert store.status_of(run_dir) == "failed"
        assert not other.try_claim(run_dir, retry_failed=False)
        assert other.try_claim(run_dir, retry_failed=True)

    def test_stale_claims_are_reclaimable_and_resettable(self, tmp_path):
        import os
        import time

        args, grid = self._entries()
        root = tmp_path / "exp"
        dirs = [g.run_dir_for(root, e, 42) for e in grid]
        store = g.ClaimStore(root, worker_id=0, stale_claim_hours=1.0)
        assert store.try_claim(dirs[0], retry_failed=False)
        assert store.try_claim(dirs[1], retry_failed=False)
        g.write_json(dirs[1] / g.RESULT_FILENAME, {"status": "completed"})
        old = time.time() - 2 * 3600
        os.utime(store.claim_path(dirs[0]), (old, old))
        assert store.status_of(dirs[0]) == "stale-claim"
        other = g.ClaimStore(root, worker_id=1, stale_claim_hours=1.0)
        assert other.try_claim(dirs[0], retry_failed=False)  # dead worker's claim re-claimed
        other.heartbeat(dirs[0], min_interval=0.0)
        assert store.status_of(dirs[0]) == "running"
        # reset releases only claims of non-completed runs
        assert store.reset_claims(dirs) == 1
        assert store.status_of(dirs[0]) == "pending"
        assert store.status_of(dirs[1]) == "completed"

    def test_status_counts_and_rotation(self, tmp_path, capsys):
        args, grid = self._entries()
        args.output_root = tmp_path / "exp"
        root = args.output_root
        g.write_json(g.run_dir_for(root, grid[0], 42) / g.RESULT_FILENAME, {"status": "completed"})
        g.write_json(g.run_dir_for(root, grid[1], 42) / g.RESULT_FILENAME, {"status": "failed"})
        g.ClaimStore(root, worker_id=0, stale_claim_hours=1.0).try_claim(
            g.run_dir_for(root, grid[2], 42), retry_failed=False)
        counts = g.print_status(grid, args)
        assert counts["instruments"] == {"completed": 1, "failed": 1, "running": 1,
                                         "stale-claim": 0, "pending": 1}
        assert "claimable=1" in capsys.readouterr().out
        assert g.rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1] and g.rotate([], 3) == []
        assert g.CACHE_SLOTS[0] == ("beauty", "sasrec") and g.CACHE_SLOTS[7] == ("instruments", "sim_embed")
        with pytest.raises(SystemExit):
            g.parse_args(["--worker-id", "0", "--index", "1"])
