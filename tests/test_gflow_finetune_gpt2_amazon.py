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
        # GPT-2 override: 512 candidate sequences per micro-batch (32 GB GPUs).
        assert a.max_candidate_batch == cfg["gpt2_amazon"]["max_candidate_batch"] == 512
        assert a.max_candidate_batch < gflow["max_candidate_batch"]
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


class TestOomFallbackAndControl:
    def _ctx(self):
        return g.DatasetContext("instruments", {"method": {"sasrec": {"checkpoint_path": "x"}}},
                                dm=None, subset_indices=[], sasrec=None, cache_dirs={})

    def test_oom_halves_candidate_cap_until_it_fits(self, tmp_path, monkeypatch):
        args = g.parse_args(["--max-candidate-batch", "4096", "--oom-retries", "3", "--device", "cpu"])
        entry = g.build_grid(g.parse_args(["--datasets", "instruments", "--combos", "sasrec",
                                           "--num-candidates", "10", "--objectives", "tb",
                                           "--powers", "1", "--weight-levels", "1"]))[0]
        caps: list[int] = []

        def fake_run_single(**kw):
            caps.append(int(kw["args"].max_candidate_batch))
            if kw["args"].max_candidate_batch > 512:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 462.00 MiB")
            return {"status": "completed", "attempt": kw["attempt"],
                    "test_batch_size": kw["args"].test_batch_size}

        monkeypatch.setattr(g, "run_single", fake_run_single)
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        result = g.execute_entry(entry, self._ctx(), run_dir, args=args, csv_path=tmp_path / "m.csv")
        assert caps == [4096, 2048, 1024, 512]
        assert result["status"] == "completed" and result["attempt"] == 3
        assert result["test_batch_size"] == 128 // 8  # halved with every retry
        assert not (run_dir / "checkpoints").exists()  # stale attempt checkpoints removed
        assert args.max_candidate_batch == 4096  # the caller's namespace is untouched

    def test_oom_gives_up_after_retries_and_records_failure(self, tmp_path, monkeypatch):
        args = g.parse_args(["--max-candidate-batch", "64", "--oom-retries", "2", "--device", "cpu"])
        entry = g.build_grid(g.parse_args(["--datasets", "instruments", "--combos", "sasrec",
                                           "--num-candidates", "10", "--objectives", "tb",
                                           "--powers", "1", "--weight-levels", "1"]))[0]
        calls = []

        def always_oom(**kw):
            calls.append(int(kw["args"].max_candidate_batch))
            raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(g, "run_single", always_oom)
        run_dir = tmp_path / "run"
        result = g.execute_entry(entry, self._ctx(), run_dir, args=args, csv_path=tmp_path / "m.csv")
        assert calls == [64, 32, 16]  # initial + 2 retries
        assert result["status"] == "failed" and result["oom_retries"] == 2
        assert result["max_candidate_batch"] == 16
        assert json.loads((run_dir / g.RESULT_FILENAME).read_text())["status"] == "failed"
        rows = (tmp_path / "m.csv").read_text().splitlines()
        assert len(rows) == 3 and "failed" in rows[1]

    def test_non_oom_errors_are_not_retried(self, tmp_path, monkeypatch):
        args = g.parse_args(["--oom-retries", "3", "--device", "cpu"])
        entry = g.build_grid(g.parse_args(["--datasets", "beauty", "--combos", "sasrec",
                                           "--num-candidates", "1", "--objectives", "tb",
                                           "--powers", "1", "--weight-levels", "1"]))[0]
        calls = []

        def boom(**kw):
            calls.append(1)
            raise ValueError("bad cache")

        monkeypatch.setattr(g, "run_single", boom)
        result = g.execute_entry(entry, self._ctx(), tmp_path / "run", args=args, csv_path=tmp_path / "m.csv")
        assert len(calls) == 1 and result["status"] == "failed" and "bad cache" in result["error"]

    def test_oom_retry_args_floor(self):
        args = g.parse_args(["--max-candidate-batch", "11", "--device", "cpu"])
        assert g.oom_retry_args(args, num_negatives=10) is None  # 11 = K+1 cannot shrink
        smaller = g.oom_retry_args(g.parse_args(["--max-candidate-batch", "40", "--device", "cpu"]), 10)
        assert smaller.max_candidate_batch == 20

    def test_sft_only_layout_is_disjoint_from_the_grid(self, tmp_path):
        root = tmp_path / "exp"
        control = g.sft_control_run_dir(root, "beauty", 42)
        assert control == root / "beauty" / "sft_only" / "seed_42"
        grid_dirs = {g.run_dir_for(root, e, 42) for e in g.build_grid(g.parse_args([]))}
        assert control not in grid_dirs
        identity = g.sft_control_identity("beauty", 42)
        assert identity["combo"] == "-" and identity["objective"] == "-"
        assert set(g.CSV_KEY) - {"variant", "split"} <= set(identity)
        args = g.parse_args(["--sft-only"])
        assert args.sft_only and args.oom_retries == 3


# --------------------------------------------------------------------------- #
# Full-user-history run (gflow_finetune_gpt2_amazon_fullhist.py)
# --------------------------------------------------------------------------- #
import gflow_finetune_gpt2_amazon_fullhist as fh  # noqa: E402


class _Examples:
    """SequenceDataset stand-in: per_user_tail_indices only needs .examples."""

    def __init__(self, user_ids):
        import pandas as pd

        self.examples = pd.DataFrame({"user_id": user_ids})

    def __len__(self):
        return len(self.examples)


class TestFullHistoryRun:
    def test_full_history_keeps_every_prefix_example(self):
        dataset = _Examples([1] * 12 + [2] * 3 + [1] * 2 + [3] * 9)
        tail = g.finetune_indices(dataset, g.parse_args(["--train-last-n", "8"]))
        assert len(tail) == 8 + 3 + 8  # users 1 and 3 lose their early prefixes
        full = g.finetune_indices(dataset, fh.parse_args([]))
        assert full == list(range(len(dataset)))
        assert set(tail) < set(full)
        assert g.cache_last_n(fh.parse_args([])) is None
        assert g.cache_last_n(g.parse_args([])) == 8

    def test_wrapper_defaults_match_the_tiger_batches_and_stay_overridable(self):
        a = fh.parse_args([])
        assert a.train_last_n == 0
        assert a.batch_size * a.grad_accum == 2048  # TIGER effective train batch
        assert a.val_batch_size == 64 and a.test_batch_size == 64
        import sft_finetune_control as control

        assert a.val_batch_size == control.DEFAULT_VAL_BATCH_SIZE
        assert all("val_batch_size" not in control.DATASETS[d] for d in ("beauty", "instruments"))
        assert a.uniform_micro_batch and a.validate_at_start and a.resume
        # The study-wide grid (finetune_base_matched.json gflow_grid), same as TIGER's.
        assert a.weight_levels == [1.0] and a.num_candidates == [10]
        assert a.similarity_modes == ["hierarchical"] and a.rho_targets == [1.0, 0.1]
        grid = g.build_grid(a)
        assert len(grid) == 2 * 48 and {e["num_negatives"] for e in grid} == {10}
        assert {e["combo_key"] if "combo_key" in e else "_x_".join(e["components"]) for e in grid} == {
            "sasrec", "sasrec_x_interaction_x_sim_hier"}
        # Later flags win, so every default can be overridden from the CLI.
        b = fh.parse_args(["--val-batch-size", "256", "--no-validate-at-start",
                           "--num-candidates", "1", "3", "--no-resume"])
        assert b.val_batch_size == 256 and not b.validate_at_start and not b.resume
        assert b.num_candidates == [1, 3] and b.train_last_n == 0
        rho = g.build_grid(fh.parse_args(["--combos", "sasrec_x_interaction_x_similarity",
                                          "--rho-targets", "0.1"]))
        assert len(rho) == 24 and {e["rho"] for e in rho} == {0.1}

    def test_outputs_are_disjoint_from_the_last8_sweep(self):
        base_args, full_args = g.parse_args([]), fh.parse_args([])
        assert g.abspath(base_args.output_root) != g.abspath(full_args.output_root)
        assert g.metrics_csv_path(base_args).name == g.DEFAULT_METRICS_CSV_NAME
        assert g.metrics_csv_path(full_args) == g.abspath(fh.METRICS_CSV)
        assert g.metrics_csv_path(full_args).parent == g.metrics_csv_path(base_args).parent
        assert g.metrics_csv_path(full_args) != g.metrics_csv_path(base_args)
        for key in g.CONCRETE_COMPONENTS:
            assert (g.component_cache_dir("beauty", key, base_args)
                    != g.component_cache_dir("beauty", key, full_args))

    def test_every_arm_gets_the_same_micro_batch(self):
        a = fh.parse_args([])
        assert g.run_batch_settings(a, 10) == (32, 64)
        assert g.run_batch_settings(a, None) == (32, 64)  # SFT-only control
        assert g.run_batch_settings(a, 1) == (32, 64)     # pinned to the largest selected K
        fh.check_uniform_micro_batch(a)
        with pytest.raises(SystemExit, match="protocol micro-batch"):
            fh.check_uniform_micro_batch(fh.parse_args(["--num-candidates", "100"]))
        legacy = g.parse_args(["--num-candidates", "1", "3"])
        assert g.run_batch_settings(legacy, 3) == (128, 16)
        assert g.run_batch_settings(legacy, 1) == (256, 8)
        assert g.run_batch_settings(legacy, None) == (256, 8)
        for args, ks in ((a, (1, 10, None)), (legacy, (1, 3, None))):
            for k in ks:
                batch, accum = g.run_batch_settings(args, k)
                assert batch * accum == 2048

    def test_batch_sizes_reach_the_datamodule_and_trainer(self, tmp_path, monkeypatch):
        a = fh.parse_args(["--device", "cpu"])
        loader = g.build_run_config("beauty", a, user_ids_count=1)["dataloader"]
        assert loader["validation_batch_size"] == 64
        captured = {}

        def fake_from_optimizer_config(**kw):
            captured.update(kw)
            return "trainer"

        monkeypatch.setattr(g.Trainer, "from_optimizer_config", staticmethod(fake_from_optimizer_config))
        for args, expected in ((a, True), (g.parse_args(["--device", "cpu"]), False)):
            g.build_trainer(None, None, tag="t", run_dir=tmp_path, run_accum=16, args=args,
                            device=torch.device("cpu"))
            assert captured["cfg"].run_validation_at_start is expected
            assert captured["cfg"].gradient_accumulation_steps == 16

    def test_cache_build_covers_every_example(self, tmp_path, monkeypatch):
        commands = []

        def fake_run(cmd, **kw):
            commands.append(list(cmd))
            out = Path(cmd[cmd.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            last_n = int(cmd[cmd.index("--per-user-last-n") + 1]) if "--per-user-last-n" in cmd else None
            (out / "manifest.json").write_text(json.dumps({
                "components": [{"name": "sasrec"}], "num_examples": 100, "per_user_last_n": last_n}))

        monkeypatch.setattr(g.subprocess, "run", fake_run)
        for parse, root, expected in ((fh.parse_args, "full", None), (g.parse_args, "tail", "8")):
            args = parse(["--cache-root", str(tmp_path / root), "--device", "cpu"])
            run_config = g.build_run_config("beauty", args, user_ids_count=1)
            g.ensure_component_cache("beauty", "sasrec", run_config, tmp_path / root / "cfg", 100, args)
            cmd = commands[-1]
            if expected is None:
                assert "--per-user-last-n" not in cmd
            else:
                assert cmd[cmd.index("--per-user-last-n") + 1] == expected

    def test_a_cache_of_another_subset_is_refused(self, tmp_path):
        spec = {"name": "sasrec"}

        def manifest(last_n):
            return {"components": [{"name": "sasrec"}], "num_examples": 100, "per_user_last_n": last_n}

        full, tail = fh.parse_args([]), g.parse_args([])
        assert g._check_component_manifest(manifest(None), tmp_path, "sasrec", spec, 100, full)
        assert g._check_component_manifest(manifest(8), tmp_path, "sasrec", spec, 100, tail)
        with pytest.raises(RuntimeError, match="last 8 examples.*full user history|--train-last-n 0"):
            g._check_component_manifest(manifest(8), tmp_path, "sasrec", spec, 100, full)
        with pytest.raises(RuntimeError, match="full user history"):
            g._check_component_manifest(manifest(None), tmp_path, "sasrec", spec, 100, tail)
        with pytest.raises(RuntimeError, match="last 4 examples"):
            g._check_component_manifest(manifest(4), tmp_path, "sasrec", spec, 100, tail)


class _FakeTrainer:
    def __init__(self, tmp_path, *, epoch, should_stop=False, max_epochs=500):
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(max_epochs=max_epochs)
        self.state = SimpleNamespace(epoch=0, best_metric=None, best_epoch=None, bad_epochs=0,
                                     should_stop=False, history=[])
        self._saved = SimpleNamespace(epoch=epoch, best_metric=0.04, best_epoch=max(epoch - 5, -1),
                                      bad_epochs=1, should_stop=should_stop, history=[{"epoch": 0}])
        self.loaded: list[tuple[str, bool]] = []
        self.fit_calls = 0

    def load_checkpoint(self, path, *, load_optimizer=True, load_scheduler=True):
        self.loaded.append((Path(path).name, load_optimizer))
        if load_optimizer:
            self.state = self._saved

    def fit(self, train_loader, val_loader):
        from types import SimpleNamespace

        self.fit_calls += 1
        self.started_at = self.state.epoch
        self.state.epoch = self.state.epoch + 4
        return SimpleNamespace(state=self.state, best_checkpoint_path=None,
                               last_checkpoint_path="last.pt")


class TestResume:
    def _interrupted_run(self, tmp_path, args, *, oom_retries=0, cap=512, examples=1000):
        entry = g.build_grid(args)[0]
        identity = g.entry_identity(entry, int(args.seed))
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        torch.save({"trainer_state": {"epoch": 3}}, run_dir / "checkpoints" / "last.pt")
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"x")
        (run_dir / "resolved_config.json").write_text(json.dumps({
            **identity, "variant": "gflow",
            "batching": {"max_candidate_batch": cap, "oom_retries": oom_retries},
            "data": {"num_finetune_examples": examples}}))
        return entry, identity, run_dir

    def test_only_an_interrupted_attempt_of_the_same_run_is_resumed(self, tmp_path):
        args = fh.parse_args(["--device", "cpu"])
        _, identity, run_dir = self._interrupted_run(tmp_path, args)
        assert g.resumable_checkpoint(run_dir, identity, 1000, args) == run_dir / "checkpoints" / "last.pt"
        assert g.resumable_checkpoint(run_dir, identity, 999, args) is None  # another subset
        assert g.resumable_checkpoint(run_dir, {**identity, "power": 123.0}, 1000, args) is None
        assert g.resumable_checkpoint(run_dir, identity, 1000, fh.parse_args(["--no-resume"])) is None
        assert g.resumable_checkpoint(run_dir, identity, 1000, fh.parse_args(["--force"])) is None
        assert g.resumable_checkpoint(run_dir, identity, 1000, g.parse_args([])) is None  # off by default
        # torch.save is not atomic: a truncated last.pt means "start over", not "fail".
        (run_dir / "checkpoints" / "last.pt").write_bytes(b"truncated")
        assert g.resumable_checkpoint(run_dir, identity, 1000, args) is None
        assert g.interrupted_attempt_args(run_dir, identity, 1000, args) == (0, args)
        (run_dir / "checkpoints" / "last.pt").unlink()
        assert g.resumable_checkpoint(run_dir, identity, 1000, args) is None

    def test_resume_keeps_the_cap_the_interrupted_attempt_had_shrunk_to(self, tmp_path, monkeypatch):
        args = fh.parse_args(["--device", "cpu"])
        entry, identity, run_dir = self._interrupted_run(tmp_path, args, oom_retries=1, cap=256)
        attempt, resumed = g.interrupted_attempt_args(run_dir, identity, 1000, args)
        assert attempt == 1 and resumed.max_candidate_batch == 256 and resumed.test_batch_size == 32
        assert args.max_candidate_batch == 512
        seen = {}

        def fake_run_single(**kw):
            seen.update(cap=kw["args"].max_candidate_batch, attempt=kw["attempt"],
                        resume_attempt=kw["resume_attempt"])
            return {"status": "completed"}

        monkeypatch.setattr(g, "run_single", fake_run_single)
        ctx = g.DatasetContext("beauty", {}, dm=None, subset_indices=list(range(1000)),
                               sasrec=None, cache_dirs={})
        g.execute_entry(entry, ctx, run_dir, args=args, csv_path=tmp_path / "m.csv")
        assert seen == {"cap": 256, "attempt": 1, "resume_attempt": 1}
        assert (run_dir / "checkpoints" / "last.pt").exists()

    def test_resume_continues_with_the_next_epoch(self, tmp_path, monkeypatch):
        trainer = _FakeTrainer(tmp_path, epoch=37)
        assert g.resume_trainer(trainer, tmp_path / "last.pt", tag="t") == 37
        assert trainer.state.epoch == 38 and trainer.state.bad_epochs == 1

        monkeypatch.setattr(g.gflow_ablation, "indexed_train_loader", lambda *a, **k: "loader")
        monkeypatch.setattr(g.gpt2_beauty, "evaluate_splits", lambda *a, **k: {"test": {}})
        from types import SimpleNamespace

        dm = SimpleNamespace(val_dataloader=lambda: "val")
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"x")
        args = fh.parse_args(["--device", "cpu"])
        fit, _ = g.fit_and_evaluate(trainer, None, dm, [0, 1], run_batch=2, run_dir=run_dir, args=args,
                                    device=torch.device("cpu"), resumed_from_epoch=37)
        assert trainer.fit_calls == 1 and trainer.started_at == 38
        # No improvement after the resume: the pre-interruption best.pt is evaluated.
        assert fit.best_checkpoint_path == str(run_dir / "checkpoints" / "best.pt")
        assert trainer.loaded[-1] == ("best.pt", False)

    def test_a_run_killed_during_the_final_evaluation_is_not_trained_again(self, tmp_path, monkeypatch):
        trainer = _FakeTrainer(tmp_path, epoch=60, should_stop=True)
        finished = g.resume_trainer(trainer, tmp_path / "last.pt", tag="t")
        monkeypatch.setattr(g.gpt2_beauty, "evaluate_splits", lambda *a, **k: {"test": {}})
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"x")
        fit, _ = g.fit_and_evaluate(trainer, None, None, [0], run_batch=2, run_dir=run_dir,
                                    args=fh.parse_args(["--device", "cpu"]),
                                    device=torch.device("cpu"), resumed_from_epoch=finished)
        assert trainer.fit_calls == 0 and fit.state.epoch == 60  # epochs_completed stays 61
        assert trainer.loaded[-1] == ("best.pt", False)


def test_full_history_defaults_agree_with_the_protocol_file():
    cfg = json.loads(g.PROTOCOL_CONFIG.read_text())
    block = cfg["gpt2_amazon_fullhist"]
    a = fh.parse_args([])
    assert a.train_last_n == block["train_last_n"] == 0
    assert (a.batch_size, a.grad_accum) == (block["batch_size"], block["grad_accum"])
    assert a.uniform_micro_batch is block["uniform_micro_batch"] is True
    assert a.val_batch_size == block["val_batch_size"]
    assert a.test_batch_size == block["test_batch_size"]
    assert a.validate_at_start is block["run_validation_at_start"]
    assert block["run_validation_at_start"] == cfg["shared_with_sft_control"]["run_validation_at_start"]
    assert "scripts/exp4-reward-ablations/gflow_finetune_gpt2_amazon_fullhist.py" in cfg["consumers"]
    # The base script ignores the block: its last-8 sweep stays reproducible.
    b = g.parse_args([])
    assert b.train_last_n == 8 and b.val_batch_size == 256 and not b.validate_at_start
