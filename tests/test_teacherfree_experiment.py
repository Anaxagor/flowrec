"""Teacher-free GFlow experiment (finetune_base_matched.json
`teacher_free_experiment`): random / policy negatives, interaction +
hierarchical + embedding-cosine rewards, no SASRec teacher anywhere; every
other parameter is the base-matched protocol's, for TIGER and GPT-2 alike."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts/exp4-reward-ablations", "scripts/exp2-tuned-params", "scripts"):
    path = str(PROJECT_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import gflow_finetune_gpt2_amazon_fullhist as gpt2_matched  # noqa: E402
import gflow_finetune_gpt2_amazon_teacherfree as gpt2_free  # noqa: E402
import gflow_finetune_tiger_matched as tiger_matched  # noqa: E402
import gflow_finetune_tiger_teacherfree as tiger_free  # noqa: E402

gflow, gpt2 = tiger_free.base, gpt2_free.base
MATCHED_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"
SUBMIT_SCRIPT = PROJECT_ROOT / "scripts/scheduler/submit_base_matched_recalc.sh"
TIGER_DATASETS = ("beauty", "instruments", "beauty_2023", "yambda_50m_listens", "yambda_500m_listens")
# argparse destinations that define data, batch and schedule: identical to the base-matched run
PROTOCOL_ARGS = ("train_last_n", "train_random_n", "train_subset_seed", "batch_size", "grad_accum",
                 "val_batch_size", "validate_every", "patience", "min_epochs", "curriculum_epochs",
                 "val_sample_fraction", "val_sample_resample", "validate_at_start", "seed", "lr",
                 "weight_decay", "max_epochs", "max_grad_norm", "amp_dtype", "num_candidates",
                 "powers", "objectives", "rho_targets", "cache_root", "resume")


@pytest.fixture(scope="module")
def block() -> dict:
    return json.loads(MATCHED_CONFIG.read_text(encoding="utf-8"))["teacher_free_experiment"]


class TestProtocolBlock:
    def test_no_teacher_anywhere(self, block) -> None:
        assert block["samplers"] == ["random", "policy"]
        assert all("sasrec" not in combo.split("_x_") for combo in block["combos"])

    def test_combos_are_every_product_of_the_three_rewards(self, block) -> None:
        import itertools

        parts = ("interaction", "similarity", "sim_embed")
        expected = {"_x_".join(c) for n in (1, 2, 3) for c in itertools.combinations(parts, n)}
        assert set(block["combos"]) == expected and len(block["combos"]) == 7
        # canonical TIGER order, which the GPT-2 wrapper follows as listed: same cell at the same index
        canonical = [gflow.combo_key(c) for c in gflow.all_combos()]
        assert block["combos"] == sorted(block["combos"], key=canonical.index)

    def test_the_rest_of_the_grid_is_the_base_matched_one(self, block) -> None:
        grid = json.loads(MATCHED_CONFIG.read_text(encoding="utf-8"))["gflow_grid"]
        for key in ("powers", "objectives", "rho_targets", "num_negatives", "component_weights"):
            assert block[key] == grid[key], key
        assert block["runs_per_dataset"] == 2 * 7 * 4 * 2 * 3 == 336


class TestTigerWrapper:
    @pytest.mark.parametrize("dataset", TIGER_DATASETS)
    def test_only_sampler_rewards_and_outputs_differ_from_the_matched_run(self, dataset, block) -> None:
        free = tiger_free.parse_args(["--dataset", dataset])
        matched = tiger_matched.parse_args(["--dataset", dataset])
        for key in PROTOCOL_ARGS:
            assert getattr(free, key) == getattr(matched, key), key
        assert free.samplers == ["random", "policy"] and matched.samplers == ["teacher"]
        assert str(free.output_root) == block["tiger"]["output_roots"][dataset] != str(matched.output_root)
        assert gflow.metrics_csv_path(free) != gflow.metrics_csv_path(matched)

    def test_yambda_500m_keeps_the_15_random_targets(self) -> None:
        free = tiger_free.parse_args(["--dataset", "yambda_500m_listens"])
        assert gflow.subset_spec(free) == {"per_user_last_n": None, "per_user_random_n": 15,
                                           "per_user_random_seed": 42}
        assert "r15" in str(free.output_root) and "r15" in str(free.cache_root)
        assert gflow.subset_spec(tiger_free.parse_args(["--dataset", "yambda_50m_listens"]))["per_user_random_n"] is None

    def test_grid_is_sampler_major_and_needs_no_teacher(self, block) -> None:
        args = tiger_free.parse_args(["--dataset", "beauty"])
        grid = gflow.build_grid(args)
        assert len(grid) == block["runs_per_dataset"]
        assert [e["sampler"] for e in grid[:168]] == ["random"] * 168
        assert [e["sampler"] for e in grid[168:]] == ["policy"] * 168
        assert not gflow.needs_teacher(grid)
        assert gflow.components_for([tuple(e["combo"]) for e in grid]) == block["tiger"]["cache_components"]
        tags = {gflow.entry_tag(e) for e in grid}
        dirs = {gflow.run_dir_for(Path("/x"), e, 42) for e in grid}
        assert len(tags) == len(dirs) == len(grid)
        assert all(tag.endswith(("_neg-random", "_neg-policy")) for tag in tags)

    def test_matched_grid_is_untouched_by_the_new_dimensions(self) -> None:
        """Queued base-matched tasks address cells by index and run dir."""
        args = tiger_matched.parse_args(["--dataset", "beauty"])
        grid = gflow.build_grid(args)
        assert len(grid) == 168 and {e["sampler"] for e in grid} == {"teacher"}
        assert gflow.entry_tag(grid[0]) == "sasrec/p0p25/tb_k10"
        assert gflow.run_dir_for(Path("/x"), grid[0], 42) == Path("/x/sasrec/pow_0p25/tb_k10/seed_42")
        assert gflow.entry_tag(grid[-1]) == "sasrec_x_interaction_x_similarity/p16/subtb_k10_rho0p1"
        assert gflow.needs_teacher(grid)
        assert [gflow.combo_key(c) for c in gflow.all_combos()[:7]] == [
            gflow.combo_key(c) for c in gflow.base.all_combos()]

    def test_teacher_runs_are_refused(self) -> None:
        with pytest.raises(SystemExit, match="teacher-free"):
            tiger_free.main(["--dataset", "beauty", "--samplers", "teacher", "--list"])
        with pytest.raises(SystemExit, match="teacher-free"):
            tiger_free.main(["--dataset", "beauty", "--combos", "sasrec", "--list"])

    def test_sim_embed_cache_is_the_similarity_model_in_embedding_mode(self, tmp_path) -> None:
        args = tiger_free.parse_args(["--dataset", "beauty"])
        spec = gflow.component_spec("sim_embed", args)
        assert spec["name"] == "similarity" and spec["reward"]["mode"] == "embedding_cosine"
        assert spec["reward"]["embeddings_path"].endswith("datasets/embeddings/beauty_content_embeddings.pt")
        config = gflow.cache_config(gflow.build_run_config(args), "sim_embed", args)
        assert config["method"]["reward_model"] == "similarity"
        manifest = {"kind": gflow.CACHE_KIND, "component": {"name": "similarity", "mode": "embedding_cosine"},
                    "num_examples": 10, "per_user_last_n": None}
        assert gflow.check_component_manifest(manifest, tmp_path, "sim_embed", 10, args)
        with pytest.raises(RuntimeError, match="embedding_cosine"):   # a hierarchical cache in the sim_embed slot
            gflow.check_component_manifest({**manifest, "component": {"name": "similarity", "mode": "hierarchical"}},
                                           tmp_path, "sim_embed", 10, args)
        yambda = tiger_free.parse_args(["--dataset", "yambda_500m_listens"])
        assert gflow.component_spec("sim_embed", yambda)["reward"]["embeddings_path"].endswith("embeddings.parquet")


class TestSamplers:
    class _Model(torch.nn.Module):
        item_semantic_ids = torch.tensor([[1, 3], [1, 4], [2, 3]])

    def _build(self, name: str, steps: int = 7):
        args = tiger_free.parse_args(["--dataset", "beauty"])
        return gflow.build_sampler(name, model=self._Model(), mapping=torch.tensor([-1, 0, 1, 2]), sasrec=None,
                                   num_negatives=10, curriculum_steps=steps, args=args)

    def test_neither_teacher_free_sampler_has_a_curriculum(self) -> None:
        """policy is always on-policy: the run's curriculum length and sampler
        temperature never reach it."""
        random, policy = self._build("random"), self._build("policy")
        assert (random.cfg.mode, random.cfg.curriculum_steps, random.scorer) == ("random", 0, None)
        assert (policy.cfg.mode, policy.cfg.curriculum_steps, policy.cfg.temperature, policy.scorer) == (
            "policy", 0, 1.0, None)
        assert policy.cfg.exclude_history and random.cfg.exclude_history
        args = tiger_free.parse_args(["--dataset", "beauty", "--sampler-temperature", "3"])
        hot = gflow.build_sampler("policy", model=self._Model(), mapping=torch.tensor([-1, 0, 1, 2]), sasrec=None,
                                  num_negatives=10, curriculum_steps=99, args=args)
        assert (hot.cfg.curriculum_steps, hot.cfg.temperature) == (0, 1.0)
        gpt2_args = gpt2_free.parse_args(["--sampler-temperature", "3"])

        class _Dm:
            pass

        import unittest.mock as mock

        with mock.patch.object(gpt2.tiger_runner, "item_id_to_semantic_row", return_value=torch.tensor([-1, 0, 1, 2])):
            hot = gpt2.build_sampler("policy", model=self._Model(), dm=_Dm(), sasrec=None, num_negatives=10,
                                     curriculum_steps=99, args=gpt2_args)
        assert (hot.cfg.mode, hot.cfg.curriculum_steps, hot.cfg.temperature) == ("policy", 0, 1.0)

    def test_teacher_sampler_still_needs_the_teacher(self) -> None:
        with pytest.raises(RuntimeError, match="SASRec teacher"):
            self._build("teacher")


class TestGpt2Wrapper:
    def test_same_grid_as_tiger_in_gpt2_component_names(self, block) -> None:
        args = gpt2_free.parse_args([])
        grid = gpt2.build_grid(args)
        assert len(grid) == 2 * block["runs_per_dataset"]             # beauty + instruments
        beauty = [e for e in grid if e["dataset"] == "beauty"]
        tiger = gflow.build_grid(tiger_free.parse_args(["--dataset", "beauty"]))
        names = block["gpt2_component_names"]
        as_gpt2 = [("_x_".join(names[p] for p in e["combo"]), e["sampler"], e["power"], e["rho"], e["objective"])
                   for e in tiger]
        assert [(gpt2.combo_key(e["components"]), e["sampler"], e["power"], e["rho"], e["objective"])
                for e in beauty] == as_gpt2                                # cell for cell, same order
        assert not gpt2.needs_teacher(grid)
        assert gpt2.components_for(grid) == block["gpt2"]["cache_components"]
        assert all(set(e["weights"]) == {1.0} for e in grid)

    def test_protocol_is_the_full_history_wrappers(self, block) -> None:
        free, matched = gpt2_free.parse_args([]), gpt2_matched.parse_args([])
        for key in ("train_last_n", "batch_size", "grad_accum", "val_batch_size", "test_batch_size",
                    "validate_every", "patience", "min_epochs", "curriculum_epochs", "seed", "powers",
                    "objectives", "rho_targets", "num_candidates", "cache_root", "resume"):
            assert getattr(free, key) == getattr(matched, key), key
        assert str(free.output_root) == block["gpt2"]["output_root"] != str(matched.output_root)
        assert str(free.metrics_csv) == block["gpt2"]["metrics_csv"]

    def test_both_similarities_fit_one_combo_and_run_dirs_are_unique(self) -> None:
        grid = gpt2.build_grid(gpt2_free.parse_args(["--datasets", "beauty"]))
        triple = [e for e in grid if e["components"] == ("interaction", "sim_hier", "sim_embed")]
        assert len(triple) == 2 * 24 and triple[0]["similarity_mode"] == "hierarchical+embedding_cosine"
        dirs = {gpt2.run_dir_for(Path("/x"), e, 42) for e in grid}
        assert len(dirs) == len(grid)
        assert any("sim_hier_x_sim_embed_neg-policy" in str(d) for d in dirs)

    def test_legacy_gpt2_grid_is_untouched(self) -> None:
        grid = gpt2.build_grid(gpt2_matched.parse_args(["--datasets", "beauty"]))
        assert len(grid) == 168 and {e["sampler"] for e in grid} == {"teacher"}
        assert gpt2.run_dir_for(Path("/x"), grid[0], 42) == Path("/x/beauty/sasrec/w1/pow_0p25/tb_k10/seed_42")
        assert gpt2.needs_teacher(grid)

    def test_bad_component_combo_is_rejected(self) -> None:
        with pytest.raises(SystemExit, match="component-combos"):
            gpt2.parse_component_combo("interaction_x_similarity")
        assert gpt2.parse_component_combo("sim_embed_x_interaction") == ("interaction", "sim_embed")


class TestLauncher:
    @staticmethod
    def _plan(**env: str) -> str:
        out = subprocess.run(["bash", str(SUBMIT_SCRIPT)], capture_output=True, text=True, check=True,
                             env={**os.environ, "DRY_RUN": "1", **env})
        return out.stdout + out.stderr

    def test_teacherfree_groups_are_opt_in(self) -> None:
        assert "teacherfree" not in self._plan(RECALC_GROUPS="all")

    def test_tiger_group_runs_the_teacherfree_script_with_its_caches(self) -> None:
        plan = self._plan(RECALC_GROUPS="teacherfree_tiger", TEACHERFREE_DATASETS="beauty instruments")
        commands = [line for line in plan.splitlines() if " submit-worker --parsable " in line]
        assert len(commands) == 2 and all("gflow_finetune_tiger_teacherfree.py" in c for c in commands)
        caches, grid = commands
        assert "CACHE_COMPONENTS=interaction similarity sim_embed" in caches and "--array=0-5 " in caches
        assert "--array=0-671 " in grid and "--dependency=afterok:DRYJOB1" in grid
        assert all("--mem" not in c and "--constraint=" in c for c in commands)
        assert all("flowrec_teacherfree_tiger" in c for c in commands)

    def test_gpt2_group_builds_no_sasrec_cache_and_no_control(self) -> None:
        plan = self._plan(RECALC_GROUPS="teacherfree_gpt2")
        commands = [line for line in plan.splitlines() if " submit-worker --parsable " in line]
        assert len(commands) == 2 and all("gflow_finetune_gpt2_amazon_teacherfree.py" in c for c in commands)
        assert "--array=1-3,5-7 " in commands[0] and "--array=0-671 " in commands[1]
        assert "SFT_ONLY=1" not in plan
