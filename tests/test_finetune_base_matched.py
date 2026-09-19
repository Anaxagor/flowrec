"""The base-matched fine-tuning protocol and its consumer scripts stay in sync.

configs/exp2-tuned-params/finetune_base_matched.json is the single source of
truth for the DATA SCOPE and the BATCH SIZES of every fine-tuning arm: each arm
trains on exactly the base SFT run's examples (no per-user subset), and every
arm of every model on a dataset uses the same micro-batch, accumulation,
validation batch and final-eval batch. The three wrappers must resolve to it:

  sft_finetune_control_matched.py        TIGER SFT-only, all five datasets
  gflow_finetune_tiger_matched.py        TIGER GFlow, all five datasets
  gflow_finetune_gpt2_amazon_fullhist.py GPT-2 SFT-only + GFlow, beauty / instruments
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_gpt2_amazon as gpt2  # noqa: E402
import gflow_finetune_gpt2_amazon_fullhist as gpt2_matched  # noqa: E402
import gflow_finetune_yambda500m as gflow  # noqa: E402
import gflow_finetune_tiger_matched as gflow_matched  # noqa: E402
import sft_finetune_control as control  # noqa: E402
import sft_finetune_control_matched as control_matched  # noqa: E402
import sft_finetune_control_yambda as control_yambda  # noqa: E402

MATCHED_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"
SFT_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_sft_only.json"
GFLOW_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_gflow.json"

AMAZON = ("beauty", "instruments")          # datasets with a GPT-2 arm: 32 x 64
AMAZON_2023 = ("beauty_2023",)               # TIGER-only Amazon dataset
YAMBDA = ("yambda_50m_listens", "yambda_500m_listens")
TIGER_ONLY = (*AMAZON_2023, *YAMBDA)         # 256 x 8
ALL = (*AMAZON, *AMAZON_2023, *YAMBDA)
BATCH_KEYS = ("batch_size", "grad_accum", "val_batch_size", "test_batch_size")


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(MATCHED_CONFIG.read_text())


@pytest.fixture()
def matched_control(monkeypatch, protocol):
    """The control module retargeted the way the wrapper's main does it,
    restored afterwards (other tests rely on the first-sweep DATASETS)."""
    monkeypatch.setattr(control, "DATASETS", control_matched.matched_datasets(protocol))
    monkeypatch.setattr(control, "DEFAULT_OUTPUT_ROOT",
                        Path(protocol["arms"]["tiger_sft_only"]["output_root"]))
    return control


def _control_args(dataset: str, protocol: dict, *extra: str):
    return control.parse_args([*control_matched.matched_defaults(dataset, protocol),
                               "--datasets", dataset, *extra])


class TestProtocolFile:
    def test_data_scope_of_the_fine_tuning_arms(self, protocol) -> None:
        """No per-user TAIL subset anywhere. One deliberate cost exception:
        yambda_500m fine-tunes on 15 RANDOM targets per user of the base
        run's last-128 layout - identically in both arms."""
        assert {n: b["train_random_n"] for n, b in protocol["datasets"].items()} == {
            "beauty": 0, "instruments": 0, "beauty_2023": 0, "yambda_50m_listens": 0, "yambda_500m_listens": 15}
        assert protocol["datasets"]["yambda_500m_listens"]["train_subset_seed"] == 42
        assert set(protocol["datasets"]) == set(ALL)
        for arm in ("tiger_sft_only", "tiger_gflow"):  # TIGER: both arms everywhere
            assert protocol["arms"][arm]["datasets"] == list(ALL)
        assert protocol["arms"]["gpt2_sft_only_and_gflow"]["datasets"] == list(AMAZON)
        for name, block in protocol["datasets"].items():
            assert block["train_last_n"] == 0, name
            assert block["val_sample_resample"] is False, name

    def test_effective_batch_is_2048_everywhere(self, protocol) -> None:
        effective = protocol["shared_schedule"]["effective_batch"]
        assert effective == 2048
        for name, block in protocol["datasets"].items():
            assert block["batch_size"] * block["grad_accum"] == effective, name

    def test_batch_sizes_are_one_setting_per_dataset_family(self, protocol) -> None:
        for family, expected in ((AMAZON, (32, 64, 64, 64)), (TIGER_ONLY, (256, 8, 256, 256))):
            settings = {tuple(protocol["datasets"][n][k] for k in BATCH_KEYS) for n in family}
            assert settings == {expected}, family

    def test_amazon_micro_batch_fits_the_gpt2_candidate_cap(self, protocol) -> None:
        cap = json.loads(GFLOW_CONFIG.read_text())["gpt2_amazon"]["max_candidate_batch"]
        k = protocol["gflow_grid"]["num_negatives"]
        for name in AMAZON:
            assert protocol["datasets"][name]["batch_size"] * (k + 1) <= cap

    def test_nothing_in_the_protocol_is_model_specific(self, protocol) -> None:
        """One K, one grid, one schedule: per-dataset blocks hold only batch
        sizes and the validation-sample rule."""
        allowed = {"base_sft_data", "train_last_n", "train_random_n", *BATCH_KEYS,
                   "val_sample_fraction", "val_sample_resample"}
        for name, block in protocol["datasets"].items():
            assert set(block) - {"train_subset_seed"} == allowed, name
        assert protocol["gflow_grid"]["num_negatives"] == 10

    def test_optimizer_and_evaluation_are_the_existing_fine_tuning_protocol(self, protocol) -> None:
        shared = protocol["shared_schedule"]
        gflow_shared = json.loads(GFLOW_CONFIG.read_text())["shared_with_sft_control"]
        for key in ("lr", "weight_decay", "max_epochs", "max_grad_norm", "amp_dtype",
                    "run_validation_at_start", "eval_num_beams", "test_ks"):
            assert shared[key] == gflow_shared[key], key
        sft = json.loads(SFT_CONFIG.read_text())
        assert shared["monitor"] == sft["monitoring"]["monitor"]
        assert shared["val_num_beams"] == sft["monitoring"]["val_num_beams"]
        assert shared["seeds_sft_only"] == sft["data"]["seeds"]
        assert shared["run_validation_at_start"] is True

    def test_stopping_schedule(self, protocol) -> None:
        """Validate every epoch, patience 5 epochs, and no patience counting
        while a GFlow curriculum could still be ramping."""
        shared = protocol["shared_schedule"]
        assert (shared["validate_every"], shared["patience"]) == (1, 5)
        assert shared["early_stopping_min_epochs"] >= shared["gflow_curriculum_epochs"] == 10
        # the first sweeps' 5 / 10 stay the defaults of the first-sweep scripts
        first = json.loads(GFLOW_CONFIG.read_text())["shared_with_sft_control"]
        assert (first["validate_every"], first["patience"]) == (5, 10)

    def test_registered_in_the_protocol_configs(self, protocol) -> None:
        for path, script in ((SFT_CONFIG, protocol["arms"]["tiger_sft_only"]["script"]),
                             (GFLOW_CONFIG, protocol["arms"]["tiger_gflow"]["script"]),
                             (GFLOW_CONFIG, protocol["arms"]["gpt2_sft_only_and_gflow"]["script"])):
            cfg = json.loads(path.read_text())
            assert script in cfg["consumers"]
            assert cfg["base_matched_protocol"]["config"] == str(MATCHED_CONFIG.relative_to(PROJECT_ROOT))
        for arm in protocol["arms"].values():
            assert (PROJECT_ROOT / arm["script"]).exists(), arm["script"]


class TestBaseDataIsWhatTheProtocolSays:
    """The fine-tuning arms build their data from the checkpoint's own captured
    config; these are the target policies the protocol file describes."""

    def test_amazon_checkpoints_trained_on_every_prefix(self, protocol) -> None:
        specs = control_matched.matched_datasets(protocol)
        for name in (*AMAZON, *AMAZON_2023):
            dataset = json.loads((PROJECT_ROOT / specs[name]["sft_config"]).read_text())["dataset"]
            assert dataset["train_target_policy"] == "all" and dataset["train_all_prefixes"] is True
            assert dataset["max_history_len"] == 20

    def test_yambda_checkpoints_trained_on_the_last_128_targets(self) -> None:
        for name in YAMBDA:
            spec = control_yambda.YAMBDA_DATASETS[name]
            dataset = json.loads((PROJECT_ROOT / spec["sft_config"]).read_text())["dataset"]
            assert (dataset["train_target_policy"], dataset["train_last_n"]) == ("last_n", 128)
            assert dataset["max_history_len"] == 50


class TestTigerSftOnlyControl:
    def test_keeps_the_first_sweeps_data_and_init_checkpoint(self, protocol) -> None:
        first = {**control.DATASETS, **control_matched.EXTRA_DATASETS, **control_yambda.YAMBDA_DATASETS}
        matched = control_matched.matched_datasets(protocol)
        assert list(matched) == protocol["arms"]["tiger_sft_only"]["datasets"]
        for name, spec in matched.items():
            for key in ("sft_config", "tiger_checkpoint", "data_overrides"):
                assert spec[key] == first[name][key], (name, key)

    @pytest.mark.parametrize("dataset", ALL)
    def test_defaults_come_from_the_protocol_file(self, dataset, protocol, matched_control) -> None:
        block, arm = protocol["datasets"][dataset], protocol["arms"]["tiger_sft_only"]
        arm = {**arm, **arm.get("dataset_roots", {}).get(dataset, {})}
        a = _control_args(dataset, protocol)
        assert a.datasets == [dataset]
        assert (a.batch_size, a.grad_accum, a.val_batch_size) == (
            block["batch_size"], block["grad_accum"], block["val_batch_size"])
        assert a.validate_at_start is True and a.resume is True
        shared = protocol["shared_schedule"]
        assert (a.validate_every, a.patience, a.min_epochs) == (
            shared["validate_every"], shared["patience"], shared["early_stopping_min_epochs"])
        assert str(a.output_root) == arm["output_root"]
        assert str(a.artifacts_root) == arm["artifacts_root"]
        assert str(a.promote_root) == arm["promote_root"]
        # Never the first sweep's trees.
        first = control.parse_args([])
        assert a.promote_root != first.promote_root and a.artifacts_root != first.artifacts_root

    @pytest.mark.parametrize("dataset", ALL)
    def test_run_config_changes_batches_and_monitor_but_not_the_data(
        self, dataset, protocol, matched_control
    ) -> None:
        block = protocol["datasets"][dataset]
        config = control.build_run_config(dataset, 3e-5, 42, _control_args(dataset, protocol))
        source = json.loads((PROJECT_ROOT / control.DATASETS[dataset]["sft_config"]).read_text())
        assert config["dataset"] == source["dataset"]  # the checkpoint's own data config
        assert config["dataloader"]["train_batch_size"] == block["batch_size"]
        assert config["dataloader"]["validation_batch_size"] == block["val_batch_size"]
        training = config["training"]
        assert training["gradient_accumulation_steps"] == block["grad_accum"]
        assert training["validate_every_epochs"] == 1
        assert training["early_stopping_patience"] == 5
        assert training["early_stopping_min_epochs"] == 10
        trainer_cfg = control.tiger_runner.build_trainer_config(config, PROJECT_ROOT)
        assert trainer_cfg.early_stopping_min_epochs == 10  # survives run_tiger's allow-list
        assert training["run_validation_at_start"] is True
        assert training["val_sample_fraction"] == block["val_sample_fraction"]
        assert training["val_sample_resample"] is False

    def test_first_sweep_scripts_keep_the_trainer_default_sample_rule(self) -> None:
        config = control.build_run_config("beauty", 3e-5, 42, control.parse_args([]))
        assert "val_sample_resample" not in config["training"]
        assert config["training"]["early_stopping_min_epochs"] == 0
        assert (config["training"]["validate_every_epochs"],
                config["training"]["early_stopping_patience"]) == (5, 10)
        config = control.build_run_config("beauty", 3e-5, 42,
                                          control.parse_args(["--no-val-sample-resample"]))
        assert config["training"]["val_sample_resample"] is False

    def test_main_runs_each_dataset_with_its_own_batch_sizes(self, monkeypatch, protocol, matched_control) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(control, "main", lambda argv=None: calls.append(list(argv)))
        control_matched.main(["--datasets", "beauty", "yambda_50m_listens", "--seeds", "42"])
        assert [c[c.index("--datasets") + 1] for c in calls] == ["beauty", "yambda_50m_listens"]
        for call, dataset in zip(calls, ("beauty", "yambda_50m_listens")):
            assert call[-2:] == ["--seeds", "42"]
            assert call[call.index("--batch-size") + 1] == str(protocol["datasets"][dataset]["batch_size"])


class TestTigerGflow:
    @pytest.mark.parametrize("dataset", ALL)
    def test_defaults_come_from_the_protocol_file(self, dataset, protocol) -> None:
        block, arm = protocol["datasets"][dataset], protocol["arms"]["tiger_gflow"]
        a = gflow_matched.parse_args(["--dataset", dataset])
        assert a.train_last_n == 0 and gflow.cache_last_n(a) is None
        assert (a.batch_size, a.grad_accum, a.val_batch_size) == (
            block["batch_size"], block["grad_accum"], block["val_batch_size"])
        assert a.num_candidates == [protocol["gflow_grid"]["num_negatives"]] == [10]
        assert a.val_sample_fraction == block["val_sample_fraction"]
        assert a.val_sample_resample is False
        assert a.validate_at_start is True and a.resume is True
        shared = protocol["shared_schedule"]
        assert (a.validate_every, a.patience, a.min_epochs) == (
            shared["validate_every"], shared["patience"], shared["early_stopping_min_epochs"])
        assert str(a.output_root) == arm["output_roots"][dataset]
        assert str(a.cache_root) == arm["cache_roots"][dataset]
        assert gflow.metrics_csv_path(a).name == arm["metrics_csv_names"][dataset]

    @pytest.mark.parametrize("dataset", YAMBDA)  # the Amazon datasets have no last-8 sweep
    def test_never_shares_outputs_or_caches_with_the_last_8_sweeps(self, dataset) -> None:
        matched = gflow_matched.parse_args(["--dataset", dataset])
        first = gflow.parse_args(["--dataset", dataset])
        assert first.train_last_n == 8
        assert matched.output_root != first.output_root
        assert matched.cache_root != first.cache_root
        assert gflow.metrics_csv_path(matched) != gflow.metrics_csv_path(first)
        assert gflow.metrics_csv_path(first).name == gflow.METRICS_CSV_NAME

    @pytest.mark.parametrize("dataset", ALL)
    def test_same_init_checkpoint_and_data_config_as_the_control(self, dataset, protocol) -> None:
        gflow_matched.parse_args(["--dataset", dataset])  # activates the dataset's assets
        spec = control_matched.matched_datasets(protocol)[dataset]
        assert gflow.SFT_CONFIG == spec["sft_config"]
        assert gflow.TIGER_CHECKPOINT == spec["tiger_checkpoint"]
        assert gflow.DATA_OVERRIDES == spec.get("data_overrides", {})

    def test_dataset_is_required(self) -> None:
        with pytest.raises(SystemExit, match="--dataset is required"):
            gflow_matched.parse_args([])

    @pytest.mark.parametrize("dataset", (*AMAZON, *AMAZON_2023))
    def test_amazon_data_is_the_controls_data_with_the_verified_rule_as_a_criterion(
        self, dataset, protocol, matched_control
    ) -> None:
        """Same captured config and parquet files as the SFT-only control; the
        only dataset keys that differ express verified-purchase as a positive
        criterion (identical examples, keeps unverified reviews for the
        interaction reward) - checked on the real data for beauty / instruments."""
        a = gflow_matched.parse_args(["--dataset", dataset])
        run = gflow.build_run_config(a)
        ctrl = control.build_run_config(dataset, 3e-5, 42, _control_args(dataset, protocol))
        assert run["data"] == ctrl["data"]
        differing = {k for k in set(run["dataset"]) | set(ctrl["dataset"])
                     if run["dataset"].get(k) != ctrl["dataset"].get(k)}
        assert differing <= {"use_verified_only", "positive_criterion"}
        assert run["dataset"]["positive_criterion"] == "verified_purchase == True"
        assert run["dataset"]["use_verified_only"] is False
        assert run["dataset"]["train_target_policy"] == "all"
        assert run["dataloader"]["train_batch_size"] == protocol["datasets"][dataset]["batch_size"]
        assert run["dataloader"]["validation_batch_size"] == protocol["datasets"][dataset]["val_batch_size"]

    @pytest.mark.parametrize("dataset", (*AMAZON, *AMAZON_2023))
    def test_amazon_assets(self, dataset) -> None:
        import gflow_finetune_gpt2_amazon as gpt2_script

        gflow_matched.parse_args(["--dataset", dataset])
        spec = gflow.component_spec("similarity", None)["reward"]
        assert spec["mode"] == "hierarchical" and spec["level_stride"] == 256
        assert Path(spec["rqvae_model_path"]).name == "model.pt"
        assert f"/{dataset}/" in spec["rqvae_model_path"]
        assert gflow.SASREC_CHECKPOINT == f"checkpoints/exp2-tuned_params/initial/sasrec/{dataset}/best_model.pt"
        if dataset in gpt2_script.DATASETS:  # the same teacher as the GPT-2 arm
            assert gflow.SASREC_CHECKPOINT == gpt2_script.DATASETS[dataset]["sasrec_checkpoint"]
            assert gflow.TEACHER_NDCG_THRESHOLD == gpt2_script.DATASETS[dataset]["teacher_ndcg_threshold"]

    def test_switching_datasets_never_leaks_the_previous_spec(self) -> None:
        gflow_matched.parse_args(["--dataset", "instruments"])
        assert gflow.DATASET_OVERRIDES == {"use_verified_only": False}
        gflow_matched.parse_args(["--dataset", "yambda_50m_listens"])
        assert gflow.DATASET_OVERRIDES == {"allowed_event_types": None, "min_played_ratio_pct": None}
        assert gflow.DATA_OVERRIDES == {} and "listen" in gflow.POSITIVE_CRITERION
        assert {frozenset(spec) for spec in gflow.DATASETS.values()} == {frozenset(gflow.DATASETS["beauty"])}

    @pytest.mark.parametrize("dataset", ALL)
    def test_grid_is_the_48_run_comparison_grid(self, dataset, protocol) -> None:
        spec = protocol["gflow_grid"]
        a = gflow_matched.parse_args(["--dataset", dataset])
        grid = gflow.build_grid(a)
        assert len(grid) == spec["runs_per_dataset"] == 48
        assert {gflow.combo_key(e["combo"]) for e in grid} == set(spec["combos"]) == {
            "sasrec", "sasrec_x_interaction_x_similarity"}
        assert {e["power"] for e in grid} == set(spec["powers"]) == {0.25, 1.0, 4.0, 16.0}
        assert {e["objective"] for e in grid} == set(spec["objectives"]) == {"tb", "db", "subtb"}
        assert {e["rho"] for e in grid} == set(spec["rho_targets"]) == {1.0, 0.1}
        assert {e["num_negatives"] for e in grid} == {10}
        # every component enters with weight 1 (no interaction-exponent dimension)
        assert {e["interaction_exponent"] for e in grid} == {1.0} and spec["component_weights"] == 1.0
        # rho gets its own run dir, so the two ratios never collide
        dirs = {gflow.run_dir_for(Path("x"), e, 42) for e in grid}
        assert len(dirs) == 48
        assert gflow.components_for([e["combo"] for e in grid]) == ["sasrec", "interaction", "similarity"]

    def test_user_flags_still_override_the_defaults(self) -> None:
        a = gflow_matched.parse_args(["--dataset", "yambda_50m_listens", "--num-candidates", "3", "10",
                                      "--combos", "sasrec"])
        assert a.num_candidates == [3, 10] and a.combos == ["sasrec"]

    def test_full_layout_means_every_train_example(self) -> None:
        class Dataset:
            examples = pd.DataFrame({"user_id": [0] * 12 + [1] * 3})

            def __len__(self) -> int:
                return len(self.examples)

        full = gflow_matched.parse_args(["--dataset", "yambda_50m_listens"])
        assert gflow.finetune_indices(Dataset(), full) == list(range(15))
        tail = gflow.parse_args(["--dataset", "yambda_50m_listens"])
        assert gflow.finetune_indices(Dataset(), tail) == [*range(4, 12), 12, 13, 14]

    def test_micro_batch_must_not_shrink_with_k(self) -> None:
        a = gflow_matched.parse_args(["--dataset", "yambda_50m_listens"])
        gflow_matched.check_uniform_micro_batch(a)  # K=10: 256 x 11 <= 4096
        scaled = gflow_matched.parse_args(["--dataset", "yambda_50m_listens", "--num-candidates", "100"])
        with pytest.raises(SystemExit, match="micro-batch"):
            gflow_matched.check_uniform_micro_batch(scaled)

    def test_full_layout_and_tail_caches_are_not_interchangeable(self, tmp_path) -> None:
        def manifest(last_n):
            return {"kind": gflow.CACHE_KIND, "component": {"name": "sasrec"},
                    "num_examples": 100, "per_user_last_n": last_n}

        full = gflow_matched.parse_args(["--dataset", "yambda_50m_listens"])
        tail = gflow.parse_args(["--dataset", "yambda_50m_listens"])
        assert gflow.check_component_manifest(manifest(None), tmp_path, "sasrec", 100, full)
        assert gflow.check_component_manifest(manifest(8), tmp_path, "sasrec", 100, tail)
        for stored, args in ((8, full), (None, tail), (4, tail)):
            with pytest.raises(RuntimeError, match="fine-tunes on"):
                gflow.check_component_manifest(manifest(stored), tmp_path, "sasrec", 100, args)


class TestYambda500mRandomSubset:
    """Cost decision: yambda_500m fine-tunes on 15 random targets per user of
    the last-128 layout. Both arms must draw the SAME examples, whatever the
    run seed, and keep everything apart from full-layout runs."""

    def test_both_arms_resolve_to_the_same_draw(self, protocol, matched_control) -> None:
        block = protocol["datasets"]["yambda_500m_listens"]
        g_args = gflow_matched.parse_args(["--dataset", "yambda_500m_listens"])
        assert gflow.subset_spec(g_args) == {"per_user_last_n": None, "per_user_random_n": 15,
                                             "per_user_random_seed": 42}
        spec = control.DATASETS["yambda_500m_listens"]
        assert (spec["train_random_n"], spec["train_subset_seed"]) == (
            block["train_random_n"], block["train_subset_seed"]) == (15, 42)

        class Dataset:
            examples = pd.DataFrame({"user_id": [0] * 128 + [1] * 9 + [2] * 40})

            def __len__(self) -> int:
                return len(self.examples)

        class DM:
            train_dataset = Dataset()

        gflow_indices = gflow.finetune_indices(Dataset(), g_args)
        for run_seed in ("42", "43", "44"):  # the run seed must not move the subset
            c_args = _control_args("yambda_500m_listens", protocol, "--seeds", run_seed)
            assert control.finetune_subset_indices(DM(), "yambda_500m_listens", c_args) == gflow_indices
        assert len(gflow_indices) == 15 + 9 + 15                     # a user with fewer keeps them all
        assert gflow_indices == sorted(gflow_indices)                # dataset order = cache row order
        first_user = [i for i in gflow_indices if i < 128]
        assert min(first_user) < 64 and first_user != list(range(113, 128))  # spread over the window, not the tail

    def test_other_datasets_keep_the_whole_layout(self, protocol, matched_control) -> None:
        class DM:
            train_dataset = None

        for dataset in ("beauty", "yambda_50m_listens"):
            assert control.finetune_subset_indices(DM(), dataset, _control_args(dataset, protocol)) is None
            a = gflow_matched.parse_args(["--dataset", dataset])
            assert gflow.subset_spec(a) == {"per_user_last_n": None, "per_user_random_n": None,
                                            "per_user_random_seed": None}

    def test_own_locations(self, protocol, matched_control) -> None:
        g_args = gflow_matched.parse_args(["--dataset", "yambda_500m_listens"])
        assert "r15" in str(g_args.output_root) and "r15" in str(g_args.cache_root)
        assert "r15" in gflow.metrics_csv_path(g_args).name
        c_args = _control_args("yambda_500m_listens", protocol)
        assert all("r15" in str(x) for x in (c_args.output_root, c_args.artifacts_root, c_args.promote_root))
        other = _control_args("yambda_50m_listens", protocol)
        assert "r15" not in str(other.output_root) and "r15" not in str(other.promote_root)

    def test_cache_must_match_the_draw(self, tmp_path) -> None:
        def manifest(**subset):
            base = {"kind": gflow.CACHE_KIND, "component": {"name": "sasrec"}, "num_examples": 100,
                    "per_user_last_n": None, "per_user_random_n": None, "per_user_random_seed": None}
            return {**base, **subset}

        args = gflow_matched.parse_args(["--dataset", "yambda_500m_listens"])
        ok = manifest(per_user_random_n=15, per_user_random_seed=42)
        assert gflow.check_component_manifest(ok, tmp_path, "sasrec", 100, args)
        for wrong in (manifest(), manifest(per_user_last_n=8), manifest(per_user_random_n=15, per_user_random_seed=7),
                      manifest(per_user_random_n=16, per_user_random_seed=42)):
            with pytest.raises(RuntimeError, match="fine-tunes on 15 random examples per user"):
                gflow.check_component_manifest(wrong, tmp_path, "sasrec", 100, args)

    def test_cache_builder_gets_the_same_draw(self, monkeypatch, tmp_path) -> None:
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            out = Path(cmd[cmd.index("--output-dir") + 1]); out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(json.dumps({
                "kind": gflow.CACHE_KIND, "component": {"name": "sasrec"}, "num_examples": 10,
                "per_user_last_n": None, "per_user_random_n": 15, "per_user_random_seed": 42}))

        monkeypatch.setattr(gflow.subprocess, "run", fake_run)
        args = gflow_matched.parse_args(["--dataset", "yambda_500m_listens", "--cache-root", str(tmp_path / "c")])
        gflow.ensure_component_cache("sasrec", gflow.build_run_config(args), tmp_path, args, None)
        cmd = commands[0]
        assert cmd[cmd.index("--per-user-random-n") + 1] == "15"
        assert cmd[cmd.index("--per-user-random-seed") + 1] == "42"
        assert "--per-user-last-n" not in cmd

    def test_runs_already_queued_for_other_datasets_are_unaffected(self, tmp_path, protocol, matched_control) -> None:
        """Queued array tasks pick up the updated code when they start, and their
        caches were built BEFORE the random-subset keys existed. Such a manifest
        must still be accepted for a full-layout run, and nothing about the
        other datasets' runs may have moved."""
        old_manifest = {"kind": gflow.CACHE_KIND, "component": {"name": "sasrec"}, "num_examples": 100,
                        "per_user_last_n": None}                      # no per_user_random_* keys
        for dataset in ("yambda_50m_listens", "beauty"):
            args = gflow_matched.parse_args(["--dataset", dataset])
            assert gflow.check_component_manifest(old_manifest, tmp_path, "sasrec", 100, args)
            assert args.train_random_n == 0 and "r15" not in str(args.output_root) + str(args.cache_root)
            assert "r15" not in gflow.metrics_csv_path(args).name
            assert gflow.finetune_indices(list(range(7)), args) == list(range(7))   # every example, as before
        old_tail = {**old_manifest, "per_user_last_n": 8}             # a first-sweep (last-8) cache
        first_sweep = gflow.parse_args(["--dataset", "yambda_50m_listens"])
        assert gflow.check_component_manifest(old_tail, tmp_path, "sasrec", 100, first_sweep)
        # the old 500M full-layout cache, on the other hand, must NOT be reused for the random subset
        with pytest.raises(RuntimeError, match="covers the full train layout"):
            gflow.check_component_manifest(old_manifest, tmp_path, "sasrec", 100,
                                           gflow_matched.parse_args(["--dataset", "yambda_500m_listens"]))
        # GPT-2 arms: their wrapper never reads the subset keys
        a = gpt2_matched.parse_args([])
        assert a.train_last_n == 0 and not hasattr(a, "train_random_n")
        assert "r15" not in str(a.output_root)

    def test_tail_and_random_are_exclusive(self) -> None:
        a = gflow_matched.parse_args(["--dataset", "yambda_500m_listens", "--train-last-n", "8"])
        with pytest.raises(SystemExit, match="exclusive"):
            gflow.subset_spec(a)


class TestCurriculumIsDefinedInEpochs:
    """GFlow arms: the easy->hard ramp lasts `gflow_curriculum_epochs` passes
    over the fine-tuning examples, whatever the data scope or micro-batch."""

    def test_epochs_to_micro_steps(self) -> None:
        # 993,926 examples at micro-batch 256 = 3,883 micro-steps per epoch
        # (the control's recorded global_step after its first epoch).
        assert gflow.base.curriculum_steps_for_epochs(1, 993_926, 256) == 3_883
        assert gflow.base.curriculum_steps_for_epochs(10, 993_926, 256) == 38_830
        assert gflow.base.curriculum_steps_for_epochs(0, 993_926, 256) == 0

    @pytest.mark.parametrize("dataset", YAMBDA)
    def test_tiger_gflow_ramp_is_ten_epochs_at_any_micro_batch(self, dataset, protocol) -> None:
        epochs = protocol["shared_schedule"]["gflow_curriculum_epochs"]
        a = gflow_matched.parse_args(["--dataset", dataset])
        assert a.curriculum_epochs == epochs == 10
        num_examples = 993_926
        for batch in (256, 128, 40):
            steps = gflow.curriculum_steps_for_batch(a, batch, num_examples)
            assert abs(steps * batch - epochs * num_examples) <= epochs * batch
        with pytest.raises(ValueError, match="number of fine-tuning examples"):
            gflow.curriculum_steps_for_batch(a, 256)

    def test_first_sweep_scripts_keep_the_micro_step_ramp(self) -> None:
        a = gflow.parse_args(["--dataset", "yambda_50m_listens"])
        assert a.curriculum_epochs is None
        assert gflow.curriculum_steps_for_batch(a, a.batch_size, 69_495) == a.curriculum_steps == 3000
        b = gpt2.parse_args([])
        assert b.curriculum_epochs is None
        assert gpt2.curriculum_steps_for_run(b, 128, 95_061) == b.curriculum_steps == 3000

    def test_gpt2_ramp_is_the_same_ten_epochs(self, protocol) -> None:
        epochs = protocol["shared_schedule"]["gflow_curriculum_epochs"]
        a = gpt2_matched.parse_args([])
        assert a.curriculum_epochs == epochs
        block = json.loads(GFLOW_CONFIG.read_text())["gpt2_amazon_fullhist"]
        assert block["curriculum_epochs"] == epochs
        # beauty full history: 131,413 prefixes at micro-batch 32 = 4,107 steps / epoch
        assert gpt2.curriculum_steps_for_run(a, 32, 131_413) == 10 * 4_107


class TestArmsAndModelsShareTheProtocol:
    @pytest.mark.parametrize("dataset", ALL)
    def test_tiger_gflow_matches_the_tiger_control(self, dataset, protocol, matched_control) -> None:
        g = gflow_matched.parse_args(["--dataset", dataset])
        c = _control_args(dataset, protocol)
        assert g.lr == c.lr[0]
        for key in ("weight_decay", "batch_size", "grad_accum", "val_batch_size", "max_epochs",
                    "validate_every", "patience", "min_epochs", "max_grad_norm", "eval_num_beams",
                    "test_ks", "validate_at_start"):
            assert getattr(g, key) == getattr(c, key), key
        assert g.amp_dtype == "float16"  # the control hardcodes float16
        for k in g.num_candidates:  # the run micro-batch, not just the protocol default
            assert gflow.base.batch_settings_for_k(k, g) == (c.batch_size, c.grad_accum)
        spec = control.DATASETS[dataset]
        assert g.val_sample_fraction == spec["val_sample_fraction"]
        assert g.val_sample_resample is spec["val_sample_resample"] is False

    @pytest.mark.parametrize("dataset", AMAZON)
    def test_gpt2_arms_match_the_tiger_control(self, dataset, protocol, matched_control) -> None:
        block = protocol["datasets"][dataset]
        c = _control_args(dataset, protocol)
        a = gpt2_matched.parse_args(["--datasets", dataset])
        assert a.train_last_n == block["train_last_n"] == 0
        k = protocol["gflow_grid"]["num_negatives"]
        assert a.num_candidates == [k]
        micro = (block["batch_size"], block["grad_accum"])
        assert micro == (32, 64)
        assert gpt2.run_batch_settings(a, None) == micro   # GPT-2 SFT-only
        assert gpt2.run_batch_settings(a, k) == micro      # GPT-2 GFlow
        assert (c.batch_size, c.grad_accum) == micro       # TIGER SFT-only
        assert a.val_batch_size == c.val_batch_size == block["val_batch_size"]
        # TIGER evaluates best.pt through the eval loaders (= the validation batch).
        assert a.test_batch_size == block["test_batch_size"] == c.val_batch_size
        assert a.validate_at_start is c.validate_at_start is True
        assert a.lr == c.lr[0] and a.weight_decay == c.weight_decay
        for key in ("max_epochs", "validate_every", "patience", "min_epochs", "max_grad_norm"):
            assert getattr(a, key) == getattr(c, key), key
        shared = protocol["shared_schedule"]
        assert (a.validate_every, a.patience, a.min_epochs) == (
            shared["validate_every"], shared["patience"], shared["early_stopping_min_epochs"])
        fullhist = json.loads(GFLOW_CONFIG.read_text())["gpt2_amazon_fullhist"]
        assert (fullhist["validate_every"], fullhist["patience"], fullhist["min_epochs"]) == (
            a.validate_every, a.patience, a.min_epochs)
        assert (fullhist["batch_size"], fullhist["grad_accum"], fullhist["num_negatives"]) == (*micro, k)
        # the last-8 sweep's base script is untouched
        b = gpt2.parse_args([])
        assert (b.validate_every, b.patience, b.min_epochs) == (5, 10, 0)


class TestTigerAndGpt2RunTheSameExperiment:
    """The logic and goals of the fine-tuning experiments do not differ between
    the two models: both GFlow wrappers and both SFT-only controls resolve,
    from the same protocol file, to the same grid, K, schedule, curriculum,
    calibration, seeds and evaluation."""

    SHARED = ("lr", "weight_decay", "max_epochs", "validate_every", "patience", "min_epochs",
              "max_grad_norm", "amp_dtype", "validate_at_start", "resume", "train_last_n",
              "curriculum_epochs", "sampler_temperature", "sasrec_transform", "sasrec_temperature",
              "gflow_weight_mode", "gflow_target_grad_ratio", "gflow_calibration_batches",
              "gflow_calibration_eps", "gflow_calibration_seed", "eval_num_beams", "test_ks", "seed",
              "powers", "objectives", "rho_targets", "num_candidates", "combos")

    def test_gflow_arms_share_every_protocol_parameter(self) -> None:
        tiger = gflow_matched.parse_args(["--dataset", "yambda_50m_listens"])
        gpt = gpt2_matched.parse_args([])
        for key in self.SHARED:
            assert getattr(tiger, key) == getattr(gpt, key), key
        assert gpt.weight_levels == [1.0] and gpt.composition == "multiply"
        assert gpt.similarity_modes == ["hierarchical"]  # TIGER's factor caches: hierarchical

    def test_grids_are_identical_cell_for_cell(self, protocol) -> None:
        tiger = gflow.build_grid(gflow_matched.parse_args(["--dataset", "yambda_50m_listens"]))
        for dataset in AMAZON:
            gpt = [e for e in gpt2.build_grid(gpt2_matched.parse_args([])) if e["dataset"] == dataset]
            assert len(gpt) == len(tiger) == protocol["gflow_grid"]["runs_per_dataset"]
            for t, g in zip(tiger, gpt):  # same order: combo, power, rho, objective
                assert len(t["combo"]) == len(g["components"])
                assert [c.replace("similarity", "sim_hier") for c in t["combo"]] == list(g["components"])
                assert (t["power"], t["rho"], t["objective"], t["num_negatives"]) == (
                    g["power"], g["rho"], g["objective"], g["num_negatives"])
                assert list(g["weights"]) == [1.0] * len(g["components"])
                assert t["interaction_exponent"] == 1.0

    def test_sft_only_controls_use_the_same_seeds(self, protocol, matched_control) -> None:
        seeds = protocol["shared_schedule"]["seeds_sft_only"]
        assert _control_args("beauty", protocol).seeds == seeds == [42, 43, 44]
        assert list(gpt2_matched.CONTROL_SEEDS) == seeds
        assert gpt2_matched.parse_args([]).seed == protocol["shared_schedule"]["seed_gflow"] == 42

    def test_the_gpt2_wrapper_has_no_constants_of_its_own(self, protocol) -> None:
        """Changing the protocol file changes both models together."""
        import copy

        changed = copy.deepcopy(protocol)
        changed["shared_schedule"]["patience"] = 7
        changed["gflow_grid"]["powers"] = [1.0]
        defaults = list(gpt2_matched.full_history_defaults(changed))
        assert defaults[defaults.index("--patience") + 1] == "7"
        assert defaults[defaults.index("--powers") + 1:defaults.index("--objectives")] == ["1.0"]
        tiger = gflow_matched.matched_defaults("yambda_50m_listens", changed)
        assert tiger[tiger.index("--patience") + 1] == "7"
        assert tiger[tiger.index("--powers") + 1:tiger.index("--objectives")] == ["1.0"]


SUBMIT_SCRIPT = PROJECT_ROOT / "scripts/slurm/submit_base_matched_recalc.sh"


@pytest.mark.skipif(not SUBMIT_SCRIPT.exists(), reason="scripts/slurm is gitignored (cluster launchers)")
class TestMassRecalculationLauncher:
    """DRY_RUN plan of the array launcher: by default only the experiments
    whose first sweep differed from the base SFT training conditions."""

    @staticmethod
    def _plan(**env: str) -> str:
        import os
        import subprocess

        out = subprocess.run(["bash", str(SUBMIT_SCRIPT)], capture_output=True, text=True, check=True,
                             env={**os.environ, "DRY_RUN": "1", **env})
        return out.stdout + out.stderr

    def test_default_selection_is_the_differing_experiments(self, protocol) -> None:
        plan = self._plan()
        assert set(protocol["launch"]["default_groups"]) == {"tiger_gflow_yambda", "gpt2_amazon"}
        assert "groups = tiger_gflow_yambda gpt2_amazon" in plan
        assert "tiger_control" not in plan and "tiger_gflow_amazon" not in plan
        # yambda_50m: a plain array (dataset x component caches -> dataset x cell grid)
        assert "DATASETS=yambda_50m_listens CACHE_ONLY=1" in plan
        assert "--dependency=afterok:DRYJOB1 --array=0-47 " in plan
        assert "--array=0-2,4-6 " in plan                      # GPT-2 caches without sim_embed
        assert "SFT_ONLY=1 CACHE_ONLY=0 SEEDS=42 43 44" in plan  # GPT-2 control, 2 datasets x 3 seeds
        assert "GRID_INDEX=1" in plan and "--array=0-95 " in plan  # GPT-2 grid over both datasets
        assert "array tasks planned: 210" in plan

    def test_no_memory_flag_and_a_node_type_on_every_array(self) -> None:
        """This Slurm takes no memory requests: host memory is handled by node
        type (fit) and by pinning (packing), never by --mem."""
        plan = self._plan(RECALC_GROUPS="all")
        commands = [line for line in plan.splitlines() if " sbatch --parsable " in line]
        assert commands
        for line in commands:
            assert "--mem" not in line, line
            assert "--constraint=" in line and "--time=" in line, line
        assert "node audit" in plan and "ERROR" not in plan

    def test_yambda_500m_is_pinned_one_task_per_node(self, protocol) -> None:
        plan = self._plan(RECALC_GROUPS="tiger_gflow_yambda tiger_control",
                          YAMBDA_DATASETS="yambda_500m_listens", CONTROL_DATASETS="yambda_500m_listens")
        commands = [line.strip() for line in plan.splitlines() if " sbatch --parsable " in line]
        assert len(commands) == 9                       # 3 lanes x (caches, grid, control)
        assert all("--nodelist=type_h-node" in c and "--constraint=type_h" in c for c in commands)
        assert all(c.split("--array=")[1].split()[0].endswith("%1") for c in commands)  # one task per lane
        grid = [c for c in commands if "CACHE_ONLY=0 BLOCK_SIZE" in c]
        assert sorted(c.split("--array=")[1].split()[0] for c in grid) == ["0-47:3%1", "1-47:3%1", "2-47:3%1"]
        # every grid lane waits for ALL caches and for its own lane's previous job
        assert all("--dependency=afterok:DRYJOB1:DRYJOB2:DRYJOB3,afterany:DRYJOB" in c for c in grid)
        control = [c for c in commands if "SEEDS=42 43 44" in c]
        assert sorted(c.split("--dependency=")[1].split()[0] for c in control) == [
            "afterany:DRYJOB4", "afterany:DRYJOB5", "afterany:DRYJOB6"]  # queued behind the grid on the same node
        assert "array tasks planned: 54" in plan

    def test_arrays_follow_a_trimmed_grid_and_the_throttle(self) -> None:
        plan = self._plan(RECALC_GROUPS="all", EXTRA_ARGS="--rho-targets 1.0", BLOCK_SIZE="2", MAX_PARALLEL="20")
        assert "24 cells per dataset, 12 tasks each" in plan
        assert "--array=0-11%20 " in plan          # one free dataset x 12 tasks
        assert "--array=0-23%20 " in plan          # beauty + instruments share node types and walltime: 2 x 12
        assert "--array=0-11:3%1 " in plan         # a yambda_500m lane
        assert "tiger_control: beauty instruments x seeds" in plan
        assert "tiger_control: yambda_500m_listens x seeds" in plan
        assert "array tasks planned: 126" in plan

    def test_live_cluster_without_memory_in_sinfo(self, tmp_path) -> None:
        """As on the real cluster: sinfo lists every node with RealMemory 1 MB.
        The audit must pass and the yambda_500m lanes must name real nodes."""
        import os

        lines = [f"cn-{i:03d}|type_a,rocky|1|gpu:v100:4|mixed" for i in range(1, 17)]
        lines += [f"cn-{i:03d}|type_c,rocky|1|gpu:v100:4|idle" for i in range(27, 30)]
        lines += ["cn-048|type_h,rocky|1|gpu:h200:8|mixed", "cn-049|type_h,rocky|1|gpu:h200:8|idle",
                  "cn-050|type_h,rocky|1|gpu:h200:4|mixed", "cn-017|type_b,rocky|1|gpu:v100:4|idle"]
        fake = tmp_path / "sinfo"
        fake.write_text("#!/bin/bash\ncat <<'EOF'\n" + "\n".join(lines) + "\nEOF\n")
        fake.chmod(0o755)
        plan = self._plan(PATH=f"{tmp_path}:{os.environ['PATH']}")
        assert "node audit (live sinfo)" in plan and "RAM per node type is taken from the hardware table" in plan
        assert "ERROR" not in plan and "FAILED" not in plan
        assert "--nodelist=cn-048 " in plan and "--nodelist=cn-050 " in plan
        assert "array tasks planned: 210" in plan

    def test_an_array_that_is_still_queued_is_not_submitted_twice(self, tmp_path) -> None:
        """Plain arrays have no claim files: two live copies of one cell would
        write to the same run dir. Lanes are chained instead, so they still go."""
        import os

        fake = tmp_path / "squeue"
        fake.write_text("#!/bin/bash\n"
                        "name=''; while [[ $# -gt 0 ]]; do [[ \"$1\" == -n ]] && name=\"$2\"; shift; done\n"
                        "[[ \"$name\" == flowrec_tiger_gflow_yambda_yambda_50m_listens_grid ]] && echo 7000001\n"
                        "exit 0\n")
        fake.chmod(0o755)
        env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "USER": os.environ.get("USER", "u"),
               "RECALC_GROUPS": "tiger_gflow_yambda"}
        plan = self._plan(**env)
        assert "already in the queue as job(s) 7000001 - not submitted again" in plan
        assert "CACHE_ONLY=0 BLOCK_SIZE=1 EXTRA_ARGS= sbatch --parsable --job-name=flowrec_tiger_gflow_yambda_yambda_50m" not in plan
        assert "--nodelist=type_h-node1 " in plan            # the yambda_500m lanes are still planned
        assert "array tasks planned: 54" in plan             # 3 + (3 + 48): the queued 48-task grid is left out
        forced = self._plan(**env, FORCE="1")
        assert "already in the queue" not in forced and "array tasks planned: 102" in forced
