import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/exp5_result_analysis/top10_manifest.py"
SPEC = importlib.util.spec_from_file_location("top10_manifest", PATH)
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def candidate(key, n, c, seed=42):
    return {"configuration_id": key, "checkpoint": f"{key}/seed_{seed}/best.pt", "seed": seed,
            "selection_metrics": {"NDCG@10": n, "Coverage@10": c}}


def test_selects_actual_pareto_compromise_and_keeps_seeds():
    inputs = [candidate("ndcg", .9, .1), candidate("coverage", .1, .9),
              candidate("middle", .65, .65, 42), candidate("middle", .75, .75, 43),
              candidate("dominated", .2, .2)]
    selected = manifest.select_three(inputs, "test")
    labels = {s["configuration_id"]: s["selection_labels"] for s in selected}
    assert labels == {"ndcg": ["best_ndcg"], "coverage": ["best_coverage"], "middle": ["pareto"]}
    middle = [s for s in selected if s["configuration_id"] == "middle"]
    assert [s["seed"] for s in middle] == [42, 43]
    assert middle[0]["selection_mean_metrics"]["NDCG@10"] == pytest.approx(.7)
    assert all(s["selection_split"] == "test" for s in selected)


def test_constant_front_merges_labels_deterministically():
    selected = manifest.select_three([candidate("z", .2, .3), candidate("a", .2, .3),
                                      candidate("bad", float("nan"), .5)])
    assert len(selected) == 1
    assert selected[0]["configuration_id"] == "a"
    assert selected[0]["selection_split"] == "test"
    assert selected[0]["selection_labels"] == ["best_ndcg", "best_coverage", "pareto"]


def test_selection_averages_seeds_before_comparing_configs():
    selected = manifest.select_three([candidate("variable", .9, .9), candidate("variable", .1, .1, 43),
                                      candidate("stable", .6, .6)])
    assert [s["configuration_id"] for s in selected] == ["stable"]


def test_legacy_paths_are_not_remapped_to_primary():
    old = "checkpoints/exp2-tuned_params/initial/gpt2/yambda_50m_listens/seed_43/best.pt"
    assert manifest.portable("/home/user/flowrec/" + old) == old
    assert manifest.portable(old) == old



def write_json(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def first_sft(root, model, dataset, seed=42):
    checkpoint = manifest.first_checkpoint(model, dataset, seed)
    if model == "gpt2":
        folder = root / checkpoint
        folder = folder.parents[1]
        write_json(folder / "result.json", {"status": "completed", "seed": seed,
                   "best_checkpoint": "/home/user/flowrec/" + checkpoint})
        write_json(folder / "resolved_config.json", {"init_checkpoint": manifest.BASE[model, dataset]})
    else:
        folder = (root / checkpoint).parent
        write_json(folder / "metrics.json", {"status": "completed", "seed": seed,
                   "best_checkpoint": "/home/user/flowrec/" + checkpoint})
        write_json(folder / "run_config.json", {"data": {"source": "amazon"}})
    return checkpoint


def exp8_run(root, model="gpt2", dataset="beauty", arm="gflow", seed=42,
             source_seed=42, name="cell", n=.4, c=.4, test_n=.4, test_c=.4):
    source_id = manifest.SOURCE_IDS[model, dataset]
    family = f"{manifest.EXP8}/{source_id}/sft_seed_{source_seed}"
    identity = {"source_id": source_id, "model": model, "dataset": dataset,
                "source_seed": source_seed, "protocol": manifest.SOURCE_PROTOCOLS[model, dataset],
                "checkpoint_sha256": "a" * 64, "source_fingerprint": "b" * 64}
    write_json(root / family / "source_identity.json", identity)
    run_dir = f"{family}/{arm}/{name}/seed_{seed}"
    path = root / family / "jobs" / f"{name}_{seed}" / "result.json"
    result = {"status": "completed", "seed": seed, "run_dir": run_dir,
              "best_checkpoint": f"{run_dir}/checkpoints/best.pt",
              "experiment_cell": {"source_id": source_id, "source_seed": source_seed,
                  "model": model, "dataset": dataset, "arm": arm, "seed": seed, "combo": name},
              "from_sft_source": identity,
              "splits": {"validation": {"NDCG@10": n, "Coverage@10": c},
                         "test": {"NDCG@10": test_n, "Coverage@10": test_c}}}
    write_json(path, result)
    return path, result


def test_all_first_sft_seeds_and_only_selected_families(tmp_path):
    expected = set()
    for model in manifest.MODELS:
        for dataset in manifest.DATASETS:
            for seed in (42, 43, 44):
                expected.add(first_sft(tmp_path, model, dataset, seed))
    wrong = tmp_path / manifest.OLD / "finetuned/tiger/yambda_50m_listens/sft_only_lr3e-05/seed_42/metrics.json"
    write_json(wrong, {"status": "completed", "seed": 42, "best_checkpoint": "wrong/best.pt"})
    runs = manifest.build_manifest(tmp_path)
    assert len(runs) == 12
    assert {r["checkpoint"] for r in runs} == expected
    assert {r["stage"] for r in runs} == {"sft"}
    assert all(r["preserve_legacy_checkpoint_path"] for r in runs)
    assert "beauty/tiger/gflow" in manifest.missing_groups(runs)


def test_missing_weights_kept_without_tuned_init_substitution(tmp_path):
    checkpoint = first_sft(tmp_path, "gpt2", "yambda_50m_listens", 43)
    runs = manifest.build_manifest(tmp_path)
    assert len(runs) == 1 and runs[0]["checkpoint"] == checkpoint
    assert not runs[0]["available"]
    assert runs[0]["init_checkpoint"].endswith("trial_36/best.pt")
    assert manifest.build_manifest(tmp_path / "empty") == []


def test_wrong_trial_rejected(tmp_path):
    checkpoint = first_sft(tmp_path, "gpt2", "yambda_50m_listens")
    folder = (tmp_path / checkpoint).parents[1]
    write_json(folder / "resolved_config.json", {"init_checkpoint": manifest.BASE["gpt2", "yambda_50m_listens"].replace("trial_36", "trial_4")})
    with pytest.raises(ValueError, match="trial_36"):
        manifest.build_manifest(tmp_path)


def test_first_tiger_experiment_records_deduplicate_promoted_metadata(tmp_path):
    checkpoint = first_sft(tmp_path, "tiger", "yambda_50m_listens", 43)
    folder = tmp_path / "experiments/exp2-sft-finetune-matched/yambda_50m_listens/sft_lr3e-05/seed_43"
    write_json(folder / "metrics.json", {"status": "completed", "seed": 43,
                                       "best_checkpoint": checkpoint})
    found = manifest.build_manifest(tmp_path)
    assert len(found) == 1
    assert found[0]["data_config"] == str(Path(checkpoint).with_name("run_config.json"))
    (tmp_path / checkpoint).with_name("metrics.json").unlink()
    found = manifest.build_manifest(tmp_path)
    assert len(found) == 1
    assert found[0]["data_config"] == (folder / "run_config.json").relative_to(tmp_path).as_posix()


def test_second_sft_all_runs_and_finetuning_selection_defaults_to_test(tmp_path):
    for seed in (42, 43):
        exp8_run(tmp_path, arm="sft", seed=seed, name="sft")
        exp8_run(tmp_path, name="valwinner", seed=seed, n=.9, c=.9, test_n=.1, test_c=.1)
        exp8_run(tmp_path, name="testwinner", seed=seed, n=.1, c=.1, test_n=.9, test_c=.9)
    # Native copies must not be counted a second time.
    path, result = exp8_run(tmp_path, arm="ce", name="ce")
    write_json(tmp_path / result["run_dir"] / "result.json", result)
    found = manifest.build_manifest(tmp_path)
    assert len([r for r in found if r["stage"] == "sft2"]) == 2
    gflow = [r for r in found if r["stage"] == "gflow"]
    assert len(gflow) == 2 and all("/testwinner/" in r["checkpoint"] for r in gflow)
    assert all(r["selection_split"] == "test" for r in gflow)
    assert len([r for r in found if r["stage"] == "ce"]) == 1
    gflow_validation = [r for r in manifest.build_manifest(tmp_path, "validation") if r["stage"] == "gflow"]
    assert len(gflow_validation) == 2
    assert all("/valwinner/" in r["checkpoint"] for r in gflow_validation)
    assert all(r["selection_split"] == "validation" for r in gflow_validation)


def test_exp8_without_best_checkpoint_uses_native_run_dir(tmp_path):
    path, result = exp8_run(tmp_path, model="tiger", dataset="yambda_50m_listens")
    del result["best_checkpoint"]
    write_json(path, result)
    found = manifest.build_manifest(tmp_path)
    assert found[0]["checkpoint"] == result["run_dir"] + "/checkpoints/best.pt"


@pytest.mark.parametrize("field,value", [("source_id", "gpt2_old"), ("protocol", "wrong"),
                                         ("checkpoint_sha256", "wrong")])
def test_exp8_wrong_source_provenance_rejected(tmp_path, field, value):
    import copy
    path, result = exp8_run(tmp_path)
    result = copy.deepcopy(result)
    result["from_sft_source"][field] = value
    write_json(path, result)
    with pytest.raises(ValueError, match="identity|provenance"):
        manifest.build_manifest(tmp_path)


def test_exp8_from_base_initialization_rejected(tmp_path):
    path, result = exp8_run(tmp_path)
    result["initial_checkpoint"] = manifest.BASE["gpt2", "beauty"]
    write_json(path, result)
    with pytest.raises(ValueError, match="Wrong first SFT checkpoint"):
        manifest.build_manifest(tmp_path)


def test_legacy_beauty_tiger_gflow_included(tmp_path):
    family = manifest.LEGACY_TIGER + "/sft_seed_42"
    run = tmp_path / family / "sasrec/pow_1/db_k10/seed_42"
    write_json(run / "result.json", {"status": "completed", "objective": "db", "seed": 42,
               "sft_checkpoint": manifest.first_checkpoint("tiger", "beauty"),
               "splits": {"test": {"NDCG@10": .4, "Coverage@10": .5}}})
    found = manifest.build_manifest(tmp_path)
    assert len(found) == 1
    assert found[0]["stage"] == "gflow" and found[0]["model"] == "tiger"
    assert found[0]["checkpoint"] == (run / "checkpoints/best.pt").relative_to(tmp_path).as_posix()
    assert "beauty/tiger/gflow" not in manifest.missing_groups(found)


def test_last_checkpoints_only_for_sft_and_only_if_present(tmp_path):
    first = first_sft(tmp_path, "gpt2", "beauty")
    last = (tmp_path / first).with_name("last.pt")
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"weights")
    _, second = exp8_run(tmp_path, arm="sft", name="second")
    write_json(tmp_path / (second["best_checkpoint"].replace("best.pt", "last.pt.json")), {})
    _, ft = exp8_run(tmp_path, arm="ce", name="ce")
    write_json(tmp_path / (ft["best_checkpoint"].replace("best.pt", "last.pt.json")), {})
    found = manifest.build_manifest(tmp_path, checkpoint_kinds=("best", "last"))
    assert len(found) == 5
    assert sorted(r["stage"] for r in found if r["checkpoint_kind"] == "last") == ["sft", "sft2"]
    defaults = manifest.build_manifest(tmp_path)
    assert len(defaults) == 3
    assert all(r["checkpoint_kind"] == "best" for r in defaults)


def test_frozen_scope_validation_rejects_from_base_and_wrong_variant(tmp_path):
    first_sft(tmp_path, "tiger", "yambda_50m_listens")
    first = manifest.build_manifest(tmp_path)[0]
    with pytest.raises(ValueError, match="selected trial/variant"):
        manifest.validate_run_scope({**first, "checkpoint": first["checkpoint"].replace("finetuned_matched", "finetuned")})
    with pytest.raises(ValueError, match="stages"):
        manifest.validate_run_scope({**first, "stage": "init"})
    exp8_run(tmp_path, model="tiger", dataset="yambda_50m_listens")
    ft = next(r for r in manifest.build_manifest(tmp_path) if r["stage"] == "gflow")
    with pytest.raises(ValueError, match="first SFT-only checkpoint"):
        manifest.validate_run_scope({**ft, "init_checkpoint": manifest.BASE["tiger", "yambda_50m_listens"]})


def test_first_gpt2_uses_bound_exp8_data_config_when_original_absent(tmp_path):
    dataset = "yambda_50m_listens"
    first_sft(tmp_path, "gpt2", dataset, 42)
    first_sft(tmp_path, "gpt2", dataset, 43)
    _, result = exp8_run(tmp_path, dataset=dataset)
    family = str(Path(result["run_dir"]).parents[2])
    fallback = f"{family}/data_reward_config.json"
    write_json(tmp_path / fallback, {"data": {"source": "yambda"},
               "from_sft_source": result["from_sft_source"]})
    runs = [r for r in manifest.build_manifest(tmp_path) if r["stage"] == "sft"]
    assert len(runs) == 2 and all(r["data_config"] == fallback for r in runs)
    # A cross-seed fallback cannot replace a different preprocessing protocol.
    changed = (tmp_path / manifest.first_checkpoint("gpt2", dataset, 43)).parents[1] / "resolved_config.json"
    write_json(changed, {"init_checkpoint": manifest.BASE["gpt2", dataset], "data": {"split_strategy": "wrong"}})
    with pytest.raises(ValueError, match="differs in data"):
        manifest.build_manifest(tmp_path)


def test_first_gpt2_prefers_original_config_to_exp8_fallback(tmp_path):
    dataset = "beauty"
    first_sft(tmp_path, "gpt2", dataset)
    original = f"{manifest.GPT2_FAMILIES[dataset]}/data_reward_config.json"
    write_json(tmp_path / original, {"data": {"source": "amazon"}})
    assert manifest.build_manifest(tmp_path)[0]["data_config"] == original
