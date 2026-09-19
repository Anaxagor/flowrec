"""Reward-FORM dimensions of the factorized-path gflow script: the sasrec
transform / temperature (teacher softmax) and the interaction exponent are grid
dimensions applied at lookup time; the script also runs on yambda_50m assets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts"),
           str(PROJECT_ROOT / "scripts" / "exp2-tuned-params"),
           str(PROJECT_ROOT / "scripts" / "exp4-reward-ablations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gflow_finetune_yambda500m as g  # noqa: E402


def _args(*argv: str):
    old = sys.argv
    sys.argv = ["x", *argv]
    try:
        return g.parse_args()
    finally:
        sys.argv = old


@pytest.fixture(autouse=True)
def _restore_default_dataset():
    yield
    g.activate_dataset(g.DEFAULT_DATASET)


class TestGrid:
    def test_reward_form_grid_is_nine_runs(self) -> None:
        a = _args("--combos", "sasrec", "sasrec_x_interaction", "--powers", "1", "--objectives", "db",
                  "--sasrec-transform", "exp", "--sasrec-temperatures", "1", "2", "4",
                  "--interaction-exponents", "1", "4")
        grid = g.build_grid(a)
        tags = [g.entry_tag(e) for e in grid]
        assert tags == [
            "sasrec/p1/db_k10_exp_T1", "sasrec/p1/db_k10_exp_T2", "sasrec/p1/db_k10_exp_T4",
            "sasrec_x_interaction/p1/db_k10_exp_T1", "sasrec_x_interaction/p1/db_k10_exp_T1_ie4",
            "sasrec_x_interaction/p1/db_k10_exp_T2", "sasrec_x_interaction/p1/db_k10_exp_T2_ie4",
            "sasrec_x_interaction/p1/db_k10_exp_T4", "sasrec_x_interaction/p1/db_k10_exp_T4_ie4",
        ]
        assert len({g.run_dir_for(Path("/r"), e, 42) for e in grid}) == 9

    def test_dimensions_collapse_where_they_do_not_apply(self) -> None:
        a = _args("--combos", "interaction", "similarity", "--powers", "1", "--objectives", "db",
                  "--sasrec-transform", "exp", "--sasrec-temperatures", "1", "2", "4",
                  "--interaction-exponents", "1", "4")
        grid = g.build_grid(a)
        assert len(grid) == 2  # no sasrec, no modulation: one run per combo
        assert {e["sasrec_transform"] for e in grid} == {"-"}
        assert {e["interaction_exponent"] for e in grid} == {1.0}

    def test_legacy_form_keeps_run_dirs_and_tags(self) -> None:
        a = _args()
        for entry in g.build_grid(a):
            assert g.variant_suffix(entry) == ""
            assert g.run_dir_for(Path("/r"), entry, 42) == g.base.run_dir_for(Path("/r"), entry, 42)
            assert g.entry_tag(entry) == g.base.entry_tag(entry)
        assert len(g.build_grid(a)) == 24

    def test_sasrec_factor_cache_is_always_built_with_the_legacy_transform(self) -> None:
        a = _args("--sasrec-transform", "exp", "--sasrec-temperatures", "2")
        spec = g.component_spec("sasrec", a)
        assert spec["reward"] == {"transform": "sigmoid", "temperature": 1.0}


class _StubCache:
    power = 1.0
    num_examples = 4
    num_items = 7

    def __init__(self, name: str) -> None:
        self.name, self.overrides = name, None

    def with_config(self, **overrides):
        view = _StubCache(self.name)
        view.overrides = overrides
        return view


class TestComposedCache:
    def test_entry_drives_the_sasrec_view_and_the_interaction_exponent(self) -> None:
        caches = {"sasrec": _StubCache("sasrec"), "interaction": _StubCache("interaction")}
        entry = {"combo": ("sasrec", "interaction"), "power": 1.0, "sasrec_transform": "exp",
                 "sasrec_temperature": 2.0, "interaction_exponent": 4.0}
        composed = g.composed_cache(entry, caches)
        assert composed.names == ["sasrec", "interaction"]
        assert composed.weights == [1.0, 4.0]
        assert composed.caches[0].overrides == {"transform": "exp", "temperature": 2.0}
        assert composed.caches[1] is caches["interaction"]
        assert caches["sasrec"].overrides is None  # the shared cache is untouched

    def test_interaction_alone_ignores_the_exponent(self) -> None:
        caches = {"interaction": _StubCache("interaction")}
        entry = {"combo": ("interaction",), "power": 16.0, "sasrec_transform": "-",
                 "sasrec_temperature": "-", "interaction_exponent": 4.0}
        composed = g.composed_cache(entry, caches)
        assert composed.weights == [1.0] and composed.power == 16.0


class TestMetricsCsv:
    def test_reward_forms_are_distinct_rows_and_legacy_rows_are_replaced(self, tmp_path: Path) -> None:
        path = tmp_path / "m.csv"
        # a row written before the reward-form columns existed
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[*g.base.CSV_KEY, "NDCG@5", "status"])
            w.writeheader()
            w.writerow({"dataset": "d", "combo": "sasrec", "power": "1.0", "objective": "db", "num_negatives": "10",
                        "seed": "42", "variant": "gflow", "split": "test", "target_grad_ratio": "1.0",
                        "NDCG@5": "0.1", "status": "completed"})
        ident = {"dataset": "d", "combo": "sasrec", "power": 1.0, "objective": "db", "num_negatives": 10,
                 "seed": 42, "variant": "gflow", "split": "test", "target_grad_ratio": 1.0}
        legacy = {"sasrec_transform": "sigmoid", "sasrec_temperature": 1.0, "interaction_exponent": 1.0}
        g.upsert_metrics_csv(path, [{**ident, **legacy, "NDCG@5": 0.2, "status": "completed"}])
        g.upsert_metrics_csv(path, [{**ident, "sasrec_transform": "exp", "sasrec_temperature": 2.0,
                                     "interaction_exponent": 1.0, "NDCG@5": 0.3, "status": "completed"}])
        rows = list(csv.DictReader(path.open()))
        assert sorted(r["NDCG@5"] for r in rows) == ["0.2", "0.3"]  # legacy row replaced, exp row added
        assert {r["sasrec_transform"] for r in rows} == {"sigmoid", "exp"}


class TestDatasets:
    def test_50m_uses_the_50m_grid_assets_and_its_own_outputs(self) -> None:
        a = _args("--dataset", "yambda_50m_listens")
        assert g.DATASET == "yambda_50m_listens"
        assert g.TIGER_CHECKPOINT == g.base.TIGER_CHECKPOINT
        assert g.SASREC_CHECKPOINT == g.base.SASREC_CHECKPOINT
        assert g.SFT_CONFIG == g.base.SFT_CONFIG
        assert "50m" in str(a.output_root) and "50m" in str(a.cache_root) and "50m" in g.METRICS_CSV_NAME
        # never mixes with the dense-cache 50m grid's outputs
        assert str(a.output_root) != "experiments/exp4-gflow-yambda50m"
        assert a.val_sample_fraction is None  # 50m monitors on the full validation split
        assert a.train_last_n == 8

    def test_default_dataset_is_restored_by_every_parse(self) -> None:
        _args("--dataset", "yambda_50m_listens")
        a = _args()
        assert g.DATASET == "yambda_500m_listens"
        assert g.TIGER_CHECKPOINT.endswith("yambda_500m_listens/best.pt")
        assert a.val_sample_fraction == 0.25

    def test_config_has_a_block_per_dataset(self) -> None:
        cfg = json.loads((PROJECT_ROOT / "configs/exp2-tuned-params/finetune_gflow.json").read_text())
        for spec in g.DATASETS.values():
            assert spec["PROTOCOL_BLOCK"] in cfg
