"""Node-type audit of the base-matched runs (scripts/cluster_resources.py).

The cluster's Slurm takes no memory requests, so host memory is handled by
WHERE tasks may run: every dataset names node types whose RAM holds a task,
and a dataset whose tasks would overflow a node when Slurm packs one per GPU
is pinned to named nodes in lanes."""

import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cluster_resources as cr  # noqa: E402
import probe_finetune_memory as probe  # noqa: E402

PROTOCOL = json.loads((PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json").read_text())
SINFO = "\n".join([
    "cn-001|type_a,v100|772000|gpu:v100:4",
    "cn-002|type_a,v100|772000|gpu:v100:4",
    "cn-002|type_a,v100|772000|gpu:v100:4",      # a node listed once per partition
    "cn-040|type_e,a100|1031000|gpu:a100:8",
    "cn-050|type_h|2063000|gpu:h200:8",
    "cn-051|type_h|2063000|gpu:h200:8",
    "cn-052|type_h|1547000|gpu:h200:4",          # H': less RAM, fewer GPUs
    "cn-030|type_d|772000|(null)",
])


@pytest.fixture()
def resources() -> dict:
    return copy.deepcopy(cr.load_resources())


class TestProtocolResources:
    def test_every_dataset_of_the_protocol_has_resources(self, resources) -> None:
        assert set(resources["datasets"]) == set(PROTOCOL["datasets"])
        for name, spec in resources["datasets"].items():
            assert spec["mem_gb"] > 0 and spec["node_types"] and spec["time"], name
            assert spec["mem_basis"] and spec["node_basis"], name

    def test_the_audit_passes(self, resources) -> None:
        for report in cr.audit(resources, cr.static_nodes(resources)):
            assert report["errors"] == [], report["errors"]
            assert report["warnings"] == [], report["warnings"]  # one GPU model per dataset
            assert report["concurrent_tasks"] >= 1

    def test_large_datasets_stay_off_nodes_that_cannot_hold_them(self, resources) -> None:
        nodes = cr.static_nodes(resources)
        for name, spec in resources["datasets"].items():
            for node in cr.dataset_nodes(spec, nodes):
                assert node["usable_ram_gb"] >= spec["mem_gb"], (name, node["type"])
                assert node["gpus"] > 0, (name, node["type"])  # type_d has no GPU
        big = resources["datasets"]["yambda_500m_listens"]
        assert big["mem_gb"] >= 4 * resources["datasets"]["yambda_50m_listens"]["mem_gb"]
        assert not {"type_a", "type_c", "type_e", "type_f"} & set(big["node_types"])

    def test_only_the_ram_bound_dataset_is_pinned_and_it_owns_its_node_types(self, resources) -> None:
        nodes = cr.static_nodes(resources)
        packing = {r["dataset"]: r["packing"] for r in cr.audit(resources, nodes)}
        assert packing == {"beauty": "free", "instruments": "free", "beauty_2023": "free",
                           "yambda_50m_listens": "free", "yambda_500m_listens": "lanes"}
        mine = set(resources["datasets"]["yambda_500m_listens"]["node_types"])
        for name, spec in resources["datasets"].items():
            if name != "yambda_500m_listens":
                assert not mine & set(spec["node_types"]), name

    def test_no_sbatch_flag_asks_for_memory(self, resources) -> None:
        for spec in resources["datasets"].values():
            flags = cr.sbatch_flags(spec)
            assert not any(f.startswith("--mem") for f in flags)
            assert flags == [f"--time={spec['time']}", "--constraint=" + "|".join(spec["node_types"])]


class TestFitAndPacking:
    def test_tasks_that_fit_is_bounded_by_ram_and_by_gpus(self) -> None:
        node = {"gpus": 8, "usable_ram_gb": 972.8}
        assert cr.tasks_that_fit(node, 1300) == 0
        assert cr.tasks_that_fit(node, 600) == 1
        assert cr.tasks_that_fit(node, 200) == 4
        assert cr.tasks_that_fit(node, 32) == 8     # one GPU per task
        assert cr.tasks_that_fit({"gpus": 0, "usable_ram_gb": 700}, 32) == 0  # CPU-only type_d

    def test_gpu_bound_means_a_full_node_of_tasks_fits(self) -> None:
        assert cr.is_gpu_bound({"gpus": 4, "usable_ram_gb": 729.6}, 96) is True     # 4 x 96 = 384
        assert cr.is_gpu_bound({"gpus": 8, "usable_ram_gb": 972.8}, 200) is False   # 8 x 200 = 1600
        assert cr.is_gpu_bound({"gpus": 4, "usable_ram_gb": 1459.2}, 200) is True   # type_b

    def test_a_node_type_that_cannot_hold_one_task_is_an_error(self, resources) -> None:
        spec = {**resources["datasets"]["yambda_500m_listens"], "node_types": ["type_e", "type_f", "type_h"]}
        report = cr.audit_dataset("yambda_500m_listens", spec, cr.static_nodes(resources))
        errors = "\n".join(report["errors"])
        assert "type_e cannot hold one task" in errors and "type_f cannot hold one task" in errors
        assert "type_h" not in errors
        assert any("mix GPU models" in w for w in report["warnings"])

    def test_gpu_memory_requirement(self, resources) -> None:
        spec = {"mem_gb": 32, "min_gpu_mem_gb": 80, "node_types": ["type_a", "type_e"], "time": "1:00:00"}
        report = cr.audit_dataset("x", spec, cr.static_nodes(resources))
        assert any("type_a GPUs have 32 GB < 80 GB" in e for e in report["errors"])
        assert not any("type_e GPUs" in e for e in report["errors"])

    def test_a_ram_bound_dataset_may_not_share_its_node_types(self, resources) -> None:
        resources["datasets"]["beauty_2023"]["node_types"] = ["type_h"]
        reports = {r["dataset"]: r for r in cr.audit(resources, cr.static_nodes(resources))}
        assert any("shares node type(s) ['type_h'] with beauty_2023" in e
                   for e in reports["yambda_500m_listens"]["errors"])

    def test_datasets_sharing_a_node_type_must_fit_together(self, resources) -> None:
        resources["datasets"]["yambda_50m_listens"]["mem_gb"] = 250  # 4 x 250 GB > a 768 GB node
        resources["datasets"]["yambda_50m_listens"]["node_types"] = ["type_a"]
        resources["datasets"]["beauty"]["node_types"] = ["type_a"]
        nodes = cr.static_nodes(resources)
        # on its own it would simply become RAM-bound; together with beauty it is an error either way
        reports = {r["dataset"]: r for r in cr.audit(resources, nodes)}
        assert reports["yambda_50m_listens"]["packing"] == "lanes"
        assert any("shares node type(s) ['type_a'] with beauty" in e
                   for e in reports["yambda_50m_listens"]["errors"])


class TestLanes:
    def test_one_lane_per_task_a_node_can_hold(self, resources) -> None:
        nodes = cr.parse_sinfo(SINFO, resources)
        spec = {**resources["datasets"]["yambda_500m_listens"], "mem_gb": 1300}
        assert cr.lanes(spec, nodes) == ["cn-050", "cn-051", "cn-052"]           # 1.3 TB: one each
        spec["mem_gb"] = 700
        assert cr.lanes(spec, nodes) == ["cn-050", "cn-051", "cn-052", "cn-050", "cn-051", "cn-052"]
        spec["mem_gb"] = 900                                                     # H' (1.5 TB) holds one
        assert cr.lanes(spec, nodes) == ["cn-050", "cn-051", "cn-052", "cn-050", "cn-051"]

    def test_live_inventory_from_sinfo(self, resources) -> None:
        nodes = cr.parse_sinfo(SINFO, resources)
        assert [n["name"] for n in nodes if n["type"] == "type_a"] == ["cn-001", "cn-002"]
        h = [n for n in nodes if n["type"] == "type_h"]
        assert [n["gpus"] for n in h] == [8, 8, 4]
        assert h[2]["ram_gb"] == pytest.approx(1547000 / 1024)
        assert h[0]["gpu_mem_gb"] == 141                # GPU memory comes from the static table
        assert [n["gpus"] for n in nodes if n["type"] == "type_d"] == [0]


class TestSinfoWithoutMemory:
    """The cluster's Slurm does not track memory: sinfo reports RealMemory 1 MB
    for every node. RAM must then come from the hardware table of the node's
    type, while names, GPU counts and states stay live."""

    TEXT = "\n".join([
        "cn-001|type_a,rocky|1|gpu:v100:4|mixed",
        "cn-002|type_a,rocky|1|gpu:v100:3|idle",        # a node with a GPU missing
        "cn-003|type_a,rocky|1|gpu:v100:4|drained",     # unusable
        "cn-020|type_b,rocky|1|gpu:v100:4|idle",
        "cn-040|type_e,rocky|1|gpu:a100:8|down*",       # unusable
        "cn-048|type_h,rocky|1|gpu:h200:8|mixed",
        "cn-049|type_h,rocky|1|gpu:h200:8|allocated",
        "cn-050|type_h,rocky|1|gpu:h200:4|mixed",       # H': 4 GPUs -> the 1.5 TB variant
    ])

    def test_ram_comes_from_the_hardware_table(self, resources) -> None:
        nodes = {n["name"]: n for n in cr.parse_sinfo(self.TEXT, resources)}
        assert set(nodes) == {"cn-001", "cn-002", "cn-020", "cn-048", "cn-049", "cn-050"}  # drained / down left out
        assert nodes["cn-001"]["ram_gb"] == 768 and nodes["cn-002"]["gpus"] == 3
        assert nodes["cn-020"]["ram_gb"] == 1536
        assert (nodes["cn-048"]["ram_gb"], nodes["cn-050"]["ram_gb"]) == (2048, 1536)
        assert all(n["ram_source"] == "hardware table" and n["usable_ram_gb"] > 0 for n in nodes.values())
        assert nodes["cn-048"]["gpu"] == "H200"

    def test_the_audit_passes_and_lanes_name_real_nodes(self, resources) -> None:
        nodes = cr.parse_sinfo(self.TEXT, resources)
        reports = {r["dataset"]: r for r in cr.audit(resources, nodes)}
        for name in ("beauty", "instruments", "yambda_50m_listens", "beauty_2023"):
            assert reports[name]["errors"] == [] and reports[name]["packing"] == "free", name
        # no type_c node is up in this listing: a warning, not a reason to refuse (type_a is there)
        assert any("no usable type_c node" in w for w in reports["beauty"]["warnings"])
        big = reports["yambda_500m_listens"]
        assert big["errors"] == [] and big["packing"] == "lanes" and big["concurrent_tasks"] == 3
        assert cr.lanes(resources["datasets"]["yambda_500m_listens"], nodes) == ["cn-048", "cn-049", "cn-050"]

    def test_a_dataset_whose_node_types_are_all_gone_is_an_error(self, resources) -> None:
        nodes = [n for n in cr.parse_sinfo(self.TEXT, resources) if n["type"] != "type_h"]
        report = cr.audit_dataset("yambda_500m_listens", resources["datasets"]["yambda_500m_listens"], nodes)
        assert any("none of its node types" in e for e in report["errors"])

    def test_a_real_memory_figure_is_still_used(self, resources) -> None:
        nodes = cr.parse_sinfo("cn-050|type_h|1547000|gpu:h200:4|idle", resources)
        assert nodes[0]["ram_gb"] == pytest.approx(1547000 / 1024) and nodes[0]["ram_source"] == "sinfo"

    def test_static_ram_by_gpu_count(self, resources) -> None:
        spec = resources["node_types"]["type_h"]
        assert cr.static_ram_gb(spec, 8) == 2048 and cr.static_ram_gb(spec, 4) == 1536
        assert cr.static_ram_gb(spec, 6) == 1536                       # unknown variant: the smaller one
        assert cr.static_ram_gb(resources["node_types"]["type_a"], 3) == 768
        assert cr.static_ram_gb(None, 4) == 0.0                        # unknown node type: holds nothing


class TestPlan:
    def test_free_datasets_with_identical_flags_share_an_array(self, resources) -> None:
        nodes = cr.static_nodes(resources)
        groups = cr.plan(resources, nodes, ["beauty", "instruments", "yambda_500m_listens", "yambda_50m_listens"])
        by_members = {tuple(g["datasets"]): g for g in groups}
        assert by_members[("beauty", "instruments")]["mode"] == "free"
        assert by_members[("yambda_50m_listens",)]["mode"] == "free"    # other walltime than beauty
        big = by_members[("yambda_500m_listens",)]
        assert big["mode"] == "lanes" and len(big["lanes"]) == 3
        with pytest.raises(KeyError):
            cr.plan(resources, nodes, ["unknown"])

    def test_cli_prints_the_plan_and_refuses_an_impossible_one(self, resources, monkeypatch, capsys) -> None:
        assert cr.main(["plan", "--datasets", "beauty", "yambda_500m_listens"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[0].split("\t") == ["--time=23:59:00 --constraint=type_a|type_c", "beauty", "free"]
        fields = lines[1].split("\t")
        assert fields[:3] == ["--time=3-00:00:00 --constraint=type_h", "yambda_500m_listens", "lanes"]
        assert len(fields[3].split()) == 3 and "--mem" not in lines[1]
        resources["datasets"]["yambda_500m_listens"]["node_types"] = ["type_f"]
        monkeypatch.setattr(cr, "load_resources", lambda path=None: resources)
        assert cr.main(["plan", "--datasets", "yambda_500m_listens"]) == 1
        assert "type_f cannot hold one task" in capsys.readouterr().err


def test_probe_recommendation() -> None:
    # build peak dominates / main + workers dominate; 30% margin, rounded up to 8 GB
    assert probe.recommended_mem_gb(15.6, 7.4, 3.0) == 24
    assert probe.recommended_mem_gb(15.6, 7.4, 43.2) == 72
    assert probe.recommended_mem_gb(160, 74, 432) == 664
