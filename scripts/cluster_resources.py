#!/usr/bin/env python
"""Node-type audit and Slurm placement for the base-matched runs.

The HSE cluster mixes node types whose host RAM ranges from 512 GB to 2 TB and
whose GPUs from V100 32 GB to H200 141 GB. A fine-tuning task holds the whole
dataset in host memory (plus copy-on-write copies in its dataloader workers),
so a large dataset does not fit every node type - and this Slurm does NOT
accept memory requests (--mem is not usable; the site documents only CPUs,
GPUs, --constraint and arrays), so nothing stops it from packing several of
our array tasks onto one node. Three yambda_500m tasks once landed on one node
that way; it swap-thrashed and was killed.

With memory out of Slurm's hands, two things keep a node from overflowing:

  fit      a dataset may only use node types whose RAM holds ONE task
           (--constraint);
  packing  how many of OUR tasks can share a node. A task takes one GPU, so a
           node never runs more tasks than it has GPUs:
             GPU-bound  gpus x task_memory <= node RAM: even a node full of
                        our tasks fits - nothing else is needed;
             RAM-bound  otherwise the tasks are pinned to named nodes
                        (--nodelist) in LANES: a node gets k =
                        floor(RAM / task_memory) lanes, every lane is its own
                        array with %1, and everything sent to a lane is
                        chained after the lane's previous job. At most k of
                        our tasks then ever run on that node.
           A RAM-bound dataset must not share its node types with any other
           dataset (their tasks would land on the same nodes uncounted).

`cluster_resources` in configs/exp2-tuned-params/finetune_base_matched.json
holds the node inventory and, per dataset, the measured task memory, the
allowed node types and the walltime.

  python scripts/cluster_resources.py audit [--live]
  python scripts/cluster_resources.py plan [--live] --datasets yambda_50m_listens yambda_500m_listens

`plan` prints one line per array group for scripts/slurm/submit_base_matched_recalc.sh:
  <sbatch flags> TAB <datasets> TAB free
  <sbatch flags> TAB <datasets> TAB lanes TAB <node> <node> ...   (a node repeated k times)
--live reads the node inventory (names, GPUs, state) from `sinfo` instead of the
static table taken from https://hpc.hse.ru/hardware/hpc-cluster; lanes need it
for a real submission because they name nodes. RAM is a different matter: this
Slurm does not track memory, so `sinfo` reports none (RealMemory 1 MB) and the
RAM of a live node comes from the static table of its node type (matched by
GPU count where a type has variants, e.g. type_h 8 x H200 / 2 TB vs 4 x H200 /
1.5 TB). Nodes that are down or drained are left out.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_CONFIG = PROJECT_ROOT / "configs/exp2-tuned-params/finetune_base_matched.json"
# The OS, the page cache and other users' jobs need their share, and Slurm's
# RealMemory sits below the nominal RAM: only this fraction of a node's RAM is
# planned for our tasks.
USABLE_RAM_FRACTION = 0.95
# Below this, a RealMemory reported by sinfo is a placeholder (Slurm does not
# track memory on this cluster: every node reports 1 MB), not a node size.
MIN_PLAUSIBLE_RAM_GB = 16.0
UNUSABLE_STATES = re.compile(r"^(down|drain|fail|maint|unk|inval|power|reboot|future|not_resp)", re.IGNORECASE)


def load_resources(path: Path = PROTOCOL_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["cluster_resources"]


# --------------------------------------------------------------------------- #
# Node inventory: one record per node
# --------------------------------------------------------------------------- #
def static_nodes(resources: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The hardware page's table expanded to nodes with placeholder names."""
    nodes: list[dict[str, Any]] = []
    for kind, spec in resources["node_types"].items():
        counter = 0
        for group in spec.get("variants") or [spec]:
            for _ in range(int(group["nodes"])):
                counter += 1
                nodes.append({"name": f"{kind}-node{counter}", "type": kind, "ram_gb": float(group["ram_gb"]),
                              "usable_ram_gb": float(group["ram_gb"]) * USABLE_RAM_FRACTION,
                              "gpus": int(group["gpus"]), "gpu": spec.get("gpu", ""),
                              "gpu_mem_gb": spec.get("gpu_mem_gb"), "source": "static"})
    return nodes


def static_ram_gb(spec: Mapping[str, Any] | None, gpus: int) -> float:
    """RAM of a node of this type per the hardware table: the variant with the
    same GPU count, else the smallest one."""
    if not spec:
        return 0.0
    variants = spec.get("variants") or [spec]
    matching = [v for v in variants if int(v.get("gpus", -1)) == int(gpus)]
    return float(min(matching or variants, key=lambda v: float(v["ram_gb"]))["ram_gb"])


def parse_sinfo(text: str, resources: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """`sinfo -h -N -o '%N|%f|%m|%G|%T'` -> records of the usable nodes.
    RealMemory (MB) is used when it is a real figure, else the static RAM of
    the node's type (see MIN_PLAUSIBLE_RAM_GB)."""
    static_types = (resources or {}).get("node_types", {})
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or parts[0] in seen:
            continue
        match = re.search(r"type_[a-z0-9_']+", parts[1])
        if not match:
            continue
        try:
            ram_gb = float(re.sub(r"[^0-9.]", "", parts[2])) / 1024.0
        except ValueError:
            continue
        seen.add(parts[0])
        if len(parts) > 4 and UNUSABLE_STATES.match(parts[4]):
            continue
        gres = re.search(r"gpu:(?:([A-Za-z0-9_]+):)?(\d+)", parts[3])
        kind = match.group(0)
        gpus = int(gres.group(2)) if gres else 0
        ram_source = "sinfo"
        if ram_gb < MIN_PLAUSIBLE_RAM_GB:
            ram_gb, ram_source = static_ram_gb(static_types.get(kind), gpus), "hardware table"
        nodes.append({"name": parts[0], "type": kind, "ram_gb": ram_gb,
                      "usable_ram_gb": ram_gb * USABLE_RAM_FRACTION, "ram_source": ram_source,
                      "gpus": gpus,
                      "gpu": static_types.get(kind, {}).get("gpu") or ((gres.group(1) or "").upper() if gres else ""),
                      "gpu_mem_gb": static_types.get(kind, {}).get("gpu_mem_gb"), "source": "sinfo"})
    return nodes


def live_nodes(resources: Mapping[str, Any], partition: str) -> list[dict[str, Any]]:
    out = subprocess.run(["sinfo", "-h", "-N", "-p", partition, "-o", "%N|%f|%m|%G|%T"],
                         capture_output=True, text=True, check=True).stdout
    nodes = parse_sinfo(out, resources)
    if not nodes:
        raise RuntimeError("sinfo returned no nodes with a type_* feature")
    return nodes


# --------------------------------------------------------------------------- #
# Fit and packing
# --------------------------------------------------------------------------- #
def tasks_that_fit(node: Mapping[str, Any], mem_gb: float) -> int:
    """Our tasks a node can hold: one GPU each, and their memory within RAM."""
    if int(node.get("gpus") or 0) <= 0 or mem_gb <= 0:
        return 0
    return min(int(node["gpus"]), int(float(node["usable_ram_gb"]) // float(mem_gb)))


def is_gpu_bound(node: Mapping[str, Any], mem_gb: float) -> bool:
    """True when the node's GPU count alone keeps our tasks within its RAM."""
    return int(node.get("gpus") or 0) > 0 and tasks_that_fit(node, mem_gb) >= int(node["gpus"])


def dataset_nodes(spec: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [n for n in nodes if n["type"] in spec["node_types"]]


def needs_lanes(spec: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> bool:
    return any(not is_gpu_bound(n, float(spec["mem_gb"])) for n in dataset_nodes(spec, nodes))


def lanes(spec: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    """Node names, each repeated once per task it can hold (round-robin over
    nodes, so consecutive lanes sit on different nodes)."""
    eligible = [(n["name"], tasks_that_fit(n, float(spec["mem_gb"]))) for n in dataset_nodes(spec, nodes)]
    out: list[str] = []
    for slot in range(max((k for _, k in eligible), default=0)):
        out.extend(name for name, k in eligible if k > slot)
    return out


def audit_dataset(name: str, spec: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mem_gb, gpu_mem_gb = float(spec["mem_gb"]), float(spec.get("min_gpu_mem_gb", 0))
    rows, errors, warnings = [], [], []
    for kind in spec["node_types"]:
        members = [n for n in nodes if n["type"] == kind]
        if not members:
            # a whole type can be down or drained on a live cluster
            warnings.append(f"{name}: no usable {kind} node in the inventory right now")
            continue
        smallest = min(members, key=lambda n: n["usable_ram_gb"])
        if tasks_that_fit(smallest, mem_gb) < 1:
            errors.append(f"{name}: {kind} cannot hold one task ({smallest['usable_ram_gb']:.0f} GB usable RAM "
                          f"< {mem_gb:.0f} GB{'' if smallest['gpus'] else ', no GPU'})")
        if smallest.get("gpu_mem_gb") is not None and float(smallest["gpu_mem_gb"]) < gpu_mem_gb:
            errors.append(f"{name}: {kind} GPUs have {smallest['gpu_mem_gb']} GB < {gpu_mem_gb:.0f} GB required")
        rows.append({"node_type": kind, "nodes": len(members), "gpu": smallest.get("gpu", ""),
                     "gpu_bound": all(is_gpu_bound(n, mem_gb) for n in members),
                     "concurrent_tasks": sum(tasks_that_fit(n, mem_gb) for n in members)})
    if not rows:
        errors.append(f"{name}: none of its node types {list(spec['node_types'])} has a usable node")
    models = {r["gpu"] for r in rows if r["gpu"]}
    if len(models) > 1:
        warnings.append(f"{name}: node types mix GPU models {sorted(models)} - training times are not comparable")
    return {"dataset": name, "mem_gb": mem_gb, "rows": rows, "errors": errors, "warnings": warnings,
            "packing": "lanes" if needs_lanes(spec, nodes) else "free",
            "concurrent_tasks": sum(r["concurrent_tasks"] for r in rows)}


def audit(resources: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reports = [audit_dataset(name, spec, nodes) for name, spec in resources["datasets"].items()]
    # Datasets that share a node type also share its nodes: a node full of the
    # LARGEST of them must still fit, or the GPU-bound guarantee is void.
    for node in nodes:
        sharing = [(name, float(spec["mem_gb"])) for name, spec in resources["datasets"].items()
                   if node["type"] in spec["node_types"] and not needs_lanes(spec, nodes)]
        if len(sharing) > 1 and int(node["gpus"]) * max(m for _, m in sharing) > float(node["usable_ram_gb"]):
            largest = max(sharing, key=lambda item: item[1])[0]
            message = (f"{largest}: {node['gpus']} tasks on a {node['type']} node exceed its RAM, and "
                       f"{sorted(n for n, _ in sharing if n != largest)} share that node type")
            for report in reports:
                if report["dataset"] == largest and message not in report["errors"]:
                    report["errors"].append(message)
    # A RAM-bound dataset counts its tasks per node; tasks of another dataset
    # on the same nodes would not be counted.
    for report in reports:
        if report["packing"] != "lanes":
            continue
        mine = set(resources["datasets"][report["dataset"]]["node_types"])
        for other, spec in resources["datasets"].items():
            shared = mine & set(spec["node_types"])
            if other != report["dataset"] and shared:
                report["errors"].append(
                    f"{report['dataset']}: RAM-bound, but shares node type(s) {sorted(shared)} with {other}")
    return reports


# --------------------------------------------------------------------------- #
# sbatch flags and array groups
# --------------------------------------------------------------------------- #
def sbatch_flags(spec: Mapping[str, Any]) -> list[str]:
    return [f"--time={spec['time']}", "--constraint=" + "|".join(spec["node_types"])]


def plan(resources: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]],
         datasets: Sequence[str]) -> list[dict[str, Any]]:
    """Array groups: datasets with identical flags share one array when they are
    GPU-bound; a RAM-bound dataset always gets its own, laned arrays."""
    groups: list[dict[str, Any]] = []
    for name in datasets:
        if name not in resources["datasets"]:
            raise KeyError(f"no cluster_resources entry for dataset {name!r}")
        spec = resources["datasets"][name]
        flags = sbatch_flags(spec)
        if needs_lanes(spec, nodes):
            groups.append({"flags": flags, "datasets": [name], "mode": "lanes", "lanes": lanes(spec, nodes)})
            continue
        for group in groups:
            if group["mode"] == "free" and group["flags"] == flags:
                group["datasets"].append(name)
                break
        else:
            groups.append({"flags": flags, "datasets": [name], "mode": "free", "lanes": []})
    return groups


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_audit(resources: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> int:
    live = nodes[0]["source"] == "sinfo"
    print(f"node inventory ({'live sinfo' if live else 'static: ' + resources['source']}):")
    if live and any(n.get("ram_source") == "hardware table" for n in nodes):
        print("  (sinfo reports no memory - Slurm does not track it here: RAM per node type is taken from the "
              "hardware table; node names, GPU counts and states are live)")
    print(f"  {'type':<9}{'nodes':>6}{'RAM GB':>14}  GPUs per node")
    for kind in dict.fromkeys(n["type"] for n in nodes):
        members = [n for n in nodes if n["type"] == kind]
        rams = sorted({round(n["ram_gb"]) for n in members})
        gpus = sorted({n["gpus"] for n in members})
        gpu = f"{'/'.join(map(str, gpus))} x {members[0]['gpu']} {members[0].get('gpu_mem_gb') or '?'} GB" if gpus[-1] else "-"
        print(f"  {kind:<9}{len(members):>6}{'/'.join(map(str, rams)):>14}  {gpu}")
    failed = 0
    print("\ndatasets (one task = 1 GPU + the host memory below; Slurm does not track memory here):")
    for report in audit(resources, nodes):
        spec = resources["datasets"][report["dataset"]]
        print(f"  {report['dataset']}: ~{int(report['mem_gb'])} GB per task | {' '.join(sbatch_flags(spec))}")
        for row in report["rows"]:
            how = "GPU-bound: a full node of tasks fits" if row["gpu_bound"] else "RAM-BOUND: pinned lanes"
            print(f"      {row['node_type']:<9}{row['nodes']:>3} nodes, {row['concurrent_tasks']:>3} tasks at once"
                  f"   ({row['gpu']}; {how})")
        if report["packing"] == "lanes":
            print(f"      lanes: {' '.join(lanes(spec, nodes))}")
        for message in report["warnings"]:
            print(f"      WARNING {message}")
        for message in report["errors"]:
            print(f"      ERROR   {message}")
        failed += len(report["errors"])
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--live", action="store_true", help="read the node inventory from sinfo")
        p.add_argument("--partition", default="rocky")
        if name == "plan":
            p.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args(argv)
    resources = load_resources()
    nodes = live_nodes(resources, args.partition) if args.live else static_nodes(resources)
    if args.command == "audit":
        return 1 if print_audit(resources, nodes) else 0
    errors = [m for report in audit(resources, nodes) if report["dataset"] in args.datasets
              for m in report["errors"]]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    for group in plan(resources, nodes, args.datasets):
        fields = [" ".join(group["flags"]), " ".join(group["datasets"]), group["mode"]]
        if group["mode"] == "lanes":
            fields.append(" ".join(group["lanes"]))
        print("\t".join(fields))
    return 0


if __name__ == "__main__":
    sys.exit(main())
