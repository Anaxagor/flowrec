"""Explicit compute accounting (spec §9) and latency benchmarking (spec §10).

Two cost regimes are distinguished by ``cache_mode``:
  - ``offline_cache``: all full-catalog TIGER/SASRec scores were precomputed;
    counters report policy arithmetic; cache construction is accounted in the
    cache-building stages' own counter rows.
  - ``online_serving``: counters report genuine model calls and wall time.

Never label cached-arithmetic time as serving latency: rows carry both
``wall_time_policy_eval_s`` (cached arithmetic) and the benchmark utility's
``end_to_end_runtime`` measurements, reported separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields

import numpy as np

COUNTER_FIELDS = [
    "tiger_encoder_forwards",
    "tiger_decoder_forward_batches",
    "tiger_prefix_state_evals",
    "tiger_token_logit_evals",
    "tiger_item_trajectory_scores",
    "sasrec_user_forwards",
    "sasrec_item_scores",
    "sasrec_full_catalog_passes",
    "prefix_mass_nodes_processed",
    "sampled_trajectory_transitions",
    "num_add_actions",
    "num_delete_actions",
    "num_stop_actions",
    "cache_hits",
    "cache_misses",
]

TIME_FIELDS = [
    "wall_time_precompute_s",
    "wall_time_policy_eval_s",
    "wall_time_end_to_end_s",
]


@dataclass
class ComputeCounters:
    cache_mode: str = "offline_cache"  # offline_cache | online_serving
    tiger_encoder_forwards: float = 0.0
    tiger_decoder_forward_batches: float = 0.0
    tiger_prefix_state_evals: float = 0.0
    tiger_token_logit_evals: float = 0.0
    tiger_item_trajectory_scores: float = 0.0
    sasrec_user_forwards: float = 0.0
    sasrec_item_scores: float = 0.0
    sasrec_full_catalog_passes: float = 0.0
    prefix_mass_nodes_processed: float = 0.0
    sampled_trajectory_transitions: float = 0.0
    num_add_actions: float = 0.0
    num_delete_actions: float = 0.0
    num_stop_actions: float = 0.0
    cache_hits: float = 0.0
    cache_misses: float = 0.0
    wall_time_precompute_s: float = 0.0
    wall_time_policy_eval_s: float = 0.0
    wall_time_end_to_end_s: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    n_users: int = 0

    def add(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + float(v))

    def merge(self, other: "ComputeCounters") -> None:
        for f in fields(self):
            if f.name == "cache_mode":
                continue
            if f.name.startswith("peak_"):
                setattr(self, f.name, max(getattr(self, f.name), getattr(other, f.name)))
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def record_peaks(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_memory_mb = max(
                    self.peak_gpu_memory_mb,
                    torch.cuda.max_memory_allocated() / 2**20)
        except Exception:
            pass
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux kilobytes
            rss_mb = rss_kb / 2**20 if rss_kb > 1 << 32 else rss_kb / 1024
            self.peak_cpu_memory_mb = max(self.peak_cpu_memory_mb, rss_mb)
        except Exception:
            pass


class LatencyBenchmark:
    """End-to-end latency measurement with CUDA discipline (spec §10)."""

    def __init__(self, device: str = "cpu", warmup: int = 3, reps: int = 20):
        self.device = str(device)
        self.warmup = warmup
        self.reps = reps

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()

    def run(self, fn, *, label: str, batch_size: int, caches_enabled: bool) -> dict:
        for _ in range(self.warmup):
            fn()
        samples = []
        for _ in range(self.reps):
            self._sync()
            t0 = time.perf_counter()
            fn()
            self._sync()
            samples.append(time.perf_counter() - t0)
        arr = np.asarray(samples)
        gpu_name = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return {
            "label": label,
            "batch_size": int(batch_size),
            "caches_enabled": bool(caches_enabled),
            "device": self.device,
            "gpu_model": gpu_name,
            "reps": self.reps,
            "median_s": float(np.median(arr)),
            "p90_s": float(np.percentile(arr, 90)),
            "p95_s": float(np.percentile(arr, 95)),
            "p99_s": float(np.percentile(arr, 99)),
            "mean_s": float(arr.mean()),
        }
