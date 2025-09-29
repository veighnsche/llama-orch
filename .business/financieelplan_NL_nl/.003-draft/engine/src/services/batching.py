"""Batching semantics (scaffold).

This module defines minimal helpers to interpret throughput samples from
`inputs/facts/tps_model_gpu.csv` and normalize them to an
"effective tokens/sec per instance" figure suitable for capacity planning
and autoscaler simulation.

Aligned with `.specs/proposals/BATCHING.md`:
- Allowed measurement_type values: `single_stream | batched_online | offline`.
- Legacy aliases mapped: `aggregate -> batched_online`, `per_user_stream -> single_stream`.

Note: In v0.1 an "instance" is assumed to be a single GPU unless specified
otherwise by the pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


def map_legacy_measurement_type(s: str) -> str:
    s_l = (s or "").strip().lower()
    if s_l == "aggregate":
        return "batched_online"
    if s_l == "per_user_stream":
        return "single_stream"
    if s_l in ("single_stream", "batched_online", "offline"):
        return s_l
    # Default to batched_online for Public Tap economics
    return "batched_online"


@dataclass(frozen=True)
class BatchingConfig:
    slots_total_default: int = 32
    batching_efficiency_pct: float = 70.0  # 0..100, aggregate efficiency vs ideal
    offline_penalty_pct: float = 40.0      # use only if offline must be estimated


def _clamp_nonneg(x: float) -> float:
    return x if x > 0 else 0.0


def effective_tps_per_instance(
    measurement_type: str,
    throughput_tokens_per_sec: float,
    gpu_count: int = 1,
    batch: Optional[int] = None,
    cfg: Optional[BatchingConfig] = None,
) -> float:
    """Return an aggregate tokens/sec per instance suitable for capacity planning.

    - For `batched_online`: assume reported throughput is aggregate across the
      measured unit; normalize by gpu_count to get per-instance.
    - For `single_stream`: estimate aggregate by multiplying with `slots_total`
      and an efficiency factor, then normalize per instance.
    - For `offline`: not representative for latency-sensitive serving; return a
      conservative value (apply a penalty) or 0 if you prefer to ignore.
    """
    cfg = cfg or BatchingConfig()
    mt = map_legacy_measurement_type(measurement_type)
    tps = _clamp_nonneg(float(throughput_tokens_per_sec or 0.0))
    g = max(int(gpu_count or 1), 1)

    if mt == "batched_online":
        per_instance = tps / g
        return _clamp_nonneg(per_instance)

    if mt == "single_stream":
        slots = int(batch) if (batch is not None and int(batch) > 0) else max(int(cfg.slots_total_default), 1)
        eff = max(min(cfg.batching_efficiency_pct / 100.0, 1.0), 0.0)
        per_instance = (tps * slots * eff) / g
        return _clamp_nonneg(per_instance)

    # mt == "offline" or unknown
    penalty = max(min(cfg.offline_penalty_pct / 100.0, 1.0), 0.0)
    per_instance = (tps * penalty) / g
    return _clamp_nonneg(per_instance)
