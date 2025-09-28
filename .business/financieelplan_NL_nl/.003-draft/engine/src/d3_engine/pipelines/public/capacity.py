"""Public capacity planning (scaffold)."""
from __future__ import annotations
from typing import Tuple

from ...services.autoscaling import (
    instances_needed,
    planner_instances_needed,
    cap_per_instance_tokens_per_hour,
)
from ...services.batching import (
    effective_tps_per_instance,
    BatchingConfig,
)

__all__ = [
    "instances_needed",
    "planner_instances_needed",
    "cap_per_instance_tokens_per_hour",
    "effective_tps",
    "plan_capacity_row",
]


def effective_tps(row: dict, cfg: BatchingConfig | None = None) -> float:
    """Compute effective aggregate tokens/sec per instance from a TPS row.

    Expects a dict shaped like a CSV row from `tps_model_gpu.csv` with keys:
    - measurement_type
    - throughput_tokens_per_sec
    - gpu_count (optional; defaults to 1)
    - batch (optional; used for single_stream estimation)
    """
    mt = (row.get("measurement_type") or "")
    tps = float(row.get("throughput_tokens_per_sec") or 0.0)
    try:
        g = int(row.get("gpu_count") or 1)
    except Exception:
        g = 1
    b_raw = row.get("batch")
    try:
        b = int(b_raw) if (b_raw is not None and str(b_raw).strip() != "") else None
    except Exception:
        b = None
    return effective_tps_per_instance(mt, tps, gpu_count=g, batch=b, cfg=cfg)


def plan_capacity_row(
    model: str,
    gpu: str,
    avg_tokens_per_hour: float,
    peak_tokens_per_hour: float,
    tps: float,
    target_utilization_pct: float,
    min_instances_per_model: int,
    max_instances_per_model: int,
) -> Tuple[list[str], list[str]]:
    """Return (headers, row) for public_tap_capacity_plan.csv using planner logic.

    This is a pure helper; writers are responsible for file I/O.
    """
    cap_per_inst = cap_per_instance_tokens_per_hour(tps, target_utilization_pct)
    instances, violation, _ = planner_instances_needed(
        peak_tokens_per_hour,
        tps,
        target_utilization_pct,
        min_instances_per_model,
        max_instances_per_model,
    )
    headers = [
        "model",
        "gpu",
        "avg_tokens_per_hour",
        "peak_tokens_per_hour",
        "tps",
        "cap_tokens_per_hour_per_instance",
        "instances_needed",
        "target_utilization_pct",
        "capacity_violation",
    ]
    row = [
        model,
        gpu,
        f"{avg_tokens_per_hour}",
        f"{peak_tokens_per_hour}",
        f"{tps}",
        f"{cap_per_inst}",
        f"{instances}",
        f"{target_utilization_pct}",
        "True" if violation else "False",
    ]
    return headers, row
