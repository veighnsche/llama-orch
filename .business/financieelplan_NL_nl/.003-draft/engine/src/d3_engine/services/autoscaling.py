"""Autoscaling helpers (scaffold)."""
import math


def instances_needed(peak_tokens_per_hour: float, tps: float, target_utilization_pct: float) -> int:
    cap = max(tps, 1e-9) * 3600.0 * max(target_utilization_pct, 1.0) / 100.0
    return max(0, math.ceil(peak_tokens_per_hour / cap))
