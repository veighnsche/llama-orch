"""Public demand model (scaffold)."""
from __future__ import annotations
import math
from typing import List


def monthly_tokens(new_customers: float, tokens_per_conversion_mean: float) -> float:
    return max(0.0, new_customers) * max(0.0, tokens_per_conversion_mean)


def hourly_timeseries_uniform(
    monthly_total_tokens: float,
    hours_in_month: int = 24 * 30,
    peak_factor: float = 1.0,
    diurnal: bool = False,
) -> List[float]:
    """Split monthly tokens into an hourly series.

    - If diurnal=False: uniform distribution; if peak_factor>1, apply a single deterministic peak and re-scale to conserve sum.
    - If diurnal=True: apply a daily sinusoidal curve with given peak_factor across the day, repeated over the month, conserving sum.
    """
    hours = max(1, hours_in_month)
    base = monthly_total_tokens / hours
    series = [base] * hours

    if peak_factor <= 1.0 and not diurnal:
        return series

    if not diurnal:
        # Single peak hour at mid-month index
        idx = hours // 2
        series[idx] = base * peak_factor
        # Adjust others to conserve total
        remaining = monthly_total_tokens - series[idx]
        per = max(0.0, remaining / (hours - 1)) if hours > 1 else 0.0
        for i in range(hours):
            if i != idx:
                series[i] = per
        return series

    # Diurnal sinusoidal pattern repeated daily
    daily = 24
    # Angle offset so that peak near hour 15 (3pm)
    def day_weight(h: int) -> float:
        # Map to [0, 2π]
        # Baseline 1.0 with amplitude a so that max/min achieve peak_factor vs trough maintaining positive weights
        # Choose amplitude a so that max/min ratio approximates peak_factor
        a = min(0.9, (peak_factor - 1.0) / (peak_factor + 1.0) * 2.0) if peak_factor > 1.0 else 0.0
        phi = (h - 15) / daily * 2 * math.pi
        return 1.0 + a * math.cos(phi)

    weights: List[float] = []
    days = math.ceil(hours / daily)
    for d in range(days):
        for h in range(daily):
            weights.append(day_weight(h))
    weights = weights[:hours]
    total_w = sum(weights) or 1.0
    series = [monthly_total_tokens * (w / total_w) for w in weights]
    return series
