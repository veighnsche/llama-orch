from __future__ import annotations
from typing import List

"""
Seasonality utilities.

- seasonal_multipliers: normalize an input pattern to mean 1.0
- apply_seasonality: multiply a base series by repeating multipliers
"""


def seasonal_multipliers(pattern: List[float]) -> List[float]:
    if not pattern:
        return [1.0]
    xs = [max(0.0, float(x)) for x in pattern]
    mean = sum(xs) / len(xs) if xs else 1.0
    mean = mean if mean > 0 else 1.0
    return [x / mean for x in xs]


def apply_seasonality(series: List[float], multipliers: List[float]) -> List[float]:
    if not multipliers:
        multipliers = [1.0]
    m = len(multipliers)
    out: List[float] = []
    for i, v in enumerate(series):
        out.append(float(v) * float(multipliers[i % m]))
    return out
