from __future__ import annotations
from typing import List, Optional
import random

"""
Stochastic noise utilities (optional, deterministic with seed when provided).

- add_gaussian_noise: additive N(0, sigma) noise
- add_log_normal_noise: multiplicative lognormal noise (exp(N(0, sigma)))
"""


def add_gaussian_noise(series: List[float], sigma: float, seed: Optional[int] = None, clip_min: float | None = 0.0) -> List[float]:
    r = random.Random(seed)
    s = max(0.0, float(sigma))
    out: List[float] = []
    for v in series:
        val = float(v) + r.gauss(0.0, s)
        if clip_min is not None and val < clip_min:
            val = clip_min
        out.append(val)
    return out


def add_log_normal_noise(series: List[float], sigma: float, seed: Optional[int] = None) -> List[float]:
    r = random.Random(seed)
    s = max(0.0, float(sigma))
    out: List[float] = []
    for v in series:
        # multiplicative factor: lognormal with mu=0
        factor = r.lognormvariate(0.0, s)
        out.append(float(v) * factor)
    return out
