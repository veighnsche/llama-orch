from __future__ import annotations
from typing import List
import math

"""
Cohort retention utilities.

- retention_exponential: exponential survival curve by monthly age
- retention_weibull: Weibull survival curve by monthly age
- cohort_active_series: accumulate actives from monthly new customers and a retention curve
"""


def retention_exponential(length: int, monthly_decay_rate: float) -> List[float]:
    """Return length-length survival curve S(age) = exp(-lambda * age)."""
    lam = max(0.0, float(monthly_decay_rate))
    return [math.exp(-lam * age) for age in range(max(0, int(length)))]


def retention_weibull(length: int, k_shape: float, lam_scale: float) -> List[float]:
    """Weibull survival S(age) = exp(-(age/lam)^k)."""
    k = max(1e-6, float(k_shape))
    lam = max(1e-6, float(lam_scale))
    return [math.exp(-((age / lam) ** k)) for age in range(max(0, int(length)))]


def cohort_active_series(new_customers: List[float], survival: List[float]) -> List[float]:
    """Compute actives per month by summing surviving cohorts.

    active[m] = sum_{age=0..m} new[m-age] * survival[age]
    """
    n = len(new_customers)
    s = survival
    mlen = len(s)
    actives: List[float] = []
    for m in range(n):
        total = 0.0
        for age in range(min(m + 1, mlen)):
            total += max(0.0, float(new_customers[m - age])) * float(s[age])
        actives.append(total)
    return actives
