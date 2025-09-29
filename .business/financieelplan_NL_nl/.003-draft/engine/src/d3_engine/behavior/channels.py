from __future__ import annotations
from typing import Dict, List, Tuple
import math

"""
Channel modeling utilities.

- Diminishing returns via power-law response: conversions = alpha * spend^beta, 0<beta<1.
- Effective CAC per channel = spend / conversions (guarded).
- Agency fee applied as overhead on spend (e.g., +15%).

Inputs are plain dict lists to keep this module pure and decoupled from I/O.
"""


def allocate_spend(total_budget: float, allocation: Dict[str, float]) -> Dict[str, float]:
    total_w = sum(max(0.0, w) for w in allocation.values()) or 1.0
    return {ch: total_budget * max(0.0, w) / total_w for ch, w in allocation.items()}


def conversions_for_channel(spend: float, alpha: float, beta: float) -> float:
    spend = max(0.0, float(spend))
    beta = max(1e-6, min(float(beta), 1.0))
    alpha = max(0.0, float(alpha))
    return alpha * (spend ** beta)


def effective_cac(spend: float, conversions: float) -> float:
    if conversions <= 0:
        return float("inf") if spend > 0 else 0.0
    return spend / conversions


def monthly_conversions(
    budgets: List[float],
    allocation: Dict[str, float],
    channel_params: Dict[str, Tuple[float, float]],  # channel -> (alpha, beta)
    agency_fee_pct: float = 0.0,
) -> Tuple[List[float], List[Dict[str, float]], List[Dict[str, float]]]:
    """Compute per-month conversions.

    Returns (total_conversions_per_month, per_channel_conversions, per_channel_cac)
    """
    total_conv: List[float] = []
    conv_by_ch: List[Dict[str, float]] = []
    cac_by_ch: List[Dict[str, float]] = []
    fee_mult = 1.0 + max(0.0, agency_fee_pct) / 100.0
    for b in budgets:
        spend_by_ch = allocate_spend(float(b) * fee_mult, allocation)
        conv_ch: Dict[str, float] = {}
        cac_ch: Dict[str, float] = {}
        total = 0.0
        for ch, s in spend_by_ch.items():
            alpha, beta = channel_params.get(ch, (0.0, 1.0))
            c = conversions_for_channel(s, alpha, beta)
            conv_ch[ch] = c
            cac_ch[ch] = effective_cac(s, c)
            total += c
        total_conv.append(total)
        conv_by_ch.append(conv_ch)
        cac_by_ch.append(cac_ch)
    return total_conv, conv_by_ch, cac_by_ch
