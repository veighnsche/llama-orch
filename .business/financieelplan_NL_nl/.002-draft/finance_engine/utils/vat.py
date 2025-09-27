from __future__ import annotations

from typing import Dict, List


def vat_set_aside(amount_eur: float, vat_rate_pct: float) -> float:
    try:
        amt = float(amount_eur)
        rate = float(vat_rate_pct)
    except Exception:
        return 0.0
    return round(amt * rate / 100.0, 2)


def vat_examples(vat_rate_pct: float) -> Dict[str, float]:
    """
    Return example VAT amounts for fixed inflows used across docs/contexts.
    Keys mirror those used in engine context.
    """
    v = float(vat_rate_pct)
    small = 1000.0
    medium = 10000.0
    large = 100000.0
    return {
        "revenue_small": small,
        "vat_small": vat_set_aside(small, v),
        "net_small": round(small - vat_set_aside(small, v), 2),
        "revenue_medium": medium,
        "vat_medium": vat_set_aside(medium, v),
        "net_medium": round(medium - vat_set_aside(medium, v), 2),
        "revenue_large": large,
        "vat_large": vat_set_aside(large, v),
        "net_large": round(large - vat_set_aside(large, v), 2),
    }


def vat_set_aside_rows(vat_rate_pct: float) -> List[Dict[str, float]]:
    """
    Rows for CSV export matching previous artifacts behavior.
    """
    v = float(vat_rate_pct)
    return [
        {"inflow_eur": 1000, "vat_set_aside_eur": vat_set_aside(1000, v)},
        {"inflow_eur": 10000, "vat_set_aside_eur": vat_set_aside(10000, v)},
        {"inflow_eur": 100000, "vat_set_aside_eur": vat_set_aside(100000, v)},
    ]
