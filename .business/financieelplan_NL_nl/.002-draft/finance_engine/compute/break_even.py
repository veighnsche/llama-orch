from __future__ import annotations

from typing import Optional


def required_inflow(fixed_total: float, margin_rate: float, marketing_pct: float) -> Optional[float]:
    if margin_rate <= marketing_pct:
        return None
    return fixed_total / (margin_rate - marketing_pct)


def compute_break_even(fixed_total: float, margin_rate: float, marketing_pct: float):
    """Return dict with required_inflow_eur and required_margin_eur.
    margin_rate and marketing_pct are fractions (e.g., 0.35 and 0.15).
    """
    inflow = required_inflow(fixed_total, margin_rate, marketing_pct)
    if inflow is None:
        return {
            "required_inflow_eur": None,
            "required_margin_eur": None,
        }
    gross_margin = margin_rate * inflow
    return {
        "required_inflow_eur": inflow,
        "required_margin_eur": gross_margin,
    }
