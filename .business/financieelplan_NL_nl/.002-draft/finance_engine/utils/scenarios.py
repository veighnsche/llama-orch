from __future__ import annotations

from typing import Dict, Any, Tuple


def make_month_case(month: Dict[str, Any], fixed_total_with_loan: float) -> Dict[str, float]:
    revenue = float(month.get("revenue_eur", 0.0))
    cogs = float(month.get("cogs_eur", 0.0))
    gross = float(month.get("gross_margin_eur", revenue - cogs))
    marketing = float(month.get("marketing_reserved_eur", 0.0))
    net = float(month.get("net_eur", gross - marketing - float(fixed_total_with_loan)))
    return {
        "public_revenue": revenue,
        "private_revenue": 0.0,
        "total_revenue": revenue,
        "cogs": cogs,
        "gross": gross,
        "marketing": marketing,
        "net": net,
    }


def scale_case(case: Dict[str, float], n: int) -> Dict[str, float]:
    return {k: (v * n if isinstance(v, (int, float)) else v) for k, v in case.items()}


def monthly_yearly_sixty(
    scenarios_tpl: Dict[str, Any], fixed_total_with_loan: float
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    worst = make_month_case(scenarios_tpl.get("worst", {}), fixed_total_with_loan)
    base = make_month_case(scenarios_tpl.get("base", {}), fixed_total_with_loan)
    best = make_month_case(scenarios_tpl.get("best", {}), fixed_total_with_loan)

    monthly = {"worst": worst, "base": base, "best": best}
    yearly = {
        "worst": scale_case(worst, 12),
        "base": scale_case(base, 12),
        "best": scale_case(best, 12),
        "fixed_total": float(fixed_total_with_loan) * 12,
    }
    sixty_m = {
        "worst": scale_case(worst, 60),
        "base": scale_case(base, 60),
        "best": scale_case(best, 60),
        "fixed_total": float(fixed_total_with_loan) * 60,
    }
    return monthly, yearly, sixty_m
