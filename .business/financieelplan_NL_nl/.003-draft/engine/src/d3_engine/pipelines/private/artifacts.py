"""Private pipeline computations (pure rows; no I/O)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import statistics
# no I/O helpers imported; compute_rows returns data only
import numpy as np


def _fx_eur_per_usd(facts: Dict[str, Any]) -> float:
    try:
        return float(
            facts.get("market_env", {})
            .get("finance", {})
            .get("eur_usd_fx_rate", {})
            .get("value")
        )
    except Exception:
        return 1.0


def _fx_buffer_pct(prv_op: Dict[str, Any]) -> float:
    try:
        return float(prv_op.get("meta", {}).get("fx_buffer_pct", 0.0))
    except Exception:
        return 0.0


def _eur_hr(usd_hr: float, eur_usd_rate: float, fx_buffer_pct: float) -> float:
    if eur_usd_rate <= 0:
        eur_usd_rate = 1.0
    return (usd_hr / eur_usd_rate) * (1.0 + max(fx_buffer_pct, 0.0) / 100.0)


    # (removed) write_all: module is now pure; use compute_rows() instead


def compute_rows(state: Dict[str, Any]) -> Dict[str, tuple[list[str], list[dict]]]:
    """Pure computation of private rows (economics, recommendation, customers).

    Returns mapping name -> (headers, rows) without performing any I/O.
    Names correspond to filenames without extension:
      - private_tap_economics
      - private_vendor_recommendation
      - private_tap_customers_by_month
    """
    econ_hdr = ["gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr", "margin_pct"]
    reco_hdr = ["gpu", "provider", "usd_hr", "eur_hr_effective", "score"]
    cust_hdr = ["month", "private_budget_eur", "private_cac_eur", "expected_new_clients", "active_clients", "hours", "sell_eur_hr", "revenue_eur"]
    costs_hdr = ["month", "gpu", "provider", "eur_hr_effective", "hours", "cost_eur"]

    rentals: List[Dict[str, Any]] = state.get("curated", {}).get("gpu_rentals", [])
    facts: Dict[str, Any] = state.get("facts", {})
    prv_op: Dict[str, Any] = state.get("operator", {}).get("private_tap", {})

    eur_usd_rate = _fx_eur_per_usd(facts)
    fx_buf = _fx_buffer_pct(prv_op)
    pp = prv_op.get("pricing_policy", {}).get("private_tap", {}) if isinstance(prv_op, dict) else {}
    markup_pct = float(pp.get("default_markup_over_provider_cost_pct", 40.0))
    mgmt_fee = float(pp.get("management_fee_eur_per_month", 0.0))

    by_gpu: Dict[str, List[tuple[str, float, float]]] = {}
    for r in rentals:
        gpu = str(r.get("gpu"))
        provider = str(r.get("provider"))
        usd = float(r.get("usd_hr"))
        eur = (usd / (eur_usd_rate if eur_usd_rate > 0 else 1.0)) * (1.0 + max(fx_buf, 0.0) / 100.0)
        by_gpu.setdefault(gpu, []).append((provider, usd, eur))

    try:
        horizon = int(state.get("simulation", {}).get("targets", {}).get("horizon_months", 12))
    except Exception:
        horizon = 12

    econ_rows: List[dict] = []
    reco_rows: List[dict] = []
    cust_rows: List[dict] = []
    costs_rows: List[dict] = []

    for gpu, lst in sorted(by_gpu.items(), key=lambda kv: kv[0]):
        if not lst:
            continue
        eur_list = [eur for (_, _, eur) in lst]
        # median
        eur_med = statistics.median(eur_list)
        sell_eur_hr = eur_med * (1.0 + max(markup_pct, 0.0) / 100.0)
        margin_eur_hr = sell_eur_hr - eur_med
        margin_pct = 0.0 if sell_eur_hr <= 0 else (margin_eur_hr / sell_eur_hr) * 100.0
        econ_rows.append({
            "gpu": gpu,
            "provider_eur_hr_med": f"{eur_med}",
            "markup_pct": f"{markup_pct}",
            "sell_eur_hr": f"{sell_eur_hr}",
            "margin_eur_hr": f"{margin_eur_hr}",
            "margin_pct": f"{margin_pct}",
        })

        # Recommendation: cheapest EUR/hr
        best = min(lst, key=lambda t: t[2])
        reco_rows.append({
            "gpu": gpu, "provider": best[0], "usd_hr": f"{best[1]}", "eur_hr_effective": f"{best[2]}", "score": "1.0"
        })

        acq = prv_op.get("acquisition", {}) if isinstance(prv_op, dict) else {}
        budget0 = float(acq.get("budget_month0_eur", 0.0) or 0.0)
        cac = float(acq.get("cac_base_eur", 0.0) or 0.0)
        hours_per_client = float(acq.get("hours_per_client_month_mean", 0.0) or 0.0)
        churn_pct = float(acq.get("churn_pct_mom", 0.0) or 0.0)
        budget_growth_pct_mom = float(acq.get("budget_growth_pct_mom", 0.0) or 0.0)
        months = np.arange(horizon, dtype=float)
        growth = (1.0 + budget_growth_pct_mom / 100.0)
        budget_series = budget0 * np.power(growth, months)
        new_series = np.where(cac > 0, budget_series / cac, 0.0)
        active = 0.0
        active_series = []
        decay = 1.0 - max(churn_pct, 0.0) / 100.0
        for new in new_series.tolist():
            active = max(0.0, active * decay + new)
            active_series.append(active)
        active_series = np.array(active_series)
        hours_series = active_series * hours_per_client
        revenue_series = hours_series * sell_eur_hr + mgmt_fee
        for month in range(horizon):
            cust_rows.append({
                "month": str(month), "private_budget_eur": f"{budget_series[month]}", "private_cac_eur": f"{cac}",
                "expected_new_clients": f"{new_series[month]}", "active_clients": f"{active_series[month]}", "hours": f"{hours_series[month]}",
                "sell_eur_hr": f"{sell_eur_hr}", "revenue_eur": f"{revenue_series[month]}"
            })
            # Costs by month using cheapest provider EUR/hr for this GPU
            costs_rows.append({
                "month": str(month),
                "gpu": gpu,
                "provider": best[0],
                "eur_hr_effective": f"{best[2]}",
                "hours": f"{hours_series[month]}",
                "cost_eur": f"{hours_series[month] * best[2]}",
            })

    return {
        "private_tap_economics": (econ_hdr, econ_rows),
        "private_vendor_recommendation": (reco_hdr, reco_rows),
        "private_tap_customers_by_month": (cust_hdr, cust_rows),
        "private_tap_costs_by_month": (costs_hdr, costs_rows),
    }
