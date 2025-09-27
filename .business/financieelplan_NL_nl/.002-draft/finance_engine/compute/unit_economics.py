from __future__ import annotations

from typing import Dict, Any

import math


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def compute_unit_economics(
    *,
    acquisition: Dict[str, Any],
    public_tpl: Dict[str, Any],
    funnel_details_base: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute simple unit economics from acquisition inputs and blended public pricing.

    Assumptions:
    - ARPU_revenue/month for a paid user = avg_paid_tokens * blended_sell_per_token
    - Contribution per user/month = ARPU_revenue * margin_rate - payment_fee%
      (Payment fee is applied on revenue)
    - CAC (blended) = marketing_eur / max(paid_new, 1)
    - Payback months = CAC / contribution_per_month (if > 0)
    - MRR snapshot (from new in base) = paid_new * ARPU_revenue
    - Steady-state (simple) active_paid = paid_new / churn_rate
      MRR_steady = active_paid * ARPU_revenue; ARR_steady = 12 * MRR_steady
    """
    acq = acquisition or {}
    g = acq.get("global") or acq.get("global_") or {}

    avg_paid_tokens = _num(g.get("avg_tokens_paid_per_user_per_month"), 0.0)
    fee_pct = _num(g.get("payment_fee_pct_of_revenue"), 0.0) / 100.0
    churn_pct = _num(g.get("churn_pct_per_month"), 0.0) / 100.0

    blended = public_tpl.get("blended", {}) if isinstance(public_tpl, dict) else {}
    sell_1m = _num(blended.get("sell_per_1m_eur"), 0.0)
    cost_1m = _num(blended.get("cost_per_1m_eur"), 0.0)
    margin_rate = _num(blended.get("margin_rate"), 0.0)

    # price per token
    sell_per_token = sell_1m / 1_000_000.0 if sell_1m > 0 else 0.0
    # ARPU (revenue) per paid user per month
    arpu_rev = avg_paid_tokens * sell_per_token
    # Contribution per user: revenue * margin_rate minus processor fee on revenue
    contribution = max(0.0, (arpu_rev * margin_rate) - (arpu_rev * fee_pct))

    paid_new = _num(funnel_details_base.get("paid_new"), 0.0)
    marketing_eur = _num(funnel_details_base.get("marketing_eur"), 0.0)
    cac = marketing_eur / paid_new if paid_new > 0 else float("inf")

    payback_months = float("inf")
    if contribution > 0:
        payback_months = cac / contribution

    # Snapshot MRR from new cohort
    mrr_snapshot = paid_new * arpu_rev

    # Steady-state MRR under constant flow assumption
    mrr_steady = 0.0
    arr_steady = 0.0
    active_paid_ss = 0.0
    if churn_pct > 0:
        active_paid_ss = paid_new / churn_pct
        mrr_steady = active_paid_ss * arpu_rev
        arr_steady = mrr_steady * 12.0

    ltv = float("inf")
    if churn_pct > 0:
        # contribution per user per month over churn
        ltv = contribution / churn_pct

    ltv_cac = float("inf")
    if cac not in (0.0, float("inf")):
        ltv_cac = ltv / cac if cac > 0 else float("inf")

    return {
        "arpu_revenue_eur": round(arpu_rev, 2),
        "arpu_contribution_eur": round(contribution, 2),
        "cac_eur": None if math.isinf(cac) else round(cac, 2),
        "ltv_eur": None if math.isinf(ltv) else round(ltv, 2),
        "ltv_cac": None if math.isinf(ltv_cac) else round(ltv_cac, 2),
        "payback_months": None if math.isinf(payback_months) else round(payback_months, 1),
        "mrr_snapshot_eur": round(mrr_snapshot, 2),
        "mrr_steady_eur": round(mrr_steady, 2),
        "arr_steady_eur": round(arr_steady, 2),
        "active_paid_steady": round(active_paid_ss, 2),
        "paid_new_base": round(paid_new, 2),
        "marketing_base_eur": round(marketing_eur, 2),
    }
