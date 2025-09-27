from __future__ import annotations

from typing import Dict, Any, Tuple
import math
import pandas as pd

from .acquisition import simulate_funnel_details
from .private_tap import compute_private_tap_economics


def _num(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d


def _pct(x: Any) -> float:
    try:
        return float(x) / 100.0
    except Exception:
        return 0.0


def _month_mult(seasonality: Dict[str, Any], idx: int) -> float:
    mm = seasonality.get("month_multipliers") or []
    if not isinstance(mm, list) or not mm:
        return 1.0
    # idx 1..N mapped to 0..11
    return float(mm[(idx - 1) % len(mm)])


def _blend(values: Dict[str, Any], key: str) -> float:
    try:
        return float(values.get(key, 0.0))
    except Exception:
        return 0.0


def _avg_or0(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())


def compute_timeseries(
    *,
    agg_inputs: Dict[str, Any],
    scenarios: Dict[str, Any],
    config: Dict[str, Any],
    gpu_df: pd.DataFrame,
    capacity_overrides: Dict[str, Any],
    gpu_pricing: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute 24-month public + private + total timeseries.
    Depends on prior steps for:
      - public_tpl.blended sell/cost per 1M
      - fixed_total_with_loan
      - unit_economics (ARPU revenue for paid user)
    Extra inputs passed via `scenarios` under __timeseries, __seasonality, __billing, __private_sales.
    """
    horizon = int(_num((scenarios.get("__timeseries") or {}).get("horizon_months"), 24))
    assert horizon == 24, "timeseries.horizon_months must be 24"
    seasonality = scenarios.get("__seasonality") or {}
    billing = scenarios.get("__billing") or {}
    private_sales = scenarios.get("__private_sales") or {}

    public_tpl = agg_inputs.get("public_tpl") or {}
    blended = public_tpl.get("blended") or {}
    sell_1m = _blend(blended, "sell_per_1m_eur")
    cost_1m = _blend(blended, "cost_per_1m_eur")
    fixed_per_month = float(agg_inputs.get("fixed_total_with_loan") or 0.0)

    # multipliers
    dr = (seasonality.get("diminishing_returns") or {}) if isinstance(seasonality, dict) else {}
    cpc_slope = _num(dr.get("cpc_slope_per_extra_1k_eur"), 0.0)
    cvr_decay_pct = _pct(dr.get("cvr_decay_per_budget_doubling_pct"))  # 0..1

    # Use base-case funnel scaled by month multipliers; diminishing returns approximated as modest CVR reduction per extra budget chunk
    acquisition = scenarios.get("__acquisition") or {}

    # ARPU revenue per paid user if provided from unit economics
    unit = agg_inputs.get("unit_economics") or {}
    arpu_rev = _num(unit.get("arpu_revenue_eur"), 0.0)

    # Private Tap price per GPU hour (average across GPUs)
    eur_usd = float((config.get("fx") or {}).get("eur_usd_rate") or 1.08)
    fxbuf = float((config.get("pricing_inputs") or {}).get("fx_buffer_pct") or 0.0)
    per_gpu = compute_private_tap_economics(
        gpu_df,
        eur_usd_rate=eur_usd,
        buffer_pct=fxbuf,
        markup_pct=float((gpu_pricing.get("default_markup_pct") or (config.get("pricing_inputs") or {}).get("private_tap_default_markup_over_provider_cost_pct") or 50.0)),
        markup_by_gpu=gpu_pricing.get("private_tap_markup_by_gpu") if isinstance(gpu_pricing, dict) else None,
    )
    sell_hr = _avg_or0(per_gpu, "sell_eur_hr")
    prov_hr = _avg_or0(per_gpu, "provider_eur_hr_med")

    # Private sales params
    L_base = _num(private_sales.get("leads_per_month"), 0)
    win_rate = _pct(private_sales.get("win_rate_pct"))
    cycle = int(_num(private_sales.get("sales_cycle_months"), 3))
    commit_hr = _num(private_sales.get("avg_commit_gpu_hours_per_month"), 0)
    mgmt_fee = _num(private_sales.get("mgmt_fee_eur_per_month"), 0)
    disc = _pct(private_sales.get("discount_pct_on_gpu_hour"))
    conv_from_public = _pct(private_sales.get("lead_conversion_from_public_pct"))

    # Churn/reactivation
    ts_cfg = scenarios.get("__timeseries") or {}
    reactivation = _pct(ts_cfg.get("reactivation_pct"))
    # For churn, derive a simple blended churn from segments globals (average across segments)
    segs = (acquisition.get("segments") or {}) if isinstance(acquisition, dict) else {}
    churn_vals = []
    for s in ("oss", "agencies", "it_teams", "compliance"):
        g = (segs.get(s) or {}).get("global") or {}
        if "churn_pct_per_month" in g:
            churn_vals.append(_pct(g.get("churn_pct_per_month")))
    churn = sum(churn_vals) / len(churn_vals) if churn_vals else 0.04

    # Billing: processor fees and credit packs
    prepaid = (billing.get("prepaid") or {}) if isinstance(billing, dict) else {}
    fee_pct = _pct(prepaid.get("processor_fee_pct_of_revenue"))
    fee_fixed = _num(prepaid.get("processor_fixed_fee_eur_per_tx"), 0.0)
    packs = prepaid.get("credit_packs") or []
    mix = prepaid.get("pack_mix_pct") or {}
    pack_prices = {str(p.get("name")): float(p.get("eur")) for p in packs if isinstance(p, dict) and p.get("name") is not None}
    wallet_buffer_months = _num(prepaid.get("wallet_buffer_months"), 0.0)
    expiry_months = int(_num(prepaid.get("expiry_months"), 12))

    # Timeseries accumulators
    pub_rows = []
    priv_rows = []
    tot_rows = []

    # Sales leads delay queue for wins
    lead_queue = [0.0] * cycle
    active_paid = 0.0
    # Credit liability buckets: list of (created_month, remaining_amount)
    credit_buckets: list[tuple[int, float]] = []
    liability = 0.0
    total_breakage = 0.0

    for m in range(1, horizon + 1):
        budget_mult = _month_mult(seasonality, m)
        # Simplified diminishing returns: reduce CVR by a factor proportional to budget multiplier
        cvr_mult = max(0.0, 1.0 - (budget_mult - 1.0) * cvr_decay_pct)
        # CAC multiplier kept neutral for now (could rise slightly with budget)
        cac_mult = 1.0
        overrides = {"base": {"budget_multiplier": budget_mult, "cvr_multiplier": cvr_mult, "cac_multiplier": cac_mult}}
        f = simulate_funnel_details(acquisition=acquisition, funnel_overrides=overrides, case="base")
        m_tokens = float(f.get("total_tokens_m", 0.0))
        marketing_eur = float(f.get("marketing_eur", 0.0))
        paid_new = float(f.get("paid_new", 0.0))

        recognized_rev = m_tokens * sell_1m
        cogs = m_tokens * cost_1m
        gross = recognized_rev - cogs
        net = gross - marketing_eur - fixed_per_month

        # Determine inflow needed to maintain wallet buffer after recognition
        # Target end-of-month liability based on current month recognized revenue
        target_liab = recognized_rev * wallet_buffer_months

        # Recognize revenue by consuming from oldest buckets first
        to_recognize = recognized_rev
        i = 0
        while to_recognize > 1e-9 and i < len(credit_buckets):
            created, amt = credit_buckets[i]
            take = min(amt, to_recognize)
            amt -= take
            to_recognize -= take
            credit_buckets[i] = (created, amt)
            i += 1
        # Remove empty buckets
        credit_buckets = [(cm, a) for (cm, a) in credit_buckets if a > 1e-9]

        # Current liability before new inflow
        liability = sum(a for _, a in credit_buckets)

        # Compute inflow so that end liability hits target after recognition and before expiry
        inflow = max(0.0, target_liab - liability)
        if inflow > 0:
            credit_buckets.append((m, inflow))

        # Processor fees on inflow transactions
        tx_count = 0.0
        if pack_prices and mix and inflow > 0:
            for pack, pct in mix.items():
                share = inflow * (_pct(pct))
                price = pack_prices.get(pack) or 0.0
                if price > 0:
                    tx_count += (share / price)
        fees = inflow * fee_pct + tx_count * fee_fixed

        # Expire buckets older than expiry_months
        remaining_buckets: list[tuple[int, float]] = []
        breakage = 0.0
        for (cm, amt) in credit_buckets:
            age = m - cm
            if age >= expiry_months and amt > 1e-9:
                breakage += amt
            else:
                remaining_buckets.append((cm, amt))
        credit_buckets = remaining_buckets
        total_breakage += breakage
        liability = sum(a for _, a in credit_buckets)

        # Active paid cohort
        churned = active_paid * churn
        reactivated = churned * reactivation
        active_paid = max(0.0, active_paid - churned + reactivated + paid_new)
        mrr = active_paid * arpu_rev

        pub_rows.append({
            "month": m,
            "m_tokens": round(m_tokens, 3),
            "recognized_revenue_eur": round(recognized_rev, 2),
            "inflow_eur": round(inflow, 2),
            "liability_end_eur": round(liability, 2),
            "breakage_eur": round(breakage, 2),
            "cogs_eur": round(cogs, 2),
            "gross_eur": round(gross, 2),
            "marketing_eur": round(marketing_eur, 2),
            "fixed_eur": round(fixed_per_month, 2),
            "fees_eur": round(fees, 2),
            "net_eur": round(net - fees, 2),
            "paid_new": round(paid_new, 2),
            "active_paid": round(active_paid, 2),
            "mrr_est": round(mrr, 2),
        })

        # Private pipeline
        leads_from_public = paid_new * conv_from_public
        leads_total = L_base + leads_from_public
        # push into queue
        lead_queue.append(leads_total)
        aged = lead_queue.pop(0)
        wins = aged * win_rate
        # Revenue and cogs from wins this month
        gpu_rev = wins * commit_hr * sell_hr * (1.0 - disc)
        gpu_cogs = wins * commit_hr * prov_hr
        mgmt = wins * mgmt_fee
        priv_rows.append({
            "month": m,
            "leads": round(leads_total, 2),
            "wins": round(wins, 2),
            "gpu_rev_eur": round(gpu_rev, 2),
            "gpu_cogs_eur": round(gpu_cogs, 2),
            "mgmt_fee_eur": round(mgmt, 2),
            "net_eur": round(gpu_rev - gpu_cogs + mgmt, 2),
        })

        tot_rows.append({
            "month": m,
            "revenue_eur": round(recognized_rev + gpu_rev + mgmt, 2),
            "inflow_eur": round(inflow, 2),
            "cogs_eur": round(cogs + gpu_cogs, 2),
            "marketing_eur": round(marketing_eur, 2),
            "fixed_eur": round(fixed_per_month, 2),
            "fees_eur": round(fees, 2),
            "net_eur": round((gross - marketing_eur - fixed_per_month - fees) + (gpu_rev - gpu_cogs + mgmt), 2),
            "active_paid": round(active_paid, 2),
            "mrr_est": round(mrr, 2),
        })

    return pd.DataFrame(pub_rows), pd.DataFrame(priv_rows), pd.DataFrame(tot_rows)
