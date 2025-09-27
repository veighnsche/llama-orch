from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd

from ...compute.pricing import compute_model_economics
from ...compute.scenarios import compute_public_scenarios
from ...compute.private_tap import compute_private_tap_economics
from ...compute.break_even import compute_break_even
from ...compute.loans import Loan, flat_interest_schedule, loan_totals


def compute_all(
    *,
    config: Dict[str, Any],
    extra: Dict[str, Any],
    lending: Dict[str, Any],
    price_sheet: pd.DataFrame,
    gpu_df: pd.DataFrame,
) -> Dict[str, Any]:
    # Model economics
    model_df = compute_model_economics(cfg=config, extra=extra, price_sheet=price_sheet, gpu_df=gpu_df)

    # Derived columns and sanitation
    def gm_pct(row):
        sell = float(row.get("sell_per_1m_eur", 0.0)) or 0.0
        return 0.0 if sell <= 0 else round((float(row.get("margin_per_1m_med", 0.0)) / sell) * 100.0, 2)

    model_df["gross_margin_pct_med"] = model_df.apply(gm_pct, axis=1)
    for col in [
        "cost_per_1m_min",
        "cost_per_1m_med",
        "cost_per_1m_max",
        "sell_per_1m_eur",
        "margin_per_1m_min",
        "margin_per_1m_med",
        "margin_per_1m_max",
        "gross_margin_pct_med",
    ]:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(0.0)

    # Public subset for scenarios and display (exclude private/service SKUs)
    pub_df = model_df[~model_df["model"].astype(str).str.startswith(("private_tap_", "priority_", "oss_support_"))].copy()

    # Finance knobs
    marketing_pct = float(config.get("finance", {}).get("marketing_allocation_pct_of_inflow", 0.0)) / 100.0
    scen = extra.get("scenarios", {}).get("monthly", {})
    worst = float(scen.get("worst_m_tokens", 1.0))
    base = float(scen.get("base_m_tokens", 5.0))
    best = float(scen.get("best_m_tokens", 15.0))
    per_model_mix = extra.get("per_model_mix", {})

    # Loan and fixed
    loan_amount = lending.get("amount_eur") or extra.get("loan", {}).get("amount_eur") or 30000
    loan_term = lending.get("term_months") or extra.get("loan", {}).get("term_months") or 60
    loan_rate = lending.get("interest_rate_pct") or extra.get("loan", {}).get("interest_rate_pct") or 9.95
    loan_obj = Loan(principal_eur=float(loan_amount), annual_rate_pct=float(loan_rate), term_months=int(loan_term))
    monthly_payment, total_repay, total_interest = loan_totals(loan_obj)

    fixed_personal = float((config.get("finance", {}).get("fixed_costs_monthly_eur", {}).get("personal") or 0.0))
    fixed_business = float((config.get("finance", {}).get("fixed_costs_monthly_eur", {}).get("business") or 0.0))
    fixed_total_with_loan = round(fixed_personal + fixed_business + float(monthly_payment), 2)

    # Public scenarios table + template dict
    public_df, public_tpl = compute_public_scenarios(
        pub_df,
        per_model_mix=per_model_mix,
        fixed_total_with_loan=fixed_total_with_loan,
        marketing_pct=marketing_pct,
        worst_base_best=(worst, base, best),
    )

    # Private tap economics
    eur_usd = float((config.get("fx", {}).get("eur_usd_rate") if isinstance(config.get("fx", {}).get("eur_usd_rate"), (int, float, str)) else None) or 1.08)
    fx_buffer_pct = float((config.get("pricing_inputs", {}).get("fx_buffer_pct") if isinstance(config.get("pricing_inputs", {}).get("fx_buffer_pct"), (int, float, str)) else None) or 0.0)
    private_markup_pct = float((config.get("pricing_inputs", {}).get("private_tap_default_markup_over_provider_cost_pct") if isinstance(config.get("pricing_inputs", {}).get("private_tap_default_markup_over_provider_cost_pct"), (int, float, str)) else None) or 50.0)
    per_gpu_markup = extra.get("pricing_inputs", {}).get("private_tap_markup_by_gpu", {}) or {}
    private_df = compute_private_tap_economics(
        gpu_df,
        eur_usd_rate=eur_usd,
        buffer_pct=fx_buffer_pct,
        markup_pct=private_markup_pct,
        markup_by_gpu=per_gpu_markup,
    )

    # Break-even
    margin_rate = float(public_tpl.get("blended", {}).get("margin_rate", 0.0))
    be = compute_break_even(fixed_total_with_loan, margin_rate, marketing_pct)

    # Loan schedule
    loan_rows = flat_interest_schedule(loan_obj)
    loan_df = pd.DataFrame(loan_rows)

    return {
        "model_df": model_df,
        "pub_df": pub_df,
        "public_df": public_df,
        "public_tpl": public_tpl,
        "private_df": private_df,
        "break_even": be,
        "loan_df": loan_df,
        "loan_obj": loan_obj,
        "monthly_payment": monthly_payment,
        "total_repay": total_repay,
        "total_interest": total_interest,
        "fixed_personal": fixed_personal,
        "fixed_business": fixed_business,
        "fixed_total_with_loan": fixed_total_with_loan,
        "eur_usd": eur_usd,
        "fx_buffer_pct": fx_buffer_pct,
        "private_markup_pct": private_markup_pct,
        "per_model_mix": per_model_mix,
        "marketing_pct": marketing_pct,
    }
