from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import pandas as pd

from ...compute.pricing import compute_model_economics
from ...compute.scenarios import compute_public_scenarios
from ...compute.private_tap import compute_private_tap_economics
from ...compute.break_even import compute_break_even
from ...compute.acquisition import simulate_m_tokens_from_funnel, simulate_funnel_details
from ...compute.unit_economics import compute_unit_economics
from ...compute.loans import Loan, flat_interest_schedule, loan_totals
from ...compute.timeseries import compute_timeseries
from ...types.inputs import Config, Lending
from ...utils.coerce import get, get_float, safe_float, pct_to_fraction


def _compute_model_block(*, config: Config, price_sheet: pd.DataFrame, gpu_df: pd.DataFrame, tps_df: pd.DataFrame, scenarios: Dict[str, Any], overrides: Dict[str, Any], capacity_overrides: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute base model economics and a public-only subset DataFrame."""
    model_df = compute_model_economics(cfg=config, price_sheet=price_sheet, gpu_df=gpu_df, tps_df=tps_df, scenarios=scenarios, overrides=overrides, capacity_overrides=capacity_overrides)

    # Derived columns and sanitation
    def gm_pct(row):
        sell = float(row.get("sell_per_1m_eur", 0.0)) or 0.0
        return 0.0 if sell <= 0 else round((float(row.get("margin_per_1m_med", 0.0)) / sell) * 100.0, 2)

    model_df["gross_margin_pct_med"] = model_df.apply(gm_pct, axis=1)
    def gm_pct_min(row):
        sell = float(row.get("sell_per_1m_eur", 0.0)) or 0.0
        return 0.0 if sell <= 0 else round((float(row.get("margin_per_1m_min", 0.0)) / sell) * 100.0, 2)

    model_df["gross_margin_pct_min"] = model_df.apply(gm_pct_min, axis=1)
    for col in [
        "cost_per_1m_min",
        "cost_per_1m_med",
        "cost_per_1m_max",
        "sell_per_1m_eur",
        "margin_per_1m_min",
        "margin_per_1m_med",
        "margin_per_1m_max",
        "gross_margin_pct_med",
        "gross_margin_pct_min",
    ]:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(0.0)

    # Public subset for scenarios and display (exclude private/service SKUs)
    pub_df = model_df[~model_df["model"].astype(str).str.startswith(("private_tap_", "priority_", "oss_support_"))].copy()
    return model_df, pub_df


def _finance_knobs(*, config: Config, scenarios: Dict[str, Any]) -> Tuple[float, Tuple[float, float, float], Dict[str, Any]]:
    marketing_pct = pct_to_fraction(get(config, ["finance", "marketing_allocation_pct_of_inflow"], 0.0), 0.0)
    worst = get_float(scenarios, ["monthly", "worst_m_tokens"], 1.0)
    base = get_float(scenarios, ["monthly", "base_m_tokens"], 5.0)
    best = get_float(scenarios, ["monthly", "best_m_tokens"], 15.0)
    per_model_mix = scenarios.get("per_model_mix", {})  # free-form map
    return marketing_pct, (worst, base, best), per_model_mix


def _loan_and_fixed(*, config: Config, lending: Lending) -> Tuple[Loan, float, float, float, float, float, float]:
    # Loan and fixed
    loan_amount = safe_float(get(lending, ["amount_eur"], 30000.0), 30000.0)
    loan_term = int(safe_float(get(lending, ["term_months"], 60.0), 60.0))
    loan_rate = safe_float(get(lending, ["interest_rate_pct"], 9.95), 9.95)
    loan_obj = Loan(
        principal_eur=loan_amount,
        annual_rate_pct=loan_rate,
        term_months=loan_term,
    )
    monthly_payment, total_repay, total_interest = loan_totals(loan_obj)

    fixed_personal = get_float(config, ["finance", "fixed_costs_monthly_eur", "personal"], 0.0)
    fixed_business = get_float(config, ["finance", "fixed_costs_monthly_eur", "business"], 0.0)
    fixed_total_with_loan = round(fixed_personal + fixed_business + float(monthly_payment), 2)
    return loan_obj, monthly_payment, total_repay, total_interest, fixed_personal, fixed_business, fixed_total_with_loan


def _compute_public_block(
    *,
    pub_df: pd.DataFrame,
    per_model_mix: Dict[str, Any],
    fixed_total_with_loan: float,
    marketing_pct: float,
    worst_base_best: Tuple[float, float, float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Placeholder; marketing_overrides applied by caller when using funnel driver via partial function call
    return compute_public_scenarios(
        pub_df,
        per_model_mix=per_model_mix,
        fixed_total_with_loan=fixed_total_with_loan,
        marketing_pct=marketing_pct,
        worst_base_best=worst_base_best,
    )


def _compute_private_block(*, config: Config, gpu_df: pd.DataFrame, gpu_pricing: Dict[str, Any]) -> Tuple[pd.DataFrame, float, float, float, Dict[str, Any]]:
    eur_usd = safe_float(get(config, ["fx", "eur_usd_rate"]), 1.08)
    fx_buffer_pct = safe_float(get(config, ["pricing_inputs", "fx_buffer_pct"]), 0.0)
    private_markup_pct = safe_float(get(config, ["pricing_inputs", "private_tap_default_markup_over_provider_cost_pct"]), 50.0)
    per_gpu_markup = get(gpu_pricing, ["private_tap_markup_by_gpu"], {}) or {}
    private_df = compute_private_tap_economics(
        gpu_df,
        eur_usd_rate=eur_usd,
        buffer_pct=fx_buffer_pct,
        markup_pct=private_markup_pct,
        markup_by_gpu=per_gpu_markup,
    )
    return private_df, eur_usd, fx_buffer_pct, private_markup_pct, per_gpu_markup

def _compute_break_even_block(*, fixed_total_with_loan: float, public_tpl: Dict[str, Any], marketing_pct: float) -> Dict[str, Any]:
    margin_rate = float(public_tpl.get("blended", {}).get("margin_rate", 0.0))
    return compute_break_even(fixed_total_with_loan, margin_rate, marketing_pct)


def compute_all(
    *,
    config: Config,
    lending: Lending,
    price_sheet: pd.DataFrame,
    gpu_df: pd.DataFrame,
    tps_df: pd.DataFrame,
    scenarios: Dict[str, Any],
    gpu_pricing: Dict[str, Any],
    overrides: Dict[str, Any],
    capacity_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    # Model block
    model_df, pub_df = _compute_model_block(config=config, price_sheet=price_sheet, gpu_df=gpu_df, tps_df=tps_df, scenarios=scenarios, overrides=overrides, capacity_overrides=capacity_overrides)

    # Finance knobs (default from config/scenarios)
    marketing_pct_default, worst_base_best_default, per_model_mix = _finance_knobs(config=config, scenarios=scenarios)

    # Loan and fixed
    (
        loan_obj,
        monthly_payment,
        total_repay,
        total_interest,
        fixed_personal,
        fixed_business,
        fixed_total_with_loan,
    ) = _loan_and_fixed(config=config, lending=lending)

    # Public scenarios
    # Curate models by pricing policy (min gross margin on worst-case bounds)
    pol = (config.get("pricing_policy") or {}).get("public_tap") or {}
    min_gm_pct = float(pol.get("min_gross_margin_pct", 25.0))
    curated_df = pub_df.copy()
    if not curated_df.empty and "gross_margin_pct_min" in curated_df.columns:
        curated_df = curated_df[curated_df["gross_margin_pct_min"] >= min_gm_pct].copy()

    # Determine driver: tokens (default) vs funnel
    driver = str(get(scenarios, ["driver"], "tokens") or "tokens").lower()
    marketing_overrides: Optional[Dict[str, float]] = None
    worst_base_best = worst_base_best_default
    marketing_pct = marketing_pct_default

    funnel_details_base: Optional[Dict[str, float]] = None
    funnel_details_worst: Optional[Dict[str, float]] = None
    funnel_details_best: Optional[Dict[str, float]] = None
    unit_econ: Optional[Dict[str, float]] = None
    unit_econ_worst: Optional[Dict[str, float]] = None
    unit_econ_best: Optional[Dict[str, float]] = None
    if driver == "funnel":
        # Simulate per-case tokens and marketing spend from acquisition inputs
        # Acquisition inputs are passed via compute_all caller; fallback to empty
        acquisition: Dict[str, Any] = get(scenarios, ["__acquisition"], {})  # late-bound by orchestrator wrapper
        funnel_overrides: Dict[str, Any] = get(scenarios, ["__funnel_overrides"], {})
        seasonality: Dict[str, Any] = get(scenarios, ["__seasonality"], {})
        dr = seasonality.get("diminishing_returns") or {}
        try:
            cpc_slope = float(dr.get("cpc_slope_per_extra_1k_eur", 0.0))
        except Exception:
            cpc_slope = 0.0
        m_worst, mk_worst = simulate_m_tokens_from_funnel(
            acquisition=acquisition,
            funnel_overrides=funnel_overrides,
            case="worst",
            cpc_slope_per_extra_1k_eur=cpc_slope,
        )
        m_base, mk_base = simulate_m_tokens_from_funnel(
            acquisition=acquisition,
            funnel_overrides=funnel_overrides,
            case="base",
            cpc_slope_per_extra_1k_eur=cpc_slope,
        )
        m_best, mk_best = simulate_m_tokens_from_funnel(
            acquisition=acquisition,
            funnel_overrides=funnel_overrides,
            case="best",
            cpc_slope_per_extra_1k_eur=cpc_slope,
        )
        worst_base_best = (m_worst, m_base, m_best)
        # Detailed base-case funnel and unit economics
        funnel_details_base = simulate_funnel_details(
            acquisition=acquisition,
            funnel_overrides=funnel_overrides,
            case="base",
            cpc_slope_per_extra_1k_eur=cpc_slope,
        )
        # Also compute worst and best snapshots
        funnel_details_worst = simulate_funnel_details(
            acquisition=acquisition,
            funnel_overrides=funnel_overrides,
            case="worst",
            cpc_slope_per_extra_1k_eur=cpc_slope,
        )
        funnel_details_best = simulate_funnel_details(
            acquisition=acquisition,
            funnel_overrides=funnel_overrides,
            case="best",
            cpc_slope_per_extra_1k_eur=cpc_slope,
        )

    # Compute public scenarios (optionally with marketing overrides)
    public_df, public_tpl = compute_public_scenarios(
        curated_df,
        per_model_mix=per_model_mix,
        fixed_total_with_loan=fixed_total_with_loan,
        marketing_pct=marketing_pct,
        worst_base_best=worst_base_best,
        marketing_overrides=marketing_overrides,
    )

    # Compute unit economics after public_tpl is available (funnel driver only)
    if driver == "funnel":
        try:
            unit_econ = compute_unit_economics(acquisition=acquisition, public_tpl=public_tpl, funnel_details_base=funnel_details_base)
        except Exception:
            unit_econ = None
        # Conservative and optimistic variants
        try:
            if funnel_details_worst is not None:
                unit_econ_worst = compute_unit_economics(
                    acquisition=acquisition,
                    public_tpl=public_tpl,
                    funnel_details_base=funnel_details_worst,
                    overrides={"arpu_multiplier": 0.8, "churn_override_pct": 6.0},
                )
        except Exception:
            unit_econ_worst = None
        try:
            if funnel_details_best is not None:
                unit_econ_best = compute_unit_economics(
                    acquisition=acquisition,
                    public_tpl=public_tpl,
                    funnel_details_base=funnel_details_best,
                    overrides={"arpu_multiplier": 1.1, "churn_override_pct": 2.0},
                )
        except Exception:
            unit_econ_best = None

    # Private economics
    private_df, eur_usd, fx_buffer_pct, private_markup_pct, per_gpu_markup = _compute_private_block(
        config=config, gpu_df=gpu_df, gpu_pricing=gpu_pricing
    )
    # Break-even: if funnel driver, prefer effective marketing % from base case (marketing / revenue)
    if marketing_overrides is not None:
        try:
            rev_base = float(public_tpl.get("base", {}).get("revenue_eur", 0.0))
            mk_base = float(marketing_overrides.get("base", 0.0))
            if rev_base > 0:
                marketing_pct_effective = mk_base / rev_base
            else:
                marketing_pct_effective = marketing_pct
        except Exception:
            marketing_pct_effective = marketing_pct
    else:
        marketing_pct_effective = marketing_pct

    be = _compute_break_even_block(
        fixed_total_with_loan=fixed_total_with_loan, public_tpl=public_tpl, marketing_pct=marketing_pct_effective
    )

    # Loan schedule
    loan_rows = flat_interest_schedule(loan_obj)
    loan_df = pd.DataFrame(loan_rows)

    # 24-month timeseries (public + private + total)
    ts_public_df, ts_private_df, ts_total_df = compute_timeseries(
        agg_inputs={
            "public_tpl": public_tpl,
            "fixed_total_with_loan": fixed_total_with_loan,
            "unit_economics": unit_econ or {},
        },
        scenarios=scenarios,
        config=config,
        gpu_df=gpu_df,
        capacity_overrides=capacity_overrides,
        gpu_pricing=gpu_pricing,
    )

    return {
        "model_df": model_df,
        "pub_df": pub_df,
        "pub_curated_df": curated_df,
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
        "marketing_pct": marketing_pct_effective,
        "marketing_overrides": marketing_overrides,
        "scenarios_driver": driver,
        "funnel_base": funnel_details_base,
        "funnel_worst": funnel_details_worst,
        "funnel_best": funnel_details_best,
        "unit_economics": unit_econ,
        "unit_economics_worst": unit_econ_worst,
        "unit_economics_best": unit_econ_best,
        "ts_public_df": ts_public_df,
        "ts_private_df": ts_private_df,
        "ts_total_df": ts_total_df,
        "policy_public_tap": {"min_gross_margin_pct": min_gm_pct},
    }
