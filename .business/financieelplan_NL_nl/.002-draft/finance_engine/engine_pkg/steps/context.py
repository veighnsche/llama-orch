from __future__ import annotations

from typing import Any, Dict

from ...config import ENGINE_VERSION
from ...utils.time import now_utc_iso
from ...utils.markdown import df_to_markdown_rows
from ...utils.vat import vat_examples
from ...utils.scenarios import monthly_yearly_sixty
from ...types.inputs import Config, Lending


def build_context(*, agg: Dict[str, Any], charts: Dict[str, str], config: Config, overrides: Dict[str, Any], lending: Lending, scenarios: Dict[str, Any]) -> Dict[str, Any]:
    # Tables for markdown rendering
    model_cols = [
        "model",
        "gpu",
        "cost_per_1m_min",
        "cost_per_1m_med",
        "cost_per_1m_max",
        "sell_per_1m_eur",
        "margin_per_1m_med",
        "gross_margin_pct_med",
    ]
    model_table_md = df_to_markdown_rows(agg["pub_df"], model_cols) if not agg["pub_df"].empty else ""

    private_cols = [
        "gpu",
        "provider_eur_hr_med",
        "markup_pct",
        "sell_eur_hr",
        "margin_eur_hr",
    ]
    private_table_md = df_to_markdown_rows(agg["private_df"], private_cols) if not agg["private_df"].empty else ""

    # Finance / pricing knobs
    raw_fx_buf = config.get("pricing_inputs", {}).get("fx_buffer_pct")
    fx_buffer_pct = float(raw_fx_buf if isinstance(raw_fx_buf, (int, float, str)) else 0.0)
    raw_priv_markup = config.get("pricing_inputs", {}).get("private_tap_default_markup_over_provider_cost_pct")
    private_markup_default = float(raw_priv_markup if isinstance(raw_priv_markup, (int, float, str)) else 50.0)
    price_overrides = overrides.get("price_overrides", {}) if isinstance(overrides, dict) else {}
    # Normalize overrides to a flat map of {model: unit_price_per_1k_tokens}
    override_prices_num: Dict[str, float] = {}
    if isinstance(price_overrides, dict) and price_overrides:
        for sku, node in price_overrides.items():
            if isinstance(node, dict) and "unit_price_eur_per_1k_tokens" in node:
                try:
                    override_prices_num[sku] = float(node["unit_price_eur_per_1k_tokens"])  # per 1k tokens
                except Exception:
                    pass

    # Break-even targets
    targets = {
        "required_prepaid_inflow_eur": agg["break_even"].get("required_inflow_eur"),
        "required_margin_eur": agg["break_even"].get("required_margin_eur"),
    }

    # Fixed costs (needed for scenario-derived sections)
    fixed = {
        "personal": agg.get("fixed_personal", 0.0),
        "business": agg.get("fixed_business", 0.0),
        "total_with_loan": agg.get("fixed_total_with_loan", 0.0),
    }

    # Scenarios for sections 3 and 5
    scenarios_tpl = agg.get("public_tpl", {})
    monthly, yearly, sixty_m = monthly_yearly_sixty(scenarios_tpl, fixed.get("total_with_loan", 0.0))

    # Loan
    loan_amount = lending.get("amount_eur")
    if loan_amount is None:
        loan_amount = lending.get("loan_request", {}).get("amount_eur")
    term_months = lending.get("term_months")
    if term_months is None:
        term_months = lending.get("repayment_plan", {}).get("term_months")
    interest_rate_pct = lending.get("interest_rate_pct")
    if interest_rate_pct is None:
        interest_rate_pct = lending.get("repayment_plan", {}).get("interest_rate_pct")

    loan = {
        "amount_eur": loan_amount,
        "term_months": term_months,
        "interest_rate_pct": interest_rate_pct,
        "monthly_payment_eur": agg.get("monthly_payment"),
        "total_repayment_eur": agg.get("total_repay"),
        "total_interest_eur": agg.get("total_interest"),
    }

    # Derive prices from computed model economics if available (sell_per_1m_eur / 1000)
    derived_prices: Dict[str, float] = {}
    try:
        mdf = agg.get("model_df")
        if mdf is not None and not mdf.empty:
            for _, row in mdf.iterrows():
                model = str(row.get("model"))
                sell_1m = float(row.get("sell_per_1m_eur", 0.0))
                if sell_1m > 0:
                    derived_prices[model] = round(sell_1m / 1000.0, 4)
    except Exception:
        pass

    model_prices_map = {**derived_prices, **override_prices_num}  # overrides take precedence

    # Compute blended public price per 1k tokens for headline
    blended_1k = None
    try:
        blended_sell_1m = float(agg.get("public_tpl", {}).get("blended", {}).get("sell_per_1m_eur", 0.0))
        if blended_sell_1m > 0:
            blended_1k = round(blended_sell_1m / 1000.0, 4)
    except Exception:
        blended_1k = None

    ctx = {
        "engine": {"version": ENGINE_VERSION, "timestamp": now_utc_iso()},
        "pricing": {
            "fx_buffer_pct": fx_buffer_pct,
            "private_tap_default_markup_over_provider_cost_pct": private_markup_default,
            # No flat public price; per-model prices derived from policy and costs
            "public_tap_flat_price_per_1k_tokens": None,
            "public_tap_blended_price_per_1k_tokens": blended_1k,
            "model_prices": model_prices_map,
            "policy": (config.get("pricing_policy") or {}).get("public_tap") or {},
        },
        "prepaid": {
            "min_topup_eur": config.get("prepaid_policy", {}).get("credits", {}).get("min_topup_eur", 25),
            "max_topup_eur": config.get("prepaid_policy", {}).get("credits", {}).get("max_topup_eur", 10000),
            "expiry_months": config.get("prepaid_policy", {}).get("credits", {}).get("expiry_months", 12),
            "non_refundable": config.get("prepaid_policy", {}).get("credits", {}).get("non_refundable", True),
            "auto_refill_default_enabled": config.get("prepaid_policy", {}).get("credits", {}).get("auto_refill_default_enabled", False),
            "auto_refill_cap_eur": config.get("prepaid_policy", {}).get("credits", {}).get("auto_refill_cap_eur", 200),
            "private_tap": {
                "billing_unit_minutes": config.get("prepaid_policy", {}).get("private_tap", {}).get("billing_unit_minutes", 60),
            },
        },
        "tax": {
            "vat_standard_rate_pct": int(round(agg.get("vat_rate", 21.0))),
            "eu_reverse_charge_enabled": True,
            "stripe_tax_enabled": True,
            "revenue_recognition": "prepaid liability until consumed",
            "example": vat_examples(agg.get("vat_rate", 21.0)),
        },
        "finance": {
            "marketing_allocation_pct_of_inflow": int(round(agg.get("marketing_pct", 0.0) * 100.0)),
            "runway_target_months": config.get("finance", {}).get("runway_target_months", 12),
        },
        "fx": {"rate_used": config.get("fx", {}).get("eur_usd_rate", 1.08)},
        "private": {
            "management_fee_eur_per_month": (
                float(config.get("prepaid_policy", {}).get("private_tap", {}).get("management_fee_eur_per_month"))
                if isinstance(config.get("prepaid_policy", {}).get("private_tap", {}).get("management_fee_eur_per_month"), (int, float))
                else config.get("prepaid_policy", {}).get("private_tap", {}).get("management_fee_eur_per_month")
            ),
            "default_markup_over_cost_pct": config.get("pricing_inputs", {}).get("private_tap_default_markup_over_provider_cost_pct")
            or config.get("prepaid_policy", {}).get("private_tap", {}).get("gpu_hour_markup_target_pct"),
        },
        "catalog": {
            # Show curated public models if present, else all public models
            "allowed_models": (
                list(agg.get("pub_curated_df", agg.get("pub_df", []) )["model"].astype(str).unique())
                if isinstance(agg.get("pub_curated_df"), type(agg.get("pub_df"))) and not agg.get("pub_curated_df").empty
                else (list(agg["pub_df"]["model"].astype(str).unique()) if not agg["pub_df"].empty else [])
            ),
            "allowed_gpus": list(agg["private_df"]["gpu"].astype(str).unique()) if not agg["private_df"].empty else [],
        },
        "fixed": fixed,
        "loan": loan,
        "targets": targets,
        "scenarios": {
            "worst": scenarios_tpl.get("worst", {}),
            "base": scenarios_tpl.get("base", {}),
            "best": scenarios_tpl.get("best", {}),
            "monthly": monthly,
            "yearly": yearly,
            "60m": sixty_m,
            "include_skus": scenarios.get("include_skus"),
            "per_model_mix": scenarios.get("per_model_mix", {}),
        },
        "tables": {
            "model_price_per_1m_tokens": model_table_md,
            "private_tap_gpu_economics": private_table_md,
            "private_tap_example_pack": "",  # not implemented yet
            "loan_schedule": "",  # not expanded to markdown; chart will visualize
        },
        "charts": charts,
    }

    return ctx
