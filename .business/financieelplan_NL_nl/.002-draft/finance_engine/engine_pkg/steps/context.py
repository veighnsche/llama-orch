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
    public_flat_1k = None
    price_overrides = overrides.get("price_overrides", {}) if isinstance(overrides, dict) else {}
    if isinstance(price_overrides, dict) and price_overrides:
        try:
            any_model = next(iter(price_overrides.values()))
            public_flat_1k = any_model.get("unit_price_eur_per_1k_tokens")
        except Exception:
            public_flat_1k = None

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
    loan = {
        "amount_eur": lending.get("amount_eur"),
        "term_months": lending.get("term_months"),
        "interest_rate_pct": lending.get("interest_rate_pct"),
        "monthly_payment_eur": agg.get("monthly_payment"),
        "total_repayment_eur": agg.get("total_repay"),
        "total_interest_eur": agg.get("total_interest"),
    }

    ctx = {
        "engine": {"version": ENGINE_VERSION, "timestamp": now_utc_iso()},
        "pricing": {
            "fx_buffer_pct": fx_buffer_pct,
            "private_tap_default_markup_over_provider_cost_pct": private_markup_default,
            "public_tap_flat_price_per_1k_tokens": public_flat_1k,
            "model_prices": price_overrides,
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
            "management_fee_eur_per_month": config.get("prepaid_policy", {}).get("private_tap", {}).get("management_fee_eur_per_month", None),
        },
        "catalog": {
            "allowed_models": list(agg["pub_df"]["model"].astype(str).unique()) if not agg["pub_df"].empty else [],
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
