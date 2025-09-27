from __future__ import annotations

from typing import TypedDict, Dict, Any, Mapping, Optional


class FinanceFixedCosts(TypedDict, total=False):
    personal: float
    business: float


class FinanceConfig(TypedDict, total=False):
    marketing_allocation_pct_of_inflow: float
    fixed_costs_monthly_eur: FinanceFixedCosts


class FXConfig(TypedDict, total=False):
    eur_usd_rate: float


class PricingInputsConfig(TypedDict, total=False):
    fx_buffer_pct: float
    private_tap_default_markup_over_provider_cost_pct: float


class PrivateTapConfig(TypedDict, total=False):
    management_fee_eur_per_month: Optional[float]


class PrepaidPrivateTapConfig(TypedDict, total=False):
    billing_unit_minutes: int


class PrepaidConfig(TypedDict, total=False):
    min_topup_eur: int
    max_topup_eur: int
    expiry_months: int
    non_refundable: bool
    auto_refill_default_enabled: bool
    auto_refill_cap_eur: int
    private_tap: PrepaidPrivateTapConfig


class TaxBillingConfig(TypedDict, total=False):
    vat_standard_rate_pct: float


class Config(TypedDict, total=False):
    finance: FinanceConfig
    fx: FXConfig
    pricing_inputs: PricingInputsConfig
    private: PrivateTapConfig
    prepaid: PrepaidConfig
    tax_billing: TaxBillingConfig


class Costs(TypedDict, total=False):
    # Left open; keyed by model/gpu/quantity etc.
    __any__: Any


class Lending(TypedDict, total=False):
    amount_eur: float
    term_months: int
    interest_rate_pct: float


class ScenarioMonthly(TypedDict, total=False):
    worst_m_tokens: float
    base_m_tokens: float
    best_m_tokens: float


class Scenarios(TypedDict, total=False):
    monthly: ScenarioMonthly


class Extra(TypedDict, total=False):
    scenarios: Scenarios
    per_model_mix: Dict[str, float]
    prepaid: PrepaidConfig
    private: PrivateTapConfig
    loan: Lending
    pricing_inputs: Dict[str, Any]
    price_overrides: Dict[str, Any]
    runway_target_months: int
