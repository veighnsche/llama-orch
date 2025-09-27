from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import pandas as pd

from ..config import OUTPUTS
from ..io.writer import write_text, ensure_dir
from .ports import RenderPort, ValidatePort
from ..types.inputs import Config, Costs, Lending
from .validation.preflight import run_preflight, build_preflight_markdown

# Step modules (keep runner thin and modular)
from .steps.load_validate import (
    load_inputs as _load_inputs,
    write_run_summary as _write_run_summary,
    validate_and_write_report as _validate_and_write_report,
)
from .steps.compute import compute_all as _compute_all
from .steps.artifacts import write_artifacts as _write_artifacts
from .steps.charts import generate_charts as _generate_charts
from .steps.context import build_context as _build_context
from .steps.render import render_plan as _render_plan


def load_inputs() -> Tuple[Config, Costs, Lending, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return _load_inputs()


def write_run_summary() -> Dict[str, Any]:
    return _write_run_summary()


def validate_and_write_report(
    config: Config,
    costs: Costs,
    lending: Lending,
    price_sheet: pd.DataFrame,
    *,
    validate_port: Optional[ValidatePort] = None,
) -> bool:
    return _validate_and_write_report(config, costs, lending, price_sheet, validate_port=validate_port)


# Pipeline steps to be filled in small patches

def compute_all(config: Config, lending: Lending, price_sheet: pd.DataFrame, gpu_df: pd.DataFrame, *, tps_df: pd.DataFrame, scenarios: Dict[str, Any], gpu_pricing: Dict[str, Any], overrides: Dict[str, Any], capacity_overrides: Dict[str, Any]) -> Dict[str, Any]:
    return _compute_all(
        config=config,
        lending=lending,
        price_sheet=price_sheet,
        gpu_df=gpu_df,
        tps_df=tps_df,
        scenarios=scenarios,
        gpu_pricing=gpu_pricing,
        overrides=overrides,
        capacity_overrides=capacity_overrides,
    )


def write_artifacts(config: Config, agg: Dict[str, Any]) -> None:
    _write_artifacts(config=config, agg=agg)


def generate_charts(agg: Dict[str, Any]) -> Dict[str, str]:
    return _generate_charts(agg=agg)


def build_context(agg: Dict[str, Any], charts: Dict[str, str], config: Config, *, overrides: Dict[str, Any], lending: Lending) -> Dict[str, Any]:
    # Synthesize a minimal extra-like dict for context rendering from domain files
    extra_ctx: Dict[str, Any] = {
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
        "private": {
            "management_fee_eur_per_month": config.get("prepaid_policy", {}).get("private_tap", {}).get("management_fee_eur_per_month", None),
        },
        "runway_target_months": config.get("finance", {}).get("runway_target_months", 12),
        "price_overrides": overrides.get("price_overrides", {}),
    }
    return _build_context(agg=agg, charts=charts, config=config, extra=extra_ctx, lending=lending)


def render_plan(context: Dict[str, Any], *, render_port: Optional[RenderPort] = None) -> None:
    _render_plan(context=context, render_port=render_port)


def run_pipeline(*, render_port: Optional[RenderPort] = None, validate_port: Optional[ValidatePort] = None) -> int:
    # Preflight validation
    ensure_dir(OUTPUTS)
    pre = run_preflight()
    pre_md = build_preflight_markdown(pre)
    write_text(OUTPUTS / "run_summary.md", pre_md)
    if not pre.ok:
        # Fail-fast before any computations
        return 1

    (
        config,
        costs,
        lending,
        price_sheet,
        gpu_df,
        tps_df,
        scenarios,
        gpu_pricing,
        capacity_overrides,
        overrides,
    ) = load_inputs()
    write_run_summary()
    has_errors = validate_and_write_report(config, costs, lending, price_sheet, validate_port=validate_port)
    if has_errors:
        return 1
    agg = compute_all(
        config,
        lending,
        price_sheet,
        gpu_df,
        tps_df=tps_df,
        scenarios=scenarios,
        gpu_pricing=gpu_pricing,
        overrides=overrides,
        capacity_overrides=capacity_overrides,
    )
    write_artifacts(config, agg)
    charts = generate_charts(agg)
    ctx = build_context(agg=agg, charts=charts, config=config, overrides=overrides, lending=lending, scenarios=scenarios)
    render_plan(context=ctx, render_port=render_port)
    # Legacy shim for older tests
    write_text(OUTPUTS / "template_filled.md", "Generated by engine v1\n")
    return 0
