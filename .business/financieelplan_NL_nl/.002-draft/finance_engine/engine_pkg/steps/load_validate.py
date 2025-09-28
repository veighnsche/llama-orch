from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import pandas as pd

from ...config import INPUTS, OUTPUTS, ENGINE_VERSION
from ...io.loader import load_yaml, read_csv
from ...io.writer import write_json, write_text, ensure_dir
from ...utils.time import now_utc_iso
from ..ports import ValidatePort, get_default_validator
from ...types.inputs import Config, Costs, Lending


def load_inputs() -> Tuple[
    Dict[str, Any],  # config
    Dict[str, Any],  # costs
    Dict[str, Any],  # lending
    pd.DataFrame,    # price_sheet
    pd.DataFrame,    # gpu_rentals
    pd.DataFrame,    # tps_model_gpu
    Dict[str, Any],  # scenarios
    Dict[str, Any],  # gpu_pricing
    Dict[str, Any],  # capacity_overrides
    Dict[str, Any],  # overrides
    Dict[str, Any],  # acquisition
    Dict[str, Any],  # funnel_overrides
]:
    """
    Load all validated domain inputs. extra.yaml is deprecated and not loaded.

    Returns:
      config, costs, lending, price_sheet, gpu_rentals, tps_model_gpu,
      scenarios, gpu_pricing, capacity_overrides, overrides,
      acquisition, funnel_overrides
    """
    config = load_yaml(INPUTS / "config.yaml") if (INPUTS / "config.yaml").exists() else {}
    costs = load_yaml(INPUTS / "costs.yaml") if (INPUTS / "costs.yaml").exists() else {}
    lending = load_yaml(INPUTS / "lending_plan.yaml") if (INPUTS / "lending_plan.yaml").exists() else {}
    price_sheet = read_csv(INPUTS / "price_sheet.csv")
    gpu_df = read_csv(INPUTS / "gpu_rentals.csv")
    tps_df = read_csv(INPUTS / "tps_model_gpu.csv")
    scenarios = load_yaml(INPUTS / "scenarios.yaml") if (INPUTS / "scenarios.yaml").exists() else {}
    model_pop = load_yaml(INPUTS / "model_popularity.yaml") if (INPUTS / "model_popularity.yaml").exists() else {}
    acquisition = load_yaml(INPUTS / "acquisition.yaml") if (INPUTS / "acquisition.yaml").exists() else {}
    funnel_overrides = load_yaml(INPUTS / "funnel_overrides.yaml") if (INPUTS / "funnel_overrides.yaml").exists() else {}
    gpu_pricing = load_yaml(INPUTS / "gpu_pricing.yaml") if (INPUTS / "gpu_pricing.yaml").exists() else {}
    capacity_overrides = load_yaml(INPUTS / "capacity_overrides.yaml") if (INPUTS / "capacity_overrides.yaml").exists() else {}
    overrides = load_yaml(INPUTS / "overrides.yaml") if (INPUTS / "overrides.yaml").exists() else {}
    # New required inputs
    seasonality = load_yaml(INPUTS / "seasonality.yaml")
    timeseries = load_yaml(INPUTS / "timeseries.yaml")
    billing = load_yaml(INPUTS / "billing.yaml")
    private_sales = load_yaml(INPUTS / "private_sales.yaml")
    competitor_benchmarks = load_yaml(INPUTS / "competitor_benchmarks.yaml") if (INPUTS / "competitor_benchmarks.yaml").exists() else {}

    # Merge pricing policy into config if present
    pricing_policy = load_yaml(INPUTS / "pricing_policy.yaml") if (INPUTS / "pricing_policy.yaml").exists() else {}
    if isinstance(pricing_policy, dict):
        config["pricing_policy"] = pricing_policy

    # Consolidated inputs (preferred if present)
    settings = load_yaml(INPUTS / "settings.yaml") if (INPUTS / "settings.yaml").exists() else {}
    market = load_yaml(INPUTS / "market.yaml") if (INPUTS / "market.yaml").exists() else {}
    finance_yaml = load_yaml(INPUTS / "finance.yaml") if (INPUTS / "finance.yaml").exists() else {}

    # Merge settings into config and gpu_pricing/pricing policy
    if isinstance(settings, dict) and settings:
        for k in ["fx", "prepaid_policy", "pricing_inputs", "catalog", "payments", "tax_billing", "ops"]:
            try:
                v = settings.get(k)
                if isinstance(v, dict) and v:
                    config[k] = v
            except Exception:
                pass
        pol = settings.get("pricing_policy")
        if isinstance(pol, dict) and pol:
            config["pricing_policy"] = pol
        gp = settings.get("gpu_pricing")
        if isinstance(gp, dict) and gp:
            gpu_pricing = gp

    # Merge market inputs
    if isinstance(market, dict) and market:
        scenarios = market.get("scenarios", scenarios)
        acquisition = market.get("acquisition", acquisition)
        funnel_overrides = market.get("funnel_overrides", funnel_overrides)
        seasonality = market.get("seasonality", seasonality)
        competitor_benchmarks = market.get("competitor_benchmarks", competitor_benchmarks)
        # Popularity list may be nested here
        model_pop = market.get("model_popularity", model_pop)
        # Capacity overrides may be provided in consolidated market.yaml
        capacity_overrides = market.get("capacity_overrides", capacity_overrides)

    # Merge finance inputs
    if isinstance(finance_yaml, dict) and finance_yaml:
        lending = finance_yaml.get("lending_plan", lending)
        billing = finance_yaml.get("billing", billing)
        private_sales = finance_yaml.get("private_sales", private_sales)
        timeseries = finance_yaml.get("timeseries", timeseries)
        costs = finance_yaml.get("costs", costs)
    # Merge competitor caps into pricing policy public_tap from competitor_benchmarks
    try:
        per_sku = competitor_benchmarks.get("per_sku") or []
        raw_caps = {str(row.get("sku")): float(row.get("eur_per_1k_tokens")) for row in per_sku if isinstance(row, dict) and row.get("sku") and row.get("eur_per_1k_tokens")}
        # Build a normalized-key map alongside originals to improve matching robustness
        def _norm_cap_key(name: str) -> str:
            s = str(name).replace("Instruct", "").strip()
            s = s.replace("/", " ")
            s = s.replace(".", "-")
            s = "-".join(s.split())
            return s
        caps = dict(raw_caps)
        for k, v in list(raw_caps.items()):
            nk = _norm_cap_key(k)
            caps[nk] = v
        apply_caps = bool((competitor_benchmarks.get("policy") or {}).get("apply_competitor_caps"))
        pp = config.setdefault("pricing_policy", {})
        pt = pp.setdefault("public_tap", {})
        pt["competitor_caps_by_sku"] = caps
        pt["apply_competitor_caps"] = apply_caps
    except Exception:
        pass

    # Derive curated model include list from TPS measurements, intersect with allowed_models if provided
    def _norm(name: str) -> str:
        s = str(name).replace("Instruct", "").strip()
        s = s.replace("/", " ")
        s = s.replace(".", "-")
        s = "-".join(s.split())
        return s

    try:
        tps_models = sorted({_norm(x) for x in tps_df["model_name"].dropna().astype(str).unique().tolist()}) if not tps_df.empty else []
    except Exception:
        tps_models = []
    allowed = []
    try:
        allowed = [str(x) for x in (config.get("catalog", {}).get("allowed_models") or [])]
    except Exception:
        allowed = []
    allowed_norm = {_norm(x) for x in allowed} if allowed else set()
    # Determine include list: prefer popularity list if present
    include = []
    pop_used = False
    try:
        top_models = [str(x) for x in (model_pop.get("top_models") or [])]
        max_count = int(model_pop.get("max_count") or 0)
        if top_models:
            top_norm = [_norm(x) for x in top_models]
            # Keep order from top_models and intersect with measured TPS models
            ordered = [m for m in top_norm if m in tps_models]
            if max_count and max_count > 0:
                ordered = ordered[:max_count]
            if ordered:
                include = ordered
                pop_used = True
    except Exception:
        pass
    if not include:
        if allowed_norm:
            include = [m for m in tps_models if m in allowed_norm]
        else:
            include = tps_models
    if isinstance(scenarios, dict):
        scenarios = {
            **scenarios,
            "include_skus": include,
            # Selection metadata
            "selection_mode": ("popularity" if pop_used else scenarios.get("selection_mode", "tps_measured")),
            "curate_models_by_min_gross_margin": scenarios.get("curate_models_by_min_gross_margin", (False if pop_used else True)),
            # Inject new required control surfaces for downstream compute
            "__acquisition": acquisition,
            "__funnel_overrides": funnel_overrides,
            "__seasonality": seasonality,
            "__timeseries": timeseries,
            "__billing": billing,
            "__private_sales": private_sales,
        }
    else:
        scenarios = {
            "include_skus": include,
            "selection_mode": ("popularity" if pop_used else "tps_measured"),
            "curate_models_by_min_gross_margin": (False if pop_used else True),
            "__acquisition": acquisition,
            "__funnel_overrides": funnel_overrides,
            "__seasonality": seasonality,
            "__timeseries": timeseries,
            "__billing": billing,
            "__private_sales": private_sales,
        }

    return (
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
        acquisition,
        funnel_overrides,
    )


def write_run_summary() -> Dict[str, Any]:
    ensure_dir(OUTPUTS)
    payload: Dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "run_at": now_utc_iso(),
        "notes": ["engine_pkg orchestrator"],
    }
    write_json(OUTPUTS / "run_summary.json", payload)
    md_path = OUTPUTS / "run_summary.md"
    md_body = "\n".join([
        "# Run Summary",
        f"- Engine version: {ENGINE_VERSION}",
        f"- Run at (UTC): {payload['run_at']}",
    ]) + "\n"
    if md_path.exists():
        # Append to preserve any existing Preflight section at the top
        existing = md_path.read_text(encoding="utf-8")
        write_text(md_path, existing + ("\n" if not existing.endswith("\n") else "") + md_body)
    else:
        write_text(md_path, md_body)
    return payload


def validate_and_write_report(
    config: Config,
    costs: Costs,
    lending: Lending,
    price_sheet: pd.DataFrame,
    *,
    validate_port: Optional[ValidatePort] = None,
) -> bool:
    # Prefer injected validator; otherwise use default implementation.
    validator: ValidatePort = validate_port or get_default_validator()
    report = validator(config=config, costs=costs, lending=lending, price_sheet=price_sheet)
    write_json(OUTPUTS / "validation_report.json", report)
    return bool(report.get("errors"))
