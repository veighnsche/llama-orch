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
    config = load_yaml(INPUTS / "config.yaml")
    costs = load_yaml(INPUTS / "costs.yaml")
    lending = load_yaml(INPUTS / "lending_plan.yaml")
    price_sheet = read_csv(INPUTS / "price_sheet.csv")
    gpu_df = read_csv(INPUTS / "gpu_rentals.csv")
    tps_df = read_csv(INPUTS / "tps_model_gpu.csv")
    scenarios = load_yaml(INPUTS / "scenarios.yaml")
    acquisition = load_yaml(INPUTS / "acquisition.yaml")
    funnel_overrides = load_yaml(INPUTS / "funnel_overrides.yaml") if (INPUTS / "funnel_overrides.yaml").exists() else {}
    gpu_pricing = load_yaml(INPUTS / "gpu_pricing.yaml")
    capacity_overrides = load_yaml(INPUTS / "capacity_overrides.yaml")
    overrides = load_yaml(INPUTS / "overrides.yaml") if (INPUTS / "overrides.yaml").exists() else {}
    # New required inputs
    seasonality = load_yaml(INPUTS / "seasonality.yaml")
    timeseries = load_yaml(INPUTS / "timeseries.yaml")
    billing = load_yaml(INPUTS / "billing.yaml")
    private_sales = load_yaml(INPUTS / "private_sales.yaml")
    competitor_benchmarks = load_yaml(INPUTS / "competitor_benchmarks.yaml")

    # Merge pricing policy into config if present
    pricing_policy = load_yaml(INPUTS / "pricing_policy.yaml") if (INPUTS / "pricing_policy.yaml").exists() else {}
    if isinstance(pricing_policy, dict):
        config["pricing_policy"] = pricing_policy
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
    if allowed_norm:
        include = [m for m in tps_models if m in allowed_norm]
    else:
        include = tps_models
    if isinstance(scenarios, dict):
        scenarios = {
            **scenarios,
            "include_skus": include,
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
