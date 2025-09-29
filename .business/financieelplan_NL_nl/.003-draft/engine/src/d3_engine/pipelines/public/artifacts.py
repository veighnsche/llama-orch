"""Public pipeline computations (pure rows; no I/O)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

from . import pricing as pub_pricing
from . import demand as pub_demand
from . import capacity as pub_capacity
from ...services.batching import effective_tps_per_instance, BatchingConfig
import numpy as np
def _fx_eur_per_usd(facts: Dict[str, Any]) -> float:
    try:
        return float(
            facts.get("market_env", {})
            .get("finance", {})
            .get("eur_usd_fx_rate", {})
            .get("value")
        )
    except Exception:
        return 1.0


def _fx_buffer_pct(pub_op: Dict[str, Any]) -> float:
    try:
        return float(pub_op.get("meta", {}).get("fx_buffer_pct", 0.0))
    except Exception:
        return 0.0


def _eur_hr(usd_hr: float, eur_usd_rate: float, fx_buffer_pct: float) -> float:
    if eur_usd_rate <= 0:
        eur_usd_rate = 1.0
    return (usd_hr / eur_usd_rate) * (1.0 + max(fx_buffer_pct, 0.0) / 100.0)


def _target_margin_cfg(pub_op: Dict[str, Any]) -> Tuple[float, float, float, float]:
    pp = pub_op.get("pricing_policy", {}).get("public_tap", {}) if isinstance(pub_op, dict) else {}
    tm = float(pp.get("target_margin_pct", 55.0))
    inc = float(pp.get("round_increment_eur_per_1k", 0.01))
    floor = float(pp.get("min_floor_eur_per_1k", 0.0))
    cap = float(pp.get("max_cap_eur_per_1k", 1e9))
    return tm, inc, floor, cap


def _typical_vram(model_row: Dict[str, Any]) -> float:
    for key in model_row.keys():
        if str(key).strip().lower().startswith("typical_vram"):
            try:
                return float(str(model_row[key]).strip())
            except Exception:
                return 0.0
    return 0.0


def _choose_gpu_for_model(typ_vram: float, rentals: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [r for r in rentals if float(r.get("vram_gb", 0.0)) >= typ_vram]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r.get("usd_hr", 1e9)))

    # (removed) write_all: module is now pure; use compute_rows() instead


def compute_rows(state: Dict[str, Any]) -> Dict[str, tuple[list[str], list[dict]]]:
    """Pure computation of public rows.

    Returns mapping name -> (headers, rows) without performing any I/O.
    Names correspond to filenames without extension:
      - public_vendor_choice
      - public_tap_prices_per_model
      - public_tap_scenarios
      - public_tap_customers_by_month
      - public_tap_capacity_plan
    """
    vendor_hdr = ["model", "gpu", "provider", "usd_hr", "eur_hr_effective", "cost_eur_per_1M"]
    prices_hdr = ["model", "gpu", "cost_eur_per_1M", "sell_eur_per_1k", "margin_pct"]
    scen_hdr = ["scenario", "month", "tokens", "revenue_eur", "cost_eur", "margin_eur"]
    cust_hdr = ["month", "budget_eur", "cac_eur", "expected_new_customers", "active_customers", "tokens"]
    cap_hdr = [
        "model", "gpu", "avg_tokens_per_hour", "peak_tokens_per_hour", "tps",
        "cap_tokens_per_hour_per_instance", "instances_needed", "target_utilization_pct", "capacity_violation"
    ]

    curated_models: List[Dict[str, Any]] = state.get("curated", {}).get("public_models", [])
    rentals: List[Dict[str, Any]] = state.get("curated", {}).get("gpu_rentals", [])
    pub_op: Dict[str, Any] = state.get("operator", {}).get("public_tap", {})
    facts: Dict[str, Any] = state.get("facts", {})
    tps_rows: List[Dict[str, Any]] = state.get("curated", {}).get("tps_model_gpu", [])

    eur_usd_rate = _fx_eur_per_usd(facts)
    fx_buf = _fx_buffer_pct(pub_op)
    target_margin_pct, round_inc, floor_eur_per_1k, cap_eur_per_1k = _target_margin_cfg(pub_op)

    acq = pub_op.get("acquisition", {}) if isinstance(pub_op, dict) else {}
    budget0 = float(acq.get("budget_month0_eur", 0.0) or 0.0)
    cac = float(acq.get("cac_base_eur", 0.0) or 0.0)
    churn_pct = float(acq.get("churn_pct_mom", 0.0) or 0.0)
    tokens_per_conv = float(acq.get("tokens_per_conversion_mean", 0.0) or 0.0)
    budget_growth_pct_mom = float(acq.get("budget_growth_pct_mom", 0.0) or 0.0)
    peak_factor = float(pub_op.get("autoscaling", {}).get("peak_factor", 1.2) if isinstance(pub_op, dict) else 1.2)
    target_util = float(pub_op.get("autoscaling", {}).get("target_utilization_pct", 75.0) if isinstance(pub_op, dict) else 75.0)
    try:
        horizon = int(state.get("simulation", {}).get("targets", {}).get("horizon_months", 12))
    except Exception:
        horizon = 12

    cfg = BatchingConfig()

    def _tps_eff_for(model: str, gpu: str) -> float:
        for row in tps_rows:
            m = (row.get("model") or row.get("Model") or "").strip()
            g = (row.get("gpu") or row.get("gpu_model") or "").strip()
            if m == model and g == gpu:
                mt = (row.get("measurement_type") or "batched_online").strip()
                try:
                    tps = float((row.get("throughput_tokens_per_sec") or row.get("tps") or 0.0))
                except Exception:
                    tps = 0.0
                try:
                    gpu_count = int((row.get("gpu_count") or 1))
                except Exception:
                    gpu_count = 1
                try:
                    batch = row.get("batch")
                    batch_i = int(batch) if batch not in (None, "") else None
                except Exception:
                    batch_i = None
                return effective_tps_per_instance(mt, tps, gpu_count=gpu_count, batch=batch_i, cfg=cfg)
        return 0.0

    vendor_rows: List[dict] = []
    prices_rows: List[dict] = []
    scen_rows: List[dict] = []
    cust_rows: List[dict] = []
    cap_rows: List[dict] = []

    for m in sorted(curated_models, key=lambda r: (str(r.get("Model") or r.get("model") or "").strip())):
        model_name = (m.get("Model") or m.get("model") or "").strip()
        if not model_name:
            continue
        typ_vram = _typical_vram(m)
        choice = _choose_gpu_for_model(typ_vram, rentals)
        if not choice:
            continue
        gpu = str(choice.get("gpu"))
        provider = str(choice.get("provider"))
        usd_hr = float(choice.get("usd_hr"))
        eur_hr = _eur_hr(usd_hr, eur_usd_rate, fx_buf)

        tps_eff = _tps_eff_for(model_name, gpu) or 1000.0
        tokens_per_hour = tps_eff * 3600.0
        cost_eur_per_1M = (float('inf') if tokens_per_hour <= 0 else (eur_hr / (tokens_per_hour / 1_000_000.0)))
        vendor_rows.append({
            "model": model_name, "gpu": gpu, "provider": provider,
            "usd_hr": f"{usd_hr}", "eur_hr_effective": f"{eur_hr}", "cost_eur_per_1M": f"{cost_eur_per_1M}"
        })

        cost_per_1k = cost_eur_per_1M / 1000.0
        target_margin_frac = max(min(target_margin_pct / 100.0, 0.95), 0.0)
        sell_per_1k = cost_per_1k / max(1e-9, (1.0 - target_margin_frac))
        sell_per_1k = pub_pricing.round_to_increment(sell_per_1k, round_inc)
        if sell_per_1k < floor_eur_per_1k:
            sell_per_1k = floor_eur_per_1k
        if sell_per_1k > cap_eur_per_1k:
            sell_per_1k = cap_eur_per_1k
        margin_pct = 0.0 if sell_per_1k <= 0 else (1.0 - (cost_per_1k / sell_per_1k)) * 100.0
        prices_rows.append({
            "model": model_name, "gpu": gpu, "cost_eur_per_1M": f"{cost_eur_per_1M}",
            "sell_eur_per_1k": f"{sell_per_1k}", "margin_pct": f"{margin_pct}"
        })

        months = np.arange(horizon, dtype=float)
        growth = (1.0 + budget_growth_pct_mom / 100.0)
        budget_series = budget0 * np.power(growth, months)
        expected_new_series = np.where(cac > 0, budget_series / cac, 0.0)
        # active customers via recurrence; keep simple loop for stability
        active = 0.0
        active_series = []
        decay = 1.0 - max(churn_pct, 0.0) / 100.0
        for new_cust in expected_new_series.tolist():
            active = max(0.0, active * decay + new_cust)
            active_series.append(active)
        active_series = np.array(active_series)
        tokens_series = np.maximum(0.0, expected_new_series * tokens_per_conv)
        revenue_series = (tokens_series / 1000.0) * sell_per_1k
        cost_series = (tokens_series / 1_000_000.0) * cost_eur_per_1M
        margin_series = revenue_series - cost_series
        # Emit rows
        for month in range(horizon):
            scen_rows.append({
                "scenario": "base", "month": str(month), "tokens": f"{tokens_series[month]}",
                "revenue_eur": f"{revenue_series[month]}", "cost_eur": f"{cost_series[month]}", "margin_eur": f"{margin_series[month]}"
            })
            cust_rows.append({
                "month": str(month), "budget_eur": f"{budget_series[month]}", "cac_eur": f"{cac}",
                "expected_new_customers": f"{expected_new_series[month]}", "active_customers": f"{active_series[month]}",
                "tokens": f"{tokens_series[month]}"
            })

        expected_new_customers_m0 = 0.0 if cac <= 0 else (budget0 / cac)
        tokens_m0 = max(0.0, expected_new_customers_m0 * tokens_per_conv)
        avg_tph = tokens_m0 / 720.0 if tokens_m0 > 0 else 0.0
        peak_tph = avg_tph * peak_factor
        instances, violation, cap_per_inst = pub_capacity.planner_instances_needed(
            peak_tph, tps_eff, target_util,
            int(pub_op.get("autoscaling", {}).get("min_instances_per_model", 0) if isinstance(pub_op, dict) else 0),
            int(pub_op.get("autoscaling", {}).get("max_instances_per_model", 100) if isinstance(pub_op, dict) else 100)
        )
        cap_rows.append({
            "model": model_name, "gpu": gpu, "avg_tokens_per_hour": f"{avg_tph}", "peak_tokens_per_hour": f"{peak_tph}",
            "tps": f"{tps_eff}", "cap_tokens_per_hour_per_instance": f"{cap_per_inst}", "instances_needed": f"{instances}",
            "target_utilization_pct": f"{target_util}", "capacity_violation": "True" if violation else "False"
        })

    return {
        "public_vendor_choice": (vendor_hdr, vendor_rows),
        "public_tap_prices_per_model": (prices_hdr, prices_rows),
        "public_tap_scenarios": (scen_hdr, scen_rows),
        "public_tap_customers_by_month": (cust_hdr, cust_rows),
        "public_tap_capacity_plan": (cap_hdr, cap_rows),
    }
