"""Public artifacts writing (MRPT v0)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

from . import pricing as pub_pricing
from . import demand as pub_demand
from . import capacity as pub_capacity


def _write_csv_header(path: Path, headers: list[str]) -> None:
    if not path.exists():
        path.write_text(",".join(headers) + "\n")


def _append_row(path: Path, row: List[str]) -> None:
    with path.open("a") as f:
        f.write(",".join(row) + "\n")


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


def write_all(out_dir: Path, state: Dict[str, Any]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Prepare headers
    vendor_hdr = ["model", "gpu", "provider", "usd_hr", "eur_hr_effective", "cost_eur_per_1M"]
    prices_hdr = ["model", "gpu", "cost_eur_per_1M", "sell_eur_per_1k", "margin_pct"]
    scen_hdr = ["scenario", "month", "tokens", "revenue_eur", "cost_eur", "margin_eur"]
    cust_hdr = ["month", "budget_eur", "cac_eur", "expected_new_customers", "active_customers", "tokens"]
    cap_hdr = [
        "model", "gpu", "avg_tokens_per_hour", "peak_tokens_per_hour", "tps",
        "cap_tokens_per_hour_per_instance", "instances_needed", "target_utilization_pct", "capacity_violation"
    ]

    vendor_path = out_dir / "public_vendor_choice.csv"
    prices_path = out_dir / "public_tap_prices_per_model.csv"
    scen_path = out_dir / "public_tap_scenarios.csv"
    cust_path = out_dir / "public_tap_customers_by_month.csv"
    cap_path = out_dir / "public_tap_capacity_plan.csv"

    for p, h in (
        (vendor_path, vendor_hdr),
        (prices_path, prices_hdr),
        (scen_path, scen_hdr),
        (cust_path, cust_hdr),
        (cap_path, cap_hdr),
    ):
        _write_csv_header(p, h)
        written.append(p.name)

    # Read config/state
    curated_models: List[Dict[str, Any]] = state.get("curated", {}).get("public_models", [])
    rentals: List[Dict[str, Any]] = state.get("curated", {}).get("gpu_rentals", [])
    pub_op: Dict[str, Any] = state.get("operator", {}).get("public_tap", {})
    facts: Dict[str, Any] = state.get("facts", {})

    eur_usd_rate = _fx_eur_per_usd(facts)
    fx_buf = _fx_buffer_pct(pub_op)
    target_margin_pct, round_inc, floor_eur_per_1k, cap_eur_per_1k = _target_margin_cfg(pub_op)

    # Simple acquisition defaults
    acq = pub_op.get("acquisition", {}) if isinstance(pub_op, dict) else {}
    budget0 = float(acq.get("budget_month0_eur", 0.0) or 0.0)
    cac = float(acq.get("cac_base_eur", 0.0) or 0.0)
    churn_pct = float(acq.get("churn_pct_mom", 0.0) or 0.0)
    tokens_per_conv = float(acq.get("tokens_per_conversion_mean", 0.0) or 0.0)
    peak_factor = float(pub_op.get("autoscaling", {}).get("peak_factor", 1.2) if isinstance(pub_op, dict) else 1.2)
    target_util = float(pub_op.get("autoscaling", {}).get("target_utilization_pct", 75.0) if isinstance(pub_op, dict) else 75.0)

    # Assume tps_eff baseline when TPS dataset is absent
    tps_eff = 1000.0  # tokens per second
    cap_per_inst = pub_capacity.cap_per_instance_tokens_per_hour(tps_eff, target_util)

    for m in curated_models:
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

        # Heuristic: without TPS dataset, approximate cost per 1M as eur_hr (1e6 tokens/hour baseline)
        cost_eur_per_1M = eur_hr
        _append_row(vendor_path, [
            model_name, gpu, provider, f"{usd_hr}", f"{eur_hr}", f"{cost_eur_per_1M}"
        ])

        # Derive sell €/1k for target margin
        cost_per_1k = cost_eur_per_1M / 1000.0
        target_margin_frac = max(min(target_margin_pct / 100.0, 0.95), 0.0)
        # sell = cost / (1 - margin)
        sell_per_1k = cost_per_1k / max(1e-9, (1.0 - target_margin_frac))
        sell_per_1k = pub_pricing.round_to_increment(sell_per_1k, round_inc)
        if sell_per_1k < floor_eur_per_1k:
            sell_per_1k = floor_eur_per_1k
        if sell_per_1k > cap_eur_per_1k:
            sell_per_1k = cap_eur_per_1k
        margin_pct = 0.0 if sell_per_1k <= 0 else (1.0 - (cost_per_1k / sell_per_1k)) * 100.0
        _append_row(prices_path, [
            model_name, gpu, f"{cost_eur_per_1M}", f"{sell_per_1k}", f"{margin_pct}"
        ])

        # Simple month 0 scenario and customers series
        month = 0
        expected_new_customers = 0.0 if cac <= 0 else (budget0 / cac)
        active_customers = expected_new_customers * (1.0 - max(churn_pct, 0.0) / 100.0)
        tokens_m = expected_new_customers * tokens_per_conv
        revenue_eur = (tokens_m / 1000.0) * sell_per_1k
        cost_eur = (tokens_m / 1_000_000.0) * cost_eur_per_1M
        margin_eur = revenue_eur - cost_eur
        _append_row(scen_path, [
            "base", str(month), f"{tokens_m}", f"{revenue_eur}", f"{cost_eur}", f"{margin_eur}"
        ])
        _append_row(cust_path, [
            str(month), f"{budget0}", f"{cac}", f"{expected_new_customers}", f"{active_customers}", f"{tokens_m}"
        ])

        # Capacity plan
        avg_tph = tokens_m / 720.0 if tokens_m > 0 else 0.0
        peak_tph = avg_tph * peak_factor
        instances, violation, _ = pub_capacity.planner_instances_needed(
            peak_tph, tps_eff, target_util, int(pub_op.get("autoscaling", {}).get("min_instances_per_model", 0) if isinstance(pub_op, dict) else 0), int(pub_op.get("autoscaling", {}).get("max_instances_per_model", 100) if isinstance(pub_op, dict) else 100)
        )
        _append_row(cap_path, [
            model_name, gpu, f"{avg_tph}", f"{peak_tph}", f"{tps_eff}", f"{cap_per_inst}", f"{instances}", f"{target_util}", "True" if violation else "False"
        ])

    return written
