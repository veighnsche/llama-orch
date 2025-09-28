from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import math

import pandas as pd


def _get_fx_and_buffer(cfg: Dict[str, Any]) -> Tuple[float, float]:
    eur_usd = cfg.get("fx", {}).get("eur_usd_rate") or 1.08
    buffer_pct = cfg.get("pricing_inputs", {}).get("fx_buffer_pct", 0)
    try:
        buffer_val = float(buffer_pct)
    except Exception:
        buffer_val = 0.0
    return float(eur_usd), buffer_val


def _select_models(*, scenarios: Dict[str, Any], price_sheet: pd.DataFrame, overrides: Dict[str, Any]) -> List[str]:
    # Priority: explicit include list set by loader (derived from OSS catalog), then overrides as a convenience.
    incl = scenarios.get("include_skus") or []
    if incl:
        return [str(m) for m in incl]
    ov = overrides.get("price_overrides", {}) if isinstance(overrides, dict) else {}
    if ov:
        return list(ov.keys())
    # As a last resort, fall back to SKUs listed in price_sheet (but pricing will still be computed, not read).
    if "sku" in price_sheet.columns:
        return price_sheet["sku"].astype(str).unique().tolist()
    return []


def _normalize_model_name(name: str) -> str:
    # Rough normalization to connect tps_model_gpu.csv model_name and price_sheet sku
    s = str(name).replace("Instruct", "").strip()
    s = s.replace("/", " ")
    s = s.replace(".", "-")
    s = "-".join(s.split())
    return s


def _policy_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pol = (cfg.get("pricing_policy") or {}).get("public_tap") or {}
    return {
        "target_gross_margin_pct": float(pol.get("target_gross_margin_pct", 45.0)),
        "min_gross_margin_pct": float(pol.get("min_gross_margin_pct", 25.0)),
        "rounding_per_1k_tokens": float(pol.get("rounding_per_1k_tokens", 0.01)),
        "price_floor_per_1k_tokens": float(pol.get("price_floor_per_1k_tokens", 0.01)),
        "price_cap_per_1k_tokens": (None if pol.get("price_cap_per_1k_tokens") in (None, "",) else float(pol.get("price_cap_per_1k_tokens"))),
        "competitor_caps_by_sku": pol.get("competitor_caps_by_sku") or {},
        "apply_competitor_caps": bool(pol.get("apply_competitor_caps", False)),
        # Optimizer settings (optional)
        "optimize": (pol.get("optimize") or pol.get("optimize_price") or {}),
    }


def _round_price_per_1k(x: float, step: float, floor: float, cap: Optional[float]) -> float:
    if step <= 0:
        y = x
    else:
        y = round(x / step) * step
    y = max(y, floor)
    if cap is not None:
        y = min(y, cap)
    return round(y, 6)


def _pick_best_gpu_and_tps(
    model: str,
    tps_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    cfg: Dict[str, Any],
    capacity_overrides: Dict[str, Any],
) -> Optional[Tuple[str, float, float, float, float]]:
    """Return (gpu_name, tps, eur_hr_min, eur_hr_med, eur_hr_max) for the cost-optimal GPU.
    Excludes the model if no TPS measurements or no matching GPU rentals exist.
    """
    ov = capacity_overrides.get("capacity_overrides", {}) if isinstance(capacity_overrides, dict) else {}
    preferred_gpu = None
    node = ov.get(model)
    if isinstance(node, dict) and node.get("preferred_gpu"):
        preferred_gpu = str(node.get("preferred_gpu"))

    if tps_df is None or tps_df.empty:
        return None
    norm = _normalize_model_name
    sub = tps_df[tps_df["model_name"].astype(str).map(norm) == norm(model)].copy()
    if sub.empty:
        return None

    # If a TPS override is present, use it with preferred GPU if set; else still need a GPU name from TPS data
    tps_override = None
    if isinstance(node, dict) and node.get("tps_override_tokens_per_sec") is not None:
        try:
            tps_override = float(node.get("tps_override_tokens_per_sec"))
        except Exception:
            tps_override = None

    eur_usd_rate, buffer_pct = _get_fx_and_buffer(cfg)
    # Effective utilization and non-GPU overhead multiplier for conservative costing
    try:
        util_pct = float((cfg.get("pricing_inputs") or {}).get("effective_utilization_pct", 10.0))
    except Exception:
        util_pct = 10.0
    try:
        overhead_mult = float((cfg.get("pricing_inputs") or {}).get("non_gpu_overhead_multiplier_on_cost", 10.0))
    except Exception:
        overhead_mult = 10.0

    def to_eur_hr(usd_hr: float) -> float:
        return (float(usd_hr) / float(eur_usd_rate)) * (1.0 + float(buffer_pct) / 100.0)

    # Decide a single allowed measurement type across all GPUs for this model
    try:
        mt_all = sub["measurement_type"].astype(str).str.lower().unique().tolist()
    except Exception:
        mt_all = []
    if "single_stream" in mt_all:
        allowed_mt = "single_stream"
    elif "per_user_stream" in mt_all:
        allowed_mt = "per_user_stream"
    else:
        allowed_mt = "aggregate"

    # Group by GPU in TPS and evaluate cost per 1M for each
    best: Optional[Tuple[str, float, float, float, float, float]] = None  # (gpu, tps, eur_hr_min, eur_hr_med, eur_hr_max, c_med)
    for gpu_name in sorted(set(sub["gpu"].dropna().astype(str))):
        # If preferred_gpu is set and doesn't match (substring), skip
        if preferred_gpu and preferred_gpu.lower() not in gpu_name.lower():
            continue
        # TPS: use override if provided; otherwise use allowed measurement type and normalize to per-GPU
        s_gpu = sub[sub["gpu"].astype(str) == gpu_name].copy()
        if tps_override is None:
            s_mt = s_gpu[s_gpu["measurement_type"].astype(str).str.lower() == allowed_mt]
            if s_mt.empty:
                continue
            # Normalize per GPU using gpu_count if present
            try:
                per_gpu_series = pd.to_numeric(s_mt["throughput_tokens_per_sec"], errors="coerce").fillna(0.0) / s_mt.get("gpu_count", 1).replace(0, 1)
            except Exception:
                per_gpu_series = pd.to_numeric(s_mt["throughput_tokens_per_sec"], errors="coerce").fillna(0.0)
            per_gpu_series = per_gpu_series[pd.to_numeric(per_gpu_series, errors="coerce") > 0]
            if per_gpu_series.empty:
                continue
            tps = float(per_gpu_series.median())
        else:
            tps = float(tps_override)
        if tps <= 0:
            continue
        # Match GPU rentals rows by substring; evaluate all matches and pick the cheapest cost per 1M
        mask = gpu_df["gpu"].astype(str).str.contains(gpu_name, case=False, regex=False)
        if not mask.any():
            # Try partial tokens (e.g., "A100 80G" -> match "A100 80GB")
            parts = [p for p in gpu_name.replace("(", " ").replace(")", " ").replace(",", " ").split() if p]
            mask = pd.Series(False, index=gpu_df.index)
            for p in parts:
                mask = mask | gpu_df["gpu"].astype(str).str.contains(p, case=False, regex=False)
        if not mask.any():
            continue
        matched_rows = gpu_df[mask]
        cheapest: Optional[Tuple[str, float, float, float, float]] = None  # (gpu_str, eur_hr_min, eur_hr_med, eur_hr_max, c_med)
        tokens_per_hour = tps * 3600.0
        if tokens_per_hour <= 0:
            continue
        for _, row in matched_rows.iterrows():
            # New schema: per-provider rows with a single usd_hr
            try:
                usd = float(row.get("usd_hr", 0.0))
            except Exception:
                continue
            if usd <= 0:
                continue
            eur_hr = to_eur_hr(usd)
            # Set min/med/max equal to the provider price (no artificial spreads)
            eur_hr_min = eur_hr
            eur_hr_med = eur_hr
            eur_hr_max = eur_hr
            c_med = eur_hr_med / (tokens_per_hour / 1_000_000.0)
            if cheapest is None or c_med < cheapest[4]:
                cheapest = (str(row.get("gpu")), eur_hr_min, eur_hr_med, eur_hr_max, c_med)
        if cheapest is None:
            continue
        row_gpu, eur_hr_min, eur_hr_med, eur_hr_max, c_med = cheapest
        if best is None or c_med < best[5]:
            best = (row_gpu, tps, eur_hr_min, eur_hr_med, eur_hr_max, c_med)

    if best is None:
        return None
    gpu_sel, tps_sel, eur_min, eur_med, eur_max, _ = best
    return gpu_sel, float(tps_sel), float(eur_min), float(eur_med), float(eur_max)


def _model_gpu(model: str, capacity_overrides: Dict[str, Any]) -> Optional[str]:
    ov = capacity_overrides.get("capacity_overrides", {}) if isinstance(capacity_overrides, dict) else {}
    node = ov.get(model)
    if isinstance(node, dict) and node.get("preferred_gpu"):
        return str(node.get("preferred_gpu"))
    return None


def _find_gpu_row(gpu_df: pd.DataFrame, gpu_name: Optional[str]) -> Tuple[float, float, float, str]:
    # Returns (usd_hr_min, usd_hr_med, usd_hr_max, matched_gpu)
    if gpu_name:
        mask = gpu_df["gpu"].astype(str).str.contains(gpu_name, case=False, regex=False)
        if mask.any():
            # Choose the cheapest provider among matches
            sub = gpu_df[mask]
            sub = sub[pd.to_numeric(sub["usd_hr"], errors="coerce") > 0]
            if sub.empty:
                raise KeyError(f"GPU not found in rentals for name: {gpu_name}")
            usd_min_val = float(sub["usd_hr"].min())
            # No spreads; use same value for min/med/max
            return usd_min_val, usd_min_val, usd_min_val, str(sub.iloc[0].get("gpu"))
    # No median fallback here to avoid guessing; signal not found by raising.
    raise KeyError(f"GPU not found in rentals for name: {gpu_name}")


def _sell_per_1m_from_cost(cost_per_1m_med: float, cfg: Dict[str, Any]) -> float:
    """Derive sell price per 1M tokens from cost and pricing policy."""
    pol = _policy_from_cfg(cfg)
    target = float(pol["target_gross_margin_pct"]) / 100.0
    step = float(pol["rounding_per_1k_tokens"])
    floor_1k = float(pol["price_floor_per_1k_tokens"])
    cap_1k = pol["price_cap_per_1k_tokens"]
    # Compute unrounded per-1k price
    if target >= 1.0:
        target = 0.99
    price_1m = 0.0 if cost_per_1m_med <= 0 else (cost_per_1m_med / max(1e-12, (1.0 - target)))
    price_1k = price_1m / 1000.0
    price_1k = _round_price_per_1k(price_1k, step, floor_1k, cap_1k)
    return float(price_1k * 1000.0)


def _optimizer_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pol = _policy_from_cfg(cfg)
    opt = pol.get("optimize") or {}
    # Backwards-safe defaults if optimizer is not configured
    return {
        "enabled": bool(opt.get("enabled", False)),
        "objective": str(opt.get("objective", "profit")),  # currently only 'profit' is supported
        "epsilon": float(opt.get("elasticity_epsilon", 1.0)),
        "ref_per_1k": (None if opt.get("reference_per_1k_tokens") in (None, "") else float(opt.get("reference_per_1k_tokens"))),
        "bounds_per_1k": opt.get("bounds_per_1k", None),  # e.g., [0.05, 5.0]
        "grid_step_per_1k": float(opt.get("grid_step_per_1k", 0.01)),
    }


def _optimizer_settings_for_model(cfg: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Merge global optimizer settings with per-SKU overrides when present."""
    base = _optimizer_settings(cfg)
    pol = _policy_from_cfg(cfg)
    opt = pol.get("optimize") or {}
    per = opt.get("per_sku") or {}
    # Try multiple key variants to find a match
    keys_to_try = [
        model,
        _normalize_model_name(model),
        model.replace(".", "-"),
        model.replace("-", "."),
    ]
    node = None
    for k in keys_to_try:
        if isinstance(per, dict) and k in per and isinstance(per[k], dict):
            node = per[k]
            break
    if not node:
        return base
    merged = dict(base)
    if "enabled" in node:
        try:
            merged["enabled"] = bool(node.get("enabled"))
        except Exception:
            pass
    if "elasticity_epsilon" in node:
        try:
            merged["epsilon"] = float(node.get("elasticity_epsilon"))
        except Exception:
            pass
    if "reference_per_1k_tokens" in node:
        try:
            v = node.get("reference_per_1k_tokens")
            merged["ref_per_1k"] = (None if v in (None, "") else float(v))
        except Exception:
            pass
    if "bounds_per_1k" in node:
        b = node.get("bounds_per_1k")
        if isinstance(b, (list, tuple)) and len(b) == 2:
            merged["bounds_per_1k"] = b
    if "grid_step_per_1k" in node:
        try:
            merged["grid_step_per_1k"] = float(node.get("grid_step_per_1k"))
        except Exception:
            pass
    return merged


def _optimize_price_per_1m(
    *,
    cost_per_1m_med: float,
    cfg: Dict[str, Any],
    model: str,
) -> float:
    """Elasticity-based grid search to maximize (relative) profit per 1k tokens.

    Demand model: D(p) = (pref / p)^epsilon, with epsilon > 0. Absolute demand scale cancels out.
    Objective: maximize D(p) * (p - c), where p = price per 1k, c = cost_per_1m_med/1000.
    Constraints: floor/cap from policy; competitor caps (if enabled) narrow the cap; optional custom bounds.
    """
    pol = _policy_from_cfg(cfg)
    opt = _optimizer_settings_for_model(cfg, model)
    if not opt["enabled"]:
        return _sell_per_1m_from_cost(cost_per_1m_med, cfg)

    floor_1k = float(pol["price_floor_per_1k_tokens"]) or 0.0
    cap_cfg = pol["price_cap_per_1k_tokens"]
    # Competitor cap bound if enabled
    cap_map = pol.get("competitor_caps_by_sku") or {}
    cap_sku = cap_map.get(model) or cap_map.get(_normalize_model_name(model))
    comp_cap_1k = None
    try:
        comp_cap_1k = float(cap_sku) if cap_sku is not None else None
    except Exception:
        comp_cap_1k = None
    cap_1k = None
    if pol.get("apply_competitor_caps") and comp_cap_1k is not None:
        cap_1k = comp_cap_1k
    elif cap_cfg is not None:
        cap_1k = float(cap_cfg)

    # Optional extra bounds
    b = opt.get("bounds_per_1k")
    if isinstance(b, (list, tuple)) and len(b) == 2:
        try:
            lo, hi = float(b[0]), float(b[1])
            floor_1k = max(floor_1k, lo)
            cap_1k = min(cap_1k, hi) if cap_1k is not None else hi
        except Exception:
            pass

    # Reference price per 1k
    c_1k = cost_per_1m_med / 1000.0
    pref = opt.get("ref_per_1k")
    if pref is None or pref <= 0:
        # Use cost-plus target (unrounded) as anchor if not specified
        target = float(pol["target_gross_margin_pct"]) / 100.0
        if target >= 1.0:
            target = 0.99
        p_unrounded_1k = 0.0 if cost_per_1m_med <= 0 else (cost_per_1m_med / max(1e-12, (1.0 - target))) / 1000.0
        pref = max(floor_1k, p_unrounded_1k)

    eps = max(1e-6, float(opt.get("epsilon", 1.0)))
    step = max(1e-6, float(opt.get("grid_step_per_1k", 0.01)))

    lo = max(1e-9, float(floor_1k))
    hi = float(cap_1k) if cap_1k is not None else max(lo + step, pref * 5.0)
    if hi <= lo:
        hi = lo + step

    # Evaluate grid
    best_p = lo
    best_val = -float("inf")
    x = lo
    while x <= hi + 1e-12:
        demand = (pref / x) ** eps
        margin = max(0.0, x - c_1k)
        val = demand * margin
        if val > best_val:
            best_val = val
            best_p = x
        x += step

    # Final rounding to policy step/floor/cap
    p1k = _round_price_per_1k(best_p, float(pol["rounding_per_1k_tokens"]), lo, (cap_1k if cap_1k is not None else None))
    return float(p1k * 1000.0)


def compute_model_economics(
    *,
    cfg: Dict[str, Any],
    price_sheet: pd.DataFrame,
    gpu_df: pd.DataFrame,
    tps_df: pd.DataFrame | None = None,
    scenarios: Dict[str, Any] | None = None,
    overrides: Dict[str, Any] | None = None,
    capacity_overrides: Dict[str, Any] | None = None,
    # Back-compat: older tests passed `extra` with include_models, price_overrides, median_gpu_for_model
    extra: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute provider cost per 1M tokens and policy-derived sell price per 1M for each model.
    Excludes models lacking TPS measurements or GPU rental matches to avoid guessed values.
    Returns a DataFrame with columns:
      model, gpu, tps, eur_hr_min, eur_hr_med, eur_hr_max,
      cost_per_1m_min, cost_per_1m_med, cost_per_1m_max,
      sell_per_1m_eur, margin_per_1m_min, margin_per_1m_med, margin_per_1m_max
    """
    # Back-compat mapping from `extra` if provided
    if extra is not None:
        scenarios = scenarios or {"include_skus": extra.get("include_models", [])}
        overrides = overrides or {"price_overrides": extra.get("price_overrides", {})}
        # capacity_overrides not available historically; ignore median_gpu_for_model here
    scenarios = scenarios or {}
    overrides = overrides or {}
    capacity_overrides = capacity_overrides or {}
    tps_df = tps_df if tps_df is not None else pd.DataFrame()

    _ = _get_fx_and_buffer(cfg)  # ensure cfg has fx defaults
    # Pull conservative costing knobs from config (estimated until telemetry)
    try:
        util_pct = float((cfg.get("pricing_inputs") or {}).get("effective_utilization_pct", 10.0))
    except Exception:
        util_pct = 10.0
    try:
        overhead_mult = float((cfg.get("pricing_inputs") or {}).get("non_gpu_overhead_multiplier_on_cost", 10.0))
    except Exception:
        overhead_mult = 10.0
    models = _select_models(scenarios=scenarios, price_sheet=price_sheet, overrides=overrides)
    rows: List[Dict[str, Any]] = []
    for model in models:
        best = _pick_best_gpu_and_tps(model, tps_df, gpu_df, cfg, capacity_overrides)
        if best is None:
            # No data → exclude model to avoid guesses
            continue
        matched_gpu, tps, eur_hr_min, eur_hr_med, eur_hr_max = best
        tokens_per_hour = float(tps) * 3600.0

        def cost_per_1m(eur_hr: float) -> float:
            if tokens_per_hour <= 0:
                return math.inf
            eff_tokens_per_hour = tokens_per_hour * max(0.0, util_pct) / 100.0
            eff_eur_hr = eur_hr * max(0.0, overhead_mult)
            if eff_tokens_per_hour <= 0:
                return math.inf
            return eff_eur_hr / (eff_tokens_per_hour / 1_000_000.0)

        c_min = cost_per_1m(eur_hr_min)
        c_med = cost_per_1m(eur_hr_med)
        c_max = cost_per_1m(eur_hr_max)

        # Pricing: prefer explicit override per 1k tokens, else compute from optimizer (if enabled) or policy target margin
        sell_1m = 0.0
        ov = overrides.get("price_overrides", {}).get(model) if isinstance(overrides, dict) else None
        if ov and isinstance(ov, dict) and "unit_price_eur_per_1k_tokens" in ov:
            try:
                sell_1m = float(ov["unit_price_eur_per_1k_tokens"]) * 1000.0
            except Exception:
                sell_1m = 0.0
        if sell_1m <= 0.0:
            pol = _policy_from_cfg(cfg)
            opt = pol.get("optimize") or {}
            if bool((_optimizer_settings(cfg)).get("enabled")):
                sell_1m = _optimize_price_per_1m(cost_per_1m_med=c_med, cfg=cfg, model=model)
            else:
                sell_1m = _sell_per_1m_from_cost(c_med, cfg)

        # Apply competitor caps per SKU if enabled (second pass to enforce strict cap even after rounding)
        pol = _policy_from_cfg(cfg)
        cap_map = pol.get("competitor_caps_by_sku") or {}
        cap_applied = False
        if pol.get("apply_competitor_caps") and isinstance(cap_map, dict):
            cap_1k = cap_map.get(model)
            if cap_1k is None:
                # try normalized model name key
                cap_1k = cap_map.get(_normalize_model_name(model))
            try:
                if cap_1k is not None:
                    cap_val_1m = float(cap_1k) * 1000.0
                    if cap_val_1m > 0 and sell_1m > cap_val_1m:
                        sell_1m = cap_val_1m
                        cap_applied = True
            except Exception:
                pass

        margin_min = max(0.0, sell_1m - c_min)
        margin_med = max(0.0, sell_1m - c_med)
        margin_max = max(0.0, sell_1m - c_max)

        rows.append({
            "model": model,
            "gpu": matched_gpu,
            "tps": tps,
            "eur_hr_min": round(eur_hr_min, 4),
            "eur_hr_med": round(eur_hr_med, 4),
            "eur_hr_max": round(eur_hr_max, 4),
            "cost_per_1m_min": round(c_min, 4),
            "cost_per_1m_med": round(c_med, 4),
            "cost_per_1m_max": round(c_max, 4),
            "sell_per_1m_eur": round(sell_1m, 4),
            "margin_per_1m_min": round(margin_min, 4),
            "margin_per_1m_med": round(margin_med, 4),
            "margin_per_1m_max": round(margin_max, 4),
            "competitor_cap_applied": cap_applied,
        })

    return pd.DataFrame(rows)
