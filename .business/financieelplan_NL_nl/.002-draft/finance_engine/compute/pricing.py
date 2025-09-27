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
    incl = scenarios.get("include_skus") or []
    if incl:
        return [str(m) for m in incl]
    ov = overrides.get("price_overrides", {}) if isinstance(overrides, dict) else {}
    if ov:
        return list(ov.keys())
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


def _model_tps(model: str, tps_df: pd.DataFrame, capacity_overrides: Dict[str, Any]) -> float:
    # capacity overrides precedence if provided
    ov = capacity_overrides.get("capacity_overrides", {}) if isinstance(capacity_overrides, dict) else {}
    node = ov.get(model)
    if isinstance(node, dict) and node.get("tps_override_tokens_per_sec") is not None:
        try:
            return float(node.get("tps_override_tokens_per_sec"))
        except Exception:
            pass
    if tps_df is None or tps_df.empty:
        return 20.0
    norm = _normalize_model_name
    # Try to match by normalized model names
    mask = tps_df["model_name"].astype(str).map(norm) == norm(model)
    sub = tps_df[mask]
    if sub.empty:
        return 20.0
    try:
        vals = pd.to_numeric(sub["throughput_tokens_per_sec"], errors="coerce").dropna()
        if vals.empty:
            return 20.0
        return float(vals.median())
    except Exception:
        return 20.0


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
            row = gpu_df[mask].iloc[0]
            usd_min = float(row.get("hourly_usd_min", 0))
            usd_max = float(row.get("hourly_usd_max", usd_min))
            usd_med = (usd_min + usd_max) / 2 if usd_max else usd_min
            return usd_min, usd_med, usd_max, str(row.get("gpu"))
    # Fallback: median across the dataset
    usd_min = float(gpu_df.get("hourly_usd_min").median()) if "hourly_usd_min" in gpu_df.columns else 0.0
    usd_max = float(gpu_df.get("hourly_usd_max").median()) if "hourly_usd_max" in gpu_df.columns else usd_min
    usd_med = (usd_min + usd_max) / 2 if usd_max else usd_min
    any_gpu = str(gpu_df.iloc[0].get("gpu")) if not gpu_df.empty else "Unknown"
    return usd_min, usd_med, usd_max, any_gpu


def _sell_per_1m(model: str, *, price_sheet: pd.DataFrame, overrides: Dict[str, Any]) -> Optional[float]:
    ov = overrides.get("price_overrides", {}).get(model) if isinstance(overrides, dict) else None
    if ov and isinstance(ov, dict) and "unit_price_eur_per_1k_tokens" in ov:
        try:
            return float(ov["unit_price_eur_per_1k_tokens"]) * 1000.0
        except Exception:
            pass
    # Fallback to price_sheet if has a matching sku
    if "sku" in price_sheet.columns and "unit_price_eur_per_1k_tokens" in price_sheet.columns:
        m = price_sheet[price_sheet["sku"].astype(str) == model]
        if not m.empty:
            raw = m.iloc[0]["unit_price_eur_per_1k_tokens"]
            # Treat NaN or empty values as missing
            if pd.isna(raw) or raw == "":
                return None
            val = float(raw) * 1000.0
            return val
    return None


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
    """Compute provider cost per 1M tokens and sell price per 1M for each model.
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

    eur_usd_rate, buffer_pct = _get_fx_and_buffer(cfg)
    models = _select_models(scenarios=scenarios, price_sheet=price_sheet, overrides=overrides)
    rows: List[Dict[str, Any]] = []
    for model in models:
        tps = _model_tps(model, tps_df, capacity_overrides)
        tokens_per_hour = tps * 3600.0
        usd_min, usd_med, usd_max, matched_gpu = _find_gpu_row(gpu_df, _model_gpu(model, capacity_overrides))

        # USD/hr → EUR/hr with buffer
        def to_eur_hr(usd_hr: float) -> float:
            eur = (usd_hr / eur_usd_rate) * (1.0 + buffer_pct / 100.0)
            return eur

        eur_hr_min = to_eur_hr(usd_min)
        eur_hr_med = to_eur_hr(usd_med)
        eur_hr_max = to_eur_hr(usd_max)

        def cost_per_1m(eur_hr: float) -> float:
            if tokens_per_hour <= 0:
                return math.inf
            return eur_hr / (tokens_per_hour / 1_000_000.0)

        c_min = cost_per_1m(eur_hr_min)
        c_med = cost_per_1m(eur_hr_med)
        c_max = cost_per_1m(eur_hr_max)

        sell_1m = _sell_per_1m(model, price_sheet=price_sheet, overrides=overrides) or 0.0
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
        })

    return pd.DataFrame(rows)
