from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import math

import pandas as pd


DEFAULT_TPS = {
    "Llama-3.1-8B": 35,
    "Llama-3.1-70B": 10,
    "Mixtral-8x7B": 12,
    "Mixtral-8x22B": 8,
    "Qwen2.5-7B": 35,
    "Qwen2.5-32B": 15,
    "Qwen2.5-72B": 10,
    "Yi-1.5-6B": 35,
    "Yi-1.5-9B": 30,
    "Yi-1.5-34B": 15,
    "DeepSeek-Coder-6.7B": 30,
    "DeepSeek-Coder-33B": 14,
    "DeepSeek-Coder-V2-16B": 18,
    "DeepSeek-Coder-V2-236B": 9,
}


def _get_fx_and_buffer(cfg: Dict[str, Any], extra: Dict[str, Any]) -> Tuple[float, float]:
    eur_usd = (
        extra.get("fx", {}).get("eur_usd_rate")
        or cfg.get("fx", {}).get("eur_usd_rate")
        or 1.08
    )
    buffer_pct = cfg.get("pricing_inputs", {}).get("fx_buffer_pct", 0)
    try:
        buffer_val = float(buffer_pct)
    except Exception:
        buffer_val = 0.0
    return float(eur_usd), buffer_val


def _select_models(extra: Dict[str, Any], price_sheet: pd.DataFrame) -> List[str]:
    incl = extra.get("include_models") or []
    if incl:
        return [str(m) for m in incl]
    # Fallback to overrides or price_sheet SKUs
    overrides = extra.get("price_overrides", {})
    if overrides:
        return list(overrides.keys())
    if "sku" in price_sheet.columns:
        return price_sheet["sku"].astype(str).unique().tolist()
    return []


def _model_tps(model: str, extra: Dict[str, Any]) -> float:
    tps_map = extra.get("assumed_tps", {})
    if model in tps_map:
        node = tps_map[model]
        if isinstance(node, dict):
            if "default" in node:
                return float(node["default"]) or DEFAULT_TPS.get(model, 20.0)
        try:
            return float(node)  # if scalar
        except Exception:
            pass
    return DEFAULT_TPS.get(model, 20.0)


def _model_gpu(model: str, extra: Dict[str, Any]) -> Optional[str]:
    return extra.get("median_gpu_for_model", {}).get(model)


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


def _sell_per_1m(model: str, extra: Dict[str, Any], price_sheet: pd.DataFrame) -> Optional[float]:
    ov = extra.get("price_overrides", {}).get(model)
    if ov and isinstance(ov, dict) and "unit_price_eur_per_1k_tokens" in ov:
        return float(ov["unit_price_eur_per_1k_tokens"]) * 1000.0
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


def compute_model_economics(*, cfg: Dict[str, Any], extra: Dict[str, Any], price_sheet: pd.DataFrame, gpu_df: pd.DataFrame) -> pd.DataFrame:
    """Compute provider cost per 1M tokens and sell price per 1M for each model.
    Returns a DataFrame with columns:
      model, gpu, tps, eur_hr_min, eur_hr_med, eur_hr_max,
      cost_per_1m_min, cost_per_1m_med, cost_per_1m_max,
      sell_per_1m_eur, margin_per_1m_min, margin_per_1m_med, margin_per_1m_max
    """
    eur_usd_rate, buffer_pct = _get_fx_and_buffer(cfg, extra)
    models = _select_models(extra, price_sheet)
    rows: List[Dict[str, Any]] = []
    for model in models:
        tps = _model_tps(model, extra)
        tokens_per_hour = tps * 3600.0
        usd_min, usd_med, usd_max, matched_gpu = _find_gpu_row(gpu_df, _model_gpu(model, extra))

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

        sell_1m = _sell_per_1m(model, extra, price_sheet) or 0.0
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
