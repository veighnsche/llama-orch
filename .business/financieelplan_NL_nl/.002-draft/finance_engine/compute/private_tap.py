from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd


def compute_private_tap_economics(
    gpu_df: pd.DataFrame,
    *,
    eur_usd_rate: float,
    buffer_pct: float,
    markup_pct: Optional[float] = None,
    markup_by_gpu: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute per-GPU economics for Private Tap.
    Returns DataFrame with columns: gpu, provider_eur_hr_med, markup_pct, sell_eur_hr, margin_eur_hr
    """
    if gpu_df.empty:
        return pd.DataFrame(columns=["gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr"])

    df = gpu_df.copy()
    # median USD/hr estimate
    if "hourly_usd_min" in df.columns and "hourly_usd_max" in df.columns:
        df["usd_hr_med"] = (df["hourly_usd_min"].astype(float) + df["hourly_usd_max"].astype(float)) / 2.0
    elif "hourly_usd" in df.columns:
        df["usd_hr_med"] = df["hourly_usd"].astype(float)
    else:
        df["usd_hr_med"] = 0.0

    df["provider_eur_hr_med"] = (df["usd_hr_med"] / float(eur_usd_rate)) * (1.0 + float(buffer_pct) / 100.0)

    # Determine markup per row: prefer per-GPU mapping by substring match, else global markup, else 50%
    default_markup = 50.0 if markup_pct is None else float(markup_pct)
    mapping = markup_by_gpu or {}

    def pick_markup(gpu_name: str) -> float:
        name_l = str(gpu_name).lower()
        for key, val in mapping.items():
            if str(key).lower() in name_l:
                try:
                    return float(val)
                except Exception:
                    continue
        return default_markup

    df["markup_pct"] = df["gpu"].apply(pick_markup)
    df["sell_eur_hr"] = df["provider_eur_hr_med"] * (1.0 + df["markup_pct"] / 100.0)
    df["margin_eur_hr"] = df["sell_eur_hr"] - df["provider_eur_hr_med"]

    cols = ["gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr"]
    out = df[cols].copy()
    # group by GPU name (if duplicates) and take median values
    out = out.groupby("gpu", as_index=False).median(numeric_only=True)
    return out
