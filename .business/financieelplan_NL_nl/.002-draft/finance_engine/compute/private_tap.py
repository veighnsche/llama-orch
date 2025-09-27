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
    New schema expects per-provider rows with 'usd_hr'.
    Returns DataFrame with columns: gpu, provider_eur_hr_med, markup_pct, sell_eur_hr, margin_eur_hr
    """
    if gpu_df.empty:
        return pd.DataFrame(columns=["gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr"])

    df = gpu_df.copy()
    # Require usd_hr per provider row, compute EUR/hr with FX and buffer
    df["provider_eur_hr"] = (pd.to_numeric(df["usd_hr"], errors="coerce").astype(float) / float(eur_usd_rate)) * (1.0 + float(buffer_pct) / 100.0)
    # Aggregate to median per GPU
    per_gpu = df.groupby("gpu", as_index=False).median(numeric_only=True)
    per_gpu = per_gpu[["gpu", "provider_eur_hr"]].rename(columns={"provider_eur_hr": "provider_eur_hr_med"})

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

    per_gpu["markup_pct"] = per_gpu["gpu"].apply(pick_markup)
    per_gpu["sell_eur_hr"] = per_gpu["provider_eur_hr_med"] * (1.0 + per_gpu["markup_pct"] / 100.0)
    per_gpu["margin_eur_hr"] = per_gpu["sell_eur_hr"] - per_gpu["provider_eur_hr_med"]

    cols = ["gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr"]
    out = per_gpu[cols].copy()
    return out
