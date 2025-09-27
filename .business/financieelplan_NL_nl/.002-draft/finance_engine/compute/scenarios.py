from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import pandas as pd


def blended_economics(models_df: pd.DataFrame, per_model_mix: Dict[str, float]) -> Tuple[float, float]:
    """Return (blended_sell_per_1m, blended_cost_per_1m_med).
    models_df must include columns: model, sell_per_1m_eur, cost_per_1m_med
    per_model_mix: model -> weight. If empty, equal weights are assumed.
    """
    if models_df.empty:
        return 0.0, 0.0
    if not per_model_mix:
        per_model_mix = {m: 1.0 for m in models_df["model"].tolist()}
    # normalize weights
    total = sum(per_model_mix.values()) or 1.0
    weights = {k: v / total for k, v in per_model_mix.items()}
    # align
    df = models_df.set_index("model")
    sell = 0.0
    cost = 0.0
    for m, w in weights.items():
        if m in df.index:
            sell += float(df.at[m, "sell_per_1m_eur"]) * w
            cost += float(df.at[m, "cost_per_1m_med"]) * w
    return sell, cost


def compute_public_scenarios(
    models_df: pd.DataFrame,
    *,
    per_model_mix: Dict[str, float],
    fixed_total_with_loan: float,
    marketing_pct: float,
    worst_base_best: Tuple[float, float, float],
    marketing_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute monthly public scenarios worst/base/best.
    marketing_pct is e.g. 0.15 for 15%.
    worst_base_best are million-token volumes.
    Returns (df, dict_for_template)
    """
    sell_1m, cost_1m = blended_economics(models_df, per_model_mix)
    scenarios = []
    keys = ["worst", "base", "best"]
    for name, m_tokens in zip(keys, worst_base_best):
        revenue = m_tokens * sell_1m
        cogs = m_tokens * cost_1m
        gross = revenue - cogs
        gross_pct = 0.0 if revenue <= 0 else (gross / revenue)
        if isinstance(marketing_overrides, dict) and name in marketing_overrides:
            try:
                marketing = float(marketing_overrides[name])
            except Exception:
                marketing = revenue * marketing_pct
        else:
            marketing = revenue * marketing_pct
        net = gross - marketing - fixed_total_with_loan
        scenarios.append({
            "case": name,
            "m_tokens": m_tokens,
            "revenue_eur": round(revenue, 2),
            "cogs_eur": round(cogs, 2),
            "gross_margin_eur": round(gross, 2),
            "gross_margin_pct": round(gross_pct * 100.0, 2),
            "marketing_reserved_eur": round(marketing, 2),
            "fixed_plus_loan_eur": round(fixed_total_with_loan, 2),
            "net_eur": round(net, 2),
        })

    df = pd.DataFrame(scenarios)
    tpl = {
        "worst": df[df["case"] == "worst"].iloc[0].to_dict(),
        "base": df[df["case"] == "base"].iloc[0].to_dict(),
        "best": df[df["case"] == "best"].iloc[0].to_dict(),
        "blended": {
            "sell_per_1m_eur": round(sell_1m, 2),
            "cost_per_1m_eur": round(cost_1m, 2),
            "margin_rate": 0.0 if sell_1m <= 0 else round((sell_1m - cost_1m) / sell_1m, 4),
        },
    }
    return df, tpl
