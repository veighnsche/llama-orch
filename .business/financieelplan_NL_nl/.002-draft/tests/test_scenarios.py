from __future__ import annotations

import pandas as pd

from finance_engine.compute.scenarios import blended_economics, compute_public_scenarios


def test_blended_economics_equal_weights():
    df = pd.DataFrame({
        "model": ["A", "B"],
        "sell_per_1m_eur": [100.0, 300.0],
        "cost_per_1m_med": [40.0, 120.0],
    })
    sell, cost = blended_economics(df, per_model_mix={})
    assert sell == (100.0 + 300.0) / 2
    assert cost == (40.0 + 120.0) / 2


def test_compute_public_scenarios_basic():
    df = pd.DataFrame({
        "model": ["A"],
        "sell_per_1m_eur": [200.0],
        "cost_per_1m_med": [100.0],
    })
    per_model_mix = {"A": 1.0}
    fixed_total_with_loan = 3000.0
    marketing_pct = 0.15
    (scen_df, tpl) = compute_public_scenarios(
        df,
        per_model_mix=per_model_mix,
        fixed_total_with_loan=fixed_total_with_loan,
        marketing_pct=marketing_pct,
        worst_base_best=(1.0, 2.0, 3.0),
    )
    assert set(scen_df["case"]) == {"worst", "base", "best"}
    # For 1M tokens: revenue=200, cogs=100, gross=100, marketing=30, net = 100 - 30 - 3000
    worst = scen_df[scen_df["case"] == "worst"].iloc[0]
    assert worst["revenue_eur"] == 200.0
    assert worst["cogs_eur"] == 100.0
    assert worst["gross_margin_eur"] == 100.0
    assert worst["marketing_reserved_eur"] == 30.0
    assert worst["net_eur"] == 100.0 - 30.0 - fixed_total_with_loan
