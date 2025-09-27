from __future__ import annotations

import pandas as pd

from finance_engine.compute.pricing import compute_model_economics


def test_compute_model_economics_with_overrides_only():
    cfg = {
        "pricing_inputs": {"fx_buffer_pct": 5},
        "fx": {"eur_usd_rate": 1.08},
    }
    overrides = {
        "price_overrides": {
            "Llama-3.1-8B": {"unit_price_eur_per_1k_tokens": 0.15},
            "Mixtral-8x7B": {"unit_price_eur_per_1k_tokens": 0.39},
        }
    }
    scenarios = {"include_skus": ["Llama-3.1-8B", "Mixtral-8x7B"]}

    price_sheet = pd.DataFrame(
        {
            "sku": ["Llama-3.1-8B", "Mixtral-8x7B"],
            "unit_price_eur_per_1k_tokens": [None, None],
            "category": ["public_tap", "public_tap"],
            "unit": ["1k_tokens", "1k_tokens"],
        }
    )
    gpu_df = pd.DataFrame(
        {"gpu": ["L40S"], "hourly_usd_min": [0.6], "hourly_usd_max": [1.2]}
    )

    df = compute_model_economics(
        cfg=cfg,
        price_sheet=price_sheet,
        gpu_df=gpu_df,
        scenarios=scenarios,
        overrides=overrides,
    )
    assert set(df["model"]) == {"Llama-3.1-8B", "Mixtral-8x7B"}
    assert (df["sell_per_1m_eur"] > 0).all()
    assert (df["cost_per_1m_med"] > 0).all()
    assert (df["margin_per_1m_med"] >= 0).all()
