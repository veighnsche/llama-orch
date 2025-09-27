from __future__ import annotations

import pandas as pd

from finance_engine.compute.pricing import compute_model_economics


def test_compute_model_economics_with_overrides_and_gpu_match():
    cfg = {
        "pricing_inputs": {"fx_buffer_pct": 5},
        "fx": {"eur_usd_rate": 1.08},
    }
    extra = {
        "include_models": ["Llama-3.1-8B", "Mixtral-8x7B"],
        "median_gpu_for_model": {"Llama-3.1-8B": "L40S", "Mixtral-8x7B": "L40S"},
        "price_overrides": {
            "Llama-3.1-8B": {"unit_price_eur_per_1k_tokens": 0.15},
            "Mixtral-8x7B": {"unit_price_eur_per_1k_tokens": 0.39},
        },
    }
    price_sheet = pd.DataFrame(
        {
            "sku": ["Llama-3.1-8B", "Mixtral-8x7B"],
            "unit_price_eur_per_1k_tokens": [0.15, 0.39],
        }
    )
    gpu_df = pd.DataFrame({
        "gpu": ["L40S", "L40S"],
        "vram_gb": [48, 48],
        "provider": ["ProviderA", "ProviderB"],
        "usd_hr": [0.9, 1.2],
    })

    # Provide measured TPS so the engine does not rely on any guessed throughput
    tps_df = pd.DataFrame(
        {
            "model_name": ["Llama-3.1-8B", "Mixtral-8x7B"],
            "gpu": ["L40S", "L40S"],
            "throughput_tokens_per_sec": [100.0, 80.0],
        }
    )

    df = compute_model_economics(cfg=cfg, extra=extra, price_sheet=price_sheet, gpu_df=gpu_df, tps_df=tps_df)
    assert set(df["model"]) == {"Llama-3.1-8B", "Mixtral-8x7B"}
    assert (df["sell_per_1m_eur"] > 0).all()
    assert (df["cost_per_1m_med"] > 0).all()
    # Margin computed as non-negative
    assert (df["margin_per_1m_med"] >= 0).all()
