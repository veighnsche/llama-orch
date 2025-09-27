from __future__ import annotations

import pandas as pd

from finance_engine.compute.private_tap import compute_private_tap_economics


def test_private_tap_global_markup_and_per_gpu_mapping():
    gpu_df = pd.DataFrame({
        "gpu": ["H100 80GB (PCIe)", "L4"],
        "hourly_usd_min": [3.0, 0.7],
        "hourly_usd_max": [4.0, 1.0],
    })
    # FX 1.0 for simplicity, buffer 0
    res_global = compute_private_tap_economics(gpu_df, eur_usd_rate=1.0, buffer_pct=0.0, markup_pct=50.0)
    assert set(res_global["gpu"]) == {"H100 80GB (PCIe)", "L4"}
    assert (res_global["sell_eur_hr"] > res_global["provider_eur_hr_med"]).all()

    # Per-GPU overrides should take precedence
    res_map = compute_private_tap_economics(
        gpu_df,
        eur_usd_rate=1.0,
        buffer_pct=0.0,
        markup_pct=50.0,
        markup_by_gpu={"H100": 70, "L4": 40},
    )
    h100 = res_map[res_map["gpu"].str.contains("H100")].iloc[0]
    l4 = res_map[res_map["gpu"] == "L4"].iloc[0]
    assert round(h100["markup_pct"], 2) == 70.0
    assert round(l4["markup_pct"], 2) == 40.0
