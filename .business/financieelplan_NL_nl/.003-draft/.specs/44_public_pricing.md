# 44 — Public Pricing & Costs

Status: Draft
Version: 0.1.0

## 1. Scope

- Testtactiek voor Public Tap kosten en prijsafleiding per model.
- Dekt: EUR/hr uit provider USD/hr + FX buffer, TPS→cost €/1M, providerkeuze g*, verkoopprijs €/1k (floors/caps/rounding), marge.

## 2. Eisen (MUST)

- **Curated GPU schema** strikt: `[gpu,vram_gb,provider,usd_hr]` (geen min/max/percent velden).
- **FX & buffers**: `eur_usd_rate` uit facts; `fx_buffer_pct` uit operator; `eur_hr = usd_hr*(1+buffer)/rate`.
- **TPS**: bij voorkeur dataset; zo niet, **heuristiek** met duidelijke WARNING + pad opnemen in `run_summary`.
- **Cost €/1M**: `eur_hr / (tokens_per_hour/1e6)`.
- **Providerkeuze**: g* minimaliseert `cost €/1M` per model; ties op laagste `eur_hr`, dan alfabetisch `gpu`.
- **Verkoopprijs €/1k**: target marge ≥ policy; floors/caps; afronding op `round_increment_eur_per_1k`.

## 3. Tests

- **Unit**: `pipelines/public/gpu_costs.py` (FX/buffer/TPS→cost €/1M), `pipelines/public/pricing.py` (marge, floor/cap/rounding).
- **Golden**: kleine fixture (2 modellen × 2 GPU’s) met vaste TPS; verwachte `public_vendor_choice.csv` en `public_tap_prices_per_model.csv` exact.
- **Heuristiek**: scenario zonder TPS → WARNING gelogd; output gelabeld als benadering.
- **Acceptance**: marge ≥ target in `public_tap_prices_per_model.csv`.

## 4. Artefacten

- `public_vendor_choice.csv` — `model,gpu,provider,usd_hr,eur_hr_effective,cost_eur_per_1M`.
- `public_tap_prices_per_model.csv` — `model,gpu,cost_eur_per_1M,sell_eur_per_1k,margin_pct`.

## 5. Referenties

- `23_sim_public_tap.md`, `21_engine_flow.md`, `40_testing.md`, `31_engine.md`.
