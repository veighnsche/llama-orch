# 46 — Private Economics (GPU‑uren)

Status: Draft
Version: 0.1.0

## 1. Scope

- Testtactiek voor Private Tap (GPU‑uren): median provider EUR/hr, verkoopprijs EUR/hr, fees, klanten/uren per maand en marges.

## 2. Eisen (MUST)

- **Curated GPU schema** strikt: `[gpu,vram_gb,provider,usd_hr]`.
- **EUR/hr median**: per GPU‑klasse de mediaan over providers (MUST).
- **Verkoop**: `sell_eur_hr = provider_eur_hr_med * (1 + default_markup_over_provider_cost_pct/100)`.
- **Fees**: management/base fees MUST in marges en voorbeeldtabellen terechtkomen.
- **Klanten/uren**: per maand via behavior, churn toegepast; `hours ≥ 0`.

## 3. Tests

- **Unit**: `pipelines/private/provider_costs.py` (median), `pipelines/private/pricing.py` (markup/fees), `pipelines/private/clients.py` (uren/cohorten).
- **Golden**: fixture met kleine set providers → exact `private_tap_economics.csv`.
- **Acceptance**: `margin_pct ≥ targets.private_margin_threshold_pct` in `private_tap_economics.csv`.

## 4. Artefacten

- `private_tap_economics.csv` — `gpu,provider_eur_hr_med,markup_pct,sell_eur_hr,margin_eur_hr,margin_pct`.
- `private_tap_customers_by_month.csv` — `month,private_budget_eur,private_cac_eur,expected_new_clients,active_clients,hours,sell_eur_hr,revenue_eur`.

## 5. Referenties

- `22_sim_private_tap.md`, `21_engine_flow.md`, `40_testing.md`, `31_engine.md`, `24_customer_behavior.md`.
