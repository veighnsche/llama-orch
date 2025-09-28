# 47 — Consolidatie (KPIs & Samenvatting)

Status: Draft
Version: 0.1.0

## 1. Scope

- Samenvoegen van outputs uit Public en Private pijplijnen over grid/replicates/MC.
- Berekenen van KPI’s en percentielen en schrijven van geconsolideerde artefacten.

## 2. Eisen (MUST)

- **Deterministische merge**: per `(grid_index, replicate_index)` vaste orde.
- **Percentielen**: volgens `stochastic.percentiles` (default `[10,50,90]`).
- **KPI’s**: omzet, kosten, marge, cashflow, blended €/1k, blended €/hr (indien relevant).
- **Overhead‑allocatie**: driver `consolidation.overhead_allocation_driver ∈ {revenue,gpu_hours,tokens}`.
- **Artefacten**: `consolidated_kpis.csv`, `consolidated_summary.{md,json}`.

## 3. Tests

- **Golden**: fixture met minimal outputs → exacte `consolidated_kpis.csv` en `consolidated_summary.json`.
- **Determinism**: twee runs met gelijke seeds → byte‑gelijke geconsolideerde artefacten.
- **Driver‑switch**: driverwisseling verandert alleen relevante KPI’s, niet overige velden.

## 4. Artefact schema (indicatief)

- `consolidated_kpis.csv` kolommen:
  - `metric, p10, p50, p90, unit`
- `consolidated_summary.json` velden:
  - `seeds, input_hashes, horizon_months, blended_prices, revenues, costs, margins, notes`

## 5. Implementatie‑mapping

- `core/aggregate.py` — aggregatie en percentielen.
- `analysis/kpis.py` — KPI’s berekenen.
- `core/consolidate.py` — artefact writers.

## 6. Referenties

- `21_engine_flow.md`, `20_simulations.md`, `40_testing.md`, `31_engine.md`.
