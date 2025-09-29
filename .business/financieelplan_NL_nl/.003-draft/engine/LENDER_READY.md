# Lender Readiness Plan (Draft 3 Engine)

This document enumerates what must be built to make the D3 finance engine credible for a €30k lender pack. It captures must‑haves (P0) to reach lender‑grade quality and nice‑to‑haves (P1) to strengthen confidence.

Current status: good scaffold for internal exploration and deterministic reporting, but missing lender‑critical economics (monthly P&L/cashflow/DSCR), realistic capacity↔cost linkage, and robust scenario/MC handling.

---

## Readiness Criteria (Definition of Done)

- **[Determinism]** All outputs reproducible with `SHA256SUMS`; inputs hashed in `run_summary.json`.
- **[Monthly financials]** Accurate P&L and Cashflow across horizon, including taxes, VAT flows, working capital, depreciation, and the loan schedule.
- **[Debt coverage]** DSCR/ICR computed monthly with a minimum DSCR threshold (default ≥ 1.2) and positive cash floor.
- **[Capacity/SLA]** Autoscaling simulated per model across the horizon; violations reported and within policy.
- **[Private margins]** Per‑GPU private margin ≥ target threshold.
- **[Scenario/MC integrity]** Distributions derived across samples (grid/replicate/MC), not pooled rows.
- **[Strict schema]** Pre‑1.0 curated GPU schema enforced: `[gpu, vram_gb, provider, usd_hr]`.
- **[Lender pack]** A compiled `LENDER_PACK.md` with executive summary, base/downside/upside tables, and appendices: inputs, quotes, TPS sources, SHA256SUMS.

---

## P0 (Must‑Have) Checklist

- [ ] Monthly P&L, Cashflow, DSCR
  - [ ] Implement in `d3_engine/core/aggregate.py`:
    - [ ] Write `pnl_by_month.csv` (month, revenue_public, revenue_private, cogs_public, cogs_private, opex_fixed, opex_variable, depreciation, EBITDA, interest, EBT, tax, NetIncome).
    - [ ] Write `cashflow_by_month.csv` (starting_cash, cash_from_ops, working_capital_delta, vat_in_out, capex, debt_service_interest, debt_service_principal, ending_cash).
    - [ ] Write `loan_schedule.csv` from `operator/general.yaml.loan.*` (principal_opening, interest, principal_repayment, principal_closing).
    - [ ] Compute and emit `kpi_summary.json` with DSCR, ICR, min cash, and runway months.

- [x] Demand→Capacity→Cost linkage (Public Tap)
  - [x] Implement per‑model monthly autoscaling in `d3_engine/pipelines/public/artifacts.py` (strict FX/TPS; no fallbacks).
  - [x] Derive monthly tokens per model → hourly (diurnal) series per model.
  - [x] Compute per‑model effective TPS from `facts/tps_model_gpu.csv` and selected GPU/vendor.
  - [x] Simulate autoscaling per model per month; aggregate monthly instance‑hours.
  - [x] Compute GPU cost: instance‑hours × EUR/hr (FX buffered) using chosen provider.
  - [x] Emit `public_tap_capacity_by_month.csv` with instances, instance_hours, peaks, violations, and `gpu_cost_eur_month`.

- [x] Capacity plan across horizon
  - [x] Produce **monthly** `public_tap_capacity_by_month.csv` aligned with autoscaling simulation outputs.

- [ ] Private Tap monthly costs
  - [ ] In `d3_engine/pipelines/private/artifacts.py` compute monthly **COGS_private**:
    - [ ] active_clients × hours_per_client × provider EUR/hr (from `private_vendor_recommendation.csv`).
    - [ ] Provide `private_tap_costs_by_month.csv` with breakdown.
  - [ ] Update `core/aggregate.py` to consume these costs instead of assuming 0.

- [ ] Taxes, VAT, Working Capital
  - [ ] Use `operator/general.yaml`:
    - [ ] `tax.{vat_pct, corporate_income_pct, dividend_withholding_pct}`
    - [ ] `working_capital.{ar_days, ap_days, inventory_days, vat_payment_lag_months}`
  - [ ] Compute VAT input/output and payment lag, AR/AP timing effects on cash; include in monthly cashflow.

- [ ] MC/Scenario integrity
  - [ ] Tag outputs with `(grid_index, replicate_index, mc_index, scenario)` or write per‑job subdirs.
  - [ ] Update `analysis/percentiles.py` and `analysis/sensitivity.py` to compute distributions **across samples**, not pooled rows.

- [ ] Acceptance gates upgrade (`d3_engine/core/acceptance.py`)
  - [ ] Add checks for `min_monthly_DSCR >= threshold`, `min_cash_floor >= 0`.
  - [ ] Ensure capacity violations are checked **across months** and per model.
  - [ ] Require private margin threshold per GPU across horizon.

- [x] Strict curated GPU schema
  - [x] Enforce `[gpu, vram_gb, provider, usd_hr]` in `d3_engine/core/validator.py` and `core/loader.py`; fail on missing/extra columns. Remove lenient legacy column paths pre‑1.0.

- [ ] Lender pack assembly
  - [ ] Add `LENDER_PACK.md` generator (can be static for now) summarizing:
    - [ ] Executive summary, funding ask, sources/uses.
    - [ ] Base/downside/upside KPIs and DSCR tables.
    - [ ] Risk & mitigations; capacity/SLA policy.
    - [ ] Appendices: `inputs/` hashes, curated GPU quotes, TPS sources, SHA256SUMS.

---

## P1 (Nice‑to‑Have) Checklist

- [ ] Channel modeling
  - [ ] Use `facts/ads_channels.csv`, `facts/agency_fees.csv` to model CAC response curves, diminishing returns, and agency overhead.
  - [ ] Add `acquisition.channel_allocation` compliance and sanity checks.

- [ ] TAM & saturation
  - [ ] Cap active customers via TAM curve; apply churn variability and seasonality.

- [ ] Stress testing
  - [ ] Wire `simulation.stress.*` (provider_price_drift_pct, tps_downshift_pct, fx_widen_buffer_pct) through public/private economics and autoscaling.
  - [ ] Generate side‑by‑side scenario pack.

- [ ] Sensitivity/Tornado charts
  - [ ] Programmatic export of top drivers (CAC, churn, TPS, usd_hr, FX) with elasticities.

- [ ] Confidence bounds on TPS & quotes
  - [ ] Use `facts/tps_model_gpu.csv.source_tag` to weight confidence; allow optimistic/base/pessimistic TPS per model/gpu.
  - [ ] Capture quote timestamps and variability in `curated_gpu.csv`.

- [ ] Docs & governance
  - [ ] Expand `.specs/`: inputs contract (strict schema), economics formulas, acceptance gates v2.
  - [ ] Add `docs/LENDER_METHOD.md`: assumptions, data sources, and validation method.

---

## Expected New Artifacts (Checklist)

- **[New CSV/JSON/MD]**
  - [x] `public_tap_capacity_by_month.csv`
  - [ ] `private_tap_costs_by_month.csv`
  - [ ] `pnl_by_month.csv`
  - [ ] `cashflow_by_month.csv`
  - [ ] `loan_schedule.csv`
  - [ ] `kpi_summary.json`
  - [ ] `LENDER_PACK.md`

---

## Implementation Plan (Phased, deterministic)

- **[Phase 1] Demand→Capacity→Costs**
  - Autoscaling per model/month; monthly capacity plan and public GPU costs.
  - Determinism: fix seed; verify `SHA256SUMS` stability.

- **[Phase 2] Private costs + Aggregation**
  - Private monthly COGS; extend aggregator to P&L/Cashflow/Loan.
  - Emit DSCR/ICR and cash runway; add acceptance checks.

- **[Phase 3] MC/Scenario tagging & analysis**
  - Tag outputs; update percentiles/sensitivity to operate across samples.

- **[Phase 4] Schema enforcement & lender pack**
  - Tighten validator/loader; generate `LENDER_PACK.md`.

Each phase ships with unit tests and a smoke integration that asserts the presence and shape of new artifacts and keeps `SHA256SUMS` stable for the base case.

---

## Code Touchpoints (Summary)

- `d3_engine/services/autoscaling_runner.py`: per‑model/month simulation; cost aggregation.
- `d3_engine/pipelines/public/artifacts.py`: capacity plan over horizon; expose TPS/cost hints.
- `d3_engine/pipelines/private/artifacts.py`: monthly private COGS.
- `d3_engine/core/aggregate.py`: P&L, Cashflow, Loan schedule, KPI summary.
- `d3_engine/core/acceptance.py`: DSCR/cash/violations gates.
- `d3_engine/core/validator.py`, `d3_engine/core/loader.py`: strict curated GPU schema.
- `d3_engine/analysis/*`: percentiles/sensitivity across samples.
- `d3_engine/core/writers.py`: (optional) add `(grid,replicate,mc,scenario)` columns when writing tables.

---

## Acceptance Metrics (Default thresholds)

- **[DSCR]** min monthly DSCR ≥ 1.2 (configurable).
- **[Cash floor]** ending cash ≥ 0 for all months; runway ≥ 6 months at start.
- **[Autoscaling]** p95_util within target±tolerance; SLA violations = 0 in base case.
- **[Private margins]** ≥ configured threshold across horizon.

---

## Notes

- The curated GPU schema must follow: `[gpu, vram_gb, provider, usd_hr]`. Pre‑1.0, fail on deviations.
- All lender‑visible figures should be traceable to inputs and logged with input hashes.
- Keep behavior deterministic between runs for the base scenario; maintainability and auditability outweigh stochastic richness for lender deliverables.
