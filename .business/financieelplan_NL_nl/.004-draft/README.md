# Financial Model v4 – Proposal and Rationale

This document proposes a v4 redesign of the financial model to address issues observed in draft v3 under `.003-draft/`. It is spec-first, code-grounded, and aims to produce lender-ready P&L/Cashflow with realistic pricing, demand, costs, and capacity interactions.

## What v3 gets wrong (evidence from the repo)
- **Public Tap price too low**
  - `outputs/public_tap_prices_per_model.csv` shows ~€0.007 per 1k tokens for multiple models on `RTX 3090`.
  - Root: `engine/src/pipelines/public/artifacts.py` uses synthesized TPS from `gpu_baselines.yaml` and a cheap `EUR/hr`, yielding GPU cost per 1M ≈ €0.47 and total cost per 1M ≈ €2.94 with configured overhead. With `target_margin_pct: 55`, sell price lands near €0.006–€0.007/1k.
  - Consequence: `public_tap_scenarios.csv` month-0 public revenue is just €7 (1M tokens).
- **Private Tap revenue is inflated by fee placement**
  - `engine/src/pipelines/private/artifacts.py` adds `management_fee_eur_per_month` per GPU row instead of per client: `revenue_series = hours_series * sell_eur_hr + mgmt_fee`. With multiple GPUs in curated rentals, this adds the management fee multiple times per month, independent of active clients.
- **Fixed opex is heavy and flat from month 0**
  - `inputs/operator/general.yaml` has fixed costs of ~€4,275/month (personal, business, office, insurance_admin, misc). `pnl_by_month.csv` shows EBITDA negative throughout.
- **TPS synthesis ignores measured dataset**
  - `engine/src/core/loader.py` synthesizes `curated.tps_model_gpu` from `facts/gpu_baselines.yaml` but does not read `facts/tps_model_gpu.csv` for model+GPU-specific throughput. Measured data should be used with validation and safety guards.
- **Overhead allocation hides unit economics split**
  - Consolidated KPIs allocate all fixed overhead to private in `aggregate/aggregator.py` (`alloc_prv = overhead_total; alloc_pub = 0`). P&L is correct, but KPI splits can mislead comparisons.

## v4 Goals
- **Realistic price surface** for Public and Private taps, anchored to market and costs, with safe floors and rounding that preserves margin.
- **Correct revenue attribution** (e.g., management fee per client, not per GPU row).
- **Credible demand curves** tied to CAC and budget with churn and channels, with sensitivity sweeps.
- **Transparent costs** including fixed opex ramps, optional hardware CapEx paths, and provider/serverless distinctions.
- **Deterministic capacity and autoscaling** that feeds cost-of-goods accurately.
- **Lender KPIs** (DSCR, ICR, runway) that are robust, with break-even, required price, and required volume helpers.

## Design Principles
- **Spec-first**: document formulas and constraints before code. Keep `.specs/` aligned; add a "Refinement Opportunities" section.
- **Deterministic core; stochastic overlays**: use fixed seeds for reproducibility, and separate Monte Carlo from base cases.
- **Guardrails**: minimum viable prices, utilization bounds, and sanity checks on TPS and provider prices.
- **Separation of concerns**: pricing, demand, capacity, costs, P&L, cashflow each in its place.

## Pricing – Public Tap
- **Inputs**
  - Provider `EUR/hr` per GPU from curated rentals (`inputs/operator/curated_gpu.csv`), with `fx_buffer_pct` from `operator/public_tap.yaml`.
  - Effective TPS per instance from a prioritized chain: `facts/tps_model_gpu.csv` (vetted) > `facts/gpu_baselines.yaml` (fallback), normalized by `services/batching.py`.
  - Policy knobs: `target_margin_pct`, `round_increment_eur_per_1k`, `min_floor_eur_per_1k`, `max_cap_eur_per_1k`, and `non_gpu_costs.{base_eur_per_1M, infra_overhead_pct_of_gpu}`.
- **Formulas**
  - `cap_per_instance_tokens_per_hour = tps * 3600 * target_utilization_pct` (`services/autoscaling.py`).
  - `gpu_cost_eur_per_1M = eur_hr / (cap_per_instance_tokens_per_hour / 1e6)`.
  - `total_cost_eur_per_1M = gpu_cost_eur_per_1M * (1 + infra_overhead_pct/100) + base_eur_per_1M`.
  - `sell_eur_per_1k = clamp_to_floor_cap( round_up( total_cost_eur_per_1M/1000 / (1 - target_margin_frac) ) )`.
    - Use round-up (not nearest) to preserve margin, then apply floor/cap.
- **Guardrails**
  - Set `min_floor_eur_per_1k` to a market-credible floor (e.g., €0.05–€0.20/1k) to prevent unrealistic underpricing when GPU is cheap or TPS is high.
  - Validate TPS: drop or down-weight community measurements; enforce percentile caps; reject outliers by source_tag.
- **Outputs**
  - Per-model price table with explicit components: GPU-only cost, overhead, base, target margin, and final price per 1k.

## Pricing – Private Tap
- **Vendor economics**
  - Medians by GPU for `EUR/hr` (post-FX). Markup policy: `default_markup_over_provider_cost_pct`.
  - Fix revenue: `revenue_eur = hours * sell_eur_hr + mgmt_fee_per_client * active_clients` (not once per GPU row).
- **Client-level units**
  - Hours per client per month (`hours_per_client_month_mean`), add variance; per-client minimum fee can be modeled as `max(mgmt_fee_per_client, hours * sell_eur_hr)`, if desired.
- **Recommendation**
  - Keep best-EUR/hr selection, but add availability/reputation weights when data exists.

## Demand Models
- **Public**
  - Acquisition budget by channel; CAC per channel; conversions produce token demand via a distribution for `tokens_per_conversion`.
  - Maintain churn and budget growth. Add seasonal and noise modules (`behavior/` exists) as optional overlays.
- **Private**
  - Acquisition budget and CAC produce new clients; active via decay `(1 - churn)`; hours = `active_clients * hours_per_client`.
  - Tie `mgmt_fee_per_client` to active clients; allow contract terms (e.g., minimum months).

## Capacity and Autoscaling
- **Planner**
  - Use `planner_instances_needed()` for monthly snapshots, backed by calibrated `effective_tps_per_instance()`.
- **Autoscaler simulation**
  - Keep `services/autoscaling.simulate_autoscaler()` for optional hour-level traces; summarize average and p95 utilization; use to cross-check planner outputs.

## Costs and Opex
- **Fixed opex ramp**
  - Move `personal` and other fixed items to `inputs/variables/general.csv` with a month-indexed ramp (e.g., personal = 0 for months 0–2, 1000 for 3–5, 2000 for 6–9, 2500 from 10+).
- **Overhead allocation**
  - For KPI splits, allocate fixed overhead by revenue share (or by direct cost drivers) rather than assigning 100% to private. Keep consolidated P&L unaffected.
- **CapEx option (optional v4.x)**
  - Add a path to model owned GPUs: CapEx, depreciation, energy (`facts/energy`), and maintenance vs rental `EUR/hr`.

## Taxes, VAT, and Working Capital
- Keep v3 mechanics (`aggregate/series.py`) but expose parameters in variables.
- Validate VAT lag and AR/AP days; add channel-level AR/AP if materially different.

## KPIs and Lender View
- Keep DSCR/ICR/runway with `aggregate/kpis.py`.
- Add break-even helpers: required price for break-even at target volume, and required volume for break-even at target price.
- Report both segment and consolidated KPIs with transparent overhead allocation method.

## Sensitivity & Scenarios
- Monte Carlo on:
  - CAC, `tokens_per_conversion`, churn, target utilization, TPS uncertainty.
- Percentiles: report [10, 50, 90] and select 50th for “representative” summary.
- Provide tornado chart-ready deltas for top drivers.

## Implementation Plan (code changes)
- **Use measured TPS when available**
  - Edit `engine/src/core/loader.py` to read `inputs/facts/tps_model_gpu.csv` and merge with synthesized rows. Prefer vetted sources; fall back to baselines.
- **Public Tap pricing**
  - `engine/src/pipelines/public/artifacts.py`:
    - Switch to round-up in `sell_eur_per_1k` (use `pricing.round_to_increment_up`).
    - Honor a higher `min_floor_eur_per_1k` from `operator/public_tap.yaml`.
  - `engine/src/pipelines/public/pricing.py`:
    - Implement `round_to_increment_up(value, inc)`.
- **Private Tap revenue fix**
  - `engine/src/pipelines/private/artifacts.py`:
    - Change `revenue_series = hours_series * sell_eur_hr + mgmt_fee` to `+ mgmt_fee * active_series`.
- **Fixed opex ramp**
  - Support overrides from `inputs/variables/general.csv` to construct `fixed_opex_series` over time; default to flat if not provided.
- **KPI split**
  - In `aggregate/aggregator.py`, add a configurable allocation method for `consolidated_kpis.csv` (e.g., revenue-share split), keeping P&L unchanged.

## Data & Validation
- Add validation and warnings in `core/validator.py` for:
  - Unrealistically low `sell_eur_per_1k` (< market floor).
  - Community TPS above vetted p90; down-weight or clamp.
  - Negative utilization or SLA violations in capacity-by-month.

## Outputs & Report
- Extend analyzed report to include:
  - Price components breakdown per model.
  - Break-even price/volume tables.
  - Overhead allocation method disclosure.

## Runbook
- After edits, regenerate via `engine/run.sh` (same interface as v3):
  ```bash
  ./.business/financieelplan_NL_nl/.003-draft/engine/run.sh \
    --inputs ./.business/financieelplan_NL_nl/.003-draft/inputs \
    --out ./.business/financieelplan_NL_nl/.003-draft/outputs \
    --pipelines public,private --seed 424242
  ```
  For v4, we’ll mirror the structure into `.004-draft/` once code changes are made.

## Refinement Opportunities
- Add provider availability and reputation models to vendor scoring.
- Model serverless duty-cycle billing for Modal-like providers explicitly vs on-demand hourly.
- Replace closed-form peak factor with diurnal demand synthesis for higher fidelity when needed.
- Introduce LTV/CAC sanity checks for private clients.
- Proof bundle: include P&L/Cashflow CSVs, autoscaling logs, and SSE transcript samples in `.proof_bundle/` for lender review.

---

If you want, I can implement the minimal code changes (public floor + round-up, private fee fix, TPS data merge) and re-run to produce a `.004-draft/outputs/` set for comparison.
