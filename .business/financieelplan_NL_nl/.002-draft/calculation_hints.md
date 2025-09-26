# Calculation Hints

This file explains how to compute each placeholder in `template.md`.

---

## General

- All monetary values in **EUR** unless explicitly USD in source data.
- Convert GPU rental USD prices to EUR using FX rate from config (`fx.rate_used`) and apply buffer (`pricing.fx_buffer_pct`).
- Throughput (tokens/sec per model+GPU) is currently assumed. Replace with **llama-orch telemetry** once available.

---

## Loan Placeholders

- `loan.monthly_payment_eur`: From `lending_plan.yaml` or compute = (loan_amount + flat_interest_total) / term_months.  
- `loan.total_repayment_eur`: loan_amount + total_interest.  
- `loan.total_interest_eur`: loan_amount × annual_rate × (term_months/12).  
- `tables.loan_schedule`: Expand month by month. Opening balance decreases linearly; each month = monthly_payment_eur.

---

## Fixed Costs

- `fixed.personal`: From `costs.yaml`.  
- `fixed.business`: From `costs.yaml`.  
- `fixed.total_with_loan`: personal + business + loan.monthly_payment_eur.

---

## Prepaid Policy

- `prepaid.min_topup_eur`, `prepaid.max_topup_eur`, `prepaid.expiry_months`: From `config.yaml`.  
- `prepaid.non_refundable`: Must be `true`.  
- `prepaid.private_tap.billing_unit_minutes`: Default = 15.  

---

## Pricing Placeholders

- Public Tap:
  - `pricing.model.<Model>_per_1k`: From `price_sheet.csv` (`unit_price_eur_per_1k_tokens`).
  - Sell €/1M tokens = unit_price_per_1k × 1000.
- Private Tap:
  - `pricing.private_tap_default_markup_over_provider_cost_pct`: From `config.yaml`.
  - `private.management_fee_eur_per_month`: From `price_sheet.csv`.

---

## Model Economics (Chapter 2)

- `tables.model_price_per_1m_tokens`: Join `oss_models.csv` + `gpu_rentals.csv` + `price_sheet.csv`.  
- For each model:
  1. Pick compatible GPU(s).  
  2. Compute cost/hr (min, median, max).  
  3. Assume throughput `tps`. Tokens/hour = tps × 3600.  
  4. Cost/1M tokens = provider_cost_per_hr / (tokens/hr ÷ 1,000,000).  
  5. Compare to sell €/1M.  
  6. Gross margin = sell − cost; margin% = (gross / sell) × 100.

---

## Public Tap Scenarios (Chapter 3)

- `scenarios.monthly.{worst,base,best}`:
  - Input: `m_tokens` (sold per month).  
  - Revenue = m_tokens × (weighted_avg_sell_per_1M).  
  - COGS = m_tokens × (weighted_avg_cost_per_1M, median).  
  - Gross = Revenue − COGS.  
  - Marketing = Revenue × marketing_allocation_pct.  
  - Net = Gross − FixedTotal − Marketing.
- Yearly = monthly × 12.  
- 60-month = monthly × 60.

---

## Private Tap Economics (Chapter 4)

- `tables.private_tap_gpu_economics`:  
  - For each GPU: median provider_cost/hr → apply markup → sell/hr → margin/hr.
- `tables.private_tap_example_pack`:  
  - Example: hours_prepaid × sell/hr → revenue.  
  - Provider cost = hours_prepaid × provider_cost/hr.  
  - Gross = revenue − provider cost + management fee.

---

## Worst/Best Projections (Chapter 5)

- Merge Public + Private results.  
- Each case uses different adoption assumptions.  
- Net = Total Gross − FixedTotal − Marketing.

---

## Taxes (Chapter 7)

- `tax.example.revenue_*`: Choose sample gross revenues (e.g. €1k, €10k, €100k).  
- VAT set-aside = revenue × vat_standard_rate_pct.  
- Net revenue = gross − VAT.  

---

## Risk Buffers

- `fx_sensitivity.csv`: Vary EUR/USD ±10%, recompute provider costs.  
- `provider_price_drift.csv`: Vary GPU hourly price ±20%, recompute cost/1M tokens.  

---

## Engine Metadata

- `engine.version`: Manual bump when Python logic changes.  
- `engine.timestamp`: UTC timestamp of run.  
