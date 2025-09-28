# 11 — Operator Inputs (Shapes & Examples)

Status: Draft
Version: 0.1.0

## 1. Scope

- Operator‑constanten wonen in drie YAML’s onder `inputs/operator/`: `general.yaml`, `public_tap.yaml`, `private_tap.yaml`.
- Curatie/whitelists en providerprijzen staan in CSV’s onder `inputs/operator/`: `curated_public_tap_models.csv`, `curated_gpu.csv`.
- Variabele ‘knoppen’ (budgets, marges, vendor‑weights, etc.) staan NIET in deze YAML’s maar in `inputs/variables/*.csv` (zie `12_oprator_variables.md`).

## 2. YAML — `inputs/operator/general.yaml`

```yaml
# Operator world facts (constanten)
meta:
  version: 1
  description: "Operator world facts (no knobs/variables). Policy levers moved to variables CSVs."

finance:
  fixed_costs_monthly_eur:
    personal: 2500
    business: 1000
    office: 400
    insurance_admin: 75
    misc_buffer: 300
  depreciation_assets_eur: 150

loan:
  amount_eur: 30000
  term_months: 60
  interest_rate_pct_flat: 9.95
  repayment_type: annuity        # annuity | flat | bullet
  equity_injection_eur: 5000
  grace_period_months: 3

insurances:
  selected: [centraal_beheer_avb, interpolis_avb]   # verwijst naar facts/insurances.csv: insurer_id

tax:
  vat_pct: 21.0
  corporate_income_pct: 19.0
  dividend_withholding_pct: 15.0
  social_security_pct: 27.65

calendar:
  start_year: 2025
  start_month: 10

working_capital:
  ar_days: 30
  ap_days: 30
  inventory_days: 0
  vat_payment_lag_months: 1

inflation:
  cpi_annual_pct: 2.5
  salary_growth_annual_pct: 0.0
  rent_growth_annual_pct: 2.0

depreciation_schedule:
  assets:
    - name: "Laptop"
      amount_eur: 2400
      months: 36
      start_month_index: 1
```

## 3. YAML — `inputs/operator/public_tap.yaml`

```yaml
meta:
  seed: 424242
  fx_buffer_pct: 5.0

pricing_policy:
  public_tap:
    target_margin_pct: 55.0
    round_increment_eur_per_1k: 0.01
    min_floor_eur_per_1k: 0.05
    max_cap_eur_per_1k: 3.00

prepaid_policy:
  credits:
    min_topup_eur: 5
    max_topup_eur: 1000
    expiry_months: 12
    non_refundable: true
    auto_refill_default_enabled: false
    auto_refill_cap_eur: null
```

Opmerking: Curated modellen/GPUs staan in CSV (zie §5). Acquisitie‑/margetargets komen als variabelen (CSV), niet als constante YAML.

## 4. YAML — `inputs/operator/private_tap.yaml`

```yaml
meta:
  seed: 12345
  fx_buffer_pct: 5.0   # conservative FX cushion

prepaid_policy:
  private_tap:
    billing_unit_minutes: 60
    non_refundable: true
    expiry_months: 12
```

## 5. Curated CSV’s — Shapes

- `inputs/operator/curated_public_tap_models.csv`

  Minimale kolommen (extra toegestaan):

  ```csv
  Model,Variant,Developer,Quantization/Runtime,Typical_VRAM_for_4bit_or_MXFP4_GB,License,Download,Benchmarks,Notes
  ```

- `inputs/operator/curated_gpu.csv`

  Provider‑granulaire rentals/offers. Minimale kolommen (extra toegestaan):

  ```csv
  provider,gpu_model,gpu_vram_gb,num_gpus,price_usd_hr,price_per_gpu_hr
  ```

  Normatief: de engine normaliseert naar `[gpu, vram_gb, provider, usd_hr]` met `usd_hr = price_per_gpu_hr` (indien gezet) anders `price_usd_hr/num_gpus`. Geen min/max/percentvelden in de bron.

## 6. Regels & Validator

- Allowed keys per bestand:
  - `inputs/operator/general.yaml`: `finance.*`, `insurances.selected`, `loan.*`, `tax.*`, `calendar.*`, `working_capital.*`, `inflation.*`, `depreciation_schedule.*`.
  - `inputs/operator/public_tap.yaml`: `meta.seed`, `meta.fx_buffer_pct`, `pricing_policy.public_tap.*`, `prepaid_policy.credits.*`.
  - `inputs/operator/private_tap.yaml`: `meta.seed`, `meta.fx_buffer_pct`, `prepaid_policy.private_tap.*`.
- Variabele paden (marges/budgets/vendor_weights/markups) horen in `inputs/variables/*.csv` (zie `12_oprator_variables.md`).
- Curated CSV’s MUST schema’s uit §5 volgen; engine normaliseert GPU‑offers naar provider‑granulaire `[gpu, vram_gb, provider, usd_hr]` (pre‑1.0 normatief, geen min/max/percentvelden).
- Seeds per pijplijn komen uit `meta.seed`; determinisme MUST gelden per pijplijn.
- `insurances.selected` MUST verwijzen naar `insurer_id` waarden in `inputs/facts/insurances.csv`; onbekend → **ERROR**.
- Conflicten/overschrijvingen door variabelen worden gelogd als `variable_override` met bron CSV en pad.

## 7. Afbakening

- Operator YAML bevat geen facts/marktdata; die staan onder `inputs/facts/`.
- Geen externe bronnen of extra bestandsformaten (alleen YAML/CSV).

## 8. Zie ook

- `19_facts.md` — facts‑datasets (ads_channels, agency_fees, insurances, market_env).
- `12_oprator_variables.md` — variabelen‑CSV’s en overlayregels.
