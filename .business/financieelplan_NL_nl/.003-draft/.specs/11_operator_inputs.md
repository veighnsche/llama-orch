# 11 — Operator Inputs (Shapes & Examples)

Status: Draft
Version: 0.1.0

## 1. Scope

- Operator‑gestuurde configuratie is gesplitst in drie bestanden: `operator_general.yaml`, `operator_public.yaml` en `operator_private.yaml`.
- Optionele CSV’s: `catalog_models.csv` / `catalog_gpus.csv` (gebruikt door de Public pijplijn).
- Beïnvloedt: globale financiën/verzekeringen/loan/horizon (General), curated catalogus en public policies (Public), private policies/fees (Private), scenarios, seeds per pijplijn.

## 2. YAML — `operator_general.yaml`

```yaml
# Globale financiën en simulatie-instellingen
finance:
  fixed_costs_monthly_eur:
    personal: 3000
    business: 0
  marketing_allocation_pct_of_inflow: 20.0

insurances:
  selected:                      # kies policies uit public insurances.csv
    - PRO-PL-001
    - OFF-CONT-001

loan:
  amount_eur: 30000
  term_months: 60
  interest_rate_pct_flat: 9.95

simulation:
  run_horizon_months: 24         # simulatieduur (aanbevolen 24)
```

## 3. YAML — `operator_public.yaml`

```yaml
meta:
  seed: 424242            # RNG seed (int, Public pijplijn)

fx:
  fx_buffer_pct: 5.0      # % buffer over public eur_usd_rate

catalog:
  models:                 # curated allow‑list (exact namen)
    - Llama-3-1-8B
    - Qwen2-5-7B
    - Mixtral-8x7B
    - DeepSeek-R1-Distill-Llama-8B
    - Llama-3-3-70B
  gpus:                   # curated GPU‑set
    - A10
    - A100-40GB-PCIe
    - A100-80GB
    - H100-80GB
    - H200-141GB
    - L4
    - L40S
    - RTX-3090
    - RTX-4090

pricing_policy:
  public_tap:
    target_margin_pct: 55.0
    round_increment_eur_per_1k: 0.01
    min_floor_eur_per_1k: 0.05   # optional
    max_cap_eur_per_1k: 3.00     # optional

prepaid_policy:
  credits:
    min_topup_eur: 5
    max_topup_eur: 1000
    expiry_months: 12
    non_refundable: true
    auto_refill_default_enabled: false
    auto_refill_cap_eur: null

acquisition:
  budget_monthly_eur: 500.0
  channel_allocation:            # som = 1.0
    google_ads: 0.5
    linkedin_ads: 0.3
    events: 0.2
  usage_tokens_per_conversion_mean: 5000
  usage_tokens_per_conversion_sd: 1500

scenarios:
  worst: { budget_multiplier: 0.7, cvr_multiplier: 0.6, cac_multiplier: 1.5 }
  base:  { budget_multiplier: 1.0, cvr_multiplier: 1.0, cac_multiplier: 1.0 }
  best:  { budget_multiplier: 1.3, cvr_multiplier: 1.2, cac_multiplier: 0.8 }
```

## 4. YAML — `operator_private.yaml`

```yaml
meta:
  seed: 777777            # RNG seed (int, Private pijplijn)

fx:
  fx_buffer_pct: 5.0      # MAY aanwezig; SHOULD gelijk zijn aan operator_public

pricing_policy:
  private_tap:
    default_markup_over_provider_cost_pct: 50.0

prepaid_policy:
  private_tap:
    billing_unit_minutes: 15
    management_fee_eur_per_month: 99.0
    base_fee_eur_per_month: 0
    base_fee_by_gpu_class:
      A100: 250
      H100: 400
    vendor_weights:      # MAY; gewichten sommeren tot 1.0
      cost: 0.7
      availability: 0.2
      reputation: 0.1

acquisition:
  budget_monthly_eur: 500.0
  channel_allocation:
    google_ads: 0.5
Qwen2-5-7B
Mixtral-8x7B
DeepSeek-R1-Distill-Llama-8B
Llama-3-3-70B
```

- `catalog_gpus.csv`

```csv
gpu
A10
A100-40GB-PCIe
A100-80GB
H100-80GB
H200-141GB
L4
L40S
RTX-3090
RTX-4090
```

## 6. Regels & Validator

- Alleen de sleutels uit de voorbeelden zijn toegestaan, gescheiden per bestand:
  - `operator_general.yaml`: `finance.*`, `insurances.selected`, `loan.*`, `simulation.run_horizon_months`.
  - `operator_public.yaml`: `meta.seed`, `fx.fx_buffer_pct`, `catalog.*`, `pricing_policy.public_tap.*`, `prepaid_policy.credits.*`, `acquisition.*`, `scenarios.*`.
  - `operator_private.yaml`: `meta.seed`, `fx.fx_buffer_pct` (MAY), `pricing_policy.private_tap.*`, `prepaid_policy.private_tap.*` (+ `base_fee*`, `vendor_weights`), `acquisition.*`, `scenarios.*`.
- CSV > YAML als beide dezelfde dataset beschrijven (shadowing WARNING). `catalog_*.csv` geldt voor de Public pijplijn.
- Seeds komen per pijplijn uit het betreffende bestand; runs zijn deterministisch per pijplijn.
- Onbekende modellen/GPUs t.o.v. public `throughput_tps`/`gpu_rentals` → WARNING/ERROR afhankelijk van impact.
- `insurances.selected` (in `operator_general.yaml`) MUST verwijzen naar bestaande `insurance_id` in `insurances.csv` (anders **ERROR**).
- `acquisition.channel_allocation` MUST som 1.0 zijn; kanalen moeten bestaan in `acquisition_benchmarks.csv`.

## 7. Afbakening

- Geen providerprijzen of FX‑spot in operator‑bundle; dat is public data.
- Geen externe bronnen of extra bestandsformaten (alleen YAML/CSV).

## 8. Zie ook

- `12_public data.md` — public CSV‑shapes (`insurances.csv`, `acquisition_benchmarks.csv`).
- `99_calc_or_input.md` — calc/input matrix en acquisitie‑simulatieontwerp.
