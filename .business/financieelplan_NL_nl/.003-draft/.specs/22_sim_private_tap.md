# 22 — Private Tap Simulatie (GPU‑uren)

Status: Draft

## 1. Scope

- Modelleert verkoop van GPU‑uren via Private Tap (prepaid uur‑/blokmodellen).
- Bepaalt verkoopprijzen per GPU‑klasse, marge, en (optioneel) LEVERANCIER‑aanbevelingen.
- Gebruikt deterministische RNG voor variabelen (grids/replicates/MC); primair prijs‑/marge‑gedreven, en rapporteert verwacht klantenvolume via een eenvoudig acquisitiemodel (groei + churn).

## 2. Invoer (MUST)

- **Operator‑constanten**: `inputs/operator/private_tap.yaml`
  - `meta.seed`, `meta.fx_buffer_pct` (MUST)
{{ ... }}
- **Curated GPU‑offers (CSV)**: `inputs/operator/curated_gpu.csv` (MUST)
  - Engine normaliseert naar `[gpu, vram_gb, provider, usd_hr]`; `usd_hr>0` (MUST)
- **Variabelen (CSV)**: `inputs/variables/private_tap.csv` (MUST)
  - Voorbeeldpaden (aanbevolen):
    - `pricing_policy.private_tap.default_markup_over_provider_cost_pct`
    - `vendor_weights.cost|availability|reputation` (optioneel)
- **Facts (YAML)**: `inputs/facts/market_env.yaml` (MUST)
  - `finance.eur_usd_fx_rate.value`

Zie `10_inputs.md`, `11_operator_constants.md`, `12_oprator_variables.md`, `19_facts.md`.

## 3. Kernberekeningen (MUST)

1) **Valuta & FX buffer**
   - `eur_hr(provider,g) = usd_hr(provider,g) / eur_usd_fx_rate * (1 + fx_buffer_pct/100)`
   - `eur_usd_fx_rate` uit facts; `fx_buffer_pct` uit `private_tap.yaml: meta.fx_buffer_pct`

2) **Provider‑samenvatting per GPU**
   - `provider_eur_hr_med(g) = median_over_providers eur_hr(provider,g)` (MUST)

3) **Verkoopprijs per GPU‑uur**
   - `markup_pct` uit variabelen (of default)
   - `sell_eur_hr(g) = provider_eur_hr_med(g) * (1 + markup_pct/100)` (MUST)

4) **Fees (optioneel)**
   - `management_fee_eur_per_month` (MAY) wordt apart gerapporteerd
   - `base_fee_eur_per_month` algemeen of `base_fee_by_gpu_class[gpu_class]` (MAY)

5) **Marge per GPU‑uur**
   - `margin_eur_hr(g) = sell_eur_hr(g) − provider_eur_hr_med(g)`
   - Rapportage toont ook `margin_pct = margin_eur_hr / sell_eur_hr`

6) **Blok‑/billing‑unit**
   - `billing_unit_minutes` (MAY) → afrond logica in voorbeelden, niet verplicht voor kernformule

7) **Maandmodel (groei + churn) (MUST)**
   - `private_budget_m = private_budget_month0_eur * (1 + private_budget_growth_pct_mom)^m`
   - `private_CAC_eur_m` uit variabelen (of kanaalgewogen, optioneel)
   - `expected_new_clients_m = private_budget_m / private_CAC_eur_m`
   - `active_clients_m = active_clients_{m-1} * (1 - churn_pct_mom) + expected_new_clients_m` (churn uit variabelen, algemeen)
   - (optioneel) `hours_m = expected_new_clients_m * hours_per_client_month_mean`
   - Omzet: `revenue_eur_m = hours_m * sell_eur_hr(g_class)` indien hours gemodelleerd; anders alleen klantenaantallen rapporteren

## 4. Leverancier‑aanbeveling (MAY)

- Default: provider met minimale `eur_hr(provider,g)` per GPU‑klasse.
- Optioneel: score met gewichten (variabelen of operator):
  - `score(provider,g) = w_cost * (min_eur_hr/eur_hr) + w_avail * avail + w_rep * rep`
  - Gewichten sommeren tot 1.0; ontbrekende `avail/rep` → 1.0 (neutraal)
- Output: `private_vendor_recommendation.csv` met `gpu,provider,usd_hr,eur_hr_effective,score`.

## 5. Variabelen & determinisme (MUST)

- Markup/gewichten leven in `private_tap` scope (CSV); treatments volgens `16_simulation_variables.md`.
- Seed‑resolutie per `15_simulation_constants.md §4`.
- Grid → replicates → MC volgt `21_sim_general.md §4`.

## 6. Artefacten (MUST)

- `private_tap_economics.csv`: `gpu,provider_eur_hr_med,markup_pct,sell_eur_hr,margin_eur_hr,margin_pct`
- `private_vendor_recommendation.csv` (indien geconfigureerd)
- `private_tap_customers_by_month.csv`: `month,private_budget_eur,private_cac_eur,expected_new_clients,active_clients,hours, sell_eur_hr, revenue_eur`
- Charts (MAY): prijzen/marges per GPU‑klasse

## 7. Validatie (MUST)

- `curated_gpu.csv` schema strikt; `usd_hr>0` (MUST); plausibiliteit `usd_hr<50` (SHOULD).
- FX veld verplicht in facts; buffer uit operator aanwezig (MUST).
- Onbekende velden in CSV/YAML → **ERROR**; renormalisatie/afronding → **WARNING** tenzij `fail_on_warning: true`.
- Indien `targets.require_monotonic_growth_private_active_customers: true` (zie `inputs/simulation.yaml`), controleer monotone groei van `active_clients_m` over de horizon.

## 8. Zie ook

- `20_simulations.md` (overzicht) · `21_sim_general.md` (algemeen)
- `11_operator_constants.md` (operator) · `12_oprator_variables.md` (variabelen)
- `19_facts.md` (facts) · `00_financial_plan_003.md §8`
