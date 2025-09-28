# 23 — Public Tap Simulatie (Tokens & Credits)

Status: Draft
Version: 0.1.0

## 1. Scope

- Modelleert verkoop van tokens (input + output) via Public Tap, afgerekend in **prepaid, non‑refundable credits** (12 maanden geldig).
- Bepaalt per model: gekozen GPU, kosten €/1M tokens, verkoop €/1k tokens, marge, en vraag/consumptie over scenario’s.
- Integreert ADR’s uit `ondernemersplan/.002-draft` m.b.t. credits en pricing.

## 2. Invoer (MUST)

- **Operator‑constanten**: `inputs/operator/public_tap.yaml`
  - `meta.seed`, `meta.fx_buffer_pct`
  - `pricing_policy.public_tap.*` — `round_increment_eur_per_1k`, optionele `min_floor_eur_per_1k`, `max_cap_eur_per_1k`
  - `prepaid_policy.credits.*` — packs, geldigheid, non_refundable
- **Curated lijsten (CSV)**: `inputs/operator/curated_public_tap_models.csv`, `inputs/operator/curated_gpu.csv`
  - GPU‑offers normaliseren naar `[gpu,vram_gb,provider,usd_hr]`
- **TPS dataset (preferred)**: `(model,gpu) → tps`
  - Indien ontbreekt: **heuristiek** afgeleid uit `Quantization/Runtime` en VRAM; MUST loggen in `run_summary`
- **Variabelen (CSV)**: `inputs/variables/public_tap.csv`, `inputs/variables/general.csv`
  - Voorbeelden: marketingbudget, allocaties per kanaal, usage per conversie, CAC/CVR parameters
- **Facts (YAML)**: `inputs/facts/market_env.yaml`
  - `finance.eur_usd_fx_rate.value`

## 3. Kernberekeningen (MUST)

1) **Grondkosten per GPU**
   - `eur_hr(g) = min_provider_over(g, usd_hr) * (1 + fx_buffer_pct/100) / eur_usd_fx_rate`
   - `eur_usd_fx_rate` uit facts; `fx_buffer_pct` uit operator.

2) **Tokens per uur**
   - `tokens_per_hour(m,g) = tps(m,g) * 3600`
   - Als TPS ontbreekt: heuristische `tokens_per_hour` (MAY) met duidelijke logging.

3) **Kosten €/1M tokens per model**
   - `cost_per_1M_tokens(m,g) = eur_hr(g) / (tokens_per_hour(m,g) / 1_000_000)`
   - Kies `g*` die `cost_per_1M_tokens` minimaliseert (ties: laagste `eur_hr`, dan alfabetisch `gpu`).

4) **Verkoopprijs per model (€/1k tokens)**
   - Bereken `sell_per_1k_tokens(m)` zodat target‑marge ≥ policy‑drempel.
   - Pas `min_floor_eur_per_1k` en `max_cap_eur_per_1k` toe (indien gezet).
   - Rond af op `round_increment_eur_per_1k` (MUST).

5) **Scenario’s & vraag/consumptie**
   - Gebruik variabelen (budget, CAC, CVR, usage per conversie) om **m_tokens** per scenario te simuleren.
   - Creditsaldo neemt af op basis van blended €/1k en volumes; **halt‑at‑zero** enforced (UI en sim).

   Maandmodel (groei + churn) (MUST):
   - `budget_m = budget_month0_eur * (1 + budget_growth_pct_mom)^m` (variabelen)
   - `CAC_eur_m` uit variabelen (of kanaalgewogen op basis van facts + allocaties)
   - `expected_new_customers_m = budget_m / CAC_eur_m`
   - `active_customers_m = active_customers_{m-1} * (1 - churn_pct_mom) + expected_new_customers_m` (churn uit variabelen)
   - `tokens_m = expected_new_customers_m * tokens_per_conversion_mean` (of ander consumptiemodel)
   - `credits_sold_eur_m = (tokens_m / 1000) * blended_sell_eur_per_1k`
   - `cumulative_credits_eur = Σ credits_sold_eur_m`

6) **Stress‑check (SHOULD)**
   - p90 providerprijs drift (`stress.provider_price_drift_pct`) → **WARNING** bij negatieve marge op p90.

## 3A. MRPT v0 — Minimale realistische pijplijn (scopeversimpeling)

- **MUST**
  - Curated modellen + GPU‑offers → normalisatie `[gpu,vram_gb,provider,usd_hr]`.
  - FX + buffer → `eur_hr(g)`.
  - TPS dataset (voorkeur) of heuristiek met duidelijke logging; **batching‑normalisatie** toepassen naar een effectieve tokens/sec per instance op basis van `measurement_type/gpu_count/batch`.
  - Per‑model `cost €/1M` en keuze `g* = argmin_g cost €/1M` (ties: laagste `eur_hr`, dan alfabetisch `gpu`).
  - `sell €/1k` zodanig dat target‑marge ≥ drempel; floor/cap/afronding toepassen.
  - Eenvoudig maandmodel: budget0/groei, CAC fallback, churn, tokens/conv → `active_customers_m`, `tokens_m`, `credits_sold_eur_m` (halt‑at‑zero).
  - Capaciteitsplanner op basis van `avg/peak tokens/hr`, `tps`, `target_util%`, `min/max instances`; `capacity_violation` via clamp/log.
  - Artefacten: `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv`.

- **SHOULD**
  - Kanaalmix/CAC per kanaal o.b.v. allocaties + facts.
  - Autoscaling simulator (hysterese, warmup/cooldown, stabilisatie) + `public_tap_scaling_events.csv`.
  - Acceptatie p95(util) binnen tolerantie rond target.

- **MAY**
  - Diurnale uurprofielen, batching‑efficiency, discount tiers, price‑parity regels.

### 3.1 Capaciteit & Autoscaling (MUST)

- Doel: plan capaciteit per model m zodanig dat piekvraag (p95) kan worden bediend met beoogde benutting.
- Parameters uit `operator/public_tap.yaml → autoscaling.*` (overridable via variabelen):
  - `target_utilization_pct`, `peak_factor`, `min_instances_per_model`, `max_instances_per_model`.
- Grootheden:
  - `avg_tokens_per_hour(m) = tokens_m / 720` (benadering over maand)
  - `peak_tokens_per_hour(m) = avg_tokens_per_hour(m) * peak_factor`
  - Per instance capaciteit bij GPU g: `cap_tokens_per_hour_per_instance(m,g) = tps_eff(m,g) * 3600 * (target_utilization_pct/100)` waarbij `tps_eff` de **batching‑genormaliseerde** tokens/sec per instance is.
  - Nodig: `instances_needed(m,g) = ceil( peak_tokens_per_hour(m) / cap_tokens_per_hour_per_instance(m,g) )`
  - Enforce: `instances_needed = clamp(instances_needed, min_instances_per_model, max_instances_per_model)`; indien clamp > max → `capacity_violation=true`.
- Kosteneffect: effectieve kosten per token worden hoger door benuttingsdoel en piekheadroom; dit is impliciet verdisconteerd via `instances_needed`×`eur_hr(g)` en draaitijd.
- Policies:
  - Indien `capacity_violation=true` voor (m,g), dan SHOULD het model op Public niet worden uitgezet met die GPU; kies alternatieve GPU of markeer als niet‑haalbaar (WARNING/ERROR afhankelijk van beleid).

## 4. Variabelen & determinisme (MUST)

- Public‑specifieke variabelen in `public_tap` scope; algemene (marketing/CVR/CAC) in `general`.
- Treatments per `16_simulation_variables.md`; grid → replicates → MC per `21_sim_general.md`.
- Seed‑resolutie per `15_simulation_constants.md §4`.

## 5. Artefacten (MUST)

- `public_vendor_choice.csv`: `model,gpu,provider,usd_hr,eur_hr_effective,cost_eur_per_1M`
- `public_tap_prices_per_model.csv`: `model,gpu,cost_eur_per_1M,sell_eur_per_1k,margin_pct`
- `public_tap_scenarios.csv`: per scenario (worst/base/best) volumes, omzet, cogs, marge, marketing, net.
- Charts: stacked scenarios, prijs/marge per model.
 - `public_tap_customers_by_month.csv`: `month,budget_eur,cac_eur,expected_new_customers,active_customers,tokens,blended_sell_eur_per_1k,credits_sold_eur,cumulative_credits_eur,cogs_eur,gross_margin_eur`
 - `public_tap_capacity_plan.csv`: `model,gpu,avg_tokens_per_hour,peak_tokens_per_hour,tps,cap_tokens_per_hour_per_instance,instances_needed,target_utilization_pct,capacity_violation`

## 6. Validatie (MUST)

- Curated CSV schema’s exact; onbekende modellen/GPUs → WARNING/ERROR.
- FX veld aanwezig; buffer aanwezig; policyvelden binnen domeinen.
- TPS dataset kolommen exact (indien aangeleverd); ontbrekend → heuristiekpad MUST loggen.

## 7. Credits & ToS (ADR‑aligned)

- Credits zijn **non‑refundable** en **12 maanden geldig** (MUST). Halt‑at‑zero MUST.
- UI moet saldo tonen en consumptie simuleren (zie `00_financial_plan_003.md §11`).

### Doel/acceptatie (MUST)

- Controleer monotone groei over `targets.horizon_months`: `active_customers_m` SHOULD niet dalen (of maximaal binnen ruis) en MUST voldoen aan `targets.public_growth_min_mom_pct` indien > 0.
- Log in `run_summary`: `{ accepted: true|false, horizon_months, public_active_customers: [...], monotonic_ok, min_mom_growth_ok }`.
 - Capaciteit: `instances_needed(m,g*)` MUST ≤ `autoscaling.max_instances_per_model` (anders **ERROR** of **WARNING** conform `fail_on_warning`).

## 8. Zie ook

- `20_simulations.md` (overzicht) · `21_sim_general.md` (algemeen)
- `11_operator_constants.md` (operator) · `12_oprator_variables.md` (variabelen)
- `19_facts.md` (facts) · `00_financial_plan_003.md §7`
