# 24 — Customer Behavior (Funnels, Budgets, Retention)

Status: Draft
Version: 0.1.0

## 1. Scope

- Definieert het klantgedrag dat in de simulaties wordt gebruikt (Public & Private).
- Omvat maandcohorten, budgetten, funnels (CAC/CVR), tokens per conversie, churn/retentie en omzetberekening.
- Deterministisch met seeds (zie `16_simulation_variables.md`). Geen netwerk tijdens runs.

## 2. Tijdsbasis & Units (MUST)

- Tijdsbasis: maanden `m ∈ [1..targets.horizon_months]` (default 18).
- Units:
  - Geld: EUR
  - Tokens: gecombineerde tokens (input + output)
  - Uren: GPU‑uren (Private)
  - Percentages: `fraction` (0..1) of `percent` (0..100), consistent met inputs.

## 3. Variabelen en inputs (MUST)

- Public (indicatief; exacte paden in `12_oprator_variables.md`):
  - `public_tap.budget_month0_eur`, `public_tap.budget_growth_pct_mom`
  - `public_tap.cac_base_eur`, `public_tap.cac_sd_eur` (optioneel; random)
  - `public_tap.cvr_base`, `public_tap.cvr_sd` (optioneel; random)
  - `public_tap.tokens_per_conversion_mean`, `public_tap.tokens_per_conversion_sd`
  - `public_tap.churn_rate_mom` (fraction per maand)
- Private:
  - `private_tap.budget_month0_eur`, `private_tap.budget_growth_pct_mom`
  - `private_tap.cac_base_eur`, `private_tap.churn_rate_mom`
  - `private_tap.hours_per_client_mean`, `private_tap.hours_per_client_sd`
  - Verkoopprijs `sell_eur_hr` komt uit de Private pipeline (pricing), niet uit behavior.

Opmerking: random variabelen worden per replicate hergetrokken (zie `16_simulation_variables.md`).

## 4. Budgetserie (MUST)

- Groeimodel per maand:
  - `budget_m = budget_month0_eur * (1 + budget_growth_pct_mom/100)^(m-1)`
  - `budget_m ≥ 0` (negatief → 0)
- Public en Private gebruiken dezelfde basis voor budgetgroei, met eigen variabelen.

## 5. Funnels & conversies (MUST)

- Public:
  - Nieuwe klanten per maand: `expected_new_customers_m = floor(budget_m / CAC_m)`
  - `CAC_m` kan per replicate random zijn (bv. normaal of lognormaal, ≥ 0.01 EUR), kwantisatie volgens `step`-rooster.
  - Tokens per maand: `tokens_m = expected_new_customers_m * tokens_per_conversion_m`
  - `tokens_per_conversion_m` kan random (≥0) of fixed zijn.
- Private:
  - Nieuwe klanten per maand: idem `floor(budget_m / CAC_m)`
  - Maanduren: `hours_m = active_clients_m * hours_per_client_m` (met max/min guards)

## 6. Retentie & churn (MUST)

- Churn per maand `c` als fractie (0..1):
  - `active_clients_m = (active_clients_{m-1} * (1 - c)) + expected_new_customers_m`
  - `c` kan uit een random of fixed variabele komen (per replicate constant, tenzij anders gespecificeerd).
- Monotoniciteit: de engine controleert per targets of `active_clients_m` niet daalt (acceptatieregel).

## 7. Determinisme & RNG (MUST)

- Seeds via `16_simulation_variables.md §4`: substreams per scope en variabele, geparametriseerd op `(grid_index, replicate_index)`.
- Random variabelen zijn per replicate constant tenzij expliciet per maand gemodelleerd.
- `variable_draws.csv` SHOULD worden vastgelegd: `scope,variable_id,path,grid_index,replicate_index,draw_value`.

## 8. Interfaces (MUST)

- Public:
  - `behavior.public.simulate_months(plan, variables, rng) -> List[Row]`
  - Output kolommen (minimaal):
    - `month,budget_eur,cac_eur,expected_new_customers,active_customers,tokens`
- Private:
  - `behavior.private.simulate_months(plan, variables, rng) -> List[Row]`
  - Output kolommen (minimaal):
    - `month,private_budget_eur,private_cac_eur,expected_new_clients,active_clients,hours,sell_eur_hr,revenue_eur`

De pipelines roepen deze functies aan om de maandtabellen te produceren; de writers schrijven de bijbehorende CSV’s.

## 9. Acceptatie & policy (MUST)

- Public/Private actieve klantenreeks is **monotoon niet‑dalend** (of policy‑gebaseerde afwijkingen met motivatie).
- Indien `targets.public_growth_min_mom_pct` is gezet: maand‑op‑maand groei ≥ drempel.
- Private marges worden elders getoetst; behavior levert de benodigde `hours` en `clients`.

## 10. Testen (MUST)

- BDD‑scenario’s (`acceptance.feature`) toetsen monotoniciteit en minimaal headers.
- Unit tests voor budgetserie, churnaccumulatie en conversies (edge‑cases: 0 budget, CAC→∞, churn→1).
- Determinisme: identieke seeds → identieke klantreeksen en tokens/hours.

## 11. Implementatie in codebase

- Map: `engine/src/d3_engine/behavior/`
  - `budgets.py` — reeks `budget_m` genereren
  - `funnels.py` — conversies/tokens afleiden uit budget + CAC/CVR
  - `retention.py` — churn/retentie en actieve klanten
  - `cohorts.py` — helpers voor cohortaggregatie (optioneel)
  - `sampling.py` — helpers voor RNG/kwantisatie
  - `public.py` — orkestratie Public behavior (roept bovenliggende helpers)
  - `private.py` — orkestratie Private behavior

## 12. Referenties

- `16_simulation_variables.md` — treatments, RNG & determinisme
- `23_sim_public_tap.md` — public pipelines en artefacten
- `22_sim_private_tap.md` — private pipelines en artefacten
- `40_testing.md` — acceptance‑regels en artefact‑schema’s
- `20_simulations.md` — runner/flow, parallel jobs
