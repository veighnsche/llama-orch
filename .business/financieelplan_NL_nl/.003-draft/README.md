# Financieel Plan — Draft 3 (D3)

Status: Draft · Taal: NL

Deze map bevat de complete D3‑specificaties, inputs en targets om een haalbaarheids‑simulatie te draaien voor 18 maanden, met maand‑op‑maand groei van (verwachte) klanten, autoscaling voor Public Tap, en gezonde marges voor Private Tap. Alle inputs zijn lokaal (YAML/CSV), deterministisch met seed.

## 1. Wat zit hier?

- Overkoepelende spec: `/.003-draft/.specs/00_financial_plan_003.md`
- Inputregels en schemas: `/.003-draft/.specs/10_inputs.md`
- Operator‑constanten (YAML): `inputs/operator/*.yaml` (zie `11_operator_constants.md`)
- Variabelen (CSV): `inputs/variables/*.csv` (zie `12_oprator_variables.md`)
- Facts (exogeen): `inputs/facts/*` (zie `19_facts.md`)
- Simulatieplan/targets: `inputs/simulation.yaml` (zie `15_simulation_constants.md`)
- Simulatie‑logica: `20_simulations.md`, `21_sim_general.md`, `22_sim_private_tap.md`, `23_sim_public_tap.md`, `16_simulation_variables.md`

## 2. Directory overzicht

```
.003-draft/
  README.md
  inputs/
    simulation.yaml
    operator/
      general.yaml
      public_tap.yaml
      private_tap.yaml
      curated_public_tap_models.csv
      curated_gpu.csv
    variables/
      general.csv
      public_tap.csv
      private_tap.csv
    facts/
      ads_channels.csv
      agency_fees.csv
      insurances.csv
      market_env.yaml
  .specs/
    00_financial_plan_003.md
    10_inputs.md
    11_operator_constants.md
    12_oprator_variables.md
    15_simulation_constants.md
    16_simulation_variables.md
    19_facts.md
    20_simulations.md
    21_sim_general.md
    22_sim_private_tap.md
    23_sim_public_tap.md
```

## 3. Invoerbundels (bron van waarheid)

- **Constants (operator)**
  - YAML: `inputs/operator/general.yaml`, `public_tap.yaml`, `private_tap.yaml`
  - Curated CSV’s: `inputs/operator/curated_public_tap_models.csv`, `curated_gpu.csv` (provider‑granulair: `[gpu,vram_gb,provider,usd_hr]`)
- **Variables (operator overlays)**
  - CSV: `inputs/variables/{general,public_tap,private_tap}.csv`
  - Model: kolommen en treatments in `12_oprator_variables.md`
- **Facts (exogeen)**
  - `inputs/facts/*` (geen netwerk). FX: `market_env.yaml → finance.eur_usd_fx_rate.value`
- **Simulation plan**
  - `inputs/simulation.yaml` (run, stochastic, stress, consolidatie, ui/logging, targets)

Regels voor overlay/precedentie, validatie en determinisme staan in `10_inputs.md`, `11_operator_constants.md`, `15_simulation_constants.md` en `16_simulation_variables.md`.

## 4. Doelen & acceptatie (default)

- **Horizon**: `targets.horizon_months = 18`
- **Groei**: `require_monotonic_growth_public_active_customers = true` en idem voor private
- **Private marge**: `targets.private_margin_threshold_pct = 20`
- **Capaciteit (Public)**: instances per model (auto‑scaling) MUST binnen `autoscaling.max_instances_per_model` vallen
- Resultaat/log wordt vastgelegd in `outputs/run_summary.{json,md}`

De lening (60m, 9,95%) is een constante onder `operator/general.yaml → loan.*`, maar géén acceptatie‑criterium binnen de 18m haalbaarheid.

## 5. Public vs Private — wat we simuleren

- **Public Tap (tokens/credits)** — `23_sim_public_tap.md`
  - Kosten per model via provider‐USD/hr → EUR/hr (FX + buffer) + TPS
  - Verkoop €/1k tokens, floors/caps/afronding
  - Vraag/consumptie per maand (budgetgroei, CAC, churn, tokens/convert)
  - Autoscaling: capaciteit per model (doel‑benutting, piekfactor, min/max instances)
  - Artefacten: `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv`

- **Private Tap (GPU‑uren)** — `22_sim_private_tap.md`
  - Median provider EUR/hr + markup + fees → verkoopprijs
  - (Optioneel) vendor aanbeveling per GPU‑klasse
  - (Eenvoudig) maandmodel: budgetgroei, CAC, churn, (optioneel) uren/klant
  - Artefacten: `private_tap_economics.csv`, `private_vendor_recommendation.csv`, `private_tap_customers_by_month.csv`

## 6. Variabelen: treatments & expansie

Zie `16_simulation_variables.md` + `12_oprator_variables.md`.

- **fixed**: gebruik `default` exact
- **low_to_high**: rooster `min..max` met `step` (inclusief eindpunten); de engine vormt de cartesische combinatie over alle `low_to_high` variabelen binnen dezelfde scope
- **random**: uniform in `[min,max]` met step‑kwantisatie (offset `min`). Voor `discrete`: gelijke kans uit `notes.values`

Expansievolgorde: `grid (low_to_high)` → `random replicates` → `Monte Carlo`

- Replicates: `run.random_runs_per_simulation`; enkel relevant voor variabelen met `treatment=random`
- MC iteraties: `stochastic.simulations_per_run`

## 7. Autoscaling (Public)

In `inputs/operator/public_tap.yaml → autoscaling.*` (overridable via `inputs/variables/public_tap.csv`):

- `target_utilization_pct` (percent), `peak_factor` (fraction)
- `min_instances_per_model` (count), `max_instances_per_model` (count)

Capaciteitsformulering en artefact: zie `23_sim_public_tap.md`.

## 8. Quick start

1) **Pas constants aan**
   - `inputs/operator/general.yaml`: vaste kosten, loan, tax, reserves
   - `inputs/operator/private_tap.yaml`: markup/fees/vendor weights, acquisition defaults

2) **Pas variables aan (CSV)**
   - `inputs/variables/public_tap.csv`: budget0, groei, churn, CAC fallback, tokens/conv, autoscaling
   - `inputs/variables/private_tap.csv`: budget0, groei, churn, CAC fallback, (optioneel) uren/klant
   - `inputs/variables/general.csv`: marketing/reserves/tax buffers

3) **Stel targets in**
   - `inputs/simulation.yaml → targets.*`
   - Horizon=18m, private marge ≥20%, (optioneel) minimale MoM groei

4) **Run de engine** (wanneer beschikbaar)
   - Output onder `.003-draft/outputs/`
   - Controleer `run_summary.{json,md}` en de CSV‑artefacten

## 9. Determinisme & seeds

- Seed‑resolutie: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR**
- Identieke inputs + seed → byte‑gelijke outputs

## 10. Belangrijke normen (pre‑1.0)

- Geen backwards compatibility
- GPU offers: provider‑granulair schema `[gpu,vram_gb,provider,usd_hr]` (geen min/max/percent‑velden in bron)
- Alleen **YAML/CSV**; bij dubbelingen wint **CSV** (shadowing WARNING)

## 11. Referenties

- `00_financial_plan_003.md` — D3: doelen, pricing, templates, acceptatie
- `10_inputs.md` — formats, layout, merge/precedence, validator
- `11_operator_constants.md` — operator‑YAML shapes
- `12_oprator_variables.md` — CSV schema, treatments, allowed paths
- `15_simulation_constants.md` — `simulation.yaml` sleutels en constraints
- `16_simulation_variables.md` — treatments, RNG, grid/replicates/MC
- `20_simulations.md` — runner/flow/artefacten
- `21_sim_general.md` — algemene regels (tijd, units, RNG, allocaties)
- `22_sim_private_tap.md` — GPU‑uren simulatie
- `23_sim_public_tap.md` — tokens/credits simulatie + autoscaling
- `19_facts.md` — facts datasets

## 12. Doel van de simulatie (samenvatting)
  
- **[waarom]** Toetsen of het plan haalbaar is over de horizon (`targets.horizon_months`, default 18) met realistische aannames en deterministische herhaalbaarheid.
- **[wat]** Twee pijplijnen:
  - Public Tap (tokens/credits): grondkosten per model uit provider‑USD/hr → EUR/hr (FX + buffer) en TPS; prijs €/1k tokens; vraag en autoscaling.
  - Private Tap (GPU‑uren): mediaan EUR/hr per GPU‑klasse → verkoop €/hr + fees → klanten/uren‑economics.
- **[hoe]** Strikte inputs:
  - Operator‑constanten in `inputs/operator/*.yaml` + curated CSV’s (`curated_public_tap_models.csv`, `curated_gpu.csv`).
  - Variabelen in `inputs/variables/*.csv` met treatments (fixed/low_to_high/random) als overlays op operator‑paden.
  - Facts (FX e.d.) in `inputs/facts/*` (read‑only; geen netwerk).
- **[acceptatie]** Monotone groei (public/private), private marges ≥ drempel, capaciteit binnen min/max, autoscaling p95(util) binnen tolerantie.
- **[eigenschappen]** Volledig lokaal, seed‑deterministisch, CSV>YAML precedence (shadowing WARNING), outputs reproduceerbaar (hash‑gelijk).
  
## 13. Engine‑architectuur & flow (conceptueel)
  
- **[flow]** Load → Validate → Variables grid → Random replicates → Monte Carlo → Pipelines (public/private) → Aggregation → Analysis → Acceptance → Artifacts → Summary.
- **[scheiding]** Pure berekeningen in `pipelines/*` en `services/*`; orchestration in `core/*`; I/O in writers/artifacts; analysis in `analysis/*`.
- **[JSONL progress]** Minimaal `ts,level,event` met events zoals `run_start`, `load_*`, `validate_*`, `grid_built`, `job_*`, `aggregate_done`, `analysis_done`, `acceptance_checked`, `run_done`.
- **[RNG]** Seed‑resolutie: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders ERROR. Substreams per scope/variable/grid/replicate/MC.
- **[autoscaling]** Planner (deterministisch: instances_needed) + Simulator (hysterese, stabilisatiewindow, warmup/cooldown), policy uit `operator/public_tap.yaml`.
  
## 14. Artefacten & rapportage (overzicht)
  
- **Public**
  - `public_vendor_choice.csv` — `model,gpu,provider,usd_hr,eur_hr_effective,cost_eur_per_1M`
  - `public_tap_prices_per_model.csv` — `model,gpu,cost_eur_per_1M,sell_eur_per_1k,margin_pct`
  - `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`
  - `public_tap_capacity_plan.csv` (+ optioneel `public_tap_scaling_events.csv`)
- **Private**
  - `private_tap_economics.csv`, `private_vendor_recommendation.csv`
  - `private_tap_customers_by_month.csv`
- **Consolidatie & rapport**
  - `consolidated_kpis.csv`, `consolidated_summary.{md,json}`
  - `run_summary.{json,md}` (seeds, input‑hashes, grid/replicates/MC, acceptatie, artefacten)
  - Eindrapport: `financial_plan_v3.md` + `charts/*.png`
  
## 15. UI (lokaal, scope)
  
- **[bediening]** Lokale UI (Vite) die parameters kan bewerken (form + editors), validaties toont, `Run Public/Private/Both` triggert en outputs previewt (MD/CSV/PNG/JSON).
- **[prepaid UX]** Zichtbare creditsaldo‑indicator; halt‑at‑zero; presets voor packs; ToS‑snippet (non‑refundable, 12m geldigheid).
- **[veiligheid]** Geen netwerk; facts zijn read‑only; runs blokkeren bij **ERROR**.
