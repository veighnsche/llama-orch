# 20 — Simulaties (Overzicht, Runner & Flow)

Status: Draft
Version: 0.1.0

## 1. Doel & Scope

- Beschrijft WAT we simuleren (pijplijnen) en HOE de engine deze uitvoert.
- De details per pijplijn staan in `22_sim_private_tap.md` (GPU‑uren) en `23_sim_public_tap.md` (tokens/credits).
- Cross‑cutting regels over determinisme, RNG, grids en replicates: zie `21_sim_general.md` en `16_simulation_variables.md`.
- Runplan en configuratie: `inputs/simulation.yaml`, gespecificeerd in `15_simulation_constants.md`.

## 2. Pijplijnen

- **PublicTapSim** — verkoopt tokens via Public Tap (prepaid credits). Prijs per model wordt afgeleid uit provider‑kosten + TPS + beleidsregels.
- **PrivateTapSim** — verkoopt GPU‑uren (prepaid blokken); prijs per klasse uit mediane provider EUR/hr + markup + fees.
- **Consolidatie** — combineert KPI’s (omzet, marge, cashflow) en schrijft een samenvatting.

Pijplijnen draaien onafhankelijk (deterministisch) en kunnen afzonderlijk of samen worden uitgevoerd (`run.pipelines`).

## 3. Engine Flow (fasen)

1) **Load inputs**
   - Laad constants: `inputs/operator/*.yaml` + curated CSV’s onder `inputs/operator/`.
   - Laad variables overlays: `inputs/variables/*.csv` (alleen toegestane paden; zie `12_oprator_variables.md`).
   - Laad facts: `inputs/facts/*` (read‑only; o.a. `market_env.yaml → finance.eur_usd_fx_rate.value`).
   - Laad runplan: `inputs/simulation.yaml`.

2) **Validate**
   - Schema’s, referenties, domeinen; zie `10_inputs.md §8`, `11_operator_constants.md §6`, `19_facts.md §6`.
   - Seed‑resolutie; zie `15_simulation_constants.md §4`.

3) **Build variable grid** (MUST)
   - Per scope (`general`, `public_tap`, `private_tap`): verzamel `fixed` en construeer grids voor `low_to_high` (cartesisch product). Zie `16_simulation_variables.md`.

4) **Replicates & Monte Carlo** (MUST)
   - Voor elke grid‑combinatie: voer `run.random_runs_per_simulation` replicates uit; trek `random` variabelen opnieuw per replicate.
   - Binnen elke replicate: voer `stochastic.simulations_per_run` iteraties uit voor funnel/uitkomststochastiek.

5) **Run pipelines (parallel jobs)** (MUST)
   - Maak jobs voor elke `(grid_index, replicate_index)` (en MC binnen replicate) met deterministische volgorde.
   - Plan jobs met `run.max_concurrency` (threads/processen) zonder resultaat‑drift.
   - `PublicTapSim`: grondkosten → keuze GPU per model → verkoopprijs per 1k → vraag/credits → KPIs.
   - `PrivateTapSim`: mediane provider‑EUR/hr → verkoop EUR/hr → fees → KPIs.

6) **Consolidatie** (MUST)
   - Combineer outputs (groepen, percentielen) volgens `consolidation.overhead_allocation_driver`.
   - Schrijf `consolidated_summary.{md,json}` en `consolidated_kpis.csv`.

7) **Artefacten & logging** (MUST)
   - `run_summary.{json,md}` met seed(s), input‑hashes, overlay‑beslissingen, shadowing‑warnings, grid/replicates/MC.
   - `variable_draws.csv` met `scope,variable_id,path,grid_index,replicate_index,draw_value`.
   - Pijplijnspecifieke CSV’s/MD/PNG (zie §6 en specs 22/23).

### 3.1 Parallelisme & determinisme (MUST)

- Jobs worden eerst volledig gegenereerd en lexicografisch gesorteerd op `(grid_index, replicate_index)`.
- Elke job krijgt een eigen seed via hashing (`H(master_seed, scope, variable_id?, grid_index, replicate_index, mc_index?)`).
- Resultaten worden deterministisch verzameld en geaggregeerd in dezelfde volgorde; schrijfbewerkingen gebeuren óf uit één aggregator, óf via staging + merge.
- JSONL progress events (indicatief):
  - `run_start` → `load_start` → `load_done` → `validate_start` → `validate_done` → `grid_built` (size=N)
  - `job_submitted`/`job_done` (met `grid_index`, `replicate_index`)
  - `pipeline_public_start`/`pipeline_public_done`, `pipeline_private_start`/`pipeline_private_done`
  - `aggregate_done` → `acceptance_checked` → `run_done`

## 4. Runner & Configuratie (`inputs/simulation.yaml`)

- `run.pipelines: [public|private]` selecteert pijplijnen.
- `stochastic.simulations_per_run` bepaalt MC‑iteraties per replicate; `run.random_runs_per_simulation` bepaalt #replicates per grid‑combinatie.
- `stress.*` definieert optionele stress‑checks (bv. provider prijsdrift voor Public Tap).
- `logging.*` bepaalt logniveau en of `run_summary` wordt geschreven.
- Volledige sleutels en constraints: `15_simulation_constants.md`.

## 5. Determinisme & Seeds (MUST)

- Seed‑resolutie: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR**.
- Identieke inputs + seed(s) → byte‑gelijke outputs (CSV/MD/JSON/PNG).
- RNG streams en hashing per scope/variabele/indices: `16_simulation_variables.md §4`.

## 6. Artefacten (indicatief)

- Public Tap: `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, charts.
- Private Tap: `private_vendor_recommendation.csv`, `private_tap_economics.csv`, charts.
- Gecombineerd: `consolidated_kpis.csv`, `consolidated_summary.{md,json}`.
- Rapport: `financial_plan_v3.md` (inclusief tabellen/grafieken).
 - Public growth: `public_tap_customers_by_month.csv` (zie `23_sim_public_tap.md`).
 - Private growth: `private_tap_customers_by_month.csv` (zie `22_sim_private_tap.md`).
 - Public capacity: `public_tap_capacity_plan.csv` (zie `23_sim_public_tap.md`).

## 7. Fouten & waarschuwingsbeleid

- `fail_on_warning: true` in `run` MAY escaleren van WARNINGs (bv. renormalisatie allocaties, YAML→CSV shadowing) naar **ERROR**.
- Schema‑/referentiefouten, seed‑ontbreken, variabele paden buiten scope → **ERROR**.

## 7.1 Doelen & Acceptatie (MUST)

- Horizon is losgekoppeld van lening: `targets.horizon_months = 18` voor haalbaarheid.
- Public: controleer monotone groei van `active_customers_m` en (optioneel) minimale MoM groei `targets.public_growth_min_mom_pct`.
- Private: idem voor `active_customers_m` (private).
- Marges: Private Tap marge per GPU‑klasse MUST ≥ `targets.private_margin_threshold_pct`.
- Log in `run_summary`: `{ accepted: true|false, horizon_months, public_active_customers: [...], private_active_customers: [...], monotonic_ok, min_mom_growth_ok, private_margin_threshold_pct }`.

## 8. Prestaties & Concurrency (SHOULD)

- `run.max_concurrency` MAY threads/processen limiteren.
- Stabiliteit verkiest determinisme boven maximale throughput.

## 9. UI‑integratie (Vue)

- UI bewerkt `inputs/operator/*.yaml` en `inputs/variables/*.csv` (facts readonly zichtbaar), zet seeds en start runs.
- UI toont voortgang, logs, en artefacten; zie `00_financial_plan_003.md §11`.

## 10. Zie ook

- `21_sim_general.md` — algemene sim‑regels (tijdsbasis, units, RNG, allocatie, rounding).
- `22_sim_private_tap.md`, `23_sim_public_tap.md` — pijplijndetails.
- `15_simulation_constants.md`, `16_simulation_variables.md` — runplan en variabelen.
- `10_inputs.md`, `11_operator_constants.md`, `19_facts.md` — inputs & validatie.
