# 21 — Engine Flow (Runner & Contracts)

Status: Draft
Version: 0.1.0

Deze pagina beschrijft het canonieke proces:

Load → Validate → Variables grid → Random replicates → MC → Parallel jobs (pipelines) → Aggregation → Analysis → Acceptance → Artifacts → Summary (CLI)

Alle regels zijn normatief in RFC‑2119 termen.

## 1. Scope & Principes

- **CLI‑only (MUST)**. Geen netwerk tijdens runs. Alle I/O is lokaal (`inputs/`, `outputs/`).
- **Determinisme (MUST)**. Identieke inputs + seeds → byte‑gelijke outputs. Concurrency mag geen drift veroorzaken.
- **Scheiding van zorgen (MUST)**. Pure berekeningen in pipelines/behavior, orchestration in `core/`, I/O in writers/cli.
- **Spec‑first (MUST)**. Dit document leidend voor implementatie en tests.

## 2. Stappen (Stages)

1) **Load inputs** (MUST)
   - Lees `inputs/simulation.yaml` → `SimulationPlan`.
   - Lees operator YAMLs (`inputs/operator/*.yaml`) + curated CSV’s (`curated_gpu.csv`, modellen).
   - Lees variabelen CSV’s (`inputs/variables/*.csv`), facts (`inputs/facts/*`).
   - Normaliseer paden; bereken input‑hashes.

2) **Validate** (MUST)
   - Schema’s (exacte headers), domeinen, referenties; CSV > YAML (shadowing WARNING).
   - Allowed roots voor variabelenpaden; onbekende paden → **ERROR**.
   - Seed‑resolutie en targets check; curated GPU shape `[gpu,vram_gb,provider,usd_hr]` afdwingen.

3) **Variables grid** (MUST)
   - Bouw cartesisch product over alle `low_to_high` variabelen per scope; kwantiseer naar `min + k*step`.
   - Log `grid_built` met `grid_size`.

4) **Random replicates** (MUST)
   - Per grid‑combinatie `run.random_runs_per_simulation` replicates; herteken alle `random` variabelen.
   - Optioneel `variable_draws.csv` transcript (`scope,variable_id,path,grid_index,replicate_index,draw_value`).

5) **Monte Carlo** (MUST)
   - Binnen een replicate `stochastic.simulations_per_run` iteraties voor funnel/uitkomststochastiek.

6) **Jobs genereren** (MUST)
   - Maak jobs `(grid_index, replicate_index, pipelines, seed)` deterministisch.
   - Sorteer lexicografisch op `(grid_index, replicate_index)`.

7) **Parallel jobs uitvoeren** (MUST)
   - Plan jobs met `run.max_concurrency` (threads/processen) zonder resultaatdrift.
   - Pipelines roepen behavior aan voor maandreeksen:
     - Public: kosten → `sell €/1k` → vraag/credits → capaciteit.
     - Private: median EUR/hr → `sell €/hr` → uren/clients.

8) **Aggregation** (MUST)
   - Verzamel job‑outputs in vaste orde; bereken KPI’s/percentielen per `stochastic.percentiles`.

9) **Analysis** (SHOULD)
   - Analyseer gevoeligheden t.o.v. variabelen (grid‑assen), scenario‑KPIs.

10) **Acceptance** (MUST)
    - Monotone groei public/private (`active_customers_m`).
    - (Optioneel) minimale MoM groei public.
    - Private marges ≥ `targets.private_margin_threshold_pct`.
    - Capaciteit: `instances_needed ≤ autoscaling.max_instances_per_model`.
    - `fail_on_warning: true` MAY escaleren naar **ERROR**.

11) **Artifacts** (MUST)
    - Public: `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv`.
    - Private: `private_tap_economics.csv`, `private_vendor_recommendation.csv`, `private_tap_customers_by_month.csv`.
    - Consolidatie: `consolidated_kpis.csv`, `consolidated_summary.{md,json}`.

12) **Summary (CLI)** (MUST)
    - `run_summary.{json,md}`: seeds, input‑hashes, grid/replicates/MC, acceptatie, artefactenlijst.

## 3. Determinisme & RNG (MUST)

- Seed‑resolutie: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR**.
- Substreams met PCG64: `H(master_seed, namespace, scope, variable_id?, grid_index, replicate_index, mc_index?)`.
- Concurrency veilig: expliciete seeds doorgeven; BLAS threads pinnen voor stabiliteit.
- Schrijfdiscipline: single writer of staging+merge om volgorde en bytes te fixeren.

## 4. JSONL Progress (MUST)

Minimaal `{ ts, level, event }`. Indicatieve events en extra velden:

- `run_start {inputs, out, pipelines, seed}`
- `load_start`, `load_done`
- `validate_start`, `validate_done`
- `grid_built {size}`
- `job_submitted {grid_index, replicate_index}`
- `pipeline_public_start`, `pipeline_public_done`
- `pipeline_private_start`, `pipeline_private_done`
- `aggregate_done`
- `analysis_done`
- `acceptance_checked`
- `run_done`

## 5. CLI‑contract (MUST)

```bash
python -m d3_engine.cli \
  --inputs .003-draft/inputs \
  --out .003-draft/outputs \
  --pipelines public,private \
  --seed 424242 \
  [--fail-on-warning] [--max-concurrency 4]
```

- Exit codes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR`.
- `--pipelines` accepteert `public,private` of subset.

## 6. Data‑contracten (MUST)

- Inputs: `SimulationPlan`, `OperatorBundle`, `VariableRow[]`, facts (`MarketEnv`, etc.). Zie `models/inputs.py`.
- Outputs: CSV‑schema’s zoals in `40_testing.md §4` en `00_financial_plan_003.md Bijlage A`.
- Transcript: `variable_draws.csv` (SHOULD) voor herleidbaarheid.

## 7. Fouten & waarschuwingen

- Onbekende paden/kolommen → **ERROR**.
- YAML→CSV shadowing → **WARNING** (of **ERROR** bij `fail_on_warning`).
- Allocatie som ≠ 1.0 → **WARNING** + renormalisatie (of **ERROR** bij policy).

## 8. Performance & Concurrency (SHOULD)

- Minimal fixture E2E ≤ N sec; RAM < M MB (waarden vastleggen in `41_engine.md`).
- `run.max_concurrency` moet identieke hashes opleveren.

## 9. Implementatie‑mapping (code)

- `core/loader.py` — Load
- `core/validator.py` — Validate
- `core/variables.py` — Grid & transcripts
- `core/rng.py` — Seeds & streams
- `core/simulate.py` — Jobs uitvoeren (roept pipelines en behavior)
- `core/aggregate.py` — Aggregation
- `analysis/*` — Analysis (KPIs/percentiles/sensitivity)
- `core/acceptance.py` — Acceptance
- `pipelines/*/artifacts.py` — Writers (CSV)
- `core/runner.py` — Orchestrator
- `cli.py` — Summary & exit code

## 10. Testen & verificatie

- BDD: `cli_contract.feature`, `determinism.feature`, `acceptance.feature`.
- Golden: byte‑gelijke artefacten + `SHA256SUMS`.
- Unit: validator, rng, behavior, services, pipelines (pure).

## 11. Referenties

- `20_simulations.md` — overzicht simulaties
- `16_simulation_variables.md` — treatments, RNG, grid/replicates/MC
- `40_testing.md` · `41_engine.md` — testregime
- `00_financial_plan_003.md` — top‑niveau eisen en artefacten
- `30_project_structure.md` — mapstructuur
