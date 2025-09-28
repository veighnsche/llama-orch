# D3 Engine — Production Readiness Checklist

Status: Draft · Versie: 0.1.0
Locatie: `.business/financieelplan_NL_nl/.003-draft/engine/`

Gebruik deze checklist als GO/NO‑GO gate voor productie. Alle MUST‑items moeten afgevinkt zijn. Verwijs bij twijfel naar specs: `30_project_structure.md`, `31_engine.md`, `40_testing.md`, `41_engine.md`, `22/23` pipeline‑specs.

## 1. Architectuur & Scope

- [ ] CLI‑only (geen server). `d3_engine/cli.py` biedt alles wat nodig is (help, flags, exit‑codes, JSONL progress).
- [ ] Geen netwerk tijdens runs (MUST). Alle I/O is lokaal (`inputs/`, `outputs/`).
- [ ] Mapstructuur volgt `30_project_structure.md` (core/, models/, services/, pipelines/public|private/).
- [ ] Core‑modules aanwezig en gedocumenteerd (zie `21_engine_flow.md`): `core/runner.py`, `core/variables.py`, `core/simulate.py`, `core/aggregate.py`, `core/acceptance.py`, `analysis/*`, `behavior/*`.
- [ ] Engine flow geïmplementeerd conform `21_engine_flow.md` met JSONL events (`run_start`, `load_*`, `validate_*`, `grid_built`, `job_*`, `aggregate_done`, `analysis_done`, `acceptance_checked`, `run_done`).
- [ ] Code in pipelines is puur (berekeningen), I/O is geconcentreerd in core/artifacts.

## 2. Inputs, Schema’s & Validatie

- [ ] Loader ondersteunt alle ingestelde inputs: `inputs/simulation.yaml`, `operator/*`, `variables/*`, `facts/*`.
- [ ] Precedence correct: CSV > YAML met duidelijke WARNING bij shadowing; escalatie naar **ERROR** bij `run.fail_on_warning: true`.
- [ ] Allowed path roots afgedwongen per scope; onbekende paden → **ERROR** (zie `12_oprator_variables.md`).
- [x] CSV kolom‑schema’s (exacte headers) en units (`percent|fraction|EUR|months|tokens|count`) strikt gevalideerd (variables/*.csv).
- [x] Curated GPU/provider schema gevalideerd: vereist óf subset `{provider,gpu_vram_gb,price_per_gpu_hr}` óf `{provider,gpu_vram_gb,price_usd_hr,num_gpus}`; afgeleide `usd_hr>0` en `gpu_vram_gb>0` (mapping gedocumenteerd in validator).
- [x] Curated public models CSV accepteert case‑insensitive kolom `model` en vereist niet‑lege waarden; ≥1 rij.
- [x] `simulation.yaml` bevat `targets.autoscaling_util_tolerance_pct` (acceptatieband p95 rond target util).

## 3. RNG & Determinisme

- [ ] Seed‑resolutie: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR**.
- [ ] RNG substreams per scope en variable_id (PCG64) met stabiele hashing (grid_index, replicate_index, mc_index).
- [ ] `low_to_high` grid + `random` replicates + MC nesting geïmplementeerd zoals `16_simulation_variables.md`.
- [ ] Optioneel transcript `variable_draws.csv` (aanbevolen) met `scope,variable_id,path,grid_index,replicate_index,draw_value`.
- [ ] Concurrency produceert identieke resultaten (hash‑gelijk) bij gelijke seeds.

## 4. Pipelines — Public

- [ ] Kostenberekening: USD/hr → EUR/hr (FX + buffer), TPS (dataset of heuristiek), `cost_eur_per_1M` per model+GPU.
- [ ] GPU/Provider keuze (`g*`) per model op minimaal `cost_eur_per_1M`; tie‑breakers gedocumenteerd.
- [ ] Pricing €/1k tokens: floors, caps, afronding, target margins volgens policy.
- [ ] Vraagmodel: budget0, groei MoM, CAC, churn, tokens per conversie → klanten/tokens per maand (via `behavior.public.simulate_months()`; zie `24_customer_behavior.md`).
- [ ] Autoscaling/capaciteit: target utilization, peak factor, min/max instances, violations.
- [ ] Artefacten schrijven: `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv` (headers conform `40_testing.md`).
- [ ] `public_tap_scaling_events.csv` met events/metrics (zie `25_autoscaling.md`).
  - [x] Autoscaler simulator policy‑sleutels aanwezig in `operator/public_tap.yaml` (`evaluation_interval_s`, thresholds, steps, windows, warmup/cooldown).

## 5. Pipelines — Private

- [ ] Provider EUR/hr medianen per GPU‑klasse (per provider, indien van toepassing).
- [ ] Pricing EUR/hr + management/base fees; margins expliciet in output.
- [ ] Klantenmodel: budget→conversies, churn, (optioneel) uren/klant per maand (via `behavior.private.simulate_months()`; zie `24_customer_behavior.md`).
- [ ] Artefacten schrijven: `private_tap_economics.csv`, `private_vendor_recommendation.csv` (indien aanwezig), `private_tap_customers_by_month.csv`.

## 6. Consolidatie & Rapportage

- [ ] `consolidated_kpis.csv` en `consolidated_summary.{md,json}` met percentielen (`stochastic.percentiles`).
- [ ] Overhead‑allocatie driver (revenue | gpu_hours | tokens) juist toegepast.
- [ ] `run_summary.{json,md}` bevat: seeds, input‑hashes, grid/replicates/MC, acceptatiekeuringen (OK/FAILED), artefactenlijst.
- [ ] JSONL progress events bevatten minimaal `ts,level,event` en nuttige velden (grid_index, replicate_index, mc_progress).
- [ ] Analyse‑laag (`analysis/*`) produceert KPI’s/percentielen/sensitivity; resultaten opgenomen in `run_summary` en consolidatie‑artefacten.
- [ ] Deterministische merge/aggregatie volgens `21_engine_flow.md §3.1`.

## 7. Acceptatie & Policy‑checks (MUST)

- [ ] Monotone groei public/private: `active_customers_m[i+1] ≥ active_customers_m[i]` over horizon (`targets.horizon_months`).
- [ ] (Optioneel) Minimale MoM groei public (indien target gezet).
- [ ] Private margin per GPU‑klasse ≥ `targets.private_margin_threshold_pct`.
- [ ] Capaciteitsviolaties: geen of duidelijk gelogd; bij overschrijding **ERROR** of **WARNING** volgens `run.fail_on_warning`.

## 8. Performance & Schaal

- [ ] Minimal fixture E2E ≤ N s, RAM < M MB (waarden ingevuld en gemeten op home‑profile).
- [ ] Vectorisatie (NumPy/Polars/Pandas) voor MC‑delen; geen diepe Python loops bij grote iteraties.
- [ ] Grote CSV’s: streaming/lazy waar nuttig; stabiele memory‑footprint; geen OOM.
- [ ] `--max-concurrency` benut parallelisme zonder nondeterminisme; documenteer limieten.

## 9. Foutafhandeling & Robuustheid

- [ ] Heldere foutcodes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR`.
- [ ] Menselijke foutboodschappen met pad/kolom/regel indicaties; geen stacktraces zonder context.
- [ ] Onbekende/extra kolommen → **ERROR** of **WARNING** per policy; alle gevallen gedocumenteerd.
- [ ] Path traversal/unsafe writes voorkomen; outputs directory wordt aangemaakt met `parents=True`.
- [ ] Timeouts of afbreken (Ctrl‑C) leidt tot nette afsluiting (partiële outputs consistent).

## 10. Beveiliging & Compliance

- [ ] Geen secrets/extern netwerk; facts zijn lokaal.
- [ ] `requirements.txt` geaudit (`pip audit`), licenties oké.
- [ ] Onbetrouwbare input‑waarden defensief gehanteerd (range checks, NaN/inf guards).

## 11. Kwaliteit & Linting

- [ ] Formatter (black) en linter (ruff/flake8) schoon; type‑hints waar zinvol (mypy optioneel).
- [ ] Dode code/experimentele paden opgeschoond; comments/docstrings up‑to‑date.

## 12. Testen (zie `40_testing.md`, `41_engine.md`)

- [ ] Unit tests voor core, services, behavior en pipelines; randgevallen gedekt.
- [ ] BDD (pytest‑bdd) scénarios slagen:
  - RNG & Determinisme (`42_rng_determinism.md`)
  - CLI‑contract (`43_cli_contract.md`)
  - Public pricing (`44_public_pricing.md`)
  - Public capacity (`45_public_capacity.md`)
  - Private economics (`46_private_economics.md`)
  - Consolidatie (`47_consolidation.md`)
  - Performance & concurrency (`48_performance_concurrency.md`)
  - Acceptance (`40_testing.md §5`) en behavior monotoniciteit (`24_customer_behavior.md`)
- [ ] Golden/determinism tests met fixtures: byte‑gelijke artefacten + `SHA256SUMS`.
- [ ] Coverage ≥ 80% voor services/pure pipeline/behavior code.

## 13. CI & Build

- [ ] CI jobs: format/lint, pytest (unit+BDD), golden‑diff, cache pnpm/pip wheels.
- [ ] Artefacten upload bij failures (CSV/MD/JSON, logs, `run_summary`).
- [ ] Makefile targets werken: `venv`, `install`, `run`, `test`.

## 14. Documentatie

- [ ] CLI usage (`--help`) en voorbeelden in README/Docs.
- [ ] Proof bundle instructies: waar outputs/`SHA256SUMS` terechtkomen.
- [ ] Behavior specs gedekt in `24_customer_behavior.md` en door pipelines gerefereerd.
- [ ] Test‑specs gedekt: `42..48` (RNG, CLI, public pricing/capacity, private economics, consolidatie, performance).
- [x] GPU‑targeting gedocumenteerd: primair consumer‑grade GPU’s (RTX 4090/3090, L‑serie waar passend) voor Public Tap benchmarks.

## 15. Go/No‑Go

- [ ] Alle MUST‑items hierboven zijn groen.
- [ ] Laatste smoke op home‑profile met standaard seeds afgerond; `run_summary` OK.

## 16. Spec Realignment (Planning Only)

Doel: deze sectie legt uitsluitend planningsbesluiten en document‑realignments vast. Er worden hier GEEN implementaties gedaan. Alle verwijzingen zijn naar specs/inputs die in deze D3‑fase leidend zijn.

### 16.1 Beslissingen om te bevriezen

- **[pricing_policy nesting]** Gebruik geneste paden in alle documentatie en voorbeelden:
  - Public: `pricing_policy.public_tap.target_margin_pct`.
  - Private: `pricing_policy.private_tap.{default_markup_over_provider_cost_pct, management_fee_eur_per_month, vendor_weights.*}`.
- **[autoscaling tolerantie]** Default voor `targets.autoscaling_util_tolerance_pct` = 25 (kan later bijgesteld worden in specs; implementatie volgt pas na design freeze).
- **[curated GPU norm]** Rijkere CSV’s toegestaan; vereiste subset kolommen: `{provider,gpu_vram_gb,price_per_gpu_hr}`. Normaliseer naar interne rentals‑vorm `[gpu, vram_gb, provider, usd_hr]` (pre‑1.0: geen min/max/percentvelden in bron).

### 16.2 Document‑updates (alleen specs, geen code)

- **`/.specs/00_financial_plan_003.md`**
  - Verduidelijk geneste `pricing_policy` paden (Public/Private) als MUST.
  - Benoem expliciet default en acceptatieband voor autoscaling util (tolerantie).
- **`/.specs/10_inputs.md`**
  - Expliciteer normalisatie: curated GPU CSV ⟶ interne rentalsvorm; CSV>YAML shadowing als WARNING loggen.
- **`/.specs/12_oprator_variables.md`**
  - Harmoniseer voorbeelden op geneste paden; verwijder ambigue/alternatieve paden.
  - Controleer unit‑labels consistentie (bijv. `EUR` vs `EUR_per_month`).
- **`/.specs/15_simulation_constants.md`**
  - Neem `targets.autoscaling_util_tolerance_pct` (default 25) op in YAML‑voorbeeld en MUST‑regels.
- **`/.specs/33_engine_flow.md`**
  - Werk loader‑architectuur uit: merge‑orde, allowed roots‑enforcement, input‑hashes, JSONL events (`grid_built`, `job_*`).
  - Documenteer RNG‑substreams (PCG64, namespacing over grid/replicate/MC) en seed‑resolutie.
- **`/.specs/25_autoscaling.md` & `/.specs/40_testing.md`**
  - Veranker acceptatieregel p95(util) ± tolerantie en (optioneel) events‑CSV als SHOULD.

### 16.3 Acceptatie & Testing verduidelijkingen

- **[monotone groei]** Public/Private: `active_customers_m[i+1] ≥ active_customers_m[i]` over `targets.horizon_months` (MUST).
- **[min MoM groei]** Optioneel voor Public; afdwingen indien target gezet (MUST wanneer geconfigureerd).
- **[private marge]** Per GPU‑klasse `margin_pct ≥ targets.private_margin_threshold_pct` (MUST).
- **[capaciteit]** Planner: `instances_needed ≤ autoscaling.max_instances_per_model`; overschrijding → violation (policy → WARNING of ERROR o.b.v. `run.fail_on_warning`).
- **[determinisme]** Identieke inputs + seeds → byte‑gelijke artefacten; concurrency mag geen drift veroorzaken; hash‑vergelijking via `SHA256SUMS` (MUST in golden tests).

### 16.4 Loader & RNG ontwerp (te documenteren; geen implementatie)

- **Loader/overlay**: volgorde = Operator YAML + curated CSV → Variables overlay → Facts (read‑only). CSV wint bij dataset‑dubbeling (shadowing WARNING). Enforce allowed roots per scope.
- **Input‑hashes & JSONL**: registreer input‑hashes, log `load_*`, `validate_*`, `grid_built`, `job_*`, `analysis_done`, `acceptance_checked`.
- **RNG substreams**: PCG64 met stabiele hashing namespaces: `(scope, variable_id, grid_index, replicate_index, mc_index)` en seed‑resolutie zoals gespecificeerd.

### 16.5 Open vragen (beslissen in specs, geen code)

- **[Private paden]** Bevestigen dat `management_fee_eur_per_month` en `vendor_weights.*` onder `pricing_policy.private_tap.*` blijven (aanbevolen) i.p.v. `prepaid_policy.private_tap.*`.
- **[Tolerantie default]** Handhaaf 25% als default voor `targets.autoscaling_util_tolerance_pct` of verlagen naar 20%/15%?
- **[TPS‑fallback]** Bij ontbrekende TPS: heuristiek toestaan met duidelijke WARNING en bronvermelding (spec staat dit toe; criteria nader aanscherpen in docs).

### 16.6 Fixtures & UI scope (documenteren)

- **Fixtures**: definieer `minimal_001` en `stress_001` (kleine curated sets, vaste seeds, verwachte artefactenlijst) onder `/.003-draft/tests/fixtures/` (alleen docs, golden later).
- **UI scopefreeze**: minimale lokale UI (Vite) zonder netwerk; bewerken van operator‑YAML en variables‑CSV, validatiefeedback, run‑trigger via CLI‑bridge, preview van MD/CSV/PNG/JSON).
