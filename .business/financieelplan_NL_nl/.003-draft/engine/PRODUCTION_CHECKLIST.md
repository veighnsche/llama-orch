# D3 Engine — Production Readiness Checklist

Status: Draft · Versie: 0.1.0
Locatie: `.business/financieelplan_NL_nl/.003-draft/engine/`

Gebruik deze checklist als GO/NO‑GO gate voor productie. Alle MUST‑items moeten afgevinkt zijn. Verwijs bij twijfel naar specs: `30_project_structure.md`, `31_engine.md`, `40_testing.md`, `41_engine.md`, `22/23` pipeline‑specs.

## 1. Architectuur & Scope

- [ ] CLI‑only (geen server). `d3_engine/cli.py` biedt alles wat nodig is (help, flags, exit‑codes, JSONL progress).
- [ ] Geen netwerk tijdens runs (MUST). Alle I/O is lokaal (`inputs/`, `outputs/`).
- [ ] Mapstructuur volgt `30_project_structure.md` (core/, models/, services/, pipelines/public|private/).
- [ ] Code in pipelines is puur (berekeningen), I/O is geconcentreerd in core/artifacts.

## 2. Inputs, Schema’s & Validatie

- [ ] Loader ondersteunt alle ingestelde inputs: `inputs/simulation.yaml`, `operator/*`, `variables/*`, `facts/*`.
- [ ] Precedence correct: CSV > YAML met duidelijke WARNING bij shadowing; escalatie naar **ERROR** bij `run.fail_on_warning: true`.
- [ ] Allowed path roots afgedwongen per scope; onbekende paden → **ERROR** (zie `12_oprator_variables.md`).
- [ ] CSV kolom‑schema’s (exacte headers) en units (`percent|fraction|EUR|months|tokens|count`) strikt gevalideerd.
- [ ] Curated GPU/provider schema is exact `[gpu,vram_gb,provider,usd_hr]` (geen min/max/percent) (MUST).

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
- [ ] Vraagmodel: budget0, groei MoM, CAC, churn, tokens per conversie → klanten/tokens per maand.
- [ ] Autoscaling/capaciteit: target utilization, peak factor, min/max instances, violations.
- [ ] Artefacten schrijven: `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv` (headers conform `40_testing.md`).

## 5. Pipelines — Private

- [ ] Provider EUR/hr medianen per GPU‑klasse (per provider, indien van toepassing).
- [ ] Pricing EUR/hr + management/base fees; margins expliciet in output.
- [ ] Klantenmodel: budget→conversies, churn, (optioneel) uren/klant per maand.
- [ ] Artefacten schrijven: `private_tap_economics.csv`, `private_vendor_recommendation.csv` (indien aanwezig), `private_tap_customers_by_month.csv`.

## 6. Consolidatie & Rapportage

- [ ] `consolidated_kpis.csv` en `consolidated_summary.{md,json}` met percentielen (`stochastic.percentiles`).
- [ ] Overhead‑allocatie driver (revenue | gpu_hours | tokens) juist toegepast.
- [ ] `run_summary.{json,md}` bevat: seeds, input‑hashes, grid/replicates/MC, acceptatiekeuringen (OK/FAILED), artefactenlijst.
- [ ] JSONL progress events bevatten minimaal `ts,level,event` en nuttige velden (grid_index, replicate_index, mc_progress).

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

- [ ] Unit tests voor core, services en pipelines; randgevallen gedekt.
- [ ] BDD (pytest‑bdd) scénarios slagen: CLI‑contract, determinisme, validatie, acceptance, capacity.
- [ ] Golden/determinism tests met fixtures: byte‑gelijke artefacten + `SHA256SUMS`.
- [ ] Coverage ≥ 80% voor services/pure pipeline code.

## 13. CI & Build

- [ ] CI jobs: format/lint, pytest (unit+BDD), golden‑diff, cache pnpm/pip wheels.
- [ ] Artefacten upload bij failures (CSV/MD/JSON, logs, `run_summary`).
- [ ] Makefile targets werken: `venv`, `install`, `run`, `test`.

## 14. Documentatie

- [ ] `31_engine.md` en `30_project_structure.md` up‑to‑date.
- [ ] CLI usage (`--help`) en voorbeelden in README/Docs.
- [ ] Proof bundle instructies: waar outputs/`SHA256SUMS` terechtkomen.

## 15. Go/No‑Go

- [ ] Alle MUST‑items hierboven zijn groen.
- [ ] Laatste smoke op home‑profile met standaard seeds afgerond; `run_summary` OK.
- [ ] Beslissing vastgelegd met datum, seed, commit‑hash en reviewers.
