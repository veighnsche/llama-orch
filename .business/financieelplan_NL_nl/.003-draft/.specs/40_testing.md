# 40 — Testing Regimen (D3 Engine & UI)

Status: Draft
Version: 0.1.0

## 1. Scope & Doelen

- Definieert het testprogramma voor D3 (engine + UI).
- Waarborgt determinisme, correctheid (acceptatiecriteria), performance, en developer‑ergonomie.
- Alle regels in RFC‑2119 termen: MUST/SHOULD/MAY.

## 2. Testniveaus

- **Unit — Engine (MUST)**
  - `core/validator.py`: schema, domeinen, referenties; foutcodes (`VALIDATION_ERROR`).
  - `core/rng.py`: seed‑hiërarchie, substreams, kwantisatie.
  - `services/*`: `fx_service`, `acquisition`, `tps_heuristics`, `autoscaling` (edge‑cases: 0/negatief/inf).
  - Pipelines: pure functies in `pipelines/public/*` en `pipelines/private/*` (geen I/O).

- **Unit — UI (SHOULD)**
  - Components renderen, props/emit, eenvoudige stores; smoke voor editors/viewers.
  - CLI‑wrapper stub (spawn) en progress parsing.

- **Contract — CLI (MUST)**
  - Exit‑codes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR`.
  - JSONL progress bevat velden: `{ ts, level, event, ... }` (minimaal `ts,level,event`).
  - `--help` output, verplichte argumenten, onbekende flags → non‑zero.

- **Golden/Determinism — Engine (MUST)**
  - Fixture(s) met vaste seeds produceren byte‑gelijke artefacten (CSV/MD/JSON) en identieke `variable_draws.csv`.
  - Hash‑vergelijking (SHA‑256) op outputs.

- **Acceptance (MUST)**
  - Public: monotone groei over `targets.horizon_months` en (indien ingesteld) minimale MoM groei.
  - Private: marge per GPU‑klasse ≥ `targets.private_margin_threshold_pct`.
  - Capaciteit: `instances_needed(m,g*) ≤ autoscaling.max_instances_per_model`.
  - Falen → **ERROR** of **WARNING** (afhankelijk van `run.fail_on_warning`).

- **Integration/E2E (SHOULD)**
  - End‑to‑end run op minimale fixture; verwacht artefacten aanwezig en headers kloppen.
  - UI integratie: spawn CLI, toon progress, laad CSV’s en render één chart/tab.

- **Performance/Smoke (SHOULD)**
  - Home‑profile budget: volledige minimale run ≤ N seconden, geheugen < M MB.
  - Concurrency vlag (`run.max_concurrency`) geen nondeterminisme.

## 3. Fixtures & Indeling

- **Locatie (SHOULD)**: `/.003-draft/tests/fixtures/`
  - `minimal_001/` — klein model/GPU set, 2–3 rijen per CSV, standaard seeds.
  - `stress_001/` — p90 drift/FX buffers, autoscaling dicht bij max.
- Fixtures MUST compleet zijn: `inputs/` subfolders + `simulation.yaml` + verwachte `outputs/` (voor golden).

## 4. Artefact Schema‑checks (MUST)

- Public:
  - `public_vendor_choice.csv`: `model,gpu,provider,usd_hr,eur_hr_effective,cost_eur_per_1M`
  - `public_tap_prices_per_model.csv`: `model,gpu,cost_eur_per_1M,sell_eur_per_1k,margin_pct`
  - `public_tap_customers_by_month.csv`: `month,budget_eur,cac_eur,expected_new_customers,active_customers,tokens,...`
  - `public_tap_capacity_plan.csv`: `model,gpu,avg_tokens_per_hour,peak_tokens_per_hour,tps,cap_tokens_per_hour_per_instance,instances_needed,target_utilization_pct,capacity_violation`
- Private:
  - `private_tap_economics.csv`: `gpu,provider_eur_hr_med,markup_pct,sell_eur_hr,margin_eur_hr,margin_pct`
  - `private_tap_customers_by_month.csv`: `month,private_budget_eur,private_cac_eur,expected_new_clients,active_clients,hours,sell_eur_hr,revenue_eur`
- Consolidatie:
  - `consolidated_kpis.csv`, `consolidated_summary.{md,json}`

## 5. Acceptance‑regels (formeel)

- **Monotone groei (Public/Private)**: `active_customers_m[i+1] ≥ active_customers_m[i]` voor `i∈[0..H-2]` (met H=`targets.horizon_months`).
- **Minimale groei (optioneel)**: `(active[i+1]-active[i])/max(active[i],1) ≥ targets.public_growth_min_mom_pct/100`.
- **Private marge**: `margin_pct(gpu_class) ≥ targets.private_margin_threshold_pct` voor alle klassen in scope.
- **Capaciteit**: `instances_needed(m,g*) ≤ autoscaling.max_instances_per_model`; overschrijding = violation.

## 6. Commando’s

- **Engine**
  - Setup: `make -C .business/financieelplan_NL_nl/.003-draft/engine install`
  - Tests: `make -C .003-draft/engine test` of `pytest -q .003-draft/engine/tests`
  - Run: `make -C .003-draft/engine run`
- **UI**
  - Install: `pnpm -w install`
  - Dev: `pnpm -F orchyra-d3-sim-frontend dev`
  - Lint/format: `pnpm -F orchyra-d3-sim-frontend lint` · `format`
  - Tests (SHOULD): `pnpm -F orchyra-d3-sim-frontend test` (vitest)

## 7. Determinisme & Proof Bundles (MUST)

- Elke golden run bewaart:
  - `run_summary.{json,md}`, `variable_draws.csv` (indien toegepast), alle output‑CSV’s, en `SHA256SUMS`.
- Bundel onder `/.docs/testing/` of `/.proof_bundle/` met datum/seed/commit.
- Identieke inputs + seeds → identieke hashes.

## 8. CI‑Richtlijnen (SHOULD)

- Jobs: format/lint (UI), pytest (engine), UI vitest (als toegevoegd), golden‑diff (engine minimal fixture).
- Cache pnpm en pip wheels. Artefacten (CSV/MD/JSON) uploaden bij failures.

## 9. Fouten & Waarschuwingen

- Onbekende kolommen/paths → **ERROR**.
- Allocatie som ≠ 1.0 → **WARNING** met renormalisatie of **ERROR** bij `fail_on_warning`.
- TPS heuristiek gebruikt (ontbrekende dataset) → **WARNING** + log ‘heuristic_path’.

## 10. Kwaliteitsbalken

- Unit coverage (engine services/pipelines) ≥ 80% (SHOULD) voor core‑functies.
- Minimal fixture e2e binnen N sec (SHOULD) op home‑profile.

## 11. Referenties

- `30_project_structure.md` — split & folders
- `31_engine.md` — engine architectuur, CLI‑contract
- `32_ui.md` — UI architectuur & CLI koppeling
- `16_simulation_variables.md` — grid/replicates/MC & RNG
- `20_simulations.md` — runner/artefacten
- `22_sim_private_tap.md` · `23_sim_public_tap.md` — pijplijnregels
