# 30 — Project Structure (Engine & UI Split)

Status: Draft
Version: 0.1.0

## 1. Scope & Principes

- Schetst de te verwachten mappenstructuur en bouwblokken voor D3.
- Strikte scheiding tussen:
  - **Engine (Python)** — loader, validator, simulatie (Public/Private), consolidatie, outputs.
  - **UI (Vue + Vite)** — inputs bewerken (YAML/CSV), runs starten, voortgang/logs/outputs tonen.
- Geen netwerk tijdens engine‑run (MUST). UI praat lokaal met de engine (CLI).
- Determinisme met seeds (MUST). Alleen YAML/CSV inputs (MUST).

## 2. Top‑level indeling (Draft folder)

De D3‑implementatie leeft onder de draft‑root `/.003-draft/`:

```
.003-draft/
  README.md                    # Overzicht & quick start (zie README)
  inputs/                      # Bron‑inputs (YAML/CSV)
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
  outputs/                     # Artefacten (CSV/MD/JSON/PNG)
  engine/                      # Python engine (broncode/tests/venv)
    src/
      d3_engine/
        __init__.py
        cli.py                 # CLI entrypoint
        core/
          loader.py            # YAML/CSV ingest, overlay/precedence
          validator.py         # schema + referenties + domeinen
          rng.py               # seeds/streams
          consolidate.py       # consolidatie outputs
          logging.py           # JSONL progress, run_summary helpers (optioneel)
        models/
          inputs.py            # Pydantic modellen voor YAML/CSV shapes
          outputs.py           # Tabel/artefact contracten
        services/
          fx_service.py        # FX & buffers
          acquisition.py       # CAC/CVR/allocatie helpers
          tps_heuristics.py    # TPS heuristieken bij ontbreken dataset
          autoscaling.py       # capaciteit & instances-needed berekening
        pipelines/
          public/
            __init__.py
            gpu_costs.py       # eur_hr, cost €/1M, vendor choice
            pricing.py         # sell €/1k, floors/caps/rounding
            demand.py          # maandmodel: budget→conversies→tokens, churn
            capacity.py        # autoscaling plan
            artifacts.py       # CSV writers (public)
          private/
            __init__.py
            provider_costs.py  # median EUR/hr per GPU
            pricing.py         # sell EUR/hr + fees
            clients.py         # maandmodel: budget→clients→uren, churn
            artifacts.py       # CSV writers (private)
        charts.py              # (optioneel) grafiekhelpers
    tests/
      test_validator.py
      test_public_sim.py
      test_private_sim.py
    requirements.txt           # of pyproject.toml (poetry/uv/pip)
    Makefile                   # convenience (lint/test/run)
  ui/                          # Vue + Vite UI
    src/
      main.ts
      App.vue
      components/
        InputsEditor.vue       # YAML/CSV editor met schema hints
        RunPanel.vue           # seed/targets/pipelines triggers
        ProgressLogs.vue
        OutputsViewer.vue      # MD/CSV/Charts preview
    public/
    index.html
    package.json               # of pnpm via workspace
    vite.config.ts
  .specs/                      # Specificaties (bron van waarheid)
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
    30_project_structure.md    # dit document
```

Opmerking: daadwerkelijke codebestanden binnen `engine/` en `ui/` zijn indicatief; exacte module/filenaam kan verschillen zolang contracten gerespecteerd worden.

## 3. Engine ↔ UI contracten

Werkmodus: **CLI (MUST)**

- UI start de engine als proces met argumenten; engine schrijft artefacten naar `outputs/` en JSON‑status naar stdout.
- Voorbeeld aanroep:

  ```bash
  python -m d3_engine.cli \
    --inputs .003-draft/inputs \
    --out .003-draft/outputs \
    --pipelines public,private \
    --seed 424242
  ```

- Exit code 0 bij succes; non‑zero bij **ERROR** (validator of run‑fout). Stdout mag JSONL‑progress bevatten.

## 4. Data‑flow & bestanden

1) UI laat de gebruiker YAML/CSV bewerken in `inputs/` (facts readonly zichtbaar).
2) Bij ‘Run’:
   - UI schrijft eventuele unsaved changes naar `inputs/`.
   - Start engine (CLI) met `inputs/simulation.yaml` als stuurfile.
3) Engine flow (zie `20_simulations.md`):
   - Load → Validate → Variables grid → Random replicates → MC → Pipelines → Consolidatie.
4) Artefacten worden in `outputs/` geschreven en in de UI weergegeven (tabellen, charts, MD‑rapport).

## 5. Determinisme & targets

- Seeds: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR**.
- Targets (`inputs/simulation.yaml → targets.*`):
  - `horizon_months` (default 18)
  - `private_margin_threshold_pct` (default 20)
  - `require_monotonic_growth_public_active_customers` (bool)
  - `require_monotonic_growth_private_active_customers` (bool)
- Public autoscaling: `operator/public_tap.yaml → autoscaling.*`; plan in `public_tap_capacity_plan.csv`.

## 6. Dev‑workflow & scripts (aanbevolen)

- Engine (Python)
  - `make venv` · `make lint` · `make test` · `make run`
  - of via uv/poetry: `uv run python -m d3_engine.cli ...`
- UI (Vue)
  - `pnpm install` · `pnpm dev` (Vite) · `pnpm build`
- Validatie quick‑loop
  - UI: edit → save → run → bekijk `outputs/run_summary.{json,md}`

## 7. Validatie & logging

- Validator MUST schema/precedentie/regeldomeinen afdwingen (zie `10_inputs.md` en `11_operator_constants.md`).
- Bij YAML→CSV shadowing MUST een duidelijke WARNING worden gelogd.
- `run_summary.{json,md}` MUST seeds, input‑hashes, grid/replicates/MC, acceptatie, en eventuele capacity‑violations loggen.

## 8. Security & I/O

- Geen netwerk tijdens runs (MUST). Engine leest uitsluitend lokale `inputs/` en schrijft naar `outputs/`.
- Secrets niet vereist; facts zijn lokaal.

## 9. Kwaliteit & linting (SHOULD)

- Python: formatter (black), linter (ruff/flake8), tests (pytest), type‑hints (mypy optioneel).
- Vue: eslint + prettier; component‑tests (vitest) optioneel.
- CI (optioneel): format/lint/test jobs; artefacten publiceren uit `outputs/`.

## 10. Migratie & compatibiliteit

- Geen backwards compatibility pre‑1.0 (MUST).
- Legacy multi‑YAML uit D2 wordt niet ondersteund. Eventueel migratiescript kan YAML samenvoegen naar D3‑vorm.
