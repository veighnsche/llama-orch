# 31 — Engine (Python) Architectuur & Best Practices

Status: Draft
Version: 0.1.0

## 1. Doel & Scope

- Definieert de engine‑architectuur voor D3: loader, validator, simulaties (Public/Private), consolidatie, outputs en CLI.
- Richtlijnen voor schaalbare, deterministische simulaties (grid → random replicates → Monte Carlo) met duidelijke scheiding tussen pipelines en services.

## 2. Kernprincipes (MUST)

- **Deterministisch** met seed‑hiërarchie en stabiele RNG‑streams (zie `16_simulation_variables.md`).
- **Pure berekeningen** in pipelines; I/O (YAML/CSV/MD/PNG) in `core/*` en `pipelines/*/artifacts.py`.
- **Scheid pipelines**: `pipelines/public/*` en `pipelines/private/*` voor groeiende complexiteit.
- **Services** voor gedeelde logica: `services/*` (rng, fx, acquisition, tps‑heuristics, autoscaling).
- **CLI‑only** bediening. Geen HTTP‑server in de engine.

## 3. Mappenstructuur

Zie `30_project_structure.md §2`. Belangrijkste directories onder `engine/src/d3_engine/`:

- `core/`: `loader.py`, `validator.py`, `rng.py`, `consolidate.py`, `logging.py` (JSONL/progress), `charts.py` (optioneel)
- `models/`: `inputs.py` (pydantic schema’s voor YAML/CSV), `outputs.py` (artefact contracten)
- `services/`: `fx_service.py`, `acquisition.py`, `tps_heuristics.py`, `autoscaling.py`
- `pipelines/public/`: `gpu_costs.py`, `pricing.py`, `demand.py`, `capacity.py`, `artifacts.py`
- `pipelines/private/`: `provider_costs.py`, `pricing.py`, `clients.py`, `artifacts.py`
- `cli.py`: CLI entrypoint

## 4. Data‑modellen (Pydantic v2) (SHOULD)

- In `models/inputs.py` definieer klassen voor:
  - `SimulationPlan` (mapping van `inputs/simulation.yaml`)
  - `OperatorGeneral`, `OperatorPublicTap`, `OperatorPrivateTap`
  - CSV‑rijen: `VariableRow` (common), curated GPU offers, curated public models
  - Facts: `MarketEnv`, `AdsChannels`, `AgencyFees`, `Insurances`
- In `models/outputs.py` definieer typed records voor tabellen (CSV/MD injectie) en helpers (to_csv with schema enforcement).

## 5. Loader & Validator (MUST)

- `core/loader.py`:
  - Lees YAML/CSV, normaliseer paden, voer overlay/precedence uit (CSV > YAML, met shadowing WARNING).
  - Bouw het variabelen‑grid (`low_to_high`) per scope; lever iterator over grid‑combinaties.
- `core/validator.py`:
  - Schema (kolomtitels), type/domeinen, referenties (curated lijstdekking), seeds, targets.
  - Warnings: allocatie som ≠ 1.0 (renormalisatie), onbekende modellen/GPUs met beperkte impact, TPS heuristiek in gebruik.

## 6. RNG & determinisme (MUST)

- `core/rng.py` levert:
  - Seed resolutie: `stochastic.random_seed` → `run.random_seed` → `meta.seed` per tap → ERROR.
  - `RNG_PARAMS(scope)` en `RNG_SIM(scope)` als `numpy.random.Generator(PCG64)` substreams:
    - Substream hashing: `H(master_seed, namespace, scope, variable_id?, grid_index, replicate_index, mc_index?)`.
  - Step‑kwantisatie: quantize naar `min + k*step` binnen `[min,max]`.
  - Cross‑process determinisme: bij parallelisatie `forkserver`/`spawn` gebruiken en substream‑seeds expliciet doorgeven.

## 7. Performance & schaal (SHOULD)

- Gebruik NumPy/Polars/Pandas voor vectorisatie. Vermijd diepe Python‑loops in MC.
- CSV I/O via Arrow/pyarrow waar nuttig (optioneel) voor snelheid en types.
- Parallelisatie opties (optioneel): `concurrent.futures.ProcessPoolExecutor` of `joblib`, maar behoud determinisme door substream‑seeds te fixeren.
- Micro‑opties: Numba/JAX zijn mogelijk later; start met pure NumPy voor eenvoud/portabiliteit.

## 8. Pipelines (MUST)

### 8.1 Public

- `gpu_costs.py`: EUR/hr uit provider USD/hr + FX buffer; bereken `cost_eur_per_1M(m,g)`, kies `g*`, log provider keuze.
- `pricing.py`: afleiding `sell_eur_per_1k(m)` met floors/caps/rounding + target marge.
- `demand.py`: maandmodel (budget0, groei, CAC, churn, tokens/conv) → klanten/tokens/credits per maand.
- `capacity.py`: autoscaling (benutting, piekfactor) → `instances_needed` en violations.
- `artifacts.py`: schrijf `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv`.

### 8.2 Private

- `provider_costs.py`: median EUR/hr per GPU‑klasse.
- `pricing.py`: verkoop EUR/hr + management/base fees.
- `clients.py`: maandmodel (budget0, groei, CAC, churn, uren/klant) → klanten/omzet.
- `artifacts.py`: schrijf `private_tap_economics.csv`, `private_vendor_recommendation.csv`, `private_tap_customers_by_month.csv`.

## 9. Consolidatie (MUST)

- `core/consolidate.py`: combineer KPI’s, percentielen, cashflows en schrijf `consolidated_kpis.csv` + `consolidated_summary.{md,json}`.
- Overhead‑allocatie driver: revenue | gpu_hours | tokens (zie `inputs/simulation.yaml`).

## 10. CLI‑contract (MUST)

- Command:
  ```bash
  python -m d3_engine.cli \
    --inputs .003-draft/inputs \
    --out .003-draft/outputs \
    --pipelines public,private \
    --seed 424242 \
    [--fail-on-warning] [--max-concurrency 4]
  ```
- Exit codes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR`.
- JSONL progress op stdout (één object/regel), velden (indicatief):
  - `{ ts, level, event, grid_index, replicate_index, mc_progress, message }`
- `run_summary.{json,md}` MUST loggen: seeds, input‑hashes, grid/replicates/MC, acceptatie, capacity‑violations, artefactenlijst.

## 11. Dependencies (aanbevolen)

- Python ≥ 3.11, `numpy`, `pandas` of `polars`, `pydantic>=2`, `pyyaml`/`ruamel.yaml`, `pyarrow` (optioneel), `typer` (CLI), `rich` (optioneel logging).

## 12. Testing (MUST)

- Unit tests: validator, rng, services.
- Golden tests: kleine fixture (2 modellen × 2 GPU’s) → prijzen/artefacten exact.
- Determinisme test: identieke inputs + seeds → byte‑gelijke CSV/MD/JSON.
- Performance smoke: run binnen N seconden op home profile.

## 13. Implementatievolgorde (SHOULD)

1) `models/*`, `core/loader.py`, `core/validator.py`, `core/rng.py`.
2) `pipelines/public/*` (costs → pricing → demand → capacity) + artifacts.
3) `pipelines/private/*` (provider costs → pricing → clients) + artifacts.
4) `core/consolidate.py`, charts helpers.
5) CLI + tests + proof bundle.
