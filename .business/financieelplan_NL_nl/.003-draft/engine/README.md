# D3 Engine (Draft 0.1.0)

CLI-only deterministic simulation engine for the FP v3 draft.

## CLI Usage

```
python -m d3_engine.cli \
  --inputs /path/to/.003-draft/inputs \
  --out /path/to/outputs \
  --pipelines public,private \
  --seed 424242 \
  --fail-on-warning \
  --max-concurrency 4
```

- Exit codes: `0=OK`, `2=VALIDATION_ERROR`, `3=RUNTIME_ERROR/INTERRUPTED`.
- Emits JSONL progress to stdout. Key events: `run_start`, `load_*`, `validate_*`, `grid_built`, `job_*`, `analysis_done`, `aggregate_done`, `acceptance_checked`, `run_done`.

## Inputs
- `simulation.yaml`, `operator/*.yaml`, `facts/*`, `variables/*.csv`, curated CSV’s under `operator/`.
- CSV > YAML precedence with JSONL WARNING on shadowing; escalates to ERROR when `run.fail_on_warning: true`.
- Strict validator: exact headers for `variables/*.csv`, curated GPU/models constraints, allowed path roots per scope, numeric NaN/inf guards.

## Determinisme & RNG
- Seed-precedentie: `stochastic.random_seed` → `run.random_seed` → `operator.<tap>.meta.seed`.
- RNG: NumPy PCG64 substreams via stabiele hashing namespaces.
- Variabelen: grid (`low_to_high`) × replicates (`random_runs_per_simulation`) × MC (`simulations_per_run`). Transcript: `variable_draws.csv`.

## Artefacten
- Public: `public_vendor_choice.csv`, `public_tap_prices_per_model.csv`, `public_tap_scenarios.csv`, `public_tap_customers_by_month.csv`, `public_tap_capacity_plan.csv`, `public_tap_scaling_events.csv`.
- Private: `private_tap_economics.csv`, `private_vendor_recommendation.csv`, `private_tap_customers_by_month.csv`.
- Consolidatie: `consolidated_kpis.csv`, `consolidated_summary.{json,md}`, `run_summary.{json,md}`, `SHA256SUMS`.

## Proof bundle
- `SHA256SUMS` bevat checksums van artefacten voor golden/determinism tests.
- Identieke inputs + seeds → byte-gelijke artefacten; parallelisme levert deterministisch dezelfde outputs op (merge is gesorteerd per tabel-specificatie).

## Performance & Concurrency
- Streaming CSV writes om geheugen stabiel te houden.
- `--max-concurrency` paralleliseert jobs; deterministische merge zorgt voor stabiele outputvolgorde.
- Zie `tests/test_performance.py` voor minimale E2E-latency en geheugenlimieten (home-profile waarden kunnen aangepast worden).

## Development
- Makefile: `venv`, `install`, `run`, `test`, `fmt`, `lint`, `audit`.
- Lint/format config in `pyproject.toml`.
