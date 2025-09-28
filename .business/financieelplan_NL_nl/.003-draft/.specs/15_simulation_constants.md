# 15 — Simulation Plan (Constants)

Status: Draft
Version: 0.1.0

## 1. Scope

- Documenteert `inputs/simulation.yaml`: run‑plan, stochastiek, stress, consolidatie, UI en logging.
- Deze waarden zijn **constanten per run**; variatie komt uit `inputs/variables/*.csv`.

## 2. YAML — `inputs/simulation.yaml` (MUST)

```yaml
run:
  random_seed: 424242                  # optional: seed for entire run
  pipelines: [public, private]         # allowed: public | private | both (list)
  output_dir: ".003-draft/outputs"     # artifacts root
  fail_on_warning: false               # promote WARNINGs to errors (optional)
  max_concurrency: null                # optional: threads/processes
  random_runs_per_simulation: 5        # #random draws per counted simulation

stochastic:
  simulations_per_run: 1000            # Monte Carlo iterations (MUST >= 1)
  percentiles: [10, 50, 90]            # cuts (values 0..100, ascending)
  random_seed: null                    # optional override for entire run

stress:
  provider_price_drift_pct: 10.0       # p90 stress drift for public margins (>=0)
  tps_downshift_pct: 0.0               # >=0 (if used)
  fx_widen_buffer_pct: 0.0             # >=0

consolidation:
  overhead_allocation_driver: revenue  # enum: revenue | gpu_hours | tokens
  include_loan_in_cashflow: true

ui:
  show_credit_packs: [5, 10, 20, 50, 100, 200, 500]
  halt_at_zero_simulation: true

logging:
  level: INFO                          # DEBUG | INFO | WARN | ERROR
  write_run_summary: true
 
targets:
  horizon_months: 18                   # feasibility horizon (MUST >= 1)
  private_margin_threshold_pct: 20     # minimum acceptable private margin (0..100)
  require_monotonic_growth_public_active_customers: true
  require_monotonic_growth_private_active_customers: true
  public_growth_min_mom_pct: 0.0       # optional minimum MoM growth requirement

## 3. Regels & Constraints (MUST)

- `run.pipelines` MUST een subset zijn van `{public, private}`; lege set → **ERROR**.
- `stochastic.simulations_per_run` MUST ≥ 1.
- `stochastic.percentiles` MUST waarden 0..100 bevatten, strikt stijgend.
- `stress.*_pct` MUST ≥ 0.
- `consolidation.overhead_allocation_driver` MUST een toegestane waarde hebben.
 - `targets.horizon_months` MUST ≥ 1; `targets.private_margin_threshold_pct` MUST in 0..100; `targets.public_growth_min_mom_pct` MUST ≥ 0.

## 4. Seed‑resolutie en determinisme (MUST)

1) `stochastic.random_seed` (indien gezet)
2) `run.random_seed` (indien gezet)
3) Pijplijnseed uit `inputs/operator/public_tap.yaml: meta.seed` of `private_tap.yaml: meta.seed`
4) Geen seed → **ERROR**

- Identieke inputs + seed geven byte‑gelijke outputs (deterministisch). 
- `run_summary.{json,md}` MUST loggen: seed(s), input‑hashes, overlay‑beslissingen, shadowing‑warnings.

## 5. Interactie met Variabelen (SHOULD)

- `run.random_runs_per_simulation` bepaalt hoeveel onafhankelijke `random` samples (per variabele) per simulatie worden getrokken (zie `16_simulation_variables.md`).
- `fail_on_warning: true` MAY escaleren van shadowing/override naar **ERROR**.

## 6. Voorbeeld (minimaal)

```yaml
run: { pipelines: [public], output_dir: ".003-draft/outputs", random_seed: 7 }
stochastic: { simulations_per_run: 500, percentiles: [5,50,95] }
consolidation: { overhead_allocation_driver: revenue, include_loan_in_cashflow: true }
logging: { level: INFO, write_run_summary: true }
```

