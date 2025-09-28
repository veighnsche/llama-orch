# 14 — Simulation Parameters (No Defaults, Fast‑Fail)

Status: Draft
Version: 0.1.0

Doel: Alle parameters die de simulatie sturen expliciet definiëren, zonder impliciete defaults. Ontbrekende of ongeldige parameters leiden tot een onmiddellijke **ERROR** (fast‑fail). Dit document is leidend voor `core/validator.py` en voor de UI‑validaties.

## 1. Locaties & Overlays (MUST)

- `inputs/simulation.yaml` → `SimulationPlan` (run/stochastic/targets/stress/consolidation/ui/logging)
- `inputs/operator/` → `OperatorBundle` (`general.yaml`, `public_tap.yaml`, `private_tap.yaml`, curated CSV’s)
- `inputs/variables/*.csv` → `VariableRow[]` per scope (`general|public_tap|private_tap`)
- `inputs/facts/*` → exogeen (read‑only)
- Overlay/precedence: **CSV > YAML** bij conflicten; validator MUST loggen met pad.

## 2. Minimale verplichte parameters (MUST, geen defaults)

### 2.1 Simulation plan (`inputs/simulation.yaml`)

- `run.pipelines` ∈ {`public`,`private`} (één of beide) — lijst
- `run.random_seed` — integer > 0
- `run.output_dir` — bestaand of aan te maken pad
- `run.random_runs_per_simulation` — integer ≥ 1
- `stochastic.simulations_per_run` — integer ≥ 1
- `stochastic.percentiles` — lijst integers ⊆ [1..99], niet leeg
- `targets.horizon_months` — integer ≥ 1
- `targets.private_margin_threshold_pct` — float ≥ 0
- `targets.require_monotonic_growth_public_active_customers` — bool
- `targets.require_monotonic_growth_private_active_customers` — bool
- `targets.autoscaling_util_tolerance_pct` — float ≥ 0

### 2.2 Operator — Public Tap (`inputs/operator/public_tap.yaml`)

- `pricing_policy.target_margin_pct` — float ∈ [0..95]
- `autoscaling.target_utilization_pct` — float ∈ [1..100]
- `autoscaling.peak_factor` — float ≥ 1.0
- `autoscaling.min_instances_per_model` — int ≥ 0
- `autoscaling.max_instances_per_model` — int ≥ 1, en ≥ min
- Simulator policy (alle vereist):
  - `evaluation_interval_s` — int > 0
  - `scale_up_threshold_pct` — 0 < down < up ≤ 100 (down/ up consistent)
  - `scale_down_threshold_pct` — idem
  - `scale_up_step_replicas` — int ≥ 1
  - `scale_down_step_replicas` — int ≥ 1
  - `stabilization_window_s` — int ≥ 0
  - `warmup_s` — int ≥ 0
  - `cooldown_s` — int ≥ 0

### 2.3 Operator — Private Tap (`inputs/operator/private_tap.yaml`)

- `pricing_policy.default_markup_over_provider_cost_pct` — float ≥ 0
- (indien aanwezig) fees >= 0

### 2.4 Variables CSV (`inputs/variables/*.csv`)

- Elke rij: `variable_id,scope,path,type,unit,min,max,step,default,treatment`
- `type ∈ {numeric,discrete}`; bij `numeric`: `min,max,step,default` aanwezig en `min ≤ default ≤ max`, `step > 0`
- `treatment ∈ {fixed,low_to_high,random}`
- Paths MUST in allowed roots per scope vallen (zie `12_oprator_variables.md`)

### 2.5 Curated CSV’s (operator)

- `curated_gpu.csv`: exact `gpu,vram_gb,provider,usd_hr`; elke `usd_hr > 0`
- `curated_public_tap_models.csv`: minimaal `model` kolom (niet leeg)
- TPS dataset (aanbevolen). Indien niet aanwezig: **ERROR** of expliciet `allow_tps_heuristic: true` in `public_tap.yaml` met **WARNING** pad.

### 2.6 Facts (`inputs/facts/market_env.yaml`)

- `finance.eur_usd_fx_rate.value` — float > 0

## 3. Derived parameters & constraints (MUST)

- `cap_tokens_per_hour_per_instance = tps(model,gpu) * 3600 * target_utilization_pct/100`
- `instances_needed = ceil(peak_tokens_per_hour / cap_per_instance)`
- `peak_tokens_per_hour` komt uit uurprofiel (uniform of diurnaal) afgeleid uit maandtokens; som (uur) == maandtokens.
- `public pricing`: floors/caps/round_increment consistent (indien gezet), marge ≥ target bij basisaanname.
- Allocaties/gewichten moeten naar 1.0 sommeren per scope; renormalisatie is **ERROR** tenzij `run.fail_on_warning=false` (policy afhankelijk).

## 4. Fast‑Fail beleid (MUST)

- Ontbrekende verplichte sleutels → **ERROR** (geen defaults invullen in de engine).
- Ongeldige ranges/waarden (negatief, NaN/inf, step ≤ 0, inconsistent thresholds) → **ERROR**.
- Onbekende CSV‑headers/paths buiten allowed roots → **ERROR**.
- TPS ontbreekt zonder `allow_tps_heuristic=true` → **ERROR**.

## 5. Validator implementatie‑note

- `core/validator.py` MUST exact deze lijst afdwingen en bij violation:
  - exit code `2=VALIDATION_ERROR`
  - duidelijke boodschap met pad (`file:key.path`) en remedie
  - JSONL event `validate_error` met details

## 6. UI gedrag (MUST)

- UI blokkeert ‘Run’ bij **ERROR** en toont welke keys ontbreken/ongeldig zijn.
- UI biedt geen “auto‑fill defaults”; enkel expliciet invullen of CSV’s leveren.

## 7. Relatie met bestaande specs

- `21_engine_flow.md` — runner‑stappen en determinisme
- `16_simulation_variables.md` — treatments, grid/replicates/MC
- `45_public_capacity.md` — planner en capaciteitsschema
- `25_autoscaling.md` — simulator policy en events
- `44_public_pricing.md`, `46_private_economics.md` — prijs/marge
- `40_testing.md` — acceptance/golden/determinism
