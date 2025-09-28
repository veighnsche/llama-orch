# 13 — Simulation Defaults (Smart Baselines)

Status: Draft
Version: 0.1.0

## 1. Doel & Scope

- Biedt “slimme” realistische baselines waarmee een simulatie kan draaien als niet alle parameters aangeleverd zijn.
- Defaults zijn **opt‑in** en **traceerbaar**; zonder opt‑in blijft de validator fast‑fail (zie `14_simulation_parameters.md`).
- Defaults zijn **deterministisch**, **versieerbaar** en **overrule‑baar** via operator/variables.

## 2. Activering & Precedentie (MUST)

- Activeren via `inputs/simulation.yaml`:

```yaml
run:
  use_defaults: true            # expliciete opt-in
  defaults_profile: baseline_v1 # verplicht als use_defaults=true
```

- Precedentie (hoog → laag):
  1) `inputs/variables/*.csv`
  2) `inputs/operator/*.yaml`
  3) Defaults (deze pagina)
- Zonder `use_defaults: true` blijft het fast‑fail beleid van kracht.

## 3. Versies & herkomst (MUST)

- Profielen: `baseline_v1` (deze), toekomstige versies `baseline_v2`, …
- Elk profiel heeft een vaste defaults‑catalogus met bronverwijzingen (indien van toepassing).
- `run_summary.json` MUST loggen:
  - `defaults: { enabled, profile, version, spec_sha256, applied: [ { path, value, source } ] }`

## 4. Defaults‑catalogus (baseline_v1)

Let op: waarden zijn realistisch bedoeld maar conservatief; pas aan via operator/variables voor specifieke cases.

### 4.1 Public Tap — vraag/funnel

- `public_tap.budget_month0_eur`: 5_000
- `public_tap.budget_growth_pct_mom`: 5    # % per maand
- `public_tap.cac_base_eur`: 200
- `public_tap.cvr_base`: 0.03              # 3%
- `public_tap.tokens_per_conversion_mean`: 50_000
- `public_tap.churn_rate_mom`: 0.03        # 3% per maand
- Uurprofiel:
  - `public_tap.demand_profile`: diurnal
  - `public_tap.hours_in_month`: 720
  - `public_tap.peak_factor_demand`: 1.5

### 4.2 Public Tap — pricing/policy

- `public_tap.pricing_policy.target_margin_pct`: 40
- `public_tap.pricing_policy.min_floor_eur_per_1k`: null
- `public_tap.pricing_policy.max_cap_eur_per_1k`: null
- `public_tap.pricing_policy.round_increment_eur_per_1k`: 0.10

### 4.3 Autoscaling (planner + simulator)

- `autoscaling.target_utilization_pct`: 75
- `autoscaling.peak_factor`: 1.5
- `autoscaling.min_instances_per_model`: 0
- `autoscaling.max_instances_per_model`: 100
- Simulator policy:
  - `evaluation_interval_s`: 60
  - `scale_up_threshold_pct`: 70
  - `scale_down_threshold_pct`: 50
  - `scale_up_step_replicas`: 1
  - `scale_down_step_replicas`: 1
  - `stabilization_window_s`: 300
  - `warmup_s`: 120
  - `cooldown_s`: 120

### 4.4 TPS & kosten (indien geen dataset aanwezig)

- `allow_tps_heuristic`: true (alleen als defaults enabled)
- Heuristiek (spec 44/45) met WARNING:
  - `tps(model,gpu) ≈ k * (vram_gb / 24)` met vaste `k` per klasse (conservatief); documenteer in `run_summary.heuristics`.

### 4.5 Private Tap — economics

- `private_tap.pricing_policy.default_markup_over_provider_cost_pct`: 30
- `private_tap.hours_per_client_mean`: 4
- `private_tap.hours_per_client_sd`: 1
- `private_tap.churn_rate_mom`: 0.02

### 4.6 Facts (FX)

- `facts.market_env.finance.eur_usd_fx_rate.value`: 1.10
  - Bron: recente gemiddelde EUR/USD; alleen gebruiken als facts ontbreken en defaults enabled

### 4.7 Targets & acceptatie

- `targets.autoscaling_util_tolerance_pct`: 25
- `targets.horizon_months`: 18
- `targets.private_margin_threshold_pct`: 20
- `targets.require_monotonic_growth_public_active_customers`: true
- `targets.require_monotonic_growth_private_active_customers`: true

## 5. Toepassing (MUST)

- Als `use_defaults=true` en een noodzakelijke key ontbreekt:
  - Injecteer default in memory‑state (niet in files) vóór simulatie.
  - Log elke injectie in `run_summary.defaults.applied`.
  - Markeer `run_summary.defaults.enabled=true` en `profile=baseline_v1`.
- Defaults mogen NOOIT randvoorwaarden verzwakken (bv. negative waarden toelaten) en zijn altijd deterministisch.

## 6. Tests (MUST)

- **BDD — defaults_enabled**: afwezige parameters → run slaagt, `defaults.applied` gelogd, artefacten aanwezig.
- **Golden**: default‑profiel produceert deterministische outputs (`SHA256SUMS`).
- **Heuristieken**: WARNING + transcript in `run_summary.heuristics` wanneer TPS‑heuristiek gebruikt is.

## 7. UI (SHOULD)

- Schakelaar “Use smart defaults” + keuze `defaults_profile`.
- Paneel “Applied defaults” met diff en bronnen.

## 8. Governance

- Elke wijziging in defaults vereist bumps van `defaults_profile` versie en aanpassing van golden tests.
- Documenteer herkomst en motivatie per default (bijv. marktomstandigheden, interne benchmarks).

## 9. Relatie met specs

- `14_simulation_parameters.md` — fast‑fail zonder opt‑in
- `21_engine_flow.md` — runner flow
- `44_public_pricing.md` / `45_public_capacity.md` / `46_private_economics.md`
- `40_testing.md` / `41_engine.md` — verificatie
