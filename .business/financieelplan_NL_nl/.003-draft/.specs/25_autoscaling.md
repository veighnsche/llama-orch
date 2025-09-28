# 25 — Autoscaling (Planner & Simulator)

Status: Draft
Version: 0.1.0

## 1. Scope

- Planner: berekent deterministisch de benodigde instanties uit piekvraag.
- Simulator: benadert real‑world autoscaler met hysterese, stabilisatiewindows, warm‑up/cooldown en thresholds.
- Toepassing: Public Tap capaciteit per model/gpu (Private gebruikt geen autoscaling in v0.1).

## 2. Inputs

- Uit operator (`inputs/operator/public_tap.yaml → autoscaling`):
  - `target_utilization_pct ∈ [1..100]`
  - `peak_factor ≥ 1.0`
  - `min_instances_per_model ≥ 0`, `max_instances_per_model ≥ 1`, `min ≤ max`
  - Simulator policy (SHOULD):
    - `evaluation_interval_s` (bijv. 60)
    - `scale_up_threshold_pct` (bijv. 70)
    - `scale_down_threshold_pct` (bijv. 50)
    - `scale_up_step_replicas` (≥1), `scale_down_step_replicas` (≥1)
    - `stabilization_window_s` (bijv. 300)
    - `warmup_s` (bijv. 120)
    - `cooldown_s` (bijv. 120)
    - Optioneel: `capacity_peak_percentile` (planner: p95 ipv pure max)
- Uit pipelines/behavior:
  - `tokens_per_hour(m,g*)` of TPS dataset (anders heuristiek met WARNING)
  - Vraagprofiel: maandvraag verdeeld over uren (uniform of diurnaal), levert `demand_tokens_per_hour[t]`.

## 3. Planner (MUST)

- `cap_tokens_per_hour_per_instance = tps_eff(m,g*) * 3600 * target_utilization_pct/100`.
- `instances_needed = ceil(peak_tokens_per_hour / cap_tokens_per_hour_per_instance)`.
- Clamp: `min_instances ≤ instances_needed ≤ max_instances`.
- `capacity_violation = instances_needed > max_instances`.
- Artefact: `public_tap_capacity_plan.csv` met kolommen:
  - `model,gpu,avg_tokens_per_hour,peak_tokens_per_hour,tps,cap_tokens_per_hour_per_instance,instances_needed,target_utilization_pct,capacity_violation`.

## 4. Simulator (SHOULD)

- Discrete tijd (stap `evaluation_interval_s`), deterministisch:
  - `observed_util = demand / (replicas * tokens_per_hour)`.
  - EWMA/EMA over `stabilization_window_s` (optioneel) → `util_ema`.
  - Actie‐regels met hysterese:
    - Als `util_ema > scale_up_threshold`: `replicas += scale_up_step_replicas`.
    - Als `util_ema < scale_down_threshold`: `replicas -= scale_down_step_replicas` (≥ min).
  - Respecteer `cooldown_s` na actie; nieuwe replicas tellen pas mee na `warmup_s`.
  - Clamp door `min/max_instances_per_model`.
  - SLA‐violation als `demand > effective_capacity` (warm‑up telt niet).
- Artefact (optioneel): `public_tap_scaling_events.csv` met kolommen:
  - `timestamp,model,gpu,demand_tokens_per_hour,effective_capacity,replicas_prev,replicas_new,reason,util_pct`.

## 5. Acceptatie & KPI’s (MUST)

- Planner‐kolom: geen `capacity_violation` voor baselines; policy bepaalt FAIL/WARN.
- Simulator: `p95(util)` rond `target_utilization_pct` binnen redelijke marge (configurabel).
- Geen `sla_violations` in baseline met voldoende `max_instances_per_model`.
- KPI’s (analysis): `scale_events`, `avg_util`, `p95_util`, `violations` in `consolidated_kpis.csv` en `consolidated_summary`.

## 6. Tests

- **Unit**: `services/autoscaling.py` randgevallen (`tps=0`, util 100%, min>max → ERROR, hysterese tegen flapping).
- **BDD**: `capacity_autoscaler.feature` — planner CSV aanwezig; (optioneel) events CSV; acceptatie‐regels.
- **Determinism**: identieke inputs + seeds + timeseries → identieke events en capaciteitsplan.

## 7. Implementatie‑mapping

- `services/autoscaling.py`: `planner_instances_needed(...)`, `simulate_autoscaler(...)`, helpers.
- `pipelines/public/capacity.py`: planner aanroepen en tabellen vullen; simulator events (optioneel) schrijven.
- `models/inputs.py → PublicAutoscaling`: simulator policy velden en bounds‑checks.
- `analysis/kpis.py`: autoscaling KPI’s.

## 8. Referenties

- `45_public_capacity.md`, `21_engine_flow.md`, `31_engine.md`, `40_testing.md`.
