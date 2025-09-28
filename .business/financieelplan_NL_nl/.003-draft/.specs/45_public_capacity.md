# 45 — Public Capacity & Autoscaling

Status: Draft
Version: 0.1.0

## 1. Scope

- Testtactiek voor Public Tap capaciteit en autoscaling‐plan per model.
- Dekt: tokens→tps, cap per instance, target utilization, peak factor, instances_needed, violations.

## 2. Eisen (MUST)

- **Autoscaling policy** (uit `operator/public_tap.yaml → autoscaling`):
  - `target_utilization_pct ∈ [1..100]`, `peak_factor ≥ 1.0`.
  - `0 ≤ min_instances_per_model ≤ max_instances_per_model`.
- **Capaciteit**:
  - `cap_tokens_per_hour_per_instance = tokens_per_hour(m,g*) * target_utilization_pct/100` (indien TPS dataset, anders heuristiek met WARNING).
  - `instances_needed = ceil(peak_tokens_per_hour / cap_tokens_per_hour_per_instance)`.
  - `capacity_violation = instances_needed > max_instances_per_model`.

## 3. Tests

- **Unit**: `services/autoscaling.py` randgevallen (tps=0, piek=0, util=100%, min>max → ERROR).
- **Golden**: fixture met bekend `tokens_per_hour` en policy → exact `public_tap_capacity_plan.csv`.
- **Acceptance**: geen `capacity_violation` of, indien aanwezig, beleid volgt (`fail_on_warning`).

## 4. Artefacten

- `public_tap_capacity_plan.csv` — kolommen:
  - `model,gpu,avg_tokens_per_hour,peak_tokens_per_hour,tps,cap_tokens_per_hour_per_instance,instances_needed,target_utilization_pct,capacity_violation`

## 5. Referenties

- `23_sim_public_tap.md`, `21_engine_flow.md`, `40_testing.md`, `31_engine.md`.
