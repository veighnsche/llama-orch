# Metrics Contract — Implementation Plan (OC-METRICS-7xxx)

Specs: `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md`
Scope: metric names/labels/units; label budgets; dashboards; linters.

## Stages and Deliverables

- Stage 5 — Observability
  - Implement metrics emission at code sites (queues, admission, decode, cancellations, errors, GPU/VRAM/KV, lifecycle, scheduling).
  - Ensure labels include `engine` and engine-specific versions; admission counters may omit `engine_version` per contract.
  - Dashboards and alerts under `ci/dashboards/`.

## Tests

- Linter expectations per `ci/metrics.lint.json`.
- Metrics contract tests under `test-harness/metrics-contract/`.
- BDD observability features: `test-harness/bdd/tests/features/observability/`.

## Acceptance Criteria

- OC-METRICS IDs mapped; linter green; dashboard render checks pass.

## Backlog (initial)

- Implement emission hooks; budget checks; dashboards for queue depth, latency, errors, fairness, deadlines, preemption.

## Proposal (Accepted)

- Align with product stages: Stage 5 observability implemented now (admission counters/gauges + /metrics); Stage 13 dashboards & alerts to be delivered; Stage 14 startup self‑tests cover telemetry emission.
- DX: keep metric registration/encoders within `orchestratord/src/metrics.rs` (or future `orch-services`), with names/labels centralized; enforce linter parity with `.specs/metrics/otel-prom.md` and `ci/metrics.lint.json`. Dashboards live under `ci/dashboards/` and render in CI with sample data.
