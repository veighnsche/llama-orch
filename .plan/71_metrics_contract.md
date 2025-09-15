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
