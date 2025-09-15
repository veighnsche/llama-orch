# Metrics Contract SPEC — Names/Labels Budgets (v1.0)

Status: Stable (draft)
Applies to: `test-harness/metrics-contract/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-METRICS-7xxx`.

## 1) Names/Labels

- [OC-METRICS-7101] Metric names and required labels MUST conform to `ci/metrics.lint.json`.
- [OC-METRICS-7102] Label cardinality budgets MUST be documented and enforced.

## 2) Traceability

- Linter: [ci/metrics.lint.json](../ci/metrics.lint.json)
- Tests: [test-harness/metrics-contract/tests/metrics_lint.rs](../test-harness/metrics-contract/tests/metrics_lint.rs)
