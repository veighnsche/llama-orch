# Investigation — Metrics Contract Alignment

Status: done · Date: 2025-09-15

## Changes

- Extended `.specs/metrics/otel-prom.md` with v3.2 metrics.
- Updated `ci/metrics.lint.json` required metrics and budgets.

## Proofs

- `cargo test -p test-harness-metrics-contract -- --nocapture`
- `bash ci/scripts/spec_lint.sh`
