# pool-managerd — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Registry behavior, preload/readiness gating, supervision/backoff.

## Test Catalog

- Registry Semantics
  - Add/remove/update replicas; idempotency on duplicate registrations.
  - Lease accounting: acquire/release changes reflected in capacity snapshots.
  - Error recording and health marks persist across restarts when configured to recover state.

- Preload/Readiness Pipeline
  - Model ensure → engine ensure → health → ready transitions with explicit states.
  - GPU-required fail-fast: when no GPU or driver mismatch detected, transitions to `Failed(GpuRequired)` with clear diagnostics.
  - Concurrency limits: preload runs within configured parallelism bounds.

- Supervision & Backoff
  - Crash loops trigger exponential (or configured) backoff; jitter (if present) stays within bounds.
  - Restart storm is bounded; circuit breaker opens after threshold.

- Capability Reporting
  - Report accurate slots/ctx/token bounds into orchestrator-core snapshot model.

## Execution & Tooling

- Run: `cargo test -p pool-managerd -- --nocapture`
- Prefer deterministic clocks in tests; inject a fake timer for backoff assertions.
- Keep integration to local mocks; do not spawn real containers/engines in unit/integration scope.

## Traceability

- GPU-only policy: tests assert fail-fast when GPU is unavailable or misconfigured.
- Preload sequencing aligns with `pool-managerd/.specs/10_contracts.md` and provisioner specs.

## Refinement Opportunities

- Fault injection for driver resets and backoff tuning verification.
