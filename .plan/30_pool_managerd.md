# Pool Managerd — Implementation Plan (OC-POOL-3xxx)

Spec: `.specs/30-pool-managerd.md`
Scope: preload/readiness lifecycle, restart/backoff, device masks, heterogeneous splits.

## Stages and Deliverables

- Stage 0 — Contracts
  - Config fields for pools, device masks, heterogeneous split ratios.

- Stage 2 — Provider/Integration
  - Implement preload at `serve` and Ready gating; fail-fast on OOM; backoff+retry.
  - Health/readiness endpoints reflect preload state and last error cause.

- Stage 5 — Observability
  - Emit preload outcomes, VRAM/RAM, driver_reset, restart counters.

- Stage 7 — Chaos
  - Reset/driver errors cause Unready + drain + backoff-restart; storms bounded via circuit breaker.

## Tests

- Integration tests under `pool-managerd/tests/`.
- BDD lifecycle and placement features: `test-harness/bdd/tests/features/lifecycle/`, `pool_manager/`.

## Acceptance Criteria

- OC-POOL IDs mapped to tests; health reflects true state; no cross-mask spillover; heterogeneous ratios enforced and capped.

## Backlog (initial)

- Preloader with memory headroom checks.
- Backoff policy and circuit breaker.
- Device mask enforcement; heterogeneous split planner.
- Health/readiness surfaces with last error snapshot.
