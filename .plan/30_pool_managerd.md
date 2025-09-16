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

- Stage 7 — Product: Pool manager readiness (aligns with README_LLM Stage 7)
  - Replica registry with heartbeat/health/readiness and version labels (`engine_version`, `model_digest`).
  - Drain/reload flows and leases integrated with orchestrator control-plane.
  - Device masks honored; heterogeneous split planner plumbs per‑GPU ratios; no cross‑mask spill.
  - Readiness gating: only Ready replicas advertised to orchestrator; pinning enforced.

- Stage 16 — Chaos (nightly)
  - Reset/driver errors cause Unready + drain + backoff‑restart; storms bounded via circuit breaker.

## Tests

- Integration tests under `pool-managerd/tests/`.
- BDD lifecycle and placement features: `test-harness/bdd/tests/features/lifecycle/`, `pool_manager/`.
- Health/readiness unit tests and integration tests for drain/reload and backoff/retry.
- Metrics contract assertions for preload outcomes and restart counters.

## Acceptance Criteria

- OC-POOL IDs mapped to tests; health reflects true state; no cross-mask spillover; heterogeneous ratios enforced and capped.
- Registry lists only Ready replicas; version labels present; orchestrator control-plane reflects state.
- Drain/reload without request loss; leases prevent double scheduling; backoff policy bounds restart storms.

## Backlog (initial)

- Preloader with memory headroom checks.
- Backoff policy and circuit breaker.
- Device mask enforcement; heterogeneous split planner.
- Health/readiness surfaces with last error snapshot.

---

## DX Modularization Proposal (Pool Manager)

Goal: isolate lifecycle concerns and make it easy to test readiness/backoff logic without pulling HTTP or orchestrator concerns.

Future layering (planning only):

- `pool-domain` (lib)
  - Types: `Replica`, `ReplicaState`, `Health`, `Readiness`, `Lease`, `DeviceMask`.
  - Errors and backoff/circuit breaker primitives.

- `pool-services` (lib)
  - Registry (heartbeats, leases), preload/health probes, drain/reload orchestrations, backoff policies.
  - Depends on: `pool-domain`; may import adapter processes via `adapter-api` helpers.

- `pool-api` (lib)
  - HTTP endpoints (health/readiness, drain/reload), JSON mapping.
  - Depends on: `pool-services`, `pool-domain`.

- `pool-managerd` (bin)
  - Thin binary: config/env, tracing, router wiring.

Dependency rules:

- `pool-api` → `pool-services` → `pool-domain`; orchestrator never imports `pool-services` directly.
- Orchestrator queries `GET /v1/replicasets` rather than linking to pool libs to prevent tight coupling.

Rollout:

1) Stage 7.1: enforce in-crate modules mirroring above; `pub(crate)` by default.
2) Extract `pool-domain` first if compile times grow; services next when endpoints stabilize.
3) Keep integration tests in `pool-managerd/tests` to minimize rebuild churn.
