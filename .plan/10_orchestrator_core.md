# Orchestrator Core — Implementation Plan (OC-CORE-1xxx)

Spec: `.specs/10-orchestrator-core.md`
Scope: queueing, scheduling/placement, guardrails, determinism, observability.

## Stages and Deliverables

- Stage 0 — Contracts
  - Validate any schema types the core depends on (priority classes, quotas) exist in `contracts/config-schema/`.

- Stage 3 — Properties & Invariants
  - Property tests: `orchestrator-core/tests/props_queue.rs` (FIFO, fairness, reject/drop policies, cancel races).
  - Implement queue abstraction: bounded FIFO per priority class.
  - Implement admission policies: `reject | drop-lru | shed-low-priority`.
  - Implement scheduler: least-loaded Ready; respect device masks; basic session affinity.

- Stage 4 — Determinism enforcement
  - Honor fixed `seed` and `sampler_profile_version` from config/context.
  - Enforce replica set pinning `engine_version`, `sampler_profile_version`.

- Stage 5 — Observability
  - Emit queue depth, reject/drop counters, latency percentiles; include engine/version labels.

- Stage 7 — Chaos & Load (nightly)
  - Verify bounded backoff and no starvation under load.

## Tests

- Property tests: `orchestrator-core/tests/props_queue.rs` (OC-CORE-1001..1013, 1020..1022).
- BDD: `test-harness/bdd/tests/features/orchestrator_core/` and `scheduling/`.
- Determinism suite integration checks via `test-harness/determinism-suite/`.

## Acceptance Criteria

- OC-CORE IDs mapped to tests; green properties and unit tests.
- Determinism constraints enforced; no cross-set mixing.
- Metrics present per `.specs/metrics/otel-prom.md`.

## Backlog (initial)

- Implement `Queue<T>` with capacity and eviction strategy.
- Scheduler: load metric, Ready gating, device mask enforcement.
- Watchdog: wall/idle timeouts.
- Log fields coverage (ids, timings).
