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

---

## DX Modularization Proposal (Orchestrator Core)

Intent: keep `orchestrator-core/` lean and stable with clear boundaries; avoid HTTP/adapter coupling; make it easy to unit test and evolve scheduling independently.

Internal modules now:

- `queue` — bounded FIFO by priority; policies `reject|drop-lru`.
- `scheduling` — placement hooks, load metrics, device masks, session affinity.
- `fairness` — priority fairness accounting; exposes `admission_share` gauges.
- `determinism` — seed policy and replica set pinning enforcement.

Dependency rules:

- No dependency on HTTP frameworks, adapters, or orchestrator binaries.
- Only depend on `serde`, `thiserror`, and workspace commons.
- Surfaces return `Result<CoreType, CoreError>`; caller maps to envelopes.

Future split triggers (if needed):

- Extract `core-queue` if experimental scheduling evolves rapidly and rebuild time becomes high.
- Extract `core-scheduler` when multiple policies (WFQ/EDF/preemption) are in flight and need independent cadence.
- Keep `core-fairness` with scheduler unless it grows large.

Rollout (after Stage 9 stabilizes):

1) Enforce module boundaries and public surface (`pub(crate)` by default).
2) Add micro-benchmarks for queue ops and scheduler decisions.
3) If compile times or coupling become problematic, extract `core-queue` first (lowest risk).

## Proposal (Accepted)

- Align with product stages: Stage 6 (placement hooks consumption by orchestrator), Stage 9 (scheduling & fairness gauges) — core provides stable APIs and invariants proven by properties.
- Adopt DX modularization intent: keep core free of HTTP/adapter deps; expose small trait-driven surfaces; plan future splits when scheduling complexity warrants.
