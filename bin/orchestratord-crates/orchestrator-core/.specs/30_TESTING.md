# orchestrator-core — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Umbrella for crate-local tests. This crate owns queue invariants, placement feasibility and tie-break determinism. Cross-crate flows are exercised by the root BDD harness.

## Owned Test Layers

- Unit: queue semantics, cancel, snapshot readouts.
- Property: invariants for FIFO-within-priority, Drop-LRU behavior.
- Integration: minimal placement flows using in-crate mocks.

## Delegated

- BDD: admission → stream/cancel via HTTP is validated by `test-harness/bdd`.

## Test Catalog

- Unit
  - Queue admission guardrails
    - Reject vs Drop-LRU decisions by priority and capacity budget.
    - Enforce FIFO-within-priority ordering; no cross-priority reordering.
  - Cancel semantics
    - Cancel on queued task removes it and compacts order; idempotent on duplicates.
    - Cancel on running task marks state and propagates a deterministic stop signal to integration boundary (mocked).
  - Capacity/position readouts
    - `queue_position` and predicted start time math for single-slot pools given synthetic service times.
    - Snapshot stability (read-only view does not mutate state).
  - Error shapes
    - Internal error types for queue/planning only (no HTTP envelopes here).

- Property
  - FIFO-within-priority invariant holds for arbitrary enqueue/cancel sequences.
  - Drop-LRU behavior: when capacity exceeded, the lowest-priority, oldest-in-priority task is dropped.
  - Stability of placement scoring: identical inputs produce identical sort order (no floating nondeterminism).

- Integration (in-crate mocks only)
  - Minimal placement feasibility over `PoolSnapshot` mocks (single-slot and multi-slot variants).
  - Predicted start time monotonicity across a sequence of enqueues with fixed service time assumptions.
  - No silent truncation of budgets; violations surface as typed errors at core boundary (HTTP mapping delegated).

- Determinism
  - Tie-breakers: define stable keys (e.g., `(priority, arrival_seq, task_id)`) and verify total ordering is reproducible.
  - Seeded RNG prohibition: core algorithms must not read wall-clock or random state; tests assert determinism under repetition.

## Execution & Tooling

- Commands
  - `cargo test -p orchestrator-core -- --nocapture`
  - Property-only focus (example): `cargo test -p orchestrator-core property -- --nocapture` (or use test name filters)
- Style
  - Pure logic; no I/O. Use lightweight mocks for placement inputs.
  - Prefer table-driven tests for edge cases and `proptest` for sequence invariants where beneficial.

## Traceability

- Align test names and comments to requirement IDs when applicable (e.g., ORCH-3016 “no silent truncation”).
- Update `.docs/testing/spec-derived-test-catalog.md` and `.docs/testing/spec-combination-matrix.md` when adding new coverage.

## Refinement Opportunities

- Add randomized stress/property tests for Drop-LRU.
- Provide golden cases for deterministic tie-breaks.
