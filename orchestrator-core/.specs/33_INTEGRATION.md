# orchestrator-core — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Minimal placement composition over mock PoolSnapshot inputs.

## Delegation

- HTTP/API integration is in root BDD; not owned here.

## Test Catalog

- Placement Feasibility
  - Single-slot pool: enqueue N tasks across priorities; verify predicted start order and monotonicity.
  - Multi-slot pools: verify parallelism and fair ordering within priority cohorts.
  - No-capacity cases: return typed `NoCapacity` with structured `IncompatibleReason` (when available).

- Budget Guardrails (Core Boundary)
  - Over-budget `ctx`/`max_tokens` produce typed errors (no silent truncation). Mapping to HTTP delegated to `orchestratord`.

- Snapshot/Readout Consistency
  - Queue position and `predicted_start_ms` remain consistent when queried repeatedly without state changes.

## Mocks & Fixtures

- Use lightweight fake `PoolSnapshot` builders with:
  - Slots (1..=N), per-slot service time hints, compatibility flags.
  - Deterministic clock abstraction to avoid wall-clock dependency.

## Execution

- `cargo test -p orchestrator-core --test integration -- --nocapture` (if a dedicated integration target exists), or filter by name.
- Keep tests hermetic; do not perform I/O or spawn threads.

## Traceability

- ORCH-3016: No silent truncation on over-budget inputs.
- ORCH-3093/3094: Capability/guardrail alignment (pre-validation at client, enforcement at server; core handles invariants).

## Refinement Opportunities

- Add mocks for perf hints and confirm predicted_end_ms ordering.
