# pool-managerd — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Registry getters/setters, lease accounting, error recording.

## Test Catalog

- Registry Operations
  - `register(replica)` idempotent on same id; `deregister(id)` removes and is idempotent.
  - `update(replica)` merges health/capacity fields without resetting immutable identity.

- Lease Accounting
  - Acquire/release within bounds; overflow/underflow guarded with typed errors.
  - Snapshot reflects live lease counts; releasing last lease toggles idle indicators.

- Error/Health Recording
  - Record last failure, failure counts; reset on successful health checks.

- Deterministic Ordering
  - Iteration over replicas (if any ordering) is explicitly sorted by key; no map nondeterminism.

## Structure & Conventions

- Keep tests pure (no I/O); use minimal fake replicas.
- Table-driven tests for lease matrix (concurrency, attempts, expected errors).

## Execution

- `cargo test -p pool-managerd -- --nocapture`

## Traceability

- Aligns with `pool-managerd/.specs/10_contracts.md` registry semantics.

## Refinement Opportunities

- Additional guardrails for negative/overflow scenarios.
