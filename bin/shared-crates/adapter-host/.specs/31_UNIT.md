# adapter-host — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Registry operations, facade happy paths, error mapping.

## Test Catalog

- Registry Operations
  - `bind(name, adapter)` registers; `rebind` swaps atomically; `unbind` idempotent
  - Lookup by name returns expected adapter; unknown yields typed error

- Facade Behavior (happy paths)
  - `submit` routes to correct adapter with normalized request
  - `cancel` routes and returns success regardless of in-flight status (idempotent)

- Error Mapping Boundaries
  - Adapter errors mapped to orchestratord domain consistently (no detail leaks)
  - Correlation id propagation asserted in logs/structures (unit-level, no HTTP)

## Structure & Conventions

- Pure logic; adapters replaced with minimal fakes/mocks
- Table-driven tests for registry matrices and mapping cases

## Execution

- `cargo test -p adapter-host -- --nocapture`

## Traceability

- Aligns with `worker-adapters` trait contracts and `orchestratord` handler expectations

## Refinement Opportunities

- Add stress tests for concurrent rebinding.
