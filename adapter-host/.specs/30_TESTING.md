# adapter-host — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Registry rebinding and facade behavior: submit/cancel routing, retries.

## Test Catalog

- Unit
  - Registry operations and rebinding logic
  - Facade request shaping and validation
  - Error mapping boundaries to orchestratord

- Integration
  - Submit/cancel routing through facade to a stub adapter
  - Retry and breaker interactions with stub failures
  - Deterministic micro-batch signaling (if applicable)

## Execution & Tooling

- `cargo test -p adapter-host -- --nocapture`
- Keep tests hermetic; use stubs for adapters and deterministic clocks for breaker/backoff

## Traceability

- Aligns with `orchestratord` integration tests and `worker-adapters` contracts

## Refinement Opportunities

- Fault injection for breaker thresholds.
