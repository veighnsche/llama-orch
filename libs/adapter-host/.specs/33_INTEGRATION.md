# adapter-host — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Orchestratord integration via in-process facade; cancel propagation.

## Test Catalog

- Submit Path
  - GIVEN a stub adapter registered under name `llamacpp`
  - WHEN orchestratord calls facade `submit`
  - THEN request is forwarded with normalized fields; correlation id is preserved; success response returns adapter handle/id

- Cancel Path
  - GIVEN an in-flight handle
  - WHEN `cancel(handle)` is invoked
  - THEN the underlying adapter receives cancel and returns success; idempotent on repeats

- Retry/Breaker Interactions
  - GIVEN transient adapter failures
  - WHEN facade retries according to policy
  - THEN attempts are bounded and breaker opens after threshold; subsequent submits fail fast until half-open

- Micro-batch Signaling (if applicable)
  - GIVEN configuration enabling micro-batching
  - WHEN streaming is initiated
  - THEN facade indicates grouping to emitter and ordering remains correct

## Fixtures & Mocks

- Minimal in-process stub adapter implementing required trait
- Deterministic clock/timer to assert breaker timing without sleeping

## Execution

- `cargo test -p adapter-host -- --nocapture`

## Traceability

- Aligns with `orchestratord` integration tests and `worker-adapters` streaming contracts

## Refinement Opportunities

- Micro-batch signaling from facade to emitter.
