# Orchestratord — Implementation Plan (OC-CTRL-2xxx)

Spec: `.specs/20-orchestratord.md`
Scope: control/data plane handlers, SSE framing, backpressure, typed errors, security.

## Stages and Deliverables

- Stage 0 — Contract Freeze
  - Author/align OpenAPI: `contracts/openapi/{control.yaml,data.yaml}` with `x-req-id`.
  - Add `x-examples` and JSON bodies per UX/DX proposal (429 policy_label, advisory fields).

- Stage 1 — CDC Consumer + Snapshots
  - `cli/consumer-tests` pact interactions for enqueue, stream (SSE frames), cancel, sessions; commit pact JSON.

- Stage 2 — Provider Verification
  - `orchestratord/tests/provider_verify.rs` verifies handlers against pact files.
  - Minimal vertical slice: admission → placement → SSE; typed errors with codes and context.

- Stage 5 — Observability
  - Ensure logs fields and metrics as per specs; expose `/metrics` scrape.

## Tests

- Provider verify: `orchestratord/tests/provider_verify.rs` (OC-CTRL-2001..2051).
- Unit/integration tests for SSE framing and backpressure headers/bodies.
- BDD features: `test-harness/bdd/tests/features/data_plane/`, `control_plane/`, `sse/`.

## Acceptance Criteria

- OpenAPI diff-clean on regen; examples compile.
- Provider verification green; SSE events in order and well-formed.
- Backpressure headers + body policy label present; error taxonomy stable.

## Backlog (initial)

- Handlers for control plane: drain/reload/health/replicasets.
- Data plane: createTask, stream SSE, cancel; correlation ID echo.
- Typed errors with codes and retry hints; backpressure headers/bodies.
- Auth middleware (API key) and logging filters.
