# test-harness-bdd — Cross-crate Integration Test Harness (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

This harness validates cross-crate behavior only:
- Orchestrator data/control plane flows (admission, placement decisions reflected, SSE streaming, cancel).
- Adapter integration (stream framing, error taxonomy mapping, timeouts/retries visible to API clients).
- Pool readiness interactions and control flows (drain/reload health signals) passing through the API.
- Observability envelopes and metrics conformance over HTTP.

Out of scope:
- Per-crate unit/behavior tests (now live under each crate).
- Schema generation/validation; preflight/apply internals.

## Consumed & Provided Contracts

- Consumes orchestrator OpenAPI contracts (control/data). Uses adapter trait via `orchestratord`.
- Provides BDD features and step implementations tied to integration endpoints only.

## Testing Policy

- Focus on golden-path and critical error-path scenarios spanning multiple crates.
- Avoid duplicating crate-scoped tests.

## Minimal Auth Scenarios

- Positive (loopback dev posture): when `AUTH_OPTIONAL=true` and requests originate from loopback, data/control endpoints succeed without `Authorization` and identity breadcrumbs log as `identity=localhost`.
- Negative (non-loopback without token): when bound to non-loopback and `AUTH_TOKEN` is required, requests without `Authorization: Bearer` MUST receive `401/403` auth errors with correlation IDs; logs MUST include `identity=token:<fp6>` when a token is presented (never full token).

## Refinement Opportunities

- Add synthetic backpressure tests with configurable queue capacity.
- Capability handshake tests once standardized across adapters.
