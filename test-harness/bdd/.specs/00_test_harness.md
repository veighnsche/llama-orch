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

## Refinement Opportunities

- Add synthetic backpressure tests with configurable queue capacity.
- Capability handshake tests once standardized across adapters.
