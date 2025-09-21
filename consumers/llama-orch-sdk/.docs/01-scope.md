# SDK Scope and Layering (Design Phase)

Status: draft (design-only)
Owner: consumers/llama-orch-sdk
Updated: 2025-09-21

## Purpose

Provide a minimal, stable SDK surface consisting of:
- Typed data models mirroring the orchestrator OpenAPI contracts.
- A small typed client for HTTP and optional streaming (SSE).

SDK exists to enable Utils. Utils owns applet logic, file I/O wrappers, prompt templating, proof bundles, and process wiring. Orchestrator serves the SDK contracts.

Relationship: Utils > SDK > Orchestrator.

## In Scope (MVP)
- Types for capabilities, catalog models, sessions, tasks/admission, SSE events, and error envelopes.
- Typed client methods for:
  - Capability discovery: `GET /v1/capabilities`.
  - Data plane: `POST /v1/tasks`, `GET /v1/tasks/{id}/stream`, `POST /v1/tasks/{id}/cancel`.
  - Sessions: `GET /v1/sessions/{id}`.
- Error mapping to a single `ErrorEnvelope` surface.
- Minimal configuration: base URL, optional API key, request timeout.

## Out of Scope
- Any applet or Blueprint logic (Utils owns these).
- Filesystem behavior or artifact proof-bundle emission.
- Retries, rate limiting, caching, circuit breakers (future enhancements).
- Engine provisioning and model fetching (server-side responsibility).

## Transport
- HTTP JSON per OpenAPI definitions in `contracts/openapi/{control.yaml,data.yaml}`.
- Streaming via Server-Sent Events (SSE) for `GET /v1/tasks/{id}/stream` with events: `started`, `token`, `metrics`, `end`, `error`.

## Configuration
- `baseURL` (default `http://127.0.0.1:8080/`).
- `apiKey` (optional header `X-API-Key` when enabled by deployment).
- `timeoutMs` default 30_000 for non-streaming calls.

## Errors
- Methods reject/return with `ErrorEnvelope` derived from OpenAPI:
  - `code`, `message`, optional `engine`, `retriable`, `retry_after_ms`, `policy_label`.
- SSE `event:error` frames carry `{ code, message, engine? }` and terminate the stream.

## Cross-language Targeting
- Rust-native crate is primary.
- WASM/TS: thin wrapper over the same shapes (no additional surface).

## Determinism & Placement Notes (for consumers)
- Determinism defaults to strict when the engine supports it; pass `seed`, `sampler_profile_version` via `TaskRequest`.
- Placement overrides are available via `TaskRequest.placement` (pin/prefer/avoid) to support explicit GPU/pool pinning when policy allows; automatic placement is the default.
