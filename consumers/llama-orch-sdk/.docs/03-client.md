# SDK Client Surface (Design Phase)

Status: draft (design-only)
Updated: 2025-09-21

The SDK provides a minimal typed client over the orchestrator HTTP + SSE APIs.
Only methods backed by OpenAPI are included. All methods return Promises (TS) or Results (Rust) and reject/err with `ErrorEnvelope`-mapped errors.

## Configuration

- `baseURL?: string` (default `http://127.0.0.1:8080/`)
- `apiKey?: string` (sent as `X-API-Key` if provided)
- `timeoutMs?: number` (default 30000 for non-streaming calls)

## Methods (public)

- `list_engines(): Promise<EngineCapability[]>`
  - Derived from `GET /v1/capabilities` → `Capabilities.engines` mapped to `{ id, workloads }`.

- `enqueue_task(req: TaskRequest): Promise<AdmissionResponse>`
  - Maps to `POST /v1/tasks`.

- `stream_task(taskId: string, opts?): AsyncIterable<SSEEvent>`
  - Maps to `GET /v1/tasks/{id}/stream`.
  - Emits `SSEEvent` union: `started | token | metrics | end | error`.

- `cancel_task(taskId: string): Promise<void>`
  - Maps to `POST /v1/tasks/{id}/cancel`.

- `get_session(sessionId: string): Promise<SessionInfo>`
  - Maps to `GET /v1/sessions/{id}`.

## Omitted in MVP (no canonical OpenAPI list endpoints)

- `list_models()` — Not provided by `contracts/openapi` (catalog has create/get/verify/state/delete only).
- `list_pools()` — Not provided by `contracts/openapi` (only per-pool health/drain/reload).

## Errors

All methods reject with an `ErrorEnvelope` derived from responses:
- Fields: `code`, `message`, optional `engine`, `retriable`, `retry_after_ms`, `policy_label` (for backpressure).
- SSE `event:error` frames terminate the stream; the iterator stops with an error value containing `{ code, message, engine? }`.

## Notes for Utils

- Utils may pre-assemble chat/message threads into a single `TaskRequest.prompt` string until chat-native schemas are introduced.
- Determinism: pass `seed` and optional `sampler_profile_version` when required for reproducibility.
- Placement: use `TaskRequest.placement` for pin/prefer/avoid according to policy; automatic placement is the default.
