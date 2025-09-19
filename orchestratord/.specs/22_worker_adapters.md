# Wiring: orchestratord ↔ worker-adapters

Status: Draft
Date: 2025-09-19

## Relationship
- `orchestratord` hosts the HTTP control/data plane and drives adapters to serve tasks. Adapters implement `WorkerAdapter` and hide engine-specific details.

### Where the connection lives (implementation notes)
- The crate depends on the shared adapter trait crate: `worker-adapters-adapter-api`.
- A default feature `mock-adapters` wires in `worker-adapters-mock` for vertical-slice/dev flows (see `orchestratord/Cargo.toml`).
- Orchestratord maintains an adapter registry (per pool/replica context) and dispatches to a Ready adapter instance after placement.

## Expectations on worker-adapters
- Implement `WorkerAdapter` (health, props, submit/cancel, engine_version) with timeouts/retries and error taxonomy mapping.
- Stream `started` → `token*` → `end` frames (and optional `metrics`), mapping determinism/log fields.
- Redact secrets and respect network egress/timeouts.

### Shared utilities & facade

- HTTP-based adapters MUST use `worker-adapters/http-util` for client construction, retries with capped jitter, HTTP/2 keep-alive, and header redaction.
- Integration inside `orchestratord` SHOULD go through the in-process Adapter Host facade (`adapter-host/.specs/00_adapter_host.md`) for bind/rebind, submit/cancel routing, and consistent narration/metrics wrappers.
- When orchestrator Minimal Auth is configured, clients MUST attach `Authorization: Bearer <token>` on calls from adapters/host to orchestrator endpoints (control/health), honoring loopback exceptions per `/.specs/11_min_auth_hooks.md`.

## Expectations on orchestratord
- Enforce admission/backpressure and dispatch to Ready adapters only.
- Surface queue position and predicted_start_ms; propagate cancel to adapters.
- Map adapter errors to HTTP error envelopes.

### Binding & lifecycle
- Maintain a mapping of `pool_id -> Box<dyn WorkerAdapter>` (or per-replica instances) initialized during preload/health transitions.
- Re-bind adapters on reload/drain events; ensure cancellation and in-flight stream handling are correct.

### Error and cancel propagation
- Map `WorkerError::{DeadlineUnmet, PoolUnavailable, DecodeTimeout, WorkerReset, Adapter, Internal}` to HTTP error taxonomy.
- On `/v1/tasks/{id}` cancel, call `adapter.cancel(task_id)` and tear down the SSE stream promptly.

### Backpressure
- Respect upstream admission decisions (429 with retry-after). Adapters must not block indefinitely; timeouts and retry caps are enforced by adapters.

## Data Flow
- Admission: `/v1/tasks` → adapter `submit()` → SSE events to client.
- Cancel: `/v1/tasks/{id}` → adapter `cancel()`.
- Health: `/v1/pools/{id}/health` → aggregate adapter health/props.

## Refinement Opportunities
- Define a standard capability snapshot exchange to surface ctx_max, workloads, features per engine.
