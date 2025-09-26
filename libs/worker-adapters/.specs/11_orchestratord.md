# Wiring: worker-adapters ↔ orchestratord

Status: Draft
Date: 2025-09-19

## Relationship
- `orchestratord` is the application boundary. It uses adapters through the shared `WorkerAdapter` trait to service data‑plane requests.

## Expectations on orchestratord
- Maintain adapter instances per pool/replica context and dispatch only to Ready adapters.
- Enforce admission/backpressure; propagate cancel; surface queue position and predicted_start_ms in SSE.
- Map adapter errors/taxonomy to HTTP error envelopes with `X-Correlation-Id`.

### Auth seam

- When Minimal Auth is configured, clients MUST attach `Authorization: Bearer <token>` on calls from adapters (or the Adapter Host facade) to orchestrator endpoints. Loopback exceptions MAY apply per `/.specs/11_min_auth_hooks.md`.

## Expectations on adapters
- Implement `WorkerAdapter` (health/props/submit/cancel/engine_version) and map engine‑specific errors to `WorkerError`.
- Stream `started` → `token*` → `end` (optional `metrics`) respecting timeouts; redact secrets.

## Data Flow
- `/v1/tasks` → adapter `submit()` → SSE events to client.
- `/v1/tasks/{id}` → adapter `cancel()`.
- `/v1/pools/{id}/health` → aggregate adapter health/props.

## Refinement Opportunities
- Capability snapshot schema for ctx_max/workloads/features and transport to `orchestratord`.
