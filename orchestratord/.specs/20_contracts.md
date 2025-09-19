# orchestratord — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts

- HTTP APIs (OpenAPI)
  - Control plane: `contracts/openapi/control.yaml` (drain, reload, health, capabilities, catalog CRUD/verify/state).
  - Data plane: `contracts/openapi/data.yaml` (admission → SSE streaming, cancel).
- Error envelope and correlation
  - All responses include structured error bodies on failure and SHOULD echo `X-Correlation-Id` when provided.
- SSE/streaming behavior
  - `metrics` and `token` events, `started`/`end` framing, cancel propagation.
- Observability
  - Logs with fields from `README_LLM.md`.
  - Metrics per `.specs/metrics/otel-prom.md`.

## Consumed Contracts (Expectations on Others)

- `orchestrator-core`
  - Queue/admission façade and placement hooks (data shapes per core spec).
- `worker-adapters/*`
  - `WorkerAdapter` trait for health/props/submit/cancel/version; adapters must enforce timeouts and map errors.
- `pool-managerd`
  - Health and readiness, capacity and perf hints, engine version; drain/reload execution.
- `model-provisioner` & `engine-provisioner`
  - Preload/reload flows (ensure present → ensure engine) orchestrated by control plane.
- `catalog-core`
  - Catalog CRUD backing for catalog endpoints.

## Data Exchange (High Level)

- Control:
  - `/v1/pools/{id}/drain`, `/reload`, `/health` → `pool-managerd` actions and registry readouts.
  - `/v1/catalog/models*` → `catalog-core` CRUD via internal services.
  - `/v1/capabilities` → aggregate of adapters/pools.
- Data:
  - `/v1/tasks` admission → SSE over adapters; cancel via `/v1/tasks/{id}`.

## Versioning & Compatibility

- OpenAPI changes MUST bump version and regenerate clients; provider verification tests MUST be updated.
- Internal crate interfaces are stable within the repo pre‑1.0 but MUST be updated in lockstep with callers.

## Security & Policy

- API key/middleware; rate limits; body size limits; correlation IDs; safe logging without secrets.

## Testing Expectations

- Provider verification against OpenAPI.
- Unit/integration tests for error mapping, headers, and SSE framing; BDD integration for cross‑crate flows.

## Refinement Opportunities

- Enforce `X-Correlation-Id` propagation end‑to‑end; tighten error taxonomy.
- Formalize capability schema across adapters/pools.
