# Orchestratord SPEC — Control/Data Plane, SSE, Backpressure (v1.0)

Status: Stable (draft)
Applies to: `orchestratord/`
Conformance language: RFC‑2119 (MUST/SHOULD/MAY)

## 0) Scope & Versioning

This SPEC defines the public control and data plane behavior of `orchestratord`, including SSE framing and backpressure. Requirements are versioned as `OC-CTRL-2xxx`.

## 1) Control Plane

- [OC-CTRL-2001] `GET /v1/pools/:id/health` MUST return liveness, readiness, draining, and metrics snapshot fields.
- [OC-CTRL-2002] `POST /v1/pools/:id/drain` MUST accept a JSON body with `deadline_ms` and MUST begin draining.
- [OC-CTRL-2003] `POST /v1/pools/:id/reload` MUST atomically switch model references or fail and roll back.
- [OC-CTRL-2004] `GET /v1/replicasets` MUST enumerate replica sets with load/SLO snapshots.

See: [contracts/openapi/control.yaml](../contracts/openapi/control.yaml), [orchestratord/src/main.rs](../orchestratord/src/main.rs), [orchestratord/tests/provider_verify.rs](../orchestratord/tests/provider_verify.rs)

## 2) Data Plane — OrchQueue v1

- [OC-CTRL-2010] `POST /v1/tasks` MUST perform admission checks (ctx, token budget) before enqueue.
- [OC-CTRL-2011] On queue full, server MUST reply `429` and include `Retry-After` and `X-Backoff-Ms`. A JSON body MUST include the full policy label.
- [OC-CTRL-2012] `POST /v1/tasks/:id/cancel` MUST be race‑free; no tokens may be emitted after cancel.

## 3) SSE Framing

- [OC-CTRL-2020] `GET /v1/tasks/:id/stream` MUST emit events `started`, `token`, `metrics`, `end`, `error`.
- [OC-CTRL-2021] `started` MUST include `queue_position` and `predicted_start_ms` when available.
- [OC-CTRL-2022] Event payloads MUST be well‑formed JSON; ordering MUST be per stream.

### 3.1 Event payload fields (authoritative)

- `started` → `{ queue_position: int, predicted_start_ms: int }`
- `token` → `{ t: string, i: int }`  // token text and incremental index
- `metrics` → `{ /* engine/pool specific snapshot; non-breaking additive */ }`
- `end` → `{ tokens_out: int, decode_ms: int }`
- `error` → `{ code: ErrorKind, message: string, engine?: Engine }`

OpenAPI component schemas:
- See `contracts/openapi/data.yaml` `components/schemas`:
  `SSEStarted`, `SSEToken`, `SSEMetrics`, `SSEEnd`, `SSEError`.

## 4) Error Taxonomy

- [OC-CTRL-2030] Errors MUST include a stable `code` field: `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DECODE_TIMEOUT`, `WORKER_RESET`, `INTERNAL`.
- [OC-CTRL-2031] Errors SHOULD include the `engine` and `pool_id` when applicable.

## 5) Security

- [OC-CTRL-2040] Control and data plane MUST be gated by AuthN/AuthZ; API keys acceptable day‑1.
- [OC-CTRL-2041] Logs MUST NOT leak secrets or API keys.

## 6) Observability

- [OC-CTRL-2050] Admission logs and `started` MUST include `queue_position` and `predicted_start_ms` when available.
- [OC-CTRL-2051] Metrics MUST include queue depth, reject/drop rates, latency percentiles, and error counts by class.

## 7) Traceability

- Code: [orchestratord/src/main.rs](../orchestratord/src/main.rs)
- Tests: [orchestratord/tests/provider_verify.rs](../orchestratord/tests/provider_verify.rs)
- Contracts: [contracts/openapi/data.yaml](../contracts/openapi/data.yaml), [contracts/openapi/control.yaml](../contracts/openapi/control.yaml)
