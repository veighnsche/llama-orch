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
- [OC-CTRL-2004] Discovery MUST use `GET /v1/capabilities`. `GET /v1/replicasets` is REMOVED pre‑1.0 and MUST NOT be served.

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
 - [OC-CTRL-2052] Responses that include headers SHOULD include `X-Correlation-Id` (except `204 No Content`) to enable client trace stitching.

## 7) Traceability

- Code: [orchestratord/src/main.rs](../orchestratord/src/main.rs)
- Tests: [orchestratord/tests/provider_verify.rs](../orchestratord/tests/provider_verify.rs)
- Contracts: [contracts/openapi/data.yaml](../contracts/openapi/data.yaml), [contracts/openapi/control.yaml](../contracts/openapi/control.yaml)

## 8) Capabilities & Discovery

- [OC-CTRL-2060] The server MUST expose a dedicated `GET /v1/capabilities` endpoint that returns engines, maximum context (`ctx_max`), declared concurrency, supported workloads, rate limits, and feature flags. This endpoint MUST be versioned and documented in OpenAPI.
- [OC-CTRL-2061] Capability payloads MUST include an API version field compatible with OpenAPI `info.version`, enabling the CLI to pin a compatible range.
- [OC-CTRL-2062] `GET /v1/replicasets` is REMOVED pre‑1.0 and MUST NOT be served.

## 9) Artifact Registry (Optional, Recommended)

- [OC-CTRL-2065] The server SHOULD provide `POST /v1/artifacts` to persist structured artifacts (plans, summaries, diffs, traces) with content-addressed IDs and tags. Request and response schemas MUST be defined in OpenAPI if implemented.
- [OC-CTRL-2066] The server SHOULD provide `GET /v1/artifacts/{id}` to retrieve artifacts by ID, including metadata (tags, lineage, timestamps). Authorization MUST be enforced.

## 10) Budgets & Guardrails

- [OC-CTRL-2068] Per-session budgets (token/time/cost) SHOULD be supported and enforced at admission or scheduling time. When budgets are active, the server SHOULD surface budget state via SSE `metrics` frames and/or response headers.

## 11) SSE Metrics – Scheduling Signals

- [OC-CTRL-2023] The `metrics` SSE frames SHOULD include fields helpful for client-side planning under load, such as `on_time_probability` (number), `queue_depth` (int), and `kv_warmth` (bool). Fields MAY be engine/pool specific and are additive only.
