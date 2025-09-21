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
- [OC-CTRL-2011] On queue full, server MUST reply `429` and include `Retry-After` and `X-Backoff-Ms`. The JSON body MUST include `policy_label`, `retriable`, and `retry_after_ms` (advisory) to guide clients.
- [OC-CTRL-2012] `POST /v1/tasks/:id/cancel` MUST be race‑free; no tokens may be emitted after cancel.

### 2.1 Optional Pin Override

- [OC-CTRL-2013] When `TaskRequest` includes a valid `pool_id` pin override and policy allows pinning, the server MUST route the task to the specified pool and MUST NOT consider other pools for placement.
- [OC-CTRL-2014] If the specified pool is unknown or not Ready, or pinning is disabled by policy, admission MUST fail with a deterministic typed error (`INVALID_PARAMS` or `POOL_UNREADY`) and MUST NOT silently fall back to automatic placement.

## 3) SSE Framing

- [OC-CTRL-2020] `GET /v1/tasks/:id/stream` MUST emit events `started`, `token`, `metrics`, `end`, `error`.
- [OC-CTRL-2021] `started` MUST include `queue_position` and `predicted_start_ms` when available.
- [OC-CTRL-2022] Event payloads MUST be well‑formed JSON; ordering MUST be per stream.

### 3.1 Transport & Performance (normative)

- [OC-CTRL-2025] The server SHOULD enable HTTP/2 for SSE where supported and MUST gracefully fallback to HTTP/1.1 when negotiation fails. Compression SHOULD be disabled for small token frames and MAY be enabled for large frames.
- [OC-CTRL-2026] The SSE encoder MUST use a buffered writer and avoid per‑token heap allocations on the hot path. An optional micro‑batch mode MAY coalesce tokens within a small latency budget; it is DISABLED by default and MUST be bounded.
- [OC-CTRL-2027] Event ordering MUST remain `started → token* → end` (with optional `metrics` frames interleaved). Heartbeat/keepalive events, if added, MUST remain compatible with existing parsers.

### 3.3 Streaming Failure Semantics (uniform)

- [OC-CTRL-2028] When a streaming failure occurs after the stream has started, the server MUST emit `event: error` with a minimal JSON body: `{ code: string, retriable: boolean, retry_after_ms?: number, message?: string }`.
- [OC-CTRL-2029] After an `event: error` is emitted, the stream MUST terminate; no further `token`/`metrics`/`end` events MAY be sent.
- [OC-CTRL-2034] Pre‑stream errors MUST use HTTP status codes. Established streams MUST stay `200 OK` and carry the error via the SSE `error` event.

### 3.2 Event payload fields (authoritative)

- `started` → `{ queue_position: int, predicted_start_ms: int }`
- `token` → `{ t: string, i: int }`  // token text and incremental index
- `metrics` → `{ /* engine/pool specific snapshot; non-breaking additive */ }`
- `end` → `{ tokens_out: int, decode_ms: int }`  // canonical name is `decode_time_ms` in root spec; implementations MAY include both during migration
- `error` → `{ code: ErrorKind, message: string, engine?: Engine }`

OpenAPI component schemas:
- See `contracts/openapi/data.yaml` `components/schemas`:
  `SSEStarted`, `SSEToken`, `SSEMetrics`, `SSEEnd`, `SSEError`.

## 4) Error Taxonomy

- [OC-CTRL-2030] Errors MUST include a stable `code` field: `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DECODE_TIMEOUT`, `WORKER_RESET`, `INTERNAL`.
- [OC-CTRL-2031] Errors SHOULD include the `engine` and `pool_id` when applicable.
- - [OC-CTRL-2032] Error envelopes SHOULD include advisory fields when available: `retriable: boolean` and `retry_after_ms: int64`. These fields are optional and non‑breaking and SHOULD be populated for backpressure and transient errors. When surfaced inside SSE `event:error`, the same fields SHOULD be included.

## 5) Security

- [OC-CTRL-2040] There is no AuthN/AuthZ in the home‑profile. Control and data plane are open locally. (Future profiles MAY introduce AuthN/AuthZ behind features.)
- [OC-CTRL-2041] Logs MUST NOT leak secrets or API keys (e.g., adapter upstream tokens). Redaction remains mandatory.

## 6) Observability

- [OC-CTRL-2050] Admission logs and `started` MUST include `queue_position` and `predicted_start_ms` when available.
- [OC-CTRL-2051] Metrics MUST include queue depth, reject/drop rates, latency percentiles, and error counts by class.
- [OC-CTRL-2052] Correlation ID: If a request includes `X-Correlation-Id`, the server MUST echo the same value in all responses and streaming (SSE) responses. If absent, the server MUST generate a UUIDv4 and include it. All non‑`204 No Content` responses MUST include this header. Narration and structured logs SHOULD include the correlation ID field.

### 6.1 Narration Hooks (repo‑wide cross‑reference)

- Hooks SHOULD emit short, human‑readable narration alongside structured fields at key points (admission, placement, stream start/end, cancel) per `/.specs/00_llama-orch.md §2.8.1`.

## 7) Traceability

- Code: [orchestratord/src/main.rs](../orchestratord/src/main.rs)
- Tests: [orchestratord/tests/provider_verify.rs](../orchestratord/tests/provider_verify.rs)

## 8) Capabilities & Discovery

- [OC-CTRL-2060] The server MUST expose a dedicated `GET /v1/capabilities` endpoint that returns, per engine/pool: `engine`, `engine_version`, `sampler_profile_version` (when applicable), `ctx_max`, `max_tokens_out`, declared concurrency/slots, `supported_workloads`, rate limits, and feature flags. This endpoint MUST be versioned and documented in OpenAPI.
- [OC-CTRL-2061] Capability payloads MUST include an API version field compatible with OpenAPI `info.version`, enabling the CLI to pin a compatible range.
- [OC-CTRL-2062] `GET /v1/replicasets` is REMOVED pre‑1.0 and MUST NOT be served.

### 8.1 Optional Output Mode Hint

- [OC-CTRL-2063] `TaskRequest` MAY include `output_mode: "text" | "json" | "edits"` as a hint for output packaging/validation and artifact tagging. Unknown values MUST be ignored. Servers MUST NOT change model semantics based on this hint.

## 9) Artifact Registry (Optional, Recommended)

- [OC-CTRL-2065] The server SHOULD provide `POST /v1/artifacts` to persist structured artifacts (plans, summaries, diffs, traces) with content-addressed IDs and tags. Request and response schemas MUST be defined in OpenAPI if implemented.
- [OC-CTRL-2066] The server SHOULD provide `GET /v1/artifacts/{id}` to retrieve artifacts by ID, including metadata (tags, lineage, timestamps). In the home‑profile, no authorization is enforced; future profiles MAY add AuthZ.
 - [OC-CTRL-2067] Each job SHOULD produce an artifact record even on failure, referencing: `job_id`, `session_id`, request params (prompt/inputs redacted, `max_tokens`, `ctx`, `engine`, `engine_version`, `sampler_profile_version`, `model_ref`, digest when known), `seed?`, key metrics (`tokens_out`, `decode_time_ms`), and an SSE transcript (inline or by reference). Failure paths SHOULD include error context and partial transcripts when available.

## 10) Budgets & Guardrails

- [OC-CTRL-2068] Per-session budgets (token/time/cost) SHOULD be supported and enforced at admission or scheduling time. When budgets are active, the server SHOULD surface budget state via SSE `metrics` frames and/or response headers.

## 11) SSE Metrics – Scheduling Signals

- [OC-CTRL-2023] The `metrics` SSE frames SHOULD include fields helpful for client-side planning under load, such as `on_time_probability` (number), `queue_depth` (int), and `kv_warmth` (bool). Fields MAY be engine/pool specific and are additive only.

## 12) OpenAPI Examples & Annotations

- [OC-CTRL-2067] Data‑plane endpoints in `contracts/openapi/data.yaml` (enqueue, stream/SSE frames, cancel, sessions) MUST include `x-examples` demonstrating typical requests and responses.
- [OC-CTRL-2069] Control‑plane endpoints SHOULD include `x-examples` for drain, reload, and capabilities.

## 13) CORS / Preflight (Optional)

- [OC-CTRL-2070] Implementations MAY support CORS for localhost tooling, replying to `OPTIONS` with appropriate `Access-Control-Allow-*` headers (including `X-Correlation-Id` and `Authorization` when Minimal Auth seam is active). CORS MUST be disabled by default; enabling MUST be non‑breaking.

## Refinement Opportunities

- Define an SSE heartbeat/keepalive event for long queues while ensuring compatibility with current parsers.
- Provide canonical `x-examples` for mixed‑GPU scenarios including `queue_position` and `predicted_start_ms` evolution.
- Explore exposing correlation ID inside SSE `started` payload (as redundant metadata) vs. header‑only.
- Consider an `X-Backoff-Policy` header to carry a stable `policy_label` alongside JSON for simpler CDC.
