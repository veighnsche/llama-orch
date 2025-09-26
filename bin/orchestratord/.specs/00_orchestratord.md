### Layering note (informative)

- **Service role:** Orchestrator defines the API ground truth via OpenAPI/specs. The **SDK** mirrors these contracts with typed models and transport only. **Utils** (applets, determinism, proof bundles) drives what the SDK needs but does not couple to orchestrator internals. The **CLI** uses the SDK to bootstrap and generate bindings/snapshots. See `consumers/.docs/.adr/006-library-split.md`.

# Orchestratord Crate Specification (Standalone)

Status: draft
Last updated: 2025-09-17

## 1. Overview
`orchestratord` is the HTTP control and data-plane daemon for llama-orch. It exposes admission, streaming (SSE), session, artifact registry, capability discovery, and pool control endpoints. This document specifies the crate’s responsibilities, public API, features, dependencies, configuration, integrations, and testing approach as a standalone unit.

Goals:
- Spec → Contract → Tests → Code alignment
- Determinism by default (contracted; to be fully enforced end-to-end)
- Strong observability (correlation IDs, Prometheus metrics, SSE transcript capture)

Non-goals:
- Implementing individual model engines (done by worker adapters)
- Multi-host cluster orchestration beyond single host reference environment

## 2. Responsibilities
- Data-plane admission and task lifecycle (enqueue, stream, cancel)
- Session lifecycle and budgets (tokens/time/cost)
- Artifact registry (content-addressed documents + metadata)
- Capability discovery (unified on /v1/capabilities)
- Control plane for pools (drain, reload, health) and lifecycle state
- Observability endpoints (/metrics) and correlation-ID propagation

## 3. Public API (HTTP)
OpenAPI sources:
- Control: `contracts/openapi/control.yaml`
- Data: `contracts/openapi/data.yaml`

Data plane:
- `POST /v1/tasks` — enqueue task, returns 202 with `queue_position`, `predicted_start_ms`, and budget headers
- `GET  /v1/tasks/:id/stream` — SSE stream: `started`, repeated `token`, optional repeated `metrics`, `end`; transcript persisted as artifact
- `POST /v1/tasks/:id/cancel` — cancel queued/running task
- `GET  /v1/sessions/:id` — session introspection (TTL, turns, KV metadata, budgets)
- `DELETE /v1/sessions/:id` — delete session

- Cancel semantics: MUST be race-free; no tokens may be emitted after cancel (planned propagation of cancel to adapters)

Artifacts:
- `POST /v1/artifacts` — create document; content-addressed by SHA-256; returns 201 and `id`
- `GET  /v1/artifacts/:id` — fetch document; 404 if not found

Capability discovery:
- `GET /v1/capabilities` — single source of truth; includes `api_version` and engine capability snapshot. Shape:
  - `{"api_version":"1.0.0","engines":[{"engine":"llamacpp","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}}, ... for vllm, tgi, triton]}`
- `GET /v1/replicasets` — REMOVED pre‑1.0; MUST NOT be served (no backwards-compatibility shims)

Control plane:
- `POST /v1/pools/:id/drain` — mark draining; readiness 0
- `POST /v1/pools/:id/reload` — model swap; readiness 1 (happy path)
- `GET  /v1/pools/:id/health` — liveness/readiness, draining flag, pool metrics JSON, last_error
- `POST /v1/catalog/models/:id/state` — set model state; only `Active|Retired` supported
- Catalog CRUD/verify stubs exist; persistence is TODO

Observability:
- `GET /metrics` — Prometheus text format; server-generated `X-Correlation-Id`

## 4. Headers & Auth
- Minimal Auth seam (spec-only): prefer `Authorization: Bearer <token>` per `/.specs/11_min_auth_hooks.md`.
  - Identity breadcrumbs in logs: `identity=localhost` for loopback, or `identity=token:<fp6>` (never log full token).
  - Startup refusal when bound to non-loopback without `AUTH_TOKEN` (see config schema seam).
- Current implementation note: historical `X-API-Key` scaffolding may exist in code/tests; this MUST be replaced by the Bearer seam in future work. Specs and contracts reference the Bearer seam going forward.
- Correlation ID: echo `X-Correlation-Id` if provided, otherwise generate UUID v4 and include in response.
- Backpressure: on 429, include `Retry-After` (s) and `X-Backoff-Ms` (ms).
- Budget headers: `X-Budget-Tokens-Remaining`, `X-Budget-Time-Remaining-Ms`, `X-Budget-Cost-Remaining`.
- Deprecation: `Deprecation: true` on `/v1/replicasets`.
- Catalog trust: `X-Trust-Policy: strict` enforces signatures; with `signed: false` yields `400 {"code":"UNTRUSTED_ARTIFACT"}`.

## 5. SSE Streaming Protocol
Response headers:
- `Content-Type: text/event-stream`
- `X-Correlation-Id: <uuid>`
- Optional budget headers at stream start (stub values in current impl)

Event order and shapes:
- `started`: `{"queue_position": <int>, "predicted_start_ms": <int>}`
- `token`: payload forwarded from adapter stream (`te.data`); the stub fallback uses `{"t": <string>, "i": <int>}`
- `metrics` (emitted after first token): `{"queue_depth": <int>, "on_time_probability": <float>, "kv_warmth": <bool>, "tokens_budget_remaining": <int>, "time_budget_remaining_ms": <int>, "cost_budget_remaining": <float>}`
- `end`: `{"tokens_out": <int>, "decode_ms": <int>}`
- `error`: `{"code": <string>, "message": <string>, "engine": <enum>}` where code maps adapter errors:
  - `DEADLINE_UNMET`, `POOL_UNAVAILABLE`, `DECODE_TIMEOUT`, `WORKER_RESET`, `INTERNAL`

Transcript persistence:
- Upon `end`, the concatenated transcript is stored as an artifact with SHA-256 ID and metadata

## 6. Session Store & Budgets
- In-memory registry keyed by `session_id`
- Tracks TTL, turns, `kv_bytes`, `kv_warmth`, and budgets: `tokens/time/cost`
- `kv_warmth` toggles to true on first token of a stream
- Defaults: TTL 600_000 ms; max turns 8 (enforcement to be wired into admission)
- Budget updates:
  - On stream end: spend `tokens_out`, spend time `decode_ms`, and spend cost using placeholder cost model `$0.000002 * tokens_out`
- Eviction:
  - Zero-TTL sessions are pruned on `create_task`, `get_session`, and stream end
- Additional in-memory stores:
  - `tasks`: maps `task_id -> session_id` for observability; mapping is cleared on stream end
  - `sse`: stores last SSE transcript per `task_id` used by `GET /v1/tasks/:id/stream` fallback

## 7. Queue Admission & Backpressure
- Basic heuristic for `queue_position` and `predicted_start_ms` based on queue length; future: active leases and GPU throughput
- Current ETA heuristic: `predicted_start_ms = queue_position * 100` (ms per item)
- 429 responses include headers `Retry-After: 1`, `X-Backoff-Ms: 1000` and advisory JSON `{policy_label:"reject", retriable:true, retry_after_ms:1000}`
- Two 429 shapes are emitted:
  - Sentinel guard path (very large `expected_tokens`) returns `code:"QUEUE_FULL_DROP_LRU"`
  - Queue reject (policy Reject) returns `code:"ADMISSION_REJECT"`

## 8. Artifact Registry
- In-memory map for now; content-addressed via SHA-256 digest of document
- Create returns 201 body subset: `{id, kind, digest:"sha256:<hex>", created_ms, tags}`; full stored doc also includes `{content, parent, metadata}`
- SSE trace artifacts are auto-persisted on stream end with shape:
  - `{id, kind:"trace", digest:"sha256:<hex>", created_ms, tags:["sse","stream", "task:<id>", "session:<id>"], content:{transcript:<string>}, parent:null, metadata:{engine,engine_version,pool_id,replica_id}}`
- TODO: persistent storage backend (filesystem path in config)

## 9. Capability Discovery
- Primary: `/v1/capabilities` with `api_version` and engine feature snapshot
- `/v1/replicasets` is REMOVED pre‑1.0 and MUST NOT be served

- Capability payload SHOULD include declared concurrency per engine/pool (e.g., `"concurrency": <int>`) to aid client scheduling

Placement behavior:
- Minimal adapter selection by requested `engine`
- On stream start, allocate a lease for `pool0` and update `active_leases{pool_id}`; release and update on stream end

## 10. Control Plane & Lifecycle
- Pools: drain/reload/health wired to in-memory `PoolRegistry` in `pool-managerd`
- Lifecycle states: model `Active|Retired` only
- Health includes `last_error` and basic metrics JSON
- Metrics side-effects:
  - `drain_pool` increments `drain_events_total{pool_id,reason="api_request"}` and sets `pool_ready{pool_id}=0`
  - `reload_pool` sets `pool_ready{pool_id}=1`

## 11. Features & Optional Deps (Cargo)
`orchestratord/Cargo.toml` defines feature flags (defaults preserve current behavior):
- `server`: enables axum/tokio/http/futures/uuid
- `metrics`: enables Prometheus metrics and tracing-subscriber setup
- `artifacts`: enables `sha2` and `chrono` for content-addressed documents
- `mock-adapters`: enables mock adapters for vertical slice flows

Guidance:
- Consumers can disable `mock-adapters` when wiring real adapters
- Future features may gate catalog persistence, NVML telemetry, or external stores

## 12. Dependencies (by area)
Core:
- `axum`, `tokio`, `http`, `futures` (server)
- `serde`, `serde_json`, `uuid`, `anyhow`, `tracing`
- `prometheus` (metrics), `tracing-subscriber` (init)
- `sha2`, `chrono` (artifacts)
Cross-crate:
- `contracts-api-types` (domain types)
- `orchestrator-core` (queue/backpressure)
- `pool-managerd` (health registry)
- `worker-adapters` (adapter API and mock)

## 13. Configuration & Environment
- API key auth (header)
- Budgets defaults (tokens/time/cost) — source from config schema (TODO for persistence across restarts)
- Artifact storage path (for persistent store) — TODO
- Optional: OpenTelemetry exporter settings — TODO

## 14. Error Taxonomy
- `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DEADLINE_UNMET`
- HTTP status mappings: 400 for invalid params/deadline; 429 for backpressure; 409 for conflict on reload; 404 for missing artifacts; 202/200 for success paths

Sentinel behaviors (planning stubs, current implementation):
- `INVALID_PARAMS` (400)
  - `ctx < 0`
  - `ctx > 32768` or `max_tokens > 50000`
- `DEADLINE_UNMET` (400)
  - `deadline_ms <= 0`
- `POOL_UNAVAILABLE` (503)
  - `model_ref == "pool-unavailable"`
  - Lifecycle `Retired` gate in create_task
- `INTERNAL` (500)
  - `prompt == "cause-internal"` sentinel
- Backpressure 429s
  - Large `expected_tokens >= 1_000_000` → `code: "QUEUE_FULL_DROP_LRU"` plus advisory headers/body
  - Queue reject via policy Reject → `code: "ADMISSION_REJECT"` plus advisory headers/body

## 15. Observability
- `/metrics` returns Prometheus text; linter compliance with `ci/metrics.lint.json`
- Correlation ID is included on `/metrics` responses (server-generated)
- Structured logs include correlation IDs and admission/stream events
- Log fields SHOULD include `engine_version` and `sampler_profile_version` when available
- Key series emitted include (non-exhaustive):
  - Counters: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total{reason}`, `tasks_rejected_total{reason}`, `admission_backpressure_events_total{policy}`
  - Gauges: `queue_depth{engine,engine_version,pool_id,priority}`, `model_state{model_id,state}`, `active_leases{pool_id}`, `pool_ready{pool_id}`, `kv_cache_usage_ratio{...}`, `gpu_utilization{...}`, `vram_used_bytes{...}`
  - Histograms: `latency_first_token_ms{engine,engine_version,pool_id,priority}`, `latency_decode_ms{...}`

## 16. Integrations Across Workspace
- `orchestrator-core`: QueueWithMetrics, policies, labels
- `pool-managerd`: Pool registry, readiness/liveness, last_error, version
- `worker-adapters`: Adapter API; default mock; planned: llamacpp/vllm/tgi/triton HTTP clients
- Contracts: OpenAPI in `contracts/openapi/*.yaml`
- Test harnesses: provider verify, BDD, determinism suite, metrics lint

## 17. Testing Strategy
- Unit tests per handler module (auth, data, control, artifacts, observability)
- Provider verification tests (`orchestratord/tests/provider_verify.rs`)
- Metrics linter tests (`test-harness/metrics-contract`)
- Determinism suite (contracts present; end-to-end enforcement TODO)
- BDD harness for SSE ordering/spec

## 18. Directory Structure (crate)
```
orchestratord/
  src/
    http/ (data, control, catalog, artifacts, auth, observability)
    admission.rs, backpressure.rs, metrics.rs, placement.rs,
    session.rs, state.rs, sse.rs, services.rs, lib.rs, main.rs
  Cargo.toml
  README.md
```

## 19. Roadmap / TODOs (crate-specific)
- Unify capability discovery: prefer `/v1/capabilities`; remove `/v1/replicasets` in code/tests after deprecation window
- Admission estimates: incorporate active leases and throughput in predicted start calculations
- Determinism enforcement via adapters: propagate `seed` to adapters and ensure single-slot/deterministic mode per engine; expand determinism tests
- Session store: expiry/eviction rigor and cost model; persist across restarts
- Artifact registry: persistent filesystem store; retention/GC; access limits
- Catalog: persistence layer and trust warnings; CRUD tests
- Observability: NVML integration and structured tracing; optional OTEL exporters
- Security/policy: HTTP tooling guardrails and secret redaction/audit
- CLI: implement `llama-orch-cli` core flows and developer ergonomics
- Cleanup: remove reduction scaffolding and placeholders

## 20. Versioning & Compatibility
- OpenAPI `info.version`: 1.0.0 (signaled via `/v1/capabilities`)
- Adheres to semver once API stabilizes; until then, alpha warnings apply

## 21. Appendix
- Sample headers: `X-API-Key`, `X-Correlation-Id`, `Retry-After`, `X-Backoff-Ms`, budget headers
- Content types: JSON; SSE uses `text/event-stream`; metrics uses `text/plain; version=0.0.4`
- Sample SSE transcript and artifact document shapes captured by unit tests under `src/http/`
