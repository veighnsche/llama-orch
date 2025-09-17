# Orchestratord v2 — Phase II (Next 4 Weeks)

Status: planned
Last updated: 2025-09-17

Guiding principle: The v2 architecture spec leads the implementation. Tests support the refactor; if glue is not ready, keep stubs that compile.

-------------------------------------------------------------------------------
WEEK 5 — App & Middleware Completion, Domain Error Mapping, Server Binary
-------------------------------------------------------------------------------

1) Middleware & Router hardening
- Implement `app/middleware.rs` tower layers and apply in `app/router.rs`:
  - ApiKeyLayer (configurable via `MiddlewareConfig { require_api_key: true }`) enforcing `X-API-Key` on all routes except `/metrics`.
  - CorrelationIdLayer: if `X-Correlation-Id` is present, echo it; otherwise generate a UUID and set it on the response. Also attach to request extensions for logging.
  - ErrorMappingLayer: map `domain::error::OrchestratorError` to `(StatusCode, headers, Json<contracts_api_types::ErrorEnvelope>)`. Ensure headers like `Retry-After`, `X-Backoff-Ms` and envelope fields like `policy_label` are set when applicable.
- Refactor handlers in `api/*` to rely on middleware instead of inline header checks:
  - Update `api/data.rs`, `api/control.rs`, `api/artifacts.rs` functions to return `Result<impl IntoResponse, domain::error::OrchestratorError>` and remove uses of `api::types::{require_api_key, correlation_id_from}`.
  - Extend `domain::error::OrchestratorError` with variants for `AdmissionReject` (429), `Canceled`, and structured invalid params to replace inline taxonomy sentinels in `api/data::create_task`.
  - Keep explicit `Content-Type` for SSE endpoints in `api/data::stream_task` but rely on middleware for auth/correlation id.
- Router tightening in `app/router.rs`:
  - Attach layers via `.layer(...)` to the main router, and split out `/metrics` into an unprotected sub-router merged with `.merge(...)`.
  - Confirm path patterns match Axum extraction (`:id` is OK as used).
- Unit tests (Rust):
  - Correlation id: missing header -> generated UUID; present header -> echoed.
  - Auth: no `X-API-Key` -> 401; bad value -> 403; `valid` -> 200.
  - Error mapping: each `OrchestratorError` maps to correct status + `ErrorEnvelope` per `contracts/api-types`.

2) Bootstrap & server executable
- `app/bootstrap.rs`:
  - Build router with layers and `state::AppState` (already returns `Router`).
  - Add `init_observability()` to set a tracing subscriber (env filter, JSON logs) and ensure metrics registry is initialized.
- `src/main.rs`:
  - Switch to `#[tokio::main]` and run the Axum server: parse `ORCHD_ADDR` (default `0.0.0.0:8080`), bind listener, `axum::serve(..., app).await`.
  - Print one-line startup log with correlation id support in middleware.
- Docs:
  - README snippet: how to run locally and curl examples for `/v1/capabilities` and `/metrics` with `X-API-Key: valid`.

Definition of Done (W5)
- All middleware unit tests green; handlers no longer call `api::types::require_api_key` or `correlation_id_from` directly.
- Server binary starts and serves all v2 routes (handlers may be partial), `/metrics` accessible without API key.

-------------------------------------------------------------------------------
WEEK 6 — Session TTL/Eviction & Budgets
-------------------------------------------------------------------------------

1) SessionService implementation
- Implement `services/session.rs` with API:
  - `get_or_create(id) -> SessionInfo` (default TTL 600_000ms, turns=0, kv_bytes=0, kv_warmth=false).
  - `tick(id, now_ms)` using `ports::clock::Clock` to decrement TTL; increment `turns` on each task; evict when TTL <= 0.
  - `delete(id)` to remove a session.
- Refactor `api/data::{get_session, delete_session}` to call `SessionService` instead of manipulating `state.sessions` directly.
- Wire `SessionService` into `AppState` (or as an Arc in router state) and provide a `infra::clock::SystemClock` implementation.
- Unit tests for TTL decrement, eviction and turns accounting (use a fake clock).

2) Budgets (framework only)
- Add `domain/budget.rs` with structs: `TokensBudget`, `TimeBudget`, `CostBudget`.
- Attach budgets to `state::SessionInfo` (or a dedicated session model in `services/session.rs`).
- Surface remaining budgets in SSE metrics frames produced by `services/streaming` (for now, placeholders derived from session until adapters provide real usage).

Definition of Done (W6)
- Session unit tests green; SSE metrics include budget placeholders; handlers use `SessionService`.

-------------------------------------------------------------------------------
WEEK 7 — Adapter Registry & Streaming Determinism & Cancel
-------------------------------------------------------------------------------

1) Ports and mock infra
- Extend `ports/adapters.rs`:
  - `AdapterProps { ctx_max, supported_workloads: Vec<String> }`.
  - Define `AdapterClient` with `stream(req, cancel) -> impl Stream<Item=StreamItem>` where `StreamItem` encodes `started`, `token`, `metrics`, `end`.
  - `AdapterRegistry` returns engines, `props(engine)`, and `client(engine)`.
- Implement `infra/adapters/mock` deterministic adapter:
  - Honors `seed` for reproducible token streams and yields stable sequence and delays.
  - Expose `props()` for `ctx_max` and `supported_workloads`.
- Capabilities: optionally switch `services/capabilities::snapshot()` to build from registry; or keep static data (current `services/capabilities.rs`).

2) StreamingService
- Implement `services/streaming.rs`:
  - Translate adapter stream into ordered SSE frames: `started` (with queue position/ETA), `token*`, periodic `metrics`, then `end | error`.
  - Handle cancel tokens from `api/data::cancel_task` (no tokens after cancel). Record `tasks_canceled_total{reason="client"}`.
  - Persist transcript to `ports::storage::ArtifactStore` at `end` and return `ArtifactId` in the final frame metadata.
- Update `api/data::{stream_task,cancel_task}` to call into `StreamingService` instead of emitting static SSE strings.
- BDD: expand SSE features to assert ordering and transcript capture.

Definition of Done (W7)
- Streaming unit tests and BDD SSE scenarios green; transcript persisted via `ArtifactStore`.
- Cancel is race-free: no tokens after cancel and `tasks_canceled_total` increments.

-------------------------------------------------------------------------------
WEEK 8 — Provider Verify & Metrics Lint & Control Semantics
-------------------------------------------------------------------------------

1) Provider verification
- Add provider verify tests against `contracts/openapi/{control.yaml,data.yaml}` for implemented paths.
- Validate status codes and `contracts_api_types::ErrorEnvelope` conform to contract (happy-path and error taxonomy cases).
- Run locally via a small harness (can reuse `tools/openapi-client` helpers if available).

2) Metrics lint compliance
- Conform to `ci/metrics.lint.json` for series and labels; ensure required names exist:
  - Counters: `tasks_enqueued_total` (already emitted in `src/admission.rs`), `tasks_started_total`, `tasks_canceled_total{reason}`, `tasks_rejected_total{reason}`, `tokens_in_total`, `tokens_out_total`, `admission_backpressure_events_total` (exists), `catalog_verifications_total`.
  - Gauges: `queue_depth` (exists; add `engine_version` where applicable), `kv_cache_usage_ratio`, `gpu_utilization{device}`, `vram_used_bytes{device}`, `model_state{model_id,state}`.
  - Histograms: `latency_first_token_ms`, `latency_decode_ms` (add simple histogram aggregator to `src/metrics.rs`).
- Emit sites:
  - Admission/accept -> `tasks_enqueued_total` (exists) and set `queue_depth` with labels {engine, engine_version, pool_id, priority}.
  - Streaming start -> `tasks_started_total`; first token latency -> `latency_first_token_ms`.
  - Each decode window -> `latency_decode_ms` samples; end frame -> `tokens_out_total` and possibly `tokens_in_total` from request/session.
  - Cancel -> `tasks_canceled_total{reason}`.
  - Backpressure path -> `admission_backpressure_events_total{policy}` and `tasks_rejected_total{reason}` (already partially done in `src/admission.rs`).
- Ensure `/metrics` always includes at least one series with `engine_version` label (already set in `api/observability.rs`).

3) Control drain/reload semantics
- Track draining state per pool in `state.rs` (augment `AppState` or maintain a `HashMap<pool_id,bool>` guarded by a mutex).
- `api/control::drain_pool`: set draining flag and call `pool_managerd::registry` drain stub; return 202.
- `api/control::reload_pool`: call `pool_managerd` reload; map `ReloadOutcome::Conflict` -> 409; update `model_state{model_id,state}` gauge.
- BDD: drain begins; reload success & rollback scenarios.

Definition of Done (W8)
- Provider verify green for control/data implemented paths.
- Metrics lint passes locally.
- Control drain/reload semantics covered by BDD.

-------------------------------------------------------------------------------
ACCEPTANCE CRITERIA (Phase II)
-------------------------------------------------------------------------------
- Server binary runs with middleware and router.
- Sessions have TTL/eviction and budgets surfaced in SSE.
- Streaming deterministic on mock adapter; cancel propagation correct.
- Provider verify and metrics lint pass for implemented endpoints.
