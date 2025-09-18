# Orchestratord — Exhaustive Wiring & Testing Checklist

Status: living document
Last reviewed: 2025-09-18

Purpose
- Single source of truth to verify that `orchestratord` is fully wired to the BDD harness and unit tests, and that every specified behavior is covered.
- Use this as a pre-merge and pre-release gate.

How to use
- Check each box once verified locally or in CI.
- For each item, the checklist references code paths, BDD feature files, and tests to execute.

Quick run commands
- BDD (all features):
  - cargo run -p orchestratord-bdd --bin bdd-runner
  - env LLORCH_BDD_FEATURE_PATH=bdd/tests/features cargo run -p orchestratord-bdd --bin bdd-runner
- BDD step sanity (no undefined/ambiguous):
  - cargo test -p orchestratord-bdd -- tests::features_have_no_undefined_or_ambiguous_steps
- Unit tests (crate):
  - cargo test -p orchestratord

---

## A) Router, Middleware, and Surface Wiring

Source: `src/app/router.rs`, `src/app/middleware.rs`, `src/api/*.rs`

- [ ] Router exposes only the intended endpoints (no legacy/back-compat):
  - GET `/v1/capabilities` → `api::control::get_capabilities`
  - GET `/v1/pools/:id/health` → `api::control::get_pool_health`
  - POST `/v1/pools/:id/drain` → `api::control::drain_pool`
  - POST `/v1/pools/:id/reload` → `api::control::reload_pool`
  - POST `/v1/tasks` → `api::data::create_task`
  - GET `/v1/tasks/:id/stream` → `api::data::stream_task`
  - POST `/v1/tasks/:id/cancel` → `api::data::cancel_task`
  - GET `/v1/sessions/:id` → `api::data::get_session`
  - DELETE `/v1/sessions/:id` → `api::data::delete_session`
  - POST `/v1/artifacts` → `api::artifacts::create_artifact`
  - GET `/v1/artifacts/:id` → `api::artifacts::get_artifact`
  - GET `/metrics` → `api::observability::metrics_endpoint`
- [ ] Middleware inserts `X-Correlation-Id` on all responses and enforces `X-API-Key` on all routes except `/metrics` (`src/app/middleware.rs`).
- [ ] No legacy `/v1/replicasets` route present (spec requirement).

Validation references
- BDD security: `bdd/tests/features/security/security.feature`
- BDD control plane: `bdd/tests/features/control_plane/control_plane.feature`
- BDD data plane & SSE: under `bdd/tests/features/data_plane/*` and `bdd/tests/features/sse/*`

---

## B) Spec Alignment (High-level)

Specs: `.specs/00_orchestratord.md`, `.specs/10_orchestratord_v2_architecture.md`

- [ ] Responsibilities match implemented surface (admission, streaming, cancel, sessions, artifacts, capabilities, control)
- [ ] Determinism by default & observability hooks present in handlers/services
- [ ] No pre-1.0 back-compat (e.g., `/v1/replicasets`) exposed

---

## C) BDD Features — Scenarios Must Pass

Run: `cargo run -p orchestratord-bdd --bin bdd-runner`

1) Control plane (`bdd/tests/features/control_plane/control_plane.feature`)
- [ ] Pool health: 200 and fields: `live`, `ready`, `draining`, `metrics` → `api::control::get_pool_health`
- [ ] Drain: 202 Accepted → `api::control::drain_pool`
- [ ] Reload success atomicity: 200 → `api::control::reload_pool`
- [ ] Reload fail rollback (conflict): 409 → `api::control::reload_pool`
- [ ] Capabilities exposed: 200, `api_version`, `engines` → `api::control::get_capabilities`

2) Data plane — enqueue/stream (`bdd/tests/features/data_plane/enqueue_stream.feature`)
- [ ] Enqueue accepted: 202 with `X-Correlation-Id` and budget headers → `api::data::create_task`
- [ ] Stream events: SSE contains `started`, `token`, `end` (and metrics) → `api::data::stream_task`, `services::streaming::render_sse_for_task`

3) Data plane — cancel (`bdd/tests/features/data_plane/cancel.feature`)
- [ ] Cancel queued/running: 204 with `X-Correlation-Id` → `api::data::cancel_task`

4) Data plane — error taxonomy (`bdd/tests/features/data_plane/error_taxonomy.feature`)
- [ ] INVALID_PARAMS → 400 → `domain::error::OrchestratorError::InvalidParams`
- [ ] POOL_UNAVAILABLE → 503 → `OrchestratorError::PoolUnavailable`
- [ ] INTERNAL → 500 → `OrchestratorError::Internal`
- [ ] Error envelope includes `engine` when applicable → `domain::error.rs`

5) Data plane — sessions (`bdd/tests/features/data_plane/sessions.feature`)
- [ ] GET session: fields `ttl_ms_remaining`, `turns`, `kv_bytes`, `kv_warmth` → `api::data::get_session`
- [ ] DELETE session: 204 → `api::data::delete_session`

6) Backpressure (`bdd/tests/features/data_plane/backpressure_429.feature`)
- [ ] Queue saturation: 429 with `Retry-After`, `X-Backoff-Ms`, `X-Correlation-Id` + body advisories → `OrchestratorError::AdmissionReject{..}`

7) SSE — metrics and deadlines (`bdd/tests/features/sse/sse_details.feature`, `sse/deadlines_sse_metrics.feature`, `sse/sse_started_with_backpressure.feature`)
- [ ] SSE metrics frame present and includes `on_time_probability` → `services::streaming::render_sse_for_task`
- [ ] `started` includes `queue_position` and `predicted_start_ms`
- [ ] Event ordering per stream: `started` → `token` → `metrics` → `end`
- [ ] Deadlines: infeasible deadline rejected with `DEADLINE_UNMET` (400) → `OrchestratorError::DeadlineUnmet`
- [ ] SSE started fields still present during backpressure; 429 headers present on admission failures

8) Security (`bdd/tests/features/security/security.feature`)
- [ ] Missing API key → 401
- [ ] Invalid API key → 403

Step registry sanity (`bdd/tests/bdd.rs`)
- [ ] All Given/When/Then/And lines match exactly one step (no undefined/ambiguous)

---

## D) Provider/OpenAPI Verification (Contract Tests)

Run: `cargo test -p orchestratord -- --nocapture provider_verify`

- [ ] Control & Artifacts contract checks pass (`tests/provider_verify.rs#control_capabilities_and_artifacts_contract`)
- [ ] Data plane budget header presence for 202 and stream 200 (`tests/provider_verify.rs#data_budgets_and_sse_metrics_contract`)
- [ ] Path and status compatibility with any pacts found (`tests/provider_verify.rs#provider_paths_match_pacts`)
- [ ] Sanity rejects unknown path/status (`tests/provider_verify.rs#rejects_unknown_paths_or_statuses`)

---

## E) Unit Tests — Present and Passing

Run: `cargo test -p orchestratord`

Existing
- [ ] Admission metrics and queue depth updates (`tests/admission_metrics.rs`) — validates counters/gauges and policy label values (reject/drop-lru)
- [ ] OpenAPI provider verification suite (`tests/provider_verify.rs`)

---

## F) Unit Tests — Recommended Additions (Checklist)

Domain
- [ ] `domain/error.rs` maps each `OrchestratorError` to correct HTTP status
- [ ] `AdmissionReject` sets `Retry-After` (s) and `X-Backoff-Ms` (ms) headers when `retry_after_ms` is present
- [ ] Error envelope fields populated: `code`, `message`, `engine`, `retriable`, `retry_after_ms`, `policy_label`
- [ ] Spec-only code `REPLICA_EXHAUSTED` — decide: implement variant + mapping or remove from spec/checklist

Middleware & Types
- [ ] `correlation_id_layer` echoes/generates header and appears on all responses including errors
- [ ] `api_key_layer` requires key for all routes except `/metrics`; 401 vs 403
- [ ] `api/types.rs`: unit tests for `require_api_key()` and `correlation_id_from()`

API Handlers
- [ ] `api::data::create_task` happy path: 202 + budget headers; logs record `queue_position` and `predicted_start_ms`
- [ ] `api::data::create_task` sentinels: INVALID_PARAMS, DEADLINE_UNMET, POOL_UNAVAILABLE, INTERNAL, ADMISSION_REJECT
- [ ] `api::data::stream_task` sets `Content-Type: text/event-stream` and budget headers; body contains 4 expected events
- [ ] `api::data::cancel_task` increments `tasks_canceled_total` with reason label; 204
- [ ] `api::control::{get_pool_health, drain_pool, reload_pool}` response codes and body fields; 409 rollback case
- [ ] `api::artifacts::{create_artifact,get_artifact}`: 201 with id kind; 200/404 on get
- [ ] `api::observability::metrics_endpoint` returns Prometheus text and seeds all required series

Services
- [ ] `services::session`: `get_or_create` defaults; `tick` eviction when TTL ≤ 0; `delete` removes; `note_turn` increments
- [ ] `services::streaming::render_sse_for_task`: event order, metrics increments, transcript persisted via `services::artifacts::put`
- [ ] `services::artifacts`: `put` mirrors into `state.artifacts`, `get` checks store first then memory
- [ ] `services::capabilities::snapshot` shape includes `api_version` and expected engines

Admission & Metrics
- [ ] `admission::QueueWithMetrics` updates `queue_depth` gauge
- [ ] Policy Reject → increments `admission_backpressure_events_total{policy="reject"}` and `tasks_rejected_total{reason="ADMISSION_REJECT"}`
- [ ] Policy DropLru → increments `admission_backpressure_events_total{policy="drop-lru"}` and `tasks_rejected_total{reason="QUEUE_FULL_DROP_LRU"}` while enqueuing second item

Infra Storage
- [ ] `infra::storage::inmem::InMemStore` assigns `sha256:<hex>` id and round-trips JSON
- [ ] `infra::storage::fs::FsStore` writes to default root (configurable via `ORCH_ARTIFACTS_FS_ROOT`), round-trips JSON, id matches content hash

State
- [ ] `state::AppState::new()` wires default `InMemStore`, `PoolRegistry`, and empty maps

---

## G) SSE Protocol and Budgets — Deep Checklist

- [ ] Response headers on GET `/v1/tasks/:id/stream`: `Content-Type: text/event-stream`, `X-Correlation-Id`, budget headers present
- [ ] Event order: `started` → `token` (one or more) → `metrics` (after first token) → `end` (and optionally `error` if failure)
- [ ] `started` contains `queue_position`, `predicted_start_ms`
- [ ] `metrics` contains at least: `queue_depth`, `on_time_probability`, `kv_warmth`, `tokens_budget_remaining`, `time_budget_remaining_ms`, `cost_budget_remaining`
- [ ] Transcript persisted as artifact with tags/metadata (currently simplified; verify presence of persisted SSE doc)
- [ ] Cancel semantics: no tokens after cancel (add BDD scenario — see Section I)

---

## H) Observability & Metrics — Coverage

- [ ] `/metrics` responds 200 with Prometheus v0.0.4 content type and `X-Correlation-Id`
- [ ] Counters incremented: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total{reason}`, `tasks_rejected_total{reason}`, `admission_backpressure_events_total{policy}`, `tokens_in_total`, `tokens_out_total`, `catalog_verifications_total{result,reason}`
- [ ] Gauges updated: `queue_depth{engine,engine_version,pool_id,priority}`, `kv_cache_usage_ratio{...}`, `gpu_utilization{...}`, `vram_used_bytes{...}`, `model_state{model_id,state}`
- [ ] Histograms observed: `latency_first_token_ms{...}`, `latency_decode_ms{...}`
- [ ] Metrics linter compliance against `ci/metrics.lint.json` (run via workspace harness if available)

---

## I) Proposed Additional BDD Scenarios (to reach exhaustive coverage)

Artifacts
- [ ] Persisted SSE transcript: after streaming, fetch the latest artifact and assert presence of `events` with `started`/`token`/`metrics`/`end`

Admission/Backpressure
- [ ] Policy Reject vs Drop-LRU: distinct scenarios validating reason codes in error envelope (`ADMISSION_REJECT` vs `QUEUE_FULL_DROP_LRU`)

Cancel Semantics
- [ ] Race-free cancel: start streaming, issue `POST /v1/tasks/:id/cancel`, assert no further `token` events are emitted

Budgets
- [ ] Budget headers on POST `/v1/tasks` (202) and stream (200) reflect session defaults and update after stream end (tokens/time/cost spend)

Security/Observability
- [ ] Logs (internal `state.logs`) include `queue_position` and `predicted_start_ms` on admission; logs contain no API keys

Error Event
- [ ] SSE `error` event emits taxonomy with `code` and `engine` and terminates stream

Capabilities
- [ ] Capability snapshot includes concurrency when available (spec note); otherwise documented as TODO

---

## J) Spec: Error Taxonomy Coverage Map

Spec codes (see `.specs/00_orchestratord.md#14-error-taxonomy`)
- [ ] INVALID_PARAMS → BDD: `error_taxonomy.feature`
- [ ] DEADLINE_UNMET → BDD: `deadlines_sse_metrics.feature`
- [ ] POOL_UNAVAILABLE → BDD: `error_taxonomy.feature`
- [ ] INTERNAL → BDD: `error_taxonomy.feature`
- [ ] ADMISSION_REJECT → BDD: `backpressure_429.feature`
- [ ] QUEUE_FULL_DROP_LRU → BDD: [ADD scenario]
- [ ] REPLICA_EXHAUSTED → Spec-only (not implemented): [DECIDE implement or remove]

---

## K) Session Lifecycle & Budgets

- [ ] `GET /v1/sessions/:id` returns all fields, including budgets
- [ ] `DELETE /v1/sessions/:id` removes the session
- [ ] `SessionService::tick()` evicts zero-TTL sessions [unit test]
- [ ] `SessionService::note_turn()` increments `turns` on accept [unit test]

---

## L) Control Plane & Lifecycle

- [ ] Drain starts (202) and reflected in health `draining` flag
- [ ] Reload success (200) updates `model_state` gauge
- [ ] Reload conflict (409) leaves prior state (atomicity)
- [ ] Health shape includes `last_error` where applicable

---

## M) CI/Automation Gates

- [ ] All BDD features pass in CI
- [ ] `features_have_no_undefined_or_ambiguous_steps` test passes (step registry complete)
- [ ] All unit tests pass (`orchestratord` crate)
- [ ] Metrics linter passes (`ci/metrics.lint.json`)
- [ ] Provider verification tests pass

---

## Appendix: Cross-References

Key files
- Router: `src/app/router.rs`
- Middleware: `src/app/middleware.rs`
- Data handlers: `src/api/data.rs`
- Control handlers: `src/api/control.rs`
- Artifacts handlers: `src/api/artifacts.rs`
- Observability handlers: `src/api/observability.rs`
- Error mapping: `src/domain/error.rs`
- SSE service: `src/services/streaming.rs`
- Session service: `src/services/session.rs`
- Artifact service: `src/services/artifacts.rs`
- Admission queue & metrics: `src/admission.rs`
- Storage: `src/infra/storage/{inmem,fs}.rs`
- State: `src/state.rs`

BDD harness
- Features: `bdd/tests/features/**/*.feature`
- Step registry: `bdd/src/steps/mod.rs`
- Steps: `bdd/src/steps/*.rs`
- World & HTTP harness: `bdd/src/steps/world.rs`
- Sanity test: `bdd/tests/bdd.rs`

Contract tests
- Provider verify: `tests/provider_verify.rs`
- Admission metrics: `tests/admission_metrics.rs`

Notes
- This checklist is generated from the current codebase and specs. Keep it updated when specs or routes change.
