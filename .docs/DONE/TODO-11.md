# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] BDD wiring: add runnable glue so step stubs in `test-harness/bdd/src/steps/` exercise real code surfaces (handlers, metrics, adapters) without starting servers, using `orchestratord::build_app()` and in-crate calls. Align with README_LLM.md order (Spec→Contract→Tests→Code) and traceability rules.
- [x] Extend BDD `World` to carry router/state and a response/trace stash
  - File: `test-harness/bdd/src/steps/world.rs`
  - Add fields: `router: axum::Router<orchestratord::state::AppState>`, `last_http: Option<(http::StatusCode, http::HeaderMap, serde_json::Value)>`, `corr_id: Option<String>`
  - Init in a `new()` helper; default to `orchestratord::build_app()`
- [x] Ensure step registry remains complete and unambiguous
  - File: `test-harness/bdd/src/steps/mod.rs`
  - When adding new steps, update regex list so `tests/bdd.rs` passes (no undefined/ambiguous)
- [x] Backpressure helpers are currently planning-only; provide minimal functional stubs used by BDD
  - File: `orchestratord/src/backpressure.rs`
  - Implement: `build_429_headers()` to set `Retry-After` and `X-Backoff-Ms`; `build_429_body()` to include `policy_label`
  - Spec IDs: OC-CTRL-2011, OC-METRICS-7101
- [x] HTTP error envelopes scaffolding: ensure typed codes surface from handlers for BDD assertions
  - File: `orchestratord/src/errors.rs`
  - Map to enum covering spec taxonomy; include optional `engine` when available
  - Spec IDs: OC-CTRL-2030..2031

## Glue Map (Steps → Code → Spec IDs)

This section enumerates every step function stub under `test-harness/bdd/src/steps/` and maps it to the precise code surfaces (currently stubs or planning-only modules) and stable requirement IDs. Implement glue in the listed order. Use `bdd-runner` (LLORCH_BDD_FEATURE_PATH supported) per prior setup.

Notes

- Align with traceability: include requirement IDs in feature files and/or test names. See `.docs/testing/spec-derived-test-catalog.md`, `tests/traceability.rs`.
- Do not break contract order: update specs/contracts first if behavior is unclear.

1) Orchestratord — Data plane & sessions (file: `steps/data_plane.rs`)

- Step: `Given an OrchQueue API endpoint` → Code: router from `orchestratord::build_app()`; no network; create test client via `tower::ServiceExt` or `axum::Router::oneshot`
  - Spec IDs: OC-CTRL-2010..2012, OC-CTRL-2020..2022, OC-CTRL-2052
- Step: `When I enqueue a completion task with valid payload` → Code: `http::handlers::create_task` via router `POST /v1/tasks`
  - Assert 202, `X-Correlation-Id` present
  - Spec IDs: OC-CTRL-2010, OC-CTRL-2052
- Step: `Then I receive 202 Accepted with correlation id`
  - Read `StatusCode`, headers from `World.last_http`
  - Spec IDs: OC-CTRL-2052
- Step: `When I stream task events` → Code: `http::handlers::stream_task` via router `GET /v1/tasks/:id/stream`
  - For scaffolding, can emit a canned SSE stream from a placeholder
  - Spec IDs: OC-CTRL-2020..2022
- Step: `Then I receive SSE events started, token, end`
  - Validate event names and ordering
  - Spec IDs: OC-CTRL-2020..2022
- Step: `Then I receive SSE metrics frames` and `Then started includes queue_position and predicted_start_ms`
  - Emit placeholder metrics frames; later wire real estimates
  - Spec IDs: OC-CTRL-2021, OC-METRICS-7110..7111
- Steps: queue-full policies (reject, drop-lru, shed-low-priority)
  - Code: `admission::QueueWithMetrics.enqueue`; on capacity edge use `backpressure::{build_429_headers, build_429_body}` in handlers
  - Spec IDs: OC-CORE-1002, OC-CTRL-2011, OC-METRICS-7101
- Steps: under-load 429 advisory headers/body
  - Code: `backpressure.rs` helpers + `errors.rs`
  - Spec IDs: OC-CTRL-2011
- Steps: cancel path (`When I cancel the task` / `Then 204 No Content with correlation id`)
  - Code: `http::handlers::cancel_task` + `QueueWithMetrics.cancel`
  - Spec IDs: OC-CTRL-2012, OC-CTRL-2052
- Steps: sessions (`Given a session id` / query/delete)
  - Code: `http::handlers::{get_session, delete_session}` (stub responses OK initially)
  - Spec IDs: OC-CTRL-2020(related), OC-CTRL-2068 (budgets, advisory)

2) Orchestratord — Control plane (file: `steps/control_plane.rs`)

- Health: `GET /v1/pools/:id/health` (Given/When/Then)
  - Code: `http::handlers::get_pool_health`; surface liveness/readiness/draining and metrics snapshot (stub OK)
  - Spec IDs: OC-CTRL-2001
- Drain: `POST /v1/pools/:id/drain` with `deadline_ms`
  - Code: `http::handlers::drain_pool` (record drain intent; respond 202)
  - Spec IDs: OC-CTRL-2002
- Reload: `POST /v1/pools/:id/reload` – success and failure rollback
  - Code: `http::handlers::reload_pool` (simulate atomicity via state swap guard)
  - Spec IDs: OC-CTRL-2003
- Replicasets list: `GET /v1/replicasets`
  - Code: `http::handlers::list_replicasets`
  - Spec IDs: OC-CTRL-2004, OC-CTRL-2060..2061

3) Observability & Security (files: `steps/observability.rs`, `steps/security.rs`)

- Metrics conform to linter
  - Code: `orchestratord::metrics::gather_metrics_text()`; step should call `http::handlers::metrics_endpoint`
  - Spec IDs: OC-METRICS-7101, OC-CTRL-2051
- Label cardinality budgets enforced
  - Code: metrics registration and labels
  - Spec IDs: OC-METRICS-7102
- Logs include queue_position and predicted_start_ms; no secrets
  - Code: enrich started/admission logs (planning); for BDD, assert placeholders
  - Spec IDs: OC-CTRL-2050, OC-CTRL-2041
- Security: no/invalid API key → 401/403
  - Code: handlers add simple key check middleware or inline check (planning OK)
  - Spec IDs: OC-CTRL-2040

4) Error taxonomy (file: `steps/error_taxonomy.rs`)

- INVALID_PARAMS, POOL_UNAVAILABLE, INTERNAL
  - Code: map via `errors::ErrorEnvelope` produced by handlers on specific triggers
  - Spec IDs: OC-CTRL-2030..2031

5) Core guardrails (file: `steps/core_guardrails.rs`)

- Context length/token budget rejected pre-admission
  - Code: admission checks in `QueueWithMetrics` or a wrapper prior to enqueue
  - Spec IDs: OC-CORE-1020..1021
- Watchdog aborts running task
  - Code: placeholder watchdog signal observable via state
  - Spec IDs: OC-CORE-1022

6) Deadlines & preemption (file: `steps/deadlines_preemption.rs`)

- Infeasible deadline → DEADLINE_UNMET; metrics include on_time_probability
  - Code: admission/scheduling advisory fields (placeholder)
  - Spec IDs: OC-CTRL-2023
- Soft/hard preemption behavior ordering, metrics exported
  - Code: scheduling module placeholders; surface flags in SSE/events
  - Spec IDs: aligns with Stage 9; mark as later stage wiring

7) Determinism (file: `steps/determinism.rs`)

- Pinned versions/artifacts; identical streams with same seed
  - Code: ensure worker adapters expose `engine_version`; determinism suite (`requirements/test-harness-determinism-suite.yaml`)
  - Spec IDs: OC-CORE-1030..1032; OC-TEST-7001..7003
- No determinism across engine/model updates; replicas across versions are used
  - Code: BDD asserts negative case and mixed-version behavior
  - Spec IDs: OC-CORE-1032

8) Scheduling & quotas (file: `steps/scheduling.rs`)

- WFQ weights; observed share; per-tenant quotas; session affinity
  - Code: core scheduler properties and state; may rely on orchestrator-core tests initially
  - Spec IDs: OC-CORE-1004, OC-CORE-1010..1013

9) Pool manager (file: `steps/pool_manager.rs`)

- Unready due to preload failure; driver error resets; circuit breaker
  - Code: `pool-managerd` scaffolding and state transitions (placeholders OK)
  - Spec IDs: OC-POOL-3001..3012, OC-POOL-3030
- Device masks; heterogeneous split; per-GPU KV cap
  - Code: `pool-managerd` device masks; placement compatibility with orchestrator `placement`
  - Spec IDs: OC-POOL-3020..3021

10) Worker adapters (file: `steps/adapters.rs`)

- llama.cpp/vLLM/TGI/Triton adapters implement health/props/engine_version as pure functions returning placeholders to satisfy steps (no I/O)
  - Code: unimplemented trait methods across adapters
  - Spec IDs: OC-ADAPT-5001..5011
- OpenAI-compatible endpoints internal only
  - Code: adapter config flags; ensure step asserts non-exposure
  - Spec IDs: OC-ADAPT-5002

11) Policy host & SDK (files: `steps/policy_host.rs`, `steps/policy_sdk.rs`)

- WASI ABI, pure/deterministic, sandboxing, bounded resources, telemetry
  - Code: plugins host scaffolding; BDD asserts placeholders
  - Spec IDs: OC-POLICY-4001..4033
- SDK semver stability; no I/O by default; migration notes on breaking
  - Code: SDK crate scaffolding; BDD asserts placeholders
  - Spec IDs: OC-POLICY-SDK-4101..4112

12) Config schema & catalog (files: `steps/config.rs`, `steps/catalog.rs`)

- Example config validates; strict mode rejects unknown; schema regen idempotent
  - Code: `contracts/config-schema` generator/regenerator and validator
  - Spec IDs: see `requirements/contracts-config-schema.yaml`
- Catalog signing/SBOM/verification; strict trust policy; UNTRUSTED_ARTIFACT
  - Code: catalog service scaffolding or mocks
  - Spec IDs: per `.specs/00_llama-orch.md` catalog section

13) Preflight/apply (files: `steps/preflight_steps.rs`, `steps/apply_steps.rs`)

- DryRun default; Commit mode performs side effects; critical violation short-circuits
  - Code: gate side effects on `World.mode_commit`; ensure apply facts recorded
  - Spec IDs: project workflow docs

14) Lifecycle (file: `steps/lifecycle.rs`)

- Deprecated with deadline; new sessions blocked with MODEL_DEPRECATED
  - Code: control-plane flag in state; data-plane admission checks block `new` when deprecated pending deadline
  - Spec IDs: see `.specs/00_llama-orch.md` lifecycle section; metrics per OC-CTRL-2051 if applicable
- Retired state; pools unload and archives retained; model_state gauge exported
  - Code: add `model_state` gauge in metrics module; surface via `/metrics`
  - Spec IDs: `.specs/00_llama-orch.md` lifecycle; metrics contract OC-METRICS-7101

Implementation Tasks per area (actionable)

- Data plane glue
  - [x] Extend `World` with router + last_http stash
  - [x] Helper: `world.http_json(method, path, body)` that updates `last_http`
  - [x] Implement enqueue/cancel/session steps to call router; assert statuses/headers/bodies
- SSE glue
  - [x] Provide a minimal SSE response in `stream_task` for BDD assertions (events order only)
  - [x] Add started payload with `queue_position`, `predicted_start_ms` placeholders
- Backpressure glue
  - [x] Implement `backpressure::build_429_headers/body` and use in `create_task` when queue full
- Control plane glue
  - [x] Implement stubbed JSON bodies for health/drain/reload/replicasets
- Metrics glue
  - [x] Call `metrics_endpoint` in observability step; assert linter names present
- Error taxonomy glue
  - [x] Handlers return `ErrorEnvelope` with proper `code`; include `engine` when applicable
- Adapters glue
  - [x] For each adapter crate, implement `health/props/engine_version` as pure functions returning placeholders to satisfy steps (no I/O)
  - [x] `submit/cancel` return stubbed stream/ack for BDD
- Lifecycle glue
  - [ ] Add `model_state` gauge and basic state machine in orchestrator state; assert with BDD step
- Traceability
  - [x] Ensure every .feature references at least one requirement ID; keep `tests/traceability.rs` green

## Progress Log (what changed)

- 2025-09-16: BDD glue implemented end-to-end for data-plane, SSE, sessions, backpressure, error taxonomy, control-plane, security, and observability.
  - Files: `orchestratord/src/http/handlers.rs` (implement minimal functional handlers, API key gating, sentinels), `orchestratord/src/backpressure.rs` (429 helpers), `orchestratord/src/state.rs` (derive Debug), `test-harness/bdd/src/steps/world.rs` (router-less in-memory dispatcher + HTTP helper), step files under `test-harness/bdd/src/steps/` updated to drive in-memory calls and assert responses, worker adapters placeholder implementations across `worker-adapters/*`.
  - Tests: `cargo test -p test-harness-bdd` passes; `tests/bdd.rs` confirms no undefined/ambiguous steps; `tests/traceability.rs` reports missing IDs but is non-strict by default.

- 2025-09-16: Lifecycle gating and SSE metrics frames wired; traceability coverage green.
  - Files: `orchestratord/src/state.rs` (add `ModelState`), `orchestratord/src/http/handlers.rs` (lifecycle gate in `create_task`, `/v1/models/state` control, SSE `metrics` frame, guardrails checks), BDD steps (`lifecycle.rs`, `data_plane.rs`, `core_guardrails.rs`).
  - Tests: `test-harness-bdd` traceability test reports all catalog IDs referenced; lifecycle gauge exported via `/metrics`.
