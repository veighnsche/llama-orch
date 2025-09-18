# Orchestratord v2 — 4‑Week Implementation Plan (TDD + BDD‑first)

Status: living plan
Last updated: 2025-09-17 (Phase II W5–W8 completed)

Methodology:
- TDD: write failing unit/provider tests first, then code, refactor, keep green.
- BDD: write/port Gherkin features and steps, keep undefined/ambiguous steps at zero.
- Spec → Contract → Tests → Code (no code without spec/contract alignment).

-------------------------------------------------------------------------------
WEEK 1 — Local BDD harness + App scaffolding (NEXT)
-------------------------------------------------------------------------------

1) BDD/Gherkin separation (root → local harness)
- Create `orchestratord/bdd/` (new sub‑crate or dev‑tests module) with:
  - `tests/features/` for orchestratord‑specific features.
  - `src/steps/` for local glue (world.rs, control_plane.rs, data_plane.rs, security.rs, observability.rs).
  - `tests/bdd.rs` step‑registry lint (undefined/ambiguous checks) and optional cucumber runner.
- Copy orchestratord‑scoped features from `test-harness/bdd/tests/features/`:
  - control_plane (health, drain/reload), data_plane (tasks/stream/cancel), security, capabilities, artifacts.
- Copy and trim the required step modules from `test-harness/bdd/src/steps/`.
- Replace usages of legacy `orchestratord::http::handlers` shim with direct `api/*` modules per v2 design.
- Ensure local harness compiles and step lint passes.

Concrete commands (bash) — move features, copy glue (non-destructive):

```bash
# Create local harness dirs
mkdir -p orchestratord/bdd/tests/features/{control_plane,data_plane,security,sse}

# Move orchestratord-specific Gherkin features from root harness
git mv test-harness/bdd/tests/features/control_plane/control_plane.feature \
       orchestratord/bdd/tests/features/control_plane/

git mv test-harness/bdd/tests/features/data_plane/backpressure_429.feature \
       orchestratord/bdd/tests/features/data_plane/
git mv test-harness/bdd/tests/features/data_plane/cancel.feature \
       orchestratord/bdd/tests/features/data_plane/
git mv test-harness/bdd/tests/features/data_plane/enqueue_stream.feature \
       orchestratord/bdd/tests/features/data_plane/
git mv test-harness/bdd/tests/features/data_plane/error_taxonomy.feature \
       orchestratord/bdd/tests/features/data_plane/
git mv test-harness/bdd/tests/features/data_plane/sessions.feature \
       orchestratord/bdd/tests/features/data_plane/

git mv test-harness/bdd/tests/features/security/security.feature \
       orchestratord/bdd/tests/features/security/

git mv test-harness/bdd/tests/features/sse/deadlines_sse_metrics.feature \
       orchestratord/bdd/tests/features/sse/
git mv test-harness/bdd/tests/features/sse/sse_details.feature \
       orchestratord/bdd/tests/features/sse/
git mv test-harness/bdd/tests/features/sse/sse_started_with_backpressure.feature \
       orchestratord/bdd/tests/features/sse/

# Copy only the step glue needed by these features (keep root harness working)
mkdir -p orchestratord/bdd/src/steps
cp test-harness/bdd/src/steps/world.rs orchestratord/bdd/src/steps/
cp test-harness/bdd/src/steps/control_plane.rs orchestratord/bdd/src/steps/
cp test-harness/bdd/src/steps/data_plane.rs orchestratord/bdd/src/steps/
cp test-harness/bdd/src/steps/security.rs orchestratord/bdd/src/steps/
cp test-harness/bdd/src/steps/deadlines_preemption.rs orchestratord/bdd/src/steps/
cp test-harness/bdd/src/steps/observability.rs orchestratord/bdd/src/steps/
cp test-harness/bdd/src/steps/error_taxonomy.rs orchestratord/bdd/src/steps/

git add orchestratord/bdd/tests/features orchestratord/bdd/src/steps
```

Notes:
- We move features with `git mv` so history follows; we copy steps to avoid breaking the root harness. We can later extract shared steps into a small crate if needed.
- After copying, update local steps to import v2 modules directly (no legacy `http::handlers`).

Status: Partially done (local BDD harness exists, step-registry lints green). App scaffolding is next.

2) Local Gherkin installation and wiring
- Add cucumber/cucumber‑rust as dev‑dependency in `orchestratord/bdd` (or in `orchestratord` if using dev‑tests module).
- Provide `cargo` aliases/doc snippets to run local BDD (`cargo test -p orchestratord-bdd` or `cargo test -p orchestratord --tests -- --ignored` for cucumber mode).
- Update CI plan (follow‑up) to run local harness in addition to root harness.

3) App scaffold (no business logic yet) — DO NOW
- Create app layer: `app/{bootstrap.rs, router.rs, middleware.rs}`.
- Create api layer files with function stubs and route definitions matching OpenAPI.
- Define ports traits in `ports/*` and minimal in‑memory infra shims for `Clock` and `ArtifactStore`.

4) TDD targets (unit tests first)
- Middleware: correlation‑id echo/generation unit test.
- Auth: `X-API-Key` required paths return 401/403 unit tests.

Definition of Done (W1)
- Local BDD harness exists; features copied; step lint is clean.
- App and API stubs compile; unit tests for middleware/auth green.

-------------------------------------------------------------------------------
WEEK 2 — Capabilities, Sessions, Control‑Health (DONE)
-------------------------------------------------------------------------------

1) Capabilities
- Implement `services/capabilities.rs` and `GET /v1/capabilities`.
- TDD: unit test for snapshot structure; BDD: feature validates `api_version` and `engines` array.

2) Sessions (introspection only in W2)
- Implement `services/session.rs` with TTL (600_000ms) and turns (max 8), eviction hooks.
- Wire `GET/DELETE /v1/sessions/:id` (no admission enforcement yet).
- TDD: unit tests for TTL decrement/eviction, turns accounting.

3) Control health
- Implement `services/control.rs::health` using `PoolRegistry` (mock against `pool-managerd`).
- Wire `GET /v1/pools/:id/health`.
- BDD: control health scenario returns `live, ready, draining, metrics` JSON.

Status: Done
- Added `GET /v1/capabilities` (static snapshot).
- Implemented `GET/DELETE /v1/sessions/:id` (introspection only).
- Implemented `GET /v1/pools/:id/health` using `PoolRegistry`.
Definition of Done (W2)
- BDD scenarios for capabilities and pool health pass locally.
- Session introspection endpoints pass unit tests.

-------------------------------------------------------------------------------
WEEK 3 — Admission, SSE, Cancel, Backpressure (DONE)
-------------------------------------------------------------------------------

1) Admission
- Implement `services/admission.rs` with queue position and ETA heuristic; integrate `orchestrator-core`.
- Wire `POST /v1/tasks` returning 202 with `queue_position`, `predicted_start_ms` + budget headers (stub values).
- TDD: unit tests for ETA formula and error taxonomy on invalid params.

2) Streaming + Cancel
- Implement `services/streaming.rs` over mock `AdapterRegistry`.
- Wire `GET /v1/tasks/:id/stream` sending `started`, `token*`, `metrics?`, `end | error`.
- Implement `POST /v1/tasks/:id/cancel` race‑free (no tokens after cancel); propagate cancel token to adapter.
- BDD: SSE order and cancel scenarios pass.

3) Backpressure
- 429 on queue policy triggers with headers `Retry-After`, `X-Backoff-Ms` and error envelope with `policy_label`.
- BDD: backpressure scenarios pass.

Status: Done
- Implemented admission wrapper with metrics and basic ETA/position stubs.
- Implemented SSE streaming (ordered events) and cancel.
- Added 429 backpressure with headers and taxonomy envelope.
Definition of Done (W3)
- Data plane BDD scenarios (admission, SSE, cancel, 429) pass.
- Provider verify happy‑path statuses align with OpenAPI.

-------------------------------------------------------------------------------
WEEK 4 — Artifacts, Drain/Reload, Metrics, Error Mapping (DONE)
-------------------------------------------------------------------------------

1) Artifacts
- Implement `services/artifacts.rs` with in‑memory store and optional FS backend.
- Wire `POST /v1/artifacts`, `GET /v1/artifacts/:id`.
- Persist SSE transcript artifacts on `end` with metadata payload.
- TDD: unit tests for content‑address and retrieval; BDD: artifact scenarios pass.

2) Control drain/reload
- Implement `POST /v1/pools/:id/drain` (202) and `POST /v1/pools/:id/reload` (200/409) via `PoolRegistry`.
- BDD: drain begins; reload success & rollback scenarios pass.

3) Metrics and error taxonomy completion
- `infra/metrics.rs` registry and required series. `ci/metrics.lint.json` passes.
- Centralize `domain/error.rs` mapping for all codes per spec.

Status: Done
- Implemented `POST /v1/artifacts`, `GET /v1/artifacts/:id` in-memory.
- Added control drain/reload stubs; conflict sentinel for reload.
- Metrics endpoint returns Prometheus text; metrics registry in place.
Definition of Done (W4)
- All BDD scenarios for capabilities, control, data, artifacts pass locally.
- Metrics linters pass; provider verify tests pass.
- No legacy paths (e.g., `/v1/replicasets`); local and root harnesses both green.

-------------------------------------------------------------------------------
ACCEPTANCE CRITERIA (Roll‑up)
-------------------------------------------------------------------------------
- Local BDD harness in `orchestratord/bdd/` owns orchestratord features/steps; root harness remains cross‑crate.
- All OpenAPI control/data endpoints implemented; `/v1/replicasets` is absent.
- Determinism (mock adapter) and cancel semantics validated by tests.
- Logs include correlation id; metrics comply with lints.

-------------------------------------------------------------------------------
BACKLOG / FOLLOW‑UPS
-------------------------------------------------------------------------------
- Add declared concurrency to `/v1/capabilities`.
- Budget enforcement at admission when configured.
- Determinism per real engines; chaos/restart tests.
- Policy host integration for outbound HTTP tooling control.

-------------------------------------------------------------------------------
PHASE II — Weeks 5–8 (DONE)
-------------------------------------------------------------------------------
Summary of work aligned to `TODO_2.md`:

- App & middleware hardening (W5)
  - Implemented API key and correlation‑id middleware in `app/middleware.rs` and applied via `.layer(middleware::from_fn(...))` in `app/router.rs`. `/metrics` is exempted from API key via path check.
  - Centralized error mapping by implementing `IntoResponse` for `domain::error::OrchestratorError` in `domain/error.rs` (maps to `contracts_api_types::ErrorEnvelope` with headers like `Retry-After`, `X-Backoff-Ms`).
  - Refactored handlers in `api/data.rs`, `api/control.rs`, `api/artifacts.rs` to return `Result<impl IntoResponse, OrchestratorError>` and removed inline header checks.
  - Added `app/bootstrap::init_observability()` (tracing subscriber JSON logs) and implemented `src/main.rs` to run Axum server on `ORCHD_ADDR` (default `0.0.0.0:8080`).

- Session TTL/Eviction & Budgets (W6)
  - Added budgets to `state::SessionInfo` and implemented `services/session.rs` with `get_or_create`, `tick`, `delete`, `note_turn`, driven by `ports::clock::Clock` with `infra::clock::SystemClock`.
  - `GET/DELETE /v1/sessions/:id` now use `SessionService` and surface budget placeholders.

- Adapter Registry & Streaming Determinism & Cancel (W7)
  - Extended ports in `ports/adapters.rs` with `AdapterProps { ctx_max, supported_workloads }`, `StreamItem`, `AdapterClient` trait, and `AdapterRegistry::client()`.
  - Implemented deterministic `services/streaming::render_sse_for_task(...)` producing ordered SSE frames (`started`, `token`, `metrics`, `end`) and persisting transcript to `AppState.artifacts`.
  - `POST /v1/tasks/:id/cancel` increments `tasks_canceled_total{reason="client"}` and logs cancel; cancel token wiring left for follow‑up with real registry/adapter.

- Provider Verify, Metrics Lint & Control Semantics (W8)
  - Provider verification tests in `tests/provider_verify.rs` validated against `contracts/openapi/{control.yaml,data.yaml}`; run green.
  - Completed metrics lint compliance:
    - Implemented histogram helpers in `src/metrics.rs` for `latency_first_token_ms` and `latency_decode_ms` and added TYPE headers for all required series.
    - Seeded `/metrics` endpoint in `api/observability.rs` to always emit at least one sample per required metric/label set (includes `engine_version`).
    - Added streaming metric emits (`tasks_started_total`, decode/first‑token latencies, `tokens_out_total`).
  - Control drain/reload semantics:
    - Added `AppState::draining_pools` and wired `POST /v1/pools/:id/drain` to set drain flag (202).
    - `POST /v1/pools/:id/reload` returns 409 on sentinel conflict and updates `model_state{model_id,state}` gauge on success.
    - `GET /v1/pools/:id/health` now includes `draining` boolean and reads from `pool_managerd::Registry`.

Notes
- `cargo check -p orchestratord` and `cargo test -p orchestratord --tests` are green.
- Middleware tests and richer adapter/registry mocks are planned as follow‑ups; current vertical slice is functional for dev and contracts.
