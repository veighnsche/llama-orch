# Orchestratord v2 — 4‑Week Implementation Plan (TDD + BDD‑first)

Status: living plan
Last updated: 2025-09-17

Methodology:
- TDD: write failing unit/provider tests first, then code, refactor, keep green.
- BDD: write/port Gherkin features and steps, keep undefined/ambiguous steps at zero.
- Spec → Contract → Tests → Code (no code without spec/contract alignment).

-------------------------------------------------------------------------------
WEEK 1 — Local BDD harness + App scaffolding
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

2) Local Gherkin installation and wiring
- Add cucumber/cucumber‑rust as dev‑dependency in `orchestratord/bdd` (or in `orchestratord` if using dev‑tests module).
- Provide `cargo` aliases/doc snippets to run local BDD (`cargo test -p orchestratord-bdd` or `cargo test -p orchestratord --tests -- --ignored` for cucumber mode).
- Update CI plan (follow‑up) to run local harness in addition to root harness.

3) App scaffold (no business logic yet)
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
WEEK 2 — Capabilities, Sessions, Control‑Health
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

Definition of DoN (W2)
- BDD scenarios for capabilities and pool health pass locally.
- Session introspection endpoints pass unit tests.

-------------------------------------------------------------------------------
WEEK 3 — Admission, SSE, Cancel, Backpressure
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

Definition of Done (W3)
- Data plane BDD scenarios (admission, SSE, cancel, 429) pass.
- Provider verify happy‑path statuses align with OpenAPI.

-------------------------------------------------------------------------------
WEEK 4 — Artifacts, Drain/Reload, Metrics, Error Mapping
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
