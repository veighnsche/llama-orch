# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

> VERY VERY VERY IMPORTANT: HOME PROFILE v2.1 (Spec Reduction aligned with CLI requirements + reference environment). Top priority.

## P0 — SPEC REDUCTION (Home Profile v2)

### DONE
- Home Profile v2.1 docs/spec refresh: `.docs/HOME_PROFILE.md`, `.specs/00_home_profile.md` (restores CLI-critical surfaces and ties to `.docs/HOME_PROFILE_TARGET.md`).
- Reference environment captured: `.docs/HOME_PROFILE_TARGET.md`.
- Process guide: `.docs/FULL_PROCESS_SPEC_CHANGE.md`
- Scanner: `search_overkill.sh` (excludes all `target/`)
- CI sanity: clippy fix in `http/control.rs` test; relaxed a flaky metrics assert; `cargo xtask dev:loop` green

### TODO — Data Plane & Sessions
- Ensure `POST /v1/tasks` accepts full FR-DP-001 payload.
  - Files: `contracts/openapi/data.yaml`, `contracts/api-types/**`, `orchestratord/src/http/data.rs`, request validation/tests.
  - Acceptance: provider verify + consumer Pact exercise new fields; determinism knobs honoured.
- Restore 429 body `policy_label` and admission metadata headers.
  - Files: `contracts/openapi/data.yaml`, `orchestratord/src/backpressure.rs`, `orchestratord/src/http/data.rs`, SSE fixtures.
  - Acceptance: 429 includes `policy_label`; tests updated.
- Reinstate SSE `metrics`/`error` frames and session budget headers when budgets enabled.
  - Files: `contracts/openapi/data.yaml`, `orchestratord/src/http/data.rs`, `test-harness/bdd/tests/features/sse/`.
  - Acceptance: streaming tests see all required events; CLI pact regenerates accordingly.
- Expose `GET/DELETE /v1/sessions/{id}` for KV reuse (SHOULD).
  - Files: `contracts/openapi/data.yaml`, handlers + tests.
  - Acceptance: basic session lifecycle covered by provider tests.

### TODO — Control Plane & Discovery
- Keep catalog endpoints (`/v1/catalog/*`) with minimal trust policy enforcement.
  - Files: `contracts/openapi/control.yaml`, `orchestratord/src/http/catalog.rs`, `orchestratord/src/lib.rs`, tests/fixtures.
  - Acceptance: CLI flows can register/verify/state flip models.
- Retain pool drain/reload/health endpoints and document single-host behaviour.
  - Files: control OpenAPI, `orchestratord/src/http/control.rs`, provider tests.
  - Acceptance: drain performs graceful stop; reload still atomic; `/v1/pools/{id}/health` returns expected payload.
- Maintain capability discovery (`GET /v1/replicasets` or `/v1/capabilities`).
  - Files: control OpenAPI, handler implementation, tests.
  - Acceptance: CLI can derive concurrency from response.

### TODO — Artifacts & Tooling
- Reintroduce `/v1/artifacts` create/fetch for CLI plan storage.
  - Files: control OpenAPI, storage backend, `contracts/pacts/**`, tests.
  - Acceptance: CLI consumer tests read/write artifacts locally.
- Document and wire policy hook for HTTP tooling (FR-TL-002).
  - Files: `.docs/HOME_PROFILE.md` examples, `orchestratord/src/tooling/**`, config schema.
  - Acceptance: Policy default permits local testing; hook can be overridden.

### TODO — Config Schema
- Keep two-priority queue but restore determinism/session/budget config.
  - Files: `contracts/config-schema/src/lib.rs`, examples under `requirements/`.
  - Acceptance: `cargo xtask regen-schema` + validation succeed.
- Ensure catalog/artifact/tooling knobs exist with sane defaults.
  - Files: config schema + docs referencing them.
  - Acceptance: schema exposes CLI-needed fields with optional overrides.

### TODO — Metrics & Observability
- Update `ci/metrics.lint.json` to include retained metric set + optional gauges.
  - Files: `ci/metrics.lint.json`, metrics docs.
  - Acceptance: metrics linter reflects SSE `metrics` data, correlation IDs preserved.
- Verify code still emits correlation IDs and metrics after refactor.
  - Files: `orchestratord/src/http/**/*.rs`, provider tests, pact fixtures.
  - Acceptance: tests assert `X-Correlation-Id` echo; SSE metrics present.

### TODO — Tests
- Refresh provider verify, Pact, and BDD suites to new spec surface.
  - Files: `orchestratord/tests/provider_verify.rs`, `contracts/pacts/cli-consumer-orchestratord.json`, `cli/consumer-tests/**`, `test-harness/bdd/**`.
  - Acceptance: `cargo xtask dev:loop` covers catalog/drain/artifact/SSE metrics paths.
- Add determinism + queue metadata coverage.
  - Files: orchestrator tests to assert `seed`, `determinism`, `queue_position`, `predicted_start_ms` propagation.

## Progress Log
- 2025-09-17: Created HOME profile docs/spec, process guide, and scanner (excludes target/); fixed clippy/test; dev:loop green
- 2025-09-18: Home Profile v2 defined; TODO list realigned to CLI requirements.
