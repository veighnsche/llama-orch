# Orchestratord — Implementation Plan (OC-CTRL-2xxx)

Spec: `.specs/20-orchestratord.md`
Scope: control/data plane handlers, SSE framing, backpressure, typed errors, security.

## Stages and Deliverables

- Stage 0 — Contract Freeze
  - Author/align OpenAPI: `contracts/openapi/{control.yaml,data.yaml}` with `x-req-id`.
  - Add `x-examples` and JSON bodies per UX/DX proposal (429 policy_label, advisory fields).

- Stage 1 — CDC Consumer + Snapshots
  - `cli/consumer-tests` pact interactions for enqueue, stream (SSE frames), cancel, sessions; commit pact JSON.

- Stage 2 — Provider Verification
  - `orchestratord/tests/provider_verify.rs` verifies handlers against pact files.
  - Minimal vertical slice: admission → placement → SSE; typed errors with codes and context.

- Stage 5 — Observability
  - Ensure logs fields and metrics as per specs; expose `/metrics` scrape.

---

Alignment with README_LLM (Product Stages)

- Stage 6 — Admission → Dispatch vertical (product)
  - Handlers: `POST /v1/tasks` calls `QueueWithMetrics::enqueue`, produces 202 with `task_id`, `correlation_id`; select a single Ready replica (placement stub) and initiate streaming pipeline.
  - Handlers: `GET /v1/tasks/:id/stream` returns SSE framing (`started|token|metrics|end|error`), pulls from WorkerAdapter; propagate `engine`, `engine_version`, `sampler_profile_version`.
  - Handlers: `POST /v1/tasks/:id/cancel` cancels active/queued tasks; emit `tasks_canceled_total`.
  - Determinism flags per engine (adapter config) and readiness gating enforced.

- Stage 7 — Pool manager readiness (interface from orchestrator)
  - Control-plane handlers: `POST /v1/pools/:id/drain`, `POST /v1/pools/:id/reload`, `GET /v1/pools/:id/health`, `GET /v1/replicasets` (readiness/health/versions surfaced).
  - Version pinning: ensure replica sets pin `engine_version` and `sampler_profile_version`; do not mix.

- Stage 8 — Worker adapters conformance (orchestrator-facing)
  - Ensure adapter trait usage supports SSE framing, backpressure, timeouts/retries, typed error envelopes.
  - Wire metrics emission for tokens/latencies using labels: `engine`, `engine_version`, `pool_id`, `replica_id`, `priority`.

- Stage 9 — Scheduling & fairness (orchestrator policy hooks)
  - Finalize priority fairness; unignore fairness property test (core crate) and add integration checks; expose `admission_share` and `deadlines_met_ratio` gauges.

- Stage 10 — Capability discovery (client planning)
  - Enrich `GET /v1/replicasets` or add `GET /v1/capabilities` with API version, `ctx_max`, features, limits. Add provider verification and snapshots for payloads.

- Stage 11 — Config & quotas
  - Config examples for engines/workers and quotas (concurrent jobs, tokens/min, KV‑MB). Enforce `REQUIRE_REAL_LLAMA=1`, TZ in E2E contexts.

- Stage 12 — BDD coverage (journeys)
  - Features under `test-harness/bdd/tests/features/{data_plane,control_plane,sse}`: admission happy path, cancel, backpressure behavior, fairness bounds, determinism toggles. Zero undefined/ambiguous steps.

- Stage 13 — Dashboards & alerts (orchestrator views)
  - Populate `/ci/dashboards` for queue depth, rejections, latencies, tokens; add alert budgets. CI render check with sample data.

- Stage 14 — Startup self‑tests
  - On boot, run: preload (adapter ping), minimal decode (single token), cancel, metrics/log emission sanity; fail fast on violations.

- Stage 15 — Real‑model E2E (Haiku) — anti‑cheat gate
  - Drive via OrchQueue v1 only; minute+nonce; metrics token delta > 0; engine/model visible. Enforce anti‑cheat scans and `REQUIRE_REAL_LLAMA=1`.

## Tests

- Provider verify: `orchestratord/tests/provider_verify.rs` (OC-CTRL-2001..2051).
- Unit/integration tests for SSE framing and backpressure headers/bodies.
- BDD features: `test-harness/bdd/tests/features/data_plane/`, `control_plane/`, `sse/`.

- Admission vertical: integration tests under `orchestratord/tests/` covering 202 accepted, SSE start/end, cancel mid‑stream; metrics side‑effects asserted.
- Capabilities payloads: snapshots (insta) + provider verify.
- Fairness: end‑to‑end assertions on `admission_share` behavior under mixed priority load.

## Acceptance Criteria

- OpenAPI diff-clean on regen; examples compile.
- Provider verification green; SSE events in order and well-formed.
- Backpressure headers + body policy label present; error taxonomy stable.

- Stage 6: POST/GET/cancel handlers pass provider verify; vertical slice streams SSE deterministically (adapter-provided); metrics emission observed.
- Stage 7: Control-plane handlers reflect real replica health/readiness; version pinning enforced.
- Stage 8: Adapters exercised through orchestrator; backpressure + errors are typed; timeouts/retries validated.
- Stage 9: Fairness property unignored and green; gauges wired and observed.
- Stage 10: Capabilities endpoint(s) return complete info; provider verify + snapshots green.
- Stage 11: Quotas enforced; config examples compile/validate.
- Stage 12: BDD features green with zero undefined/ambiguous; artifacts captured.
- Stage 13: Dashboards render and alerts configured; CI render check passes.
- Stage 14: Startup self‑tests pass; startup fails fast on violation.
- Stage 15: Haiku anti‑cheat E2E passes within time budget.

## Backlog (initial)

- Handlers for control plane: drain/reload/health/replicasets.
- Data plane: createTask, stream SSE, cancel; correlation ID echo.
  - Typed errors with codes and retry hints; backpressure headers/bodies.
  - Auth middleware (API key) and logging filters.

Next slice (Stage 6 focus): implement POST /v1/tasks → enqueue + 202, GET stream SSE via mock adapter first (llama.cpp next), POST cancel, and update provider verify accordingly.

---

## DX Modularization Proposal (Orchestratord)

Goal: reduce compile/test churn, clarify boundaries, and enable parallel ownership. This is a planning section; do not create crates yet. Execute after Stage 6 vertical stabilizes.

Proposed layering (future sub‑crates):

- `orch-domain` (lib)
  - Shared domain types and typed error envelopes; SSE frame struct; backpressure policy labels.
  - No HTTP or engine deps; minimal `serde` and `thiserror` only.

- `orch-services` (lib)
  - Admission service (wraps `QueueWithMetrics`), placement policy, SSE streaming bridge, backpressure helpers.
  - Depends on: `orch-domain`, `worker-adapters/adapter-api` (trait only), `orchestrator-core`.
  - No Axum/HTTP; pure services for easy unit tests and reuse.

- `orch-api` (lib)
  - Axum routers and HTTP handlers: admission, stream, cancel, control, capabilities; request/response mapping.
  - Depends on: `orch-services`, `orch-domain`. No engine/adapters directly.

- `orchestratord` (bin)
  - Thin binary wiring env/config → `orch-api` router; logging/tracing init; metrics endpoint.

Dependency rules:

- `orch-api` → `orch-services` → `orch-domain`.
- `orch-services` → `worker-adapters/adapter-api`, `orchestrator-core`.
- No cycles; no handler importing adapter engine crates directly.

Rollout plan:

1) Stage 6.1: enforce in‑crate module boundaries mirroring the above (`src/http`, `src/services`, `src/domain`, `src/sse`, `src/backpressure`, `src/placement`, `src/state`, `src/errors`).
2) Stage 6.2: measure compile/test times; if hot, extract `orch-domain` first (lowest risk).
3) Stage 7–8: extract `orch-services` when adapter API is stable; finally `orch-api` when handler signatures settle.
4) Keep integration tests under `orchestratord/tests/` to avoid rebuilding libs on test changes.

Ergonomics:

- Public APIs on `orch-services` should be small, trait‑driven, returning `Result<DomainType, ErrorEnvelope>`.
- Prefer `pub(crate)` defaults; export only handler constructors from `orch-api`.

## Proposal (Accepted)

- Adopt product stages 6–15 for orchestrator (handlers, vertical slice, readiness, adapters conformance, fairness, capabilities, quotas, BDD, dashboards, startup self‑tests, Haiku E2E).
- Adopt DX modularization layering: `orch-domain` → `orch-services` → `orch-api` → thin `orchestratord` bin.
- Enforce module boundaries now; extract crates post Stage 6 stabilization per rollout plan.
