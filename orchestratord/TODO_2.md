# Orchestratord v2 — Phase II (Next 4 Weeks)

Status: planned
Last updated: 2025-09-17

Guiding principle: The v2 architecture spec leads the implementation. Tests support the refactor; if glue is not ready, keep stubs that compile.

-------------------------------------------------------------------------------
WEEK 5 — App & Middleware Completion, Domain Error Mapping, Server Binary
-------------------------------------------------------------------------------

1) Middleware & Router hardening
- Implement `app/middleware.rs`:
  - Auth middleware: require `X-API-Key` on protected routes.
  - Correlation-id middleware: echo or generate `X-Correlation-Id`.
  - Central error mapping to HTTP via `domain::error`.
- Unit tests for middleware behaviors.

2) Bootstrap & server executable
- `app/bootstrap.rs`: initialize tracing subscriber and metrics registry.
- `src/main.rs`: run Axum server; configurable addr via env (default 0.0.0.0:8080).
- README snippet for running the server locally.

Definition of Done (W5)
- All middleware unit tests green.
- Server binary starts and serves all v2 routes (handlers may be partial).

-------------------------------------------------------------------------------
WEEK 6 — Session TTL/Eviction & Budgets
-------------------------------------------------------------------------------

1) SessionService implementation
- Implement `services/session.rs` with TTL decrement using `ports::clock::Clock`.
- Evict sessions when TTL <= 0. Track `turns`, `kv_bytes`, `kv_warmth`.
- Unit tests for TTL decrements, eviction, and turns accounting.

2) Budgets (framework only)
- Define budget structs (tokens/time/cost) and attach to sessions.
- Surface remaining budgets in SSE metrics frames.

Definition of Done (W6)
- Session unit tests green; SSE metrics include budget placeholders.

-------------------------------------------------------------------------------
WEEK 7 — Adapter Registry & Streaming Determinism & Cancel
-------------------------------------------------------------------------------

1) Ports and mock infra
- Flesh out `ports::adapters::{AdapterRegistry, AdapterClient}` trait(s).
- Implement `infra::adapters::mock` deterministic adapter:
  - Honors `seed` for reproducible token streams.
  - Exposes `props()` for `ctx_max` and supported workloads.

2) StreamingService
- `services/streaming.rs` produces ordered SSE frames and handles cancel token.
- Persist transcript to ArtifactStore at `end`.
- BDD: expand SSE features to assert ordering and transcript capture.

Definition of Done (W7)
- Streaming unit tests and BDD SSE scenarios green.
- Cancel is race-free: no tokens after cancel.

-------------------------------------------------------------------------------
WEEK 8 — Provider Verify & Metrics Lint & Control Semantics
-------------------------------------------------------------------------------

1) Provider verification
- Add provider verify tests against OpenAPI for the implemented paths.
- Validate status codes and envelopes conform to contract.

2) Metrics lint compliance
- Conform to `ci/metrics.lint.json` for series and labels.
- Add missing metrics and labels (e.g., engine_version, policy_label).

3) Control drain/reload semantics
- Implement drain (state flag) and reload with proper outcomes.
- BDD: drain begins and reload success/rollback scenarios.

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
