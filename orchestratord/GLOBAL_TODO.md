# Orchestratord v2 — GLOBAL TODO (End‑to‑End Plan)

Status: living plan
Last updated: 2025-09-17

This document is the comprehensive plan to deliver the refactored `orchestratord` per the specs:
- Crate spec: `orchestratord/.specs/00_orchestratord.md`
- Architecture: `orchestratord/.specs/10_orchestratord_v2_architecture.md`
- Contracts: `contracts/openapi/{control.yaml,data.yaml}`

Guiding principles:
- Spec → Contract → Tests → Code.
- TDD and BDD first.
- No back‑compat pre‑1.0. Legacy endpoints (e.g., `/v1/replicasets`) MUST NOT be served.

## 0) BDD/Gherkin Strategy (First Priority)
- Separate workspace‑level BDD (cross‑crate) from crate‑local BDD (orchestratord only).
- Create a local BDD harness inside `orchestratord/` with Cucumber/Gherkin, and migrate all orchestratord‑scoped features and steps locally.

Deliverables:
- `orchestratord/bdd/` (new sub‑crate or dev‑tests module) containing:
  - `tests/features/` with orchestratord features: control plane, data plane, SSE framing, capabilities, security, artifacts.
  - `src/steps/` with only the steps/glue needed by these features (`world.rs`, `control_plane.rs`, `data_plane.rs`, `security.rs`, minimal `observability.rs`).
  - `tests/bdd.rs` that lints undefined/ambiguous steps (pattern registry) and can run Cucumber when desired.
- Root `test-harness/bdd/` remains for cross‑crate features only; update docs to state the split of ownership.

Migration plan:
- Copy orchestratord‑specific feature files from `test-harness/bdd/tests/features/*` to `orchestratord/bdd/tests/features/*`.
- Copy and trim the matching step modules from `test-harness/bdd/src/steps/*` into `orchestratord/bdd/src/steps/*`.
- Replace any usage of legacy `http::handlers` shim with direct API module calls per v2 architecture.
- Update the local step registry; ensure `features_have_no_undefined_or_ambiguous_steps` passes locally.

## 1) Phased Delivery Plan

Phase 1 — Scaffolding and Middleware
- App layer: `app/{bootstrap,router,middleware}.rs` (auth, correlation‑id, error mapping).
- API layer files created, routes wired to `todo!()` service calls.
- Ports traits defined; in‑memory infra shims (Clock, ArtifactStore).
- Local BDD harness in place (from section 0) with initial failing scenarios for `/v1/capabilities`, control health, admission skeleton.

Phase 2 — Capabilities, Sessions, Control‑Health
- Implement `CapabilitiesService` and `GET /v1/capabilities` with `api_version`.
- Implement `SessionService` basic TTL/turn counters (admission enforcement stubbed).
- Implement `ControlService::health` and `GET /v1/pools/:id/health` minimal shape.
- TDD: unit tests for each service; BDD features pass for capabilities and health.

Phase 3 — Admission, SSE, Cancel, Backpressure
- Implement `AdmissionService` with ETA/position heuristics; backpressure 429 policy.
- Implement `StreamingService` on mock adapter; SSE event ordering and metrics frames.
- Implement `POST /v1/tasks/:id/cancel` with race‑free semantics (no tokens after cancel).
- BDD: data plane features pass; provider verify updated if needed.

Phase 4 — Artifacts & Transcript Capture
- Implement `ArtifactService` with in‑memory store + optional filesystem backend.
- Persist SSE transcript artifacts on `end`.
- Expose `POST /v1/artifacts` and `GET /v1/artifacts/:id` per contract.

Phase 5 — Control Drain/Reload, Error Mapping, Metrics
- Implement `POST /v1/pools/:id/drain|reload` against PoolRegistry.
- Complete error taxonomy mapping for all paths.
- Metrics registry and minimal required series; metrics lint passes.

Phase 6 — Determinism, Budgets, Hardening
- Propagate `seed` to adapters; enforce single‑slot mode in mock adapter tests.
- Admission budget enforcement (tokens/time/cost) gated by config.
- Finalize logs (include `engine_version`, `sampler_profile_version`).
- Determinism and chaos tests; finalize provider verification.

Exit criteria:
- All OpenAPI paths implemented; `/v1/replicasets` absent.
- BDD and provider verify pass locally for `orchestratord`.
- Metrics lints pass; determinism suite green for mock engine.

## 2) Work Breakdown (Trackers)
- Specs & Contracts
  - Keep `00_orchestratord.md` + `10_orchestratord_v2_architecture.md` as single source of truth.
  - OpenAPI diffs reviewed and versioned.
- Code
  - App/API/Services/Ports/Infra/Domain as per architecture.
- Tests
  - Local BDD harness, unit tests, provider verify, determinism suite, metrics lints.

## 3) Risks & Mitigations
- Step duplication across root and local harnesses → keep minimal local copies; consider future `common-bdd` shared crate if duplication grows.
- SSE determinism across real engines → keep mock deterministic; document gaps per engine.
- Feature creep → enforce “spec first” and weekly demoable increments.

## 4) Communication & Review
- Each phase lands behind passing local BDD and unit tests.
- PR template references requirement IDs from `requirements/`.
- Update `TODO.md` weekly with status.
