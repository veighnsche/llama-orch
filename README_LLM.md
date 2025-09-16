# README for LLM Developers — Decision Rules and Workflow

This document defines how an LLM developer must make decisions and contribute to this repository. Keep it short, decisive, and traceable.

## Golden Rules

- No backwards compatibility pre‑v1.0.0
  - Do not preserve pre‑1.0 behavior for compatibility. Do not write shims, adapters, or keep dead code for BC. Break and rebuild if it’s better. Remove unused paths aggressively.

- Spec is the source of truth
  - For every technical decision, consult `.specs/` first. If something is ambiguous, propose a spec update before writing code.

- Proposal required for spec changes
  - Do not change a spec without a proposal and review. Keep proposals minimal (problem, change, impact, new/changed requirement IDs, migration/rollback). Once accepted, update the spec and proceed.

- Order of work: Spec → Contract → Tests → Code
  - Spec: author/adjust normative text in `.specs/` using RFC‑2119 language with stable requirement IDs.
  - Contract: update formal interfaces (e.g., `contracts/openapi/`, `contracts/config-schema/`, ABIs) to reflect the spec.
  - Tests: write proofs (unit/integration/property/CDC) that reference the requirement IDs.
  - Code: implement the behavior to satisfy the tests and spec.
  - Discipline: do not work out of order. If a prerequisite (spec/contract/tests) is missing, stop and fulfill it before coding.

- Always update the TODO tracker with progress
  - After each meaningful change, update the root `TODO.md`. Keep it factual: what changed, where, and why.

- Finish what you start
  - Do not leave items half‑done. If blocked, explicitly note the blocker in the TODO and open a proposal/issue.

- Investigations are written down, not chatted
  - When tasked to investigate, write your findings in a committed `*.md` file (location at your discretion where it makes sense: near the component or under `.docs/`). You do not need to paste findings in chat—prefer durable, reviewable write‑ups.

- Tests follow the testing rules and proof bundle
  - When dealing with any tests, first read `.docs/testing/` for the applicable rules. Treat the “proof bundle” in `VIBE_CHECK.md` as mandatory: produce testing artifacts (logs, pact files, snapshots, metrics dumps) as specified.

## Traceability and Quality Gates

- Requirement IDs everywhere
  - Reference requirement IDs (e.g., `ORCH-…` umbrella and `OC-…` component-specific) in code comments near key logic and in test names/docs.

- Idempotent regeneration
  - Regeneration tools MUST be diff‑clean on a second run. Favor determinism and reproducibility over cleverness.

- Contracts are single source of truth for APIs
  - Update OpenAPI (`contracts/openapi/`), config schema, and metrics per `.specs/metrics/otel-prom.md` before or with code. Generated clients/servers and CDC tests should pass before merging. Metrics labels MUST include `engine` and engine‑specific version labels (e.g., `engine_version`, `trtllm_version`).

- No premature optimization
  - Optimize only after correctness and contract proofs are in place. Remove dead code early.

## Minimal Spec/Proposal Workflow

1) Write/adjust spec in `.specs/` with RFC‑2119 terms and stable IDs.
2) If a change is material, open a small proposal in the PR description and commit message. Include: problem, change summary, impacted areas, new/changed IDs, migration/rollback.
3) Update contracts (OpenAPI/config schema/ABIs) to match the spec.
4) Add/adjust tests that prove the requirements.
5) Implement code. Keep diffs tight and focused.
6) Update the TODO tracker with progress and links to proofs.

## TODO.md Lifecycle and Discipline

- If there is no `TODO.md` at the repository root:
  - Read `.specs/` and `.docs/workflow.md` to determine the next actionable items in proper order.
  - Create a root `TODO.md` and list those items in execution order.

- Active tracker
  - Keep the root `TODO.md` as the single active tracker. After each task, update it with what changed, where, why, and links to proofs (tests, CI logs, spec IDs).

- Archiving when done
  - When `TODO.md` is fully completed and reflects all work done, run the archive script (to be created): `bash ci/scripts/archive_todo.sh`.
  - The script MUST move `TODO.md` to `.docs/DONE/TODO-[auto-increment].md` and optionally create a fresh empty root `TODO.md`.

## PR Checklist (run before merge)

- Specs/Contracts
  - Spec updated or explicitly confirmed unchanged; requirement IDs present where applicable.
  - OpenAPI/config schema updated when surfaces change; examples compile.

- Proofs
  - Requirements regen is clean: `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - Links are valid: `bash ci/scripts/check_links.sh`
  - Workspace is healthy: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
  - Tests pass: `cargo test --workspace --all-features -- --nocapture`

- Hygiene
  - No BC shims or dead code left behind.
  - Commits and PR description reference relevant requirement IDs.
  - TODO tracker updated with what changed; if `TODO.md` is complete, run `ci/scripts/archive_todo.sh` to move it under `.docs/DONE/` with the next auto-incremented filename.

## Workflow alignment (SPEC→SHIP v2)

- Guiding principles (see `.docs/workflow.md`)
  - Spec is law with stable IDs (`ORCH-XXXX`); contracts are versioned artifacts.
  - Contract‑first; TDD; determinism by default; fail fast; short sessions.
  - Real‑model proof: the Haiku E2E test MUST pass; prefer a GPU worker over LAN; mocks cannot satisfy release gates.

- Stages and gates (brief)
  - Stage 0 — Contract freeze: OpenAPI + config schema regenerated; CI fails on diffs.
  - Stage 1 — CDC + snapshots: Pact + insta green before provider code.
  - Stage 2 — Provider verify: orchestrator passes pact verification.
  - Stage 3 — Properties: core invariants via proptest.
  - Stage 4 — Determinism: two replicas per engine; byte‑exact streams.
  - Stage 5 — Observability: metrics exactly per `.specs/metrics/otel-prom.md`.
  - Stage 6 — Admission → Dispatch vertical: queue → scheduler/placement → WorkerAdapter `submit()` (single ready replica); health/readiness gating; pin `engine_version` & `sampler_profile_version`; engine determinism flags
  - Stage 7 — Pool manager readiness: worker registry, heartbeat/health/readiness, drain/reload, leases; propagate `engine_version`/`model_digest`
  - Stage 8 — Worker adapters conformance: SSE framing, backpressure, timeouts/retries, typed errors; metrics emission per contract; engines: mock, llamacpp-http, vllm-http, tgi-http, triton
  - Stage 9 — Scheduling & fairness: finalize policy; unignore fairness property; wire `admission_share` & `deadlines_met_ratio` gauges; tune backpressure
  - Stage 10 — Capability discovery: `GET /v1/replicasets` or `GET /v1/capabilities` with API version, `ctx_max`, features, limits; provider verify + snapshots
  - Stage 11 — Config & quotas: examples per engine/worker; enforce quotas; env conventions (`REQUIRE_REAL_LLAMA=1`, TZ)
  - Stage 12 — BDD coverage: admission happy path, cancel, backpressure, fairness bounds, determinism toggles; zero undefined/ambiguous steps; proof artifacts
  - Stage 13 — Dashboards & alerts: Grafana panels (depth, rejections, latencies, tokens); alert budgets; CI render check
  - Stage 14 — Startup self‑tests: preload, minimal decode, cancel, telemetry emission
  - Stage 15 — Real‑model E2E (Haiku): pass within budget; metrics delta observed (anti‑cheat gate)
  - Stage 16 — Chaos & Load (nightly): failure injections and load SLO budgets
  - Stage 17 — Compliance & Release: requirements extract, COMPLIANCE.md, CHANGELOG_SPEC.md, tag & artifacts

  Anti‑cheat note (Haiku E2E): This test is the anti‑cheat gate. See `.docs/workflow.md` §4 “Haiku Test (Normative Protocol)” (Anti‑cheat). Explicit criteria:
  - MUST run against a real Worker (GPU preferred; CPU allowed only as a clearly marked CI fallback);
  - Forbid fixtures (no `fixtures/haiku*`), and disallow hardcoded haiku content anywhere in source;
  - Scan the repo for lines containing both the current minute words and the test nonce;
  - Fail if a mock/stub engine is detected or if `/metrics` is absent;
  - Require `REQUIRE_REAL_LLAMA=1` during the test run.

- Developer loop (deterministic)
  - `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo xtask regen-openapi && cargo xtask regen-schema`
  - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - `cargo test --workspace --all-features -- --nocapture`
  - `bash ci/scripts/check_links.sh`

- Environment conventions
  - `TZ=Europe/Amsterdam` for E2E‑Haiku; `REQUIRE_REAL_LLAMA=1` to enforce a real Worker.

- Definition of Done (per requirement)
  - Contract coverage in OpenAPI/Schema; consumer pact exists; provider verification passes; unit/property tests cover edges; observability proves it (metrics/logs); `requirements/*.yaml` links req → tests → code.

## Non‑Goals and Anti‑Patterns

- Do not invent behavior that is not in the spec; propose first.
- Do not keep unused flags/ENV/feature gates “just in case”.
- Do not bypass contracts/tests to “unblock” code.
- Do not rely on hidden state or nondeterministic outputs.

## When in Doubt

- Prefer smaller, well‑scoped proposals and PRs.
- Ask the spec for guidance; if it’s silent, propose the minimum change to make it explicit.
- Choose clarity and determinism over compatibility until v1.0.0.

## Branching Policy (pre‑v1.0.0)

- Trunk‑based development: develop directly on `master`.
- Avoid feature branches before v1.0.0; keep changes small and incremental.
- Commit messages must reference relevant requirement IDs and be descriptive (what changed, where, why).
- Use PRs for review when useful, but do not gate on long‑lived branches.

## Quick Status for LLMs (Progress & Global TODO)

Overall project progress: 33% (Stages 0–5 of 18 complete)

- [x] Stage 0 — Contract freeze: OpenAPI + config schema regenerated; CI fails on diffs
- [x] Stage 1 — CDC + snapshots: Pact + insta green before provider code
- [x] Stage 2 — Provider verify: orchestrator passes pact verification
- [x] Stage 3 — Properties: core invariants via proptest
- [x] Stage 4 — Determinism: two replicas per engine; byte‑exact streams
- [x] Stage 5 — Observability: metrics exactly per `.specs/metrics/otel-prom.md` (linter parity, /metrics endpoint, admission emissions wired)
- [ ] Stage 6 — Admission → Dispatch vertical
- [ ] Stage 7 — Pool manager readiness
- [ ] Stage 8 — Worker adapters conformance
- [ ] Stage 9 — Scheduling & fairness
- [ ] Stage 10 — Capability discovery
- [ ] Stage 11 — Config & quotas
- [ ] Stage 12 — BDD coverage
- [ ] Stage 13 — Dashboards & alerts
- [ ] Stage 14 — Startup self‑tests
- [ ] Stage 15 — Real‑model E2E (Haiku)
- [ ] Stage 16 — Chaos & Load (nightly)
- [ ] Stage 17 — Compliance & Release

  Detailed 4‑week execution summary and prior progress logs are archived under `.docs/DONE/` (see the latest e.g. `.docs/DONE/TODO-9.md`). Read these to understand exactly what has been completed, with dates and file references.
  For expected code layout at release, see `.plan/code-distribution.md`.

### Global TODO to v1.0.0 (High‑Level)

- [x] Spec discipline and planning artifacts (`.specs/`, `.plan/`, `.docs/` test catalog)
- [x] Contracts regenerated and frozen (OpenAPI, Config Schema)
- [x] CDC consumer tests and provider verification green
- [x] Core queue invariants proven (bounded, full policies, FIFO)
- [x] Determinism suite (byte‑exact per engine/version + seeds) green
- [x] Observability metrics: linter/spec alignment, /metrics endpoint, admission emissions (enqueue/cancel/backpressure)
- [ ] Scheduling fairness properties (priority fairness) — finalize and unignore test
- [ ] Pool manager scaffolding and replica lifecycle wiring
- [ ] Worker adapters conformance suite (mock + llamacpp-http + vllm-http + tgi-http + triton)
- [ ] Policy host + SDK minimal flow and CDC tests
- [ ] E2E Haiku on real worker (GPU) — success within SLO budget with metrics deltas captured
- [ ] CI hardening (dashboards, alerts, budgets) and release hygiene (SECURITY.md, license checks)
- [ ] Documentation pass (component READMEs, capabilities docs) and versioned release notes

Tip for LLMs: Start by scanning `.docs/DONE/` for the most recent archived TODO to get detailed, dated changes; then use the sections below to navigate the codebase.

## Project Structure (Navigation Map)

- `.specs/` — Normative specs and proposals; includes `metrics/otel-prom.md` and core specs
- `.plan/` — Per‑spec execution plans (00..71) aligned with README_LLM workflow
- `.docs/` — Docs; see `DONE/` for archived TODOs and detailed progress logs
- `contracts/` — API types, config schema, OpenAPI, and pact artifacts
- `orchestrator-core/` — Core library (queue, traits); property tests under `tests/`
- `orchestratord/` — Orchestrator binary; routes, admission metrics, `/metrics` endpoint
- `pool-managerd/` — Pool manager daemon (scaffolding)
- `worker-adapters/` — Engine adapters (mock, llamacpp-http, vllm-http, tgi-http, triton)
- `plugins/` — Policy host and SDK crates
- `test-harness/` — BDD, determinism suite, chaos, E2E Haiku harness
- `tools/` — Spec extraction, OpenAPI client, README indexer
- `ci/` — Pipelines, scripts, metrics linter config

;; with love, Vince :) ;;
