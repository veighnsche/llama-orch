# Llama-Orch Meta Implementation Plan (SPEC→SHIP)

Status: draft · Last updated: 2025-09-15 · Scope: implement all specs in `.specs/` with contract-first, test-first discipline.

## 1) Executive Summary

This plan operationalizes the workflow defined in `README_LLM.md` and `.docs/workflow.md` to ship the orchestrator with NVIDIA-only inference and multi-engine support. Work proceeds in the strict order Spec → Contract → Tests → Code, with deterministic regeneration, traceability, and hard quality gates. All requirements with RFC-2119 language and stable IDs (e.g., `ORCH-xxxx`) must be proven via tests and mapped in `requirements/*.yaml`.

Key specs in `.specs/`:

- `00_llama-orch.md` — Umbrella requirements (ORCH-*)
- `10-orchestrator-core.md` — Queue, scheduling, determinism (OC-CORE-1xxx)
- `20-orchestratord.md` — Control/Data plane, SSE, backpressure (OC-CTRL-2xxx)
- `30-pool-managerd.md` — Preload, readiness, restart/backoff (OC-POOL-3xxx)
- `40-43-worker-adapters-*.md` — Engine adapter contracts (OC-ADAPT-5xxx)
- `50-plugins-policy-host.md`, `51-plugins-policy-sdk.md` — Policy ABI and SDK (OC-POLICY-4xxx)
- `60-config-schema.md` — JSON Schema generation/validation (OC-CONFIG-6xxx)
- `70-determinism-suite.md` — Suite semantics (OC-TEST-7xxx)
- `71-metrics-contract.md` and `.specs/metrics/otel-prom.md` — Metrics names/labels (OC-METRICS-7xxx)

## 2) Guardrails and Conventions

- Spec is law with stable IDs. Proposals precede spec changes: `.specs/proposals/*.md`.
- Contract-first. OpenAPI in `contracts/openapi/{control.yaml,data.yaml}`; config schema in `contracts/config-schema/` via `schemars`.
- Tests-before-code. CDC consumer, provider verify, property tests, determinism, metrics contract, E2E Haiku.
- Deterministic regen: second run of generators is diff-clean.
- Metrics must include `engine` and engine-specific version labels; see `.specs/metrics/otel-prom.md` and `ci/metrics.lint.json`.
- Real model E2E gate: prefer GPU worker; CI-only CPU fallback allowed and clearly marked.

References:

- `README_LLM.md` (rules, gates, PR checklist)
- `.docs/workflow.md` (Stage 0–8)
- `.docs/testing/test-case-discovery-method.md`, `.docs/testing/spec-derived-test-catalog.md`

## 2.1) DX & Modularization Blueprint (planning)

Purpose: make developer experience first-class. Encode how we split crates to reduce compile/test churn, enable parallel ownership, and keep boundaries clean.

Workspace layering (future-proof):

- Domain libs (no HTTP): `orch-domain`, `pool-domain` — types, error envelopes, SSE frames, policy labels.
- Service libs: `orch-services`, `pool-services` — admission, placement, SSE, backpressure, registries, drain/reload.
- API libs: `orch-api`, `pool-api` — Axum routers/handlers; map HTTP ↔ domain.
- Adapter contract: `worker-adapters/adapter-api` — trait + helpers; engines implement this only.
- Binaries: `orchestratord`, `pool-managerd` — thin wiring, tracing, metrics endpoint.

Dependency rules:

- `api` → `services` → `domain`; `services` → `adapter-api`, `orchestrator-core`.
- Orchestrator depends on `adapter-api` only; never import engine crates directly.
- No cross‑layer cycles; prefer `pub(crate)` surfaces and small trait‑based APIs.

Extraction triggers (when to split):

- Handler changes trigger large rebuilds or slow tests.
- Multiple teams working in parallel on adapters and handlers.
- LOC in `orchestratord/` approaches >6–7k or crate compile time exceeds budget.

Rollout (after Stage 6 vertical stabilizes):

1) Enforce module boundaries inside crates mirroring the layering; keep integration tests in bin crates to avoid rebuilding libs on test edits.
2) Extract `orch-domain` and `adapter-api` first (lowest coupling risk), then `orch-services`, and finally `orch-api`.
3) For pool manager: extract `pool-domain` first, then `pool-services` if needed.

Compile‑time ergonomics:

- Feature flags to build with `mock` adapter only by default during inner loops; enable engines via features.
- Keep generated code (OpenAPI client/types) in their own crates to avoid touching app crates.
- Avoid proc‑macro heavy dependencies on hot paths; prefer plain derive and small helpers.
- CI uses incremental workflows: run unit/property fast path before integration/E2E.

## 3) Phased Plan (Stages and Gates)

The following stages mirror `.docs/workflow.md` (§3) and add component-specific deliverables and ownership.

### Stage 0 — Contract Freeze (seed)

Deliverables:

- OpenAPI authored with `x-req-id` linking to spec IDs; examples added per UX/DX proposal.
- Config schema types covering pools, engines, devices, quotas, preemption.
- Metrics contract in `.specs/metrics/otel-prom.md` aligned with `ci/metrics.lint.json`.
Gates:
- `cargo xtask regen-openapi && cargo xtask regen-schema` → diff-clean on second run.
- Linkcheck green: `bash ci/scripts/check_links.sh`.
Artifacts/Owners:
- Contracts: `contracts/openapi/*.yaml`, `contracts/config-schema/` (Contracts Owner)

### Stage 1 — CDC (consumer) + Snapshots

Deliverables:

- Pact tests in `cli/consumer-tests` against OrchQueue v1 `POST /v1/tasks`, `GET /v1/tasks/:id/stream`, cancel, sessions.
- Snapshots with `insta` for CLI transcripts.
Gates:
- Pact consumer tests green; pact files committed under `contracts/pacts/`.

### Stage 2 — Provider Verification (server)

Deliverables:

- Provider verify test in `orchestratord/tests/provider_verify.rs` loading pact files.
- Minimal vertical slice: admission → placement to one ready replica; typed errors; SSE framing.
Gates:
- Provider verification green.
Artifacts/Owners:
- `orchestratord/` (Orchestratord Owner)

### Stage 3 — Properties & Invariants (core)

Deliverables:

- Proptest suites in `orchestrator-core/tests/props_queue.rs` for FIFO, fairness, reject/drop policies, cancel races.
Gates:
- Property tests green.
Artifacts/Owners:
- `orchestrator-core/` (Core Owner)

### Stage 4 — Determinism Suite (replica set)

Deliverables:

- `test-harness/determinism-suite/`: 64 seeds corpus; per-engine settings (llama.cpp `--parallel 1 --no-cont-batching`, etc.).
Gates:
- Byte-exact token streams across two replicas per engine; failure emits token diff artifact.

### Stage 5 — Observability & SLOs

Deliverables:

- Metrics implemented per `.specs/metrics/otel-prom.md` and `71-metrics-contract.md`.
- Dashboards in `ci/dashboards/`, alerts defined; logs carry required fields.
Gates:
- Metrics linter green (`ci/metrics.lint.json`), dashboard render checks pass.

### Stage 6 — Admission → Dispatch vertical (product)

Deliverables:

- `POST /v1/tasks` enqueues via core queue (`QueueWithMetrics`) and returns 202 with `task_id`, `correlation_id`.
- Placement selects a single Ready replica; `GET /v1/tasks/:id/stream` streams SSE (`started|token|metrics|end|error`).
- `POST /v1/tasks/:id/cancel` cancels queued/active tasks; determinism flags per engine; readiness gating.
Gates:
- Provider verify green for POST/GET/cancel; SSE events well-formed; metrics side-effects observed.

### Stage 7 — Pool manager readiness

Deliverables:

- Replica registry (heartbeat/health/readiness), drain/reload, leases; surface `engine_version`/`model_digest`.
- Control-plane: `POST /v1/pools/:id/{drain,reload}`, `GET /v1/pools/:id/health`, `GET /v1/replicasets`.
Gates:
- Only Ready replicas advertised; pin `engine_version` and `sampler_profile_version`; do not mix.

### Stage 8 — Worker adapters conformance

Deliverables:

- Adapters: mock, llamacpp-http, vllm-http, tgi-http, triton. SSE framing/backpressure/timeouts/typed errors.
- Metrics emission per contract: tokens in/out, latencies, labels {engine,engine_version,pool_id,replica_id,priority}.
Gates:
- Adapter integration tests pass; determinism normalization per engine; version labels present.

### Stage 9 — Scheduling & fairness

Deliverables:

- Finalize policy; wire `admission_share` and `deadlines_met_ratio`; tune backpressure.
Gates:
- Fairness property unignored and green; end-to-end fairness behavior validated.

### Stage 10 — Capability discovery

Deliverables:

- `GET /v1/replicasets` or `GET /v1/capabilities` with API version, `ctx_max`, features, limits; snapshots + provider verify.

### Stage 11 — Config & quotas

Deliverables:

- Engine/worker examples; quotas (concurrent jobs, tokens/min, KV-MB); env conventions enforced.

### Stage 12 — BDD coverage (journeys)

Deliverables:

- Features for admission, cancel, backpressure, fairness bounds, determinism toggles; zero undefined/ambiguous.

### Stage 13 — Dashboards & alerts

Deliverables:

- Panels (queue_depth, rejections, latencies, tokens) + alert budgets in `/ci/dashboards`; CI render check.

### Stage 14 — Startup self-tests

Deliverables:

- Preload, minimal decode, cancel, telemetry; fail fast on violation.

### Stage 15 — Real-Model E2E (Haiku) — anti-cheat gate

Deliverables:

- E2E test in `test-harness/e2e-haiku/` driving only OrchQueue v1; minute+nonce; metrics token delta > 0; engine/model visible.
Gates:
- Pass within time budget; anti-cheat enforced (real Worker, no fixtures, repo scan, REQUIRE_REAL_LLAMA=1).

### Stage 16 — Chaos & Load (nightly)

Deliverables:

- `test-harness/chaos/` scenarios (kill/restart/drain/reset) and short load SLO checks.
Gates:
- Nightly-only pass.

### Stage 17 — Compliance & Release

Deliverables:

- `tools/spec-extract` → `requirements/*.yaml` refreshed; `COMPLIANCE.md` generated; `CHANGELOG_SPEC.md` updated; artifacts published.
Gates:
- Spec-extract diff-clean; linkcheck green; workspace fmt/lint/tests green.

## 4) Workstreams by Spec (Tracks)

For each spec, plan and deliver tests and code per the stages above. Per-spec detailed plans will live in `.plan/` (see §10).

- `00_llama-orch.md` (ORCH-*):
  - Contracts: OpenAPI operations for control/data plane; typed errors and SSE framing.
  - Tests: CDC, provider verify, BDD features under `test-harness/bdd/tests/features/` (admission, scheduling, sse, security, observability, lifecycle).
  - Code: Implement across `orchestratord/`, `orchestrator-core/`, `pool-managerd/` and adapters.

- `10-orchestrator-core.md` (OC-CORE-1xxx):
  - Tests: `orchestrator-core/tests/props_queue.rs` (properties) and targeted unit tests.
  - Code: Queue, scheduling, determinism enforcement.

- `20-orchestratord.md` (OC-CTRL-2xxx):
  - Tests: `orchestratord/tests/provider_verify.rs`; SSE payload/unit tests.
  - Code: Handlers, SSE stream, backpressure headers and bodies.

- `30-pool-managerd.md` (OC-POOL-3xxx):
  - Tests: preload/readiness/backoff; device masks; heterogeneous splits.
  - Code: Lifecycle, restart/backoff, readiness endpoints.

- `40-43-worker-adapters-*.md` (OC-ADAPT-5xxx):
  - Tests: adapter integration; determinism normalization; version capture.
  - Code: Health, properties, completion SSE, cancel, metrics scrape.

- `50-plugins-policy-host.md`, `51-plugins-policy-sdk.md` (OC-POLICY-4xxx):
  - Tests: WASI ABI pure/deterministic functions; sandbox/time/mem bounds; SDK stability.
  - Code: Host bridge and SDK surfaces.

- `60-config-schema.md` (OC-CONFIG-6xxx):
  - Tests: strict validation; example configs validate; idempotent generation.
  - Code: Rust types and schema emitter.

- `70-determinism-suite.md` (OC-TEST-7xxx):
  - Tests: suite semantics and corpus; harness alignment.

- `71-metrics-contract.md` and `.specs/metrics/otel-prom.md` (OC-METRICS-7xxx):
  - Tests: metrics linter; BDD observability checks; dashboards.

## 5) Proposals in Flight (v3.2)

Incorporate `.specs/proposals/2025-09-15-spec-v3.2-catalog-scheduling.md`:

- Catalog & Trust (ORCH-3060..3068), Lifecycle (ORCH-3069..3074), WFQ/EDF & Preemption (ORCH-3075..3090), glue (ORCH-3091..3094).
- Update contracts (`contracts/openapi/*`, `contracts/config-schema/`), metrics (`.specs/metrics/otel-prom.md`), tests (BDD scheduling/lifecycle, provider tests), dashboards.

Incorporate `.specs/proposals/2025-09-15-ux-dx-improvements.md`:

- OpenAPI `x-examples`, 429 body `policy_label`, `ErrorEnvelope` advisory fields, `X-Correlation-Id` behavior.

## 6) Test Strategy and Catalog

- Method: `.docs/testing/test-case-discovery-method.md`
- Catalog: `.docs/testing/spec-derived-test-catalog.md` kept diff-clean; IDs map to concrete tests and code paths; cross-spec combinations enumerated.
- Harnesses:
  - BDD: `test-harness/bdd/` binary `bdd-runner` supports `LLORCH_BDD_FEATURE_PATH` to target `tests/features/` or a subset.
  - Determinism: `test-harness/determinism-suite/`
  - Metrics: `test-harness/metrics-contract/`
  - CDC Consumer: `cli/consumer-tests/`

## 7) Developer Loop (deterministic)

Run in order from repo root:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo xtask regen-openapi && cargo xtask regen-schema
cargo run -p tools-spec-extract --quiet && git diff --exit-code
cargo test --workspace --all-features -- --nocapture
bash ci/scripts/check_links.sh
```

Useful wrappers:

```bash
# BDD (all or targeted)
cargo run -p test-harness-bdd --bin bdd-runner --quiet
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/sse \
  cargo run -p test-harness-bdd --bin bdd-runner --quiet

# Determinism and Haiku
cargo xtask ci:determinism
cargo xtask ci:haiku:cpu
```

## 8) Quality Gates (per PR)

- Specs/Contracts updated or confirmed unchanged; requirement IDs present.
- Regen tools diff-clean; linkcheck passes.
- Workspace fmt/clippy as errors.
- Tests green: CDC consumer, provider verify, unit/property, determinism, BDD, metrics linter; E2E gate on protected branches.
- TODO tracker updated (`TODO.md` → archived via `ci/scripts/archive_todo.sh` when complete).

## 9) RACI (roles by stream)

- Spec Owner: writes/updates `.specs/*.md`; owns proposals and IDs.
- Contracts Owner: OpenAPI and config schema; examples; regeneration.
- Test Harness Owner: BDD, determinism, metrics contract, CDC consumer; keeps catalog in sync.
- Core Owner: `orchestrator-core/` properties and invariants.
- Orchestratord Owner: provider verify and server handlers; SSE & backpressure.
- Pool Manager Owner: preload/readiness/backoff; device masks; heterogeneous splits.
- Adapters Owner(s): each engine adapter crate and determinism normalization.
- Observability Owner: metrics emission, linter compliance, dashboards.

Notes:

- Owners may be shared across maintainers; assignments are per-PR.

## 10) Deliverables and Documentation

- Per-spec plan stubs in `.plan/` (to be added):
  - `.plan/10_orchestrator_core.md`
  - `.plan/20_orchestratord.md`
  - `.plan/30_pool_managerd.md`
  - `.plan/40_worker_adapters.md`
  - `.plan/50_policy.md`
  - `.plan/60_config_schema.md`
  - `.plan/70_determinism_suite.md`
  - `.plan/71_metrics_contract.md`
- Compliance and Traceability:
  - `requirements/*.yaml` kept up-to-date by `tools/spec-extract`.
  - `COMPLIANCE.md` generated at Stage 8.

## 11) Risks and Mitigations

- GPU availability for real-model tests — maintain a reachable LAN GPU worker; provide CI-only CPU carve-out with clear marking.
- Determinism across engines — pin `engine_version` and sampler profiles; normalize templates in adapters; do not compare across engines.
- Metrics cardinality — adhere to label budgets; admission counters may omit `engine_version` by contract.
- Spec drift vs contracts — enforce contract freeze and regen diffs as CI blockers.

## 12) Acceptance and Exit Criteria

Per requirement (DoD):

- Contract coverage present; consumer pact exists; provider verification passes; unit/property cover edges; observability proves it; `requirements/*.yaml` links req → tests → code.

Repo release (Stage 8):

- All gates green; E2E Haiku passes; COMPLIANCE and CHANGELOG updated; dashboards committed.
