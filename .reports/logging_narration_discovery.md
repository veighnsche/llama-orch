# Logging Narration Discovery — llama-orch

This report inventories the current logging and telemetry setup and surfaces concrete integration points for adding human‑readable narration to logs, repo‑wide. No code changes were made; findings are sourced from manifests, source files, specs, and CI configuration.

## Overview & Method

- Read spec/process docs: `README_LLM.md`, `.specs/proposals/**`.
- Enumerated crates via workspace `Cargo.toml` and per‑crate `Cargo.toml`.
- Searched codebase for logging libraries, initialization points, and macro usage.
- Mapped cross‑cutting flows and current log/metric touchpoints.
- Recorded config/env/feature controls and CI behavior.

Ambiguities are called out explicitly.

## SPEC Mechanics Summary (from README_LLM.md)

Source: `README_LLM.md` and example proposal `/.specs/proposals/2025-09-19-testing-ownership-and-scope.md`.

- Proposal location: `/.specs/proposals/` (repository‑relative).
- Naming: dated filename `YYYY-MM-DD-<slug>.md`. Example: `2025-09-19-testing-ownership-and-scope.md`.
- Document structure conventions observed:
  - Header with `Status`, `Owner`, `Date`.
  - Sections: Motivation, Scope, Normative Requirements (RFC‑2119, with scoped IDs like `ORCH-32xx`), Layer Responsibilities, Mapping, CI & Tooling, Migration Plan, Acceptance Criteria, and a required "Refinement Opportunities" section.
- Approval flow & labels: Not explicitly defined in `README_LLM.md`.
  - Process gates inferred from README and CI:
    - Spec‑first: Spec → Contract → Tests → Code.
    - Contracts regen required: `cargo xtask regen-openapi`, `cargo xtask regen-schema`, `cargo run -p tools-spec-extract --quiet`.
    - PR Checklist requires specs/contracts updated, tests referencing requirement IDs, and `cargo xtask dev:loop` green.
  - Requirement IDs appear within proposals (e.g., `ORCH-3200` series in the example). No repo‑wide label taxonomy is declared beyond this.
- Review gates: formatting (`cargo fmt`), clippy (warnings denied), spec extraction and link check, workspace tests, plus any contract/provider/BDD tests per CI.

Ambiguity: README does not state a formal approver set or label workflow; acceptance is functionally gated by CI and the spec‑first policy.

## Workspace Map (Crates & Binaries)

Derived from root `Cargo.toml` members and per‑crate manifests.

- contracts/api-types
  - Kind: lib (`contracts_api_types`)
  - Role: Generated/handcrafted OpenAPI data-plane types.
- contracts/config-schema
  - Kind: lib (`contracts_config_schema`)
  - Role: Config schema model + emit helpers for JSON schema.
- orchestrator-core
  - Kind: lib (`orchestrator_core`)
  - Role: Core invariants and primitives (e.g., in‑memory queue `queue.rs`).
- orchestratord
  - Kind: both (lib + bin `orchestratord`)
  - Role: HTTP service (Axum), admission → SSE vertical, metrics scrape, auth middleware.
  - Entrypoints/modules: `src/main.rs`, `app/bootstrap.rs` (init), `app/router.rs`, `app/middleware.rs`, `api/*` (control/data/catalog/observability), `services/streaming.rs`, `admission.rs`, `metrics.rs`, `infra/*`, `ports/*`, `domain/*`, `state.rs`.
- pool-managerd
  - Kind: both (implicit lib via `src/lib.rs`, bin `pool-managerd`)
  - Role: Registry/health/leases planning (stubs for lifecycle & readiness).
  - Modules: `registry.rs`, `health.rs`, `leases.rs`, `drain.rs`, `preload.rs`, `devicemasks.rs`, `backoff.rs`, `hetero_split.rs`.
- catalog-core
  - Kind: lib
  - Role: Model catalog resolve/verify/cache (HTTP via `ureq`).
- provisioners/engine-provisioner
  - Kind: lib
  - Role: Engine provisioning planner/ensurer (notably llama.cpp source build & run), tool preflight, pacman installs (Arch), model fetch via model‑provisioner.
- provisioners/model-provisioner
  - Kind: lib
  - Role: Ensures models present (file‑only flow in current code).
- worker-adapters/adapter-api
  - Kind: lib
  - Role: Adapter traits/types shared by HTTP adapters.
- worker-adapters/llamacpp-http, vllm-http, tgi-http, triton, mock
  - Kind: lib
  - Role: HTTP adapters for engines; mock for dev/BDD.
- cli/llama-orch-cli
  - Kind: bin (`llama-orch`)
  - Role: CLI frontend for design/verification flows.
- cli/consumer-tests
  - Kind: bin (`cli-consumer-tests`)
  - Role: Pact consumer and stub tests.
- test-harness/* (bdd, determinism-suite, chaos, e2e-haiku, metrics-contract)
  - Kinds: mix of bin (bdd/determinism/chaos/e2e-haiku) and lib (metrics-contract)
  - Role: System harnesses for BDD, determinism, e2e smoke, and metrics contract checks.
- orchestratord/bdd
  - Kind: bin (`bdd-runner`)
  - Role: Orchestratord-specific BDD steps and runner (local to orchestrator crate).
- tools/spec-extract, tools/readme-index, tools/openapi-client
  - Kinds: bin (spec-extract, readme-index) and lib (openapi-client)
  - Role: Docs/spec tooling and generated clients.
- xtask
  - Kind: bin (`xtask`)
  - Role: Workspace utility (regen, docs, pact verify, engine plan/up/down/status).

## Current Logging/Telemetry Stack

- Libraries (from root workspace dependencies and per‑crate manifests):
  - `tracing = 0.1` (workspace dep)
  - `tracing-subscriber = 0.3` with `fmt`, `env-filter`, `json` (workspace dep; optional in `orchestratord` behind `metrics` feature; default features include `metrics`)
  - No OpenTelemetry crates present in manifests.
- Shared telemetry/logging crate or module: None workspace‑level. `orchestratord/src/metrics.rs` provides a simple in‑process metrics registry; `infra/metrics.rs` re‑exports it.
- Initialization points:
  - `orchestratord/src/app/bootstrap.rs::init_observability()` uses `tracing_subscriber::{fmt, EnvFilter}` with `EnvFilter::try_from_default_env()` falling back to `info`, and forces JSON output via `.json().try_init()`.
  - Other binaries (`pool-managerd`, `cli`, test harness bins, `xtask`) do not set up `tracing`; they use `println!/eprintln!` for output.
- Output formatting and controls:
  - `orchestratord` format is JSON only (no pretty mode wired). Controlled by `RUST_LOG` environment via `EnvFilter`; default level `info`.
  - No repo‑wide toggle for pretty vs JSON detected; no feature flags for verbosity besides `orchestratord`'s `metrics` feature enabling `tracing-subscriber`.
- Field taxonomy in practice:
  - README requires logs include: `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`.
  - Current code does not use `tracing::info!` or similar; instead, `orchestratord` collects ad‑hoc JSON strings into `state.logs` for BDD checks, e.g., `{"queue_position":3,"predicted_start_ms":420}` in `api/data.rs`.
  - SSE transcripts include `queue_position`, `predicted_start_ms`, and at end `tokens_out` and `decode_ms` (note: `decode_ms` vs spec’s `decode_time_ms`).

## Log Call Inventory (per crate)

Method: ripgrep for `info!`, `warn!`, `error!`, `debug!`, `trace!`, `span!`, `event!`, `#[instrument]`, across `**/*.rs`.

- orchestratord
  - Counts: info=0, warn=0, error=0, debug=0, trace=0, span=0, instrument=0
  - Examples (current behavior using alternatives):
    - Log capture vector: `orchestratord/src/api/data.rs:69-75` pushes `{"queue_position":..., "predicted_start_ms":...}` to `state.logs`.
    - SSE assembly: `orchestratord/src/services/streaming.rs` creates events with `queue_position`, `predicted_start_ms`, `tokens_out`, `decode_ms`.
    - Initialization: `orchestratord/src/app/bootstrap.rs:9-14` sets JSON `tracing_subscriber` with `EnvFilter`.
  - Human‑readable phrasing: minimal; JSON fragments without narrative.
- orchestrator-core
  - Counts: all zero
  - Notes: pure library logic (`queue.rs`).
- pool-managerd
  - Counts: all zero (bin prints "pool-managerd stub"; library modules are planning stubs).
- catalog-core, provisioners/*, worker-adapters/*, cli/*, tools/*, test-harness/*
  - Counts: all zero for tracing macros.
  - Notable stdout/stderr usages:
    - `println!/eprintln!` across `xtask` and `provisioners/engine-provisioner` (e.g., CUDA fallback warnings and `spawned llama-server pid=...`).

No secret/PII instances found in logs. BDD tests explicitly assert no `secret`/`api_key` occurrences in logs (`orchestratord/bdd/src/steps/observability.rs`).

## Representative Flows (story points)

1) Admission → SSE streaming → metrics → artifact persist
- `orchestratord/src/api/data.rs:36-85:create_task`
  - Pushes ad‑hoc log: `{"queue_position":3,"predicted_start_ms":420}`
- `orchestratord/src/api/data.rs:87-99:stream_task`
  - Returns SSE built by service; seeds budget headers
- `orchestratord/src/services/streaming.rs:8-44:render_sse_for_task`
  - Increments `tasks_started_total`, observes `latency_first_token_ms`, later increments `tokens_out_total`, observes `latency_decode_ms`
- `orchestratord/src/services/streaming.rs:80-85`
  - Persists transcript via `services::artifacts::put`

2) Cancellation path
- `orchestratord/src/api/data.rs:101-112:cancel_task`
  - Records `tasks_canceled_total`; sets cancellation flag; appends log line `{"canceled":true,"task_id":"..."}` to `state.logs`
- `orchestratord/src/services/streaming.rs:48-78`
  - Checks cancellation at several points to cut off subsequent events/metrics

3) Provisioner failure/fallback (engine bring‑up)
- `provisioners/engine-provisioner/src/providers/llamacpp.rs`
  - Preflight tools; Arch pacman installs if allowed; prints warnings on CUDA misconfig: e.g., `warning: CUDA configure still failing; retrying with CPU-only (-DGGML_CUDA=OFF)`
  - On success: prints `spawned llama-server pid=... (pool=...)`

4) Admission with backpressure
- `orchestratord/src/admission.rs:43-88:QueueWithMetrics::enqueue`
  - Increments `admission_backpressure_events_total` and either `tasks_rejected_total` or enqueues and updates `queue_depth` gauge depending on policy.

## Config, Flags, and Env Controls

- Logging verbosity/format:
  - `RUST_LOG` (read by `EnvFilter` in `init_observability`) controls level; default `info`.
  - Output format is hard‑wired to JSON in `orchestratord`.
- Service address: `ORCHD_ADDR` (server bind address for `orchestratord`).
- Features:
  - `orchestratord` features: `server`, `metrics` (enables `tracing-subscriber` and `prometheus`), `artifacts`, `mock-adapters`. Defaults include `metrics`.
- Test/runtime envs noted in README (not logging‑specific): `LLORCH_API_TOKEN`, `REQUIRE_REAL_LLAMA`.

## OpenTelemetry (if any)

- No OTEL setup detected. No `opentelemetry` crates in manifests, no exporters configured.

## Tests & CI Touchpoints

- Log assertions:
  - `orchestratord/bdd/src/steps/observability.rs`
    - Verifies metrics names/labels in `/metrics` output.
    - Checks that logs include `queue_position` and `predicted_start_ms` via `state.logs`.
    - Ensures logs do not contain secrets / API keys.
- CI (`ci/pipelines.yml`):
  - Runs fmt, clippy, spec extract, regen tasks, workspace tests, link checks.
  - No explicit `RUST_LOG` or JSON formatting flags set in CI; relies on defaults.

## Candidate Centralization Points (options)

1) New workspace crate: `observability` (or `telemetry-core`)
- Pros: Single place to own log narration API, field taxonomy enforcement, init helpers (fmt/json/pretty), future OTEL exporters, and test shims.
- Cons: Cross‑crate wiring required; must avoid heavy deps (e.g., Axum) to remain consumable by CLI/tools.

2) Existing module extension in `orchestratord` (e.g., `orchestratord/src/infra/logging.rs`)
- Pros: Quick path for `orchestratord` binary; can wrap `tracing` with narration helpers; minimal initial blast radius.
- Cons: Not shared; other crates still emit `println!/eprintln!` or nothing; limits repo‑wide consistency.

3) Hybrid: new crate for core narration + thin per‑binary init
- Pros: Shared taxonomy and macros; small adapters in `orchestratord` and future bins (`pool-managerd`, harnesses) to choose pretty vs JSON.
- Cons: Slightly more upfront design; need to refactor tests that currently rely on `state.logs`.

Early adopters (fewest touches, highest coverage):
- `orchestratord` request handlers and services (`api/*`, `services/streaming.rs`, `admission.rs`).
- `provisioners/engine-provisioner` provider logs (replace `println!/eprintln!` with narration wrappers).
- `xtask` (optional): keep stdout, or route through narration when useful.

## Risks & Constraints

- Tests depend on `state.logs` JSON strings (BDD assertions). Replacing with `tracing` logs will require a test adapter (e.g., capture layer) or preserving the in‑memory vector for BDD.
- Performance: SSE path is hot; narration should use structured fields and avoid heavy string formatting. Prefer `tracing` fields + json formatter over preformatted strings.
- Cross‑crate coupling: A shared crate must stay dependency‑light to be used by tools and provisioners.
- Spec alignment: README requires specific fields; current code uses `decode_ms` instead of `decode_time_ms` in SSE logs—mismatch to address during implementation.
- No OTEL today: adding exporters changes runtime deps; feature‑gate to avoid impacting minimal profiles.

## Appendix: Examples (short snippets)

- Init observability (JSON + EnvFilter): `orchestratord/src/app/bootstrap.rs`
```
let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
let _ = fmt().with_env_filter(filter).json().try_init();
```

- Admission log (ad‑hoc): `orchestratord/src/api/data.rs`
```
lg.push(format!(
    "{{\"queue_position\":{},\"predicted_start_ms\":{}}}",
    admission.queue_position, admission.predicted_start_ms
));
```

- SSE end event fields: `orchestratord/src/services/streaming.rs`
```
let end = json!({"tokens_out": 1, "decode_ms": 5});
```

- Provisioner fallback warning: `provisioners/engine-provisioner/src/providers/llamacpp.rs`
```
eprintln!("warning: CUDA configure still failing; retrying with CPU-only (-DGGML_CUDA=OFF)");
```
