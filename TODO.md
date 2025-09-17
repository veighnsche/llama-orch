# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

> VERY VERY VERY IMPORTANT: HOME PROFILE v2.1 (Spec alignment + reference environment bringing up the actual system). Top priority until the CLI can drive the workstation deployment end-to-end.

## P0 — Home Profile v2.1 Compliance (Spec-Driven Backlog)

### Implementation — Data Plane & Sessions
- Correlation ID passthrough & request metadata propagation
  - Implementation: parse `X-Correlation-Id` + request provenance headers, generate UUID when absent, echo on every response (data/control) and store on task/session context.
  - Files: `orchestratord/src/http/data.rs`, `orchestratord/src/http/control.rs`, `orchestratord/src/http/catalog.rs`, `orchestratord/src/http/observability.rs`, `orchestratord/src/http/auth.rs`, tests under `orchestratord/tests` & `cli/consumer-tests`.
  - Acceptance: provider + pact suites prove echoed IDs; regression tests cover both provided and generated IDs.
- Admission primitive implementation (queue position/predicted start/backoff)
  - Implementation: compute queue depth + estimated start using `QueueWithMetrics`, GPU throughput and active leases; return real `queue_position`, `predicted_start_ms`, `backoff_ms` and align `Retry-After`/`X-Backoff-Ms` values.
  - Files: `orchestratord/src/http/data.rs`, `orchestratord/src/admission.rs`, `pool-managerd/src/leases.rs`, metrics helpers, tests.
  - Acceptance: integration tests demonstrate dynamic queue metadata; metrics mirror live queue depth.
- SSE streaming, budgets, and determinism
  - Implementation: wire worker adapter streams to real SSE (started/token/metrics/end/error), include budget deltas, queue depth, and determinism guarantees; honor `seed`/`determinism` flags through adapters.
  - Files: `orchestratord/src/http/data.rs`, `worker-adapters/**`, `contracts/openapi/data.yaml`, SSE fixtures under `test-harness/bdd`.
  - Acceptance: BDD + pact suites validate streaming order and payloads; determinism smoke test verifies reproducible token streams.
- Session store & budgets
  - Implementation: add in-memory (and optional persistent) session registry with TTL/turn counters, KV metadata, and token/time/cost budgets; surface via GET/DELETE endpoints and headers/SSE metrics.
  - Files: `orchestratord/src/state.rs`, new `orchestratord/src/session.rs`, `orchestratord/src/http/data.rs`, config schema for default budgets.
  - Acceptance: unit tests exercise TTL expiration + eviction; provider tests show accurate session introspection and budget math.
- Cancel path & task lifecycle
  - Implementation: implement `/v1/tasks/{id}/cancel` to remove queued work or signal adapters to stop running jobs; update metrics + logs.
  - Files: `orchestratord/src/http/data.rs`, `orchestrator-core/src/queue.rs`, adapters, tests.
  - Acceptance: integration test cancels queued and running tasks; queue depth/active leases adjust accordingly.
- Lifecycle states aligned to Active/Retired
  - Implementation: replace `ModelState::{Draft,Deprecated,Retired}` with `Active|Retired`; update OpenAPI enums, metrics, logs, and gating logic.
  - Files: `orchestratord/src/state.rs`, `orchestratord/src/http/control.rs`, `contracts/openapi/control.yaml`, `contracts/api-types`, tests.
  - Acceptance: attempts to set unsupported states fail 400; metrics only emit `Active|Retired` labels.

### Implementation — Control Plane & Catalog
- Catalog persistence & trust warnings
  - Implementation: back catalog endpoints with local storage (sqlite or JSON) storing model manifests, signatures, SBOM metadata; default to permissive trust with warning logs.
  - Files: `orchestratord/src/http/catalog.rs`, new storage module, `contracts/openapi/control.yaml`, config.
  - Acceptance: CRUD round-trip persists to disk; unsigned uploads emit warnings not hard failures.
- Model state API parity with spec
  - Implementation: update `/v1/catalog/models/{id}/state` payloads/enums; propagate state to pool manager + metrics.
  - Files: `orchestratord/src/http/control.rs`, `contracts/openapi/control.yaml`, provider tests.
  - Acceptance: state changes reflect in `/v1/pools/{id}/health` + metrics.
- Pool drain/reload/health on real workers
  - Implementation: connect control endpoints to pool-managerd + worker adapters; ensure drain stops new admissions, reload swaps model with rollback, health exposes last error / readiness.
  - Files: `orchestratord/src/http/control.rs`, `pool-managerd/**`, adapters, tests.
  - Acceptance: integration test drives drain→reload cycle touching GPU adapters, verifying leases + metrics.

### Implementation — Artifact Registry
- Storage backend + HTTP handlers
  - Implementation: add `/v1/artifacts` POST/GET handlers storing blobs + metadata (content-addressed) on local disk; include lineage + tags.
  - Files: new `orchestratord/src/http/artifacts.rs`, router wiring in `orchestratord/src/lib.rs`, storage utilities under `orchestratord/src/storage`, `contracts/openapi/control.yaml`, `contracts/api-types`.
  - Acceptance: provider + pact tests upload and download artifacts; CLI stores plan diffs successfully.
- Access control + retention policies
  - Implementation: enforce API key auth, optional size limits, retention GC based on config.
  - Files: artifacts module, config schema, docs.
  - Acceptance: exceeding limits yields 413; GC job tested via unit/integration test.

### Implementation — Capability Discovery & Placement
- Capability snapshot endpoints
  - Implementation: populate `/v1/replicasets` and `/v1/capabilities` with engine versions, ctx limits, concurrency, rate limits derived from active pools + adapters.
  - Files: `orchestratord/src/http/control.rs`, `pool-managerd/src/registry.rs`, config, tests.
  - Acceptance: CLI can derive concurrency; provider tests validate schema.
- Least-loaded GPU scheduling for RTX 3090 + 3060
  - Implementation: integrate NVML stats, track VRAM/free slots per GPU, schedule by heuristic that respects mixed VRAM; handle overflow gracefully.
  - Files: `orchestratord/src/placement.rs`, `pool-managerd/src/leases.rs`, GPU telemetry module, tests.
  - Acceptance: reference environment load test shows jobs distributed across 3090/3060 based on free VRAM.
- Concurrency + lease accounting
  - Implementation: move from placeholder lease counters to real tracker respecting per-engine concurrency; expose via metrics + capabilities.
  - Files: `pool-managerd/src/registry.rs`, `orchestratord/src/http/data.rs`, metrics.
  - Acceptance: queue throughput tests confirm concurrency caps; metrics reflect active leases per pool.

### Implementation — Worker Runtime & Adapters
- Replace mock adapter with real engine clients
  - Implementation: implement HTTP/WebSocket clients for llamacpp, vLLM, TGI, Triton adapters using workstation GPU; configure endpoints via home profile config.
  - Files: `worker-adapters/**`, `orchestratord/src/placement.rs`, `orchestratord/src/services`, config.
  - Acceptance: smoke tests decode sample prompts through each engine; determinism tests pass.
- Pool-managerd process orchestration
  - Implementation: flesh out `pool-managerd` modules (preload, drain, health, hetero split) to manage worker lifecycles on single host, including GPU pinning.
  - Files: `pool-managerd/src/*.rs`, `services` wiring, system tests.
  - Acceptance: orchestrator can start/stop workers, update health, and recover from failure scenarios.

### Implementation — Config Schema & Auth
- Home profile config schema refresh
  - Implementation: trim fairness/preemption/tensor_split from `contracts/config-schema`, add fields for budgets, determinism, API key, bind address, artifact storage path, capability overrides.
  - Files: `contracts/config-schema/src/lib.rs`, generated schema artifacts, docs under `.docs/`.
  - Acceptance: `cargo xtask regen-schema` succeeds; examples validate; config matches spec requirements.
- Auth token configuration
  - Implementation: load API token (and optional mTLS/OIDC) from config/env; update auth middleware + docs.
  - Files: `orchestratord/src/http/auth.rs`, config schema, CLI docs.
  - Acceptance: missing/incorrect tokens reject requests; CLI handshake documented.

### Observability & Metrics
- Metric set alignment
  - Implementation: keep required gauges/counters, remove fairness/preemption metrics from HOME build or guard behind feature flags; update `ci/metrics.lint.json` and docs.
  - Files: `orchestratord/src/metrics.rs`, `ci/metrics.lint.json`, `.specs/metrics/otel-prom.md`.
  - Acceptance: linter matches new set; `/metrics` exposes required series only.
- GPU + NVML integration
  - Implementation: scrape NVML for utilization/VRAM per device, wire into metrics + SSE `metrics` frame.
  - Files: `orchestratord/src/metrics.rs`, new telemetry module, placement.
  - Acceptance: metrics reflect live GPU stats on reference workstation.
- Logging & tracing
  - Implementation: add per-request structured logs with correlation IDs, budgets, placement decisions; expose optional OpenTelemetry exporters.
  - Files: logging config, `orchestratord/src/http/**/*.rs`, docs.
  - Acceptance: dev box CLI traces requests end-to-end with matching IDs.

### Security & Tool Policy Hooks
- HTTP tooling guardrail implementation
  - Implementation: add policy service/module controlling outbound HTTP tools (allow-list, redact secrets); surface config + docs.
  - Files: `orchestratord/src/tooling/**`, config schema, docs.
  - Acceptance: integration test shows blocked domain yields policy error, allowlisted domain succeeds.
- Secrets handling & audit
  - Implementation: ensure sensitive fields scrubbed from logs/artifacts; add audit trail for policy decisions.
  - Files: logging middleware, artifact storage, docs.
  - Acceptance: unit tests confirm redaction; audit log accessible via CLI.

### CLI Consumer & Tooling
- Implement `llama-orch-cli`
  - Implementation: build async CLI covering admission, SSE streaming, session reuse, artifact uploads, capability discovery, policy feedback; include config for remote host + token.
  - Files: `cli/llama-orch-cli/src`, CLI docs/tests, pact fixtures.
  - Acceptance: CLI drives full Spec→Contract→Tests→Code loop against reference environment.
- Developer ergonomics
  - Implementation: add commands for queue status, metrics tailing, artifact diffing; integrate with `auto coder` workflow.
  - Files: CLI crate, docs.
  - Acceptance: manual QA verifies commands; automated smoke tests cover core flows.

### Removal & Simplification
- Remove legacy reduction scaffolding
  - Tasks: delete `orchestratord/src/http/handlers.rs` re-export, scrub comments referencing reduction; drop unused fairness/preemption modules where not applicable.
  - Files: `orchestratord/src/http/handlers.rs`, metrics module, docs, `search_overkill.sh` hints.
  - Acceptance: build/tests pass without compatibility shim; search script reflects new guardrails.
- Clean up placeholder logic
  - Tasks: replace dummy correlation IDs, zero budgets, static SSE transcripts, and stub catalog responses with real implementations noted above.
  - Files: as per implementation sections.
  - Acceptance: no TODO/placeholder strings remain tied to removed behaviour.

### Testing & Tooling
- Update provider/pact/BDD suites
  - Implementation: regenerate OpenAPI-derived tests, update pact fixtures, expand BDD scenarios for budgets, determinism, artifacts, policy enforcement, mixed GPU scheduling.
  - Files: `orchestratord/tests/provider_verify.rs`, `contracts/pacts/**`, `cli/consumer-tests`, `test-harness/bdd`.
  - Acceptance: `cargo xtask dev:loop` covers all reinstated surfaces; CI green.
- Reference environment integration test
  - Implementation: add scripted test (behind feature flag) that runs against workstation via tunnel; collects latency/budget metrics.
  - Files: new test harness under `test-harness/reference-env`, docs.
  - Acceptance: manual run succeeds; documented for release readiness.

### Documentation & Examples
- Update spec/process docs
  - Tasks: refresh `.docs/HOME_PROFILE.md`, `.specs/00_home_profile.md`, `.docs/HOME_PROFILE_TARGET.md` with implementation details once tasks land.
  - Acceptance: docs describe actual behaviour; changelog entries recorded.
- Provide sample configs & runbooks
  - Tasks: add `examples/home-profile/` with orchestrator + CLI configs for RTX 3090 + 3060, include run scripts, troubleshooting guide.
  - Acceptance: new user can set up reference environment following docs.

## P1 — Reference Environment Automation & Delivery
- Workstation bootstrap scripts (systemd/docker-compose) for orchestrator + adapters; verify GPU detection.
- Dev box tooling: helper scripts for SSH tunnelling, CLI env setup, artifact sync.
- Continuous validation job (nightly) running smoke suite against reference hardware, collecting artifacts for regressions.
- Packaging: build release binaries + checksums for orchestrator/CLI with instructions tailored to home profile.

## P2 — Quality, Stretch, and Sustainability
- Optional mTLS/OIDC support for remote dev teams.
- Local artifact browser UI (static site) served from workstation.
- Adaptive concurrency auto-tuning based on GPU telemetry.
- Incremental model download/cache management for limited disk environments.
- Explore pluggable policy engine (Rego/WASM) for tool guardrails.

## Progress Log
- 2025-09-17: Created HOME profile docs/spec, process guide, and scanner (excludes target/); fixed clippy/test; dev:loop green
- 2025-09-18: Home Profile v2.1 defined; TODO rebuilt to cover full spec alignment and reference environment roadmap; `.docs/HOME_PROFILE_TARGET.md` added.
