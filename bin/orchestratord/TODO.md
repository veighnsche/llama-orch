# orchestratord TODO

**Scope**: This crate only (`bin/orchestratord`)  
**Status**: Alpha (v0.1.0) — No backwards compatibility guarantees pre-1.0.0  
**Last Updated**: 2025-09-30

**See Also**:
- Workspace-level e2e checklist: `/CHECKLIST_HAIKU.md`
- Analysis documents: `IMPLEMENTATION_STATUS.md`, `ANALYSIS_SUMMARY.md`
- Quick wins plan: `PHASE1_QUICK_WINS.md`

---

## Completed ✅

### Owner F Tasks (from TODO_OWNERS_MVP_pt3.md)
- [x] Handoff autobind watcher (`ORCHD-HANDOFF-AUTOBIND-0002`)
- [x] Health-gated dispatch in streaming
- [x] Admission response `streams` and `preparation` fields populated
- [x] Unit tests for handoff autobind and health-gated streaming
- [x] Integration test scaffolding

### Core Infrastructure
- [x] HTTP routing for v2 endpoints (data, control, catalog, artifacts, observability)
- [x] Middleware: correlation ID, API key enforcement, bearer identity
- [x] Session management (get/delete)
- [x] Artifact storage (create/get with CAS)
- [x] Catalog operations (create/get/delete/verify/set_state)
- [x] Basic SSE streaming with deterministic fallback
- [x] Cancellation support
- [x] Pool health endpoint
- [x] Capabilities endpoint with caching
- [x] Metrics endpoint (Prometheus text format)

---

## High Priority 🔴

### Control Plane (OC-CTRL-2xxx)

#### Pool Management
- [ ] **OC-CTRL-2002**: `POST /v2/pools/:id/drain` — Implement actual draining logic
  - Current: Stub returns 202 Accepted
  - Need: Respect `deadline_ms`, prevent new admissions, wait for in-flight tasks
  - File: `src/api/control.rs::drain_pool()`
  - Ref: `.specs/20-orchestratord.md#1-control-plane`

- [ ] **OC-CTRL-2003**: `POST /v2/pools/:id/reload` — Implement atomic model reload
  - Current: Stub returns 202 Accepted
  - Need: Atomic switch or rollback, coordinate with pool-managerd
  - File: `src/api/control.rs::reload_pool()`
  - Ref: `.specs/20-orchestratord.md#1-control-plane`

#### Discovery & Capabilities
- [ ] **OC-CTRL-2004**: Remove legacy `/v1/replicasets` endpoint (if exists)
  - Verify not served; spec says MUST NOT serve pre-1.0
  - File: `src/app/router.rs`
  - Ref: `.specs/20-orchestratord.md#1-control-plane`

- [ ] **OC-CTRL-2060**: Enhance `/v2/meta/capabilities` with complete metadata
  - Current: Basic capabilities from cache
  - Need: Per-engine/pool: `engine_version`, `sampler_profile_version`, `ctx_max`, `max_tokens_out`, concurrency/slots, `supported_workloads`, rate limits, feature flags
  - File: `src/api/control.rs::get_capabilities()`, `src/services/capabilities.rs`
  - Ref: `.specs/20-orchestratord.md#8-capabilities-discovery`

- [ ] **OC-CTRL-2061**: Include API version in capabilities payload
  - Need: Compatible with OpenAPI `info.version`
  - File: `src/services/capabilities.rs`

### Data Plane & Admission

#### Admission Policy
- [ ] **ORCHD-DATA-1001**: Replace sentinel validations with real policy
  - Current: String sentinels like `model_ref == "pool-unavailable"`
  - Need: Use `orchestrator-core` placement/admission hooks, `pool-managerd` state
  - File: `src/api/data.rs::create_task()`
  - Ref: Inline TODO comment

- [ ] **ORCHD-CATALOG-CHECK-0006**: Integrate catalog-core for model presence check
  - Need: Check if `model_ref` is present and Active
  - Need: Decide provision vs reject based on policy
  - Need: Surface HF auth/cache errors clearly
  - File: `src/api/data.rs::create_task()`
  - Ref: Inline TODO comment

- [ ] **ORCHD-PROVISION-POLICY-0005**: Orchestrate provisioners on-demand
  - Need: Trigger `engine-provisioner` + `model-provisioner` when model/engine missing
  - Need: Integrate with `pool_managerd::registry` readiness
  - Need: Respect user pool/GPU pin overrides
  - File: `src/api/data.rs::create_task()`
  - Ref: Inline TODO comment

- [ ] **OC-CTRL-2010**: Full admission checks (ctx, token budget) before enqueue
  - Current: Basic ctx/deadline checks
  - Need: Token budget enforcement, session linkage
  - File: `src/api/data.rs::create_task()`
  - Ref: `.specs/20-orchestratord.md#2-data-plane-orchqueue-v2`

- [ ] **OC-CTRL-2011**: Queue full handling with proper headers
  - Current: Returns error with retry_after_ms
  - Need: Ensure `429` status, `Retry-After` header, `X-Backoff-Ms` header, JSON body with `policy_label`, `retriable`, `retry_after_ms`
  - File: `src/api/data.rs::create_task()`, `src/domain/error.rs`
  - Ref: `.specs/20-orchestratord.md#2-data-plane-orchqueue-v2`

- [ ] **ORCHD-ADMISSION-2002**: Real ETA calculation for `predicted_start_ms`
  - Current: Heuristic `queue_position * 100`
  - Need: Derive from pool throughput, `slots_free`, `perf_tokens_per_s`
  - File: `src/api/data.rs::create_task()`
  - Ref: Inline TODO comment

#### Pin Override Support
- [ ] **OC-CTRL-2013**: Implement pool pin override routing
  - Need: When `TaskRequest.placement.pin_pool_id` is set and policy allows, route to that pool exclusively
  - File: `src/api/data.rs::create_task()`, placement logic
  - Ref: `.specs/20-orchestratord.md#21-optional-pin-override`

- [ ] **OC-CTRL-2014**: Fail deterministically on invalid pin
  - Need: If pinned pool unknown/not Ready or pinning disabled, fail with `INVALID_PARAMS` or `POOL_UNREADY` (no silent fallback)
  - File: `src/api/data.rs::create_task()`
  - Ref: `.specs/20-orchestratord.md#21-optional-pin-override`

### Streaming & SSE

#### Event Ordering & Payloads
- [ ] **OC-CTRL-2020**: Verify all SSE event types emitted
  - Current: `started`, `token`, `metrics`, `end` implemented
  - Need: Ensure `error` event properly emitted on failures
  - File: `src/services/streaming.rs`
  - Ref: `.specs/20-orchestratord.md#3-sse-framing`

- [ ] **OC-CTRL-2021**: Include `queue_position` and `predicted_start_ms` in `started`
  - Current: Implemented
  - Verify: Matches spec payload shape
  - File: `src/services/streaming.rs`

- [ ] **OC-CTRL-2022**: Ensure event payloads are well-formed JSON
  - Current: Using `serde_json::json!`
  - Verify: Ordering is `started → token* → end`
  - File: `src/services/streaming.rs`

#### Transport & Performance
- [ ] **OC-CTRL-2025**: HTTP/2 support for SSE with graceful fallback (50% DONE)
  - ✅ Environment variable `ORCHD_PREFER_H2` exists
  - ✅ Narration emitted when H2 preference set
  - ⚠️ TODO: Configure Axum to actually enable HTTP/2
  - ⚠️ TODO: Configure compression settings
  - File: `src/app/bootstrap.rs::start_server()`
  - Ref: `.specs/20-orchestratord.md#31-transport-performance-normative`

- [ ] **OC-CTRL-2026**: SSE encoder optimization
  - Current: Uses `BufWriter`
  - Verify: No per-token heap allocations on hot path
  - Optional: Micro-batch mode (disabled by default, bounded)
  - File: `src/services/streaming.rs::build_and_persist_sse()`
  - Ref: `.specs/20-orchestratord.md#31-transport-performance-normative`

- [ ] **OC-CTRL-2027**: Event ordering enforcement
  - Current: Deterministic path follows order
  - Verify: Adapter path maintains `started → token* → end`
  - Verify: Heartbeat/keepalive (if added) remains compatible
  - File: `src/services/streaming.rs`

#### Streaming Failure Semantics
- [ ] **OC-CTRL-2028**: Emit `event: error` on streaming failures
  - Need: Minimal JSON body: `{ code, retriable, retry_after_ms?, message? }`
  - File: `src/services/streaming.rs`
  - Ref: `.specs/20-orchestratord.md#33-streaming-failure-semantics-uniform`

- [ ] **OC-CTRL-2029**: Terminate stream after `error` event
  - Need: No further `token`/`metrics`/`end` after error
  - File: `src/services/streaming.rs`

- [ ] **OC-CTRL-2034**: Pre-stream errors use HTTP status, established streams use SSE error
  - Current: Likely correct
  - Verify: Established streams stay `200 OK` and carry error via SSE
  - File: `src/services/streaming.rs`, `src/api/data.rs::stream_task()`

#### Streaming TODOs (from code)
- [ ] **ORCHD-STREAM-1101**: Build TaskRequest from actual admission context
  - Current: Stub request in fallback path
  - Need: Use admission snapshot with session linkage, placement decision, budgets
  - File: `src/services/streaming.rs::try_dispatch_via_adapter()`
  - Ref: Inline TODO comment (partially done)

- [ ] **ORCHD-STREAM-1102**: Propagate cancellation via structured token
  - Current: Shared state polling
  - Need: Structured `CancellationToken` to adapters
  - Need: Verify no tokens after cancel across all adapters
  - File: `src/services/streaming.rs`
  - Ref: Inline TODO comment

- [ ] **ORCHD-STREAM-1103**: Map adapter errors to domain errors
  - Need: Emit `error` SSE frames with `code/message/engine` per spec
  - File: `src/services/streaming.rs::try_dispatch_via_adapter()`
  - Ref: Inline TODO comment

- [ ] **ORCHD-SSE-1301**: Include additional `started` fields
  - Need: `engine`, `pool`, `replica`, `sampler_profile_version` when available
  - File: `src/services/streaming.rs`
  - Ref: Inline TODO comment

- [ ] **ORCHD-STREAM-VERBOSE-0011**: Parse `?verbose=true` query param
  - Need: Propagate to `render_sse_for_task_verbose()` (to be added)
  - Need: Include `{"human": "...", "phase": "..."}` breadcrumbs in `metrics` frames
  - File: `src/api/data.rs::stream_task()`
  - Ref: Inline TODO comment

#### Cancellation
- [ ] **OC-CTRL-2012**: Race-free cancellation
  - Current: Sets flag in shared state
  - Verify: No tokens emitted after cancel
  - File: `src/api/data.rs::cancel_task()`, `src/services/streaming.rs`
  - Ref: `.specs/20-orchestratord.md#2-data-plane-orchqueue-v2`

### Error Taxonomy ✅ COMPLETE

- [x] **OC-CTRL-2030**: Stable error codes
  - ✅ All codes present in `contracts/api-types/src/generated.rs::ErrorKind`
  - ✅ Includes: `AdmissionReject`, `QueueFullDropLru`, `InvalidParams`, `PoolUnready`, `PoolUnavailable`, `ReplicaExhausted`, `DecodeTimeout`, `WorkerReset`, `Internal`, `DeadlineUnmet`
  - ✅ Plus: `ModelDeprecated`, `UntrustedArtifact`
  - File: `contracts/api-types/src/generated.rs`, `src/domain/error.rs`

- [x] **OC-CTRL-2031**: Include `engine` and `pool_id` in errors
  - ✅ `ErrorEnvelope` has `engine: Option<Engine>` field
  - ✅ Domain error mapping includes engine in all error variants
  - File: `src/domain/error.rs::into_response()`

- [x] **OC-CTRL-2032**: Advisory fields in error envelopes
  - ✅ `ErrorEnvelope` has: `retriable: Option<bool>`, `retry_after_ms: Option<i64>`, `policy_label: Option<String>`
  - ✅ Retry headers (`Retry-After`, `X-Backoff-Ms`) emitted for queue full errors
  - File: `contracts/api-types/src/generated.rs`, `src/domain/error.rs`

### Observability

- [ ] **OC-CTRL-2050**: Admission logs include queue metadata
  - Current: Logs `queue_position` and `predicted_start_ms`
  - Verify: Structured JSON with all required fields
  - File: `src/api/data.rs::create_task()`
  - Ref: `.specs/20-orchestratord.md#6-observability`

- [ ] **OC-CTRL-2051**: Metrics coverage
  - Need: Queue depth, reject/drop rates, latency percentiles, error counts by class
  - File: `src/metrics.rs`, `src/api/data.rs`, `src/services/streaming.rs`
  - Verify: Matches `ci/metrics.lint.json`
  - Ref: `.specs/20-orchestratord.md#6-observability`

- [x] **OC-CTRL-2052**: Correlation ID handling (MOSTLY COMPLETE)
  - ✅ Middleware extracts or generates UUIDv4
  - ✅ Attached to request extensions
  - ✅ Added to all responses (including auth failures)
  - ⚠️ TODO: Verify included in SSE response headers
  - ⚠️ TODO: Extract from extensions in logging sites
  - File: `src/app/middleware.rs::correlation_id_layer()`
  - Ref: `.specs/20-orchestratord.md#6-observability`

- [x] **ORCHD-METRICS-1201**: Metrics infrastructure (COMPLETE)
  - ✅ Full metrics system in `src/metrics.rs` (counters, gauges, histograms)
  - ✅ Label support and throttling for high-cardinality metrics
  - ✅ Pre-registration of common metrics
  - ✅ Prometheus text format export at `/metrics`
  - ⚠️ TODO: Add more call sites (queue depth updates, error counters)
  - File: `src/metrics.rs`
  - Ref: Inline TODO comment

### Budgets & Sessions

- [ ] **OC-CTRL-2068**: Per-session budget enforcement
  - Current: Budget headers computed from session info (best-effort)
  - Need: Enforce at admission/scheduling time
  - Need: Surface budget state via SSE `metrics` frames and/or response headers
  - File: `src/api/data.rs`, `src/services/session.rs`
  - Ref: `.specs/20-orchestratord.md#10-budgets-guardrails`

- [ ] **ORCHD-BUDGETS-3001**: Real budget policy
  - Current: Best-effort lookup
  - Need: Compute from real budget policy and session linkage
  - File: `src/api/data.rs::create_task()`
  - Ref: Inline TODO comment

### Artifacts

- [ ] **OC-CTRL-2065**: Artifact persistence with CAS
  - Current: Basic implementation exists
  - Verify: Content-addressed IDs (SHA-256), tags, OpenAPI schema
  - File: `src/api/artifacts.rs`, `src/services/artifacts.rs`
  - Ref: `.specs/20-orchestratord.md#9-artifact-registry-optional-recommended`

- [ ] **OC-CTRL-2066**: Artifact retrieval with metadata
  - Current: Basic get implemented
  - Need: Include metadata (tags, lineage, timestamps)
  - File: `src/api/artifacts.rs`
  - Ref: `.specs/20-orchestratord.md#9-artifact-registry-optional-recommended`

- [ ] **OC-CTRL-2067**: Job artifact records
  - Need: Each job produces artifact with: `job_id`, `session_id`, request params (redacted), metrics, SSE transcript
  - Need: Failure paths include error context and partial transcripts
  - File: `src/services/streaming.rs`, `src/services/artifacts.rs`
  - Ref: `.specs/20-orchestratord.md#9-artifact-registry-optional-recommended`

---

## Medium Priority 🟡

### Configuration & Bootstrap

- [ ] **ORCHD-CONFIG-VALIDATE-0001**: Load and validate orchestrator config
  - Need: Config schema for GPU pools, placement policy
  - Need: Wire reload/drain lifecycle
  - File: `src/app/bootstrap.rs`
  - Ref: Inline TODO comment

- [ ] **OwnerB-ORCH-BINDING-SHIM**: Replace MVP adapter binding shim
  - Current: Feature gate + env vars (`ORCHD_LLAMACPP_URL`)
  - Need: Pool-manager driven registration, config-schema backed sources
  - Need: Reload/drain lifecycle integration with AdapterHost
  - File: `src/app/bootstrap.rs::build_app()`
  - Ref: Inline TODO comment

### SSE Metrics Frames

- [ ] **OC-CTRL-2023**: SSE metrics with scheduling signals
  - Current: Basic metrics frame
  - Need: `on_time_probability`, `queue_depth`, `kv_warmth`
  - File: `src/services/streaming.rs`
  - Ref: `.specs/20-orchestratord.md#11-sse-metrics-scheduling-signals`

### OpenAPI Examples

- [ ] **OC-CTRL-2067**: Data-plane `x-examples` in OpenAPI
  - Need: Examples for enqueue, stream/SSE frames, cancel, sessions
  - File: `contracts/openapi/data.yaml`
  - Ref: `.specs/20-orchestratord.md#12-openapi-examples-annotations`

- [ ] **OC-CTRL-2069**: Control-plane `x-examples` in OpenAPI
  - Need: Examples for drain, reload, capabilities
  - File: `contracts/openapi/control.yaml`
  - Ref: `.specs/20-orchestratord.md#12-openapi-examples-annotations`

### Security

- [ ] **OC-CTRL-2040**: Document AuthN/AuthZ policy
  - Current: Home-profile has no AuthN/AuthZ (open locally)
  - Verify: Documented clearly
  - Future: May introduce AuthN/AuthZ behind features
  - File: README.md, `.specs/20-orchestratord.md`
  - Ref: `.specs/20-orchestratord.md#5-security`

- [ ] **OC-CTRL-2041**: Secret redaction in logs
  - Current: `http-util` has redaction
  - Verify: Orchestrator logs never leak API keys, adapter tokens
  - File: All logging sites
  - Ref: `.specs/20-orchestratord.md#5-security`

### Optional Features

- [ ] **OC-CTRL-2063**: Output mode hint support
  - Optional: `TaskRequest.output_mode: "text" | "json" | "edits"`
  - Need: Hint for artifact tagging, ignore unknown values
  - File: `contracts/api-types/src/generated.rs`, `src/api/data.rs`
  - Ref: `.specs/20-orchestratord.md#81-optional-output-mode-hint`

- [ ] **OC-CTRL-2070**: CORS support (optional)
  - Optional: For localhost tooling
  - Need: Reply to `OPTIONS` with appropriate headers
  - Must: Disabled by default, non-breaking when enabled
  - File: `src/app/middleware.rs`, `src/app/router.rs`
  - Ref: `.specs/20-orchestratord.md#13-cors-preflight-optional`

---

## Low Priority / Future 🟢

### Determinism & Reliability

- [ ] **OwnerB-ORCH-SSE-FALLBACK**: Remove or gate deterministic SSE fallback
  - Current: Falls back to deterministic SSE when no adapter bound
  - Future: Replace with full admission→dispatch→stream path exclusively
  - Alternative: Gate behind dev/testing feature flag
  - File: `src/services/streaming.rs::render_sse_for_task()`
  - Ref: Inline TODO comment

- [ ] Graceful shutdown with in-flight stream handling
  - Need: Drain semantics for shutdown
  - File: `src/app/bootstrap.rs::start_server()`
  - Ref: CHECKLIST.md

- [ ] Idempotence for task creation
  - Optional: Where applicable
  - File: `src/api/data.rs::create_task()`
  - Ref: CHECKLIST.md

### Testing Gaps

- [ ] Provider verification tests
  - Current: Test file exists
  - Verify: Green against OpenAPI
  - File: `tests/provider_verify.rs`
  - Ref: CHECKLIST.md

- [ ] BDD features coverage
  - Need: Local BDD features green for data/control/SSE/security
  - File: `bdd/` directory
  - Ref: CHECKLIST.md

- [ ] Determinism suite
  - Need: Mock engines pass
  - Need: Document real engine gaps
  - File: Test harness
  - Ref: CHECKLIST.md

- [ ] Metrics lint
  - Need: Pass locally and in CI
  - File: `ci/metrics.lint.json`
  - Ref: CHECKLIST.md

### Documentation

- [ ] README High/Mid/Low sections
  - Current: Basic sections exist
  - Need: Match code, keep current, richer descriptions
  - File: `README.md`
  - Ref: CHECKLIST.md, user memory

- [ ] Spec Refinement Opportunities sections
  - Need: Every `.specs/*.md` includes actionable follow-ups
  - File: `.specs/20-orchestratord.md` (has section)
  - Ref: User memory

### Operations

- [ ] Config via env or file
  - Need: Listen addr, auth policy, artifact root, exporters
  - File: `src/app/bootstrap.rs`
  - Ref: CHECKLIST.md

- [ ] Health/readiness endpoints
  - Current: `/metrics` exists
  - Need: Dedicated health/readiness for service management
  - File: `src/api/observability.rs`
  - Ref: CHECKLIST.md

- [ ] Release artifacts and versioning
  - Need: Document CHANGELOG pointers
  - File: Root CHANGELOG.md
  - Ref: CHECKLIST.md

---

## Notes

- **Spec-first workflow**: Update `.specs/` and `contracts/` before runtime code
- **No backwards compat pre-1.0**: Breaking changes allowed until v1.0.0
- **Determinism by default**: Follow Spec → Contract → Tests → Code
- **Proof bundles**: Document investigations in `.docs/`, ship with behavior changes
- **TODO.md maintenance**: Update after each change, archive via `ci/scripts/archive_todo.sh` when done

## Verification Commands

```bash
# Format and lint
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Tests
cargo test -p orchestratord -- --nocapture
cargo test -p orchestratord --test provider_verify -- --nocapture

# Regenerate artifacts
cargo xtask regen-openapi
cargo xtask regen-schema

# Dev loop (all checks)
cargo xtask dev:loop
```

## References

- Spec: `.specs/20-orchestratord.md`
- Requirements: `requirements/orchestratord.yaml`
- Checklist: `CHECKLIST.md`
- OpenAPI: `contracts/openapi/{control.yaml,data.yaml}`
- Metrics: `ci/metrics.lint.json`
