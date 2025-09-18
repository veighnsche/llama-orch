# Orchestratord TODO (Per File) — Feature Parity With src-old

Goal: Complete all stubs and ensure that every capability present in `orchestratord/src-old/` is available (and improved) in the new `orchestratord/src/` layout. Each item below is actionable and references the legacy source for parity.

Legend:
- [ ] pending
- [x] done (as of this commit)
- (src-old → reference) indicates where to look for legacy behavior

## Root (src/)

### `src/lib.rs`
- [ ] Re-export public modules intentionally (API surface definition) and ensure doc comments link to OpenAPI contracts.
- [ ] Gate modules with feature flags as needed (e.g., `server`, `metrics`, `artifacts`) to match workspace policy.
- [ ] Add crate-level docs that reference `.specs/00_orchestratord.md` and `.specs/10_orchestratord_v2_architecture.md`.
- (src-old → `src-old/lib.rs`)

### `src/main.rs`
- [x] Initialize observability and serve router.
- [ ] Add graceful shutdown (SIGINT/SIGTERM) and drain behavior.
- [ ] Read config/env for listen addr, auth mode, and feature flags.
- [ ] Emit startup logs with version, feature flags, and metric endpoints.
- (src-old → `src-old/main.rs`)

### `src/state.rs`
- [x] Basic shared state (logs, sessions, artifacts, pool_manager, draining flags).
- [ ] Replace coarse `Mutex` with finer-grained `RwLock` where read-heavy.
- [ ] Add handles for: `AdapterRegistry`, `ArtifactStore` (trait-backed), optional catalog cache.
- [ ] Background tasks: session TTL eviction and draining progression (if applicable).
- (src-old → `src-old/state.rs`)

### `src/metrics.rs`
- [x] Minimal in-crate registry with counters/gauges/histograms.
- [ ] Replace with real Prometheus/OTel integration or bridge existing registry to exporter.
- [ ] Enforce label budgets and sensible defaults; add unit tests for cardinality.
- (src-old → `src-old/metrics.rs`)

### `src/admission.rs`
- [x] Queue wrapper with metrics and policies (reject/drop-lru) using `orchestrator_core`.
- [ ] Integrate with budgeting (tokens/time/cost) and priority queues per spec.
- [ ] Emit backpressure metrics and reasons aligned with taxonomy.
- [ ] Wire to data API: admission returns queue position, ETA, and backoff hints.
- (src-old → `src-old/backpressure.rs`, `src-old/admission.rs`)

## API Layer (src/api/)

### `src/api/mod.rs`
- [x] Module declarations.
- [ ] Ensure visibility and re-exports documented for downstream crates/tests.

### `src/api/types.rs`
- [x] Basic helpers for API key and correlation id.
- [ ] Remove duplicate logic now in middleware; keep only API-specific helpers.
- [ ] Add helpers for parsing budget headers as defined in OpenAPI (tokens/time/cost/session TTL), with validation.
- (src-old → `src-old/http/auth.rs`)

### `src/api/data.rs`
- [x] Handlers for: create task, stream SSE, cancel task, session get/delete.
- [ ] Replace sentinels with real logic: integrate `admission::QueueWithMetrics` and budget headers.
- [ ] Stream via selected engine adapter (see `ports/adapters.rs`) and propagate cancel tokens.
- [ ] Ensure `X-Backoff-Ms` and `Retry-After` from actual queue state; shape SSE frames per spec (metrics, started, token, end).
- [ ] Structured logging for admission and started frames (no secrets); add tracing spans using correlation-id.
- [ ] Full error taxonomy mapping for all branches; remove stub “cause-internal” pathway.
- (src-old → `src-old/http/data.rs`)

### `src/api/control.rs`
- [x] Capabilities, pool health/drain/reload handlers.
- [ ] Implement atomic drain and reload behavior via `PoolRegistry` (deadline honoring, rollback on failure) with durable state.
- [ ] Populate health `metrics` with real queue depth and replica info; include last_error details.
- [ ] Authorization policy for control plane endpoints if configured.
- (src-old → `src-old/http/control.rs`)

### `src/api/artifacts.rs`
- [x] Content-addressed artifact creation and retrieval (in-memory map).
- [ ] Switch to `ports::storage::ArtifactStore` and `infra::storage::{inmem, fs}` backend depending on config.
- [ ] Enforce size limits, content types, and retention policies; include ETag/If-None-Match semantics where useful.
- (src-old → `src-old/http/artifacts.rs`)

### `src/api/observability.rs`
- [x] `/metrics` endpoint returns Prometheus text; seeds series for lints.
- [ ] Replace seed with gather from real registry; include build info/version.
- [ ] Add `/healthz` and `/readyz` endpoints if not covered elsewhere (optional).
- (src-old → `src-old/http/observability.rs`)

## App Layer (src/app/)

### `src/app/bootstrap.rs`
- [x] Builds router and initializes observability.
- [ ] Implement `start_server()` or remove stub; centralize server bootstrap here.
- [ ] Inject constructed state: `AdapterRegistry`, `ArtifactStore`, `PoolRegistry`, config.

### `src/app/router.rs`
- [x] Routes mounted for control, data, sessions, artifacts, and metrics.
- [ ] Validate route guards and method sets against OpenAPI; add unit test asserting no extra paths.

### `src/app/middleware.rs`
- [x] `correlation_id_layer` and `api_key_layer` implemented with basic policy.
- [ ] Make API key policy configurable (allowlist, issuer verification); add rate-limits and request-size limits.
- [ ] Add structured error mapping layer and tracing spans; ensure correlation-id in all responses.
- (src-old → `src-old/http/auth.rs`)

## Domain (src/domain/)

### `src/domain/error.rs`
- [x] Error taxonomy mapping to envelopes and HTTP codes; advisory headers for backpressure.
- [ ] Populate `engine` field dynamically from selected adapter/pool when applicable (not a stub).
- [ ] Include correlation id in error body or ensure via headers consistently (spec decision).
- (src-old → `src-old/errors.rs`)

### `src/domain/sse.rs`
- [x] Basic SSE event enum.
- [ ] Align event payloads with final spec and adapter outputs; provide serializer helpers.
- (src-old → `src-old/sse.rs`)

### `src/domain/ids.rs`
- [x] Strong ID newtypes.
- [ ] Add validation helpers and parsing from headers/paths where appropriate.

## Infra (src/infra/)

### `src/infra/metrics.rs`
- [x] Facade for metrics module.
- [ ] Optionally integrate Prometheus registry or OTel exporter; provide feature-gated backends.

### `src/infra/clock.rs`
- [x] System clock implementation.
- [ ] Add mock/monotonic test clock (if not already elsewhere) with drift controls.

### `src/infra/storage/inmem.rs`
- [x] In-memory ArtifactStore (content-addressed by sha256).
- [ ] Add list/query operations and retention policies.

### `src/infra/storage/fs.rs`
- [ ] Implement filesystem ArtifactStore: CAS directory layout, atomic writes, checksum verification, and GC.
- [ ] Configurable root path and disk quotas; error handling and permissions.
- (src-old → `src-old/http/artifacts.rs` for behavior expectations)

## Ports (src/ports/)

### `src/ports/adapters.rs`
- [x] Traits for `AdapterRegistry` and `AdapterClient`, and stream item enum.
- [ ] Expand `AdapterClient` to support workloads: completion, embedding, rerank; include properties and errors.
- [ ] Add async streaming interface with backpressure and cancellation.
- [ ] Provide `AdapterRegistry` impl in infra (future) and wire via app bootstrap.
- (src-old → implicit in data/http handlers; adapter layer was inlined before)

### `src/ports/pool.rs`
- [x] PoolRegistry trait and outcomes.
- [ ] Extend with replicas, load, and health metrics; timeouts and error taxonomy mapping.
- [ ] Implement adapter discovery per pool, model mapping, and drain progress.
- (src-old → `src-old/placement.rs`, `src-old/http/control.rs`)

### `src/ports/storage.rs`
- [x] ArtifactStore trait.
- [ ] Add streaming/attachment interfaces and metadata.

### `src/ports/clock.rs`
- [x] Clock trait.

## Services (src/services/)

### `src/services/session.rs`
- [x] Session get/create, tick/delete, and bookkeeping of turns.
- [ ] Replace fixed 100ms tick with last_seen timestamps and real TTL logic; enforce session budgets from headers.
- [ ] Expose structured metrics for session counts and TTL expirations.
- (src-old → `src-old/session.rs`)

### `src/services/streaming.rs`
- [x] Deterministic SSE rendering; persists transcript artifact.
- [ ] Replace string-built SSE with streaming (e.g., axum SSE or chunked) and connect to AdapterClient stream.
- [ ] Record tokens_in/out and latency metrics from actual stream; propagate cancellations.
- [ ] Ensure per-stream ordering and include admission context in `started`.
- (src-old → `src-old/http/data.rs` SSE sections)

### `src/services/capabilities.rs`
- [x] Static snapshot of engines and features.
- [ ] Build from `AdapterRegistry` and `PoolRegistry` at runtime; include ctx_max, workloads, and dynamic flags.
- (src-old → `src-old/http/catalog.rs` for capabilities/catelog aspects)

### `src/services/artifacts.rs`
- [ ] Implement service on top of `ports::storage::ArtifactStore` to decouple API from storage backend; add indexing.

### `src/services/catalog.rs` (stub)
- [ ] Implement model catalog (list/query) from pools and adapters; ensure compatibility with CLI consumer tests.
- (src-old → `src-old/http/catalog.rs`)

### `src/services/control.rs` (stub)
- [ ] Encapsulate control plane orchestration (drain/reload/health) and leave API thin; include retries/backoff.
- (src-old → `src-old/http/control.rs`)

### `src/services/mod.rs`
- [x] Module wiring.
- [ ] Consider feature flags to include/exclude services.

## Cross-Cutting Tasks
- [ ] Remove remaining stubs (seed metrics in `api/observability.rs`, config-less API key acceptance).
- [ ] Replace sentinel conditions in APIs with real orchestration logic.
- [ ] Ensure all responses carry `X-Correlation-Id` via middleware; add tests.
- [ ] Add rate limiting and body size limits as per spec.
- [ ] Port any missing metrics from `src-old/metrics.rs` and align label sets with `ci/metrics.lint.json`.
- [ ] Verify OpenAPI contract coverage for every endpoint/field (add tests if gaps appear).
- [ ] Remove `src-old/` after parity is confirmed and docs updated.

## Parity Map (src-old → new)
- `src-old/http/data.rs` → `src/api/data.rs` + `src/services/streaming.rs` + `src/admission.rs`
- `src-old/http/control.rs` → `src/api/control.rs` + `src/services/control.rs` + `src/ports/pool.rs`
- `src-old/http/artifacts.rs` → `src/api/artifacts.rs` + `src/ports/storage.rs` + `src/infra/storage/*`
- `src-old/http/observability.rs` → `src/api/observability.rs` + `src/infra/metrics.rs`
- `src-old/http/auth.rs` → `src/app/middleware.rs` + `src/api/types.rs`
- `src-old/metrics.rs` → `src/metrics.rs` (plus exporter integration TBD)
- `src-old/session.rs` → `src/services/session.rs`
- `src-old/backpressure.rs`/`src-old/admission.rs` → `src/admission.rs`
- `src-old/sse.rs` → `src/domain/sse.rs`
- `src-old/state.rs` → `src/state.rs`

## Acceptance Criteria
- [ ] All BDD scenarios continue to pass (no sentinel shortcuts).
- [ ] Provider verification tests pass and cover new/updated fields.
- [ ] Metrics linter passes with required names and label sets.
- [ ] No references to `src-old/` remain in code or docs after parity.
- [ ] README and `.specs/` updated to reflect implemented features.
