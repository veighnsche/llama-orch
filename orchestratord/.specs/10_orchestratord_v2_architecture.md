# Orchestratord v2 — Architecture and Crate Structure (Proposal)

Status: proposal
Last updated: 2025-09-17
Authors: Orchestratord maintainers

## 0. Purpose and Golden Rules
The goal of this document is to define a clean, modular architecture and crate structure for a full rewrite of `orchestratord`, aligning strictly with workspace specs and contracts.

Golden rules (from README_LLM and root specs):
- Spec → Contract → Tests → Code. No code without prior spec/contract updates.
- No backwards compatibility pre-1.0.0. Legacy paths (e.g., `/v1/replicasets`) MUST NOT be served.
- Determinism by default and strong observability are first-class.

Non-goals:
- Multi-host clustering (single host reference environment only).
- Implementing model engines (handled by worker adapters).

## 1. Responsibilities (Scope)
- Data-plane admission & lifecycle (enqueue, stream, cancel).
- Session lifecycle & budgets (tokens/time/cost).
- Artifact registry (content-addressed documents, local persistence).
- Capability discovery via `/v1/capabilities`.
- Control plane (pools drain/reload/health; simple lifecycle states).
- Observability: metrics endpoint, correlation IDs, structured logs.

## 2. External Contracts and Dependencies
- OpenAPI: `contracts/openapi/control.yaml`, `contracts/openapi/data.yaml`.
- Cross-crate integrations:
  - `orchestrator-core`: queue/backpressure policies and metrics helpers.
  - `pool-managerd`: pool registry (health/readiness/drain/reload, last_error).
  - `worker-adapters/*`: adapter API and concrete adapters (mock, llamacpp, vllm, tgi, triton).
  - `contracts-api-types`: shared API types for request/response bodies.

## 3. High-level Architecture (Ports & Services)
Layered architecture with clear seams:

- API Layer (Axum handlers, request/response mapping):
  - `api::data`, `api::control`, `api::observability`, `api::artifacts`, `api::catalog`.
  - Thin, stateless; delegates to Services.
- Services Layer (domain orchestration):
  - `services::admission` — admission checks, enqueue, ETA estimates.
  - `services::streaming` — SSE streaming orchestration, transcript capture, cancel propagation.
  - `services::session` — session registry, TTL/turns, budgets, eviction.
  - `services::artifacts` — content-addressed storage (inmem + fs backends).
  - `services::capabilities` — compile capability snapshot.
  - `services::control` — drain/reload/health using Pool port.
  - `services::catalog` — stub CRUD/verify with trust policy enforcement.
- Ports (interfaces to external systems):
  - `ports::adapters::AdapterRegistry` — list adapters and per-adapter `props()` and `stream()`/`infer()`.
  - `ports::pool::PoolRegistry` — get/set pool state, drain/reload, health.
  - `ports::clock::Clock` — abstract time for testing.
  - `ports::storage::ArtifactStore` — abstraction over artifact persistence.
- Infrastructure (adapters and stores):
  - `infra::adapters::mock`, `infra::adapters::http::*` (llamacpp/vllm/tgi/triton).
  - `infra::storage::inmem`, `infra::storage::fs`.
  - `infra::metrics` — Prometheus registry and metric helpers.
  - `infra::clock::SystemClock`.
- Domain:
  - `domain::sse` (events, typed frames), `domain::error` (typed errors and HTTP mapping), `domain::ids`.

### 3.1 SSE Emitter Design (normative per root spec)

- Transport preference: enable HTTP/2 where supported for SSE; gracefully fallback to HTTP/1.1 (no behavior change). Compression is typically disabled for small token frames.
- Buffered writer: the SSE encoder MUST use a buffered writer and avoid per-token heap allocations on the hot path.
- Optional micro-batch: a bounded, disabled-by-default coalescing mode MAY group tokens within a small latency budget to reduce syscalls; ordering remains `started → token* → end` with optional `metrics` interleaves.
- CPU budget: keep encoder CPU usage small and predictable; avoid allocations and JSON re-serialization per token; prefer pre-encoded slices where feasible.
- Narration hooks: emit short, human-readable narration strings alongside structured logs at key events (admission, placement, stream start/end, cancel) per `/.specs/00_llama-orch.md §2.8.1`.

## 4. Proposed Crate Structure
```
orchestratord/
  src/
    app/
      bootstrap.rs        # server init, feature flags, metrics init
      router.rs           # axum Router construction, routes and middleware wiring
      middleware.rs       # auth (X-API-Key), correlation-id, error mapping
    api/
      control.rs          # POST /v1/pools/:id/drain|reload, GET /v1/pools/:id/health
      data.rs             # POST /v1/tasks, GET /v1/tasks/:id/stream, POST /v1/tasks/:id/cancel
      artifacts.rs        # POST /v1/artifacts, GET /v1/artifacts/:id
      catalog.rs          # POST /v1/catalog/models (stubs) + verify/state
      observability.rs    # GET /metrics
      types.rs            # helper extractors, headers, shared response helpers (thin)
    services/
      admission.rs        # AdmissionService
      streaming.rs        # StreamingService (SSE), cancel propagation
      session.rs          # SessionService (TTL, turns, budgets)
      artifacts.rs        # ArtifactService
      capabilities.rs     # CapabilitiesService
      control.rs          # ControlService
      catalog.rs          # CatalogService (trust policy checks)
    ports/
      adapters.rs         # trait AdapterRegistry + AdapterClient
      pool.rs             # trait PoolRegistry
      clock.rs            # trait Clock
      storage.rs          # trait ArtifactStore
    infra/
      adapters/
        mock.rs           # default mock adapters for vertical slice
        http/
          llamacpp.rs
          vllm.rs
          tgi.rs
          triton.rs
      storage/
        inmem.rs
        fs.rs
      metrics.rs          # Prometheus registry, lints compliance helpers
      clock.rs            # SystemClock
    domain/
      error.rs            # OrchestratorError -> http::StatusCode
      sse.rs              # typed SSE events and encoding helpers
      ids.rs              # newtype wrappers for TaskId, SessionId, ArtifactId
    state.rs              # AppState aggregation: ports, services, config, metrics
    lib.rs
    main.rs
```

Notes:
- Legacy `http::handlers` shim is deleted; no shims pre-1.0.
- `/v1/replicasets` is removed. Discovery must use `/v1/capabilities`.
- `contracts-api-types` continue to define public request/response bodies; `api/*` maps to/from them.

## 5. Key Interfaces (Signatures — Rust-ish pseudocode)

Ports:
```rust
pub trait AdapterRegistry: Send + Sync {
    fn engines(&self) -> Vec<String>; // e.g., ["llamacpp","vllm",...]
    fn props(&self, engine: &str) -> anyhow::Result<AdapterProps>;
    fn stream(
        &self,
        engine: &str,
        req: StreamRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<AdapterStream>; // yields tokens & metrics
}

pub trait PoolRegistry: Send + Sync {
    fn health(&self, pool_id: &str) -> anyhow::Result<PoolHealth>;
    fn drain(&self, pool_id: &str, deadline_ms: u64) -> anyhow::Result<()>;
    fn reload(&self, pool_id: &str, new_model_ref: &str) -> anyhow::Result<ReloadOutcome>;
}

pub trait ArtifactStore: Send + Sync {
    fn put(&self, doc: Artifact) -> anyhow::Result<ArtifactId>;
    fn get(&self, id: &ArtifactId) -> anyhow::Result<Artifact>;
}

pub trait Clock: Send + Sync { fn now_ms(&self) -> u64; }
```

Services:
```rust
pub struct AdmissionService<'a> { /* ports: PoolRegistry, OrchestratorCore, Clock */ }
impl<'a> AdmissionService<'a> {
    pub fn enqueue(&self, req: TaskRequest) -> Result<AdmissionResult, OrchestratorError>;
}

pub struct StreamingService<'a> { /* ports: AdapterRegistry, ArtifactStore, SessionService, Clock */ }
impl<'a> StreamingService<'a> {
    pub async fn stream(
        &self,
        task_id: TaskId,
        params: StreamParams,
        cancel: CancellationToken,
    ) -> Result<SseTranscript, OrchestratorError>;
}

pub struct SessionService { /* inmem + eviction */ }
impl SessionService {
    pub fn get(&self, session_id: &SessionId) -> Option<SessionInfo>;
    pub fn delete(&self, session_id: &SessionId) -> bool;
}

pub struct CapabilitiesService<'a> { /* ports: AdapterRegistry */ }
impl<'a> CapabilitiesService<'a> {
    pub fn snapshot(&self) -> CapabilityPayload; // includes api_version
}

pub struct ControlService<'a> { /* ports: PoolRegistry */ }
impl<'a> ControlService<'a> { /* drain, reload, health */ }
```

API-Middleware:
```rust
// app/middleware.rs
// - Auth: require X-API-Key
// - Correlation: echo X-Correlation-Id or generate
// - Error mapping: OrchestratorError -> (StatusCode, Json)
```

## 6. HTTP Surface (Explicit, No Back-Compat)
- Data plane:
  - `POST /v1/tasks` → 202 {task_id, queue_position, predicted_start_ms}; budget headers.
  - `GET  /v1/tasks/:id/stream` → SSE: started, token, metrics?, end, error.
  - `POST /v1/tasks/:id/cancel` → race-free cancel; no tokens after cancel.
  - Sessions: `GET /v1/sessions/:id`, `DELETE /v1/sessions/:id`.
- Artifacts:
  - `POST /v1/artifacts` (201), `GET /v1/artifacts/:id` (200/404).
- Capability discovery:
  - `GET /v1/capabilities` only.
- Control:
  - `POST /v1/pools/:id/drain` (202), `POST /v1/pools/:id/reload` (200/409), `GET /v1/pools/:id/health` (200).
- Observability:
  - `GET /metrics` Prometheus v0.0.4; always includes `X-Correlation-Id`.

## 7. Determinism, Budgets, Backpressure
- Determinism: propagate `seed`, pin `engine_version` and `sampler_profile_version` per pool.
- Budgets: tokens/time/cost enforced at admission when configured; surfacing remaining budgets in SSE metrics frames.
- Backpressure: return 429 with `Retry-After` and `X-Backoff-Ms`; include error envelope with `policy_label` and advisories.

## 8. Error Taxonomy and Mapping
- Codes: `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNREADY`, `POOL_UNAVAILABLE`, `REPLICA_EXHAUSTED`, `DEADLINE_UNMET`, `INTERNAL`.
- Mapping: 400 (invalid params/deadline), 404 (artifact missing), 409 (reload conflict), 429 (backpressure), 503 (pool unavailable), 5xx (internal).

## 9. Metrics and Logging
- Metrics must comply with `ci/metrics.lint.json`.
- SSE metrics frames include additive JSON: `queue_depth`, `on_time_probability`, `kv_warmth`, remaining budgets.
- Logs: include `job_id`, `session_id`, `engine`, `engine_version`, `sampler_profile_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`.

## 10. Configuration
- `X-API-Key` auth (stub); on 401/403 errors ensure correlation id present.
- Budget defaults; artifact storage path for fs store; optional OTEL exporters.

## 11. Testing Strategy
- Unit tests per API and Service module.
- Provider verification against OpenAPI.
- BDD features for SSE framing & control plane.
- Determinism suite (byte-exact for engines in single-slot mode).
- Metrics linter.

## 12. Migration & Work Plan
Phased refactor (src-old is reference only; do not import code):

1) Scaffolding
- Create `app::{bootstrap,router,middleware}` and `api::*` with routes matching OpenAPI.
- Wire thin handlers to `todo!()` service calls; compile behind feature flags.

2) Services and Ports
- Define traits in `ports::*`; create in-memory `infra` implementations for: `ArtifactStore`, `Clock`.
- Implement `SessionService`, `CapabilitiesService`, `ControlService` (using `pool-managerd`).

3) Data Plane Core
- Implement `AdmissionService` using `orchestrator-core` and `pool-managerd`.
- Implement `StreamingService` wired to `worker-adapters` mock; add cancel token support.

4) Observability & Artifacts
- Implement Prometheus metrics registry; populate counters/gauges/histograms per spec.
- Implement artifact persistence (inmem + fs backend) and SSE transcript capture.

5) Hardening & Tests
- Fill out error mapping; BDD and provider verify must pass.
- Determinism suite for mock engine; document gaps for real engines.

6) Cleanup & Removal
- Ensure no legacy shims, no `/v1/replicasets`, and doc parity.

## 13. Directory Diff vs v1
- v1 (src-old) co-located handlers and services; v2 separates API, Services, Ports, and Infra clearly.
- v2 adds explicit ports for clock/storage/adapters/pool and domain error/SSE modules.

## 14. Open Questions / Future Work
- Policy engine hooks (allow/deny outbound HTTP tooling per deployment).
- NVML/GPU telemetry integration for richer metrics.
- Persistent session store and budget snapshots across restarts.

## 15. Acceptance Criteria
- All OpenAPI control/data paths implemented; `/v1/replicasets` absent.
- Provider verify, BDD, metrics lints pass.
- Code compiles with `default` features and runs server.
- Capability snapshot includes `api_version`; concurrency added when available.
