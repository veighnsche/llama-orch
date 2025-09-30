# orchestratord Binary Analysis & Modularization Proposal

**Date**: 2025-10-01  
**Total Lines**: 3,382 lines across 46 files  
**Status**: Analysis for potential modularization

---

## Current Structure

### Module Breakdown

```
bin/orchestratord/src/
├── admission.rs (105 lines) — Task admission logic
├── api/ (9 files, ~1,500 lines) — HTTP endpoint handlers
│   ├── artifacts.rs — Artifact registry endpoints
│   ├── catalog.rs — Model catalog CRUD
│   ├── catalog_availability.rs (179 lines) — Multi-node catalog queries
│   ├── control.rs (124 lines) — Pool control (drain/reload)
│   ├── data.rs (236 lines) — Task admission + streaming
│   ├── nodes.rs (448 lines) — Cloud profile node management
│   ├── observability.rs (141 lines) — Metrics, health, capabilities
│   └── types.rs — API-specific types
├── app/ (5 files, ~350 lines) — Application bootstrap
│   ├── auth_min.rs (105 lines) — Bearer token middleware
│   ├── bootstrap.rs (84 lines) — App initialization
│   ├── middleware.rs — Request middleware
│   ├── router.rs (51 lines) — Route assembly
│   └── mod.rs
├── clients/ (2 files, ~110 lines) — External service clients
│   └── pool_manager.rs (105 lines) — Pool manager HTTP client
├── domain/ (4 files, ~200 lines) — Domain types
│   ├── error.rs (115 lines) — Error taxonomy
│   ├── ids.rs — ID types
│   ├── sse.rs — SSE event types
│   └── mod.rs
├── infra/ (6 files, ~150 lines) — Infrastructure adapters
│   ├── clock.rs — Time abstraction
│   ├── metrics.rs — Metrics infrastructure
│   └── storage/ — Filesystem + in-memory storage
├── ports/ (5 files, ~100 lines) — Port traits (hexagonal architecture)
│   ├── adapters.rs — Worker adapter port
│   ├── clock.rs — Clock port
│   ├── pool.rs — Pool manager port
│   └── storage.rs — Storage port
├── services/ (9 files, ~1,200 lines) — Business logic
│   ├── artifacts.rs — Artifact service
│   ├── capabilities.rs — Capabilities service
│   ├── catalog.rs — Catalog service
│   ├── control.rs — Control service
│   ├── placement.rs (45 lines) — Legacy placement
│   ├── placement_v2.rs (302 lines) — Model-aware placement
│   ├── session.rs (67 lines) — Session management
│   └── streaming.rs (391 lines) — SSE streaming
├── state.rs (187 lines) — AppState and shared state
└── metrics.rs (199 lines) — Prometheus metric definitions
```

### File Size Distribution

**Largest files** (top 10):
1. `api/nodes.rs` — 448 lines (Cloud profile node endpoints)
2. `services/streaming.rs` — 391 lines (SSE streaming logic)
3. `services/placement_v2.rs` — 302 lines (Model-aware placement)
4. `api/data.rs` — 236 lines (Task admission + events)
5. `metrics.rs` — 199 lines (Prometheus metrics)
6. `state.rs` — 187 lines (AppState)
7. `api/catalog_availability.rs` — 179 lines (Catalog queries)
8. `api/observability.rs` — 141 lines (Health/metrics/caps)
9. `api/control.rs` — 124 lines (Pool control)
10. `api/catalog.rs` — 118 lines (Catalog CRUD)

---

## What orchestratord Does

### 1. **HTTP API Server** (Axum)
- Binds to `0.0.0.0:8080` (configurable via `ORCHD_ADDR`)
- Serves OpenAPI v2 endpoints (data plane, control plane, artifacts, observability)
- Handles Bearer token authentication (cloud profile)
- Routes: `/v2/tasks`, `/v2/nodes`, `/v2/catalog`, `/v2/pools`, `/metrics`

### 2. **Task Admission & Queueing**
- Accepts `POST /v2/tasks` requests
- Validates context, budgets, model availability
- Enqueues to `orchestrator-core` queue
- Returns `202 Accepted` with `AdmissionResponseV2`

### 3. **SSE Streaming**
- Serves `GET /v2/tasks/{id}/events` (text/event-stream)
- Dispatches jobs to worker adapters
- Streams: `started → token → metrics → end` (or `error`)
- Handles cancellation and correlation IDs

### 4. **Cloud Profile Management**
- Node registration (`POST /v2/nodes/register`)
- Heartbeat tracking (`POST /v2/nodes/{id}/heartbeat`)
- Node deregistration (`DELETE /v2/nodes/{id}`)
- Service registry integration (node health, pool status)

### 5. **Model-Aware Placement**
- Filters pools by model availability
- Placement strategies: round-robin, least-loaded
- Queries catalog availability across nodes
- Selects pools based on slots, VRAM, model presence

### 6. **Session Management**
- TTL tracking (default 10 minutes)
- Budget enforcement (tokens, time, cost)
- KV warmth metadata
- Session cleanup

### 7. **Catalog Management**
- Model CRUD (`POST /v2/catalog/models`, `GET`, verify, state transitions)
- Multi-node catalog availability (`GET /v2/catalog/availability`)
- Integration with `catalog-core` library

### 8. **Pool Control**
- Health checks (`GET /v2/pools/{id}/health`)
- Drain/reload/purge operations
- Readiness tracking

### 9. **Artifacts Registry**
- Content-addressed storage for plans/diffs/traces
- `POST /v2/artifacts`, `GET /v2/artifacts/{id}`

### 10. **Observability**
- Prometheus metrics (`/metrics`)
- 7 cloud-specific metrics + core metrics
- Health endpoint
- Capabilities endpoint
- Structured logging (tracing-subscriber)

### 11. **Adapter Binding**
- Adapter host integration
- Optional llamacpp-http adapter binding
- Adapter dispatch for streaming

---

## Responsibilities Analysis

### Core Responsibilities (Keep in Binary)
1. **HTTP server lifecycle** — Axum setup, bind, serve
2. **Route assembly** — Wiring endpoints to handlers
3. **Application state** — AppState initialization
4. **Bootstrap** — Service startup, feature gates, env config
5. **Entrypoint** — `main.rs` binary

### Modularizable Candidates

#### ✅ **HIGH PRIORITY: Extract to Libraries**

##### 1. **Cloud Profile Node Management** (`api/nodes.rs` — 448 lines)
**Proposal**: Extract to `libs/control-plane/node-api/`

**Why**:
- Large, self-contained module (448 lines)
- Cloud profile-specific functionality
- Clear domain: node registration, heartbeat, deregistration
- Already depends on `service-registry` library
- Could be reused if we build additional control plane tools

**Interface**:
```rust
// libs/control-plane/node-api/src/handlers.rs
pub async fn register_node(...) -> Result<...>
pub async fn heartbeat(...) -> Result<...>
pub async fn deregister_node(...) -> Result<...>
pub async fn list_nodes(...) -> Result<...>
```

##### 2. **Streaming Service** (`services/streaming.rs` — 391 lines)
**Proposal**: Extract to `libs/streaming-core/`

**Why**:
- Largest service module (391 lines)
- Complex SSE logic (started/token/metrics/end frames)
- Adapter dispatch and correlation
- Could be reused in other streaming contexts
- Well-defined interface

**Interface**:
```rust
// libs/streaming-core/src/lib.rs
pub struct StreamingService;
impl StreamingService {
    pub async fn stream_task(&self, ...) -> Result<Response<Body>>;
}
```

##### 3. **Placement Logic** (`services/placement_v2.rs` — 302 lines)
**Proposal**: Extract to `libs/placement/` or keep in `orchestrator-core`

**Why**:
- Substantial module (302 lines)
- Model-aware placement is core orchestration logic
- May fit better in `orchestrator-core` alongside queue
- Clear separation from HTTP layer

**Interface**:
```rust
// libs/placement/src/lib.rs
pub struct PlacementEngine;
pub enum PlacementStrategy { RoundRobin, LeastLoaded }
impl PlacementEngine {
    pub fn select_pool(&self, ...) -> Option<PoolSelection>;
}
```

#### ⚠️ **MEDIUM PRIORITY: Consider Extraction**

##### 4. **Metrics Definitions** (`metrics.rs` — 199 lines)
**Proposal**: Extract to `libs/observability/metrics-core/`

**Why**:
- Self-contained Prometheus metric definitions
- Could be shared across binaries (orchestratord, pool-managerd)
- Clean interface (register, increment, observe)

**Concern**:
- Metrics are often binary-specific
- May not benefit from extraction unless pool-managerd needs same metrics

##### 5. **Catalog Availability** (`api/catalog_availability.rs` — 179 lines)
**Proposal**: Merge into `libs/catalog-core/`

**Why**:
- Catalog-specific logic
- Already depends on catalog-core
- Could live in catalog-core as an HTTP layer

**Interface**:
```rust
// libs/catalog-core/src/availability.rs
pub struct CatalogAvailability;
impl CatalogAvailability {
    pub fn query_availability(...) -> AvailabilityReport;
}
```

##### 6. **Session Service** (`services/session.rs` — 67 lines)
**Proposal**: Extract to `libs/session-management/`

**Why**:
- Self-contained session logic (TTL, budgets, cleanup)
- Could be reused in other contexts
- Small but well-defined

**Concern**:
- Only 67 lines — may not justify a separate crate
- Tightly coupled to orchestratord state

#### ❌ **LOW PRIORITY: Keep in Binary**

##### 7. **API Handlers** (`api/*.rs`)
**Keep in binary because**:
- Thin HTTP layer (converts requests → service calls → responses)
- Binary-specific routing and middleware
- Tight coupling to AppState

##### 8. **Domain Types** (`domain/*.rs`)
**Keep in binary because**:
- Small modules (error, IDs, SSE types)
- Binary-specific domain language
- Already have `contracts/api-types` for shared types

##### 9. **Infrastructure Adapters** (`infra/*.rs`)
**Keep in binary because**:
- Thin wrappers around external libs
- Binary-specific configuration

##### 10. **Ports** (`ports/*.rs`)
**Keep in binary because**:
- Small trait definitions
- Used only within orchestratord

---

## Modularization Proposal

### Option A: **Aggressive Extraction** (Recommended)

Extract 3-5 core libraries:

```
libs/
├── control-plane/
│   ├── node-api/ (NEW) — Node registration, heartbeat, deregistration handlers
│   └── service-registry/ (existing)
├── streaming-core/ (NEW) — SSE streaming service, adapter dispatch
├── placement/ (NEW) — Model-aware placement engine
└── observability/
    ├── metrics-core/ (NEW) — Shared Prometheus metric definitions
    └── narration-core/ (existing)
```

**Benefits**:
- Clearer separation of concerns
- Reusable components for future tools
- Easier to test in isolation
- Smaller orchestratord binary (focus on HTTP + routing)

**Costs**:
- More crates to maintain
- Potential over-engineering if components aren't reused
- Increased compilation time (more crate boundaries)

**Estimate**: orchestratord reduces from 3,382 lines → ~2,000 lines

---

### Option B: **Conservative Extraction** (Minimal Risk)

Extract only the largest, most self-contained modules:

```
libs/
├── control-plane/
│   └── node-api/ (NEW) — Node management (448 lines)
└── streaming-core/ (NEW) — SSE streaming (391 lines)
```

**Benefits**:
- Lower risk, focused extraction
- Immediate size reduction (~840 lines)
- Clear wins (largest modules)

**Costs**:
- orchestratord still has 2,500+ lines
- Placement and metrics remain in binary

**Estimate**: orchestratord reduces from 3,382 lines → ~2,500 lines

---

### Option C: **Keep as Binary** (Status Quo)

**Arguments for keeping orchestratord as-is**:

1. **Size is reasonable**: 3,382 lines is not excessively large for a binary
2. **Good internal structure**: Already well-organized into modules (api/, services/, domain/)
3. **Clear responsibilities**: Each module has a focused purpose
4. **Pre-1.0 flexibility**: Easier to refactor within a binary than across crates
5. **No reuse yet**: No concrete plans to reuse streaming/placement outside orchestratord

**When to reconsider**:
- If orchestratord grows beyond 5,000 lines
- If we build additional control plane tools that need node-api
- If streaming logic becomes generic enough to use elsewhere
- Post-1.0 when API stability matters more

---

## Recommendation

### **Option B: Conservative Extraction** ✅

**Extract immediately**:
1. `libs/control-plane/node-api/` — Node management (448 lines)
   - Clear domain boundary
   - Cloud profile-specific
   - May be reused by monitoring/admin tools

**Consider for future extraction** (post-v0.2.0):
2. `libs/streaming-core/` — SSE streaming (391 lines)
   - Wait until streaming patterns stabilize
   - May need adapter protocol changes

3. `libs/placement/` or move to `orchestrator-core`
   - Placement is core queue logic
   - May fit better in orchestrator-core than as separate lib

**Keep in binary** (for now):
- api/ handlers (thin HTTP layer)
- services/ (except streaming, consider later)
- domain/, infra/, ports/ (small, binary-specific)
- state.rs, metrics.rs (app-specific)

### Rationale

1. **orchestratord is well-structured but not too large** (3,382 lines is manageable)
2. **Node API is the clearest extraction candidate** (large, self-contained, cloud-specific)
3. **Premature extraction is costly** (compilation time, maintenance overhead)
4. **Pre-1.0 flexibility is valuable** (easier to refactor within a binary)
5. **Extract when there's proven reuse** (don't speculate on future needs)

### Next Steps

1. ✅ **Document this analysis** (this file)
2. ⏸️ **Defer extraction to post-v0.2.0** (focus on feature completion)
3. 📋 **Add to backlog**: Extract `node-api` when building admin/monitoring tools
4. 🔄 **Revisit at 5,000 lines** (or when concrete reuse emerges)

---

## Summary

| Metric | Current | After Option A | After Option B | After Option C |
|--------|---------|---------------|---------------|---------------|
| orchestratord lines | 3,382 | ~2,000 | ~2,500 | 3,382 |
| New libs | 0 | 4 | 1-2 | 0 |
| Maintenance burden | Low | Medium | Low | Low |
| Reusability | Low | High | Medium | Low |
| Risk | Low | Medium | Low | Low |

**Recommendation**: **Option C (Keep as Binary)** for v0.2.0, revisit extraction post-release when reuse patterns emerge.

---

## Appendix: Comparison with Other Binaries

- `pool-managerd`: Not analyzed yet (binary + lib)
- `orchestrator-core`: Pure library (~500 lines, queue logic)
- Typical Rust binary: 2,000-5,000 lines is common and manageable

**Conclusion**: orchestratord at 3,382 lines is within normal range for a Rust HTTP service binary.
