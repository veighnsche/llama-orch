# orchestratord Responsibility Audit

**Date**: 2025-09-30  
**Purpose**: Verify orchestratord is not overextending responsibilities and other crates aren't doing orchestratord's job

---

## ✅ VERDICT: Clean Boundaries

**orchestratord is correctly scoped** and other crates respect boundaries. No violations found.

---

## Orchestratord's Responsibilities (Per Spec)

From `.specs/20-orchestratord.md` and `.specs/00_llama-orch.md`:

### ✅ What orchestratord SHOULD Do:

1. **HTTP API Surface** (Control + Data Plane)
   - `POST /v2/tasks` - Admission & enqueue
   - `GET /v2/tasks/:id/events` - SSE streaming
   - `POST /v2/tasks/:id/cancel` - Cancellation
   - `GET /v2/pools/:id/health` - Pool health queries
   - `POST /v2/pools/:id/drain` - Drain orchestration
   - `POST /v2/pools/:id/reload` - Reload orchestration
   - `GET /v2/meta/capabilities` - Capability discovery
   - `POST /v2/artifacts`, `GET /v2/artifacts/:id` - Artifact registry
   - `GET /v2/sessions/:id`, `DELETE /v2/sessions/:id` - Session management
   - `POST /v2/catalog/models`, etc. - Catalog HTTP endpoints

2. **Admission & Queueing**
   - Validate requests (ctx, deadline, budgets)
   - Enqueue into priority queues
   - Apply backpressure policies (reject/drop-lru)
   - Emit admission responses with ETA

3. **Placement & Routing**
   - Query pool-managerd for health/slots
   - Select target pool (least-loaded, VRAM-aware)
   - Respect pin overrides when provided
   - Route tasks to adapters

4. **SSE Streaming Orchestration**
   - Build SSE frames (`started`, `token`, `metrics`, `end`, `error`)
   - Coordinate with adapters for token streaming
   - Handle cancellation propagation
   - Persist SSE transcripts to artifacts

5. **Observability Coordination**
   - Emit metrics (queue depth, tasks enqueued/started/canceled/rejected)
   - Correlation ID middleware
   - Narration breadcrumbs at key decision points
   - Aggregate metrics from pools/adapters

6. **Lifecycle Coordination**
   - Trigger drain/reload on pools (via pool-managerd)
   - Coordinate provisioning (call engine-provisioner + model-provisioner)
   - Handoff autobind watcher (react to provisioner outputs)

### ❌ What orchestratord Should NOT Do:

1. **Engine Provisioning** → `engine-provisioner`
2. **Model Fetching** → `model-provisioner`
3. **Pool Supervision** → `pool-managerd`
4. **Adapter Implementation** → `worker-adapters/*`
5. **Queue Implementation** → `orchestrator-core`
6. **Catalog Storage** → `catalog-core`

---

## Audit Results by Crate

### ✅ orchestratord (bin/orchestratord)

**Current Responsibilities**:
- HTTP routing ✅
- Middleware (correlation ID, API key, auth) ✅
- Admission validation & enqueue ✅
- SSE streaming coordination ✅
- Session management ✅
- Artifact storage coordination ✅
- Catalog HTTP endpoints ✅
- Pool health queries ✅
- Handoff autobind watcher ✅
- Metrics aggregation ✅

**Dependencies** (Correct):
- `orchestrator-core` - Queue implementation ✅
- `pool-managerd` - Registry (embedded as library in home profile) ✅
- `adapter-host` - Adapter binding/routing ✅
- `catalog-core` - Catalog storage ✅
- `contracts-api-types` - OpenAPI types ✅
- `observability-narration-core` - Logging ✅

**Violations**: ❌ **NONE**

**Notes**:
- Correctly uses `pool-managerd::Registry` as embedded library (home profile pattern)
- Does NOT implement queue logic (delegates to `orchestrator-core`)
- Does NOT implement catalog storage (delegates to `catalog-core`)
- Does NOT provision engines (delegates to `engine-provisioner` via handoff files)
- Handoff autobind watcher is correct: orchestratord's job is to *react* to provisioner outputs

---

### ✅ engine-provisioner (libs/provisioners/engine-provisioner)

**Current Responsibilities**:
- Fetch/build engine binaries ✅
- Preflight tool checks ✅
- CUDA discovery & GPU enforcement ✅
- Write handoff JSON files ✅
- Spawn engine processes (or prepare artifacts) ✅

**What It Does NOT Do**:
- ❌ Does NOT call orchestratord HTTP APIs
- ❌ Does NOT manage admission/queueing
- ❌ Does NOT implement SSE streaming
- ❌ Does NOT manage pool health (writes handoff, pool-managerd reads it)

**Handoff Pattern** (Correct):
```rust
// engine-provisioner writes:
pub fn write_handoff_file(filename: &str, payload: &serde_json::Value) -> Result<PathBuf> {
    let dir = PathBuf::from(".runtime").join("engines");
    std::fs::create_dir_all(&dir)?;
    // ... write JSON
}

// orchestratord reads via handoff watcher:
pub fn spawn_handoff_autobind_watcher(state: AppState) {
    // watches .runtime/engines/*.json
    // binds adapters
    // updates pool_manager registry
}
```

**Violations**: ❌ **NONE**

---

### ✅ pool-managerd (libs/pool-managerd)

**Current Responsibilities**:
- Pool registry (health, slots, metadata) ✅
- Preload lifecycle tracking ✅
- Device mask management ✅
- Snapshot generation for placement ✅

**What It Does NOT Do**:
- ❌ Does NOT implement HTTP APIs (orchestratord does)
- ❌ Does NOT implement admission/queueing (orchestrator-core does)
- ❌ Does NOT provision engines (engine-provisioner does)
- ❌ Does NOT stream tokens (adapters do)

**Integration Pattern** (Correct):
```rust
// Home profile: orchestratord embeds as library
// bin/orchestratord/src/state.rs:
pub pool_manager: Arc<Mutex<PoolRegistry>>,

// Cloud profile (future): standalone daemon
// orchestratord would query via HTTP control API
```

**Comments in Code** (Correct):
```rust
// libs/pool-managerd/src/main.rs:
// Home Profile (Current):
// - orchestratord embeds pool-managerd::registry as library
// Cloud Profile (Future):
// - pool-managerd as standalone daemon
// - orchestratord queries via HTTP control API
```

**Violations**: ❌ **NONE**

---

### ✅ catalog-core (libs/catalog-core)

**Current Responsibilities**:
- Model reference parsing ✅
- Filesystem catalog storage ✅
- Lifecycle state management (Active/Retired) ✅
- Digest verification ✅
- Fetcher abstraction ✅

**What It Does NOT Do**:
- ❌ Does NOT implement HTTP endpoints (orchestratord does)
- ❌ Does NOT implement admission logic
- ❌ Does NOT provision models (model-provisioner does)

**Usage Pattern** (Correct):
```rust
// catalog-core is a library
// orchestratord uses it for HTTP endpoints:
// bin/orchestratord/src/api/catalog.rs:
pub async fn create_model(...) {
    let cat = get_catalog()?;
    cat.put(&entry)?;
    // ...
}
```

**Violations**: ❌ **NONE**

---

### ✅ orchestrator-core (libs/orchestrator-core)

**Current Responsibilities**:
- Queue implementation (InMemoryQueue) ✅
- Priority handling (Interactive/Batch) ✅
- Policy enforcement (Reject/DropLru) ✅
- Enqueue/cancel operations ✅

**What It Does NOT Do**:
- ❌ Does NOT implement HTTP APIs
- ❌ Does NOT implement SSE streaming
- ❌ Does NOT manage pool health
- ❌ Does NOT provision engines

**Usage Pattern** (Correct):
```rust
// orchestratord uses it:
// bin/orchestratord/src/admission.rs:
pub struct QueueWithMetrics {
    queue: InMemoryQueue,
    labels: MetricLabels,
}
```

**Violations**: ❌ **NONE**

---

### ✅ adapter-host (libs/adapter-host)

**Current Responsibilities**:
- Adapter binding/unbinding ✅
- Adapter routing (pool_id + replica_id → adapter) ✅
- Adapter trait abstraction ✅

**What It Does NOT Do**:
- ❌ Does NOT implement HTTP APIs
- ❌ Does NOT implement admission
- ❌ Does NOT implement specific adapters (worker-adapters/* do)

**Usage Pattern** (Correct):
```rust
// orchestratord uses it:
// bin/orchestratord/src/state.rs:
pub adapter_host: Arc<AdapterHost>,

// bin/orchestratord/src/services/handoff.rs:
state.adapter_host.bind(pool.clone(), replica, Arc::new(adapter));
```

**Violations**: ❌ **NONE**

---

## Potential Concerns (Resolved)

### 1. ⚠️ Handoff Autobind Watcher - Is this orchestratord's job?

**Answer**: ✅ **YES**

**Reasoning**:
- Engine-provisioner's job: Build engine, write handoff file
- Orchestratord's job: React to handoff files, bind adapters, update registry
- This is coordination/orchestration, which is orchestratord's core responsibility
- Alternative would be engine-provisioner calling orchestratord HTTP API, which violates separation

**Spec Support**:
- ORCH-3207: "Provisioning MUST produce a deterministic plan... pool MUST only transition to Ready after successful engine provisioning"
- Orchestratord coordinates the "transition to Ready" by reading handoffs

### 2. ⚠️ Pool Registry Embedded in orchestratord - Is this correct?

**Answer**: ✅ **YES (for home profile)**

**Reasoning**:
- Home profile: Single binary, single workstation → embed as library
- Cloud profile (future): Separate daemon → query via HTTP
- This is explicitly documented in `pool-managerd/src/main.rs`
- Follows "lightweight configuration" goal (ORCH-3002)

**Spec Support**:
- ORCH-3002: "Keep configuration lightweight: filesystem storage, no clustered control plane"
- Home profile optimizes for simplicity

### 3. ⚠️ Catalog HTTP Endpoints in orchestratord - Should catalog-core have its own API?

**Answer**: ✅ **NO, current design is correct**

**Reasoning**:
- `catalog-core` is a library (storage + logic)
- Orchestratord provides the HTTP surface
- This follows standard layering: storage library + API server
- Catalog operations need orchestrator context (correlation ID, auth, metrics)

**Spec Support**:
- OC-CTRL-2065: "The server SHOULD provide `POST /v1/artifacts`..." (orchestratord is "the server")
- ORCH-3037: "Catalog APIs MUST persist model metadata locally" (APIs are orchestratord's, storage is catalog-core's)

---

## Recommendations

### ✅ Current Design is Sound

No changes needed. Boundaries are clean and well-documented.

### 📝 Documentation Improvements

1. **Add Architecture Diagram** showing:
   - orchestratord as HTTP API layer
   - Embedded libraries (pool-managerd, catalog-core, orchestrator-core)
   - External processes (engine-provisioner writes handoffs)
   - Adapters (worker-adapters/*)

2. **Clarify Home vs Cloud Profile** in README:
   - Home: Single binary, embedded libraries
   - Cloud: Distributed, HTTP between components

3. **Document Handoff Pattern** as a design pattern:
   - Provisioners write files
   - Orchestratord watches and reacts
   - Avoids circular dependencies

---

## Summary

| Crate | Responsibilities | Violations | Status |
|-------|-----------------|------------|--------|
| **orchestratord** | HTTP APIs, admission, streaming coordination, placement | None | ✅ Correct |
| **engine-provisioner** | Engine build/fetch, handoff generation | None | ✅ Correct |
| **pool-managerd** | Registry, health, slots (embedded library) | None | ✅ Correct |
| **catalog-core** | Model storage, verification (library) | None | ✅ Correct |
| **orchestrator-core** | Queue implementation (library) | None | ✅ Correct |
| **adapter-host** | Adapter routing (library) | None | ✅ Correct |

**Conclusion**: All crates respect their boundaries. No overreach detected. Design follows spec. 🎯
