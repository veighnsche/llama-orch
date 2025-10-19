# rbee-hive HTTP Update Plan

**Created:** 2025-10-20  
**Based on:** New architecture docs (a_human_wrote_this.md, a_Claude_Sonnet_4_5_refined_this.md)  
**Current Implementation:** /bin/20_rbee_hive/src/http (copied from old.rbee-hive)

---

## Executive Summary

The rbee-hive HTTP implementation needs to be updated to align with the new architecture. The current implementation is **mostly correct** but is missing several critical endpoints and functionality.

**Status:** 🟡 60% Complete (6/10 endpoints implemented)

---

## Architecture Alignment Analysis

### ✅ Already Implemented (Correct)

1. **POST /v1/workers/spawn** - Spawn worker with model/backend/device
   - ✅ Model provisioning from catalog
   - ✅ Port allocation logic
   - ✅ Worker binary spawning
   - ✅ Worker registry (RAM)
   - ✅ Callback URL to hive (not queen directly)

2. **POST /v1/workers/ready** - Worker ready callback
   - ✅ Updates worker state to Idle
   - ✅ Includes queen callback notification (TEAM-124)

3. **GET /v1/workers/list** - List all workers

4. **POST /v1/models/download** - Download model
   - ✅ Model catalog (SQLite)
   - ✅ Model provisioner with download tracking

5. **GET /v1/models/download/progress** - SSE progress stream
   - ✅ Industry-standard SSE with [DONE] marker

6. **POST /v1/heartbeat** - Receive heartbeat from workers
   - ✅ Updates last_heartbeat timestamp
   - ⚠️ **ISSUE:** Does NOT relay to queen (see below)

### ❌ Missing Critical Endpoints

According to the new architecture (Phase 4, 6, 12):

1. **GET /v1/devices** - Device detection
   - **Architecture reference:** Phase 4 (lines 136-164)
   - **Returns:** CPU info, GPU list (ID, name, VRAM), model count, worker count
   - **Crate exists:** `/bin/25_rbee_hive_crates/device-detection/`
   - **Called by:** Queen-rbee on first heartbeat or stale capabilities

2. **GET /v1/capacity** - VRAM capacity check
   - **Architecture reference:** Phase 6 (lines 181-189)
   - **Query params:** `device=gpu1&model=HF:author/minillama`
   - **Returns:** 204 No Content (OK) or 409 Conflict (insufficient)
   - **Crate exists:** `/bin/25_rbee_hive_crates/vram-checker/`
   - **Called by:** Queen-rbee before worker spawning

3. **POST /v1/shutdown** - Graceful shutdown
   - **Architecture reference:** Phase 12 (lines 368-387)
   - **Action:** Shutdown all workers, then hive daemon
   - **Called by:** Queen-rbee during cascading shutdown

4. **POST /v1/workers/provision** - Provision worker binary (optional for dev)
   - **Architecture reference:** Phase 8 (lines 230-242)
   - **Dev mode:** Use hardcoded target path
   - **Crate exists:** `/bin/25_rbee_hive_crates/worker-catalog/`
   - **Called by:** Queen-rbee if worker binary not in catalog

### ⚠️ Needs Enhancement

1. **Heartbeat Relay to Queen**
   - **Current:** Hive receives worker heartbeats, updates registry
   - **Missing:** Hive should send its own heartbeat to queen with nested worker heartbeats
   - **Architecture reference:** Phase 10 (lines 300-313)
   - **Pattern:** Worker → Hive (POST /v1/heartbeat) → Queen (POST /heartbeat with nested workers array)

---

## Implementation Tasks

### Task 1: Add Device Detection Endpoint

**File:** `src/http/devices.rs` (new)

```rust
//! Device detection endpoint
//!
//! Per architecture Phase 4: Device Detection
//! Returns CPU info, GPU list, model count, worker count

use axum::{extract::State, Json};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct DevicesResponse {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuInfo>,
    pub models: usize,
    pub workers: usize,
}

#[derive(Debug, Serialize)]
pub struct CpuInfo {
    pub cores: u32,
    pub ram_gb: u32,
}

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub id: String,
    pub name: String,
    pub vram_gb: u32,
}

/// Handle GET /v1/devices
pub async fn handle_devices(
    State(state): State<AppState>,
) -> Json<DevicesResponse> {
    // Use rbee-hive-device-detection crate
    // Count models from catalog
    // Count workers from registry
}
```

**Reference:** `bin/25_rbee_hive_crates/device-detection/`

---

### Task 2: Add VRAM Capacity Check Endpoint

**File:** `src/http/capacity.rs` (new)

```rust
//! VRAM capacity check endpoint
//!
//! Per architecture Phase 6: VRAM Check

use axum::{extract::{Query, State}, http::StatusCode};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct CapacityQuery {
    pub device: String,     // e.g., "gpu1"
    pub model: String,      // e.g., "HF:author/minillama"
}

/// Handle GET /v1/capacity?device=gpu1&model=HF:author/minillama
pub async fn handle_capacity_check(
    State(state): State<AppState>,
    Query(params): Query<CapacityQuery>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Use rbee-hive-vram-checker crate
    // Check: Total VRAM - loaded models - estimated model size
    // Return: 204 No Content (OK) or 409 Conflict (insufficient)
}
```

**Reference:** `bin/25_rbee_hive_crates/vram-checker/`

---

### Task 3: Add Shutdown Endpoint

**File:** `src/http/shutdown.rs` (new)

```rust
//! Graceful shutdown endpoint
//!
//! Per architecture Phase 12: Cascading Shutdown

use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ShutdownResponse {
    pub message: String,
}

/// Handle POST /v1/shutdown
pub async fn handle_shutdown(
    State(state): State<AppState>,
) -> Json<ShutdownResponse> {
    // 1. Shutdown all workers in registry
    // 2. Signal server to shutdown gracefully
    // 3. Return acknowledgment
}
```

---

### Task 4: Add Worker Binary Provision Endpoint (Optional)

**File:** `src/http/workers.rs` (extend)

```rust
/// Worker provision request
#[derive(Debug, Deserialize)]
pub struct ProvisionWorkerRequest {
    pub kind: String,  // e.g., "cuda-llm-worker-rbee"
}

/// Worker provision response
#[derive(Debug, Serialize)]
pub struct ProvisionWorkerResponse {
    pub message: String,
    pub path: String,
}

/// Handle POST /v1/workers/provision
pub async fn handle_provision_worker(
    State(state): State<AppState>,
    Json(request): Json<ProvisionWorkerRequest>,
) -> Result<Json<ProvisionWorkerResponse>, (StatusCode, String)> {
    // Dev mode: Use hardcoded target path
    // Prod mode: Download binary from artifact store
    // Register in worker-catalog (SQLite)
}
```

**Reference:** `bin/25_rbee_hive_crates/worker-catalog/`

---

### Task 5: Enhance Heartbeat Handling (Relay to Queen)

**File:** `src/http/heartbeat.rs` (update)

**Current behavior:**
- Worker sends heartbeat to hive
- Hive updates `last_heartbeat` in worker registry
- **STOPS HERE** ❌

**Required behavior:**
- Worker sends heartbeat to hive
- Hive updates `last_heartbeat` in worker registry
- **Hive sends its own heartbeat to queen** with nested worker heartbeats ✅

**Implementation:**

```rust
// In heartbeat.rs, after updating registry:

// Spawn async task to relay to queen (if queen_callback_url configured)
if let Some(ref queen_url) = state.queen_callback_url {
    let hive_heartbeat_payload = serde_json::json!({
        "hive_id": "localhost",  // From config
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "workers": [
            {
                "worker_id": payload.worker_id,
                "state": "Ready",  // From registry
                "last_heartbeat": payload.timestamp,
            }
        ]
    });

    tokio::spawn(async move {
        // POST to queen_url/v1/heartbeat
        // (Similar pattern to queen callback in workers.rs line 418)
    });
}
```

**Architecture reference:** Phase 10 (lines 300-313)

---

### Task 6: Update Routes Configuration

**File:** `src/http/routes.rs`

Add new endpoints to router:

```rust
pub fn create_router(...) -> Router {
    // ... existing code ...

    let protected_routes = Router::new()
        // Existing endpoints
        .route("/v1/workers/spawn", post(workers::handle_spawn_worker))
        .route("/v1/workers/ready", post(workers::handle_worker_ready))
        .route("/v1/workers/list", get(workers::handle_list_workers))
        .route("/v1/heartbeat", post(heartbeat::handle_heartbeat))
        .route("/v1/models/download", post(models::handle_download_model))
        .route("/v1/models/download/progress", get(models::handle_download_progress))
        
        // NEW: Device detection (protected or public?)
        .route("/v1/devices", get(devices::handle_devices))
        
        // NEW: Capacity check
        .route("/v1/capacity", get(capacity::handle_capacity_check))
        
        // NEW: Shutdown
        .route("/v1/shutdown", post(shutdown::handle_shutdown))
        
        // OPTIONAL: Worker provisioning
        .route("/v1/workers/provision", post(workers::handle_provision_worker))
        
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // ... rest of router setup ...
}
```

---

### Task 7: Update mod.rs

**File:** `src/http/mod.rs`

```rust
pub mod devices;     // NEW
pub mod capacity;    // NEW
pub mod shutdown;    // NEW
pub mod health;
pub mod heartbeat;
pub mod metrics;
pub mod middleware;
pub mod models;
pub mod routes;
pub mod server;
pub mod workers;
```

---

## Comparison with Worker HTTP

### Worker HTTP Endpoints (Reference)

From `/bin/30_llm_worker_rbee/src/http/routes.rs`:

- ✅ `GET /health` - Basic health (public)
- ✅ `POST /v1/inference` - Execute inference (protected, SSE)

### Key Differences

1. **Worker is simpler:** Only health + inference
2. **Worker sends heartbeats TO hive:** Worker doesn't expose heartbeat endpoint
3. **Worker callback pattern:** Worker calls hive's `/v1/workers/ready` when ready
4. **Authentication:** Worker uses same auth pattern (token-based, optional local mode)

### Patterns to Copy from Worker

1. **Narration fields** (worker uses observability_narration_core)
2. **Request queue pattern** (worker uses RequestQueue for inference)
3. **Graceful shutdown** (both use broadcast channel)

---

## Architecture Flow Verification

### Happy Path: Worker Spawning (Phase 9)

**Current Implementation:**
1. ✅ Queen → Hive: `POST /v1/workers/spawn`
2. ✅ Hive spawns worker process with `--hive-url http://127.0.0.1:8600`
3. ✅ Worker starts, calls `POST http://127.0.0.1:8600/v1/workers/ready`
4. ✅ Hive updates registry, notifies queen

**Correct!** ✅

### Happy Path: Worker Heartbeat (Phase 10)

**Current Implementation:**
1. ✅ Worker → Hive: `POST http://127.0.0.1:8600/v1/heartbeat`
2. ✅ Hive updates `last_heartbeat` in registry
3. ❌ **MISSING:** Hive → Queen: `POST http://127.0.0.1:8500/heartbeat` (nested)

**Needs Fix!** ⚠️

### Happy Path: Device Detection (Phase 4)

**Current Implementation:**
1. ❌ **MISSING:** Queen → Hive: `GET /v1/devices`

**Needs Implementation!** 🚧

### Happy Path: VRAM Check (Phase 6)

**Current Implementation:**
1. ❌ **MISSING:** Queen → Hive: `GET /v1/capacity?device=gpu1&model=...`

**Needs Implementation!** 🚧

### Happy Path: Shutdown (Phase 12)

**Current Implementation:**
1. ❌ **MISSING:** Queen → Hive: `POST /v1/shutdown`
2. ❌ **MISSING:** Hive shutdowns workers then itself

**Needs Implementation!** 🚧

---

## Mock Hive Server Analysis

From `/xtask/src/tasks/worker.rs`:

**Mock Hive Implements:**
- ✅ `POST /v1/heartbeat` - Receives worker heartbeats
- ✅ Prints heartbeat count and details

**Mock Hive Does NOT Implement:**
- ❌ Device detection
- ❌ Capacity check
- ❌ Worker spawning
- ❌ Model provisioning
- ❌ Shutdown

**Purpose:** Testing worker heartbeat in isolation

**Conclusion:** Mock is correct for its purpose (worker isolation testing)

---

## Implementation Order

### Phase 1: Critical Missing Endpoints (1-2 days)
1. **Device detection** (`devices.rs`)
2. **VRAM capacity** (`capacity.rs`)
3. **Shutdown** (`shutdown.rs`)

### Phase 2: Enhanced Heartbeat (1 day)
4. **Heartbeat relay to queen** (update `heartbeat.rs`)

### Phase 3: Optional Features (0.5 days)
5. **Worker binary provisioning** (extend `workers.rs`)

### Phase 4: Testing & Documentation (1 day)
6. Update integration tests
7. Update README.md
8. Add BDD scenarios

**Total Estimate:** 3-4 days

---

## Testing Strategy

### Unit Tests (per module)
- `devices.rs`: Mock device-detection crate
- `capacity.rs`: Mock vram-checker crate
- `shutdown.rs`: Mock worker registry shutdown
- `heartbeat.rs`: Verify queen callback

### Integration Tests
- Full happy path: spawn → heartbeat → inference → shutdown
- Device detection with real hardware info
- Capacity check with mock VRAM data
- Heartbeat relay chain: worker → hive → queen

### BDD Scenarios (test-harness/bdd)
- Scenario: "Hive reports device capabilities to queen"
- Scenario: "Hive checks VRAM capacity before spawning worker"
- Scenario: "Hive relays worker heartbeats to queen"
- Scenario: "Hive cascades shutdown to all workers"

---

## Dependencies to Verify

Ensure these crates are properly wired in `Cargo.toml`:

```toml
[dependencies]
# Existing
rbee-hive-worker-registry = { path = "../25_rbee_hive_crates/worker-registry" }
rbee-hive-model-catalog = { path = "../25_rbee_hive_crates/model-catalog" }
rbee-hive-model-provisioner = { path = "../25_rbee_hive_crates/model-provisioner" }
rbee-hive-download-tracker = { path = "../25_rbee_hive_crates/download-tracker" }

# NEW (verify these exist)
rbee-hive-device-detection = { path = "../25_rbee_hive_crates/device-detection" }
rbee-hive-vram-checker = { path = "../25_rbee_hive_crates/vram-checker" }
rbee-hive-worker-catalog = { path = "../25_rbee_hive_crates/worker-catalog" }
```

---

## Security Considerations

### Authentication
- All new endpoints should use existing auth middleware
- Device detection: **Public or Protected?** → Probably protected (queen-only)
- Capacity check: **Protected** (queen-only)
- Shutdown: **Protected** (queen-only, critical operation)

### Input Validation
- Device ID validation (e.g., "gpu0", "gpu1", "cpu")
- Model reference validation (already exists: `input_validation::validate_model_ref`)

### Audit Logging
- Shutdown events should be logged (use existing `audit_logger`)
- Device detection queries (optional, for security monitoring)

---

## Checklist

- [ ] Create `devices.rs` module
- [ ] Create `capacity.rs` module
- [ ] Create `shutdown.rs` module
- [ ] Update `heartbeat.rs` to relay to queen
- [ ] Extend `workers.rs` with provision endpoint (optional)
- [ ] Update `routes.rs` with new endpoints
- [ ] Update `mod.rs` with new modules
- [ ] Verify crate dependencies in `Cargo.toml`
- [ ] Write unit tests for each new module
- [ ] Update integration tests
- [ ] Add BDD scenarios
- [ ] Update README.md with new endpoints
- [ ] Verify alignment with mock hive server in xtask
- [ ] Test full happy path: spawn → heartbeat → inference → shutdown

---

## Questions for User

1. **Device detection endpoint:** Should it be public or protected?
   - **Recommendation:** Protected (queen-only access)

2. **Worker binary provisioning:** Dev mode only or full implementation?
   - **Current state:** Dev uses hardcoded target paths
   - **Recommendation:** Implement basic version, stub prod artifact download

3. **Heartbeat relay:** Should hive send heartbeat to queen proactively or only when worker heartbeat received?
   - **Architecture suggests:** Hive sends periodic heartbeats with nested worker status
   - **Recommendation:** Implement periodic heartbeat task (every 30s) + relay on worker heartbeat

4. **Shutdown behavior:** Should shutdown wait for workers to finish inference?
   - **Recommendation:** Yes, graceful shutdown with timeout (30s)

---

**END OF PLAN**
