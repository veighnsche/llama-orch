# TEAM-159: Heartbeat Consolidation Plan

**Goal:** Move ALL heartbeat logic (including narration) into `rbee-heartbeat` shared crate

---

## Current State Analysis

### Queen-rbee: `/bin/10_queen_rbee/src/http/heartbeat.rs` (269 LOC)

**What it does:**
1. Receives `HiveHeartbeatPayload` from hives
2. Updates `last_heartbeat_ms` in hive catalog
3. **First heartbeat detection** - checks if status is `Unknown`
4. **Device detection trigger** - HTTP GET to `http://hive:port/v1/devices`
5. **Device storage** - converts `DeviceResponse` to `DeviceCapabilities` and stores
6. **Narration** - emits 5+ narration events during the flow
7. **Status update** - changes hive status from `Unknown` to `Online`

**Dependencies:**
- `observability_narration_core::Narration`
- `queen_rbee_hive_catalog::{HiveCatalog, HiveStatus, DeviceCapabilities}`
- `rbee_heartbeat::{HeartbeatAcknowledgement, HiveHeartbeatPayload}`
- `reqwest` for HTTP client
- `chrono` for timestamp parsing

### Rbee-hive: `/bin/20_rbee_hive/src/http/heartbeat.rs` (179 LOC)

**What it does:**
1. Receives `WorkerHeartbeatPayload` from workers
2. Updates `last_heartbeat` timestamp in worker registry
3. Returns `HeartbeatResponse`
4. **Note:** Relay to queen handled by periodic task (already in shared crate)

**Dependencies:**
- `crate::http::routes::AppState` (registry)
- `rbee_heartbeat::{HealthStatus, WorkerHeartbeatPayload}`
- `tracing` for logging

### Shared crate: `/bin/99_shared_crates/heartbeat/src/` (48K total)

**What it has:**
- `types.rs` - Payload types, enums
- `worker.rs` - Worker → Hive heartbeat sender
- `hive.rs` - Hive → Queen heartbeat sender (aggregates workers)
- `queen.rs` - Basic queen receiver types

**What it's missing:**
- Queen's device detection logic
- Queen's narration integration
- Rbee-hive's worker heartbeat receiver logic

---

## Consolidation Strategy

### Phase 1: Move Rbee-hive Worker Receiver to Shared Crate

**Create:** `heartbeat/src/hive_receiver.rs`

**Move from rbee-hive:**
```rust
// Handle POST /v1/heartbeat from workers
pub async fn handle_worker_heartbeat<R: WorkerRegistry>(
    registry: Arc<R>,
    payload: WorkerHeartbeatPayload,
) -> Result<HeartbeatResponse, HeartbeatError>
```

**Benefits:**
- Removes 90 LOC from rbee-hive
- Reusable worker heartbeat receiver logic
- Testable in isolation

### Phase 2: Move Queen Receiver to Shared Crate

**Create:** `heartbeat/src/queen_receiver.rs`

**Move from queen-rbee:**
```rust
// Handle POST /v1/heartbeat from hives
pub async fn handle_hive_heartbeat<C: HiveCatalog>(
    catalog: Arc<C>,
    payload: HiveHeartbeatPayload,
    device_detector: impl DeviceDetector,
    narrator: impl Narrator,
) -> Result<HeartbeatAcknowledgement, HeartbeatError>
```

**Key abstractions needed:**
1. `HiveCatalog` trait - for catalog operations
2. `DeviceDetector` trait - for triggering device detection
3. `Narrator` trait - for emitting narration events

**Benefits:**
- Removes 200 LOC from queen-rbee
- Reusable hive heartbeat receiver logic
- Testable with mocks

### Phase 3: Add Narration Support

**Option A: Make narration-core a dependency**
- Add `observability-narration-core` to heartbeat crate
- Use directly in `queen_receiver.rs`

**Option B: Narration trait abstraction**
```rust
pub trait Narrator {
    fn emit(&self, actor: &str, action: &str, target: &str, message: &str);
}
```

**Recommendation:** Option A - narration is core infrastructure, not domain logic

---

## Migration Steps

### Step 1: Create Trait Abstractions

**File:** `heartbeat/src/traits.rs`

```rust
// Trait for worker registry operations
pub trait WorkerRegistry: Send + Sync {
    async fn update_heartbeat(&self, worker_id: &str) -> bool;
}

// Trait for hive catalog operations
pub trait HiveCatalog: Send + Sync {
    async fn update_heartbeat(&self, hive_id: &str, timestamp_ms: i64) -> Result<(), CatalogError>;
    async fn get_hive(&self, hive_id: &str) -> Result<Option<HiveRecord>, CatalogError>;
    async fn update_devices(&self, hive_id: &str, devices: DeviceCapabilities) -> Result<(), CatalogError>;
    async fn update_hive_status(&self, hive_id: &str, status: HiveStatus) -> Result<(), CatalogError>;
}

// Trait for device detection
pub trait DeviceDetector: Send + Sync {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse, DetectionError>;
}
```

### Step 2: Move Hive Worker Receiver

**File:** `heartbeat/src/hive_receiver.rs`

```rust
pub async fn handle_worker_heartbeat<R>(
    registry: Arc<R>,
    payload: WorkerHeartbeatPayload,
) -> Result<HeartbeatResponse, HeartbeatError>
where
    R: WorkerRegistry,
{
    // Move logic from rbee-hive/src/http/heartbeat.rs
}
```

### Step 3: Move Queen Hive Receiver

**File:** `heartbeat/src/queen_receiver.rs`

```rust
pub async fn handle_hive_heartbeat<C, D>(
    catalog: Arc<C>,
    payload: HiveHeartbeatPayload,
    device_detector: Arc<D>,
) -> Result<HeartbeatAcknowledgement, HeartbeatError>
where
    C: HiveCatalog,
    D: DeviceDetector,
{
    // Move logic from queen-rbee/src/http/heartbeat.rs
    // Include narration calls
}
```

### Step 4: Update Binaries to Use Shared Logic

**Queen-rbee:**
```rust
// Implement traits
impl HiveCatalog for queen_rbee_hive_catalog::HiveCatalog { ... }
impl DeviceDetector for HttpDeviceDetector { ... }

// Use shared handler
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    rbee_heartbeat::queen_receiver::handle_hive_heartbeat(
        state.hive_catalog,
        payload,
        state.device_detector,
    ).await
    .map(Json)
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

**Rbee-hive:**
```rust
// Implement trait
impl WorkerRegistry for crate::registry::WorkerRegistry { ... }

// Use shared handler
pub async fn handle_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HeartbeatRequest>,
) -> Result<(StatusCode, Json<HeartbeatResponse>), (StatusCode, String)> {
    rbee_heartbeat::hive_receiver::handle_worker_heartbeat(
        state.registry,
        payload,
    ).await
    .map(|resp| (StatusCode::OK, Json(resp)))
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

---

## Dependencies to Add

**heartbeat/Cargo.toml:**
```toml
[dependencies]
# Existing
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
tracing = "0.1"

# NEW for queen receiver
observability-narration-core = { path = "../observability-narration-core" }
reqwest = { version = "0.12", features = ["json"] }
chrono = "0.4"
```

---

## LOC Savings

| Binary | Current | After | Savings |
|--------|---------|-------|---------|
| queen-rbee | 269 LOC | ~50 LOC (trait impls + wrapper) | **-219 LOC** |
| rbee-hive | 179 LOC | ~30 LOC (trait impl + wrapper) | **-149 LOC** |
| heartbeat crate | 48K | +400 LOC (receivers + traits) | +400 LOC |

**Net savings:** ~368 LOC removed from binaries, centralized in shared crate

---

## Testing Strategy

### Unit Tests in Shared Crate

**heartbeat/src/hive_receiver.rs:**
```rust
#[cfg(test)]
mod tests {
    // Mock WorkerRegistry
    struct MockRegistry { ... }
    
    #[tokio::test]
    async fn test_handle_worker_heartbeat_success() { ... }
    
    #[tokio::test]
    async fn test_handle_worker_heartbeat_unknown_worker() { ... }
}
```

**heartbeat/src/queen_receiver.rs:**
```rust
#[cfg(test)]
mod tests {
    // Mock HiveCatalog, DeviceDetector
    struct MockCatalog { ... }
    struct MockDetector { ... }
    
    #[tokio::test]
    async fn test_handle_hive_heartbeat_first_time() { ... }
    
    #[tokio::test]
    async fn test_handle_hive_heartbeat_subsequent() { ... }
}
```

### Integration Tests in Binaries

Keep existing integration tests in binaries, but they now test the thin wrappers.

---

## Migration Checklist

- [ ] Create `heartbeat/src/traits.rs` with abstractions
- [ ] Create `heartbeat/src/hive_receiver.rs` and move rbee-hive logic
- [ ] Create `heartbeat/src/queen_receiver.rs` and move queen-rbee logic
- [ ] Add narration-core, reqwest, chrono dependencies
- [ ] Implement traits in queen-rbee (HiveCatalog, DeviceDetector)
- [ ] Implement trait in rbee-hive (WorkerRegistry)
- [ ] Update queen-rbee to use shared handler
- [ ] Update rbee-hive to use shared handler
- [ ] Run all tests
- [ ] Update documentation

---

## Benefits

1. **Single source of truth** - All heartbeat logic in one place
2. **Reusability** - Other binaries can use the same logic
3. **Testability** - Mock implementations for unit tests
4. **Maintainability** - Changes in one place, not three
5. **Consistency** - Same behavior across all binaries
6. **LOC reduction** - 368 LOC removed from binaries

---

## Risks & Mitigations

**Risk:** Trait abstractions add complexity  
**Mitigation:** Keep traits minimal, well-documented

**Risk:** Breaking changes to existing code  
**Mitigation:** Incremental migration, keep tests passing

**Risk:** Circular dependencies  
**Mitigation:** Heartbeat crate depends on narration-core (not vice versa)

---

**TEAM-159: Ready to consolidate heartbeat logic into shared crate**
