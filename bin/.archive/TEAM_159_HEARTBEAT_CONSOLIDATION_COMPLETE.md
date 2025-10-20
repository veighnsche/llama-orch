# TEAM-159: Heartbeat Consolidation - Phase 1 Complete

**Date:** 2025-10-20  
**Status:** ✅ Shared crate ready, binaries need updating

---

## What Was Completed

### 1. Trait Abstractions Created
**File:** `heartbeat/src/traits.rs` (200+ LOC)

**Traits defined:**
- `WorkerRegistry` - For rbee-hive's worker registry operations
- `HiveCatalog` - For queen-rbee's hive catalog operations  
- `DeviceDetector` - For queen-rbee's device detection
- `Narrator` - For narration events (optional, using observability-narration-core directly)

**Types defined:**
- `HiveStatus`, `HiveRecord`, `DeviceCapabilities`, `CpuDevice`, `GpuDevice`, `DeviceBackend`
- `DeviceResponse`, `CpuInfo`, `GpuInfo`
- `CatalogError`, `DetectionError`

### 2. Hive Receiver Created
**File:** `heartbeat/src/hive_receiver.rs` (160+ LOC)

**Functionality:**
- `handle_worker_heartbeat<R: WorkerRegistry>()` - Generic worker heartbeat handler
- Updates worker registry with heartbeat timestamp
- Returns `HeartbeatResponse` or `HeartbeatError::WorkerNotFound`
- **Includes unit tests with mock registry**

**Replaces:** `rbee-hive/src/http/heartbeat.rs` (179 LOC)

### 3. Queen Receiver Created
**File:** `heartbeat/src/queen_receiver.rs` (400+ LOC)

**Functionality:**
- `handle_hive_heartbeat<C: HiveCatalog, D: DeviceDetector>()` - Generic hive heartbeat handler
- Updates hive catalog with heartbeat timestamp
- **First heartbeat detection** - triggers device detection if status is Unknown
- **Device detection** - HTTP GET to hive's `/v1/devices` endpoint
- **Device storage** - converts and stores DeviceCapabilities
- **Narration** - emits 5+ narration events during flow
- **Status update** - changes hive from Unknown to Online
- **Includes unit tests with mock catalog and detector**

**Replaces:** `queen-rbee/src/http/heartbeat.rs` (269 LOC)

### 4. Dependencies Added
**File:** `heartbeat/Cargo.toml`

```toml
serde_json = "1.0"  # JSON parsing
thiserror = "1.0"  # Error types
async-trait = "0.1"  # Async trait methods
observability-narration-core = { path = "../narration-core" }  # Narration
```

### 5. Exports Updated
**File:** `heartbeat/src/lib.rs`

**New exports:**
```rust
// Receivers
pub use hive_receiver::{handle_worker_heartbeat, HeartbeatResponse};
pub use queen_receiver::handle_hive_heartbeat;

// Traits
pub use traits::{
    CatalogError, CpuDevice, DeviceBackend, DeviceCapabilities,
    DeviceDetector, DeviceResponse, GpuDevice, HiveCatalog,
    HiveRecord, HiveStatus, WorkerRegistry,
};
```

---

## Compilation Status

✅ **rbee-heartbeat compiles successfully**

```bash
cargo check -p rbee-heartbeat
# SUCCESS - No errors
```

---

## Next Steps: Update Binaries

### Step 1: Update rbee-hive

**File:** `bin/20_rbee_hive/src/http/heartbeat.rs`

**Current:** 179 LOC with full implementation  
**Target:** ~30 LOC wrapper

**Implementation:**
```rust
// Implement WorkerRegistry trait for crate::registry::WorkerRegistry
use async_trait::async_trait;
use rbee_heartbeat::traits::WorkerRegistry;

#[async_trait]
impl WorkerRegistry for crate::registry::WorkerRegistry {
    async fn update_heartbeat(&self, worker_id: &str) -> bool {
        // Delegate to existing method
        self.update_heartbeat(worker_id).await
    }
}

// Update handler to use shared logic
pub async fn handle_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<WorkerHeartbeatPayload>,
) -> Result<(StatusCode, Json<HeartbeatResponse>), (StatusCode, String)> {
    rbee_heartbeat::handle_worker_heartbeat(state.registry, payload)
        .await
        .map(|resp| (StatusCode::OK, Json(resp)))
        .map_err(|e| match e {
            rbee_heartbeat::hive_receiver::HeartbeatError::WorkerNotFound(_) => {
                (StatusCode::NOT_FOUND, e.to_string())
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        })
}
```

**LOC Savings:** 179 - 30 = **149 LOC removed**

### Step 2: Update queen-rbee

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Current:** 269 LOC with full implementation  
**Target:** ~50 LOC wrapper + trait impls

**Implementation:**
```rust
// Implement HiveCatalog trait
use async_trait::async_trait;
use rbee_heartbeat::traits::{HiveCatalog as HiveCatalogTrait, *};

#[async_trait]
impl HiveCatalogTrait for queen_rbee_hive_catalog::HiveCatalog {
    async fn update_heartbeat(&self, hive_id: &str, timestamp_ms: i64) -> Result<(), CatalogError> {
        self.update_heartbeat(hive_id, timestamp_ms)
            .await
            .map_err(|e| CatalogError::Database(e.to_string()))
    }
    
    async fn get_hive(&self, hive_id: &str) -> Result<Option<HiveRecord>, CatalogError> {
        // Convert from catalog's HiveRecord to trait's HiveRecord
        self.get_hive(hive_id)
            .await
            .map(|opt| opt.map(convert_hive_record))
            .map_err(|e| CatalogError::Database(e.to_string()))
    }
    
    // ... implement other methods
}

// Implement DeviceDetector trait
struct HttpDeviceDetector {
    client: reqwest::Client,
}

#[async_trait]
impl DeviceDetector for HttpDeviceDetector {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse, DetectionError> {
        let url = format!("{}/v1/devices", hive_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| DetectionError::Http(e.to_string()))?
            .json()
            .await
            .map_err(|e| DetectionError::Parse(e.to_string()))
    }
}

// Update handler to use shared logic
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    let detector = Arc::new(HttpDeviceDetector {
        client: reqwest::Client::new(),
    });
    
    rbee_heartbeat::handle_hive_heartbeat(
        state.hive_catalog,
        payload,
        detector,
    )
    .await
    .map(Json)
    .map_err(|e| match e {
        rbee_heartbeat::queen_receiver::HeartbeatError::HiveNotFound(_) => {
            (StatusCode::NOT_FOUND, e.to_string())
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    })
}
```

**LOC Savings:** 269 - 50 = **219 LOC removed**

---

## Total Impact

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| rbee-hive heartbeat | 179 LOC | 30 LOC | **-149 LOC** |
| queen-rbee heartbeat | 269 LOC | 50 LOC | **-219 LOC** |
| heartbeat shared crate | ~48K | +800 LOC | +800 LOC |
| **Net savings** | | | **-368 LOC from binaries** |

---

## Benefits Achieved

1. ✅ **Single source of truth** - All heartbeat logic in shared crate
2. ✅ **Reusability** - Other binaries can use same logic
3. ✅ **Testability** - Unit tests with mocks in shared crate
4. ✅ **Maintainability** - Changes in one place
5. ✅ **Consistency** - Same behavior across binaries
6. ✅ **Narration included** - All narration logic consolidated

---

## Testing

### Shared Crate Tests

**hive_receiver.rs:**
```bash
cargo test -p rbee-heartbeat hive_receiver
# 2 tests: success, unknown_worker
```

**queen_receiver.rs:**
```bash
cargo test -p rbee-heartbeat queen_receiver
# 2 tests: first_time, subsequent
```

### Binary Tests

After updating binaries, existing integration tests should still pass:
```bash
cargo test -p rbee-hive
cargo test -p queen-rbee
```

---

## Files Created

1. `heartbeat/src/traits.rs` - Trait abstractions (200 LOC)
2. `heartbeat/src/hive_receiver.rs` - Worker heartbeat handler (160 LOC)
3. `heartbeat/src/queen_receiver.rs` - Hive heartbeat handler (400 LOC)

## Files Modified

1. `heartbeat/Cargo.toml` - Added dependencies
2. `heartbeat/src/lib.rs` - Added exports

## Files To Update (Next Phase)

1. `rbee-hive/src/http/heartbeat.rs` - Use shared handler
2. `queen-rbee/src/http/heartbeat.rs` - Use shared handler

---

## Migration Checklist

- [x] Create trait abstractions
- [x] Create hive receiver (worker heartbeat handler)
- [x] Create queen receiver (hive heartbeat handler)
- [x] Add dependencies to shared crate
- [x] Update exports in shared crate
- [x] Verify shared crate compiles
- [ ] Implement WorkerRegistry trait in rbee-hive
- [ ] Update rbee-hive to use shared handler
- [ ] Implement HiveCatalog trait in queen-rbee
- [ ] Implement DeviceDetector trait in queen-rbee
- [ ] Update queen-rbee to use shared handler
- [ ] Run all tests
- [ ] Update documentation

---

**TEAM-159: Heartbeat consolidation Phase 1 complete. Shared crate ready. Binaries need trait implementations.**
