# TEAM-158 STARTING POINT

**Previous Team:** TEAM-157 (Add Local PC & Start Rbee-Hive)  
**Your Mission:** Implement Heartbeat Listener & Device Detection (Happy Flow Lines 37-48)

---

## 🎯 What TEAM-157 Built for You

### 1. Localhost Added to Catalog ✅
```rust
// Localhost is now in the hive catalog with status: Unknown
HiveRecord {
    id: "localhost",
    host: "127.0.0.1",
    port: 8600,
    status: HiveStatus::Unknown,
    last_heartbeat_ms: None,
    ...
}
```

### 2. Rbee-Hive Running ✅
- Subprocess spawned on port 8600
- Hive will automatically send heartbeats
- Queen is waiting for heartbeat

---

## 📋 Your TODO List (Lines 37-48)

From `a_human_wrote_this.md`:

### 1. **Implement Heartbeat Endpoint**
   - Add POST /heartbeat endpoint to queen
   - Hive will call this automatically
   - Update `last_heartbeat_ms` in catalog
   - Narration: "First heartbeat from a bee hive is received from localhost. Checking its capabilities..."

### 2. **Check Hive Capabilities**
   - When first heartbeat received, check catalog for devices
   - Will be undefined (first time)
   - Narration: "Unknown capabilities of beehive localhost. Asking the beehive to detect devices"

### 3. **Request Device Detection**
   - Call rbee-hive API: GET http://localhost:8600/devices
   - Hive responds with CPU, GPUs, model catalog, worker catalog
   - Parse response

### 4. **Update Catalog with Devices**
   - Store devices in catalog
   - Update hive status to Online
   - Narration: "The beehive localhost has cpu, gpu0 rtx 3060, gpu1 rtx 3090, model catalog has 0 models, 0 workers available"

---

## 🔧 Implementation Guide

### Step 1: Create Heartbeat Endpoint

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs` (NEW)

```rust
//! Heartbeat endpoint for hive health monitoring
//!
//! Created by: TEAM-158

use axum::{extract::State, http::StatusCode, Json};
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const ACTOR_QUEEN_HEARTBEAT: &str = "👑 queen-heartbeat";
const ACTION_HEARTBEAT: &str = "heartbeat";

#[derive(Debug, Deserialize)]
pub struct HeartbeatRequest {
    pub hive_id: String,
    pub timestamp_ms: i64,
}

#[derive(Debug, Serialize)]
pub struct HeartbeatResponse {
    pub acknowledged: bool,
}

pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,
}

pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<HeartbeatResponse>, (StatusCode, String)> {
    // TEAM-158: Update heartbeat in catalog
    state.hive_catalog
        .update_heartbeat(&req.hive_id, req.timestamp_ms)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_HEARTBEAT, &req.hive_id)
        .human(format!("Heartbeat received from {}", req.hive_id))
        .emit();

    // TEAM-158: Check if this is first heartbeat
    let hive = state.hive_catalog
        .get_hive(&req.hive_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Hive not found".to_string()))?;

    // TEAM-158: If first heartbeat (status is Unknown), trigger device detection
    if matches!(hive.status, HiveStatus::Unknown) {
        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_HEARTBEAT, &req.hive_id)
            .human(format!("First heartbeat from {}. Checking capabilities...", req.hive_id))
            .emit();

        // TEAM-158: TODO - Trigger device detection
        // For now, just acknowledge
    }

    Ok(Json(HeartbeatResponse { acknowledged: true }))
}
```

### Step 2: Register Heartbeat Route

**File:** `bin/10_queen_rbee/src/http/mod.rs`

```rust
pub mod heartbeat;  // TEAM-158: Add this line
```

**File:** `bin/10_queen_rbee/src/main.rs`

```rust
// In create_router():
let heartbeat_state = http::heartbeat::HeartbeatState {
    hive_catalog: hive_catalog.clone(),
};

axum::Router::new()
    .route("/health", get(http::health::handle_health))
    .route("/shutdown", post(http::shutdown::handle_shutdown))
    .route("/jobs", post(http::jobs::handle_create_job))
    .route("/jobs/{job_id}/stream", get(http::jobs::handle_stream_job))
    .with_state(job_state)
    // TEAM-158: Heartbeat endpoint
    .route("/heartbeat", post(http::heartbeat::handle_heartbeat))
    .with_state(heartbeat_state)
```

### Step 3: Device Detection

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

```rust
// After first heartbeat detected:

// TEAM-158: Request device detection from hive
let hive_url = format!("http://{}:{}/devices", hive.host, hive.port);
let client = reqwest::Client::new();
let response = client.get(&hive_url)
    .send()
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

let devices: DeviceResponse = response.json()
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_HEARTBEAT, &req.hive_id)
    .human(format!(
        "The beehive {} has {} devices, {} models, {} workers",
        req.hive_id,
        devices.devices.len(),
        devices.models.len(),
        devices.workers.len()
    ))
    .emit();

// TEAM-158: Update catalog with devices
// TODO: Store devices in catalog (may need schema update)

// TEAM-158: Update hive status to Online
state.hive_catalog
    .update_hive_status(&req.hive_id, HiveStatus::Online)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
```

---

## 📚 Reference Files

**Read these first:**
- `bin/a_human_wrote_this.md` (lines 37-48) - Your mission
- `bin/TEAM_157_SUMMARY.md` - What TEAM-157 built
- `bin/10_queen_rbee/src/http/jobs.rs` - Current implementation
- `bin/15_queen_rbee_crates/hive-catalog/src/lib.rs` - Catalog API

---

## 🎓 Tips

1. **Heartbeat is async** - Don't block job creation waiting for heartbeat
2. **First heartbeat detection** - Check if status is Unknown
3. **Device detection is separate** - Triggered after first heartbeat
4. **Error handling** - Hive might not respond, handle gracefully
5. **Narration everywhere** - Stream updates via SSE if possible

---

## ⚠️ Important Notes

### Heartbeat Flow
```
Hive starts → Sends POST /heartbeat → Queen updates catalog → 
If first heartbeat → Request devices → Update catalog → Status = Online
```

### Don't Implement Yet
- Model provisioning (later)
- Worker provisioning (later)
- Actual inference routing (later)

### Your Job
- Implement heartbeat endpoint
- Detect first heartbeat
- Request device detection
- Update catalog with devices
- Update hive status to Online

---

**Good luck, TEAM-158! 🚀**

**You got this! 💪**
