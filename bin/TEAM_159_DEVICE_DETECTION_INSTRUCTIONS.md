# TEAM-159: Device Detection & Capability Storage

**Date:** 2025-10-20  
**Previous Team:** TEAM-158  
**Mission:** Implement device detection flow from hive to queen

---

## ✅ What TEAM-158 Completed

### 1. Device Types Created

**Location:** `bin/15_queen_rbee_crates/hive-catalog/src/device_types.rs`

**Types:**
```rust
pub enum DeviceBackend {
    Cpu,    // Always available
    Cuda,   // NVIDIA GPUs
    Metal,  // Apple GPUs
}

pub struct CpuDevice {
    pub cores: u32,
    pub ram_gb: u32,
}

pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub vram_gb: u32,
    pub backend: DeviceBackend,
}

pub struct DeviceCapabilities {
    pub cpu: Option<CpuDevice>,
    pub gpus: Vec<GpuDevice>,
}
```

### 2. Hive Catalog Updated

**Added:**
- `devices: Option<DeviceCapabilities>` field to `HiveRecord`
- `devices_json TEXT` column to database schema
- `update_devices(id, devices)` method for updating device capabilities

**Usage:**
```rust
use queen_rbee_hive_catalog::{HiveCatalog, DeviceCapabilities, CpuDevice, GpuDevice, DeviceBackend};

// Update devices after detection
let mut devices = DeviceCapabilities::none();
devices.cpu = Some(CpuDevice { cores: 8, ram_gb: 32 });
devices.gpus.push(GpuDevice {
    index: 0,
    name: "RTX 3060".to_string(),
    vram_gb: 12,
    backend: DeviceBackend::Cuda,
});

catalog.update_devices("localhost", devices).await?;
```

---

## 🎯 YOUR MISSION (TEAM-159)

Implement the device detection flow from the happy flow document (lines 38-48).

### Flow Overview

```
1. Queen receives first heartbeat from hive (status = Unknown)
2. Queen checks hive catalog → devices = None
3. Queen requests device detection from hive (GET /devices)
4. Hive calls device-detection crate
5. Hive responds with device capabilities
6. Queen stores devices in hive catalog
7. Queen updates hive status to Online
```

---

## 📋 Implementation Checklist

### ✅ Already Done (TEAM-158)
- [x] Device types created
- [x] Hive catalog schema updated
- [x] `update_devices()` method added
- [x] All tests passing

### ✅ DONE (TEAM-159)

#### 1. Rbee-Hive: Add Device Detection Endpoint

**File:** `bin/20_rbee_hive/src/http/devices.rs` (create new)

**Endpoint:** `GET /devices`

**Implementation:**
```rust
use rbee_hive_device_detection::{detect_backends, detect_gpus};
use axum::{Json, response::IntoResponse};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct DeviceResponse {
    pub cpu: Option<CpuDeviceInfo>,
    pub gpus: Vec<GpuDeviceInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct CpuDeviceInfo {
    pub cores: u32,
    pub ram_gb: u32,
}

#[derive(Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub index: u32,
    pub name: String,
    pub vram_gb: u32,
    pub backend: String, // "cuda" or "metal"
}

pub async fn handle_device_detection() -> impl IntoResponse {
    // 1. Detect backends (CPU, CUDA, Metal)
    let backend_caps = detect_backends();
    
    // 2. Detect GPUs
    let gpu_info = detect_gpus();
    
    // 3. Build response
    let cpu = Some(CpuDeviceInfo {
        cores: num_cpus::get() as u32,
        ram_gb: get_system_ram_gb(),
    });
    
    let gpus = gpu_info.devices.iter().map(|gpu| {
        GpuDeviceInfo {
            index: gpu.index,
            name: gpu.name.clone(),
            vram_gb: gpu.vram_total_gb() as u32,
            backend: if backend_caps.has_backend(Backend::Cuda) {
                "cuda"
            } else if backend_caps.has_backend(Backend::Metal) {
                "metal"
            } else {
                "cpu"
            }.to_string(),
        }
    }).collect();
    
    Json(DeviceResponse { cpu, gpus })
}

fn get_system_ram_gb() -> u32 {
    // Use sysinfo crate or similar
    // For now, return a placeholder
    32
}
```

**Add to router:**
```rust
// bin/20_rbee_hive/src/main.rs
.route("/devices", get(http::devices::handle_device_detection))
```

#### 2. Queen-Rbee: Request Device Detection on First Heartbeat

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Current code (lines 78-177):**
```rust
// TEAM-158: If first heartbeat (status is Unknown), trigger device detection
if matches!(hive.status, HiveStatus::Unknown) {
    // ... existing narration ...
    
    // TEAM-158: Request device detection from hive
    let hive_url = format!("http://{}:{}/devices", hive.host, hive.port);
    
    // ... existing HTTP call ...
    
    let devices: DeviceResponse = response.json().await.map_err(|e| {
        // ... error handling ...
    })?;
    
    // TODO TEAM-159: Convert DeviceResponse to DeviceCapabilities and store
}
```

**What YOU need to add:**
```rust
// After getting DeviceResponse, convert and store:

// Convert DeviceResponse to DeviceCapabilities
let mut device_caps = DeviceCapabilities::none();

// Add CPU
if let Some(cpu) = devices.cpu {
    device_caps.cpu = Some(CpuDevice {
        cores: cpu.cores,
        ram_gb: cpu.ram_gb,
    });
}

// Add GPUs
for gpu in devices.gpus {
    let backend = match gpu.backend.as_str() {
        "cuda" => DeviceBackend::Cuda,
        "metal" => DeviceBackend::Metal,
        _ => continue,
    };
    
    device_caps.gpus.push(GpuDevice {
        index: gpu.index,
        name: gpu.name,
        vram_gb: gpu.vram_gb,
        backend,
    });
}

// Store in hive catalog
state.hive_catalog.update_devices(&payload.hive_id, device_caps).await.map_err(|e| {
    Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
        .human(format!("Failed to store device capabilities: {}", e))
        .error_kind("device_storage_failed")
        .emit();
    (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to store devices: {}", e))
})?;

// Add narration
Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, &payload.hive_id)
    .human(format!("Stored device capabilities for hive {}", payload.hive_id))
    .emit();
```

#### 3. Add Narration

**Expected narration (from happy flow):**
```
Line 39: "first heartbeat from a bee hive is received from localhost. checking its capabilities..."
Line 41: "unknown capabilities of beehive localhost. asking the beehive to detect devices"
Line 47: "the beehive localhost has a cpu gpu0 and 1 and blabla and model catalog has 0 models and 0 workers available"
```

**Implementation:**
```rust
// Line 39 - Already exists in heartbeat.rs

// Line 41 - Already exists in heartbeat.rs

// Line 47 - Add after storing devices:
let device_summary = format!(
    "CPU: {} cores, {} GB RAM | GPUs: {}",
    device_caps.cpu.as_ref().map(|c| c.cores).unwrap_or(0),
    device_caps.cpu.as_ref().map(|c| c.ram_gb).unwrap_or(0),
    device_caps.gpus.iter()
        .map(|g| format!("{} ({} GB)", g.name, g.vram_gb))
        .collect::<Vec<_>>()
        .join(", ")
);

Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, &payload.hive_id)
    .human(format!(
        "The beehive {} has {} | Model catalog: 0 models | Workers: 0 available",
        payload.hive_id,
        device_summary
    ))
    .emit();
```

---

## 🧪 Testing

### Manual Test

```bash
# Terminal 1: Start rbee-hive
cargo run --bin rbee-hive -- --port 8600

# Terminal 2: Test device detection endpoint
curl http://localhost:8600/devices | jq

# Expected output:
# {
#   "cpu": {
#     "cores": 8,
#     "ram_gb": 32
#   },
#   "gpus": [
#     {
#       "index": 0,
#       "name": "RTX 3060",
#       "vram_gb": 12,
#       "backend": "cuda"
#     }
#   ]
# }
```

### Integration Test

```bash
# Terminal 1: Start queen
cargo run --bin queen-rbee

# Terminal 2: Start hive
cargo run --bin rbee-hive -- --port 8600

# Terminal 3: Check hive catalog
sqlite3 queen-hive-catalog.db "SELECT id, devices_json FROM hives WHERE id='localhost';"

# Expected: devices_json should contain CPU and GPU info
```

---

## 📚 Reference Files

### Device Detection Crate
- `bin/25_rbee_hive_crates/device-detection/src/backend.rs` - Backend enum, detect_backends()
- `bin/25_rbee_hive_crates/device-detection/src/detection.rs` - detect_gpus()
- `bin/25_rbee_hive_crates/device-detection/src/types.rs` - GpuDevice, GpuInfo

### Hive Catalog
- `bin/15_queen_rbee_crates/hive-catalog/src/device_types.rs` - Device types
- `bin/15_queen_rbee_crates/hive-catalog/src/catalog.rs` - update_devices() method

### Happy Flow
- `bin/a_human_wrote_this.md` - Lines 38-48 (device detection flow)

---

## ✅ Acceptance Criteria

- [x] Rbee-hive has `/devices` endpoint that returns device capabilities
- [x] Queen requests device detection on first heartbeat (status = Unknown)
- [x] Queen converts DeviceResponse to DeviceCapabilities
- [x] Queen stores devices in hive catalog using `update_devices()`
- [x] Narration shows device summary (CPU cores, RAM, GPU names/VRAM)
- [x] Manual test: `curl http://localhost:8600/devices` returns valid JSON
- [x] Integration test: Hive catalog contains devices_json after first heartbeat
- [x] All existing tests still pass

---

## 🚨 Important Notes

1. **Device detection crate is READY** - Just call `detect_backends()` and `detect_gpus()`
2. **Hive catalog is READY** - Just call `update_devices(id, capabilities)`
3. **Types are READY** - Import from `queen_rbee_hive_catalog::device_types`
4. **This is the "first heartbeat" flow** - Only runs when hive status is Unknown
5. **After storing devices, update status to Online** - Already implemented in heartbeat.rs

---

## 💡 Tips

- Use the existing `DeviceResponse` struct from heartbeat.rs as reference
- The device-detection crate already handles all the hard work (nvidia-smi, etc.)
- Just map from device-detection types to hive-catalog types
- Add good narration so users can see what's happening
- Test with and without GPUs (CPU-only should work too)

---

**TEAM-158: Device types ready. TEAM-159: Implement the flow!**

**Good luck! 🚀**
