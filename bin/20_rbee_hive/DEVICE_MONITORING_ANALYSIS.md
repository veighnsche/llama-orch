# Device Monitoring and Heartbeat Analysis

**Date:** Oct 30, 2025  
**Status:** DEPRECATED - See `.specs/HEARTBEAT_ARCHITECTURE.md`  
**Purpose:** ~~Analyze device detection, monitoring, and heartbeat architecture~~

---

## ⚠️ DEPRECATED

**This document contains outdated heartbeat architecture.**

**See canonical specification:** `/home/vince/Projects/llama-orch/bin/.specs/HEARTBEAT_ARCHITECTURE.md`

**Key changes:**
- Pull-based discovery (Queen initiates, not hive)
- No blind heartbeats (hives only send after discovery)
- Monitor data included in heartbeat payload
- Capability changes sent incrementally

---

# Original Analysis (DEPRECATED)

---

## Executive Summary

**Current State:**
- ✅ **device-detection** - COMPLETE (GPU detection, CPU cores, RAM)
- ✅ **Hive heartbeat** - COMPLETE (sends HiveInfo to Queen every 30s)
- ✅ **/capabilities endpoint** - COMPLETE (bypasses job server, returns device list)
- ⏳ **vram-checker** - STUB (placeholder only) - REDUNDANT!
- ⏳ **monitor** - STUB (placeholder only) - NEEDED!
- ⏳ **download-tracker** - STUB (placeholder only) - Nice-to-have
- ✅ **model-preloader** - STUB (created for RAM caching) - Nice-to-have

**Critical Finding:**
- HiveInfo contract has TODO for system stats (CPU%, RAM, VRAM per device)
- Heartbeat sends static HiveInfo, NOT live stats
- Queen scheduler needs live stats for decisions

**Recommendation:**
- Enhance HiveInfo with live system stats
- Use device-detection for real-time metrics
- Consolidate vram-checker into device-detection (already has VRAM!)
- Implement monitor crate for CPU/RAM usage tracking
- download-tracker is nice-to-have for model downloads

---

## 1. device-detection Crate

**Location:** `bin/25_rbee_hive_crates/device-detection/`

**Status:** ✅ COMPLETE

**Purpose:** Runtime GPU detection and system information

### What It Provides

```rust
// GPU Detection
pub fn detect_gpus() -> GpuInfo;
pub fn detect_gpus_or_fail() -> Result<GpuInfo>;
pub fn has_gpu() -> bool;
pub fn gpu_count() -> usize;

// System Information (TEAM-159)
pub fn get_cpu_cores() -> u32;
pub fn get_system_ram_gb() -> u32;

// Backend Detection
pub fn detect_backends() -> Vec<Backend>;
pub struct Backend {
    name: String,
    available: bool,
    capabilities: BackendCapabilities,
}
```

### GpuInfo Structure

```rust
pub struct GpuInfo {
    pub available: bool,
    pub count: usize,
    pub devices: Vec<GpuDevice>,
}

pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub vram_total_bytes: usize,
    pub vram_free_bytes: usize,      // ← ALREADY HAS VRAM!
    pub compute_capability: (u32, u32),
    pub pci_bus_id: String,
}

impl GpuDevice {
    pub fn vram_utilization(&self) -> f64;
    pub fn vram_free_gb(&self) -> f64;
    pub fn vram_total_gb(&self) -> f64;
    pub fn vram_used_gb(&self) -> f64;
    pub fn has_free_vram(&self, required_bytes: usize) -> bool;
}
```

### Current Usage in Hive

**In main.rs:**
```rust
// Detect GPUs
let gpu_info = rbee_hive_device_detection::detect_gpus();

// Get CPU/RAM
let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();

// Build capabilities response
let devices = if gpu_info.available {
    gpu_info.devices.iter().map(|gpu| HiveDevice {
        id: format!("cuda:{}", gpu.index),
        device_type: "cuda".to_string(),
        name: gpu.name.clone(),
        vram_total_gb: gpu.vram_total_gb(),
        vram_free_gb: gpu.vram_free_gb(),
    }).collect()
} else {
    vec![HiveDevice {
        id: "cpu:0".to_string(),
        device_type: "cpu".to_string(),
        name: format!("CPU ({} cores, {} GB RAM)", cpu_cores, system_ram_gb),
        vram_total_gb: 0.0,
        vram_free_gb: 0.0,
    }]
};
```

### Key Features

1. **GPU Detection** - Uses nvidia-smi to detect NVIDIA GPUs
2. **VRAM Tracking** - Already tracks total and free VRAM per GPU
3. **CPU Detection** - Uses num_cpus crate
4. **RAM Detection** - Uses sysinfo crate
5. **Backend Detection** - Detects CUDA, Metal, Vulkan, etc.

### Assessment

**✅ SUFFICIENT for device detection**

This crate already provides:
- GPU detection with VRAM tracking
- CPU core count
- System RAM

**No additional crate needed for basic device info.**

---

## 2. vram-checker Crate

**Location:** `bin/25_rbee_hive_crates/vram-checker/`

**Status:** ⏳ STUB (placeholder only)

**Current Implementation:**
```rust
//! VRAM checker for rbee-hive
//!
//! This crate provides VRAM availability checking and admission control
//! for GPU devices to ensure models can be loaded.

// Placeholder implementation
```

**Dependencies:**
- rbee-hive-device-detection (already has VRAM!)

### Analysis

**❌ REDUNDANT - device-detection already has VRAM tracking!**

**What vram-checker SHOULD do (if kept):**
- Admission control (can model fit on GPU?)
- VRAM reservation system (prevent oversubscription)
- Model size estimation (predict VRAM usage)

**Recommendation:**
- **Option 1:** Delete vram-checker, use device-detection directly
- **Option 2:** Implement admission control logic in vram-checker

**Current Status:** Not needed for basic functionality. device-detection already provides:
- `vram_total_bytes`, `vram_free_bytes` per GPU
- `has_free_vram(required_bytes)` method
- `vram_utilization()` calculation

---

## 3. monitor Crate

**Location:** `bin/25_rbee_hive_crates/monitor/`

**Status:** ⏳ STUB (placeholder only)

**Current Implementation:**
```rust
//! rbee-hive-monitor
//!
//! System monitoring functionality for rbee-hive

// TODO: Implement monitoring functionality
```

**Dependencies:** None (needs to be added)

### What It SHOULD Provide

```rust
// Live system metrics
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub vram_per_device: Vec<VramInfo>,
    pub temperature_celsius: Option<f32>,
    pub uptime_seconds: u64,
}

pub struct VramInfo {
    pub device_index: u32,
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
    pub vram_utilization_percent: f32,
}

// Monitoring functions
pub fn get_system_metrics() -> SystemMetrics;
pub fn get_cpu_usage() -> f32;
pub fn get_ram_usage() -> (f32, f32); // (used, total)
pub fn get_vram_usage() -> Vec<VramInfo>;
```

### Recommended Implementation

**Use sysinfo crate for CPU/RAM:**
```rust
use sysinfo::{System, SystemExt, ProcessorExt};

pub fn get_cpu_usage() -> f32 {
    let mut sys = System::new_all();
    sys.refresh_cpu();
    sys.global_processor_info().cpu_usage()
}

pub fn get_ram_usage() -> (f32, f32) {
    let mut sys = System::new_all();
    sys.refresh_memory();
    let used = sys.used_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let total = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    (used, total)
}
```

**Use device-detection for VRAM:**
```rust
pub fn get_vram_usage() -> Vec<VramInfo> {
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    gpu_info.devices.iter().map(|gpu| VramInfo {
        device_index: gpu.index,
        vram_used_gb: gpu.vram_used_gb() as f32,
        vram_total_gb: gpu.vram_total_gb() as f32,
        vram_utilization_percent: (gpu.vram_utilization() * 100.0) as f32,
    }).collect()
}
```

### Assessment

**⏳ NEEDED for live system stats**

This crate should:
- Track CPU usage (%)
- Track RAM usage (used/total)
- Track VRAM usage (delegate to device-detection)
- Provide unified SystemMetrics struct

**Priority:** HIGH (needed for heartbeat enhancement)

---

## 4. download-tracker Crate

**Location:** `bin/25_rbee_hive_crates/download-tracker/`

**Status:** ⏳ STUB (placeholder only)

**Current Implementation:**
```rust
//! rbee-hive-download-tracker
//!
//! Download progress tracking for model downloads

// TODO: Implement download tracker functionality
```

### What It SHOULD Provide

```rust
pub struct DownloadTracker {
    active_downloads: HashMap<String, DownloadProgress>,
}

pub struct DownloadProgress {
    pub model_id: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_bytes_per_sec: u64,
    pub eta_seconds: u64,
    pub started_at: DateTime<Utc>,
    pub status: DownloadStatus,
}

pub enum DownloadStatus {
    Downloading,
    Paused,
    Completed,
    Failed { error: String },
}

impl DownloadTracker {
    pub fn start_download(&mut self, model_id: String, total_bytes: u64);
    pub fn update_progress(&mut self, model_id: &str, bytes_downloaded: u64);
    pub fn get_progress(&self, model_id: &str) -> Option<&DownloadProgress>;
    pub fn list_active(&self) -> Vec<&DownloadProgress>;
    pub fn complete_download(&mut self, model_id: &str);
    pub fn fail_download(&mut self, model_id: &str, error: String);
}
```

### Assessment

**📋 NICE-TO-HAVE for model downloads**

This crate is:
- Not critical for MVP
- Needed for TEAM-269 (model provisioner)
- Provides better UX (progress bars, ETA)

**Priority:** MEDIUM (implement with model provisioner)

---

## 5. Hive Heartbeat

**Location:** `bin/20_rbee_hive/src/heartbeat.rs`

**Status:** ✅ IMPLEMENTED (but incomplete)

### Current Implementation

```rust
pub async fn send_heartbeat_to_queen(hive_info: &HiveInfo, queen_url: &str) -> Result<()> {
    let heartbeat = HiveHeartbeat::new(hive_info.clone());
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/hive-heartbeat", queen_url))
        .json(&heartbeat)
        .send()
        .await?;
    // ... error handling ...
}

pub fn start_heartbeat_task(hive_info: HiveInfo, queen_url: String) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            if let Err(e) = send_heartbeat_to_queen(&hive_info, &queen_url).await {
                tracing::warn!("Failed to send hive heartbeat: {}", e);
            }
        }
    })
}
```

### HiveInfo Contract

**Current (hive-contract/src/types.rs):**
```rust
pub struct HiveInfo {
    pub id: String,
    pub hostname: String,
    pub port: u16,
    pub operational_status: OperationalStatus,
    pub health_status: HealthStatus,
    pub version: String,
    
    // TODO: TEAM-284: Add system stats
    // pub cpu_usage_percent: f32,
    // pub ram_used_gb: f32,
    // pub ram_total_gb: f32,
    // pub vram_per_device: Vec<VramInfo>,
    // pub temperature_celsius: Option<f32>,
}
```

### Problem

**Heartbeat sends STATIC HiveInfo, not live stats!**

```rust
// In main.rs - created ONCE at startup
let hive_info = hive_contract::HiveInfo {
    id: args.hive_id.clone(),
    hostname: args.hostname.clone(),
    port: args.port,
    operational_status: OperationalStatus::Ready,
    health_status: HealthStatus::Healthy,
    version: env!("CARGO_PKG_VERSION").to_string(),
};

// Heartbeat task uses SAME hive_info forever
let _heartbeat_handle = heartbeat::start_heartbeat_task(hive_info, args.queen_url.clone());
```

**This means:**
- CPU usage is never updated
- RAM usage is never updated
- VRAM usage is never updated
- Operational status is always "Ready"
- Health status is always "Healthy"

**Queen scheduler cannot make decisions based on stale data!**

---

## 6. /capabilities Endpoint

**Location:** `bin/20_rbee_hive/src/main.rs`

**Status:** ✅ IMPLEMENTED (bypasses job server)

### Current Implementation

```rust
async fn get_capabilities() -> Json<CapabilitiesResponse> {
    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    
    // Get CPU/RAM
    let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
    let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();
    
    // Build device list
    let devices = if gpu_info.available {
        gpu_info.devices.iter().map(|gpu| HiveDevice {
            id: format!("cuda:{}", gpu.index),
            device_type: "cuda".to_string(),
            name: gpu.name.clone(),
            vram_total_gb: gpu.vram_total_gb(),
            vram_free_gb: gpu.vram_free_gb(),
        }).collect()
    } else {
        vec![HiveDevice {
            id: "cpu:0".to_string(),
            device_type: "cpu".to_string(),
            name: format!("CPU ({} cores, {} GB RAM)", cpu_cores, system_ram_gb),
            vram_total_gb: 0.0,
            vram_free_gb: 0.0,
        }]
    };
    
    Json(CapabilitiesResponse { devices })
}
```

### Assessment

**✅ SUFFICIENT for device capabilities**

This endpoint:
- Bypasses job server (direct GET request)
- Returns real-time device info
- Includes VRAM free/total per GPU
- Includes CPU cores and RAM

**However:**
- Does NOT include CPU usage %
- Does NOT include RAM usage (only total)
- Does NOT include VRAM usage % (only free/total)

**Recommendation:** Enhance with live stats from monitor crate

---

## Recommendations

### 1. Enhance HiveInfo Contract

**Add system stats fields:**
```rust
pub struct HiveInfo {
    // ... existing fields ...
    
    // Live system stats
    pub cpu_usage_percent: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub vram_per_device: Vec<VramInfo>,
    pub uptime_seconds: u64,
}

pub struct VramInfo {
    pub device_index: u32,
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
    pub vram_utilization_percent: f32,
}
```

### 2. Implement monitor Crate

**Priority:** HIGH

**Implementation:**
```rust
// bin/25_rbee_hive_crates/monitor/src/lib.rs
use rbee_hive_device_detection;
use sysinfo::{System, SystemExt};

pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub vram_per_device: Vec<VramInfo>,
    pub uptime_seconds: u64,
}

pub fn get_system_metrics() -> SystemMetrics {
    let mut sys = System::new_all();
    sys.refresh_all();
    
    // CPU usage
    let cpu_usage = sys.global_processor_info().cpu_usage();
    
    // RAM usage
    let ram_used = sys.used_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let ram_total = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    
    // VRAM usage (delegate to device-detection)
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    let vram_per_device = gpu_info.devices.iter().map(|gpu| VramInfo {
        device_index: gpu.index,
        vram_used_gb: gpu.vram_used_gb() as f32,
        vram_total_gb: gpu.vram_total_gb() as f32,
        vram_utilization_percent: (gpu.vram_utilization() * 100.0) as f32,
    }).collect();
    
    // Uptime
    let uptime_seconds = sys.uptime();
    
    SystemMetrics {
        cpu_usage_percent: cpu_usage,
        ram_used_gb: ram_used,
        ram_total_gb: ram_total,
        vram_per_device,
        uptime_seconds,
    }
}
```

### 3. Update Heartbeat to Use Live Stats

**Modify heartbeat task:**
```rust
pub fn start_heartbeat_task(
    hive_id: String,
    hostname: String,
    port: u16,
    queen_url: String,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Get LIVE system metrics
            let metrics = rbee_hive_monitor::get_system_metrics();
            
            // Build HiveInfo with live stats
            let hive_info = HiveInfo {
                id: hive_id.clone(),
                hostname: hostname.clone(),
                port,
                operational_status: OperationalStatus::Ready, // TODO: calculate based on load
                health_status: HealthStatus::Healthy,         // TODO: calculate based on metrics
                version: env!("CARGO_PKG_VERSION").to_string(),
                cpu_usage_percent: metrics.cpu_usage_percent,
                ram_used_gb: metrics.ram_used_gb,
                ram_total_gb: metrics.ram_total_gb,
                vram_per_device: metrics.vram_per_device,
                uptime_seconds: metrics.uptime_seconds,
            };
            
            if let Err(e) = send_heartbeat_to_queen(&hive_info, &queen_url).await {
                tracing::warn!("Failed to send hive heartbeat: {}", e);
            }
        }
    })
}
```

### 4. Enhance /capabilities Endpoint

**Add live stats:**
```rust
async fn get_capabilities() -> Json<CapabilitiesResponse> {
    let metrics = rbee_hive_monitor::get_system_metrics();
    
    // ... existing device detection ...
    
    Json(CapabilitiesResponse {
        devices,
        cpu_usage_percent: metrics.cpu_usage_percent,
        ram_used_gb: metrics.ram_used_gb,
        ram_total_gb: metrics.ram_total_gb,
        uptime_seconds: metrics.uptime_seconds,
    })
}
```

### 5. Delete or Implement vram-checker

**Option 1: Delete** (recommended)
- device-detection already has VRAM tracking
- No need for separate crate

**Option 2: Implement admission control**
- Model size estimation
- VRAM reservation system
- Prevent oversubscription

**Recommendation:** Delete for now, implement later if needed

### 6. Implement download-tracker (Later)

**Priority:** MEDIUM

Implement when TEAM-269 adds model provisioner:
- Track active downloads
- Show progress, speed, ETA
- Emit narration events

### 7. Implement model-preloader (Later)

**Priority:** LOW (nice-to-have optimization)

**Purpose:** Pre-load GGUF models into RAM for faster VRAM transfer

**Benefits:**
- 3-5x faster worker startup for cached models
- Disk → VRAM (slow, 5-10s) vs RAM → VRAM (fast, 1-2s)
- Better UX for burst worker spawns

**Implementation:**
```rust
// Stub created at: bin/25_rbee_hive_crates/model-preloader/

pub struct ModelPreloader {
    cache: HashMap<String, Vec<u8>>,
    config: PreloadConfig,
}

pub struct PreloadConfig {
    max_cache_size_gb: f64,      // Default: 16GB
    eviction_policy: EvictionPolicy, // LRU, LFU, FIFO
    auto_preload_count: usize,   // Pre-load top N models
}

// Usage:
let preloader = ModelPreloader::new(PreloadConfig::default());
preloader.preload("llama-3.2-1b").await?;

// When spawning worker:
if let Some(model_data) = preloader.get(&model_id).await? {
    // Fast path: model in RAM
    spawn_worker_with_preloaded_model(model_data).await?;
} else {
    // Slow path: load from disk
    spawn_worker_from_disk(&model_path).await?;
}
```

**Strategies:**
- mmap (memory-mapped file)
- Read into buffer
- Hybrid (mmap + madvise hints)

**See:** `bin/25_rbee_hive_crates/model-preloader/README.md`

---

## Summary

### Current State

| Crate | Status | Purpose | Assessment |
|-------|--------|---------|------------|
| device-detection | ✅ COMPLETE | GPU/CPU/RAM detection | Sufficient |
| vram-checker | ⏳ STUB | VRAM admission control | Redundant (delete) |
| monitor | ⏳ STUB | Live system metrics | NEEDED (implement) |
| download-tracker | ⏳ STUB | Download progress | Nice-to-have |
| model-preloader | ✅ STUB | RAM caching for models | Nice-to-have (optimization) |
| heartbeat | ✅ IMPLEMENTED | Hive → Queen status | Incomplete (no live stats) |
| /capabilities | ✅ IMPLEMENTED | Device info endpoint | Sufficient (enhance later) |

### Action Items

**Priority 1 (Critical):**
1. ✅ Implement monitor crate (CPU%, RAM usage, VRAM usage)
2. ✅ Enhance HiveInfo contract with system stats
3. ✅ Update heartbeat to use live metrics from monitor

**Priority 2 (Important):**
4. ✅ Enhance /capabilities endpoint with live stats
5. ✅ Delete vram-checker crate (redundant)

**Priority 3 (Nice-to-have):**
6. ⏳ Implement download-tracker (with TEAM-269)
7. ⏳ Implement model-preloader (RAM caching optimization)

### Why This Matters

**Queen scheduler needs live stats to make decisions:**
- Which hive has free VRAM for new worker?
- Which hive is least loaded (CPU%, RAM%)?
- Which hive is healthy vs degraded?
- Which hive can handle inference request?

**Without live stats, scheduler is blind!**

**With live stats, scheduler can:**
- Load balance across hives
- Avoid overloaded hives
- Detect unhealthy hives
- Optimize resource utilization

**This is critical for the Rhai scheduler to work correctly.**
