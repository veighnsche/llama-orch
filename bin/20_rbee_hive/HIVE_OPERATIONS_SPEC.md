# Hive Operations Specification

**Date:** Oct 30, 2025  
**Status:** COMPREHENSIVE SPEC  
**Purpose:** Define all operations Hive exposes through the job server

---

## Table of Contents

1. [Overview](#overview)
2. [Current Operations (8)](#current-operations-8)
3. [Proposed Operations (15+)](#proposed-operations-15)
4. [Operation Categories](#operation-categories)
5. [Implementation Status](#implementation-status)
6. [Future Enhancements](#future-enhancements)

---

## Overview

### Architecture

```
Client (rbee-keeper, UI, API)
    ↓
POST /v1/jobs (Operation enum)
    ↓
Hive Job Server (job_router.rs)
    ↓
Match operation → Execute handler
    ↓
GET /v1/jobs/{job_id}/stream (SSE narration)
```

### Endpoints

**Job Server (all operations):**
- `POST /v1/jobs` - Submit operation, get job_id
- `GET /v1/jobs/{job_id}/stream` - Stream narration via SSE

**Direct Endpoints (bypass job server):**
- `GET /health` - Health check (returns "ok")
- `GET /capabilities` - Device capabilities (GPU/CPU list)

---

## Current Operations (8)

### 1. Worker Lifecycle (4 operations)

#### WorkerSpawn
**Purpose:** Start a worker process on the hive

**Request:**
```rust
pub struct WorkerSpawnRequest {
    pub hive_id: String,    // Hive ID
    pub model: String,      // Model ID to load
    pub worker: String,     // Worker type: "cpu", "cuda", "metal"
    pub device: u32,        // Device index (0, 1, 2, ...)
}
```

**Response:**
```rust
pub struct WorkerSpawnResponse {
    pub worker_id: String,  // Assigned worker ID
    pub port: u16,          // HTTP port worker is listening on
    pub pid: u32,           // Process ID
    pub status: String,     // "running", "starting"
}
```

**Narration Events:**
- `worker_spawn_start` - Starting worker spawn
- `worker_spawn_health_check` - Polling worker health
- `worker_spawn_complete` - Worker ready (PID, port)
- `worker_spawn_error` - Spawn failed

**Current Implementation:** ✅ COMPLETE

---

#### WorkerProcessList
**Purpose:** List all running worker processes (via `ps`)

**Request:**
```rust
pub struct WorkerProcessListRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct WorkerProcessListResponse {
    pub workers: Vec<WorkerProcessInfo>,
}

pub struct WorkerProcessInfo {
    pub pid: u32,
    pub worker_id: String,
    pub model: String,
    pub port: u16,
    pub status: String,
}
```

**Narration Events:**
- `worker_proc_list_start` - Starting process list
- `worker_proc_list_entry` - Each worker found
- `worker_proc_list_empty` - No workers found
- `worker_proc_list_result` - Summary (N workers)

**Current Implementation:** ✅ COMPLETE (basic ps parsing)

---

#### WorkerProcessGet
**Purpose:** Get details of a specific worker process

**Request:**
```rust
pub struct WorkerProcessGetRequest {
    pub hive_id: String,
    pub pid: u32,
}
```

**Response:**
```rust
pub struct WorkerProcessGetResponse {
    pub worker: WorkerProcessInfo,
}
```

**Current Implementation:** ✅ COMPLETE (basic ps lookup)

---

#### WorkerProcessDelete
**Purpose:** Kill a worker process (SIGTERM → SIGKILL)

**Request:**
```rust
pub struct WorkerProcessDeleteRequest {
    pub hive_id: String,
    pub pid: u32,
}
```

**Response:**
```rust
pub struct WorkerProcessDeleteResponse {
    pub message: String,
    pub pid: u32,
}
```

**Narration Events:**
- `worker_proc_del_start` - Starting kill
- `worker_proc_del_sigterm` - Sent SIGTERM
- `worker_proc_del_sigkill` - Sent SIGKILL (if needed)
- `worker_proc_del_ok` - Process killed

**Current Implementation:** ✅ COMPLETE

---

### 2. Model Management (4 operations)

#### ModelList
**Purpose:** List all models in catalog

**Request:**
```rust
pub struct ModelListRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
}

pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size_bytes: u64,
    pub status: String,  // "available", "downloading", "failed"
}
```

**Current Implementation:** ✅ COMPLETE

---

#### ModelGet
**Purpose:** Get details of a specific model

**Request:**
```rust
pub struct ModelGetRequest {
    pub hive_id: String,
    pub model_id: String,
}
```

**Response:**
```rust
pub struct ModelGetResponse {
    pub model: ModelInfo,
}
```

**Current Implementation:** ✅ COMPLETE

---

#### ModelDelete
**Purpose:** Delete a model from disk

**Request:**
```rust
pub struct ModelDeleteRequest {
    pub hive_id: String,
    pub model_id: String,
}
```

**Response:**
```rust
pub struct ModelDeleteResponse {
    pub message: String,
    pub model_id: String,
}
```

**Current Implementation:** ✅ COMPLETE

---

#### ModelDownload
**Purpose:** Download a model from HuggingFace/URL

**Request:**
```rust
pub struct ModelDownloadRequest {
    pub hive_id: String,
    pub model: String,  // Model name or URL
}
```

**Response:**
```rust
pub struct ModelDownloadResponse {
    pub model_id: String,
    pub status: String,
}
```

**Narration Events:**
- `model_download_start` - Starting download
- `model_download_progress` - Progress updates (bytes, %)
- `model_download_complete` - Download finished
- `model_download_error` - Download failed

**Current Implementation:** ⏳ STUB (TEAM-269)

---

## Proposed Operations (15+)

### 3. Device Management (3 operations)

#### DeviceList
**Purpose:** List all devices (GPUs, CPUs) with real-time stats

**Request:**
```rust
pub struct DeviceListRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct DeviceListResponse {
    pub devices: Vec<DeviceInfo>,
}

pub struct DeviceInfo {
    pub device_id: u32,
    pub device_type: String,  // "cpu", "cuda", "metal"
    pub name: String,
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub vram_used_mb: Option<u64>,
    pub temperature_celsius: Option<f32>,
    pub in_use: bool,
    pub workers: Vec<String>,  // Worker IDs using this device
}
```

**Status:** 📋 PROPOSED

---

#### DeviceGet
**Purpose:** Get detailed info about a specific device

**Request:**
```rust
pub struct DeviceGetRequest {
    pub hive_id: String,
    pub device_id: u32,
}
```

**Response:**
```rust
pub struct DeviceGetResponse {
    pub device: DeviceInfo,
}
```

**Status:** 📋 PROPOSED

---

#### DeviceStats
**Purpose:** Get real-time device statistics

**Request:**
```rust
pub struct DeviceStatsRequest {
    pub hive_id: String,
    pub device_id: Option<u32>,  // None = all devices
}
```

**Response:**
```rust
pub struct DeviceStatsResponse {
    pub devices: Vec<DeviceStats>,
}

pub struct DeviceStats {
    pub device_id: u32,
    pub vram_utilization_percent: f32,
    pub temperature_celsius: Option<f32>,
    pub power_usage_watts: Option<f32>,
    pub fan_speed_percent: Option<f32>,
}
```

**Status:** 📋 PROPOSED

---

### 4. System Monitoring (3 operations)

#### SystemStats
**Purpose:** Get live system metrics (CPU%, RAM%, uptime)

**Request:**
```rust
pub struct SystemStatsRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct SystemStatsResponse {
    pub cpu_usage_percent: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub uptime_seconds: u64,
    pub load_average: (f32, f32, f32),  // 1min, 5min, 15min
}
```

**Status:** 📋 PROPOSED (needs monitor crate)

---

#### ProcessList
**Purpose:** List all processes (not just workers)

**Request:**
```rust
pub struct ProcessListRequest {
    pub hive_id: String,
    pub filter: Option<String>,  // Filter by name
}
```

**Response:**
```rust
pub struct ProcessListResponse {
    pub processes: Vec<ProcessInfo>,
}

pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_percent: f32,
    pub memory_mb: u64,
    pub status: String,
}
```

**Status:** 📋 PROPOSED

---

#### HiveStatus
**Purpose:** Get overall hive health and status

**Request:**
```rust
pub struct HiveStatusRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct HiveStatusResponse {
    pub hive_id: String,
    pub status: String,  // "healthy", "degraded", "unhealthy"
    pub uptime_seconds: u64,
    pub workers_running: u32,
    pub models_available: u32,
    pub devices: Vec<DeviceInfo>,
    pub system_stats: SystemStatsResponse,
}
```

**Status:** 📋 PROPOSED

---

### 5. Worker Binary Management (3 operations)

**Note:** Worker binaries are READ ONLY from Hive's perspective. Queen installs them via PackageSync.

#### WorkerBinaryCatalogList
**Purpose:** List available worker binaries in catalog

**Request:**
```rust
pub struct WorkerBinaryCatalogListRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct WorkerBinaryCatalogListResponse {
    pub binaries: Vec<WorkerBinaryInfo>,
}

pub struct WorkerBinaryInfo {
    pub id: String,
    pub worker_type: String,  // "cpu", "cuda", "metal"
    pub platform: String,     // "linux", "macos", "windows"
    pub version: String,
    pub path: String,
    pub size_bytes: u64,
}
```

**Status:** 📋 PROPOSED

---

#### WorkerBinaryCatalogGet
**Purpose:** Get details of a specific worker binary

**Request:**
```rust
pub struct WorkerBinaryCatalogGetRequest {
    pub hive_id: String,
    pub binary_id: String,
}
```

**Response:**
```rust
pub struct WorkerBinaryCatalogGetResponse {
    pub binary: WorkerBinaryInfo,
}
```

**Status:** 📋 PROPOSED

---

#### WorkerBinaryCatalogRefresh
**Purpose:** Refresh worker binary catalog (scan filesystem)

**Request:**
```rust
pub struct WorkerBinaryCatalogRefreshRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct WorkerBinaryCatalogRefreshResponse {
    pub binaries_found: u32,
    pub binaries_added: u32,
    pub binaries_removed: u32,
}
```

**Status:** 📋 PROPOSED

---

### 6. Model Pre-loading (3 operations)

#### ModelPreload
**Purpose:** Pre-load a model into RAM for faster worker startup

**Request:**
```rust
pub struct ModelPreloadRequest {
    pub hive_id: String,
    pub model_id: String,
}
```

**Response:**
```rust
pub struct ModelPreloadResponse {
    pub model_id: String,
    pub size_bytes: u64,
    pub preload_time_ms: u64,
}
```

**Narration Events:**
- `model_preload_start` - Starting pre-load
- `model_preload_progress` - Loading into RAM
- `model_preload_complete` - Model cached in RAM

**Status:** 📋 PROPOSED (needs model-preloader crate)

---

#### ModelPreloadEvict
**Purpose:** Evict a model from RAM cache

**Request:**
```rust
pub struct ModelPreloadEvictRequest {
    pub hive_id: String,
    pub model_id: String,
}
```

**Response:**
```rust
pub struct ModelPreloadEvictResponse {
    pub model_id: String,
    pub freed_bytes: u64,
}
```

**Status:** 📋 PROPOSED

---

#### ModelPreloadStats
**Purpose:** Get pre-load cache statistics

**Request:**
```rust
pub struct ModelPreloadStatsRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct ModelPreloadStatsResponse {
    pub total_size_bytes: u64,
    pub cached_models: Vec<String>,
    pub hit_rate: f64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}
```

**Status:** 📋 PROPOSED

---

### 7. Download Management (3 operations)

#### DownloadList
**Purpose:** List active downloads

**Request:**
```rust
pub struct DownloadListRequest {
    pub hive_id: String,
}
```

**Response:**
```rust
pub struct DownloadListResponse {
    pub downloads: Vec<DownloadInfo>,
}

pub struct DownloadInfo {
    pub download_id: String,
    pub model_id: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_bytes_per_sec: u64,
    pub eta_seconds: u64,
    pub status: String,  // "downloading", "paused", "completed", "failed"
}
```

**Status:** 📋 PROPOSED (needs download-tracker crate)

---

#### DownloadPause
**Purpose:** Pause an active download

**Request:**
```rust
pub struct DownloadPauseRequest {
    pub hive_id: String,
    pub download_id: String,
}
```

**Response:**
```rust
pub struct DownloadPauseResponse {
    pub download_id: String,
    pub status: String,
}
```

**Status:** 📋 PROPOSED

---

#### DownloadCancel
**Purpose:** Cancel an active download

**Request:**
```rust
pub struct DownloadCancelRequest {
    pub hive_id: String,
    pub download_id: String,
}
```

**Response:**
```rust
pub struct DownloadCancelResponse {
    pub download_id: String,
    pub bytes_downloaded: u64,
}
```

**Status:** 📋 PROPOSED

---

## Operation Categories

### By Functionality

| Category | Operations | Status |
|----------|-----------|--------|
| **Worker Lifecycle** | Spawn, List, Get, Delete | ✅ COMPLETE |
| **Model Management** | List, Get, Delete, Download | ⏳ 3/4 (Download pending) |
| **Device Management** | List, Get, Stats | 📋 PROPOSED |
| **System Monitoring** | SystemStats, ProcessList, HiveStatus | 📋 PROPOSED |
| **Worker Binary Catalog** | List, Get, Refresh | 📋 PROPOSED |
| **Model Pre-loading** | Preload, Evict, Stats | 📋 PROPOSED |
| **Download Management** | List, Pause, Cancel | 📋 PROPOSED |

### By Priority

**Priority 1 (Critical - MVP):**
- ✅ WorkerSpawn, WorkerProcessList, WorkerProcessDelete
- ✅ ModelList, ModelGet, ModelDelete
- ⏳ ModelDownload (TEAM-269)

**Priority 2 (Important - Enhanced UX):**
- 📋 DeviceList, DeviceGet (show GPU/CPU with live stats)
- 📋 SystemStats (CPU%, RAM%, uptime)
- 📋 HiveStatus (overall health dashboard)

**Priority 3 (Nice-to-have - Optimizations):**
- 📋 ModelPreload, ModelPreloadEvict, ModelPreloadStats
- 📋 DownloadList, DownloadPause, DownloadCancel
- 📋 WorkerBinaryCatalogList, WorkerBinaryCatalogGet

---

## Implementation Status

### ✅ Complete (8 operations)

1. WorkerSpawn - Start worker process
2. WorkerProcessList - List running workers
3. WorkerProcessGet - Get worker details
4. WorkerProcessDelete - Kill worker process
5. ModelList - List models in catalog
6. ModelGet - Get model details
7. ModelDelete - Delete model from disk
8. HiveCheck - Diagnostic narration test

### ⏳ In Progress (1 operation)

9. ModelDownload - Download model from HuggingFace (TEAM-269)

### 📋 Proposed (15+ operations)

**Device Management:**
10. DeviceList
11. DeviceGet
12. DeviceStats

**System Monitoring:**
13. SystemStats
14. ProcessList
15. HiveStatus

**Worker Binary Catalog:**
16. WorkerBinaryCatalogList
17. WorkerBinaryCatalogGet
18. WorkerBinaryCatalogRefresh

**Model Pre-loading:**
19. ModelPreload
20. ModelPreloadEvict
21. ModelPreloadStats

**Download Management:**
22. DownloadList
23. DownloadPause
24. DownloadCancel

---

## Future Enhancements

### Phase 1: Core Functionality (Current)
- ✅ Worker lifecycle (spawn, list, kill)
- ✅ Model management (list, get, delete)
- ⏳ Model download (TEAM-269)

### Phase 2: Monitoring & Stats
- Device management (list, stats)
- System monitoring (CPU%, RAM%, uptime)
- Hive status dashboard

### Phase 3: Optimizations
- Model pre-loading (RAM cache)
- Download management (pause, cancel)
- Worker binary catalog

### Phase 4: Advanced Features
- Batch operations (spawn multiple workers, delete multiple models)
- Worker logs streaming
- Model validation (checksum verification)
- GGUF metadata parsing
- Auto-scaling (spawn workers based on load)

---

## Adding a New Operation

### 3-File Pattern

**1. operations-contract/src/lib.rs:**
```rust
// Add to Operation enum
pub enum Operation {
    // ...
    MyNewOperation(MyNewOperationRequest),
}
```

**2. operations-contract/src/requests.rs:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyNewOperationRequest {
    pub hive_id: String,
    // ... other fields
}
```

**3. operations-contract/src/responses.rs:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyNewOperationResponse {
    // ... response fields
}
```

**4. rbee-hive/src/job_router.rs:**
```rust
match operation {
    // ...
    Operation::MyNewOperation(request) => {
        // Implementation
        n!("my_new_op_start", "Starting operation");
        // ... logic ...
        n!("my_new_op_complete", "Operation complete");
    }
}
```

---

## Summary

**Current:** 8 operations (7 complete, 1 in progress)

**Proposed:** 15+ operations across 7 categories

**Priority:**
1. Complete ModelDownload (TEAM-269)
2. Add Device & System monitoring (Priority 2)
3. Add optimizations (Priority 3)

**The Hive job server is the central API for all Hive management operations. Every operation follows the same pattern: POST /v1/jobs → SSE streaming → completion.**
