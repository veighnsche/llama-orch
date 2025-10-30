# Hive Responsibilities & User Controls

**Date:** Oct 30, 2025  
**Status:** SPECIFICATION  
**Purpose:** Define what the Hive is, what it does, and what controls it exposes

---

## Table of Contents

1. [What is a Hive?](#what-is-a-hive)
2. [Core Responsibilities](#core-responsibilities)
3. [What Hive Does NOT Do](#what-hive-does-not-do)
4. [Data Models](#data-models)
5. [User Controls (SDK/UI)](#user-controls-sdkui)
6. [Implementation Status](#implementation-status)
7. [Future Enhancements](#future-enhancements)

---

## What is a Hive?

### Definition

A **Hive** is a **daemon** that runs on a single machine (physical or virtual) and manages:
- **Models** (GGUF files stored on disk)
- **Workers** (LLM inference processes) // could also be for images, fine tuning, could be for all sorts of workers. Maybe even games

### Key Characteristics

- **Daemon-only** - No CLI, all management via HTTP API
- **Single machine** - Manages resources on ONE machine only
- **HTTP API** - Port 7835 (job-based operations)
- **SSE Streaming** - Real-time progress via narration
- **Stateless** - No persistent state (catalogs are disk-based)
- **No heartbeat** - Workers report directly to Queen /// yes and also that hives send heartbeats to the queen

### Architecture Position

```
Queen (Orchestrator)
    ↓
    ├─→ Hive 1 (192.168.1.100)
    │     ├─→ Worker 1 (PID 1234, llama-3.2-1b, GPU 0)
    │     ├─→ Worker 2 (PID 1235, llama-3.2-3b, GPU 1)
    │     └─→ Models: llama-3.2-1b, llama-3.2-3b, llama-3.2-7b
    │
    └─→ Hive 2 (192.168.1.101)
          ├─→ Worker 3 (PID 5678, llama-3.2-1b, CPU)
          └─→ Models: llama-3.2-1b
```

**Flow:**
1. User → Queen → Hive (spawn worker)
2. Queen → Worker (inference requests) - **DIRECT, bypasses Hive**
3. Worker → Queen (heartbeat) - **DIRECT, bypasses Hive**

**Critical:** Hive is **NOT** in the inference path. It only manages worker lifecycle.

---

## Core Responsibilities

### 1. Model Management

**What Hive Does:**
- ✅ **List models** - Show all GGUF files in catalog
- ✅ **Get model details** - Show size, path, status
- ✅ **Delete model** - Remove GGUF file from disk
- ⏳ **Download model** - Fetch GGUF from HuggingFace/URL (TEAM-269)
- ⏳ **Track downloads** - Show progress, speed, ETA (TEAM-269)

**Storage:**
- Models stored in: `~/.cache/rbee/models/` (Linux/Mac) or `%LOCALAPPDATA%\rbee\models\` (Windows)
- Catalog: `ModelCatalog` (filesystem-based, uses `FilesystemCatalog<ModelEntry>`)
- Format: GGUF files only
- Metadata: Each model has `metadata.json` in its subdirectory

**Catalog Architecture (TEAM-267, TEAM-273):**
```
artifact-catalog (shared abstraction)
    ├─→ Artifact trait (id, path, size, status)
    ├─→ ArtifactCatalog trait (add, get, list, remove)
    ├─→ FilesystemCatalog<T> (generic implementation)
    └─→ ArtifactProvisioner trait (download/provision)

model-catalog (concrete implementation)
    └─→ ModelEntry (implements Artifact)
    └─→ ModelCatalog (wraps FilesystemCatalog<ModelEntry>)
```

**Current Implementation (ModelEntry):**

```rust
// TEAM-267: Actual implementation in bin/25_rbee_hive_crates/model-catalog/src/types.rs
pub struct ModelEntry {
    // Identity
    id: String,              // Unique ID (e.g., "meta-llama/Llama-2-7b")
    name: String,            // Human-readable name
    path: PathBuf,           // Absolute path to GGUF file
    
    // Metadata
    size: u64,               // File size in bytes
    
    // Status
    status: ArtifactStatus,  // Available, Downloading, Failed { error }
    added_at: DateTime<Utc>, // When added to catalog
}

pub enum ArtifactStatus {
    Available,
    Downloading,
    Failed { error: String },
}
```

**Missing Properties (TODO):**
```rust
// These should be added to ModelEntry:
pub struct ModelEntry {
    // ... existing fields ...
    
    // GGUF metadata (to be parsed from file)
    format: String,              // "GGUF"
    architecture: String,        // "llama", "mistral", "qwen", etc.
    quantization: String,        // "Q4_K_M", "Q8_0", "F16", etc.
    parameter_count: String,     // "1B", "3B", "7B", "13B", "70B"
    context_length: u32,         // 2048, 4096, 8192, 32768, etc.
    gguf_version: u32,
    tensor_count: u32,
    kv_count: u32,
    
    // Download tracking (if downloading)
    download_progress: Option<DownloadProgress>,
    
    // Validation
    checksum: Option<String>,    // SHA256 hash
    verified: bool,              // Checksum verified?
}

pub struct DownloadProgress {
    bytes_downloaded: u64,
    total_bytes: u64,
    speed_bytes_per_sec: u64,
    eta_seconds: u64,
    started_at: DateTime<Utc>,
    source_url: String,
}
```

### 2. Worker Management

**What Hive Does:**
- ✅ **Spawn worker** - Start worker process with model + device
- ✅ **List workers** - Show all running worker processes (via `ps`)
- ✅ **Get worker details** - Show PID, model, device, port
- ✅ **Kill worker** - Stop worker process (SIGTERM → SIGKILL)

**Worker Lifecycle:**
```
1. Spawn: Hive starts worker binary with args
2. Health check: Hive polls worker HTTP endpoint
3. Register: Worker sends heartbeat to Queen (NOT Hive!)
4. Inference: Queen routes requests DIRECTLY to worker
5. Kill: Hive sends SIGTERM/SIGKILL to process
```

**Properties to Track:**

```rust
pub struct Worker {
    // Process info
    pub pid: u32,                // Process ID
    pub worker_id: String,       // Unique worker ID
    pub port: u16,               // HTTP port (9000-9999)
    
    // Configuration
    pub model: String,           // Model ID being served
    pub worker_type: WorkerType, // Cpu, Cuda, Metal, Vulkan, etc.
    pub device: u32,             // Device index (0, 1, 2, ...)
    
    // Status
    pub status: WorkerStatus,    // Starting, Running, Stopping, Dead
    pub started_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    
    // Resource usage (from ps)
    pub cpu_percent: f32,        // CPU usage %
    pub memory_mb: u64,          // Memory usage (MB)
    pub vram_mb: Option<u64>,    // VRAM usage (MB) - if GPU worker
    
    // Health
    pub health_url: String,      // http://localhost:{port}/health
    pub health_status: HealthStatus,
    pub last_health_check: DateTime<Utc>,
}

pub enum WorkerType {
    CpuLlm,      // CPU inference
    CudaLlm,     // NVIDIA GPU
    MetalLlm,    // Apple Metal
    VulkanLlm,   // Vulkan (future)
    RocmLlm,     // AMD GPU (future)
}

pub enum WorkerStatus {
    Starting,    // Process spawned, waiting for health check
    Running,     // Health check passed, ready for inference
    Stopping,    // SIGTERM sent, waiting for graceful shutdown
    Dead,        // Process not found in ps
}

pub enum HealthStatus {
    Healthy,     // HTTP 200 from /health
    Unhealthy,   // HTTP error or timeout
    Unknown,     // Not checked yet
}
```

### 3. Device Detection & Monitoring

**What Hive Does:**
- ✅ **Detect GPUs** - List available NVIDIA/AMD/Metal GPUs (device-detection crate)
- ✅ **Check VRAM** - Show total/free VRAM per GPU (device-detection crate)
- ✅ **Detect CPU** - Show CPU cores (device-detection crate)
- ✅ **Detect RAM** - Show total system RAM (device-detection crate)
- ⏳ **Monitor CPU usage** - Track CPU % (monitor crate - TODO)
- ⏳ **Monitor RAM usage** - Track RAM used/total (monitor crate - TODO)
- ⏳ **Monitor VRAM usage** - Track VRAM per device (monitor crate - TODO)

**Crates:**
- `device-detection` - Static device info (GPU list, CPU cores, RAM total)
- `monitor` - Live system metrics (CPU%, RAM%, VRAM%)
- `vram-checker` - REDUNDANT (device-detection already has VRAM!)

**Current Implementation (device-detection):**

```rust
// GPU Detection
pub struct GpuInfo {
    pub available: bool,
    pub count: usize,
    pub devices: Vec<GpuDevice>,
}

pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub vram_total_bytes: usize,
    pub vram_free_bytes: usize,      // ← Already has VRAM!
    pub compute_capability: (u32, u32),
    pub pci_bus_id: String,
}

impl GpuDevice {
    pub fn vram_utilization(&self) -> f64;
    pub fn vram_free_gb(&self) -> f64;
    pub fn vram_total_gb(&self) -> f64;
    pub fn vram_used_gb(&self) -> f64;
}

// System Information
pub fn get_cpu_cores() -> u32;
pub fn get_system_ram_gb() -> u32;
```

**Missing (monitor crate - TODO):**

```rust
pub struct SystemMetrics {
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

pub fn get_system_metrics() -> SystemMetrics;
```

### 4. Worker Binary Management

**What Hive Does:**
- ✅ **List worker binaries** - Show available worker types
- ✅ **Find worker binary** - Locate binary by type + platform
- ❌ **Install worker binary** - NOT Hive's job! (Queen's PackageSync)
- ❌ **Update worker binary** - NOT Hive's job! (Queen's PackageSync)

**Catalog Architecture (TEAM-273, TEAM-277):**
```
worker-catalog (concrete implementation)
    └─→ WorkerBinary (implements Artifact)
    └─→ WorkerCatalog (wraps FilesystemCatalog<WorkerBinary>)
    └─→ READ ONLY from Hive's perspective!
```

**Current Implementation (WorkerBinary):**
```rust
// TEAM-273: Actual implementation in bin/25_rbee_hive_crates/worker-catalog/src/types.rs
pub struct WorkerBinary {
    // Identity
    id: String,                  // e.g., "cpu-llm-worker-rbee-v0.1.0-linux"
    worker_type: WorkerType,     // CpuLlm, CudaLlm, MetalLlm
    platform: Platform,          // Linux, MacOS, Windows
    
    // Location
    path: PathBuf,               // Absolute path to binary
    version: String,             // Semantic version (e.g., "0.1.0")
    
    // Metadata
    size: u64,                   // File size in bytes
    status: ArtifactStatus,      // Available, Downloading, Failed
    added_at: DateTime<Utc>,     // When added to catalog
}

pub enum WorkerType {
    CpuLlm,      // CPU inference
    CudaLlm,     // NVIDIA GPU
    MetalLlm,    // Apple Metal
}

pub enum Platform {
    Linux,
    MacOS,
    Windows,
}

impl WorkerType {
    pub fn binary_name(&self) -> &str {
        match self {
            WorkerType::CpuLlm => "cpu-llm-worker-rbee",
            WorkerType::CudaLlm => "cuda-llm-worker-rbee",
            WorkerType::MetalLlm => "metal-llm-worker-rbee",
        }
    }
}

impl Platform {
    pub fn current() -> Self { /* detect current platform */ }
    pub fn extension(&self) -> &str { /* "" or ".exe" */ }
}
```

**Critical (TEAM-277):** 
- Hive does NOT install worker binaries. That's Queen's job via PackageSync.
- WorkerCatalog is READ ONLY from Hive's perspective.
- Hive discovers workers installed by Queen via SSH.
- Hive only manages worker PROCESSES, not binaries.

### 5. Health Monitoring & Heartbeat

**See canonical specification:** `/home/vince/Projects/llama-orch/bin/.specs/HEARTBEAT_ARCHITECTURE.md`

**What Hive Does:**
- ✅ **Health endpoint** - `/health` returns 200 if Hive is alive (bypasses job server)
- ✅ **Capabilities endpoint** - `/capabilities` returns device list + accepts `queen_url` parameter
- ✅ **Hive heartbeat** - Sends monitor data to Queen every 30s (after discovery)
- ✅ **Worker health checks** - Poll worker `/health` endpoints
- ✅ **Resource monitoring** - Track CPU/memory/VRAM usage (included in heartbeat)

**Heartbeat Discovery Protocol (Pull-based):**
```
1. Queen starts, reads SSH config
2. Queen sends GET /capabilities?queen_url=http://queen:7833 to each hive
3. Hive receives request, extracts queen_url, starts heartbeat task
4. Hive responds with capabilities (devices, models, workers)
5. Hive sends heartbeat every 30s with monitor data
```

**Hive Heartbeat Payload:**
```json
{
  "hive_id": "localhost",
  "timestamp": "2025-10-30T14:52:00Z",
  "monitor_data": {
    "cpu_usage_percent": 45.2,
    "ram_used_gb": 12.5,
    "ram_total_gb": 64.0,
    "uptime_seconds": 86400,
    "devices": [
      {
        "device_id": "GPU-0",
        "vram_used_gb": 8.2,
        "vram_total_gb": 24.0,
        "temperature_celsius": 65.0
      }
    ]
  },
  "capability_changes": null  // Only set when models/workers added/removed
}
```

**Capability Changes:**
When hive downloads/removes models or workers, next heartbeat includes:
```json
{
  "capability_changes": {
    "models": [
      { "action": "added", "model_id": "llama-3.2-1b", "size_gb": 2.5 }
    ],
    "workers": []
  }
}
```

**Worker Heartbeat Flow:**
```
1. Hive spawns worker with --queen-url flag
2. Worker sends initial heartbeat to Queen (status=starting)
3. Worker loads model, sends heartbeat (status=ready)
4. Worker continues heartbeat every 30s
5. Queen tracks worker state (NOT Hive!)
```

**Critical:** 
- Hive only sends heartbeats AFTER Queen discovery (no blind heartbeats)
- Monitor data included in heartbeat (no separate monitoring endpoint)
- Workers report directly to Queen (NOT Hive!)

---

## What Hive Does NOT Do

### ❌ Inference Routing

**Why:** Queen routes inference requests **DIRECTLY** to workers.

```
❌ WRONG: Queen → Hive → Worker (adds latency)
✅ RIGHT: Queen → Worker (direct, fast)
```

**Hive is NOT in the inference path!**

### ❌ Worker State Management

**Why:** Workers send heartbeats **DIRECTLY** to Queen.

```
❌ WRONG: Worker → Hive → Queen (state aggregation)
✅ RIGHT: Worker → Queen (direct, simple)
```

**Queen is the single source of truth for worker state.**

### ❌ Worker Binary Installation

**Why:** Queen's PackageSync distributes binaries to Hives.

```
❌ WRONG: Hive downloads/installs worker binaries
✅ RIGHT: Queen pushes binaries to Hive via PackageSync
```

**Hive only USES worker binaries, doesn't install them.**

### ❌ Model Provisioning (Yet)

**Status:** ⏳ TEAM-269 implementing model download

**Why:** Model download is complex (HuggingFace API, progress tracking, etc.)

```
⏳ FUTURE: Hive downloads models from HuggingFace
✅ CURRENT: Models must be manually placed in ~/.rbee/models/
```

### ❌ Hive Heartbeat

**Why:** Simplified architecture - workers report directly to Queen.

```
❌ OLD: Worker → Hive (heartbeat) → Queen (aggregated state)
✅ NEW: Worker → Queen (heartbeat, direct)
```

**See:** `bin/.plan/TEAM_261_SIMPLIFICATION_AUDIT.md`

---

## Catalog Architecture (TEAM-273)

### Design Pattern

The Hive uses a **shared catalog abstraction** to manage both models and worker binaries consistently:

```
artifact-catalog (shared abstraction)
    ├─→ Artifact trait
    │   ├─→ id() -> &str
    │   ├─→ path() -> &Path
    │   ├─→ size() -> u64
    │   ├─→ status() -> &ArtifactStatus
    │   └─→ set_status(status)
    │
    ├─→ ArtifactCatalog trait
    │   ├─→ add(artifact)
    │   ├─→ get(id)
    │   ├─→ list()
    │   ├─→ remove(id)
    │   ├─→ contains(id)
    │   └─→ len()
    │
    ├─→ FilesystemCatalog<T: Artifact>
    │   └─→ Generic implementation using JSON metadata files
    │
    └─→ ArtifactProvisioner trait
        ├─→ provision(id, job_id) -> Artifact
        └─→ supports(id) -> bool
```

### Filesystem Layout

**Models:**
```
~/.cache/rbee/models/
├── meta-llama-Llama-2-7b/
│   ├── metadata.json          # ModelEntry serialized
│   └── model.gguf             # Actual GGUF file
├── mistralai-Mistral-7B/
│   ├── metadata.json
│   └── model.gguf
└── ...
```

**Workers:**
```
~/.cache/rbee/workers/
├── cpu-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json          # WorkerBinary serialized
│   └── cpu-llm-worker-rbee    # Actual binary
├── cuda-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json
│   └── cuda-llm-worker-rbee
└── ...
```

### Metadata Example

**Model metadata.json:**
```json
{
  "artifact": {
    "id": "meta-llama/Llama-2-7b",
    "name": "Llama 2 7B",
    "path": "/home/user/.cache/rbee/models/meta-llama-Llama-2-7b/model.gguf",
    "size": 7340032000,
    "status": "Available",
    "added_at": "2025-10-30T14:00:00Z"
  },
  "added_at": "2025-10-30T14:00:00Z",
  "last_accessed": "2025-10-30T15:30:00Z"
}
```

### Provisioner Pattern (TEAM-269)

**Multi-vendor provisioning:**
```rust
pub trait VendorSource {
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64>;
    fn supports(&self, id: &str) -> bool;
    fn name(&self) -> &str;
}

pub trait ArtifactProvisioner<T: Artifact> {
    async fn provision(&self, id: &str, job_id: &str) -> Result<T>;
    fn supports(&self, id: &str) -> bool;
}

// Example vendors:
// - HuggingFaceVendor (supports "HF:meta-llama/Llama-2-7b")
// - GitHubReleaseVendor (supports "GH:owner/repo@v1.0.0")
// - LocalBuildVendor (supports "local:path/to/binary")
```

**Usage:**
```rust
let provisioner = MultiVendorProvisioner::new(vec![
    Box::new(HuggingFaceVendor::new()),
    Box::new(GitHubReleaseVendor::new()),
]);

// Provision model from HuggingFace
let model = provisioner.provision("HF:meta-llama/Llama-2-7b", job_id).await?;
catalog.add(model)?;
```

### Benefits

1. **Consistency** - Same pattern for models and workers
2. **Type Safety** - Artifact trait ensures all types have required methods
3. **Extensibility** - Easy to add new artifact types (e.g., datasets, configs)
4. **Testability** - Generic implementation is well-tested
5. **Metadata** - Automatic tracking of added_at, last_accessed
6. **Multi-vendor** - Provisioner pattern supports multiple sources

---

## Data Models

### Complete Type Definitions

```rust
// ============================================================================
// MODEL TYPES
// ============================================================================

pub struct Model {
    // Identity
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    
    // Metadata
    pub size_bytes: u64,
    pub format: String,              // "GGUF"
    pub architecture: String,        // "llama", "mistral", "qwen", etc.
    pub quantization: String,        // "Q4_K_M", "Q8_0", "F16", etc.
    pub parameter_count: String,     // "1B", "3B", "7B", "13B", "70B", etc.
    pub context_length: u32,         // 2048, 4096, 8192, 32768, etc.
    
    // GGUF metadata (parsed from file)
    pub gguf_version: u32,
    pub tensor_count: u32,
    pub kv_count: u32,
    
    // Status
    pub status: ModelStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    
    // Download tracking (if downloading)
    pub download_progress: Option<DownloadProgress>,
    
    // Validation
    pub checksum: Option<String>,    // SHA256 hash
    pub verified: bool,              // Checksum verified?
}

pub enum ModelStatus {
    Available,
    Downloading,
    Failed { error: String },
}

pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_bytes_per_sec: u64,
    pub eta_seconds: u64,
    pub started_at: DateTime<Utc>,
    pub source_url: String,
}

// ============================================================================
// WORKER TYPES
// ============================================================================

pub struct Worker {
    // Process info
    pub pid: u32,
    pub worker_id: String,
    pub port: u16,
    
    // Configuration
    pub model: String,
    pub worker_type: WorkerType,
    pub device: u32,
    pub queen_url: String,           // Where to send heartbeats
    
    // Status
    pub status: WorkerStatus,
    pub started_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub uptime_seconds: u64,
    
    // Resource usage (from ps)
    pub cpu_percent: f32,
    pub memory_mb: u64,
    pub vram_mb: Option<u64>,
    
    // Health
    pub health_url: String,
    pub health_status: HealthStatus,
    pub last_health_check: DateTime<Utc>,
    
    // Performance (from worker metrics)
    pub requests_handled: u64,
    pub tokens_generated: u64,
    pub avg_tokens_per_sec: f32,
    pub errors: u64,
}

pub enum WorkerType {
    CpuLlm,
    CudaLlm,
    MetalLlm,
    VulkanLlm,
    RocmLlm,
}

pub enum WorkerStatus {
    Starting,
    Running,
    Stopping,
    Dead,
}

pub enum HealthStatus {
    Healthy,
    Unhealthy { reason: String },
    Unknown,
}

// ============================================================================
// DEVICE TYPES
// ============================================================================

pub struct Device {
    // Identity
    pub device_id: u32,
    pub device_type: DeviceType,
    pub name: String,
    pub vendor: String,              // "NVIDIA", "AMD", "Apple", "Intel"
    
    // Capabilities
    pub vram_total_mb: Option<u64>,
    pub vram_available_mb: Option<u64>,
    pub compute_capability: Option<String>,
    pub max_threads: Option<u32>,
    
    // Status
    pub available: bool,
    pub in_use: bool,
    pub workers: Vec<String>,
    pub temperature_celsius: Option<f32>,
    pub power_usage_watts: Option<f32>,
}

pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

// ============================================================================
// WORKER BINARY TYPES
// ============================================================================

pub struct WorkerBinary {
    // Identity
    pub id: String,
    pub worker_type: WorkerType,
    pub platform: Platform,
    
    // Location
    pub path: PathBuf,
    pub version: String,
    
    // Metadata
    pub size_bytes: u64,
    pub checksum: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    
    // Capabilities
    pub supported_models: Vec<String>,  // Model architectures supported
    pub min_vram_mb: Option<u64>,       // Minimum VRAM required
}

pub enum Platform {
    LinuxX64,
    LinuxArm64,
    MacOSX64,
    MacOSArm64,
    WindowsX64,
}

// ============================================================================
// HIVE STATUS TYPES
// ============================================================================

pub struct HiveStatus {
    // Identity
    pub hive_id: String,
    pub hostname: String,
    pub ip_address: String,
    pub port: u16,
    
    // Status
    pub status: String,              // "running", "starting", "stopping"
    pub uptime_seconds: u64,
    pub started_at: DateTime<Utc>,
    
    // Resources
    pub cpu_cores: u32,
    pub memory_total_mb: u64,
    pub memory_available_mb: u64,
    pub disk_total_gb: u64,
    pub disk_available_gb: u64,
    
    // Devices
    pub devices: Vec<Device>,
    
    // Workers
    pub workers_running: u32,
    pub workers_starting: u32,
    pub workers_stopping: u32,
    pub workers_total: u32,
    
    // Models
    pub models_available: u32,
    pub models_downloading: u32,
    pub models_total_gb: f64,
}
```

---

## User Controls (SDK/UI)

### Operations Exposed via HTTP API

**All operations follow job-based pattern:**
```
POST /v1/jobs → {job_id, sse_url}
GET /v1/jobs/{job_id}/stream → SSE narration
```

### 1. Model Operations

#### List Models
```typescript
// Request
const op = OperationBuilder.modelList(hiveId)
await client.submitAndStream(op, handleLine)

// Response (via narration)
data: {"action":"model_list_entry","message":"llama-3.2-1b | 1.23 GB | available"}
data: {"action":"model_list_entry","message":"llama-3.2-3b | 3.45 GB | downloading"}
data: [DONE]

// Parsed result
interface Model {
  id: string
  name: string
  size_bytes: number
  status: 'available' | 'downloading' | 'failed'
  architecture?: string
  quantization?: string
  parameter_count?: string
  download_progress?: {
    bytes_downloaded: number
    total_bytes: number
    speed_bytes_per_sec: number
    eta_seconds: number
  }
}
```

#### Get Model Details
```typescript
const op = OperationBuilder.modelGet(hiveId, modelId)
await client.submitAndStream(op, handleLine)

// Response includes full model metadata
```

#### Delete Model
```typescript
const op = OperationBuilder.modelDelete(hiveId, modelId)
await client.submitAndStream(op, handleLine)

// Response
data: {"action":"model_delete_complete","message":"✅ Model deleted successfully"}
data: [DONE]
```

#### Download Model (⏳ TEAM-269)
```typescript
const op = OperationBuilder.modelDownload(hiveId, modelName)
await client.submitAndStream(op, handleLine)

// Response (streaming progress)
data: {"action":"model_download_start","message":"📥 Downloading llama-3.2-1b"}
data: {"action":"model_download_progress","message":"Downloaded 123 MB / 1230 MB (10%)"}
data: {"action":"model_download_progress","message":"Downloaded 246 MB / 1230 MB (20%)"}
...
data: {"action":"model_download_complete","message":"✅ Download complete"}
data: [DONE]
```

### 2. Worker Operations

#### List Workers
```typescript
const op = OperationBuilder.workerList(hiveId)
await client.submitAndStream(op, handleLine)

// Response
data: {"action":"worker_proc_list_entry","message":"PID 1234 | llama-3.2-1b | GPU 0 | running"}
data: {"action":"worker_proc_list_entry","message":"PID 1235 | llama-3.2-3b | GPU 1 | running"}
data: [DONE]

// Parsed result
interface Worker {
  pid: number
  worker_id: string
  model: string
  worker_type: 'cpu' | 'cuda' | 'metal'
  device: number
  port: number
  status: 'starting' | 'running' | 'stopping' | 'dead'
  cpu_percent?: number
  memory_mb?: number
  vram_mb?: number
  uptime_seconds?: number
}
```

#### Spawn Worker
```typescript
const op = OperationBuilder.workerSpawn(
  hiveId,
  'llama-3.2-1b',  // model
  'cuda',          // worker type
  0                // device index
)
await client.submitAndStream(op, handleLine)

// Response
data: {"action":"worker_spawn_start","message":"🚀 Spawning worker..."}
data: {"action":"worker_spawn_health_check","message":"Waiting for worker to start..."}
data: {"action":"worker_spawn_complete","message":"✅ Worker spawned (PID: 1234, port: 9301)"}
data: [DONE]
```

#### Get Worker Details
```typescript
const op = OperationBuilder.workerGet(hiveId, pid)
await client.submitAndStream(op, handleLine)

// Response includes full worker details
```

#### Kill Worker
```typescript
const op = OperationBuilder.workerDelete(hiveId, pid)
await client.submitAndStream(op, handleLine)

// Response
data: {"action":"worker_proc_del_start","message":"🗑️ Killing worker PID 1234"}
data: {"action":"worker_proc_del_sigterm","message":"Sent SIGTERM"}
data: {"action":"worker_proc_del_ok","message":"✅ Worker killed successfully"}
data: [DONE]
```

### 3. Device Operations (Future)

```typescript
// List devices
const op = OperationBuilder.deviceList(hiveId)

// Get device details
const op = OperationBuilder.deviceGet(hiveId, deviceId)

// Response
interface Device {
  device_id: number
  device_type: 'cpu' | 'cuda' | 'metal'
  name: string
  vram_total_mb?: number
  vram_available_mb?: number
  available: boolean
  in_use: boolean
  workers: string[]
}
```

### 4. Hive Status (Future)

```typescript
// Get hive status
const op = OperationBuilder.hiveStatus(hiveId)

// Response
interface HiveStatus {
  hive_id: string
  hostname: string
  ip_address: string
  status: string
  uptime_seconds: number
  cpu_cores: number
  memory_total_mb: number
  memory_available_mb: number
  devices: Device[]
  workers_running: number
  models_available: number
}
```

---

## Implementation Status

### ✅ Complete

1. **Job-based architecture** - POST /v1/jobs, SSE streaming
2. **Catalog abstraction (TEAM-273)** - Shared Artifact trait, FilesystemCatalog, ArtifactProvisioner
3. **Model catalog (TEAM-267)** - ModelEntry, ModelCatalog using FilesystemCatalog
4. **Worker catalog (TEAM-273)** - WorkerBinary, WorkerCatalog using FilesystemCatalog
5. **Model list** - Show all models in catalog
6. **Model get** - Show model details
7. **Model delete** - Remove model from disk
8. **Worker spawn** - Start worker process
9. **Worker list** - Show running workers (via ps)
10. **Worker get** - Show worker details
11. **Worker kill** - Stop worker process
12. **Device detection** - Detect GPUs/CPUs
13. **Worker catalog** - List available worker binaries

### ⏳ In Progress

1. **Model download** (TEAM-269) - Download from HuggingFace
2. **Download tracking** (TEAM-269) - Progress, speed, ETA
3. **Model provisioner** (TEAM-269) - Implement HuggingFaceVendor, wire up MultiVendorProvisioner
4. **GGUF metadata parsing** - Extract architecture, quantization, parameter count from GGUF files
5. **Enhanced ModelEntry** - Add missing properties (architecture, quantization, etc.)

### 📋 Planned

1. **Resource monitoring** - CPU/memory/VRAM tracking
2. **Worker health checks** - Periodic health polling
3. **Device operations** - List/get device details
4. **Hive status** - Overall hive health/metrics
5. **Worker metrics** - Requests, tokens, performance
6. **Model validation** - Checksum verification
7. **Worker logs** - Stream worker stdout/stderr
8. **Batch operations** - Spawn multiple workers, delete multiple models
9. **Vendor sources** - GitHubReleaseVendor, LocalBuildVendor
10. **Artifact metadata** - Track last_accessed, usage stats

---

## Future Enhancements

### Phase 1: Core Functionality (Current)
- ✅ Model management (list, get, delete)
- ✅ Worker management (spawn, list, kill)
- ⏳ Model download (TEAM-269)

### Phase 2: Monitoring & Metrics
- Resource monitoring (CPU, memory, VRAM)
- Worker health checks
- Performance metrics (tokens/sec, requests/sec)
- Device utilization tracking

### Phase 3: Advanced Features
- Worker logs streaming
- Batch operations
- Model validation & checksums
- GGUF metadata parsing
- Auto-scaling (spawn workers based on load)

### Phase 4: Polish
- Error recovery
- Graceful shutdown
- Configuration hot-reload
- Metrics export (Prometheus)
- Admin UI improvements

---

## UI Controls Needed

### Model Management Panel

**List View:**
- Table with columns: Name, Size, Architecture, Quantization, Status
- Actions: View Details, Delete
- Filter: By architecture, size, status
- Sort: By name, size, date

**Detail View:**
- Full metadata display
- Download progress (if downloading)
- Checksum verification status
- Actions: Delete, Cancel Download

**Download Form:**
- Model name/URL input
- Source selector (HuggingFace, URL, local file)
- Progress bar during download
- Cancel button

### Worker Management Panel

**List View:**
- Table with columns: PID, Model, Device, Type, Status, CPU%, Memory, Uptime
- Actions: View Details, Kill
- Filter: By model, device, status
- Sort: By PID, uptime, resource usage

**Detail View:**
- Full worker info
- Resource usage graphs
- Performance metrics
- Logs viewer
- Actions: Kill, Restart

**Spawn Form:**
- Model selector (dropdown from available models)
- Worker type selector (CPU, CUDA, Metal)
- Device selector (0, 1, 2, ... based on detected devices)
- Advanced options: Port, context length, batch size
- Spawn button

### Device Panel

**List View:**
- Cards for each device
- Show: Name, Type, VRAM, Status, Workers using it
- Visual indicator: Available (green), In Use (yellow), Unavailable (red)

### Hive Status Panel

**Dashboard:**
- Overall status indicator
- Uptime
- Resource usage (CPU, memory, disk)
- Worker count (running/total)
- Model count
- Recent activity log

---

## Summary

**Hive is responsible for:**
1. ✅ Managing models on disk (list, get, delete, download)
2. ✅ Managing worker processes (spawn, list, kill)
3. ✅ Detecting devices (GPUs, CPUs)
4. ✅ Providing HTTP API for all operations
5. ✅ Streaming progress via SSE narration

**Hive is NOT responsible for:**
1. ❌ Inference routing (Queen does this)
2. ❌ Worker state management (Queen does this)
3. ❌ Worker binary installation (Queen's PackageSync does this)
4. ❌ Hive heartbeat (Workers report directly to Queen)

**User controls needed:**
1. Model list/details/delete/download
2. Worker list/details/spawn/kill
3. Device list/details
4. Hive status dashboard

**Next steps:**
1. Complete model download (TEAM-269)
2. Enhance data models with all properties
3. Build UI components for each control
4. Add resource monitoring
5. Add worker health checks
