# GPU Inventory SPEC — System-Wide GPU State Tracking (GPUINV-12xxx)

**Status**: Draft  
**Applies to**: `bin/pool-managerd-crates/gpu-inventory/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `gpu-inventory` crate provides system-wide GPU monitoring and VRAM accounting for pool-managerd. It uses NVML (NVIDIA Management Library) to query GPU state without allocating resources.

**Why it exists:**
- Pool manager needs to track VRAM across ALL GPUs on the system
- NVML provides read-only queries (separate from CUDA allocation)
- Need preflight validation before spawning workers ("Does GPU 0 have 16GB free?")
- Update VRAM accounting when workers start/stop

**What it does:**
- Query all NVIDIA GPUs on system via NVML FFI
- Track total/free/allocated VRAM per GPU
- Update allocated_vram when workers start/stop
- Expose GPU capabilities (compute capability, model name, PCI bus)
- Monitor GPU health (temperature, utilization, ECC errors)

**What it does NOT do:**
- ❌ Allocate VRAM (workers do this via CUDA)
- ❌ Make placement decisions (orchestratord does this)
- ❌ Spawn workers (worker-lifecycle does this)
- ❌ Validate model compatibility (capability-matcher does this)

**FFI Boundary:**
- Uses **NVML** (NVIDIA Management Library) - read-only query API
- Does NOT use CUDA (workers use CUDA for allocation)

---

## 1. Core Responsibilities

### [GPUINV-12001] GPU Discovery
The crate MUST discover all NVIDIA GPUs on the local system at initialization.

### [GPUINV-12002] VRAM Tracking
The crate MUST track per-GPU VRAM: `total_bytes`, `free_bytes`, `allocated_bytes`.

### [GPUINV-12003] VRAM Accounting Updates
The crate MUST update `allocated_bytes` when workers start/stop (via callback from worker-lifecycle).

### [GPUINV-12004] GPU Properties
The crate MUST expose GPU properties: compute capability, model name, PCI bus ID, driver version.

---

## 2. NVML Initialization

### [GPUINV-12010] NVML Init
At initialization, the crate MUST:
1. Call `nvmlInit()` to initialize NVML library
2. Query device count via `nvmlDeviceGetCount()`
3. For each device, get device handle via `nvmlDeviceGetHandleByIndex()`
4. Query device properties
5. Store device handles for subsequent queries

### [GPUINV-12011] NVML Cleanup
At shutdown, the crate MUST call `nvmlShutdown()`.

### [GPUINV-12012] NVML Version Check
The crate SHOULD verify NVML API version compatibility at init.

---

## 3. VRAM Tracking

### [GPUINV-12020] VRAM Query
The crate MUST query VRAM via `nvmlDeviceGetMemoryInfo()`:
```c
nvmlMemory_t memory;
nvmlDeviceGetMemoryInfo(device, &memory);
// memory.total   = Total VRAM in bytes
// memory.free    = Free VRAM in bytes
// memory.used    = Used VRAM in bytes
```

### [GPUINV-12021] Accounting Model
For each GPU, track:
```rust
struct GpuVramState {
    total_bytes: u64,        // From NVML (constant)
    system_used_bytes: u64,  // From NVML memory.used (includes OS, display, other processes)
    allocated_bytes: u64,    // Tracked by pool manager (sum of worker allocations)
    available_bytes: u64,    // Calculated: total - system_used - allocated
}
```

### [GPUINV-12022] Update on Worker Start
When worker starts and reports VRAM usage:
```rust
fn record_worker_allocation(&mut self, gpu_id: u32, worker_vram_bytes: u64) {
    self.gpus[gpu_id].allocated_bytes += worker_vram_bytes;
    self.gpus[gpu_id].update_available();
}
```

### [GPUINV-12023] Update on Worker Stop
When worker stops:
```rust
fn release_worker_allocation(&mut self, gpu_id: u32, worker_vram_bytes: u64) {
    self.gpus[gpu_id].allocated_bytes -= worker_vram_bytes;
    self.gpus[gpu_id].update_available();
}
```

---

## 4. GPU Properties

### [GPUINV-12030] Device Properties
The crate MUST expose per-GPU:
```rust
struct GpuInfo {
    id: u32,
    model_name: String,          // via nvmlDeviceGetName()
    compute_capability: (u32, u32), // via nvmlDeviceGetCudaComputeCapability()
    pci_bus_id: String,          // via nvmlDeviceGetPciInfo()
    driver_version: String,      // via nvmlSystemGetDriverVersion()
    cuda_version: String,        // via nvmlSystemGetCudaDriverVersion()
}
```

### [GPUINV-12031] Persistence Mode
The crate SHOULD check persistence mode via `nvmlDeviceGetPersistenceMode()` and warn if disabled.

---

## 5. Health Monitoring

### [GPUINV-12040] Temperature Monitoring
The crate SHOULD query GPU temperature via `nvmlDeviceGetTemperature()`:
- Warn if temperature > 85°C
- Critical if temperature > 95°C

### [GPUINV-12041] Utilization Monitoring
The crate SHOULD query GPU utilization via `nvmlDeviceGetUtilizationRates()`:
- Track GPU compute utilization percentage
- Track memory bandwidth utilization percentage

### [GPUINV-12042] ECC Errors
The crate SHOULD query ECC memory errors via `nvmlDeviceGetMemoryErrorCounter()`:
- Track corrected/uncorrected errors
- Warn on uncorrected errors (GPU instability)

### [GPUINV-12043] Power Monitoring
The crate MAY query power draw via `nvmlDeviceGetPowerUsage()`.

---

## 6. API

### [GPUINV-12050] Public Interface
```rust
pub struct GpuInventory {
    gpus: Vec<GpuState>,
}

impl GpuInventory {
    /// Initialize NVML and discover GPUs
    pub fn init() -> Result<Self>;
    
    /// Get all GPUs with current state
    pub fn get_all_gpus(&self) -> &[GpuState];
    
    /// Get single GPU state
    pub fn get_gpu(&self, id: u32) -> Option<&GpuState>;
    
    /// Update VRAM accounting when worker starts
    pub fn record_worker_allocation(&mut self, gpu_id: u32, vram_bytes: u64);
    
    /// Update VRAM accounting when worker stops
    pub fn release_worker_allocation(&mut self, gpu_id: u32, vram_bytes: u64);
    
    /// Refresh VRAM state from NVML (query current free/used)
    pub fn refresh_vram_state(&mut self) -> Result<()>;
    
    /// Get GPU health status
    pub fn get_health(&self, gpu_id: u32) -> Result<GpuHealth>;
}

pub struct GpuState {
    pub info: GpuInfo,
    pub vram: GpuVramState,
    pub health: GpuHealth,
}

pub struct GpuHealth {
    pub temperature_celsius: u32,
    pub utilization_percent: u32,
    pub power_watts: Option<u32>,
    pub ecc_errors: Option<EccErrors>,
}
```

---

## 7. Error Handling

### [GPUINV-12060] Error Types
```rust
pub enum GpuInventoryError {
    NvmlInitFailed(String),
    NvmlNotAvailable,
    DeviceNotFound(u32),
    QueryFailed(String),
    InvalidGpuId(u32),
}
```

### [GPUINV-12061] Graceful Degradation
If NVML is unavailable (no NVIDIA drivers), the crate SHOULD return error at init (fail fast).

---

## 8. Dependencies

### [GPUINV-12070] Required Crates
```toml
[dependencies]
nvml-wrapper = { workspace = true }  # NVML FFI bindings
thiserror = { workspace = true }
tracing = { workspace = true }
serde = { workspace = true, features = ["derive"] }
```

---

## 9. Testing

### [GPUINV-12080] Unit Tests
The crate MUST include tests for:
- VRAM accounting updates (record/release)
- Available VRAM calculation
- Error handling

### [GPUINV-12081] Integration Tests
The crate SHOULD include tests for:
- NVML initialization (requires GPU)
- GPU discovery (requires GPU)
- VRAM queries (requires GPU)

### [GPUINV-12082] Mock Support
The crate SHOULD support mocking for tests without GPUs.

---

## 10. Traceability

**Code**: `bin/pool-managerd-crates/gpu-inventory/src/`  
**Tests**: `bin/pool-managerd-crates/gpu-inventory/tests/`  
**Parent**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Used by**: `pool-managerd`, `worker-lifecycle`, `control-api`  
**Spec IDs**: GPUINV-12001 to GPUINV-12082

---

**End of Specification**
