# rbee-hive-device-detection

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** Compute device detection for rbee-hive  
**Location:** `bin/rbee-hive-crates/device-detection/` (hive-specific, NOT shared)

---

## Overview

The `rbee-hive-device-detection` crate provides **device detection functionality** for rbee-hive. It detects available compute devices (NVIDIA GPUs, Apple Metal, CPU) on the local machine and reports their capabilities to the orchestrator.

### System Context

In the llama-orch architecture:

```
┌─────────────────┐
│   rbee-hive     │  ← Pool manager (THIS CRATE)
│ (pool-managerd) │  ← Detects local devices
└────────┬────────┘
         │
         │ Reports device state via heartbeat
         │ POST /v2/pools/{id}/heartbeat
         │ { pool_id, gpus: [...], workers: [...] }
         ↓
┌─────────────────┐
│   queen-rbee    │  ← Orchestrator (uses device info for scheduling)
│ (orchestratord) │
└─────────────────┘
```

**Key Responsibilities:**
- Detect available compute devices on the local machine
- Query device capabilities (VRAM, compute capability, etc.)
- Report device state to queen-rbee via heartbeats
- Support multiple device types (NVIDIA, Apple Metal, CPU)

---

## Why This is Hive-Specific (Not Shared)

### Who Needs Device Detection?

**ONLY rbee-hive:**
- ✅ rbee-hive runs on GPU nodes
- ✅ rbee-hive detects local devices
- ✅ rbee-hive reports device state to queen via heartbeats
- ✅ rbee-hive uses device info for preflight validation

**NOT other binaries:**
- ❌ **Workers:** Told which device to use (don't detect)
- ❌ **Queen:** Receives device state from hives (doesn't detect)
- ❌ **Keeper:** CLI tool (no device access)

### Therefore: Binary-Specific, Not Shared

Since only 1 binary uses it, it belongs in `rbee-hive-crates/`, not `shared-crates/`.

---

## Migration from shared-crates/gpu-info

**This crate replaces the deprecated `shared-crates/gpu-info`.**

See: `MIGRATION_FROM_GPU_INFO.md` for full migration guide.

**Key changes:**
- **Old:** `gpu-info` (GPU-only, incorrectly in shared-crates)
- **New:** `device-detection` (multi-device, correctly in rbee-hive-crates)
- **Scope:** Expanded to support Apple Metal, CPU, and future accelerators

---

## Architecture Principles

### 1. Multi-Device Support

**Supported device types:**
- **NVIDIA GPUs:** Via nvidia-smi (primary) or CUDA runtime (fallback)
- **Apple Metal:** Via Metal framework (future)
- **CPU:** Fallback for inference (future)
- **Other accelerators:** Extensible design (future)

### 2. Detection Strategy

**Hierarchical detection:**
1. Try nvidia-smi (most reliable for NVIDIA)
2. Fall back to CUDA runtime API (if available)
3. Try Metal framework (on macOS)
4. Fall back to CPU (always available)

### 3. Security-First Design

**Security considerations:**
- Validate all parsed data (bounds checking, null byte rejection)
- Use absolute paths for executables (prevent PATH manipulation)
- Limit string lengths (prevent memory exhaustion)
- Saturating arithmetic (prevent overflow)

---

## API Design

### Core Types

```rust
/// Device information
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub device_type: DeviceType,
    pub memory_bytes: usize,
    pub memory_free_bytes: usize,
    pub memory_architecture: MemoryArchitecture,
    pub capabilities: DeviceCapabilities,
}

/// Device type
pub enum DeviceType {
    NvidiaGpu,      // CUDA-capable NVIDIA GPU
    AppleMetal,     // Apple Metal device
    AmdGpu,         // AMD GPU (future)
    IntelGpu,       // Intel GPU (future)
    Cpu,            // CPU fallback
}

/// Memory architecture
pub enum MemoryArchitecture {
    Discrete,       // Separate VRAM (NVIDIA, AMD)
    Unified,        // Unified memory (Apple Silicon)
}

/// Device capabilities
pub struct DeviceCapabilities {
    pub compute_capability: Option<(u32, u32)>,  // NVIDIA only
    pub max_threads: Option<usize>,
    pub pci_bus_id: Option<String>,
}
```

### Detection Functions

```rust
/// Detect all available devices
pub fn detect_devices() -> Vec<DeviceInfo>;

/// Detect devices or fail if none found
pub fn detect_devices_or_fail() -> Result<Vec<DeviceInfo>>;

/// Check if any device is available
pub fn has_device() -> bool;

/// Get device count
pub fn device_count() -> usize;

/// Get specific device info
pub fn get_device_info(device_id: u32) -> Result<DeviceInfo>;
```

---

## Usage Examples

### Basic Detection

```rust
use rbee_hive_device_detection::{detect_devices, DeviceType};

// Detect all devices
let devices = detect_devices();

for device in devices {
    println!("Device {}: {} ({:?})", 
        device.id, 
        device.name, 
        device.device_type
    );
    println!("  Memory: {} GB total, {} GB free",
        device.memory_bytes / (1024 * 1024 * 1024),
        device.memory_free_bytes / (1024 * 1024 * 1024)
    );
}
```

### NVIDIA GPU Detection

```rust
use rbee_hive_device_detection::{detect_devices, DeviceType};

// Find NVIDIA GPUs
let nvidia_gpus: Vec<_> = detect_devices()
    .into_iter()
    .filter(|d| matches!(d.device_type, DeviceType::NvidiaGpu))
    .collect();

for gpu in nvidia_gpus {
    if let Some((major, minor)) = gpu.capabilities.compute_capability {
        println!("GPU {}: {} (compute {}.{})",
            gpu.id, gpu.name, major, minor);
    }
}
```

### Preflight Validation

```rust
use rbee_hive_device_detection::{get_device_info, DeviceType};

// Validate device before spawning worker
fn validate_device_for_model(
    device_id: u32, 
    model_vram_required: usize
) -> Result<(), String> {
    let device = get_device_info(device_id)
        .map_err(|e| format!("Device {} not found: {}", device_id, e))?;
    
    // Check device type
    if !matches!(device.device_type, DeviceType::NvidiaGpu) {
        return Err(format!("Device {} is not an NVIDIA GPU", device_id));
    }
    
    // Check VRAM availability
    if device.memory_free_bytes < model_vram_required {
        return Err(format!(
            "Insufficient VRAM: {} GB required, {} GB available",
            model_vram_required / (1024 * 1024 * 1024),
            device.memory_free_bytes / (1024 * 1024 * 1024)
        ));
    }
    
    Ok(())
}
```

### Integration with Heartbeat

```rust
use rbee_hive_device_detection::detect_devices;
use rbee_heartbeat::{HeartbeatConfig, start_heartbeat_task};

// Detect devices
let devices = detect_devices();

// Configure heartbeat with device state
let config = HeartbeatConfig::new(
    "pool-1".to_string(),
    "http://queen-rbee:8080/v2/pools/pool-1/heartbeat".to_string(),
    15, // 15 seconds
    move |pool_id| {
        // Take snapshot of device state
        let gpus = devices.iter().map(|d| GpuState {
            id: d.id,
            device_name: d.name.clone(),
            total_vram_bytes: d.memory_bytes,
            free_vram_bytes: d.memory_free_bytes,
        }).collect();
        
        PoolHeartbeat {
            pool_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            gpus,
            workers: vec![], // From worker registry
        }
    },
);

start_heartbeat_task(config);
```

---

## Dependencies

### Required

- **`which`**: Find nvidia-smi in PATH
- **`tracing`**: Structured logging
- **`serde`**: Serialization for device info

### Platform-Specific

- **NVIDIA:**
  - `nvidia-smi` executable (runtime dependency)
  - Optional: CUDA runtime library (compile-time feature)

- **Apple Metal:**
  - `metal-rs` crate (future)
  - macOS 10.13+ (future)

---

## Implementation Status

### Phase 1: NVIDIA GPU Support (M1)
- [ ] Migrate from `gpu-info` crate
- [ ] nvidia-smi detection
- [ ] CUDA runtime detection (optional feature)
- [ ] Security hardening (bounds checking, validation)
- [ ] Unit tests with mock nvidia-smi output
- [ ] Integration tests with real GPUs

### Phase 2: Apple Metal Support (M2)
- [ ] Metal framework integration
- [ ] Unified memory architecture support
- [ ] Device capability detection
- [ ] Tests on Apple Silicon

### Phase 3: Multi-Device Support (M3)
- [ ] CPU fallback detection
- [ ] AMD GPU support (ROCm)
- [ ] Intel GPU support
- [ ] Device selection strategies

### Phase 4: Advanced Features (M4+)
- [ ] Device health monitoring
- [ ] Temperature/power monitoring
- [ ] Multi-GPU topology detection
- [ ] PCIe bandwidth detection

---

## Detection Methods

### NVIDIA GPUs

**Primary: nvidia-smi**
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id \
           --format=csv,noheader,nounits
```

**Output:**
```
0, NVIDIA GeForce RTX 4090, 24576, 23456, 8.9, 0000:01:00.0
1, NVIDIA GeForce RTX 3090, 24576, 20123, 8.6, 0000:02:00.0
```

**Fallback: CUDA Runtime API**
```c
cudaGetDeviceCount(&count);
cudaGetDeviceProperties(&props, device);
cudaMemGetInfo(&free, &total);
```

### Apple Metal (Future)

**Metal Framework:**
```swift
MTLCopyAllDevices()
device.recommendedMaxWorkingSetSize
device.hasUnifiedMemory
```

---

## Security Considerations

### Input Validation

**All parsed data is validated:**
- String length limits (prevent memory exhaustion)
- Null byte rejection (prevent injection)
- Numeric bounds checking (prevent overflow)
- Reasonable value ranges (detect malformed data)

### Executable Safety

**nvidia-smi execution:**
- Use absolute path (prevent PATH manipulation)
- Validate output format (prevent injection)
- Timeout protection (prevent hanging)

### Memory Safety

**Arithmetic operations:**
- Saturating multiplication (prevent overflow)
- Bounds clamping (free <= total)
- Maximum limits (1TB VRAM is unreasonable)

---

## Testing Strategy

### Unit Tests

- Parse nvidia-smi output (various formats)
- Validate compute capability parsing
- Test error handling (malformed input)
- Security tests (injection, overflow)

### Integration Tests

- Detect real GPUs (if available)
- Query device properties
- Validate VRAM reporting
- Test fallback mechanisms

### Mock Tests

- Mock nvidia-smi output
- Simulate detection failures
- Test error recovery

---

## Performance Considerations

### Detection Overhead

- **nvidia-smi:** ~50-100ms per invocation
- **CUDA runtime:** ~10-20ms per query
- **Caching:** Device list cached, refreshed on demand

### Optimization Strategies

1. **Cache device list:** Detect once at startup
2. **Lazy detection:** Only detect when needed
3. **Parallel queries:** Query multiple devices concurrently
4. **Incremental updates:** Only query changed properties

---

## Related Crates

### Used By
- **`rbee-hive`**: Main binary that uses this crate

### Integrates With
- **`rbee-hive-crates/worker-lifecycle`**: Uses device info for worker spawning
- **`rbee-hive-crates/http-server`**: Reports device state via heartbeats
- **`shared-crates/heartbeat`**: Sends device state to queen-rbee

### Replaces
- **`shared-crates/gpu-info`**: DEPRECATED (see `MIGRATION_FROM_GPU_INFO.md`)

---

## Specification References

- **SYS-6.2.1**: Pool Manager Execution (device queries)
- **SYS-6.2.2**: State Reporting (device state in heartbeats)
- **SYS-6.2.3**: Preflight Validation (device capability checks)
- **SYS-6.2.4**: Heartbeat Protocol (device state aggregation)

See: `/home/vince/Projects/llama-orch/bin/.specs/00_llama-orch.md`

---

## Team History

- **TEAM-115**: Original `gpu-info` implementation
- **TEAM-135**: Scaffolding for new crate-based architecture
- **2025-10-19**: Moved from `shared-crates/gpu-info` to `rbee-hive-crates/device-detection`

---

## Migration Notes

**Migrating from `shared-crates/gpu-info`?**

See: `MIGRATION_FROM_GPU_INFO.md` for complete migration guide.

**Quick summary:**
- `gpu-info` → `rbee-hive-device-detection`
- `GpuInfo` → `DeviceInfo`
- `detect_gpus()` → `detect_devices()`
- `vram_bytes` → `memory_bytes`

---

**Next Steps:**
1. Migrate implementation from `shared-crates/gpu-info`
2. Expand API to support multiple device types
3. Add Apple Metal detection
4. Add CPU fallback detection
5. Integrate with rbee-hive heartbeat system
