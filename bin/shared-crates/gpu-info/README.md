# gpu-info

**Runtime GPU detection and information for llama-orch**

`bin/shared-crates/gpu-info` — Detects available NVIDIA GPUs at runtime, queries VRAM capacity, and enforces GPU-only policy. Used by pool-managerd for preflight checks, worker-orcd for CUDA initialization, and vram-residency for test auto-detection.

---

## What This Crate Offers

`gpu-info` provides **runtime GPU detection** for all llama-orch services. Here's what we offer:

### 🎮 Core Capabilities

**1. GPU Detection**
- Automatic detection of NVIDIA GPUs via `nvidia-smi`
- Fallback to CUDA runtime API (optional feature)
- Fail-fast validation for GPU-only enforcement
- Zero-cost when no GPU available (graceful degradation)

**2. GPU Information**
- Device count and indices
- GPU names and models
- VRAM capacity (total and free)
- Compute capability version
- PCI bus IDs

**3. Multi-GPU Support**
- Enumerate all available GPUs
- Query per-device information
- Select best GPU for workload (most free VRAM)
- Validate device indices

**4. Testing Support**
- Auto-detect GPU for test selection
- Mock-friendly API (no GPU required)
- Runtime detection (not build-time)

---

## What You Get

### For pool-managerd (Preflight Checks)

```rust
use gpu_info::assert_gpu_available;

// Fail fast if no GPU detected
pub fn assert_gpu_only() -> Result<()> {
    assert_gpu_available()
        .context("GPU-only enforcement: CUDA not detected")
}
```

### For worker-orcd (CUDA Initialization)

```rust
use gpu_info::GpuInfo;

// Initialize CUDA context with validation
pub fn initialize_cuda(device: u32) -> Result<CudaContext> {
    let gpu_info = GpuInfo::detect_or_fail()?;
    
    // Validate device index
    if device as usize >= gpu_info.count {
        return Err(CudaError::InvalidDevice(device));
    }
    
    let gpu = &gpu_info.devices[device as usize];
    tracing::info!(
        device = %device,
        name = %gpu.name,
        vram_gb = %(gpu.vram_total_bytes / 1024 / 1024 / 1024),
        "Initializing CUDA on {}", gpu.name
    );
    
    Ok(CudaContext::new(device)?)
}
```

### For vram-residency (Test Auto-Detection)

```rust
use gpu_info::GpuInfo;

#[test]
fn test_vram_operations() {
    let gpu_info = GpuInfo::detect();
    
    if gpu_info.available {
        println!("🎮 Detected {} GPU(s)", gpu_info.count);
        for gpu in &gpu_info.devices {
            println!("   GPU {}: {} ({} GB VRAM)", 
                gpu.index, gpu.name, gpu.vram_total_bytes / 1024 / 1024 / 1024);
        }
        test_with_real_gpu(&gpu_info);
    } else {
        println!("💻 No GPU detected, using mock");
        test_with_mock();
    }
}
```

### For orchestratord (Node Registration)

```rust
use gpu_info::GpuInfo;

// Report GPU capabilities during node registration
pub fn register_node() -> Result<NodeInfo> {
    let gpu_info = GpuInfo::detect_or_fail()?;
    
    Ok(NodeInfo {
        node_id: generate_node_id(),
        gpu_count: gpu_info.count,
        total_vram_gb: gpu_info.total_vram_bytes() / 1024 / 1024 / 1024,
        gpus: gpu_info.devices.iter().map(|gpu| GpuSpec {
            index: gpu.index,
            name: gpu.name.clone(),
            vram_gb: gpu.vram_total_bytes / 1024 / 1024 / 1024,
        }).collect(),
    })
}
```

---

## API Reference

### Core Types

#### `GpuInfo`

Complete GPU information for all detected devices.

```rust
pub struct GpuInfo {
    pub available: bool,        // True if any GPU detected
    pub count: usize,           // Number of GPUs
    pub devices: Vec<GpuDevice>, // Per-device information
}
```

**Methods**:
```rust
impl GpuInfo {
    /// Detect GPUs (returns empty if none found)
    pub fn detect() -> Self;
    
    /// Detect GPUs or fail if none found
    pub fn detect_or_fail() -> Result<Self>;
    
    /// Get total VRAM across all GPUs
    pub fn total_vram_bytes(&self) -> usize;
    
    /// Get GPU with most free VRAM
    pub fn best_gpu_for_workload(&self) -> Option<&GpuDevice>;
    
    /// Validate device index
    pub fn validate_device(&self, device: u32) -> Result<&GpuDevice>;
}
```

---

#### `GpuDevice`

Information for a single GPU device.

```rust
pub struct GpuDevice {
    pub index: u32,                      // Device index (0, 1, 2, ...)
    pub name: String,                    // GPU name (e.g., "NVIDIA GeForce RTX 3090")
    pub vram_total_bytes: usize,         // Total VRAM in bytes
    pub vram_free_bytes: usize,          // Free VRAM in bytes
    pub compute_capability: (u32, u32),  // Compute capability (e.g., (8, 6))
    pub pci_bus_id: String,              // PCI bus ID (e.g., "0000:01:00.0")
}
```

**Methods**:
```rust
impl GpuDevice {
    /// Get VRAM utilization percentage (0.0 to 1.0)
    pub fn vram_utilization(&self) -> f64;
    
    /// Get free VRAM in GB
    pub fn vram_free_gb(&self) -> f64;
    
    /// Get total VRAM in GB
    pub fn vram_total_gb(&self) -> f64;
    
    /// Check if GPU has sufficient free VRAM
    pub fn has_free_vram(&self, required_bytes: usize) -> bool;
}
```

---

### Convenience Functions

```rust
/// Check if any GPU is available
pub fn has_gpu() -> bool;

/// Get number of available GPUs
pub fn gpu_count() -> usize;

/// Assert GPU is available (fail fast if not)
pub fn assert_gpu_available() -> Result<()>;

/// Get GPU info for specific device
pub fn get_device_info(device: u32) -> Result<GpuDevice>;
```

---

## Detection Methods

### Primary: nvidia-smi (Most Reliable)

**Command**:
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id \
           --format=csv,noheader,nounits
```

**Output**:
```
0, NVIDIA GeForce RTX 3090, 24576, 23456, 8.6, 0000:01:00.0
1, NVIDIA GeForce RTX 3060 Lite Hash Rate, 12288, 11234, 8.6, 0000:02:00.0
```

**Advantages**:
- ✅ No CUDA runtime required
- ✅ Works even if CUDA driver is outdated
- ✅ Provides real-time VRAM usage
- ✅ Available on all NVIDIA driver installations

---

### Fallback: CUDA Runtime API (Optional)

**Requires**: `cuda-runtime` feature flag

**APIs used**:
- `cudaGetDeviceCount()` — Number of devices
- `cudaGetDeviceProperties()` — Device info
- `cudaMemGetInfo()` — VRAM capacity

**Advantages**:
- ✅ Direct CUDA integration
- ✅ No external process spawning
- ✅ Slightly faster

**Disadvantages**:
- ⚠️ Requires CUDA runtime library
- ⚠️ Requires feature flag
- ⚠️ More complex build

---

## Error Handling

### GpuError Enum

```rust
pub enum GpuError {
    NoGpuDetected,                       // No GPU found
    NvidiaSmiNotFound,                   // nvidia-smi not in PATH
    NvidiaSmiParseFailed(String),        // Failed to parse output
    InvalidDevice(u32),                  // Device index out of range
    CudaRuntimeError(String),            // CUDA runtime error
    Io(std::io::Error),                  // I/O error
}
```

**Error messages**:
- ✅ Actionable (tell user what to do)
- ✅ Specific (exact failure reason)
- ✅ Safe (no sensitive data)

---

## Performance

**Detection overhead**:
- `nvidia-smi`: ~50-100ms (spawns process)
- CUDA runtime: ~10-20ms (direct API)

**Caching strategy**:
- Detection results can be cached
- VRAM usage should be queried fresh
- Recommended: Cache device list, query VRAM on-demand

---

## Platform Support

| Platform | nvidia-smi | CUDA Runtime |
|----------|------------|--------------|
| **Linux** | ✅ Primary | ✅ Optional |
| **Windows** | ✅ Supported | ✅ Optional |
| **macOS** | ❌ N/A | ❌ N/A |

**Note**: macOS does not support NVIDIA GPUs (Apple Silicon uses Metal).

---

## Integration Examples

### Example 1: Startup Validation

```rust
use gpu_info::GpuInfo;

fn main() -> Result<()> {
    // Fail fast if no GPU
    let gpu_info = GpuInfo::detect_or_fail()
        .context("This application requires an NVIDIA GPU")?;
    
    tracing::info!(
        "Detected {} GPU(s) with {} GB total VRAM",
        gpu_info.count,
        gpu_info.total_vram_bytes() / 1024 / 1024 / 1024
    );
    
    // Continue with GPU-dependent initialization
    Ok(())
}
```

---

### Example 2: Multi-GPU Selection

```rust
use gpu_info::GpuInfo;

fn select_best_gpu() -> Result<u32> {
    let gpu_info = GpuInfo::detect_or_fail()?;
    
    // Select GPU with most free VRAM
    let best_gpu = gpu_info.best_gpu_for_workload()
        .ok_or_else(|| anyhow!("No suitable GPU found"))?;
    
    tracing::info!(
        "Selected GPU {}: {} ({} GB free)",
        best_gpu.index,
        best_gpu.name,
        best_gpu.vram_free_gb()
    );
    
    Ok(best_gpu.index)
}
```

---

### Example 3: VRAM Capacity Check

```rust
use gpu_info::GpuInfo;

fn can_load_model(model_size_bytes: usize, device: u32) -> Result<bool> {
    let gpu_info = GpuInfo::detect_or_fail()?;
    let gpu = gpu_info.validate_device(device)?;
    
    Ok(gpu.has_free_vram(model_size_bytes))
}
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p gpu-info

# Specific test suites
cargo test -p gpu-info detection    # Detection logic
cargo test -p gpu-info parsing      # nvidia-smi parsing
cargo test -p gpu-info validation   # Device validation
```

### Integration Tests

```bash
# Test with real GPU (if available)
cargo test -p gpu-info --test integration

# Test without GPU (mock)
cargo test -p gpu-info --test no_gpu
```

---

## Dependencies

```toml
[dependencies]
thiserror.workspace = true   # Error types
tracing.workspace = true     # Logging
serde = { workspace = true } # Serialization

[features]
cuda-runtime = []  # Optional CUDA runtime API
```

**Why minimal dependencies?**
- ✅ Fast compilation
- ✅ Small binary size
- ✅ Reduced attack surface
- ✅ Easy to audit

---

## Specifications

Implements requirements from:
- **GPU-INFO-001 to GPU-INFO-020**: GPU detection requirements
- **ORCH-1102**: GPU-only policy enforcement
- **ORCH-3202**: Preflight checks

See `.specs/` for full requirements:
- `00_gpu-info.md` — Functional specification
- `10_expectations.md` — Consumer expectations
- `20_security.md` — Security considerations

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Security Tier**: TIER 2 (High-Importance)
- **Priority**: P0 (blocking for worker-orcd)

---

## Roadmap

### Phase 1: Core Detection (Current)
- ✅ nvidia-smi detection
- ✅ CSV parsing
- ✅ Error handling
- ✅ Multi-GPU support
- ⬜ Unit tests
- ⬜ Integration tests

### Phase 2: CUDA Runtime (Optional)
- ⬜ CUDA runtime API integration
- ⬜ Feature flag support
- ⬜ Performance comparison

### Phase 3: Advanced Features
- ⬜ GPU topology detection
- ⬜ NVLink detection
- ⬜ Power usage monitoring
- ⬜ Temperature monitoring

---

## Contributing

**Before implementing**:
1. Read `.specs/00_gpu-info.md` — Functional specification
2. Follow TIER 2 Clippy configuration (no panics, no unwrap)
3. Add tests for all detection paths

**Testing requirements**:
- Unit tests for parsing logic
- Integration tests with/without GPU
- Error handling tests

---

## For Questions

See:
- `.specs/` — Complete specifications
- `bin/pool-managerd/src/validation/preflight.rs` — Current GPU detection
- `bin/worker-orcd/src/cuda_ffi/mod.rs` — CUDA integration point
