# GPU Info SPEC — Runtime GPU Detection

**Status**: Draft  
**Applies to**: `bin/shared-crates/gpu-info/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This crate provides runtime GPU detection and information for all llama-orch services. It enforces GPU-only policy and provides device information for CUDA initialization.

**Used by**: pool-managerd, worker-orcd, vram-residency, queen-rbee

---

## 1. GPU Detection Requirements

### 1.1 Detection Methods

**GPU-INFO-001**: The crate MUST support detection via `nvidia-smi` as the primary method.

**GPU-INFO-002**: The crate MAY support detection via CUDA runtime API as an optional feature.

**GPU-INFO-003**: Detection MUST NOT require CUDA runtime library in default configuration.

**GPU-INFO-004**: Detection MUST work even if CUDA driver is outdated (nvidia-smi only).

---

### 1.2 Information Provided

**GPU-INFO-010**: For each detected GPU, the crate MUST provide:
- Device index (0, 1, 2, ...)
- GPU name/model
- Total VRAM in bytes
- Free VRAM in bytes
- Compute capability (major, minor)
- PCI bus ID

**GPU-INFO-011**: The crate MUST provide aggregate information:
- Total number of GPUs
- Total VRAM across all GPUs
- Total free VRAM across all GPUs

**GPU-INFO-012**: VRAM information SHOULD be real-time (not cached).

---

## 2. API Requirements

### 2.1 Detection Functions

```rust
/// Detect GPUs (returns empty if none found)
pub fn detect_gpus() -> GpuInfo;

/// Detect GPUs or fail if none found
pub fn detect_gpus_or_fail() -> Result<GpuInfo>;

/// Check if any GPU is available
pub fn has_gpu() -> bool;

/// Get number of available GPUs
pub fn gpu_count() -> usize;

/// Assert GPU is available (fail fast if not)
pub fn assert_gpu_available() -> Result<()>;
```

**GPU-INFO-020**: `detect_gpus()` MUST NOT fail if no GPU is detected (returns empty).

**GPU-INFO-021**: `detect_gpus_or_fail()` MUST fail with `GpuError::NoGpuDetected` if no GPU found.

**GPU-INFO-022**: `assert_gpu_available()` MUST be suitable for startup validation (fail fast).

---

### 2.2 Device Selection

```rust
impl GpuInfo {
    /// Get GPU with most free VRAM
    pub fn best_gpu_for_workload(&self) -> Option<&GpuDevice>;
    
    /// Validate device index
    pub fn validate_device(&self, device: u32) -> Result<&GpuDevice>;
}
```

**GPU-INFO-030**: `best_gpu_for_workload()` MUST select GPU with most free VRAM.

**GPU-INFO-031**: `validate_device()` MUST return error if device index out of range.

---

## 3. nvidia-smi Integration

### 3.1 Command Execution

**GPU-INFO-040**: The crate MUST execute:
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id \
           --format=csv,noheader,nounits
```

**GPU-INFO-041**: The crate MUST handle nvidia-smi not found gracefully.

**GPU-INFO-042**: The crate MUST parse CSV output robustly (skip malformed lines).

---

### 3.2 Output Parsing

**GPU-INFO-050**: The crate MUST parse nvidia-smi output format:
```
0, NVIDIA GeForce RTX 3090, 24576, 23456, 8.6, 0000:01:00.0
1, NVIDIA GeForce RTX 3060 Lite Hash Rate, 12288, 11234, 8.6, 0000:02:00.0
```

**GPU-INFO-051**: Memory values MUST be interpreted as megabytes and converted to bytes.

**GPU-INFO-052**: Compute capability MUST be parsed as "major.minor" (e.g., "8.6" → (8, 6)).

**GPU-INFO-053**: Parsing errors MUST NOT cause panic (return error instead).

---

## 4. Error Handling

### 4.1 Error Types

```rust
pub enum GpuError {
    NoGpuDetected,
    NvidiaSmiNotFound,
    NvidiaSmiParseFailed(String),
    InvalidDevice(u32, usize),
    Io(std::io::Error),
}
```

**GPU-INFO-060**: All errors MUST be actionable (tell user what to do).

**GPU-INFO-061**: Error messages MUST be specific (exact failure reason).

**GPU-INFO-062**: Errors MUST NOT contain sensitive data.

---

## 5. Performance Requirements

**GPU-INFO-070**: Detection via nvidia-smi SHOULD complete within 100ms.

**GPU-INFO-071**: Detection MUST NOT block for more than 5 seconds.

**GPU-INFO-072**: Detection results MAY be cached (caller's responsibility).

---

## 6. Platform Support

**GPU-INFO-080**: The crate MUST support Linux.

**GPU-INFO-081**: The crate SHOULD support Windows.

**GPU-INFO-082**: The crate MUST gracefully handle unsupported platforms (return no GPU).

---

## 7. Security Requirements

**GPU-INFO-090**: The crate MUST use TIER 2 Clippy configuration (no panics, no unwrap).

**GPU-INFO-091**: nvidia-smi output MUST be validated before parsing.

**GPU-INFO-092**: Command injection MUST be prevented (no user input in command).

---

## 8. Testing Requirements

**GPU-INFO-100**: The crate MUST have unit tests for parsing logic.

**GPU-INFO-101**: The crate MUST have integration tests that work without GPU.

**GPU-INFO-102**: The crate SHOULD have integration tests that run with GPU if available.

---

## 9. Dependencies

**Allowed**:
- `thiserror` — Error types
- `tracing` — Logging
- `serde` — Serialization

**Forbidden**:
- CUDA runtime libraries (in default build)
- Heavy dependencies (keep minimal)

---

## 10. Traceability

**Code**: `bin/shared-crates/gpu-info/src/`  
**Tests**: `bin/shared-crates/gpu-info/tests/`  
**Consumers**:
- `bin/pool-managerd/src/validation/preflight.rs`
- `bin/worker-orcd/src/cuda_ffi/mod.rs`
- `bin/worker-orcd-crates/vram-residency/tests/`

---

## 11. Refinement Opportunities

### 11.1 Enhanced Detection

**Future work**:
- Add GPU topology detection (NVLink, PCIe lanes)
- Add power usage monitoring
- Add temperature monitoring
- Add GPU utilization percentage

### 11.2 Performance Optimization

**Future work**:
- Implement caching strategy
- Add async detection
- Optimize nvidia-smi parsing

### 11.3 Platform Expansion

**Future work**:
- Add AMD GPU support (rocm-smi)
- Add Intel GPU support (intel_gpu_top)
- Add macOS Metal detection

---

**End of Specification**
