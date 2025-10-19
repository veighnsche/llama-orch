# Migration from shared-crates/gpu-info

**Date:** 2025-10-19  
**Reason:** GPU detection is hive-specific, not shared

---

## Why This Move?

### Original Location (WRONG)
- `bin/shared-crates/gpu-info/`
- Assumed to be shared across multiple binaries

### New Location (CORRECT)
- `bin/rbee-hive-crates/device-detection/`
- Hive-specific functionality

---

## Architecture Rationale

### Who Needs GPU Detection?

**ONLY rbee-hive:**
- ✅ rbee-hive runs on GPU nodes
- ✅ rbee-hive detects local GPUs/devices
- ✅ rbee-hive reports GPU state to queen via heartbeats

**NOT other binaries:**
- ❌ Workers: Told which GPU to use (don't detect)
- ❌ Queen: Receives GPU state from hives (doesn't detect)
- ❌ Keeper: CLI tool (no GPU access)

### Therefore: Binary-Specific, Not Shared

Since only 1 binary uses it, it belongs in `rbee-hive-crates/`, not `shared-crates/`.

---

## What Changed

### Crate Rename
```
FROM: gpu-info (gpu_info)
TO:   rbee-hive-device-detection (rbee_hive_device_detection)
```

### Location
```
FROM: bin/shared-crates/gpu-info/
TO:   bin/rbee-hive-crates/device-detection/
```

### Scope Expansion

**Old (gpu-info):**
- Only NVIDIA GPUs via NVML
- GPU-centric naming

**New (device-detection):**
- NVIDIA GPUs via NVML
- Apple Metal devices (future)
- Other accelerators (future)
- Device-agnostic naming

---

## Migration Guide

### Update Imports

**Before:**
```rust
use gpu_info::{detect_gpus, GpuInfo};
```

**After:**
```rust
use rbee_hive_device_detection::{detect_devices, DeviceInfo};
```

### Update Function Calls

**Before:**
```rust
let gpus = detect_gpus()?;
for gpu in gpus {
    println!("GPU {}: {} ({}GB VRAM)", 
        gpu.id, gpu.name, gpu.vram_gb);
}
```

**After:**
```rust
let devices = detect_devices()?;
for device in devices {
    println!("Device {}: {} ({}GB memory)", 
        device.id, device.name, device.memory_gb);
}
```

### Update Cargo.toml

**Before:**
```toml
[dependencies]
gpu-info = { path = "../shared-crates/gpu-info" }
```

**After:**
```toml
[dependencies]
rbee-hive-device-detection = { path = "../rbee-hive-crates/device-detection" }
```

---

## API Changes

### Struct Renames

| Old (gpu-info) | New (device-detection) | Notes |
|----------------|------------------------|-------|
| `GpuInfo` | `DeviceInfo` | More generic |
| `detect_gpus()` | `detect_devices()` | Supports non-GPU devices |
| `vram_bytes` | `memory_bytes` | Generic memory |

### New Features

**Device Types:**
```rust
pub enum DeviceType {
    NvidiaGpu,      // CUDA-capable NVIDIA GPU
    AppleMetal,     // Apple Metal device (future)
    AmdGpu,         // AMD GPU (future)
    IntelGpu,       // Intel GPU (future)
}
```

**Memory Architecture:**
```rust
pub enum MemoryArchitecture {
    Discrete,       // Separate VRAM (NVIDIA, AMD)
    Unified,        // Unified memory (Apple Silicon)
}
```

---

## Deprecation Timeline

### Phase 1: Mark as Deprecated (DONE)
- ✅ Create `DEPRECATED.md` in `shared-crates/gpu-info/`
- ✅ Create this migration guide
- ✅ Update documentation

### Phase 2: Migrate rbee-hive (TODO)
- [ ] Update rbee-hive to use `device-detection`
- [ ] Remove `gpu-info` dependency from rbee-hive
- [ ] Verify compilation and tests

### Phase 3: Remove from Workspace (TODO)
- [ ] Remove `shared-crates/gpu-info` from root `Cargo.toml`
- [ ] Verify no other crates depend on it
- [ ] Delete `shared-crates/gpu-info/` directory

---

## Related Documentation

- **Deprecated crate:** `bin/shared-crates/gpu-info/DEPRECATED.md`
- **New crate:** `bin/rbee-hive-crates/device-detection/README.md`
- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`

---

## Questions?

If you're trying to detect GPUs in a binary other than rbee-hive, **you're doing it wrong**.

- Workers don't detect GPUs (they're told which GPU to use)
- Queen doesn't detect GPUs (it receives GPU state from hives)
- Keeper doesn't detect GPUs (it's a CLI tool)

Only rbee-hive needs device detection.
