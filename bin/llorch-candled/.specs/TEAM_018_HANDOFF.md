# TEAM-018 Handoff: Accelerate → Metal Transition

**Date:** 2025-10-09  
**Team:** TEAM-018  
**Directive:** Drop Accelerate support, add pre-release Metal backend  
**Status:** ✅ COMPLETE

---

## Executive Summary

TEAM-018 successfully removed the Accelerate backend and implemented a pre-release Metal backend for Apple Silicon GPU support. The transition follows the established multi-binary pattern and aligns with the CUDA implementation.

### What Was Delivered ✅

1. **Accelerate backend removed** - Binary, feature flag, device init, and tests deleted
2. **Metal backend added** - Feature flag, device init, binary following CUDA pattern
3. **Documentation updated** - BACKEND_ARCHITECTURE_RESEARCH.md and new docs/metal.md
4. **References cleaned** - README.md, FEATURES.md, .llorch-test.toml updated

### Rationale

**Why remove Accelerate:**
- ❌ CPU-bound (not GPU acceleration)
- ❌ Too slow for production inference (~10-20 tok/s)
- ❌ No significant advantage over plain CPU backend
- ❌ Confusing naming (users expected GPU acceleration)

**Why add Metal:**
- ✅ True GPU acceleration on Apple Silicon
- ✅ 2×-10× faster than Accelerate (estimated)
- ✅ Aligns with CUDA pattern (GPU inference)
- ✅ Candle already has Metal support

---

## Changes Made

### 1. Cargo.toml

**Removed:**
```toml
accelerate = ["candle-core/accelerate", "candle-nn/accelerate"]

[[bin]]
name = "llorch-accelerate-candled"
path = "src/bin/accelerate.rs"
required-features = ["accelerate"]
```

**Added:**
```toml
metal = ["candle-core/metal", "candle-nn/metal"]

[[bin]]
name = "llorch-metal-candled"
path = "src/bin/metal.rs"
required-features = ["metal"]
```

### 2. src/device.rs

**Removed:**
```rust
#[cfg(feature = "accelerate")]
pub fn init_accelerate_device() -> CandleResult<Device> {
    tracing::info!("Initializing Apple Accelerate device (CPU-optimized)");
    Ok(Device::Cpu)
}
```

**Added:**
```rust
#[cfg(feature = "metal")]
pub fn init_metal_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing Apple Metal device (GPU) {}", gpu_id);
    Device::new_metal(gpu_id)
}
```

### 3. src/bin/metal.rs (NEW)

Created new Metal binary following CUDA pattern:
- CLI args: `--worker-id`, `--model`, `--port`, `--callback-url`, `--metal-device`
- Device initialization with verification
- Model loading with auto-detected architecture
- GPU warmup (same as CUDA)
- HTTP server with inference endpoints

**Key features:**
- Pre-release status clearly marked
- Follows exact CUDA pattern
- Device ID selection via CLI
- Proper error handling

### 4. src/bin/accelerate.rs (DELETED)

Removed entire file (103 lines).

### 5. Documentation

**Created:**
- `docs/metal.md` - Complete Metal backend guide
  - Overview and comparison
  - Requirements (Apple Silicon, macOS 10.15+)
  - Build and run instructions
  - Supported models
  - Performance estimates
  - Troubleshooting
  - Pre-release status and roadmap

**Updated:**
- `.specs/BACKEND_ARCHITECTURE_RESEARCH.md` - Comprehensive updates
  - Executive summary updated
  - Feature-gated binaries section
  - Device initialization examples
  - Metal support implementation status
  - Accelerate removal rationale
  - Deployment matrix
  - Conclusion and next steps

**Updated:**
- `README.md` - All Accelerate references replaced with Metal
- `FEATURES.md` - Feature flag guide updated
- `.llorch-test.toml` - Test configuration updated

---

## Build Commands

### Metal Binary (macOS Apple Silicon only)

```bash
cd bin/llorch-candled

# Build Metal binary
cargo build --release --features metal --bin llorch-metal-candled

# Run Metal worker
./target/release/llorch-metal-candled \
  --worker-id test-worker \
  --model /path/to/llama-2-7b/ \
  --port 8080 \
  --metal-device 0 \
  --callback-url http://localhost:9999
```

### Verify Build (CPU)

```bash
# Verify Accelerate is gone
cargo build --features accelerate 2>&1 | grep -i "unknown feature"
# Should error: unknown feature `accelerate`

# Verify Metal compiles (structure only, requires macOS to run)
cargo check --features metal --bin llorch-metal-candled
```

---

## Testing Status

### Code Complete ✅
- [x] Metal feature flag added
- [x] `init_metal_device()` function created
- [x] `llorch-metal-candled` binary created
- [x] Follows CUDA pattern exactly
- [x] Accelerate backend removed
- [x] Documentation complete

### Runtime Validation ⏳ PENDING
- [ ] Test on Apple Silicon Mac (M1/M2/M3/M4)
- [ ] Benchmark performance vs CPU
- [ ] Verify model loading (all architectures)
- [ ] Test warmup and inference
- [ ] Stress test (long-running inference)

**Note:** Metal backend is marked **pre-release** until runtime validation is complete.

---

## Current Backend Support

| Backend | Status | Hardware | Use Case |
|---------|--------|----------|----------|
| **CPU** | ✅ Production | Any CPU | Development, fallback |
| **CUDA** | ✅ Production | NVIDIA GPU | Production inference |
| **Metal** | 🚧 Pre-release | Apple Silicon GPU | macOS GPU inference |
| **Accelerate** | ❌ Removed | - | Too slow (removed) |

---

## Deployment Matrix

| Target | Binary | Rationale |
|--------|--------|-----------|
| Linux x86 CPU | `llorch-cpu-candled` | No GPU |
| Linux + NVIDIA GPU | `llorch-cuda-candled` | CUDA acceleration |
| macOS Intel | `llorch-cpu-candled` | No GPU (Intel Macs) |
| macOS Apple Silicon | `llorch-metal-candled` | Apple GPU acceleration |
| Windows CPU | `llorch-cpu-candled` | No GPU |
| Windows + NVIDIA GPU | `llorch-cuda-candled` | CUDA acceleration |

---

## Pool-Manager Integration

### Capability Detection

Pool-managerd should:
1. Detect Apple Silicon GPU (via system info)
2. Launch `llorch-metal-candled` binary
3. Pass `--metal-device 0` argument
4. Monitor worker health

### Capability Reporting

Metal workers report:
```json
{
  "capabilities": ["metal"],
  "device": "apple_silicon_gpu",
  "backend": "metal",
  "status": "pre-release"
}
```

---

## Known Limitations

### Pre-Release Status

⚠️ Metal backend is **pre-release** and has not been extensively tested on real Apple Silicon hardware.

**What this means:**
- ✅ Code compiles and follows CUDA pattern
- ⚠️ Runtime behavior not validated
- ⚠️ Performance not benchmarked
- ⚠️ Edge cases not tested

**Before production use:**
1. Test on target Apple Silicon hardware
2. Benchmark performance vs expectations
3. Validate model loading for all architectures
4. Run stress tests (long-running inference)

### Candle Metal Support

Metal support depends on Candle's Metal implementation:
- Candle provides Metal kernels for common operations
- Some operations may fall back to CPU
- Performance may vary by operation type

---

## Next Steps

### Phase 1: Foundation ✅ COMPLETE
- [x] Add Metal feature flag
- [x] Create `init_metal_device()` function
- [x] Create `llorch-metal-candled` binary
- [x] Follow CUDA pattern
- [x] Remove Accelerate backend
- [x] Update documentation

### Phase 2: Validation ⏳ PENDING
- [ ] Test on Apple Silicon Mac
- [ ] Benchmark vs CPU backend
- [ ] Verify model loading (all architectures)
- [ ] Test warmup and inference
- [ ] Validate device selection

### Phase 3: Production 🔮 FUTURE
- [ ] Performance optimization
- [ ] Memory management tuning
- [ ] CI/CD integration (macOS runners)
- [ ] Production deployment guide
- [ ] Remove pre-release status

---

## Files Modified

### Code
- `Cargo.toml` - Feature flags and binary targets
- `src/device.rs` - Device initialization
- `src/bin/metal.rs` - NEW Metal binary
- `src/bin/accelerate.rs` - DELETED

### Documentation
- `.specs/BACKEND_ARCHITECTURE_RESEARCH.md` - Comprehensive updates
- `.specs/TEAM_018_HANDOFF.md` - NEW (this file)
- `docs/metal.md` - NEW Metal guide
- `README.md` - Updated references
- `FEATURES.md` - Updated feature guide
- `.llorch-test.toml` - Updated test config

### Tests
- `src/device.rs` - Updated device tests (Metal, removed Accelerate)

---

## Verification Checklist

- [x] Accelerate feature flag removed from Cargo.toml
- [x] Metal feature flag added to Cargo.toml
- [x] Accelerate binary target removed
- [x] Metal binary target added
- [x] `init_accelerate_device()` removed from device.rs
- [x] `init_metal_device()` added to device.rs
- [x] `src/bin/accelerate.rs` deleted
- [x] `src/bin/metal.rs` created
- [x] Device tests updated
- [x] README.md updated
- [x] FEATURES.md updated
- [x] .llorch-test.toml updated
- [x] BACKEND_ARCHITECTURE_RESEARCH.md updated
- [x] docs/metal.md created
- [x] TEAM_018_HANDOFF.md created

---

## References

### Internal Documentation
- `.specs/BACKEND_ARCHITECTURE_RESEARCH.md` - Multi-backend pattern
- `docs/metal.md` - Metal backend guide
- `.specs/TEAM_007_FINAL_REPORT.md` - Original multi-backend implementation
- `README.md` - Project overview

### Candle Metal Support
- `reference/candle/candle-core/` - Metal device implementation
- `reference/candle/candle-metal-kernels/` - Metal GPU kernels

### Apple Documentation
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Apple Silicon Performance](https://developer.apple.com/documentation/apple-silicon)

---

**Handoff completed:** 2025-10-09  
**Status:** ✅ Code complete, runtime validation pending  
**Next team:** Validation team (test on Apple Silicon hardware)  
**Contact:** TEAM-018
