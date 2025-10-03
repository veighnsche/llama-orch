# CUDA Feature Flag

## Overview

The `worker-orcd` binary requires CUDA for production use but can be built **without CUDA** for development on CUDA-less devices using the `cuda` feature flag.

## Building

### Development Mode (Without CUDA)

```bash
# Build without CUDA (default)
cargo build -p worker-orcd

# Run tests without CUDA
cargo test -p worker-orcd
```

**Behavior:**
- ✅ Compiles successfully on machines without CUDA toolkit
- ✅ rust-analyzer works properly
- ✅ Allows development of non-CUDA code paths
- ⚠️ CUDA operations will fail with stub errors at runtime

### Production Mode (With CUDA)

```bash
# Build with CUDA support
cargo build -p worker-orcd --features cuda

# Requires:
# - NVIDIA GPU
# - CUDA toolkit (nvcc, cmake)
# - CUDA_PATH environment variable (or default paths)
```

**Behavior:**
- ✅ Full CUDA functionality enabled
- ✅ Model loading to VRAM
- ✅ GPU inference execution
- ❌ Fails to build without CUDA toolkit installed

## Implementation Details

### Feature Gating

The `cuda` feature gates:
1. **build.rs**: CMake compilation of C++/CUDA code
2. **FFI declarations**: `extern "C"` blocks linking to CUDA library
3. **Runtime behavior**: Stub implementations return errors when CUDA is disabled

### Stub Behavior

When built **without** `cuda` feature:
- `ContextHandle::new()` → Returns `CudaError::DeviceNotFound`
- `ModelHandle::load()` → Returns `CudaError::ModelLoadFailed`
- `InferenceHandle::start()` → Returns `CudaError::InferenceFailed`
- All operations log warnings indicating stub mode

### Feature Detection at Runtime

The worker binary will fail gracefully at startup when CUDA is disabled:

```rust
let cuda_ctx = cuda::safe::ContextHandle::new(args.gpu_device)?;
// Returns Err(CudaError::DeviceNotFound) when built without CUDA
```

## CI/CD Configuration

### GitHub Actions

```yaml
# Development CI (CUDA-less)
- name: Build worker-orcd (stub mode)
  run: cargo build -p worker-orcd

# Production CI (with CUDA)
- name: Build worker-orcd (CUDA)
  run: cargo build -p worker-orcd --features cuda
  # Requires self-hosted runner with NVIDIA GPU
```

### Rust Analyzer

By default, rust-analyzer will build without CUDA, enabling IDE support on development machines.

To enable CUDA in rust-analyzer, add to `.vscode/settings.json`:

```json
{
  "rust-analyzer.cargo.features": ["cuda"]
}
```

## Testing Strategy

### Unit Tests (Without CUDA)

```bash
cargo test -p worker-orcd
```

Tests run with stub implementations. Focus on:
- HTTP API logic
- Error handling paths
- Configuration parsing
- Startup sequence (up to CUDA init)

### Integration Tests (With CUDA)

```bash
cargo test -p worker-orcd --features cuda
# Requires GPU machine
```

Tests validate actual CUDA operations:
- Model loading to VRAM
- Inference execution
- VRAM residency checks
- Device health monitoring

## Workspace Impact

The `cuda` feature is **opt-in** (not default), allowing:
- ✅ `cargo build` to succeed on all machines
- ✅ `cargo test` to run without GPU hardware
- ✅ rust-analyzer to provide IDE support
- ✅ Development on laptops/dev boxes without NVIDIA GPU

Production deployments must explicitly enable:
```bash
cargo build --release --features cuda
```

## Related Files

- `Cargo.toml` - Feature definition
- `build.rs` - Conditional CUDA compilation
- `src/cuda/mod.rs` - Feature-gated FFI and stubs
- `.specs/00_worker-orcd.md` - Worker architecture spec
- `.specs/01_cuda_ffi_boundary.md` - CUDA FFI boundary spec
