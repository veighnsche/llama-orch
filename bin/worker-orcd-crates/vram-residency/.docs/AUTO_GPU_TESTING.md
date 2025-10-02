# Automatic GPU Testing

**Status**: ✅ Operational  
**Date**: 2025-10-02

---

## Overview

The `vram-residency` crate **automatically detects GPU availability** and runs tests on real VRAM when possible. No manual configuration or separate test scripts required.

---

## How It Works

### Build-Time Detection

The `build.rs` script automatically:

1. **Detects NVIDIA GPU** via `nvidia-smi --query-gpu=count`
2. **Detects CUDA Toolkit** via `nvcc --version` (checks `/opt/cuda/bin` and `/usr/local/cuda/bin`)
3. **Queries GPU Compute Capability** via `nvidia-smi --query-gpu=compute_cap`
4. **Selects Correct Architecture** (e.g., `sm_86` for Ampere RTX 30xx)
5. **Compiles CUDA Kernels** if both GPU and CUDA toolkit found
6. **Falls Back to Mock** if either is missing

### Runtime Behavior

| Environment | Detection Result | Test Mode |
|-------------|------------------|-----------|
| Dev machine with GPU + CUDA | ✅ GPU detected | **Real VRAM** |
| CI/CD runner (no GPU) | ❌ No GPU | **Mock VRAM** |
| CI/CD GPU runner | ✅ GPU detected | **Real VRAM** |
| Forced mock mode | 🔧 Override | **Mock VRAM** |

---

## Usage

### Standard Testing (Auto-Detect)

```bash
# Just run cargo test - auto-detects GPU
cargo test -p vram-residency

# BDD tests also auto-detect
cd bin/worker-orcd-crates/vram-residency/bdd
cargo test
```

**Output when GPU detected**:
```
warning: vram-residency@0.0.0: GPU detected - building with real CUDA
warning: vram-residency@0.0.0: Tests will run on real GPU VRAM
warning: vram-residency@0.0.0: Compiling for GPU architecture: sm_86
warning: vram-residency@0.0.0: CUDA kernels compiled successfully
```

**Output when no GPU**:
```
warning: vram-residency@0.0.0: Building with mock VRAM (no GPU/CUDA detected)
warning: vram-residency@0.0.0: Tests will use mock VRAM allocator
warning: vram-residency@0.0.0: Mock CUDA compiled for testing
```

### Force Mock Mode

```bash
# Override auto-detection to use mock (useful for testing mock behavior)
VRAM_RESIDENCY_FORCE_MOCK=1 cargo test -p vram-residency
```

---

## Architecture Detection

The build script automatically detects your GPU's compute capability and selects the appropriate CUDA architecture:

| GPU Series | Compute Capability | CUDA Architecture |
|------------|-------------------|-------------------|
| RTX 40xx (Ada) | 8.9 | `sm_89` |
| RTX 30xx (Ampere) | 8.6 | `sm_86` |
| RTX 20xx (Turing) | 7.5 | `sm_75` |
| GTX 10xx (Pascal) | 6.1 | `sm_61` |

**Fallback**: If detection fails, defaults to `sm_86` (Ampere).

---

## Test Coverage

### Unit Tests (87 tests)

All unit tests work in both mock and GPU modes:
- ✅ Allocator operations
- ✅ Seal/verify workflows
- ✅ Security validation
- ✅ Input validation

### CUDA Kernel Tests (25 tests)

These tests **require real GPU** and are skipped in mock mode:
- ✅ Context creation
- ✅ Memory allocation
- ✅ Host↔Device transfers
- ✅ Bounds checking
- ✅ Error recovery

### BDD Tests (7 features)

BDD tests automatically adapt to available hardware:
- ✅ `seal_model.feature`
- ✅ `verify_seal.feature`
- ✅ `multi_shard.feature`
- ✅ `security.feature`
- ✅ `error_recovery.feature`
- ✅ `seal_verification_extended.feature`
- ⏭️ `vram_policy.feature` (requires real GPU for UMA detection)

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# CPU-only runner (uses mock)
test-mock:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - run: cargo test -p vram-residency

# GPU runner (uses real VRAM)
test-gpu:
  runs-on: [self-hosted, gpu]
  steps:
    - uses: actions/checkout@v4
    - run: cargo test -p vram-residency
    # Auto-detects GPU and runs on real VRAM!
```

### Local Development

```bash
# On your dev machine with GPU
cargo test -p vram-residency
# → Automatically uses real GPU

# On a laptop without GPU
cargo test -p vram-residency
# → Automatically uses mock

# Force mock for testing mock behavior
VRAM_RESIDENCY_FORCE_MOCK=1 cargo test -p vram-residency
```

---

## Implementation Details

### Build Script Logic

```rust
fn should_use_real_cuda() -> bool {
    // Allow explicit override
    if env::var("VRAM_RESIDENCY_FORCE_MOCK").is_ok() {
        return false;
    }
    
    // Check for GPU
    let has_gpu = Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    
    // Check for CUDA toolkit
    setup_cuda_paths(); // Add /opt/cuda/bin to PATH
    let has_nvcc = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    
    has_gpu && has_nvcc
}
```

### BDD Test Adaptation

```rust
// BDD tests detect GPU at runtime
let gpu_info = gpu_info::detect_gpus();
let using_real_gpu = gpu_info.available;

if !using_real_gpu {
    // Only call mock functions in mock mode
    unsafe { vram_reset_mock_state(); }
    std::env::set_var("MOCK_VRAM_MB", capacity_mb.to_string());
}
```

---

## Benefits

### For Developers

✅ **Zero Configuration** - Just run `cargo test`  
✅ **Automatic Optimization** - Uses real GPU when available  
✅ **Consistent Experience** - Same commands work everywhere  
✅ **Fast Feedback** - Real GPU tests complete in ~2 seconds

### For CI/CD

✅ **Flexible Runners** - Works on both CPU and GPU runners  
✅ **No Special Setup** - Standard `cargo test` commands  
✅ **Cost Optimization** - Use cheap CPU runners for most tests, GPU runners for validation  
✅ **Parallel Execution** - Run mock and GPU tests in parallel

### For Testing

✅ **Real Hardware Validation** - Catches GPU-specific bugs  
✅ **Mock Fallback** - Tests work without GPU  
✅ **Same Test Suite** - No separate test files for mock vs GPU  
✅ **Architecture Portability** - Auto-adapts to different GPU models

---

## Troubleshooting

### GPU Detected But Tests Use Mock

**Symptom**: Build says "GPU detected" but tests still use mock

**Causes**:
1. CUDA toolkit not installed
2. `nvcc` not in PATH
3. CUDA libraries not found

**Solution**:
```bash
# On CachyOS/Arch
sudo pacman -S cuda

# Verify nvcc is accessible
nvcc --version

# Check CUDA paths
ls /opt/cuda/bin/nvcc
ls /opt/cuda/lib64/libcudart.so
```

### Build Fails with "Unsupported gpu architecture"

**Symptom**: `nvcc fatal: Unsupported gpu architecture 'sm_XX'`

**Cause**: GPU compute capability detection failed or CUDA version too old

**Solution**:
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Update CUDA toolkit if needed
sudo pacman -S cuda
```

### Tests Fail on Real GPU

**Symptom**: Tests pass in mock mode but fail on real GPU

**Cause**: Real GPU has different behavior (alignment, timing, errors)

**Solution**:
1. Check test output for specific error
2. Verify GPU has enough free VRAM: `nvidia-smi`
3. Run with `--nocapture` to see detailed output:
   ```bash
   cargo test -p vram-residency -- --nocapture
   ```

---

## Comparison: Before vs After

### Before (Manual Script)

```bash
# Had to use separate script
./test_on_real_gpu.sh

# Different commands for mock vs GPU
cargo test  # mock only
VRAM_RESIDENCY_BUILD_CUDA=1 cargo test  # GPU
```

### After (Automatic)

```bash
# Single command works everywhere
cargo test -p vram-residency

# Auto-detects and uses best mode
# No environment variables needed
# No separate scripts
```

---

## Related Documentation

- `README.md` - Testing section with auto-detection info
- `build.rs` - Implementation of auto-detection logic
- `.docs/testing/BDD_RUST_MOCK_LESSONS_LEARNED.md` - BDD testing patterns
- `.docs/CUDA_SETUP.md` - CUDA installation guide

---

**Last Updated**: 2025-10-02  
**Status**: Production-ready  
**Tested On**: CachyOS with RTX 3060/3090, CUDA 13.0
