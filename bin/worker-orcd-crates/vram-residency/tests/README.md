# VRAM Residency - Dual-Mode Testing

This directory implements the **mandatory dual-mode testing requirement** from spec `42_dual_mode_testing.md`.

## Overview

All tests run in **TWO modes**:
1. **Mock VRAM** (always) - Fast, no GPU required
2. **Real CUDA** (if available) - Full coverage with actual GPU

## Quick Start

### Run All Tests

```bash
# From workspace root
cargo test -p vram-residency

# Or from this directory
cd bin/worker-orcd-crates/vram-residency
cargo test
```

### Expected Output

**With GPU:**
```
🧪 Running with MOCK VRAM...
✅ Mock mode: PASSED
🎮 GPU detected: NVIDIA RTX 4090
   VRAM: 24 GB
🧪 Running with REAL CUDA...
✅ Real CUDA mode: PASSED

test result: ok. 162 passed; 0 failed
```

**Without GPU:**
```
🧪 Running with MOCK VRAM...
✅ Mock mode: PASSED

⚠️  ═══════════════════════════════════════════════
⚠️  WARNING: NO CUDA FOUND
⚠️  ONLY MOCK VRAM HAS BEEN TESTED!
⚠️  CUDA FFI layer NOT verified
⚠️  Real VRAM operations NOT tested
⚠️  Install NVIDIA GPU + CUDA for full coverage
⚠️  ═══════════════════════════════════════════════

test result: ok. 162 passed; 0 failed
```

## Test Files

### Integration Tests

- **`dual_mode_example.rs`** - Example tests demonstrating the dual-mode pattern
- **`test_runner.rs`** - Test runner that prints mode information
- **`robustness_concurrent.rs`** - Concurrent access tests
- **`robustness_properties.rs`** - Property-based tests (proptest)
- **`robustness_stress.rs`** - Stress tests
- **`cuda_kernel_tests.rs`** - CUDA kernel tests
- **`proof_bundle_generator.rs`** - Proof bundle generation

### Common Utilities

- **`common/mod.rs`** - Shared test utilities
  - `run_dual_mode_test()` - Helper to run tests in both modes
  - `emit_no_cuda_warning()` - Emit the mandatory warning
  - `has_cuda()` - Check if GPU is available

## Writing Dual-Mode Tests

### Basic Pattern

```rust
use vram_residency::{VramManager, VramError};

mod common;
use common::run_dual_mode_test;

#[test]
fn test_my_feature() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing with real CUDA");
        } else {
            println!("   → Testing with mock VRAM");
        }
        
        // Your test logic here
        let data = vec![0x42u8; 1024];
        let shard = manager.seal_model(&data, 0)?;
        assert!(manager.verify_sealed(&shard).is_ok());
        
        Ok::<(), VramError>(())
    });
}
```

### Conditional Logic

```rust
#[test]
fn test_with_conditional_logic() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        // Use different sizes for mock vs real
        let size = if is_real_cuda {
            10 * 1024 * 1024 // 10MB for real GPU
        } else {
            1 * 1024 * 1024  // 1MB for mock
        };
        
        let data = vec![0x42u8; size];
        let shard = manager.seal_model(&data, 0)?;
        
        Ok::<(), VramError>(())
    });
}
```

## BDD Tests

### Run BDD Tests

```bash
# From workspace root
cargo run -p vram-residency-bdd

# Or with specific features
LLORCH_BDD_FEATURE_PATH=tests/features/seal_verify.feature cargo run -p vram-residency-bdd
```

### BDD Output

The BDD runner automatically runs scenarios twice (mock + real CUDA) and emits the same warning if no GPU is found.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: VRAM Residency Tests

on: [push, pull_request]

jobs:
  test-cpu-only:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests (CPU only)
        run: cargo test -p vram-residency
      # Warning will be emitted but tests pass
      
  test-with-gpu:
    runs-on: self-hosted-gpu
    steps:
      - uses: actions/checkout@v3
      - name: Run tests (with GPU)
        run: cargo test -p vram-residency
      # Full coverage achieved
```

## Coverage

### Mock Mode
- **Coverage**: ~95% of codebase
- **What's tested**: Business logic, cryptography, validation, audit
- **What's NOT tested**: CUDA FFI layer, real VRAM operations

### Real CUDA Mode
- **Coverage**: 100% of codebase
- **What's tested**: Everything including CUDA FFI and real VRAM

## Troubleshooting

### "NO CUDA FOUND" Warning

This is **expected** if you don't have an NVIDIA GPU. Tests still pass with mock VRAM.

**To enable full testing:**
1. Install NVIDIA GPU with CUDA support
2. Install CUDA toolkit (`nvidia-cuda-toolkit` on Ubuntu/Debian)
3. Verify with: `nvidia-smi`
4. Re-run tests

### Tests Fail in Mock Mode

If tests fail in mock mode, this indicates a **business logic bug** (not GPU-related).
Fix the issue - it will affect both mock and real CUDA modes.

### Tests Pass in Mock, Fail in Real CUDA

This indicates a **CUDA-specific issue**:
- Check CUDA driver version
- Check VRAM availability
- Check for CUDA errors in logs
- Verify GPU is not in use by other processes

## Compliance

This testing approach is **mandatory** per spec `42_dual_mode_testing.md`.

All new tests MUST:
- ✅ Run with mock VRAM first
- ✅ Attempt real CUDA testing
- ✅ Emit warning if no CUDA found
- ✅ Not fail if no GPU present

## References

- **Spec**: `../.specs/42_dual_mode_testing.md`
- **Testing Strategy**: `../.specs/40_testing.md`
- **GPU Detection**: `/home/vince/Projects/llama-orch/bin/shared-crates/gpu-info`
