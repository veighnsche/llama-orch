# 42. Dual-Mode Testing Requirement

**Status**: ✅ **MANDATORY**  
**Created**: 2025-10-02  
**Updated**: 2025-10-02  

---

## 0. Executive Summary

**Requirement**: All tests (unit and BDD) MUST run in **both mock mode AND real CUDA mode** when available.

**Key Policy**:
- ✅ Tests MUST always run with mock VRAM first
- ✅ Tests MUST then attempt to run with real CUDA (if available)
- ⚠️ Tests MUST emit clear warning if no CUDA found
- ❌ Tests MUST NOT silently skip real CUDA testing

---

## 1. Testing Modes

### 1.1 Mock Mode (Always Required)

**Purpose**: Validate business logic without GPU dependency

**Characteristics**:
- Uses `MockVramAllocator` from `40_testing.md`
- No GPU required
- Fast execution
- CI/CD friendly
- Covers 95% of code (cryptography, validation, audit)

**Status**: ✅ ALWAYS RUNS

---

### 1.2 Real CUDA Mode (Conditional)

**Purpose**: Validate actual GPU operations and CUDA FFI

**Characteristics**:
- Uses real CUDA via `CudaContext`
- Requires NVIDIA GPU with CUDA support
- Validates actual memory operations
- Covers 100% of code including CUDA FFI layer

**Status**: ✅ RUNS IF GPU AVAILABLE, ⚠️ WARNING IF NOT

---

## 2. Test Execution Flow

### 2.1 Required Test Sequence

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RUN TESTS WITH MOCK VRAM                                │
│    ✅ Always executes                                       │
│    ✅ No GPU required                                       │
│    ✅ Validates business logic                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DETECT GPU AVAILABILITY                                  │
│    Use: gpu-info crate                                      │
│    Path: /home/vince/Projects/llama-orch/bin/shared-crates/│
│          gpu-info                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────┴───────┐
                    │               │
            GPU FOUND?          NO GPU FOUND?
                    │               │
                    ↓               ↓
    ┌───────────────────────┐   ┌─────────────────────────────┐
    │ 3a. RUN WITH REAL CUDA│   │ 3b. EMIT WARNING            │
    │    ✅ Test CUDA FFI   │   │    ⚠️ NO CUDA FOUND        │
    │    ✅ Test real VRAM  │   │    ⚠️ ONLY MOCK VRAM TESTED│
    │    ✅ 100% coverage   │   │    ⚠️ CUDA FFI NOT VERIFIED│
    └───────────────────────┘   └─────────────────────────────┘
```

---

## 3. Implementation Requirements

### 3.1 Test Structure (Unit Tests)

**REQUIRED**: Every test module MUST follow this pattern:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use gpu_info::GpuInfo;
    
    /// Helper to run test in both modes
    fn run_dual_mode_test<F>(test_fn: F) 
    where
        F: Fn(bool) -> Result<()>  // bool = is_real_cuda
    {
        // PHASE 1: Always run with mock
        println!("🧪 Running with MOCK VRAM...");
        test_fn(false).expect("Mock mode test failed");
        println!("✅ Mock mode: PASSED");
        
        // PHASE 2: Attempt real CUDA
        match GpuInfo::detect() {
            Ok(gpu_info) => {
                println!("🎮 GPU detected: {}", gpu_info.name);
                println!("🧪 Running with REAL CUDA...");
                test_fn(true).expect("Real CUDA test failed");
                println!("✅ Real CUDA mode: PASSED");
            }
            Err(_) => {
                eprintln!("");
                eprintln!("⚠️  ═══════════════════════════════════════════════");
                eprintln!("⚠️  WARNING: NO CUDA FOUND");
                eprintln!("⚠️  ONLY MOCK VRAM HAS BEEN TESTED!");
                eprintln!("⚠️  CUDA FFI layer NOT verified");
                eprintln!("⚠️  Install NVIDIA GPU + CUDA for full coverage");
                eprintln!("⚠️  ═══════════════════════════════════════════════");
                eprintln!("");
            }
        }
    }
    
    #[test]
    fn test_seal_model() {
        run_dual_mode_test(|is_real_cuda| {
            let manager = if is_real_cuda {
                VramManager::new_with_real_cuda()?
            } else {
                VramManager::new_with_mock()
            };
            
            let data = vec![0x42u8; 1024];
            let shard = manager.seal_model(&data, 0)?;
            
            assert!(manager.verify_sealed(&shard).is_ok());
            Ok(())
        });
    }
}
```

---

### 3.2 Test Structure (BDD Tests)

**REQUIRED**: BDD test harness MUST run scenarios twice:

```rust
// In bdd/src/main.rs or test runner

use cucumber::World;
use gpu_info::GpuInfo;

#[tokio::main]
async fn main() {
    // PHASE 1: Mock mode
    println!("🧪 Running BDD scenarios with MOCK VRAM...");
    std::env::set_var("VRAM_MODE", "mock");
    
    let mock_result = VramWorld::cucumber()
        .run("tests/features")
        .await;
    
    println!("✅ Mock mode: {} scenarios passed", mock_result.passed);
    
    // PHASE 2: Real CUDA mode
    match GpuInfo::detect() {
        Ok(gpu_info) => {
            println!("🎮 GPU detected: {}", gpu_info.name);
            println!("🧪 Running BDD scenarios with REAL CUDA...");
            std::env::set_var("VRAM_MODE", "cuda");
            
            let cuda_result = VramWorld::cucumber()
                .run("tests/features")
                .await;
            
            println!("✅ Real CUDA mode: {} scenarios passed", cuda_result.passed);
        }
        Err(_) => {
            eprintln!("");
            eprintln!("⚠️  ═══════════════════════════════════════════════");
            eprintln!("⚠️  WARNING: NO CUDA FOUND");
            eprintln!("⚠️  ONLY MOCK VRAM HAS BEEN TESTED!");
            eprintln!("⚠️  BDD scenarios NOT verified with real GPU");
            eprintln!("⚠️  Install NVIDIA GPU + CUDA for full coverage");
            eprintln!("⚠️  ═══════════════════════════════════════════════");
            eprintln!("");
        }
    }
}
```

---

### 3.3 Warning Message Requirements

**REQUIRED**: Warning message MUST include:

1. ⚠️ Clear visual separator (box or banner)
2. ⚠️ "NO CUDA FOUND" headline
3. ⚠️ "ONLY MOCK VRAM HAS BEEN TESTED" statement
4. ⚠️ What was NOT verified (CUDA FFI layer)
5. ⚠️ Actionable guidance (install NVIDIA GPU + CUDA)

**Example**:
```
⚠️  ═══════════════════════════════════════════════
⚠️  WARNING: NO CUDA FOUND
⚠️  ONLY MOCK VRAM HAS BEEN TESTED!
⚠️  CUDA FFI layer NOT verified
⚠️  Real VRAM operations NOT tested
⚠️  Install NVIDIA GPU + CUDA for full coverage
⚠️  ═══════════════════════════════════════════════
```

---

## 4. GPU Detection

### 4.1 Detection Method

**REQUIRED**: Use `gpu-info` crate for GPU detection

**Path**: `/home/vince/Projects/llama-orch/bin/shared-crates/gpu-info`

**API**:
```rust
use gpu_info::GpuInfo;

// Attempt detection (returns Result)
match GpuInfo::detect() {
    Ok(info) => {
        println!("GPU found: {}", info.name);
        println!("VRAM: {} GB", info.total_vram_gb);
        // Run real CUDA tests
    }
    Err(e) => {
        eprintln!("No GPU detected: {}", e);
        // Emit warning
    }
}
```

---

### 4.2 Detection Timing

**REQUIRED**: GPU detection MUST happen:
- ✅ After mock tests complete successfully
- ✅ Before attempting real CUDA tests
- ✅ Once per test run (not per test)

**FORBIDDEN**: GPU detection MUST NOT:
- ❌ Block mock tests from running
- ❌ Cause test failures if no GPU found
- ❌ Be cached across test runs

---

## 5. Test Coverage Requirements

### 5.1 Mock Mode Coverage

**MUST cover**:
- ✅ All cryptographic operations (HMAC, SHA-256, HKDF)
- ✅ All validation logic (shard_id, size, bounds)
- ✅ All audit logging
- ✅ All error handling
- ✅ All business logic

**Coverage target**: ≥ 95% of codebase

---

### 5.2 Real CUDA Mode Coverage

**MUST cover** (when GPU available):
- ✅ CUDA FFI calls (`cudaMalloc`, `cudaFree`, `cudaMemcpy`)
- ✅ Real VRAM allocation/deallocation
- ✅ Actual memory operations
- ✅ GPU-specific error handling
- ✅ Device property queries

**Coverage target**: 100% of codebase (including CUDA layer)

---

## 6. CI/CD Integration

### 6.1 CPU-Only Runners

**Behavior**:
- ✅ Mock tests run successfully
- ⚠️ Warning emitted (no CUDA found)
- ✅ Build passes (not a failure)

**Example output**:
```
Running 100 tests with MOCK VRAM...
test result: ok. 100 passed; 0 failed

⚠️  WARNING: NO CUDA FOUND
⚠️  ONLY MOCK VRAM HAS BEEN TESTED!

Build: SUCCESS (with warnings)
```

---

### 6.2 GPU-Enabled Runners

**Behavior**:
- ✅ Mock tests run successfully
- ✅ Real CUDA tests run successfully
- ✅ No warnings emitted
- ✅ Full coverage achieved

**Example output**:
```
Running 100 tests with MOCK VRAM...
test result: ok. 100 passed; 0 failed

🎮 GPU detected: NVIDIA RTX 4090
Running 100 tests with REAL CUDA...
test result: ok. 100 passed; 0 failed

Build: SUCCESS (full coverage)
```

---

## 7. Compliance Checklist

### 7.1 For Test Authors

Before merging tests:
- [ ] Test runs with mock VRAM first
- [ ] Test attempts real CUDA detection
- [ ] Test runs with real CUDA if available
- [ ] Warning emitted if no CUDA found
- [ ] Warning message follows required format
- [ ] Test doesn't fail if no GPU present

---

### 7.2 For Code Reviewers

Before approving PR:
- [ ] Dual-mode pattern implemented correctly
- [ ] `gpu-info` crate used for detection
- [ ] Warning message is clear and actionable
- [ ] Tests pass on CPU-only systems
- [ ] Tests pass on GPU systems (if available)

---

## 8. Examples

### 8.1 Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use gpu_info::GpuInfo;
    
    #[test]
    fn test_vram_allocation() {
        // PHASE 1: Mock mode
        println!("🧪 Testing with MOCK VRAM...");
        {
            let manager = VramManager::new_with_mock();
            let data = vec![0u8; 1024];
            let shard = manager.seal_model(&data, 0).unwrap();
            assert!(manager.verify_sealed(&shard).is_ok());
        }
        println!("✅ Mock mode: PASSED");
        
        // PHASE 2: Real CUDA mode
        match GpuInfo::detect() {
            Ok(info) => {
                println!("🎮 GPU detected: {}", info.name);
                println!("🧪 Testing with REAL CUDA...");
                let manager = VramManager::new_with_real_cuda().unwrap();
                let data = vec![0u8; 1024];
                let shard = manager.seal_model(&data, 0).unwrap();
                assert!(manager.verify_sealed(&shard).is_ok());
                println!("✅ Real CUDA mode: PASSED");
            }
            Err(_) => {
                eprintln!("\n⚠️  WARNING: NO CUDA FOUND");
                eprintln!("⚠️  ONLY MOCK VRAM HAS BEEN TESTED!\n");
            }
        }
    }
}
```

---

### 8.2 BDD Test Example

```gherkin
Feature: Dual-Mode VRAM Testing
  All scenarios MUST run in both mock and real CUDA modes

  Background:
    Given the test harness detects GPU availability
    And mock mode is always enabled
    And real CUDA mode is enabled if GPU found

  Scenario: Seal model in dual mode
    # Runs twice: once with mock, once with real CUDA (if available)
    Given a VramManager in current mode
    When I seal a 1MB model
    Then the seal should succeed
    And verification should pass
    
  Scenario: Warning emitted if no GPU
    Given no NVIDIA GPU is detected
    When all scenarios complete with mock mode
    Then a warning MUST be emitted
    And the warning MUST state "NO CUDA FOUND"
    And the warning MUST state "ONLY MOCK VRAM HAS BEEN TESTED"
```

---

## 9. Rationale

### 9.1 Why Dual-Mode Testing?

1. **Confidence**: Validates both business logic AND GPU operations
2. **Coverage**: Achieves 100% code coverage when GPU available
3. **Flexibility**: Tests work on CPU-only and GPU systems
4. **Visibility**: Clear warnings prevent false confidence
5. **CI/CD**: Works on standard runners without GPU

---

### 9.2 Why Mock First?

1. **Fast feedback**: Mock tests run quickly
2. **Isolation**: Validates logic without GPU dependency
3. **Debugging**: Easier to debug without GPU complexity
4. **Baseline**: Ensures core functionality works before GPU testing

---

### 9.3 Why Mandatory Warning?

1. **Transparency**: Developers know what was NOT tested
2. **Accountability**: Prevents shipping untested CUDA code
3. **Actionable**: Guides developers to install GPU for full testing
4. **Compliance**: Satisfies audit requirements for test coverage

---

## 10. Migration Guide

### 10.1 Updating Existing Tests

**Before** (single mode):
```rust
#[test]
fn test_seal() {
    let manager = VramManager::new();
    let shard = manager.seal_model(&[0u8; 1024], 0).unwrap();
    assert!(manager.verify_sealed(&shard).is_ok());
}
```

**After** (dual mode):
```rust
#[test]
fn test_seal() {
    run_dual_mode_test(|is_real_cuda| {
        let manager = if is_real_cuda {
            VramManager::new_with_real_cuda()?
        } else {
            VramManager::new_with_mock()
        };
        
        let shard = manager.seal_model(&[0u8; 1024], 0)?;
        assert!(manager.verify_sealed(&shard).is_ok());
        Ok(())
    });
}
```

---

## 11. Enforcement

### 11.1 Build-Time Checks

**REQUIRED**: Add to `build.rs` or test harness:

```rust
// Emit warning at build time if no GPU detected
fn main() {
    if gpu_info::GpuInfo::detect().is_err() {
        println!("cargo:warning=⚠️  NO CUDA FOUND - Tests will run in mock mode only");
    }
}
```

---

### 11.2 Runtime Checks

**REQUIRED**: Test runner MUST:
- ✅ Track whether real CUDA tests ran
- ✅ Emit summary at end of test run
- ✅ Include warning in test output

---

## 12. Summary

**Key Requirements**:
1. ✅ All tests MUST run with mock VRAM first
2. ✅ All tests MUST attempt real CUDA testing
3. ⚠️ Clear warning MUST be emitted if no CUDA found
4. ✅ Tests MUST NOT fail if no GPU present
5. ✅ Use `gpu-info` crate for detection

**Compliance**: This spec is **MANDATORY** for all vram-residency tests.

---

**Status**: ✅ ACTIVE  
**Enforcement**: REQUIRED  
**Review**: Quarterly
