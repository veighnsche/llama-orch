# FT-012: FFI Integration Tests

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: M (2 days)  
**Days**: 24 - 25  
**Spec Ref**: M0-W-1810, M0-W-1811

---

## Story Description

Implement comprehensive integration tests for FFI boundary, validating Rust-to-C++-to-CUDA flow with real CUDA operations. This ensures the FFI contract works correctly end-to-end.

---

## Acceptance Criteria

- [ ] Integration test suite covers all FFI functions
- [ ] Tests validate context initialization and cleanup
- [ ] Tests validate error code propagation from C++ to Rust
- [ ] Tests validate VRAM allocation and tracking
- [ ] Tests validate pointer lifetime management (no leaks)
- [ ] Tests run with real CUDA (not mocked)
- [ ] Tests include negative cases (invalid params, OOM simulation)
- [ ] CI integration with CUDA feature flag
- [ ] Test output includes VRAM usage metrics

---

## Dependencies

### Upstream (Blocks This Story)
- FT-010: CUDA context initialization (Expected completion: Day 17)
- FT-011: VRAM tracking (Expected completion: Day 23)

### Downstream (This Story Blocks)
- FT-023: Integration test framework needs FFI tests as foundation
- FT-024: HTTP-FFI-CUDA integration builds on these tests

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/ffi_test.rs` - Rust integration tests
- `bin/worker-orcd/cuda/tests/integration/ffi_integration_test.cpp` - C++ integration tests
- `bin/worker-orcd/.github/workflows/cuda-tests.yml` - CI configuration

### Key Interfaces
```rust
// tests/integration/ffi_test.rs
#![cfg(feature = "cuda")]

use worker_orcd::cuda::{CudaContext, CudaError};

#[test]
fn test_context_init_valid_device() {
    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
    assert_eq!(ctx.device(), 0);
    
    // Context should be dropped here, triggering cleanup
}

#[test]
fn test_context_init_invalid_device() {
    let result = CudaContext::new(999);
    assert!(result.is_err());
    
    match result {
        Err(CudaError::InvalidDevice(_)) => {
            // Expected error
        }
        _ => panic!("Expected InvalidDevice error"),
    }
}

#[test]
fn test_device_count() {
    let count = CudaContext::device_count();
    assert!(count > 0, "No CUDA devices found");
}

#[test]
fn test_context_device_properties() {
    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
    
    let name = ctx.device_name();
    assert!(!name.is_empty(), "Device name should not be empty");
    
    let compute_cap = ctx.compute_capability();
    assert!(compute_cap >= 70, "Compute capability should be >= 7.0 (Volta)");
    
    let total_vram = ctx.total_vram();
    assert!(total_vram > 0, "Total VRAM should be positive");
    
    let free_vram = ctx.free_vram();
    assert!(free_vram <= total_vram, "Free VRAM should be <= total VRAM");
}

#[test]
fn test_vram_tracking() {
    let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
    
    let initial_usage = ctx.vram_tracker().total_usage();
    assert_eq!(initial_usage, 0, "Initial VRAM usage should be 0");
    
    // Allocate some VRAM (via model loading in future tests)
    // For now, just verify tracker is accessible
    let breakdown = ctx.vram_tracker().usage_breakdown();
    assert!(breakdown.is_empty(), "No allocations yet");
}

#[test]
fn test_error_code_conversion() {
    // Test that C++ error codes convert to Rust errors correctly
    let err = CudaError::from_code(1);
    assert!(matches!(err, CudaError::InvalidDevice(_)));
    
    let err = CudaError::from_code(2);
    assert!(matches!(err, CudaError::OutOfMemory(_)));
    
    let err = CudaError::from_code(99);
    assert!(matches!(err, CudaError::Unknown(_)));
}

#[test]
fn test_context_cleanup() {
    // Test that context cleanup doesn't leak VRAM
    let initial_free = {
        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
        ctx.free_vram()
    };
    
    // Context dropped here
    
    let ctx2 = CudaContext::new(0).expect("Failed to initialize CUDA context");
    let final_free = ctx2.free_vram();
    
    // Free VRAM should be approximately the same (within 1MB tolerance)
    let diff = if final_free > initial_free {
        final_free - initial_free
    } else {
        initial_free - final_free
    };
    
    assert!(diff < 1024 * 1024, "VRAM leak detected: {} bytes", diff);
}

#[test]
fn test_multiple_contexts_sequential() {
    // Test creating multiple contexts sequentially
    for i in 0..3 {
        let ctx = CudaContext::new(0).expect(&format!("Failed to create context {}", i));
        assert_eq!(ctx.device(), 0);
        // Context dropped at end of iteration
    }
}

#[test]
#[should_panic(expected = "Invalid device")]
fn test_context_negative_device() {
    let _ = CudaContext::new(-1).expect("Should fail with negative device");
}

// C++ integration tests
// cuda/tests/integration/ffi_integration_test.cpp
#include <gtest/gtest.h>
#include "context.h"
#include "vram_tracker.h"
#include "cuda_error.h"

using namespace worker;

class FFIIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure at least one CUDA device available
        ASSERT_GT(Context::device_count(), 0) << "No CUDA devices found";
    }
};

TEST_F(FFIIntegrationTest, ContextInitialization) {
    Context ctx(0);
    EXPECT_EQ(ctx.device(), 0);
    EXPECT_GT(ctx.total_vram(), 0);
}

TEST_F(FFIIntegrationTest, ContextInvalidDevice) {
    EXPECT_THROW({
        Context ctx(999);
    }, CudaError);
}

TEST_F(FFIIntegrationTest, VramTrackerIntegration) {
    Context ctx(0);
    auto& tracker = ctx.vram_tracker();
    
    EXPECT_EQ(tracker.total_usage(), 0);
    EXPECT_EQ(tracker.allocation_count(), 0);
    
    // Simulate allocation
    void* test_ptr = reinterpret_cast<void*>(0x1000);
    tracker.record_allocation(test_ptr, 1024, VramPurpose::ModelWeights, "test");
    
    EXPECT_EQ(tracker.total_usage(), 1024);
    EXPECT_EQ(tracker.allocation_count(), 1);
    
    tracker.record_deallocation(test_ptr);
    EXPECT_EQ(tracker.total_usage(), 0);
}

TEST_F(FFIIntegrationTest, ErrorCodeConversion) {
    try {
        throw CudaError::invalid_device("test device");
    } catch (const CudaError& e) {
        EXPECT_EQ(e.code(), CUDA_ERROR_INVALID_DEVICE);
        EXPECT_STREQ(e.what(), "Invalid device: test device");
    }
}

TEST_F(FFIIntegrationTest, ContextCleanup) {
    size_t initial_free;
    {
        Context ctx(0);
        initial_free = ctx.free_vram();
    }
    // Context destroyed here
    
    Context ctx2(0);
    size_t final_free = ctx2.free_vram();
    
    // Free VRAM should be approximately the same
    size_t diff = (final_free > initial_free) ? 
                  (final_free - initial_free) : 
                  (initial_free - final_free);
    
    EXPECT_LT(diff, 1024 * 1024) << "VRAM leak detected: " << diff << " bytes";
}

TEST_F(FFIIntegrationTest, DeviceProperties) {
    Context ctx(0);
    
    EXPECT_NE(ctx.device_name(), nullptr);
    EXPECT_GT(strlen(ctx.device_name()), 0);
    
    EXPECT_GE(ctx.compute_capability(), 70) << "Compute capability should be >= 7.0";
    
    EXPECT_GT(ctx.total_vram(), 0);
    EXPECT_LE(ctx.free_vram(), ctx.total_vram());
}
```

### Implementation Notes
- Tests run with `--features cuda` flag
- CI skips CUDA tests if no GPU available
- Use `#[cfg(feature = "cuda")]` to conditionally compile tests
- C++ tests use Google Test framework
- Rust tests use standard `#[test]` attribute
- Memory leak detection via VRAM comparison
- Error propagation tested with negative cases
- Tests are deterministic (no flaky tests)

---

## Testing Strategy

### Unit Tests
- None (this story IS the test suite)

### Integration Tests
- Test context initialization (valid device)
- Test context initialization (invalid device)
- Test device count query
- Test device properties query
- Test VRAM tracking integration
- Test error code conversion
- Test context cleanup (no leaks)
- Test multiple contexts sequentially
- Test negative device ID
- Test error message retrieval

### Manual Verification
1. Run Rust tests: `cargo test --features cuda --test ffi_test`
2. Run C++ tests: `./build/tests/ffi_integration_test`
3. Verify all tests pass
4. Check VRAM usage with `nvidia-smi` during test run

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Rust integration tests passing (9+ tests)
- [ ] C++ integration tests passing (6+ tests)
- [ ] CI configuration updated for CUDA tests
- [ ] Documentation updated (test suite docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §12.2 Unit Tests (M0-W-1810, M0-W-1811)
- Related Stories: FT-010 (context init), FT-011 (VRAM tracking)
- Google Test: https://github.com/google/googletest

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Test suite started**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_start",
       target: "ffi-integration".to_string(),
       human: "Starting FFI integration test suite".to_string(),
       ..Default::default()
   });
   ```

2. **Memory leak detected**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_complete",
       target: "ffi-integration".to_string(),
       error_kind: Some("memory_leak".to_string()),
       human: format!("Memory leak detected: {} bytes", diff),
       ..Default::default()
   });
   ```

3. **Test suite completed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_complete",
       target: "ffi-integration".to_string(),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("FFI integration tests passed ({} ms)", elapsed.as_millis()),
       ..Default::default()
   });
   ```

**Why this matters**: Integration tests validate the FFI boundary. Narration creates an audit trail of test runs and helps diagnose FFI-related issues.

---
*Narration guidance added by Narration-Core Team 🎀*
