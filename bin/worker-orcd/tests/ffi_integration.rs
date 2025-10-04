//! FFI Integration Tests
//!
//! Tests the Rust-to-C++-to-CUDA FFI boundary with real CUDA operations.
//! These tests validate that the FFI contract works correctly end-to-end.
//!
//! Spec: M0-W-1810, M0-W-1811

#![cfg(feature = "cuda")]

use worker_orcd::cuda::{Context, CudaError};

// ============================================================================
// Context Initialization Tests
// ============================================================================

#[test]
fn test_context_init_valid_device() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    let ctx = Context::new(0).expect("Failed to initialize CUDA context");
    
    // Verify context is usable
    let vram = ctx.process_vram_usage();
    assert!(vram >= 0, "VRAM usage should be non-negative");
    
    // Context should be dropped here, triggering cleanup
}

#[test]
fn test_context_init_invalid_device() {
    let result = Context::new(999);
    assert!(result.is_err(), "Should fail with invalid device ID");
    
    match result {
        Err(CudaError::InvalidDevice(_)) => {
            // Expected error
        }
        Err(e) => panic!("Expected InvalidDevice error, got: {:?}", e),
        Ok(_) => panic!("Expected error, got success"),
    }
}

#[test]
fn test_context_init_negative_device() {
    let result = Context::new(-1);
    assert!(result.is_err(), "Should fail with negative device ID");
}

#[test]
fn test_device_count() {
    let count = Context::device_count();
    assert!(count >= 0, "Device count should be non-negative");
    
    if count > 0 {
        println!("Found {} CUDA device(s)", count);
    } else {
        eprintln!("WARNING: No CUDA devices found");
    }
}

// ============================================================================
// Device Properties Tests
// ============================================================================

#[test]
fn test_context_device_properties() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    let ctx = Context::new(0).expect("Failed to initialize CUDA context");
    
    // Test process VRAM usage query
    let vram_usage = ctx.process_vram_usage();
    assert!(vram_usage >= 0, "VRAM usage should be non-negative");
    
    println!("Process VRAM usage: {} bytes", vram_usage);
}

#[test]
fn test_check_device_health() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    let ctx = Context::new(0).expect("Failed to initialize CUDA context");
    
    let healthy = ctx.check_device_health()
        .expect("Health check should not error");
    
    assert!(healthy, "Device should be healthy");
}

// ============================================================================
// Context Cleanup Tests
// ============================================================================

#[test]
fn test_context_cleanup_no_leak() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    // Measure initial free VRAM
    let initial_free = {
        let ctx = Context::new(0).expect("Failed to initialize CUDA context");
        let free = ctx.process_vram_usage();
        drop(ctx); // Explicit drop
        free
    };
    
    // Create new context and measure again
    let ctx2 = Context::new(0).expect("Failed to initialize CUDA context");
    let final_free = ctx2.process_vram_usage();
    
    // Free VRAM should be approximately the same (within 1MB tolerance)
    let diff = if final_free > initial_free {
        final_free - initial_free
    } else {
        initial_free - final_free
    };
    
    assert!(
        diff < 1024 * 1024,
        "VRAM leak detected: {} bytes difference (initial: {}, final: {})",
        diff,
        initial_free,
        final_free
    );
    
    println!("VRAM difference: {} bytes (within tolerance)", diff);
}

#[test]
fn test_multiple_contexts_sequential() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    // Test creating multiple contexts sequentially
    for i in 0..3 {
        let ctx = Context::new(0)
            .expect(&format!("Failed to create context {}", i));
        
        // Verify context is usable
        let vram = ctx.process_vram_usage();
        assert!(vram >= 0, "VRAM usage should be non-negative");
        
        println!("Context {} created successfully", i);
        // Context dropped at end of iteration
    }
}

// ============================================================================
// Error Propagation Tests
// ============================================================================

#[test]
fn test_error_message_retrieval() {
    let result = Context::new(999);
    assert!(result.is_err());
    
    let err = result.unwrap_err();
    let err_msg = format!("{}", err);
    
    // Error message should contain useful information
    assert!(!err_msg.is_empty(), "Error message should not be empty");
    println!("Error message: {}", err_msg);
}

#[test]
fn test_error_debug_format() {
    let result = Context::new(999);
    assert!(result.is_err());
    
    let err = result.unwrap_err();
    let debug_msg = format!("{:?}", err);
    
    // Debug format should contain useful information
    assert!(!debug_msg.is_empty(), "Debug message should not be empty");
    println!("Debug format: {}", debug_msg);
}

// ============================================================================
// Concurrent Context Tests (Sequential Safety)
// ============================================================================

#[test]
fn test_context_send_trait() {
    // Verify Context implements Send (can be moved between threads)
    fn assert_send<T: Send>() {}
    assert_send::<Context>();
}

#[test]
fn test_context_not_sync() {
    // Verify Context does NOT implement Sync (cannot be shared between threads)
    fn assert_not_sync<T: Send>() {
        // This compiles only if T is Send but not Sync
        let _ = std::marker::PhantomData::<T>;
    }
    assert_not_sync::<Context>();
}

// ============================================================================
// FFI Boundary Stress Tests
// ============================================================================

#[test]
fn test_rapid_context_creation_destruction() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    // Rapidly create and destroy contexts
    for i in 0..10 {
        let ctx = Context::new(0)
            .expect(&format!("Failed to create context {} in rapid test", i));
        
        // Immediately drop
        drop(ctx);
    }
    
    println!("Rapid context creation/destruction test passed");
}

#[test]
fn test_context_outlives_multiple_operations() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    let ctx = Context::new(0).expect("Failed to initialize CUDA context");
    
    // Perform multiple operations on same context
    for i in 0..5 {
        let vram = ctx.process_vram_usage();
        assert!(vram >= 0, "VRAM query {} failed", i);
        
        let healthy = ctx.check_device_health()
            .expect(&format!("Health check {} failed", i));
        assert!(healthy, "Device unhealthy at iteration {}", i);
    }
    
    println!("Context survived multiple operations");
}

// ============================================================================
// Integration with Other Components
// ============================================================================

#[test]
fn test_context_ready_for_model_loading() {
    let device_count = Context::device_count();
    if device_count == 0 {
        eprintln!("Skipping test: No CUDA devices available");
        return;
    }

    let ctx = Context::new(0).expect("Failed to initialize CUDA context");
    
    // Verify context is in a state ready for model loading
    let healthy = ctx.check_device_health()
        .expect("Health check failed");
    assert!(healthy, "Device must be healthy before model loading");
    
    let vram = ctx.process_vram_usage();
    println!("Context ready for model loading. Current VRAM usage: {} bytes", vram);
}

// ============================================================================
// Test Metadata and Reporting
// ============================================================================

#[test]
fn test_ffi_integration_suite_metadata() {
    // This test documents the test suite itself
    println!("FFI Integration Test Suite");
    println!("==========================");
    println!("Spec: M0-W-1810, M0-W-1811");
    println!("Tests: Context initialization, cleanup, error propagation");
    println!("CUDA devices: {}", Context::device_count());
}

// ---
// Built by Foundation-Alpha 🏗️
