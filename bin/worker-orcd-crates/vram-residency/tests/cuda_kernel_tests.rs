//! Unit tests for CUDA kernels
//!
//! These tests validate the CUDA C++ kernels via FFI.
//! Tests run on real GPU if available, skip otherwise.

use vram_residency::cuda_ffi::{CudaContext, SafeCudaPtr};

/// Helper to check if GPU is available
fn has_gpu() -> bool {
    gpu_info::detect_gpus().available
}

/// Skip test if no GPU available
macro_rules! require_gpu {
    () => {
        if !has_gpu() {
            eprintln!("⏭️  Skipping GPU test (no GPU detected)");
            return;
        }
    };
}

// ============================================================================
// Context Initialization Tests
// ============================================================================

#[test]
fn test_context_creation_with_gpu() {
    require_gpu!();
    
    let ctx = CudaContext::new(0);
    assert!(ctx.is_ok(), "Should create context with GPU");
}

#[test]
fn test_context_creation_invalid_device() {
    require_gpu!();
    
    let ctx = CudaContext::new(999);
    assert!(ctx.is_err(), "Should fail for invalid device");
}

// ============================================================================
// Allocation Tests
// ============================================================================

#[test]
fn test_allocate_valid_size() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let ptr = ctx.allocate_vram(1024);
    assert!(ptr.is_ok(), "Should allocate 1KB");
    assert_eq!(ptr.unwrap().size(), 1024);
}

#[test]
fn test_allocate_zero_size() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let ptr = ctx.allocate_vram(0);
    assert!(ptr.is_err(), "Should reject zero size");
}

#[test]
fn test_allocate_huge_size() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let size = 101 * 1024 * 1024 * 1024; // 101GB (exceeds MAX_ALLOCATION_SIZE)
    let ptr = ctx.allocate_vram(size);
    assert!(ptr.is_err(), "Should reject size > 100GB");
}

#[test]
fn test_allocate_large_valid_size() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let size = 1 * 1024 * 1024 * 1024; // 1GB
    let ptr = ctx.allocate_vram(size);
    
    // May fail if not enough VRAM, but should not panic
    if let Ok(p) = ptr {
        assert_eq!(p.size(), size);
    }
}

#[test]
fn test_allocate_multiple() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    
    let ptr1 = ctx.allocate_vram(1024).unwrap();
    let ptr2 = ctx.allocate_vram(2048).unwrap();
    let ptr3 = ctx.allocate_vram(4096).unwrap();
    
    assert_eq!(ptr1.size(), 1024);
    assert_eq!(ptr2.size(), 2048);
    assert_eq!(ptr3.size(), 4096);
    
    // Pointers should be different
    assert_ne!(ptr1.as_ptr() as usize, ptr2.as_ptr() as usize);
    assert_ne!(ptr2.as_ptr() as usize, ptr3.as_ptr() as usize);
}

// ============================================================================
// Memory Copy Tests
// ============================================================================

#[test]
fn test_write_and_read() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Write data
    let data = vec![1u8, 2, 3, 4, 5];
    let result = ptr.write_at(0, &data);
    assert!(result.is_ok(), "Should write data");
    
    // Read back
    let read_data = ptr.read_at(0, 5).unwrap();
    assert_eq!(read_data, data, "Read data should match written data");
}

#[test]
fn test_write_at_offset() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Write at offset
    let data = vec![0xAA, 0xBB, 0xCC, 0xDD];
    let result = ptr.write_at(100, &data);
    assert!(result.is_ok(), "Should write at offset");
    
    // Read back from offset
    let read_data = ptr.read_at(100, 4).unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_write_out_of_bounds() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Try to write beyond allocation
    let data = vec![1u8; 100];
    let result = ptr.write_at(1000, &data);
    assert!(result.is_err(), "Should reject out-of-bounds write");
}

#[test]
fn test_read_out_of_bounds() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let ptr = ctx.allocate_vram(1024).unwrap();
    
    // Try to read beyond allocation
    let result = ptr.read_at(1000, 100);
    assert!(result.is_err(), "Should reject out-of-bounds read");
}

#[test]
fn test_write_overflow() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Try to trigger overflow
    let result = ptr.write_at(usize::MAX, &[1]);
    assert!(result.is_err(), "Should detect overflow");
}

#[test]
fn test_read_overflow() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let ptr = ctx.allocate_vram(1024).unwrap();
    
    // Try to trigger overflow
    let result = ptr.read_at(usize::MAX, 1);
    assert!(result.is_err(), "Should detect overflow");
}

#[test]
fn test_write_zero_bytes() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Write zero bytes (should be no-op)
    let result = ptr.write_at(0, &[]);
    assert!(result.is_ok(), "Zero-byte write should succeed");
}

#[test]
fn test_read_zero_bytes() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let ptr = ctx.allocate_vram(1024).unwrap();
    
    // Read zero bytes (should be no-op)
    let result = ptr.read_at(0, 0);
    assert!(result.is_ok(), "Zero-byte read should succeed");
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_large_copy() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(10 * 1024 * 1024).unwrap(); // 10MB
    
    // Write 1MB of data
    let data = vec![0x42u8; 1024 * 1024];
    let result = ptr.write_at(0, &data);
    assert!(result.is_ok(), "Should write 1MB");
    
    // Read back
    let read_data = ptr.read_at(0, data.len()).unwrap();
    assert_eq!(read_data.len(), data.len());
    assert_eq!(read_data[0], 0x42);
    assert_eq!(read_data[data.len() - 1], 0x42);
}

// ============================================================================
// VRAM Info Tests
// ============================================================================

#[test]
fn test_get_free_vram() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let free = ctx.get_free_vram();
    assert!(free.is_ok(), "Should query free VRAM");
    assert!(free.unwrap() > 0, "Free VRAM should be > 0");
}

#[test]
fn test_get_total_vram() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let total = ctx.get_total_vram();
    assert!(total.is_ok(), "Should query total VRAM");
    assert!(total.unwrap() > 0, "Total VRAM should be > 0");
}

#[test]
fn test_vram_info_consistency() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let free = ctx.get_free_vram().unwrap();
    let total = ctx.get_total_vram().unwrap();
    
    assert!(free <= total, "Free VRAM should not exceed total");
    assert!(total > 1024 * 1024 * 1024, "Total VRAM should be > 1GB");
}

// ============================================================================
// Drop/Cleanup Tests
// ============================================================================

#[test]
fn test_drop_frees_memory() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let free_before = ctx.get_free_vram().unwrap();
    
    {
        let _ptr = ctx.allocate_vram(100 * 1024 * 1024).unwrap(); // 100MB
        let free_during = ctx.get_free_vram().unwrap();
        assert!(free_during < free_before, "VRAM should be allocated");
    }
    
    // After drop, VRAM should be freed
    let free_after = ctx.get_free_vram().unwrap();
    
    // Allow some tolerance (CUDA may not immediately report freed memory)
    let tolerance = 10 * 1024 * 1024; // 10MB
    assert!(
        free_after >= free_before - tolerance,
        "VRAM should be freed after drop"
    );
}

// ============================================================================
// Alignment Tests
// ============================================================================

#[test]
fn test_allocation_alignment() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let ptr = ctx.allocate_vram(1024).unwrap();
    
    // CUDA should return 256-byte aligned pointers
    let addr = ptr.as_ptr() as usize;
    assert_eq!(addr % 256, 0, "Pointer should be 256-byte aligned");
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_many_small_allocations() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptrs = Vec::new();
    
    // Allocate 100 small buffers
    for _ in 0..100 {
        let ptr = ctx.allocate_vram(1024);
        if let Ok(p) = ptr {
            ptrs.push(p);
        } else {
            break; // Out of VRAM
        }
    }
    
    assert!(ptrs.len() >= 10, "Should allocate at least 10 buffers");
}

#[test]
fn test_write_read_pattern() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Write pattern
    for i in 0..10 {
        let data = vec![i as u8; 10];
        ptr.write_at(i * 10, &data).unwrap();
    }
    
    // Read and verify pattern
    for i in 0..10 {
        let data = ptr.read_at(i * 10, 10).unwrap();
        assert_eq!(data, vec![i as u8; 10]);
    }
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

#[test]
fn test_error_recovery_after_failed_allocation() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    
    // Try to allocate too much
    let _ = ctx.allocate_vram(1000 * 1024 * 1024 * 1024); // 1TB (will fail)
    
    // Should still be able to allocate normally
    let ptr = ctx.allocate_vram(1024);
    assert!(ptr.is_ok(), "Should recover after failed allocation");
}

#[test]
fn test_error_recovery_after_invalid_operation() {
    require_gpu!();
    
    let ctx = CudaContext::new(0).unwrap();
    let mut ptr = ctx.allocate_vram(1024).unwrap();
    
    // Try invalid write
    let _ = ptr.write_at(usize::MAX, &[1]);
    
    // Should still be able to write normally
    let result = ptr.write_at(0, &[1, 2, 3]);
    assert!(result.is_ok(), "Should recover after invalid operation");
}
