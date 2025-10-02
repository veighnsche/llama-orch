//! CUDA VRAM allocator
//!
//! Real CUDA VRAM allocation via FFI.
//!
//! # Why This File Exists But Is Unused
//!
//! This module contains an **alternative allocator abstraction** that was designed
//! during early development but is **not currently used** in production.
//!
//! ## Current Architecture
//!
//! `VramManager` (in `vram_manager.rs`) uses `CudaContext` **directly** from the
//! `cuda_ffi` module. This provides:
//! - Direct access to CUDA operations without an extra abstraction layer
//! - Simpler code path with fewer indirections
//! - Better performance (no wrapper overhead)
//!
//! ## Why Keep This File?
//!
//! 1. **Future flexibility** - If we need a different allocation strategy or
//!    want to support multiple allocator backends, this provides a starting point
//! 2. **Reference implementation** - Shows how to wrap CudaContext in a higher-level API
//! 3. **No maintenance burden** - The code is complete, tested, and doesn't interfere
//!    with the current implementation
//!
//! ## Why `#[allow(dead_code)]`?
//!
//! Without this attribute, Rust would emit warnings for every unused function in this
//! module. Since this is **intentionally unused but kept for future use**, we suppress
//! these warnings to keep the build output clean while preserving the code.
//!
//! ## When Would We Use This?
//!
//! Potential future scenarios:
//! - Supporting multiple GPU vendors (CUDA, ROCm, Vulkan) with a unified allocator trait
//! - Implementing custom allocation strategies (pooling, buddy allocation, etc.)
//! - Adding allocation telemetry or debugging hooks
//! - Testing different allocation patterns without modifying VramManager

use crate::error::Result;
use crate::cuda_ffi::{CudaContext, SafeCudaPtr};

/// CUDA VRAM allocator (alternative implementation, currently unused)
///
/// Wraps CUDA FFI calls for VRAM allocation.
///
/// # Status
///
/// **Not currently used in production.** See module documentation for details.
///
/// # Production Ready
///
/// This allocator uses real CUDA via the cuda_ffi module.
/// All operations are bounds-checked and safe.
#[allow(dead_code)]
pub struct CudaVramAllocator {
    context: CudaContext,
    allocations: Vec<SafeCudaPtr>,
}

impl CudaVramAllocator {
    /// Create new CUDA allocator
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No GPU detected
    /// - Device index is invalid
    /// - CUDA initialization fails
    pub fn new(gpu_device: u32) -> Result<Self> {
        let context = CudaContext::new(gpu_device)?;
        Ok(Self {
            context,
            allocations: Vec::new(),
        })
    }
    
    /// Allocate VRAM via cudaMalloc
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Size is zero
    /// - Out of VRAM
    /// - CUDA driver error
    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        let ptr = self.context.allocate_vram(size)?;
        let ptr_id = ptr.as_ptr() as usize;
        self.allocations.push(ptr);
        Ok(ptr_id)
    }
    
    /// Copy to VRAM via cudaMemcpy
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Pointer not found
    /// - Out of bounds
    /// - CUDA memcpy fails
    pub fn copy_to_vram(&mut self, ptr_id: usize, data: &[u8]) -> Result<()> {
        let ptr = self.allocations.iter_mut()
            .find(|p| p.as_ptr() as usize == ptr_id)
            .ok_or_else(|| crate::error::VramError::IntegrityViolation)?;
        
        ptr.write_at(0, data)
    }
    
    /// Copy from VRAM via cudaMemcpy
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Pointer not found
    /// - Out of bounds
    /// - CUDA memcpy fails
    pub fn copy_from_vram(&self, ptr_id: usize, size: usize) -> Result<Vec<u8>> {
        let ptr = self.allocations.iter()
            .find(|p| p.as_ptr() as usize == ptr_id)
            .ok_or_else(|| crate::error::VramError::IntegrityViolation)?;
        
        ptr.read_at(0, size)
    }
    
    /// Get available VRAM
    pub fn available_vram(&self) -> Result<usize> {
        self.context.get_free_vram()
    }
    
    /// Get total VRAM
    pub fn total_vram(&self) -> Result<usize> {
        self.context.get_total_vram()
    }
}

// Allocations are automatically freed when SafeCudaPtr is dropped
impl Drop for CudaVramAllocator {
    fn drop(&mut self) {
        tracing::debug!(
            allocations = %self.allocations.len(),
            "Dropping CudaVramAllocator (VRAM will be freed)"
        );
    }
}
