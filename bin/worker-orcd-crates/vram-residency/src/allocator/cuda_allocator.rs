//! CUDA VRAM allocator
//!
//! Real CUDA VRAM allocation via FFI.

use crate::error::Result;
use crate::cuda_ffi::{CudaContext, SafeCudaPtr};

/// CUDA VRAM allocator
///
/// Wraps CUDA FFI calls for VRAM allocation.
///
/// # Production Ready
///
/// This allocator uses real CUDA via the cuda_ffi module.
/// All operations are bounds-checked and safe.
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
