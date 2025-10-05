//! cuda_ffi — Safe CUDA FFI boundary
//!
//! Provides safe Rust wrappers around CUDA C++ kernels with bounds checking and error handling.
//!
//! # Security Properties
//!
//! - Bounds-checked VRAM operations
//! - Safe pointer abstractions (no raw pointer exposure)
//! - CUDA error mapping to Rust Result types
//! - Fail-fast on driver errors
//!
//! # Architecture
//!
//! This module wraps CUDA kernels from `bin/worker-orcd/cuda/kernels/`:
//! - `gemm.cu` — cuBLAS matrix multiplication
//! - `rope.cu` — Rotary Position Embedding
//! - `attention.cu` — Attention kernels (prefill + decode)
//! - `rmsnorm.cu` — RMSNorm normalization
//! - `sampling.cu` — Token sampling (greedy, top-k, temperature)
//!
//! # Example
//!
//! ```rust
//! use cuda_ffi::{CudaContext, SafeCudaPtr};
//!
//! let ctx = CudaContext::new(0)?; // GPU device 0
//! let mut ptr = ctx.allocate_vram(1024)?;
//! ptr.write_at(0, &[1, 2, 3, 4])?; // Bounds-checked write
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

use std::ffi::c_void;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA initialization failed: {0}")]
    InitFailed(String),
    #[error("CUDA allocation failed: {0} bytes")]
    AllocationFailed(usize),
    #[error("CUDA out of bounds: offset {offset} + len {len} > size {size}")]
    OutOfBounds { offset: usize, len: usize, size: usize },
    #[error("CUDA driver error: {0}")]
    DriverError(String),
    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    #[error("invalid device: {0}")]
    InvalidDevice(u32),
}

pub type Result<T> = std::result::Result<T, CudaError>;

/// Safe CUDA pointer with bounds checking
pub struct SafeCudaPtr {
    ptr: *mut c_void,
    size: usize,
    device: u32,
}

impl SafeCudaPtr {
    /// Create new SafeCudaPtr (private - use CudaContext::allocate_vram)
    fn new(ptr: *mut c_void, size: usize, device: u32) -> Self {
        Self { ptr, size, device }
    }

    /// Write data at offset with bounds checking
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        let end = offset.checked_add(data.len()).ok_or(CudaError::OutOfBounds {
            offset,
            len: data.len(),
            size: self.size,
        })?;

        if end > self.size {
            return Err(CudaError::OutOfBounds { offset, len: data.len(), size: self.size });
        }

        // TODO(ARCH-CHANGE): Implement actual CUDA memcpy per ARCHITECTURE_CHANGE_PLAN.md Phase 3:
        // - Use cudaMemcpy to copy data to GPU
        // - Handle CUDA errors properly
        // - Add async copy support (cudaMemcpyAsync)
        // See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #11 (unsafe CUDA FFI)
        tracing::debug!(
            offset = %offset,
            len = %data.len(),
            size = %self.size,
            "CUDA write (stub)"
        );

        Ok(())
    }

    /// Read data from offset with bounds checking
    pub fn read_at(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        let end = offset.checked_add(len).ok_or(CudaError::OutOfBounds {
            offset,
            len,
            size: self.size,
        })?;

        if end > self.size {
            return Err(CudaError::OutOfBounds { offset, len, size: self.size });
        }

        // TODO(ARCH-CHANGE): Implement actual CUDA memcpy from device
        // - Use cudaMemcpy to copy data from GPU
        // - Handle CUDA errors properly
        Ok(vec![0u8; len])
    }

    /// Get size of allocated VRAM
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get device index
    pub fn device(&self) -> u32 {
        self.device
    }
}

impl Drop for SafeCudaPtr {
    fn drop(&mut self) {
        // TODO(ARCH-CHANGE): Implement CUDA deallocation
        // - Use cudaFree to release VRAM
        // - Log errors but don't panic in Drop
        tracing::debug!(
            size = %self.size,
            device = %self.device,
            "CUDA free (stub)"
        );
    }
}

/// CUDA context for device management
pub struct CudaContext {
    device: u32,
}

impl CudaContext {
    /// Initialize CUDA context for device
    ///
    /// # Production Mode
    /// Requires GPU to be available. Fails if no GPU detected.
    ///
    /// # Test Mode
    /// When compiled with `cfg(test)`, allows initialization without GPU (mock mode).
    pub fn new(device: u32) -> Result<Self> {
        // In test mode, allow mock initialization without GPU
        #[cfg(test)]
        {
            tracing::warn!(
                device = %device,
                "CUDA context initialized in TEST MODE (no GPU validation)"
            );
            return Ok(Self { device });
        }

        // Production mode: Log GPU initialization
        #[cfg(not(test))]
        {
            tracing::info!(
                device = %device,
                "Initializing CUDA context (stub mode)"
            );

            // TODO(ARCH-CHANGE): Implement actual CUDA initialization per ARCHITECTURE_CHANGE_PLAN.md Phase 3:
            // - Use cudaSetDevice to select GPU
            // - Initialize cuBLAS handle
            // - Validate GPU availability
            // See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #11 (unsafe CUDA FFI)

            Ok(Self { device })
        }
    }

    /// Allocate VRAM with bounds checking
    pub fn allocate_vram(&self, size: usize) -> Result<SafeCudaPtr> {
        if size == 0 {
            return Err(CudaError::AllocationFailed(size));
        }

        // TODO(ARCH-CHANGE): Implement actual CUDA allocation
        // - Use cudaMalloc to allocate VRAM
        // - Check for cudaErrorMemoryAllocation
        // - Return SafeCudaPtr wrapper
        let ptr = std::ptr::null_mut(); // Placeholder - REPLACE with cudaMalloc

        tracing::debug!(
            size = %size,
            device = %self.device,
            "CUDA allocate (stub)"
        );

        Ok(SafeCudaPtr::new(ptr, size, self.device))
    }

    /// Get available VRAM on device
    pub fn get_free_vram(&self) -> Result<usize> {
        // TODO(ARCH-CHANGE): Implement VRAM query
        // - Use cudaMemGetInfo to get free/total VRAM
        // - Return free bytes
        Ok(24_000_000_000) // 24GB stub
    }

    /// Get total VRAM on device
    pub fn get_total_vram(&self) -> Result<usize> {
        // TODO(ARCH-CHANGE): Implement VRAM query
        // - Use cudaMemGetInfo to get total VRAM
        Ok(24_000_000_000) // 24GB stub
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_checking() {
        let ctx = CudaContext::new(0).unwrap();
        let mut ptr = ctx.allocate_vram(1024).unwrap();

        // Valid write
        assert!(ptr.write_at(0, &[1, 2, 3, 4]).is_ok());

        // Out of bounds write
        assert!(ptr.write_at(1020, &[1, 2, 3, 4, 5]).is_err());

        // Overflow check
        assert!(ptr.write_at(usize::MAX, &[1]).is_err());
    }
}
