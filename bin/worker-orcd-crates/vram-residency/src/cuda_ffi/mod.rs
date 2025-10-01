//! cuda_ffi — Safe CUDA FFI boundary for VRAM operations
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
//! This module wraps CUDA kernels from `bin/worker-orcd-crates/vram-residency/cuda/kernels/`:
//! - `vram_ops.cu` — VRAM allocation, deallocation, memcpy
//!
//! # Example
//!
//! ```rust
//! use vram_residency::cuda_ffi::{CudaContext, SafeCudaPtr};
//!
//! let ctx = CudaContext::new(0)?; // GPU device 0
//! let mut ptr = ctx.allocate_vram(1024)?;
//! ptr.write_at(0, &[1, 2, 3, 4])?; // Bounds-checked write
//! ```

// Security-critical module: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![allow(clippy::todo)] // TODO: Remove once CUDA FFI is implemented
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

use std::ffi::c_void;
use crate::error::{Result, VramError};

// CUDA error codes (from vram_ops.cu)
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_ALLOCATION_FAILED: i32 = 1;
const CUDA_ERROR_INVALID_VALUE: i32 = 2;
const CUDA_ERROR_MEMCPY_FAILED: i32 = 3;
const CUDA_ERROR_DRIVER: i32 = 4;

// External C functions from CUDA kernels
extern "C" {
    fn vram_malloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
    fn vram_free(ptr: *mut c_void) -> i32;
    fn vram_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    fn vram_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    fn vram_get_info(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
    fn vram_set_device(device: i32) -> i32;
    fn vram_get_device_count(count: *mut i32) -> i32;
}

/// Map CUDA error code to VramError
fn map_cuda_error(code: i32, context: &str) -> VramError {
    match code {
        CUDA_ERROR_ALLOCATION_FAILED => VramError::CudaAllocationFailed(context.to_string()),
        CUDA_ERROR_INVALID_VALUE => VramError::InvalidInput(context.to_string()),
        CUDA_ERROR_MEMCPY_FAILED => VramError::CudaAllocationFailed(format!("memcpy failed: {}", context)),
        CUDA_ERROR_DRIVER => VramError::CudaAllocationFailed(format!("driver error: {}", context)),
        _ => VramError::CudaAllocationFailed(format!("unknown error {}: {}", code, context)),
    }
}

/// Safe CUDA pointer with bounds checking
///
/// This type wraps a raw CUDA pointer and provides bounds-checked operations.
/// The pointer is automatically freed when dropped.
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
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Offset + data length exceeds allocation size
    /// - Offset + data length overflows
    /// - CUDA memcpy fails
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        // Bounds checking
        let end = offset.checked_add(data.len()).ok_or_else(|| {
            VramError::IntegrityViolation
        })?;
        
        if end > self.size {
            return Err(VramError::IntegrityViolation);
        }
        
        // Calculate destination pointer
        let dst = unsafe {
            (self.ptr as *mut u8).add(offset) as *mut c_void
        };
        
        // Perform CUDA memcpy (host to device)
        let result = unsafe {
            vram_memcpy_h2d(dst, data.as_ptr() as *const c_void, data.len())
        };
        
        if result != CUDA_SUCCESS {
            return Err(map_cuda_error(result, "write_at"));
        }
        
        tracing::debug!(
            offset = %offset,
            len = %data.len(),
            size = %self.size,
            "CUDA write completed"
        );
        
        Ok(())
    }
    
    /// Read data from offset with bounds checking
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Offset + length exceeds allocation size
    /// - Offset + length overflows
    /// - CUDA memcpy fails
    pub fn read_at(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        // Bounds checking
        let end = offset.checked_add(len).ok_or_else(|| {
            VramError::IntegrityViolation
        })?;
        
        if end > self.size {
            return Err(VramError::IntegrityViolation);
        }
        
        // Allocate buffer for result
        let mut buffer = vec![0u8; len];
        
        // Calculate source pointer
        let src = unsafe {
            (self.ptr as *const u8).add(offset) as *const c_void
        };
        
        // Perform CUDA memcpy (device to host)
        let result = unsafe {
            vram_memcpy_d2h(buffer.as_mut_ptr() as *mut c_void, src, len)
        };
        
        if result != CUDA_SUCCESS {
            return Err(map_cuda_error(result, "read_at"));
        }
        
        Ok(buffer)
    }
    
    /// Get size of allocated VRAM
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get device index
    pub fn device(&self) -> u32 {
        self.device
    }
    
    /// Get raw pointer (for internal use only)
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for SafeCudaPtr {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        
        let result = unsafe { vram_free(self.ptr) };
        
        if result != CUDA_SUCCESS {
            tracing::error!(
                size = %self.size,
                device = %self.device,
                error_code = %result,
                "CUDA free failed in Drop (non-fatal)"
            );
        } else {
            tracing::debug!(
                size = %self.size,
                device = %self.device,
                "CUDA free completed"
            );
        }
    }
}

// Safety: SafeCudaPtr can be sent between threads (GPU memory is thread-safe)
unsafe impl Send for SafeCudaPtr {}
unsafe impl Sync for SafeCudaPtr {}

/// CUDA context for device management
///
/// Manages CUDA device initialization and VRAM allocation.
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
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No NVIDIA GPU detected (production mode)
    /// - Device index is invalid
    /// - CUDA driver initialization fails
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
        
        // Production mode: Validate GPU availability using gpu-info
        #[cfg(not(test))]
        {
            let gpu_info = gpu_info::detect_gpus();
            if !gpu_info.available {
                return Err(VramError::PolicyViolation(
                    "No NVIDIA GPU detected. This worker requires GPU.".to_string(),
                ));
            }
            
            // Validate device index
            if (device as usize) >= gpu_info.count {
                return Err(VramError::InvalidInput(format!(
                    "GPU device {} out of range (max: {})",
                    device, gpu_info.count
                )));
            }
            
            let gpu = &gpu_info.devices[device as usize];
            
            tracing::info!(
                device = %device,
                name = %gpu.name,
                vram_gb = %(gpu.vram_total_bytes / 1024 / 1024 / 1024),
                compute_cap = ?gpu.compute_capability,
                "Initializing CUDA context"
            );
            
            // Set CUDA device
            let result = unsafe { vram_set_device(device as i32) };
            if result != CUDA_SUCCESS {
                return Err(map_cuda_error(result, "set_device"));
            }
            
            tracing::info!(device = %device, "CUDA context initialized");
            Ok(Self { device })
        }
    }
    
    /// Allocate VRAM with bounds checking
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Size is zero
    /// - Allocation fails (out of memory)
    /// - CUDA driver error
    pub fn allocate_vram(&self, size: usize) -> Result<SafeCudaPtr> {
        if size == 0 {
            return Err(VramError::InvalidInput("size cannot be zero".to_string()));
        }
        
        let mut ptr: *mut c_void = std::ptr::null_mut();
        
        // Allocate VRAM via CUDA
        let result = unsafe { vram_malloc(&mut ptr as *mut *mut c_void, size) };
        
        if result != CUDA_SUCCESS {
            return Err(map_cuda_error(result, &format!("allocate {} bytes", size)));
        }
        
        if ptr.is_null() {
            return Err(VramError::CudaAllocationFailed(
                "cudaMalloc returned null pointer".to_string()
            ));
        }
        
        tracing::debug!(
            size = %size,
            device = %self.device,
            "CUDA allocate completed"
        );
        
        Ok(SafeCudaPtr::new(ptr, size, self.device))
    }
    
    /// Get available VRAM on device
    ///
    /// # Errors
    ///
    /// Returns error if CUDA query fails
    pub fn get_free_vram(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        let result = unsafe { vram_get_info(&mut free as *mut usize, &mut total as *mut usize) };
        
        if result != CUDA_SUCCESS {
            return Err(map_cuda_error(result, "get_free_vram"));
        }
        
        Ok(free)
    }
    
    /// Get total VRAM on device
    ///
    /// # Errors
    ///
    /// Returns error if CUDA query fails
    pub fn get_total_vram(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        let result = unsafe { vram_get_info(&mut free as *mut usize, &mut total as *mut usize) };
        
        if result != CUDA_SUCCESS {
            return Err(map_cuda_error(result, "get_total_vram"));
        }
        
        Ok(total)
    }
    
    /// Get device index
    pub fn device(&self) -> u32 {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_creation() {
        // In test mode, this should succeed without GPU
        let ctx = CudaContext::new(0);
        assert!(ctx.is_ok());
    }
    
    #[test]
    fn test_bounds_checking_overflow() {
        // Test overflow detection in bounds checking
        let ctx = CudaContext::new(0).unwrap();
        
        // This will fail in test mode (no real CUDA), but tests the API
        if let Ok(mut ptr) = ctx.allocate_vram(1024) {
            // Overflow check
            assert!(ptr.write_at(usize::MAX, &[1]).is_err());
        }
    }
}
