//! Safe wrapper for CUDA context
//!
//! This module provides a safe Rust wrapper around the C CUDA context API.
//! The `Context` type implements RAII pattern - resources are automatically
//! freed when the context is dropped.

use super::error::CudaError;
use super::ffi;
use super::model::Model;
// use std::ffi::CString;  // Unused in stub mode

/// Safe wrapper for CUDA context
///
/// Represents an initialized CUDA device context. The context owns
/// the CUDA device and all resources allocated on it.
///
/// # Thread Safety
///
/// `Context` is `Send` but not `Sync`. Each context is single-threaded
/// and must not be accessed concurrently. However, contexts can be moved
/// between threads.
///
/// # Example
///
/// ```no_run
/// use worker_orcd::cuda::Context;
///
/// let ctx = Context::new(0)?; // GPU device 0
/// let device_count = Context::device_count();
/// println!("Found {} CUDA devices", device_count);
/// # Ok::<(), worker_orcd::cuda::CudaError>(())
/// ```
#[derive(Debug)]
pub struct Context {
    ptr: *mut ffi::CudaContext,
}

impl Context {
    /// Initialize CUDA context for specified GPU device
    ///
    /// # Arguments
    ///
    /// * `gpu_device` - GPU device ID (0, 1, 2, ...)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Device ID is invalid
    /// - CUDA initialization fails
    /// - No CUDA devices found
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn new(gpu_device: i32) -> Result<Self, CudaError> {
        let mut error_code = 0;

        // SAFETY: error_code is valid pointer to writable i32
        let ptr = unsafe { ffi::cuda_init(gpu_device, &mut error_code) };

        if ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }

        Ok(Self { ptr })
    }

    /// Get number of available CUDA devices
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let count = Context::device_count();
    /// println!("Found {} CUDA devices", count);
    /// ```
    pub fn device_count() -> i32 {
        // SAFETY: No pointer parameters, safe to call
        unsafe { ffi::cuda_get_device_count() }
    }

    /// Load model from GGUF file to VRAM
    ///
    /// # Arguments
    ///
    /// * `model_path` - Absolute path to .gguf file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model file not found
    /// - Invalid GGUF format
    /// - Insufficient VRAM
    /// - Model path contains null byte
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let model = ctx.load_model("/path/to/model.gguf")?;
    /// println!("Model loaded: {} bytes VRAM", model.vram_bytes());
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn load_model(&self, model_path: &str) -> Result<Model, CudaError> {
        Model::load(self, model_path)
    }

    /// Get process-wide VRAM usage
    ///
    /// Returns total VRAM bytes allocated by this process across all models.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let vram_used = ctx.process_vram_usage();
    /// println!("Process VRAM usage: {} bytes", vram_used);
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn process_vram_usage(&self) -> u64 {
        // SAFETY: ptr is valid (checked in constructor)
        unsafe { ffi::cuda_get_process_vram_usage(self.ptr) }
    }

    /// Check CUDA device health
    ///
    /// Returns `Ok(true)` if device is healthy, `Ok(false)` if device has errors.
    ///
    /// # Errors
    ///
    /// Returns error if health check fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let healthy = ctx.check_device_health()?;
    /// if !healthy {
    ///     eprintln!("Device is unhealthy!");
    /// }
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn check_device_health(&self) -> Result<bool, CudaError> {
        let mut error_code = 0;

        // SAFETY: ptr is valid (checked in constructor), error_code is valid pointer
        let healthy = unsafe { ffi::cuda_check_device_health(self.ptr, &mut error_code) };

        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }

        Ok(healthy)
    }

    /// Get raw pointer (for internal use only)
    #[doc(hidden)]
    pub(crate) fn as_ptr(&self) -> *mut ffi::CudaContext {
        self.ptr
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe { ffi::cuda_destroy(self.ptr) };
        }
    }
}

// SAFETY: Context can be moved between threads (CUDA context is per-process)
unsafe impl Send for Context {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_count() {
        let count = Context::device_count();
        // Should return 0 when built without CUDA feature
        #[cfg(not(feature = "cuda"))]
        assert_eq!(count, 0);

        // Should return >= 0 when built with CUDA feature
        #[cfg(feature = "cuda")]
        assert!(count >= 0);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_context_new_without_cuda() {
        let result = Context::new(0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CudaError::DeviceNotFound(_)));
    }

    #[test]
    fn test_context_new_invalid_device() {
        let result = Context::new(-1);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_context_new_valid_device() {
        let count = Context::device_count();
        if count > 0 {
            let result = Context::new(0);
            assert!(result.is_ok());
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_process_vram_usage() {
        let count = Context::device_count();
        if count > 0 {
            let ctx = Context::new(0).unwrap();
            let vram = ctx.process_vram_usage();
            // VRAM usage should be non-negative
            assert!(vram >= 0);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_check_device_health() {
        let count = Context::device_count();
        if count > 0 {
            let ctx = Context::new(0).unwrap();
            let result = ctx.check_device_health();
            // Should not error
            assert!(result.is_ok());
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
