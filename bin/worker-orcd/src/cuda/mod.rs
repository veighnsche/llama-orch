//! CUDA FFI bindings for worker-orcd
//!
//! This module provides safe Rust wrappers around the C API
//! exposed by the CUDA C++ implementation.
//!
//! When built WITHOUT the `cuda` feature, this module provides
//! stub implementations for development on CUDA-less devices.
//!
//! # Architecture
//!
//! The module is organized into:
//! - `ffi` - Raw unsafe FFI declarations
//! - `error` - Error types and conversion
//! - `context` - Safe CUDA context wrapper
//! - `model` - Safe model wrapper
//! - `inference` - Safe inference session wrapper
//!
//! # Example
//!
//! ```no_run
//! use worker_orcd::cuda::Context;
//!
//! let ctx = Context::new(0)?;
//! let model = ctx.load_model("/path/to/model.gguf")?;
//! let mut inference = model.start_inference("Write a haiku", 100, 0.7, 42)?;
//!
//! while let Some((token, idx)) = inference.next_token()? {
//!     print!("{}", token);
//! }
//! # Ok::<(), worker_orcd::cuda::CudaError>(())
//! ```

pub mod context;
mod error;
mod ffi;  // Keep private for now
mod inference;
mod model;

pub use context::Context;
pub use error::{CudaError, CudaErrorCode};
pub use inference::Inference;
pub use model::Model;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are exported
        let _ = std::any::type_name::<Context>();
        let _ = std::any::type_name::<Model>();
        let _ = std::any::type_name::<Inference>();
        let _ = std::any::type_name::<CudaError>();
        let _ = std::any::type_name::<CudaErrorCode>();
    }

    #[test]
    fn test_device_count() {
        let count = Context::device_count();
        // Should return >= 0
        assert!(count >= 0);
    }
}
