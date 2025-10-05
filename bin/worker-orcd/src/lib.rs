//! Worker-orcd library
//!
//! This library exposes the core modules for integration testing.

pub mod cuda;
pub mod cuda_ffi;
pub mod http;
pub mod inference;
pub mod inference_executor;
pub mod model;
pub mod tests;

// Re-export from worker-gguf
pub use worker_gguf as gguf;

// Re-export commonly used types
pub use cuda::CudaError;
pub use inference_executor::InferenceExecutor;
// ---
// Built by Foundation-Alpha 🏗️
