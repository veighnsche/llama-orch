//! Worker-orcd library
//!
//! This library exposes the core modules for integration testing.

pub mod cuda;
pub mod cuda_ffi;
pub mod error;
pub mod http;
pub mod inference;
pub mod inference_executor;
pub mod inference_result;
pub mod model;
pub mod sampling_config;
pub mod startup;
pub mod tests;

// Re-export from worker-gguf
pub use worker_gguf as gguf;

// Re-export commonly used types
pub use cuda::CudaError;
pub use error::WorkerError;
pub use inference_executor::InferenceExecutor;
pub use inference_result::InferenceResult;
pub use sampling_config::SamplingConfig;
// ---
// Built by Foundation-Alpha 🏗️
