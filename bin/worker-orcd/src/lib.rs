//! Worker-orcd library
//!
//! This library exposes the core modules for integration testing.

pub mod cuda;
pub mod error;
pub mod http;
pub mod inference_executor;
pub mod inference_result;
pub mod sampling_config;

// Re-export commonly used types
pub use cuda::CudaError;
pub use error::WorkerError;
pub use inference_executor::InferenceExecutor;
pub use inference_result::InferenceResult;
pub use sampling_config::SamplingConfig;

// ---
// Built by Foundation-Alpha 🏗️
