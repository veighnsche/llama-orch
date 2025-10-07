//! Worker-orcd library
//!
//! This library exposes the core modules for integration testing.

pub mod cuda;
pub mod cuda_ffi;
pub mod inference;
pub mod inference_executor;
pub mod model;
pub mod tests;

// [TEAM PICASSO 2025-10-07T16:13Z] Numeric parity logging moved to C++ side
// Rust-side logging removed - single source of truth in C++ where logits live

// Re-export commonly used types
pub use cuda::CudaError;
pub use inference_executor::InferenceExecutor;
// ---
// Built by Foundation-Alpha 🏗️
