//! Platform-agnostic compute trait for llama-orch workers
//!
//! Defines the `ComputeBackend` trait that abstracts platform-specific
//! compute operations (CUDA, Metal, etc.).
//!
//! # Architecture
//!
//! ```
//! ┌─────────────────────────────────────┐
//! │ ComputeBackend trait (this crate)   │
//! └──────────┬──────────────────────────┘
//!            │
//!     ┌──────┴──────┐
//!     ↓             ↓
//! CudaBackend   MetalBackend
//! (worker-orcd) (worker-aarmd)
//! ```
//!
//! # Example Implementation
//!
//! ```no_run
//! use worker_compute::{ComputeBackend, ComputeError};
//!
//! struct MyBackend;
//!
//! impl ComputeBackend for MyBackend {
//!     type Context = MyContext;
//!     type Model = MyModel;
//!     type InferenceResult = MyInferenceResult;
//!     
//!     fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
//!         // Platform-specific initialization
//!         todo!()
//!     }
//!     
//!     // ... implement other trait methods
//! }
//! ```

use thiserror::Error;

/// Compute backend errors
#[derive(Debug, Error)]
pub enum ComputeError {
    /// Device not found or unavailable
    #[error("Device not found")]
    DeviceNotFound,
    
    /// Insufficient memory
    #[error("Insufficient memory: required {required}, available {available}")]
    InsufficientMemory { required: u64, available: u64 },
    
    /// Model load failed
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    
    /// Inference failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Platform-agnostic compute backend trait
///
/// Implementations provide platform-specific compute operations:
/// - CUDA backend (NVIDIA GPUs)
/// - Metal backend (Apple Silicon)
/// - ROCm backend (AMD GPUs)
/// - etc.
pub trait ComputeBackend: Sized {
    /// Platform-specific context type (e.g., CudaContext, MetalContext)
    type Context;
    
    /// Platform-specific model type (e.g., CudaModel, MetalModel)
    type Model;
    
    /// Platform-specific inference result type
    type InferenceResult;
    
    /// Initialize compute context for specified device
    ///
    /// # Arguments
    /// - `device_id`: Device ID (0, 1, 2, ...)
    ///
    /// # Returns
    /// Initialized context or error if device unavailable
    fn init(device_id: i32) -> Result<Self::Context, ComputeError>;
    
    /// Load model from file path
    ///
    /// # Arguments
    /// - `ctx`: Compute context
    /// - `path`: Path to model file (GGUF format)
    ///
    /// # Returns
    /// Loaded model or error if load failed
    fn load_model(ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError>;
    
    /// Start inference with given prompt
    ///
    /// # Arguments
    /// - `model`: Loaded model
    /// - `prompt`: Input prompt text
    /// - `max_tokens`: Maximum tokens to generate
    /// - `temperature`: Sampling temperature (0.0-2.0)
    /// - `seed`: Random seed for reproducibility
    ///
    /// # Returns
    /// Inference result handle or error
    fn inference_start(
        model: &Self::Model,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Self::InferenceResult, ComputeError>;
    
    /// Generate next token in inference sequence
    ///
    /// # Arguments
    /// - `result`: Inference result handle
    ///
    /// # Returns
    /// - `Ok(Some(token))`: Next token generated
    /// - `Ok(None)`: Sequence complete
    /// - `Err(...)`: Inference error
    fn inference_next_token(
        result: &mut Self::InferenceResult,
    ) -> Result<Option<String>, ComputeError>;
    
    /// Get memory usage for model
    ///
    /// # Arguments
    /// - `model`: Loaded model
    ///
    /// # Returns
    /// Memory usage in bytes
    fn get_memory_usage(model: &Self::Model) -> u64;
    
    /// Get memory architecture type
    ///
    /// # Returns
    /// - `"vram-only"`: NVIDIA CUDA (separate VRAM)
    /// - `"unified"`: Apple Metal (unified memory)
    /// - etc.
    fn memory_architecture() -> &'static str;
}
