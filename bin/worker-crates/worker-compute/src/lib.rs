//! Platform-agnostic compute trait for llama-orch workers
//!
//! Defines the `ComputeBackend` trait that abstracts platform-specific
//! compute operations (CUDA, Metal, etc.).
//!
//! # Architecture
//!
//! ```text
//! +-------------------------------------+
//! | ComputeBackend trait (this crate)   |
//! +------------------+------------------+
//!                    |
//!         +----------+----------+
//!         |                     |
//!    CudaBackend          MetalBackend
//!   (worker-orcd)        (worker-aarmd)
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

impl ComputeError {
    /// Check if error is retriable
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            ComputeError::InsufficientMemory { .. } | ComputeError::InferenceFailed(_)
        )
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            ComputeError::DeviceNotFound => "device",
            ComputeError::InsufficientMemory { .. } => "memory",
            ComputeError::ModelLoadFailed(_) => "model",
            ComputeError::InferenceFailed(_) => "inference",
            ComputeError::InvalidParameter(_) => "parameter",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock backend for testing trait behavior
    struct MockBackend;

    #[derive(Debug)]
    struct MockContext {
        device_id: i32,
    }

    #[derive(Debug)]
    struct MockModel {
        path: String,
        memory_usage: u64,
    }

    #[derive(Debug)]
    struct MockInferenceResult {
        tokens: Vec<String>,
        current: usize,
        max_tokens: usize,
    }

    impl ComputeBackend for MockBackend {
        type Context = MockContext;
        type Model = MockModel;
        type InferenceResult = MockInferenceResult;

        fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
            if device_id < 0 {
                return Err(ComputeError::DeviceNotFound);
            }
            Ok(MockContext { device_id })
        }

        fn load_model(_ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError> {
            if path.is_empty() {
                return Err(ComputeError::InvalidParameter("path cannot be empty".to_string()));
            }
            if path.contains("nonexistent") {
                return Err(ComputeError::ModelLoadFailed("file not found".to_string()));
            }
            Ok(MockModel {
                path: path.to_string(),
                memory_usage: 8_000_000_000,
            })
        }

        fn inference_start(
            _model: &Self::Model,
            prompt: &str,
            max_tokens: usize,
            temperature: f32,
            _seed: u64,
        ) -> Result<Self::InferenceResult, ComputeError> {
            if prompt.is_empty() {
                return Err(ComputeError::InvalidParameter("prompt cannot be empty".to_string()));
            }
            if temperature < 0.0 || temperature > 2.0 {
                return Err(ComputeError::InvalidParameter(
                    "temperature must be 0.0-2.0".to_string(),
                ));
            }
            if max_tokens == 0 {
                return Err(ComputeError::InvalidParameter("max_tokens must be > 0".to_string()));
            }

            let tokens = vec!["Hello".to_string(), " world".to_string(), "!".to_string()];
            Ok(MockInferenceResult {
                tokens,
                current: 0,
                max_tokens,
            })
        }

        fn inference_next_token(
            result: &mut Self::InferenceResult,
        ) -> Result<Option<String>, ComputeError> {
            if result.current >= result.max_tokens {
                return Ok(None);
            }
            if result.current >= result.tokens.len() {
                return Ok(None);
            }
            let token = result.tokens[result.current].clone();
            result.current += 1;
            Ok(Some(token))
        }

        fn get_memory_usage(model: &Self::Model) -> u64 {
            model.memory_usage
        }

        fn memory_architecture() -> &'static str {
            "mock"
        }
    }

    // ComputeError tests
    #[test]
    fn test_device_not_found_error() {
        let err = ComputeError::DeviceNotFound;
        assert_eq!(err.to_string(), "Device not found");
        assert!(!err.is_retriable());
        assert_eq!(err.category(), "device");
    }

    #[test]
    fn test_insufficient_memory_error() {
        let err = ComputeError::InsufficientMemory {
            required: 16_000_000_000,
            available: 8_000_000_000,
        };
        assert!(err.to_string().contains("16000000000"));
        assert!(err.to_string().contains("8000000000"));
        assert!(err.is_retriable());
        assert_eq!(err.category(), "memory");
    }

    #[test]
    fn test_model_load_failed_error() {
        let err = ComputeError::ModelLoadFailed("file not found".to_string());
        assert!(err.to_string().contains("Model load failed"));
        assert!(err.to_string().contains("file not found"));
        assert!(!err.is_retriable());
        assert_eq!(err.category(), "model");
    }

    #[test]
    fn test_inference_failed_error() {
        let err = ComputeError::InferenceFailed("CUDA OOM".to_string());
        assert!(err.to_string().contains("Inference failed"));
        assert!(err.to_string().contains("CUDA OOM"));
        assert!(err.is_retriable());
        assert_eq!(err.category(), "inference");
    }

    #[test]
    fn test_invalid_parameter_error() {
        let err = ComputeError::InvalidParameter("temperature out of range".to_string());
        assert!(err.to_string().contains("Invalid parameter"));
        assert!(err.to_string().contains("temperature out of range"));
        assert!(!err.is_retriable());
        assert_eq!(err.category(), "parameter");
    }

    #[test]
    fn test_error_retriability() {
        // Retriable errors
        assert!(ComputeError::InsufficientMemory {
            required: 1,
            available: 0
        }
        .is_retriable());
        assert!(ComputeError::InferenceFailed("test".to_string()).is_retriable());

        // Non-retriable errors
        assert!(!ComputeError::DeviceNotFound.is_retriable());
        assert!(!ComputeError::ModelLoadFailed("test".to_string()).is_retriable());
        assert!(!ComputeError::InvalidParameter("test".to_string()).is_retriable());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(ComputeError::DeviceNotFound.category(), "device");
        assert_eq!(
            ComputeError::InsufficientMemory {
                required: 1,
                available: 0
            }
            .category(),
            "memory"
        );
        assert_eq!(
            ComputeError::ModelLoadFailed("test".to_string()).category(),
            "model"
        );
        assert_eq!(
            ComputeError::InferenceFailed("test".to_string()).category(),
            "inference"
        );
        assert_eq!(
            ComputeError::InvalidParameter("test".to_string()).category(),
            "parameter"
        );
    }

    // Mock backend tests
    #[test]
    fn test_mock_backend_init_success() {
        let ctx = MockBackend::init(0).unwrap();
        assert_eq!(ctx.device_id, 0);

        let ctx = MockBackend::init(1).unwrap();
        assert_eq!(ctx.device_id, 1);
    }

    #[test]
    fn test_mock_backend_init_failure() {
        let result = MockBackend::init(-1);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ComputeError::DeviceNotFound));
    }

    #[test]
    fn test_mock_backend_load_model_success() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/llama-3.1-8b.gguf").unwrap();
        assert_eq!(model.path, "/models/llama-3.1-8b.gguf");
        assert_eq!(model.memory_usage, 8_000_000_000);
    }

    #[test]
    fn test_mock_backend_load_model_empty_path() {
        let ctx = MockBackend::init(0).unwrap();
        let result = MockBackend::load_model(&ctx, "");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComputeError::InvalidParameter(_)
        ));
    }

    #[test]
    fn test_mock_backend_load_model_nonexistent() {
        let ctx = MockBackend::init(0).unwrap();
        let result = MockBackend::load_model(&ctx, "/models/nonexistent.gguf");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComputeError::ModelLoadFailed(_)
        ));
    }

    #[test]
    fn test_mock_backend_inference_start_success() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();
        let result = MockBackend::inference_start(&model, "Hello", 100, 0.7, 42).unwrap();
        assert_eq!(result.current, 0);
        assert_eq!(result.max_tokens, 100);
    }

    #[test]
    fn test_mock_backend_inference_start_empty_prompt() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();
        let result = MockBackend::inference_start(&model, "", 100, 0.7, 42);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComputeError::InvalidParameter(_)
        ));
    }

    #[test]
    fn test_mock_backend_inference_start_invalid_temperature() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();

        let result = MockBackend::inference_start(&model, "Hello", 100, -0.1, 42);
        assert!(result.is_err());

        let result = MockBackend::inference_start(&model, "Hello", 100, 2.1, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_inference_start_zero_max_tokens() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();
        let result = MockBackend::inference_start(&model, "Hello", 0, 0.7, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_backend_inference_next_token() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();
        let mut result = MockBackend::inference_start(&model, "Hello", 100, 0.7, 42).unwrap();

        let token1 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token1, Some("Hello".to_string()));

        let token2 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token2, Some(" world".to_string()));

        let token3 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token3, Some("!".to_string()));

        let token4 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token4, None);
    }

    #[test]
    fn test_mock_backend_inference_max_tokens_limit() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();
        let mut result = MockBackend::inference_start(&model, "Hello", 2, 0.7, 42).unwrap();

        let token1 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token1, Some("Hello".to_string()));

        let token2 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token2, Some(" world".to_string()));

        // Should stop at max_tokens
        let token3 = MockBackend::inference_next_token(&mut result).unwrap();
        assert_eq!(token3, None);
    }

    #[test]
    fn test_mock_backend_get_memory_usage() {
        let ctx = MockBackend::init(0).unwrap();
        let model = MockBackend::load_model(&ctx, "/models/test.gguf").unwrap();
        let usage = MockBackend::get_memory_usage(&model);
        assert_eq!(usage, 8_000_000_000);
    }

    #[test]
    fn test_mock_backend_memory_architecture() {
        assert_eq!(MockBackend::memory_architecture(), "mock");
    }

    #[test]
    fn test_complete_inference_workflow() {
        // Initialize backend
        let ctx = MockBackend::init(0).unwrap();

        // Load model
        let model = MockBackend::load_model(&ctx, "/models/llama-3.1-8b.gguf").unwrap();

        // Check memory usage
        let memory = MockBackend::get_memory_usage(&model);
        assert_eq!(memory, 8_000_000_000);

        // Start inference
        let mut result = MockBackend::inference_start(&model, "Hello world", 100, 0.7, 42).unwrap();

        // Generate tokens
        let mut tokens = vec![];
        while let Some(token) = MockBackend::inference_next_token(&mut result).unwrap() {
            tokens.push(token);
        }

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "Hello");
        assert_eq!(tokens[1], " world");
        assert_eq!(tokens[2], "!");
    }
}

// ---
// Verified by Testing Team 🔍
