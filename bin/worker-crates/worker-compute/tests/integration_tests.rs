//! Integration tests for worker-compute
//!
//! Tests trait implementations and cross-module behavior.

use worker_compute::{ComputeBackend, ComputeError};

// Mock backend for integration testing
struct TestBackend;

#[derive(Debug)]
struct TestContext {
    device_id: i32,
    initialized: bool,
}

#[derive(Debug)]
struct TestModel {
    path: String,
    memory_usage: u64,
    loaded: bool,
}

#[derive(Debug)]
struct TestInferenceResult {
    prompt: String,
    tokens_generated: Vec<String>,
    current_index: usize,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
}

impl ComputeBackend for TestBackend {
    type Context = TestContext;
    type Model = TestModel;
    type InferenceResult = TestInferenceResult;

    fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
        if device_id < 0 || device_id > 7 {
            return Err(ComputeError::DeviceNotFound);
        }
        Ok(TestContext { device_id, initialized: true })
    }

    fn load_model(ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError> {
        if !ctx.initialized {
            return Err(ComputeError::InvalidParameter("context not initialized".to_string()));
        }
        if path.is_empty() {
            return Err(ComputeError::InvalidParameter("empty path".to_string()));
        }
        if !path.ends_with(".gguf") {
            return Err(ComputeError::ModelLoadFailed("invalid format".to_string()));
        }

        let memory_usage = if path.contains("8b") {
            8_000_000_000
        } else if path.contains("70b") {
            70_000_000_000
        } else {
            16_000_000_000
        };

        Ok(TestModel { path: path.to_string(), memory_usage, loaded: true })
    }

    fn inference_start(
        model: &Self::Model,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Self::InferenceResult, ComputeError> {
        if !model.loaded {
            return Err(ComputeError::InferenceFailed("model not loaded".to_string()));
        }
        if prompt.is_empty() {
            return Err(ComputeError::InvalidParameter("empty prompt".to_string()));
        }
        if max_tokens == 0 {
            return Err(ComputeError::InvalidParameter("max_tokens must be > 0".to_string()));
        }
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ComputeError::InvalidParameter("temperature out of range".to_string()));
        }

        let tokens =
            vec!["The".to_string(), " answer".to_string(), " is".to_string(), " 42".to_string()];

        Ok(TestInferenceResult {
            prompt: prompt.to_string(),
            tokens_generated: tokens,
            current_index: 0,
            max_tokens,
            temperature,
            seed,
        })
    }

    fn inference_next_token(
        result: &mut Self::InferenceResult,
    ) -> Result<Option<String>, ComputeError> {
        if result.current_index >= result.max_tokens {
            return Ok(None);
        }
        if result.current_index >= result.tokens_generated.len() {
            return Ok(None);
        }

        let token = result.tokens_generated[result.current_index].clone();
        result.current_index += 1;
        Ok(Some(token))
    }

    fn get_memory_usage(model: &Self::Model) -> u64 {
        model.memory_usage
    }

    fn memory_architecture() -> &'static str {
        "test-unified"
    }
}

#[test]
fn test_multi_device_initialization() {
    // Test multiple devices
    for device_id in 0..4 {
        let ctx = TestBackend::init(device_id).unwrap();
        assert_eq!(ctx.device_id, device_id);
        assert!(ctx.initialized);
    }
}

#[test]
fn test_device_bounds() {
    // Valid devices
    assert!(TestBackend::init(0).is_ok());
    assert!(TestBackend::init(7).is_ok());

    // Invalid devices
    assert!(TestBackend::init(-1).is_err());
    assert!(TestBackend::init(8).is_err());
}

#[test]
fn test_model_size_detection() {
    let ctx = TestBackend::init(0).unwrap();

    let model_8b = TestBackend::load_model(&ctx, "/models/llama-3.1-8b.gguf").unwrap();
    assert_eq!(TestBackend::get_memory_usage(&model_8b), 8_000_000_000);

    let model_70b = TestBackend::load_model(&ctx, "/models/llama-3.1-70b.gguf").unwrap();
    assert_eq!(TestBackend::get_memory_usage(&model_70b), 70_000_000_000);

    let model_default = TestBackend::load_model(&ctx, "/models/custom.gguf").unwrap();
    assert_eq!(TestBackend::get_memory_usage(&model_default), 16_000_000_000);
}

#[test]
fn test_model_format_validation() {
    let ctx = TestBackend::init(0).unwrap();

    // Valid format
    assert!(TestBackend::load_model(&ctx, "/models/test.gguf").is_ok());

    // Invalid formats
    let result = TestBackend::load_model(&ctx, "/models/test.bin");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ComputeError::ModelLoadFailed(_)));
}

#[test]
fn test_inference_parameter_validation() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();

    // Valid parameters
    assert!(TestBackend::inference_start(&model, "Hello", 100, 0.7, 42).is_ok());

    // Invalid temperature
    assert!(TestBackend::inference_start(&model, "Hello", 100, -0.1, 42).is_err());
    assert!(TestBackend::inference_start(&model, "Hello", 100, 2.1, 42).is_err());

    // Invalid max_tokens
    assert!(TestBackend::inference_start(&model, "Hello", 0, 0.7, 42).is_err());

    // Invalid prompt
    assert!(TestBackend::inference_start(&model, "", 100, 0.7, 42).is_err());
}

#[test]
fn test_inference_token_generation() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();
    let mut result =
        TestBackend::inference_start(&model, "What is the answer?", 100, 0.7, 42).unwrap();

    let mut tokens = vec![];
    while let Some(token) = TestBackend::inference_next_token(&mut result).unwrap() {
        tokens.push(token);
    }

    assert_eq!(tokens.len(), 4);
    assert_eq!(tokens[0], "The");
    assert_eq!(tokens[1], " answer");
    assert_eq!(tokens[2], " is");
    assert_eq!(tokens[3], " 42");
}

#[test]
fn test_inference_max_tokens_enforcement() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();
    let mut result = TestBackend::inference_start(&model, "Test", 2, 0.7, 42).unwrap();

    let token1 = TestBackend::inference_next_token(&mut result).unwrap();
    assert!(token1.is_some());

    let token2 = TestBackend::inference_next_token(&mut result).unwrap();
    assert!(token2.is_some());

    // Should stop at max_tokens
    let token3 = TestBackend::inference_next_token(&mut result).unwrap();
    assert!(token3.is_none());
}

#[test]
fn test_memory_architecture_reporting() {
    assert_eq!(TestBackend::memory_architecture(), "test-unified");
}

#[test]
fn test_complete_workflow_with_different_models() {
    let ctx = TestBackend::init(0).unwrap();

    // Test with 8B model
    let model_8b = TestBackend::load_model(&ctx, "/models/llama-3.1-8b.gguf").unwrap();
    assert_eq!(TestBackend::get_memory_usage(&model_8b), 8_000_000_000);

    let mut result = TestBackend::inference_start(&model_8b, "Hello", 10, 0.7, 42).unwrap();
    let mut tokens = vec![];
    while let Some(token) = TestBackend::inference_next_token(&mut result).unwrap() {
        tokens.push(token);
    }
    assert_eq!(tokens.len(), 4);

    // Test with 70B model
    let model_70b = TestBackend::load_model(&ctx, "/models/llama-3.1-70b.gguf").unwrap();
    assert_eq!(TestBackend::get_memory_usage(&model_70b), 70_000_000_000);

    let mut result = TestBackend::inference_start(&model_70b, "World", 10, 0.5, 123).unwrap();
    let mut tokens = vec![];
    while let Some(token) = TestBackend::inference_next_token(&mut result).unwrap() {
        tokens.push(token);
    }
    assert_eq!(tokens.len(), 4);
}

#[test]
fn test_error_propagation() {
    // Device error
    let device_err = TestBackend::init(-1);
    assert!(device_err.is_err());
    assert!(matches!(device_err.unwrap_err(), ComputeError::DeviceNotFound));

    // Model load error
    let ctx = TestBackend::init(0).unwrap();
    let model_err = TestBackend::load_model(&ctx, "/models/test.bin");
    assert!(model_err.is_err());
    assert!(matches!(model_err.unwrap_err(), ComputeError::ModelLoadFailed(_)));

    // Inference parameter error
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();
    let inference_err = TestBackend::inference_start(&model, "", 100, 0.7, 42);
    assert!(inference_err.is_err());
    assert!(matches!(inference_err.unwrap_err(), ComputeError::InvalidParameter(_)));
}

#[test]
fn test_temperature_range_boundaries() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();

    // Boundary values
    assert!(TestBackend::inference_start(&model, "Test", 10, 0.0, 42).is_ok());
    assert!(TestBackend::inference_start(&model, "Test", 10, 2.0, 42).is_ok());

    // Just outside boundaries
    assert!(TestBackend::inference_start(&model, "Test", 10, -0.001, 42).is_err());
    assert!(TestBackend::inference_start(&model, "Test", 10, 2.001, 42).is_err());
}

#[test]
fn test_seed_reproducibility_tracking() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();

    let result1 = TestBackend::inference_start(&model, "Test", 10, 0.7, 42).unwrap();
    assert_eq!(result1.seed, 42);

    let result2 = TestBackend::inference_start(&model, "Test", 10, 0.7, 999).unwrap();
    assert_eq!(result2.seed, 999);
}

#[test]
fn test_prompt_preservation() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();

    let prompt = "What is the meaning of life?";
    let result = TestBackend::inference_start(&model, prompt, 10, 0.7, 42).unwrap();
    assert_eq!(result.prompt, prompt);
}

#[test]
fn test_large_max_tokens() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();
    let mut result = TestBackend::inference_start(&model, "Test", 10000, 0.7, 42).unwrap();

    // Should still only generate available tokens
    let mut count = 0;
    while TestBackend::inference_next_token(&mut result).unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 4); // Only 4 tokens available
}

#[test]
fn test_multiple_models_same_context() {
    let ctx = TestBackend::init(0).unwrap();

    let model1 = TestBackend::load_model(&ctx, "/models/model1.gguf").unwrap();
    let model2 = TestBackend::load_model(&ctx, "/models/model2.gguf").unwrap();

    assert_ne!(model1.path, model2.path);
    assert_eq!(TestBackend::get_memory_usage(&model1), 16_000_000_000);
    assert_eq!(TestBackend::get_memory_usage(&model2), 16_000_000_000);
}

#[test]
fn test_inference_state_isolation() {
    let ctx = TestBackend::init(0).unwrap();
    let model = TestBackend::load_model(&ctx, "/models/test.gguf").unwrap();

    let mut result1 = TestBackend::inference_start(&model, "First", 10, 0.7, 42).unwrap();
    let result2 = TestBackend::inference_start(&model, "Second", 10, 0.5, 123).unwrap();

    // Advance result1
    TestBackend::inference_next_token(&mut result1).unwrap();
    TestBackend::inference_next_token(&mut result1).unwrap();

    // result2 should be independent
    assert_eq!(result2.current_index, 0);
    assert_eq!(result2.prompt, "Second");
    assert_eq!(result2.seed, 123);
}

// ---
// Verified by Testing Team 🔍
