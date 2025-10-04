//! Test fixtures for integration tests
//!
//! Provides mock models and test configurations.
//!
//! # Spec References
//! - M0-W-1820: Integration test framework

use std::path::PathBuf;

/// Test model configuration
#[derive(Debug, Clone)]
pub struct TestModel {
    pub name: String,
    pub path: PathBuf,
    pub num_layers: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
}

impl TestModel {
    /// Qwen2.5-0.5B test model
    pub fn qwen2_5_0_5b() -> Self {
        Self {
            name: "Qwen2.5-0.5B".to_string(),
            path: PathBuf::from("models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
            num_layers: 24,
            num_kv_heads: 2,
            head_dim: 64,
            vocab_size: 151936,
        }
    }
    
    /// Mock model for fast tests (no real file needed)
    pub fn mock() -> Self {
        Self {
            name: "Mock".to_string(),
            path: PathBuf::from("/dev/null"),
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 32,
            vocab_size: 1000,
        }
    }
    
    /// Check if model file exists
    pub fn exists(&self) -> bool {
        self.path.exists()
    }
}

/// Test configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub gpu_device: i32,
    pub timeout_secs: u64,
    pub max_tokens: u32,
}

impl TestConfig {
    /// Default test configuration
    pub fn default() -> Self {
        Self {
            gpu_device: 0,
            timeout_secs: 30,
            max_tokens: 10,
        }
    }
    
    /// Fast test configuration (small max_tokens)
    pub fn fast() -> Self {
        Self {
            gpu_device: 0,
            timeout_secs: 10,
            max_tokens: 5,
        }
    }
    
    /// Long test configuration (large max_tokens)
    pub fn long() -> Self {
        Self {
            gpu_device: 0,
            timeout_secs: 60,
            max_tokens: 100,
        }
    }
}

/// Test prompt templates
pub struct TestPrompts;

impl TestPrompts {
    /// Simple prompt for basic tests
    pub fn simple() -> &'static str {
        "Hello"
    }
    
    /// Short prompt that should generate quickly
    pub fn short() -> &'static str {
        "Write a haiku"
    }
    
    /// Longer prompt for prefill testing
    pub fn long() -> &'static str {
        "Write a detailed explanation of how transformers work in machine learning"
    }
    
    /// Prompt for JSON generation (with stop sequences)
    pub fn json() -> &'static str {
        "Generate a JSON object with name and age fields"
    }
    
    /// Prompt for testing stop sequences
    pub fn with_stop() -> &'static str {
        "Count from 1 to 10"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_model() {
        let model = TestModel::qwen2_5_0_5b();
        assert_eq!(model.name, "Qwen2.5-0.5B");
        assert_eq!(model.num_layers, 24);
        assert_eq!(model.num_kv_heads, 2);
    }

    #[test]
    fn test_mock_model() {
        let model = TestModel::mock();
        assert_eq!(model.name, "Mock");
        assert_eq!(model.num_layers, 2);
    }

    #[test]
    fn test_default_config() {
        let config = TestConfig::default();
        assert_eq!(config.gpu_device, 0);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_tokens, 10);
    }

    #[test]
    fn test_fast_config() {
        let config = TestConfig::fast();
        assert_eq!(config.max_tokens, 5);
        assert_eq!(config.timeout_secs, 10);
    }

    #[test]
    fn test_prompts() {
        assert!(!TestPrompts::simple().is_empty());
        assert!(!TestPrompts::short().is_empty());
        assert!(!TestPrompts::long().is_empty());
        assert!(!TestPrompts::json().is_empty());
    }
}

// ---
// Built by Foundation-Alpha 🏗️
