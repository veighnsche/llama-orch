//! Adapter Factory Pattern
//!
//! Factory for creating the correct adapter based on model architecture.
//! Supports automatic architecture detection from GGUF metadata.
//!
//! # Example
//!
//! ```no_run
//! use worker_orcd::models::factory::AdapterFactory;
//!
//! // Create adapter from GGUF file (auto-detect architecture)
//! let adapter = AdapterFactory::from_gguf("model.gguf")?;
//!
//! // Or specify architecture explicitly
//! let adapter = AdapterFactory::from_gguf_with_arch("model.gguf", "llama")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Spec: FT-034

use super::{
    AdapterForwardConfig, LlamaModelAdapter, ModelType,
    gpt::{GPTConfig, GPTWeightLoader},
    phi3::{Phi3Config, Phi3WeightLoader},
    qwen::{QwenConfig, QwenWeightLoader},
};
use crate::gguf::GGUFMetadata;
use thiserror::Error;

/// Factory errors
#[derive(Debug, Error)]
pub enum FactoryError {
    /// Unknown architecture
    #[error("Unknown architecture: {0}")]
    UnknownArchitecture(String),

    /// Model loading failed
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),

    /// GGUF parsing failed
    #[error("GGUF parsing failed: {0}")]
    GGUFParsingFailed(String),

    /// Unsupported model variant
    #[error("Unsupported model variant: {0}")]
    UnsupportedVariant(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Architecture type detected from GGUF metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Llama architecture (includes Qwen, Phi-3, Llama 2/3, Mistral)
    Llama,
    /// GPT architecture (GPT-2, GPT-3)
    GPT,
    // Future: Other architectures
}

impl Architecture {
    /// Parse architecture from string
    pub fn from_str(s: &str) -> Result<Self, FactoryError> {
        match s.to_lowercase().as_str() {
            "llama" => Ok(Architecture::Llama),
            "gpt" | "gpt2" | "gpt-2" | "gpt3" | "gpt-3" => Ok(Architecture::GPT),
            _ => Err(FactoryError::UnknownArchitecture(s.to_string())),
        }
    }
}

/// Adapter factory for creating adapters from GGUF files
pub struct AdapterFactory;

impl AdapterFactory {
    /// Create adapter from GGUF file with automatic architecture detection
    ///
    /// # Arguments
    /// - `path`: Path to GGUF file
    ///
    /// # Returns
    /// Adapter instance with correct model type
    ///
    /// # Errors
    /// Returns error if:
    /// - File not found
    /// - GGUF parsing fails
    /// - Architecture not supported
    /// - Model loading fails
    pub fn from_gguf(path: &str) -> Result<LlamaModelAdapter, FactoryError> {
        // Parse GGUF metadata
        let metadata = GGUFMetadata::from_file(path)
            .map_err(|e| FactoryError::GGUFParsingFailed(e.to_string()))?;

        // Detect architecture from metadata
        let arch_str = metadata.architecture()
            .map_err(|e| FactoryError::GGUFParsingFailed(e.to_string()))?;
        let arch = Architecture::from_str(&arch_str)?;

        Self::from_gguf_with_arch(path, arch)
    }

    /// Create adapter from GGUF file with explicit architecture
    ///
    /// # Arguments
    /// - `path`: Path to GGUF file
    /// - `arch`: Architecture type
    ///
    /// # Returns
    /// Adapter instance with correct model type
    pub fn from_gguf_with_arch(
        path: &str,
        arch: Architecture,
    ) -> Result<LlamaModelAdapter, FactoryError> {
        match arch {
            Architecture::Llama => Self::load_llama_model(path),
            Architecture::GPT => Self::load_gpt_model(path),
        }
    }

    /// Create adapter from GGUF file with architecture string
    ///
    /// # Arguments
    /// - `path`: Path to GGUF file
    /// - `arch_str`: Architecture name ("llama", "gpt", etc.)
    pub fn from_gguf_with_arch_str(
        path: &str,
        arch_str: &str,
    ) -> Result<LlamaModelAdapter, FactoryError> {
        let arch = Architecture::from_str(arch_str)?;
        Self::from_gguf_with_arch(path, arch)
    }

    /// Detect architecture from filename
    ///
    /// This is a fallback heuristic when GGUF metadata is not available.
    fn detect_architecture_from_filename(path: &str) -> Result<Architecture, FactoryError> {
        let path_lower = path.to_lowercase();

        if path_lower.contains("qwen") || path_lower.contains("phi") || path_lower.contains("llama") || path_lower.contains("mistral") {
            Ok(Architecture::Llama)
        } else if path_lower.contains("gpt") {
            Ok(Architecture::GPT)
        } else {
            Err(FactoryError::UnknownArchitecture(
                "Cannot detect architecture from filename".to_string(),
            ))
        }
    }

    /// Detect model variant from filename
    fn detect_model_variant(path: &str) -> Result<ModelType, FactoryError> {
        let path_lower = path.to_lowercase();

        if path_lower.contains("qwen") {
            Ok(ModelType::Qwen2_5)
        } else if path_lower.contains("phi") {
            Ok(ModelType::Phi3)
        } else if path_lower.contains("llama-2") {
            Ok(ModelType::Llama2)
        } else if path_lower.contains("llama-3") || path_lower.contains("llama3") {
            Ok(ModelType::Llama3)
        } else if path_lower.contains("gpt-2") || path_lower.contains("gpt2") {
            Ok(ModelType::GPT2)
        } else if path_lower.contains("gpt-3") || path_lower.contains("gpt3") {
            Ok(ModelType::GPT3)
        } else {
            Err(FactoryError::UnsupportedVariant(
                "Cannot detect model variant from filename".to_string(),
            ))
        }
    }

    /// Load Llama-family model (Qwen, Phi-3, Llama 2/3)
    fn load_llama_model(path: &str) -> Result<LlamaModelAdapter, FactoryError> {
        let variant = Self::detect_model_variant(path)?;

        match variant {
            ModelType::Qwen2_5 => {
                // TODO: Detect size from GGUF metadata
                let config = QwenConfig::qwen2_5_0_5b();
                let model = QwenWeightLoader::load_to_vram(path, &config)
                    .map_err(|e| FactoryError::ModelLoadingFailed(e.to_string()))?;
                Ok(LlamaModelAdapter::new_qwen(model))
            }
            ModelType::Phi3 => {
                let config = Phi3Config::phi3_mini_4k();
                let model = Phi3WeightLoader::load_to_vram(path, &config)
                    .map_err(|e| FactoryError::ModelLoadingFailed(e.to_string()))?;
                Ok(LlamaModelAdapter::new_phi3(model))
            }
            ModelType::Llama2 | ModelType::Llama3 => {
                Err(FactoryError::UnsupportedVariant("Llama 2/3 not yet implemented".to_string()))
            }
            _ => Err(FactoryError::UnsupportedVariant(format!("{:?}", variant))),
        }
    }

    /// Load GPT-family model (GPT-2, GPT-3)
    fn load_gpt_model(path: &str) -> Result<LlamaModelAdapter, FactoryError> {
        let variant = Self::detect_model_variant(path)?;

        match variant {
            ModelType::GPT2 => {
                // TODO: Detect size from GGUF metadata
                let config = GPTConfig::gpt2_small();
                let model = GPTWeightLoader::load_to_vram(path, &config)
                    .map_err(|e| FactoryError::ModelLoadingFailed(e.to_string()))?;
                Ok(LlamaModelAdapter::new_gpt2(model))
            }
            ModelType::GPT3 => {
                Err(FactoryError::UnsupportedVariant("GPT-3 not yet implemented".to_string()))
            }
            _ => Err(FactoryError::UnsupportedVariant(format!("{:?}", variant))),
        }
    }

    /// Create default adapter for testing
    ///
    /// Creates a Qwen 0.5B adapter with dummy GGUF file.
    /// Useful for tests and examples.
    pub fn default_for_testing() -> Result<LlamaModelAdapter, FactoryError> {
        let config = QwenConfig::qwen2_5_0_5b();
        let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config)
            .map_err(|e| FactoryError::ModelLoadingFailed(e.to_string()))?;
        Ok(LlamaModelAdapter::new_qwen(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_str() {
        assert_eq!(Architecture::from_str("llama").unwrap(), Architecture::Llama);
        assert_eq!(Architecture::from_str("gpt").unwrap(), Architecture::GPT);
        assert_eq!(Architecture::from_str("gpt2").unwrap(), Architecture::GPT);
        assert_eq!(Architecture::from_str("GPT-2").unwrap(), Architecture::GPT);
        assert!(Architecture::from_str("unknown").is_err());
    }

    #[test]
    fn test_detect_architecture_from_filename() {
        assert_eq!(
            AdapterFactory::detect_architecture_from_filename("qwen-2.5-0.5b.gguf").unwrap(),
            Architecture::Llama
        );
        assert_eq!(
            AdapterFactory::detect_architecture_from_filename("phi-3-mini.gguf").unwrap(),
            Architecture::Llama
        );
        assert_eq!(
            AdapterFactory::detect_architecture_from_filename("gpt2-small.gguf").unwrap(),
            Architecture::GPT
        );
        assert!(AdapterFactory::detect_architecture_from_filename("unknown.gguf").is_err());
    }

    #[test]
    fn test_detect_model_variant() {
        assert_eq!(
            AdapterFactory::detect_model_variant("qwen-2.5-0.5b.gguf").unwrap(),
            ModelType::Qwen2_5
        );
        assert_eq!(
            AdapterFactory::detect_model_variant("phi-3-mini.gguf").unwrap(),
            ModelType::Phi3
        );
        assert_eq!(
            AdapterFactory::detect_model_variant("gpt2-small.gguf").unwrap(),
            ModelType::GPT2
        );
        assert_eq!(
            AdapterFactory::detect_model_variant("llama-3-8b.gguf").unwrap(),
            ModelType::Llama3
        );
    }

    #[test]
    fn test_from_gguf_qwen() {
        let adapter = AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap();
        assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
        assert_eq!(adapter.vocab_size().unwrap(), 151936);
    }

    #[test]
    fn test_from_gguf_phi3() {
        let adapter = AdapterFactory::from_gguf("phi-3-mini.gguf").unwrap();
        assert_eq!(adapter.model_type(), ModelType::Phi3);
        assert_eq!(adapter.vocab_size().unwrap(), 32064);
    }

    #[test]
    fn test_from_gguf_gpt2() {
        let adapter = AdapterFactory::from_gguf("gpt2-small.gguf").unwrap();
        assert_eq!(adapter.model_type(), ModelType::GPT2);
        assert_eq!(adapter.vocab_size().unwrap(), 50257);
    }

    #[test]
    fn test_from_gguf_with_arch_str() {
        let adapter = AdapterFactory::from_gguf_with_arch_str("gpt2-model.gguf", "gpt").unwrap();
        assert_eq!(adapter.model_type(), ModelType::GPT2);
    }

    #[test]
    fn test_default_for_testing() {
        let adapter = AdapterFactory::default_for_testing().unwrap();
        assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
    }

    #[test]
    fn test_unsupported_variant() {
        let result = AdapterFactory::from_gguf("llama-2-7b.gguf");
        assert!(result.is_err());
        match result {
            Err(FactoryError::UnsupportedVariant(_)) => {}
            _ => panic!("Expected UnsupportedVariant error"),
        }
    }

    #[test]
    fn test_unknown_architecture() {
        let result = AdapterFactory::from_gguf("unknown-model.gguf");
        assert!(result.is_err());
        match result {
            Err(FactoryError::UnknownArchitecture(_)) => {}
            _ => panic!("Expected UnknownArchitecture error"),
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
