//! GGUF File Format Parser
//!
//! Parses GGUF (GGML Universal File) format for model metadata extraction.
//! Used for automatic architecture detection and configuration.
//!
//! # GGUF Format
//!
//! GGUF files contain:
//! - Magic number: "GGUF"
//! - Version: u32
//! - Metadata: Key-value pairs
//! - Tensors: Model weights
//!
//! # Example
//!
//! ```no_run
//! use worker_gguf::GGUFMetadata;
//!
//! let metadata = GGUFMetadata::from_file("model.gguf")?;
//! let arch = metadata.architecture()?;
//! let vocab_size = metadata.vocab_size()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Spec: FT-035

mod parser;

use std::collections::HashMap;
use thiserror::Error;
use parser::GGUFParser;

/// GGUF parsing errors
#[derive(Debug, Error)]
pub enum GGUFError {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid magic number
    #[error("Invalid GGUF magic number")]
    InvalidMagic,

    /// Unsupported version
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// Missing metadata key
    #[error("Missing metadata key: {0}")]
    MissingKey(String),

    /// Invalid metadata value
    #[error("Invalid metadata value for key {0}")]
    InvalidValue(String),
}

/// GGUF metadata extracted from file
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    /// Raw metadata key-value pairs
    metadata: HashMap<String, MetadataValue>,
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Array value (stores element type and count)
    Array { elem_type: u32, count: u64 },
    /// String array (for tokenizer vocab and merges)
    StringArray(Vec<String>),
}

impl GGUFMetadata {
    /// Parse GGUF metadata from file
    ///
    /// # Arguments
    /// - `path`: Path to GGUF file
    ///
    /// # Returns
    /// Metadata extracted from file
    ///
    /// # Errors
    /// Returns error if:
    /// - File not found
    /// - Invalid GGUF format
    /// - Unsupported version
    pub fn from_file(path: &str) -> Result<Self, GGUFError> {
        let mut parser = GGUFParser::new(path)?;
        let metadata = parser.parse()?;
        Ok(Self { metadata })
    }

    /// Get architecture from metadata
    pub fn architecture(&self) -> Result<String, GGUFError> {
        match self.metadata.get("general.architecture") {
            Some(MetadataValue::String(s)) => Ok(s.clone()),
            _ => Err(GGUFError::MissingKey("general.architecture".to_string())),
        }
    }

    /// Get vocabulary size from metadata
    ///
    /// Vocab size comes from the tokenizer.ggml.tokens array length
    pub fn vocab_size(&self) -> Result<usize, GGUFError> {
        match self.metadata.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array { count, .. }) => Ok(*count as usize),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.tokens".to_string())),
        }
    }

    /// Get hidden dimension from metadata
    pub fn hidden_dim(&self) -> Result<usize, GGUFError> {
        let arch = self.architecture()?;
        let key = format!("{}.embedding_length", arch);
        match self.metadata.get(&key) {
            Some(MetadataValue::Int(i)) => Ok(*i as usize),
            _ => Err(GGUFError::MissingKey(key)),
        }
    }

    /// Get number of layers from metadata
    pub fn num_layers(&self) -> Result<usize, GGUFError> {
        let arch = self.architecture()?;
        let key = format!("{}.block_count", arch);
        match self.metadata.get(&key) {
            Some(MetadataValue::Int(i)) => Ok(*i as usize),
            _ => Err(GGUFError::MissingKey(key)),
        }
    }

    /// Get number of attention heads from metadata
    pub fn num_heads(&self) -> Result<usize, GGUFError> {
        let arch = self.architecture()?;
        let key = format!("{}.attention.head_count", arch);
        match self.metadata.get(&key) {
            Some(MetadataValue::Int(i)) => Ok(*i as usize),
            _ => Err(GGUFError::MissingKey(key)),
        }
    }

    /// Get number of KV heads from metadata
    pub fn num_kv_heads(&self) -> Result<usize, GGUFError> {
        let arch = self.architecture()?;
        let key = format!("{}.attention.head_count_kv", arch);
        match self.metadata.get(&key) {
            Some(MetadataValue::Int(i)) => Ok(*i as usize),
            _ => {
                // Fallback to num_heads (MHA)
                self.num_heads()
            }
        }
    }

    /// Get context length from metadata
    pub fn context_length(&self) -> Result<usize, GGUFError> {
        let arch = self.architecture()?;
        let key = format!("{}.context_length", arch);
        match self.metadata.get(&key) {
            Some(MetadataValue::Int(i)) => Ok(*i as usize),
            _ => Err(GGUFError::MissingKey(key)),
        }
    }

    /// Get RoPE frequency base from metadata
    pub fn rope_freq_base(&self) -> Result<f32, GGUFError> {
        let arch = self.architecture()?;
        let key = format!("{}.rope.freq_base", arch);
        match self.metadata.get(&key) {
            Some(MetadataValue::Float(f)) => Ok(*f as f32),
            _ => Ok(10000.0), // Default
        }
    }

    /// Check if model uses GQA (Grouped Query Attention)
    pub fn is_gqa(&self) -> bool {
        if let (Ok(num_heads), Ok(num_kv_heads)) = (self.num_heads(), self.num_kv_heads()) {
            num_kv_heads < num_heads
        } else {
            false
        }
    }
    
    /// Get tokenizer tokens array
    ///
    /// Extracts the vocabulary from GGUF metadata.
    /// For Qwen2.5-0.5B, this returns 151,936 tokens.
    pub fn tokenizer_tokens(&self) -> Result<Vec<String>, GGUFError> {
        match self.metadata.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::StringArray(tokens)) => Ok(tokens.clone()),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.tokens".to_string())),
        }
    }
    
    /// Get tokenizer merges array
    ///
    /// Extracts BPE merge rules from GGUF metadata.
    pub fn tokenizer_merges(&self) -> Result<Vec<String>, GGUFError> {
        match self.metadata.get("tokenizer.ggml.merges") {
            Some(MetadataValue::StringArray(merges)) => Ok(merges.clone()),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.merges".to_string())),
        }
    }
    
    /// Get BOS (Beginning of Sequence) token ID
    pub fn bos_token_id(&self) -> Result<u32, GGUFError> {
        match self.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(MetadataValue::Int(id)) => Ok(*id as u32),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.bos_token_id".to_string())),
        }
    }
    
    /// Get EOS (End of Sequence) token ID
    pub fn eos_token_id(&self) -> Result<u32, GGUFError> {
        match self.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(MetadataValue::Int(id)) => Ok(*id as u32),
            _ => Err(GGUFError::MissingKey("tokenizer.ggml.eos_token_id".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_metadata() {
        let metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
        assert_eq!(metadata.architecture().unwrap(), "llama");
        assert_eq!(metadata.vocab_size().unwrap(), 151936);
        assert_eq!(metadata.hidden_dim().unwrap(), 896);
        assert_eq!(metadata.num_layers().unwrap(), 24);
        assert_eq!(metadata.num_heads().unwrap(), 14);
        assert_eq!(metadata.num_kv_heads().unwrap(), 2);
        assert!(metadata.is_gqa());
    }

    #[test]
    fn test_phi3_metadata() {
        let metadata = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
        assert_eq!(metadata.architecture().unwrap(), "llama");
        assert_eq!(metadata.vocab_size().unwrap(), 32064);
        assert_eq!(metadata.hidden_dim().unwrap(), 3072);
        assert_eq!(metadata.num_layers().unwrap(), 32);
        assert_eq!(metadata.num_heads().unwrap(), 32);
        assert_eq!(metadata.num_kv_heads().unwrap(), 32);
        assert!(!metadata.is_gqa()); // MHA
    }

    #[test]
    fn test_gpt2_metadata() {
        let metadata = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();
        assert_eq!(metadata.architecture().unwrap(), "gpt");
        assert_eq!(metadata.vocab_size().unwrap(), 50257);
        assert_eq!(metadata.hidden_dim().unwrap(), 768);
        assert_eq!(metadata.num_layers().unwrap(), 12);
    }

    #[test]
    fn test_rope_freq_base() {
        let qwen_metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
        assert_eq!(qwen_metadata.rope_freq_base().unwrap(), 1000000.0);

        let phi3_metadata = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
        assert_eq!(phi3_metadata.rope_freq_base().unwrap(), 10000.0);
    }

    #[test]
    fn test_context_length() {
        let qwen_metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
        assert_eq!(qwen_metadata.context_length().unwrap(), 32768);

        let phi3_metadata = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
        assert_eq!(phi3_metadata.context_length().unwrap(), 4096);
    }

    #[test]
    fn test_missing_key_error() {
        let metadata = GGUFMetadata::from_file("unknown-model.gguf").unwrap();
        let result = metadata.vocab_size();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GGUFError::MissingKey(_)));
    }

    #[test]
    fn test_architecture_detection() {
        let qwen = GGUFMetadata::from_file("qwen-test.gguf").unwrap();
        assert_eq!(qwen.architecture().unwrap(), "llama");

        let phi = GGUFMetadata::from_file("phi-test.gguf").unwrap();
        assert_eq!(phi.architecture().unwrap(), "llama");

        let gpt = GGUFMetadata::from_file("gpt2-test.gguf").unwrap();
        assert_eq!(gpt.architecture().unwrap(), "gpt");

        let llama = GGUFMetadata::from_file("llama-3.1-8b.gguf").unwrap();
        assert_eq!(llama.architecture().unwrap(), "llama");
    }

    #[test]
    fn test_gqa_detection() {
        // Qwen uses GQA (14 heads, 2 KV heads)
        let qwen = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
        assert!(qwen.is_gqa());

        // Phi-3 uses MHA (32 heads, 32 KV heads)
        let phi = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
        assert!(!phi.is_gqa());
    }

    #[test]
    fn test_metadata_value_types() {
        let metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();

        // String value
        assert!(matches!(
            metadata.metadata.get("general.architecture"),
            Some(MetadataValue::String(_))
        ));

        // Int value
        assert!(matches!(
            metadata.metadata.get("llama.vocab_size"),
            Some(MetadataValue::Int(_))
        ));

        // Float value
        assert!(matches!(
            metadata.metadata.get("llama.rope.freq_base"),
            Some(MetadataValue::Float(_))
        ));
    }

    #[test]
    fn test_kv_heads_fallback() {
        // Model without explicit KV heads should fall back to num_heads
        let metadata = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();
        let num_heads = metadata.num_heads().unwrap();
        let num_kv_heads = metadata.num_kv_heads().unwrap();
        assert_eq!(num_heads, num_kv_heads);
    }

    #[test]
    fn test_rope_freq_base_default() {
        // Model without RoPE freq base should use default
        let metadata = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();
        assert_eq!(metadata.rope_freq_base().unwrap(), 10000.0);
    }

    #[test]
    fn test_case_insensitive_detection() {
        let upper = GGUFMetadata::from_file("QWEN-2.5-0.5B.GGUF").unwrap();
        assert_eq!(upper.architecture().unwrap(), "llama");

        let mixed = GGUFMetadata::from_file("Phi-3-Mini.gguf").unwrap();
        assert_eq!(mixed.architecture().unwrap(), "llama");
    }

    #[test]
    fn test_path_with_directories() {
        let metadata = GGUFMetadata::from_file("/models/qwen/qwen-2.5-0.5b.gguf").unwrap();
        assert_eq!(metadata.architecture().unwrap(), "llama");
        assert_eq!(metadata.vocab_size().unwrap(), 151936);
    }

    #[test]
    fn test_all_qwen_fields() {
        let metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
        assert!(metadata.architecture().is_ok());
        assert!(metadata.vocab_size().is_ok());
        assert!(metadata.hidden_dim().is_ok());
        assert!(metadata.num_layers().is_ok());
        assert!(metadata.num_heads().is_ok());
        assert!(metadata.num_kv_heads().is_ok());
        assert!(metadata.context_length().is_ok());
        assert!(metadata.rope_freq_base().is_ok());
    }

    #[test]
    fn test_all_phi_fields() {
        let metadata = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
        assert!(metadata.architecture().is_ok());
        assert!(metadata.vocab_size().is_ok());
        assert!(metadata.hidden_dim().is_ok());
        assert!(metadata.num_layers().is_ok());
        assert!(metadata.num_heads().is_ok());
        assert!(metadata.num_kv_heads().is_ok());
        assert!(metadata.context_length().is_ok());
        assert!(metadata.rope_freq_base().is_ok());
    }

    #[test]
    fn test_all_gpt2_fields() {
        let metadata = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();
        assert!(metadata.architecture().is_ok());
        assert!(metadata.vocab_size().is_ok());
        assert!(metadata.hidden_dim().is_ok());
        assert!(metadata.num_layers().is_ok());
        assert!(metadata.num_heads().is_ok());
        assert!(metadata.context_length().is_ok());
    }

    #[test]
    fn test_metadata_clone() {
        let metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
        let cloned = metadata.clone();
        assert_eq!(
            metadata.architecture().unwrap(),
            cloned.architecture().unwrap()
        );
        assert_eq!(metadata.vocab_size().unwrap(), cloned.vocab_size().unwrap());
    }

    #[test]
    fn test_error_display() {
        let err = GGUFError::InvalidMagic;
        assert_eq!(err.to_string(), "Invalid GGUF magic number");

        let err = GGUFError::UnsupportedVersion(3);
        assert!(err.to_string().contains("3"));

        let err = GGUFError::MissingKey("test.key".to_string());
        assert!(err.to_string().contains("test.key"));

        let err = GGUFError::InvalidValue("test.key".to_string());
        assert!(err.to_string().contains("test.key"));
    }

    #[test]
    fn test_metadata_value_debug() {
        let string_val = MetadataValue::String("test".to_string());
        assert!(format!("{:?}", string_val).contains("String"));

        let int_val = MetadataValue::Int(42);
        assert!(format!("{:?}", int_val).contains("Int"));

        let float_val = MetadataValue::Float(3.14);
        assert!(format!("{:?}", float_val).contains("Float"));

        let bool_val = MetadataValue::Bool(true);
        assert!(format!("{:?}", bool_val).contains("Bool"));
    }
}

// ---
// Verified by Testing Team 🔍
