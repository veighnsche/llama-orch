//! Tokenizer backend abstraction
//!
//! Provides a unified interface for different tokenizer backends:
//! - GGUF BPE: For Llama-family models (Qwen, Phi-3)
//! - HF JSON: For GPT-family models (GPT-OSS-20B)
//!
//! Spec: M0-W-1361, M0-W-1362

use crate::{BPEEncoder, BPEDecoder, hf_json::HfJsonTokenizer, TokenizerError};
use std::path::Path;

/// Tokenizer backend selection
///
/// Determines which tokenization backend to use based on model requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerBackend {
    /// GGUF byte-level BPE (for Llama-family models)
    GgufBpe,
    /// HuggingFace tokenizer.json (for GPT-family models)
    HfJson,
}

impl TokenizerBackend {
    /// Detect backend from file extension
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer file
    ///
    /// # Returns
    /// Detected backend type
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        let path_ref = path.as_ref();
        
        if path_ref.extension().and_then(|s| s.to_str()) == Some("json") {
            TokenizerBackend::HfJson
        } else {
            TokenizerBackend::GgufBpe
        }
    }
    
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            TokenizerBackend::GgufBpe => "GGUF BPE",
            TokenizerBackend::HfJson => "HuggingFace JSON",
        }
    }
}

/// Unified tokenizer interface
///
/// Abstracts over different tokenizer backends to provide a consistent API.
pub enum Tokenizer {
    /// GGUF BPE tokenizer
    GgufBpe {
        encoder: BPEEncoder,
        decoder: BPEDecoder,
    },
    /// HuggingFace JSON tokenizer
    HfJson(HfJsonTokenizer),
}

impl Tokenizer {
    /// Load tokenizer from file
    ///
    /// Automatically detects backend from file extension.
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer file (*.json for HF, GGUF for BPE)
    ///
    /// # Returns
    /// * `Ok(Tokenizer)` - Successfully loaded tokenizer
    /// * `Err(TokenizerError)` - Failed to load
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let backend = TokenizerBackend::from_path(&path);
        
        match backend {
            TokenizerBackend::HfJson => {
                let hf = HfJsonTokenizer::from_file(path)?;
                Ok(Tokenizer::HfJson(hf))
            }
            TokenizerBackend::GgufBpe => {
                // Note: GGUF BPE loading requires vocab and merges from GGUF file
                // This is handled separately in model loading
                Err(TokenizerError::EncodeFailed(
                    "GGUF BPE tokenizer must be loaded from GGUF file".to_string()
                ))
            }
        }
    }
    
    /// Load tokenizer from GGUF file
    ///
    /// Extracts vocabulary and merges from GGUF metadata.
    ///
    /// # Arguments
    /// * `path` - Path to GGUF file
    ///
    /// # Returns
    /// * `Ok(Tokenizer)` - Successfully loaded tokenizer
    /// * `Err(TokenizerError)` - Failed to load
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        use worker_gguf::GGUFMetadata;
        use crate::{Vocabulary, MergeTable};
        
        // Parse GGUF file
        let path_str = path.as_ref().to_str()
            .ok_or_else(|| TokenizerError::LoadFailed("Invalid path".to_string()))?;
        
        let metadata = GGUFMetadata::from_file(path_str)
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to parse GGUF: {}", e)))?;
        
        // Extract tokens
        let tokens = metadata.tokenizer_tokens()
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to extract tokens: {}", e)))?;
        
        // Extract merges
        let merge_strings = metadata.tokenizer_merges()
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to extract merges: {}", e)))?;
        
        // Extract special token IDs
        let bos_token_id = metadata.bos_token_id()
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to extract BOS token: {}", e)))?;
        
        let eos_token_id = metadata.eos_token_id()
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to extract EOS token: {}", e)))?;
        
        // Build vocabulary
        let vocab = Vocabulary::new(tokens, bos_token_id, eos_token_id, Some(eos_token_id))
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to build vocab: {}", e)))?;
        
        // Build merge table
        let merges = MergeTable::new(merge_strings)
            .map_err(|e| TokenizerError::LoadFailed(format!("Failed to build merges: {}", e)))?;
        
        // Create encoder and decoder
        let encoder = BPEEncoder::new(vocab.clone(), merges);
        let decoder = BPEDecoder::new(vocab);
        
        Ok(Tokenizer::GgufBpe { encoder, decoder })
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        match self {
            Tokenizer::GgufBpe { encoder, .. } => {
                encoder.encode(text).map_err(|e| TokenizerError::EncodeFailed(e.to_string()))
            }
            Tokenizer::HfJson(hf) => {
                hf.encode(text, add_special_tokens)
            }
        }
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        match self {
            Tokenizer::GgufBpe { decoder, .. } => {
                decoder.decode(token_ids).map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
            }
            Tokenizer::HfJson(hf) => {
                hf.decode(token_ids, skip_special_tokens)
            }
        }
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::GgufBpe { .. } => {
                // TODO: Add vocab_size accessor to BPEEncoder
                0
            }
            Tokenizer::HfJson(hf) => hf.vocab_size(),
        }
    }
    
    /// Get backend type
    pub fn backend(&self) -> TokenizerBackend {
        match self {
            Tokenizer::GgufBpe { .. } => TokenizerBackend::GgufBpe,
            Tokenizer::HfJson(_) => TokenizerBackend::HfJson,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backend_detection() {
        assert_eq!(
            TokenizerBackend::from_path("tokenizer.json"),
            TokenizerBackend::HfJson
        );
        assert_eq!(
            TokenizerBackend::from_path("model.gguf"),
            TokenizerBackend::GgufBpe
        );
    }
    
    #[test]
    fn test_backend_names() {
        assert_eq!(TokenizerBackend::GgufBpe.name(), "GGUF BPE");
        assert_eq!(TokenizerBackend::HfJson.name(), "HuggingFace JSON");
    }
}

// ---
// Crafted by GPT-Gamma 🤖
