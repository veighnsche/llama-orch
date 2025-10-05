//! HuggingFace tokenizer.json backend
//!
//! Provides pure-Rust tokenization using the HuggingFace tokenizers crate.
//! Used for GPT-OSS-20B and other models that ship with tokenizer.json files.
//!
//! Spec: M0-W-1361, M0-W-1365

use std::path::Path;
use tokenizers::Tokenizer;
use crate::tokenizer::error::TokenizerError;

/// HuggingFace JSON tokenizer backend
///
/// Wraps the HuggingFace `tokenizers` crate to provide tokenization
/// for models that use tokenizer.json format (e.g., GPT-OSS-20B).
///
/// # Features
/// - Pure Rust implementation (no Python dependencies)
/// - Fast BPE tokenization
/// - Special token handling
/// - Vocabulary metadata access
#[derive(Debug)]
pub struct HfJsonTokenizer {
    /// Inner HuggingFace tokenizer instance
    inner: Tokenizer,
    /// Cached vocabulary size
    vocab_size: usize,
}

impl HfJsonTokenizer {
    /// Load tokenizer from tokenizer.json file
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    ///
    /// # Returns
    /// * `Ok(HfJsonTokenizer)` - Successfully loaded tokenizer
    /// * `Err(TokenizerError)` - Failed to load or parse tokenizer
    ///
    /// # Example
    /// ```no_run
    /// use worker_orcd::tokenizer::hf_json::HfJsonTokenizer;
    ///
    /// let tokenizer = HfJsonTokenizer::from_file("models/gpt-oss-20b/tokenizer.json")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let path_ref = path.as_ref();
        
        // Load tokenizer from file
        let inner = Tokenizer::from_file(path_ref).map_err(|e| {
            TokenizerError::LoadFailed(format!(
                "Failed to load tokenizer from {}: {}",
                path_ref.display(),
                e
            ))
        })?;
        
        // Extract vocabulary size
        let vocab_size = inner.get_vocab_size(true);
        
        tracing::info!(
            path = %path_ref.display(),
            vocab_size = vocab_size,
            "Loaded HuggingFace tokenizer"
        );
        
        Ok(Self { inner, vocab_size })
    }
    
    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `add_special_tokens` - Whether to add BOS/EOS tokens
    ///
    /// # Returns
    /// * `Ok(Vec<u32>)` - Encoded token IDs
    /// * `Err(TokenizerError)` - Encoding failed
    ///
    /// # Example
    /// ```no_run
    /// # use worker_orcd::tokenizer::hf_json::HfJsonTokenizer;
    /// # let tokenizer = HfJsonTokenizer::from_file("tokenizer.json")?;
    /// let tokens = tokenizer.encode("Hello, world!", true)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
        
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode token IDs to text
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to decode
    /// * `skip_special_tokens` - Whether to skip BOS/EOS tokens
    ///
    /// # Returns
    /// * `Ok(String)` - Decoded text
    /// * `Err(TokenizerError)` - Decoding failed
    ///
    /// # Example
    /// ```no_run
    /// # use worker_orcd::tokenizer::hf_json::HfJsonTokenizer;
    /// # let tokenizer = HfJsonTokenizer::from_file("tokenizer.json")?;
    /// let text = tokenizer.decode(&[123, 456, 789], true)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }
    
    /// Get vocabulary size
    ///
    /// # Returns
    /// Total number of tokens in vocabulary (including special tokens)
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get special token IDs
    ///
    /// # Returns
    /// Tuple of (bos_token_id, eos_token_id, pad_token_id)
    /// Returns None for tokens that don't exist
    pub fn special_tokens(&self) -> (Option<u32>, Option<u32>, Option<u32>) {
        let bos = self.inner.token_to_id("<|begin_of_text|>")
            .or_else(|| self.inner.token_to_id("<s>"))
            .or_else(|| self.inner.token_to_id("[BOS]"));
        
        let eos = self.inner.token_to_id("<|end_of_text|>")
            .or_else(|| self.inner.token_to_id("</s>"))
            .or_else(|| self.inner.token_to_id("[EOS]"));
        
        let pad = self.inner.token_to_id("<|pad|>")
            .or_else(|| self.inner.token_to_id("[PAD]"));
        
        (bos, eos, pad)
    }
    
    /// Get model type from tokenizer metadata
    ///
    /// # Returns
    /// Model type string (e.g., "gpt2", "gpt-neox")
    pub fn model_type(&self) -> Option<String> {
        // Try to extract model type from tokenizer metadata
        // This is best-effort and may not always be available
        None // TODO: Extract from tokenizer metadata if available
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    
    fn create_test_tokenizer_json() -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        
        // Minimal valid tokenizer.json
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel"
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "vocab": {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                    "ab": 3,
                    "bc": 4
                },
                "merges": [
                    "a b",
                    "b c"
                ]
            }
        }"#;
        
        file.write_all(json.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }
    
    #[test]
    fn test_load_tokenizer() {
        let file = create_test_tokenizer_json();
        let tokenizer = HfJsonTokenizer::from_file(file.path());
        assert!(tokenizer.is_ok());
    }
    
    #[test]
    fn test_load_missing_file() {
        let result = HfJsonTokenizer::from_file("/nonexistent/tokenizer.json");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TokenizerError::LoadFailed { .. }));
    }
    
    #[test]
    fn test_vocab_size() {
        let file = create_test_tokenizer_json();
        let tokenizer = HfJsonTokenizer::from_file(file.path()).unwrap();
        assert_eq!(tokenizer.vocab_size(), 5);
    }
    
    #[test]
    fn test_encode_decode_roundtrip() {
        let file = create_test_tokenizer_json();
        let tokenizer = HfJsonTokenizer::from_file(file.path()).unwrap();
        
        let text = "abc";
        let tokens = tokenizer.encode(text, false).unwrap();
        assert!(!tokens.is_empty());
        
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert_eq!(decoded, text);
    }
    
    #[test]
    fn test_encode_empty_string() {
        let file = create_test_tokenizer_json();
        let tokenizer = HfJsonTokenizer::from_file(file.path()).unwrap();
        
        let tokens = tokenizer.encode("", false).unwrap();
        assert_eq!(tokens.len(), 0);
    }
}

// ---
// Crafted by GPT-Gamma 🤖
