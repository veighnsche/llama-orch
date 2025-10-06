// Tokenizer metadata
//
// Metadata extracted from tokenizers for observability and validation.
//
// Spec: M0-W-1361, M0-W-1364
// Story: GT-003

use serde::{Deserialize, Serialize};

/// Tokenizer metadata
///
/// Contains information about the tokenizer configuration including
/// special tokens, vocabulary size, and context length.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerMetadata {
    /// End-of-sequence token ID
    pub eos_id: Option<u32>,

    /// Beginning-of-sequence token ID
    pub bos_id: Option<u32>,

    /// Padding token ID
    pub pad_id: Option<u32>,

    /// Unknown token ID
    pub unk_id: Option<u32>,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum context length (if specified in tokenizer)
    pub model_max_context: Option<usize>,

    /// Tokenizer type/kind
    pub tokenizer_kind: String,
}

impl TokenizerMetadata {
    /// Create new tokenizer metadata
    pub fn new(
        eos_id: Option<u32>,
        bos_id: Option<u32>,
        pad_id: Option<u32>,
        unk_id: Option<u32>,
        vocab_size: usize,
        model_max_context: Option<usize>,
        tokenizer_kind: String,
    ) -> Self {
        Self { eos_id, bos_id, pad_id, unk_id, vocab_size, model_max_context, tokenizer_kind }
    }

    /// Validate metadata is reasonable
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("Vocabulary size cannot be zero".to_string());
        }

        if self.vocab_size > 1_000_000 {
            return Err(format!("Vocabulary size {} is unreasonably large", self.vocab_size));
        }

        // Validate special token IDs are within vocab range
        if let Some(eos) = self.eos_id {
            if eos as usize >= self.vocab_size {
                return Err(format!("EOS token ID {} exceeds vocab size {}", eos, self.vocab_size));
            }
        }

        if let Some(bos) = self.bos_id {
            if bos as usize >= self.vocab_size {
                return Err(format!("BOS token ID {} exceeds vocab size {}", bos, self.vocab_size));
            }
        }

        if let Some(pad) = self.pad_id {
            if pad as usize >= self.vocab_size {
                return Err(format!("PAD token ID {} exceeds vocab size {}", pad, self.vocab_size));
            }
        }

        if let Some(unk) = self.unk_id {
            if unk as usize >= self.vocab_size {
                return Err(format!("UNK token ID {} exceeds vocab size {}", unk, self.vocab_size));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = TokenizerMetadata::new(
            Some(50256),
            Some(50256),
            None,
            Some(0),
            50257,
            Some(2048),
            "hf-json".to_string(),
        );

        assert_eq!(metadata.eos_id, Some(50256));
        assert_eq!(metadata.bos_id, Some(50256));
        assert_eq!(metadata.vocab_size, 50257);
        assert_eq!(metadata.model_max_context, Some(2048));
    }

    #[test]
    fn test_metadata_validation() {
        let metadata = TokenizerMetadata::new(
            Some(50256),
            Some(50256),
            None,
            None,
            50257,
            None,
            "hf-json".to_string(),
        );

        assert!(metadata.validate().is_ok());
    }

    #[test]
    fn test_metadata_validation_zero_vocab() {
        let metadata =
            TokenizerMetadata::new(None, None, None, None, 0, None, "hf-json".to_string());

        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_metadata_validation_token_out_of_range() {
        let metadata = TokenizerMetadata::new(
            Some(100000), // Out of range
            None,
            None,
            None,
            50257,
            None,
            "hf-json".to_string(),
        );

        assert!(metadata.validate().is_err());
    }
}

// ---
// Crafted by GPT-Gamma 🤖
