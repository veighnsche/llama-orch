// GGUF Vocabulary Parser (LT-007)
//
// Parses vocabulary from GGUF metadata to build token-to-ID and ID-to-token mappings
// for byte-level BPE tokenizer.
//
// Spec: M0-W-1362

use super::error::VocabError;
use std::collections::HashMap;

/// Vocabulary structure with bidirectional token↔ID mappings
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token string → Token ID
    pub token_to_id: HashMap<String, u32>,

    /// Token ID → Token string
    pub id_to_token: HashMap<u32, String>,

    /// Total vocabulary size
    pub vocab_size: u32,

    /// Beginning of sequence token ID
    pub bos_token_id: u32,

    /// End of sequence token ID
    pub eos_token_id: u32,

    /// Padding token ID (optional)
    pub pad_token_id: Option<u32>,
}

impl Vocabulary {
    /// Create new vocabulary from token list and special token IDs
    pub fn new(
        tokens: Vec<String>,
        bos_token_id: u32,
        eos_token_id: u32,
        pad_token_id: Option<u32>,
    ) -> Result<Self, VocabError> {
        let vocab_size = tokens.len() as u32;

        // Validate special tokens are within vocab range
        if bos_token_id >= vocab_size {
            return Err(VocabError::InvalidSpecialToken {
                token_type: "BOS".to_string(),
                id: bos_token_id,
                vocab_size,
            });
        }

        if eos_token_id >= vocab_size {
            return Err(VocabError::InvalidSpecialToken {
                token_type: "EOS".to_string(),
                id: eos_token_id,
                vocab_size,
            });
        }

        if let Some(pad_id) = pad_token_id {
            if pad_id >= vocab_size {
                return Err(VocabError::InvalidSpecialToken {
                    token_type: "PAD".to_string(),
                    id: pad_id,
                    vocab_size,
                });
            }
        }

        // Build bidirectional maps
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_token = HashMap::with_capacity(tokens.len());

        for (id, token) in tokens.into_iter().enumerate() {
            let id = id as u32;

            // Check for duplicates
            if token_to_id.contains_key(&token) {
                return Err(VocabError::DuplicateToken { id, token: token.clone() });
            }

            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Ok(Self { token_to_id, id_to_token, vocab_size, bos_token_id, eos_token_id, pad_token_id })
    }

    /// Get token ID for a token string
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string for a token ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Check if token exists in vocabulary
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Check if token ID is valid
    pub fn contains_id(&self, id: u32) -> bool {
        id < self.vocab_size
    }

    /// Get BOS token string
    pub fn bos_token(&self) -> Option<&str> {
        self.get_token(self.bos_token_id)
    }

    /// Get EOS token string
    pub fn eos_token(&self) -> Option<&str> {
        self.get_token(self.eos_token_id)
    }

    /// Get PAD token string
    pub fn pad_token(&self) -> Option<&str> {
        self.pad_token_id.and_then(|id| self.get_token(id))
    }
}

/// Vocabulary parser for GGUF metadata
pub struct VocabParser;

impl VocabParser {
    /// Parse vocabulary from GGUF metadata
    ///
    /// Extracts:
    /// - `tokenizer.ggml.tokens` array (token strings)
    /// - `tokenizer.ggml.bos_token_id` (BOS token ID)
    /// - `tokenizer.ggml.eos_token_id` (EOS token ID)
    /// - `tokenizer.ggml.pad_token_id` (optional PAD token ID)
    pub fn parse_from_metadata(
        tokens: Vec<String>,
        bos_token_id: u32,
        eos_token_id: u32,
        pad_token_id: Option<u32>,
    ) -> Result<Vocabulary, VocabError> {
        if tokens.is_empty() {
            return Err(VocabError::MissingMetadata("tokenizer.ggml.tokens is empty".to_string()));
        }

        tracing::info!(
            "Parsing vocabulary: {} tokens, BOS={}, EOS={}, PAD={:?}",
            tokens.len(),
            bos_token_id,
            eos_token_id,
            pad_token_id
        );

        Vocabulary::new(tokens, bos_token_id, eos_token_id, pad_token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vocab() -> Vocabulary {
        let tokens = vec![
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "Ġ".to_string(), // space token
        ];

        Vocabulary::new(tokens, 0, 1, None).unwrap()
    }

    #[test]
    fn test_vocab_creation() {
        let vocab = create_test_vocab();

        assert_eq!(vocab.vocab_size, 5);
        assert_eq!(vocab.bos_token_id, 0);
        assert_eq!(vocab.eos_token_id, 1);
        assert_eq!(vocab.pad_token_id, None);
    }

    #[test]
    fn test_token_to_id_lookup() {
        let vocab = create_test_vocab();

        assert_eq!(vocab.get_id("<BOS>"), Some(0));
        assert_eq!(vocab.get_id("<EOS>"), Some(1));
        assert_eq!(vocab.get_id("hello"), Some(2));
        assert_eq!(vocab.get_id("world"), Some(3));
        assert_eq!(vocab.get_id("Ġ"), Some(4));
        assert_eq!(vocab.get_id("unknown"), None);
    }

    #[test]
    fn test_id_to_token_lookup() {
        let vocab = create_test_vocab();

        assert_eq!(vocab.get_token(0), Some("<BOS>"));
        assert_eq!(vocab.get_token(1), Some("<EOS>"));
        assert_eq!(vocab.get_token(2), Some("hello"));
        assert_eq!(vocab.get_token(3), Some("world"));
        assert_eq!(vocab.get_token(4), Some("Ġ"));
        assert_eq!(vocab.get_token(99), None);
    }

    #[test]
    fn test_special_tokens() {
        let vocab = create_test_vocab();

        assert_eq!(vocab.bos_token(), Some("<BOS>"));
        assert_eq!(vocab.eos_token(), Some("<EOS>"));
        assert_eq!(vocab.pad_token(), None);
    }

    #[test]
    fn test_contains_token() {
        let vocab = create_test_vocab();

        assert!(vocab.contains_token("hello"));
        assert!(vocab.contains_token("world"));
        assert!(!vocab.contains_token("unknown"));
    }

    #[test]
    fn test_contains_id() {
        let vocab = create_test_vocab();

        assert!(vocab.contains_id(0));
        assert!(vocab.contains_id(4));
        assert!(!vocab.contains_id(5));
        assert!(!vocab.contains_id(100));
    }

    #[test]
    fn test_invalid_bos_token() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let result = Vocabulary::new(tokens, 10, 1, None);

        assert!(matches!(result, Err(VocabError::InvalidSpecialToken { .. })));
    }

    #[test]
    fn test_invalid_eos_token() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let result = Vocabulary::new(tokens, 0, 10, None);

        assert!(matches!(result, Err(VocabError::InvalidSpecialToken { .. })));
    }

    #[test]
    fn test_duplicate_token() {
        let tokens = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let result = Vocabulary::new(tokens, 0, 1, None);

        assert!(matches!(result, Err(VocabError::DuplicateToken { .. })));
    }

    #[test]
    fn test_vocab_parser() {
        let tokens = vec!["<BOS>".to_string(), "<EOS>".to_string(), "test".to_string()];
        let vocab = VocabParser::parse_from_metadata(tokens, 0, 1, None).unwrap();

        assert_eq!(vocab.vocab_size, 3);
        assert_eq!(vocab.bos_token_id, 0);
        assert_eq!(vocab.eos_token_id, 1);
    }

    #[test]
    fn test_empty_vocab() {
        let result = VocabParser::parse_from_metadata(vec![], 0, 1, None);

        assert!(matches!(result, Err(VocabError::MissingMetadata(_))));
    }
}
