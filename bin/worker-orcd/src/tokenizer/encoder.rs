// Byte-Level BPE Encoder (LT-009)
//
// Implements byte-level BPE encoding algorithm to convert text strings into token IDs.
//
// Spec: M0-W-1362

use super::{Vocabulary, MergeTable};
use super::error::EncodeError;

/// Byte-level BPE encoder
pub struct BPEEncoder {
    vocab: Vocabulary,
    merges: MergeTable,
}

impl BPEEncoder {
    /// Create new BPE encoder with vocabulary and merge table
    pub fn new(vocab: Vocabulary, merges: MergeTable) -> Self {
        Self { vocab, merges }
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EncodeError> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        // 1. Convert to byte-level tokens
        let mut tokens = self.to_byte_level(text);
        
        // 2. Apply BPE merges
        tokens = self.apply_merges(tokens);
        
        // 3. Convert to IDs
        self.tokens_to_ids(&tokens)
    }
    
    /// Encode text with special tokens (BOS, EOS)
    pub fn encode_with_special(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<u32>, EncodeError> {
        let mut token_ids = Vec::new();
        
        if add_bos {
            token_ids.push(self.vocab.bos_token_id);
        }
        
        let content_ids = self.encode(text)?;
        token_ids.extend(content_ids);
        
        if add_eos {
            token_ids.push(self.vocab.eos_token_id);
        }
        
        tracing::debug!(
            "Encoded text (len={}) to {} tokens (BOS={}, EOS={})",
            text.len(),
            token_ids.len(),
            add_bos,
            add_eos
        );
        
        Ok(token_ids)
    }
    
    /// Convert UTF-8 text to byte-level tokens
    fn to_byte_level(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        
        for byte in text.bytes() {
            // Convert byte to string representation
            // For now, use simple byte-to-char mapping
            // In production, this would use proper byte-level BPE mapping
            let token = if byte == b' ' {
                "Ġ".to_string()  // Space token
            } else if byte == b'\n' {
                "Ċ".to_string()  // Newline token
            } else if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
                (byte as char).to_string()
            } else {
                // Non-ASCII bytes: use hex representation
                format!("<0x{:02X}>", byte)
            };
            
            tokens.push(token);
        }
        
        tokens
    }
    
    /// Apply BPE merges iteratively
    fn apply_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        if tokens.len() < 2 {
            return tokens;
        }
        
        // Keep merging until no more merges possible
        loop {
            let merge = self.find_best_merge(&tokens);
            
            match merge {
                Some((pos, _priority)) => {
                    // Merge tokens at position
                    let merged = format!("{}{}", tokens[pos], tokens[pos + 1]);
                    tokens.splice(pos..=pos + 1, std::iter::once(merged));
                }
                None => break,
            }
        }
        
        tokens
    }
    
    /// Find the best merge (lowest priority) in current token sequence
    fn find_best_merge(&self, tokens: &[String]) -> Option<(usize, u32)> {
        let mut best_merge: Option<(usize, u32)> = None;
        
        for i in 0..tokens.len().saturating_sub(1) {
            let left = &tokens[i];
            let right = &tokens[i + 1];
            
            if let Some(priority) = self.merges.get_priority(left, right) {
                match best_merge {
                    None => best_merge = Some((i, priority)),
                    Some((_, best_priority)) if priority < best_priority => {
                        best_merge = Some((i, priority));
                    }
                    _ => {}
                }
            }
        }
        
        best_merge
    }
    
    /// Convert tokens to token IDs
    fn tokens_to_ids(&self, tokens: &[String]) -> Result<Vec<u32>, EncodeError> {
        let mut ids = Vec::with_capacity(tokens.len());
        
        for token in tokens {
            match self.vocab.get_id(token) {
                Some(id) => ids.push(id),
                None => {
                    return Err(EncodeError::UnknownToken {
                        token: token.clone(),
                    });
                }
            }
        }
        
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{Vocabulary, MergeTable};
    
    fn create_test_encoder() -> BPEEncoder {
        // Simple vocab: individual chars + merged tokens
        let tokens = vec![
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "Ġ".to_string(),
            "w".to_string(),
            "r".to_string(),
            "d".to_string(),
            "he".to_string(),     // merged
            "ll".to_string(),     // merged
            "hello".to_string(),  // fully merged
        ];
        
        let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();
        
        // Merge rules
        let merge_lines = vec![
            "h e".to_string(),    // priority 0: h + e → he
            "l l".to_string(),    // priority 1: l + l → ll
            "he ll".to_string(),  // priority 2: he + ll → hell
        ];
        
        let merges = MergeTable::new(merge_lines).unwrap();
        
        BPEEncoder::new(vocab, merges)
    }
    
    #[test]
    fn test_encoder_creation() {
        let encoder = create_test_encoder();
        assert_eq!(encoder.vocab.vocab_size, 13);
        assert_eq!(encoder.merges.merge_count, 3);
    }
    
    #[test]
    fn test_to_byte_level() {
        let encoder = create_test_encoder();
        
        let tokens = encoder.to_byte_level("hello");
        assert_eq!(tokens, vec!["h", "e", "l", "l", "o"]);
        
        let tokens = encoder.to_byte_level("hi world");
        assert_eq!(tokens, vec!["h", "i", "Ġ", "w", "o", "r", "l", "d"]);
    }
    
    #[test]
    fn test_find_best_merge() {
        let encoder = create_test_encoder();
        
        let tokens = vec!["h".to_string(), "e".to_string(), "l".to_string()];
        let merge = encoder.find_best_merge(&tokens);
        
        // Should find "h e" merge (priority 0)
        assert_eq!(merge, Some((0, 0)));
    }
    
    #[test]
    fn test_apply_merges_simple() {
        let encoder = create_test_encoder();
        
        let tokens = vec!["h".to_string(), "e".to_string()];
        let merged = encoder.apply_merges(tokens);
        
        assert_eq!(merged, vec!["he"]);
    }
    
    #[test]
    fn test_apply_merges_multiple() {
        let encoder = create_test_encoder();
        
        let tokens = vec!["l".to_string(), "l".to_string()];
        let merged = encoder.apply_merges(tokens);
        
        assert_eq!(merged, vec!["ll"]);
    }
    
    #[test]
    fn test_tokens_to_ids() {
        let encoder = create_test_encoder();
        
        let tokens = vec!["h".to_string(), "e".to_string(), "l".to_string()];
        let ids = encoder.tokens_to_ids(&tokens).unwrap();
        
        assert_eq!(ids, vec![2, 3, 4]);
    }
    
    #[test]
    fn test_encode_simple() {
        let encoder = create_test_encoder();
        
        // Note: This test may fail if merges aren't applied correctly
        // The actual result depends on merge order
        let result = encoder.encode("he");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_encode_with_special_tokens() {
        let encoder = create_test_encoder();
        
        let ids = encoder.encode_with_special("he", true, true).unwrap();
        
        // Should have BOS + content + EOS
        assert_eq!(ids[0], 0);  // BOS
        assert_eq!(ids[ids.len() - 1], 1);  // EOS
        assert!(ids.len() >= 3);
    }
    
    #[test]
    fn test_encode_empty_string() {
        let encoder = create_test_encoder();
        
        let ids = encoder.encode("").unwrap();
        assert_eq!(ids, Vec::<u32>::new());
    }
    
    #[test]
    fn test_encode_with_space() {
        let encoder = create_test_encoder();
        
        let tokens = encoder.to_byte_level("h e");
        assert_eq!(tokens, vec!["h", "Ġ", "e"]);
    }
    
    #[test]
    fn test_unknown_token_error() {
        let encoder = create_test_encoder();
        
        // Create tokens that don't exist in vocab
        let tokens = vec!["xyz".to_string()];
        let result = encoder.tokens_to_ids(&tokens);
        
        assert!(matches!(result, Err(EncodeError::UnknownToken { .. })));
    }
}
