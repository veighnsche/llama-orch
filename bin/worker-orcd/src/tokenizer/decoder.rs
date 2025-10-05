// Byte-Level BPE Decoder (LT-010)
//
// Implements byte-level BPE decoding algorithm to convert token IDs back into text.
//
// Spec: M0-W-1362

use super::Vocabulary;
use super::error::DecodeError;

/// Byte-level BPE decoder
pub struct BPEDecoder {
    vocab: Vocabulary,
}

impl BPEDecoder {
    /// Create new BPE decoder with vocabulary
    pub fn new(vocab: Vocabulary) -> Self {
        Self { vocab }
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, DecodeError> {
        self.decode_with_special(token_ids, true)
    }
    
    /// Decode token IDs to text with special token handling
    pub fn decode_with_special(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, DecodeError> {
        if token_ids.is_empty() {
            return Ok(String::new());
        }
        
        // 1. Convert IDs to tokens
        let tokens = self.ids_to_tokens(token_ids, skip_special_tokens)?;
        
        // 2. Convert byte-level tokens to bytes
        let bytes = self.from_byte_level(&tokens)?;
        
        // 3. Convert bytes to UTF-8 string
        let text = self.bytes_to_utf8(&bytes)?;
        
        tracing::debug!(
            "Decoded {} tokens to text (len={}, skip_special={})",
            token_ids.len(),
            text.len(),
            skip_special_tokens
        );
        
        Ok(text)
    }
    
    /// Convert token IDs to token strings
    fn ids_to_tokens(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, DecodeError> {
        let mut tokens = Vec::new();
        
        for &id in token_ids {
            // Skip special tokens if requested
            if skip_special_tokens {
                if id == self.vocab.bos_token_id
                    || id == self.vocab.eos_token_id
                    || Some(id) == self.vocab.pad_token_id
                {
                    continue;
                }
            }
            
            // Lookup token
            match self.vocab.get_token(id) {
                Some(token) => tokens.push(token.to_string()),
                None => {
                    return Err(DecodeError::UnknownTokenId { id });
                }
            }
        }
        
        Ok(tokens)
    }
    
    /// Convert byte-level tokens to byte sequence
    fn from_byte_level(&self, tokens: &[String]) -> Result<Vec<u8>, DecodeError> {
        let mut bytes = Vec::new();
        
        for token in tokens {
            if token == "Ġ" {
                // Space token
                bytes.push(b' ');
            } else if token == "Ċ" {
                // Newline token
                bytes.push(b'\n');
            } else if token.starts_with("<0x") && token.ends_with('>') {
                // Hex byte token: <0xXX>
                let hex = &token[3..token.len() - 1];
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => bytes.push(byte),
                    Err(_) => {
                        return Err(DecodeError::DecodingFailed {
                            reason: format!("Invalid hex byte token: {}", token),
                        });
                    }
                }
            } else {
                // Regular token: convert chars to bytes
                bytes.extend(token.bytes());
            }
        }
        
        Ok(bytes)
    }
    
    /// Convert byte sequence to UTF-8 string
    fn bytes_to_utf8(&self, bytes: &[u8]) -> Result<String, DecodeError> {
        String::from_utf8(bytes.to_vec()).map_err(|_| DecodeError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Vocabulary;
    
    fn create_test_decoder() -> BPEDecoder {
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
        ];
        
        let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();
        BPEDecoder::new(vocab)
    }
    
    #[test]
    fn test_decoder_creation() {
        let decoder = create_test_decoder();
        assert_eq!(decoder.vocab.vocab_size, 10);
    }
    
    #[test]
    fn test_ids_to_tokens() {
        let decoder = create_test_decoder();
        
        let ids = vec![2, 3, 4, 4, 5];  // h e l l o
        let tokens = decoder.ids_to_tokens(&ids, false).unwrap();
        
        assert_eq!(tokens, vec!["h", "e", "l", "l", "o"]);
    }
    
    #[test]
    fn test_ids_to_tokens_skip_special() {
        let decoder = create_test_decoder();
        
        let ids = vec![0, 2, 3, 1];  // BOS h e EOS
        let tokens = decoder.ids_to_tokens(&ids, true).unwrap();
        
        assert_eq!(tokens, vec!["h", "e"]);
    }
    
    #[test]
    fn test_from_byte_level_simple() {
        let decoder = create_test_decoder();
        
        let tokens = vec!["h".to_string(), "e".to_string(), "l".to_string()];
        let bytes = decoder.from_byte_level(&tokens).unwrap();
        
        assert_eq!(bytes, b"hel");
    }
    
    #[test]
    fn test_from_byte_level_space() {
        let decoder = create_test_decoder();
        
        let tokens = vec!["h".to_string(), "Ġ".to_string(), "e".to_string()];
        let bytes = decoder.from_byte_level(&tokens).unwrap();
        
        assert_eq!(bytes, b"h e");
    }
    
    #[test]
    fn test_from_byte_level_newline() {
        let decoder = create_test_decoder();
        
        let tokens = vec!["h".to_string(), "Ċ".to_string(), "e".to_string()];
        let bytes = decoder.from_byte_level(&tokens).unwrap();
        
        assert_eq!(bytes, b"h\ne");
    }
    
    #[test]
    fn test_from_byte_level_hex() {
        let decoder = create_test_decoder();
        
        let tokens = vec!["<0x41>".to_string()];  // 0x41 = 'A'
        let bytes = decoder.from_byte_level(&tokens).unwrap();
        
        assert_eq!(bytes, b"A");
    }
    
    #[test]
    fn test_bytes_to_utf8() {
        let decoder = create_test_decoder();
        
        let bytes = b"hello";
        let text = decoder.bytes_to_utf8(bytes).unwrap();
        
        assert_eq!(text, "hello");
    }
    
    #[test]
    fn test_decode_simple() {
        let decoder = create_test_decoder();
        
        let ids = vec![2, 3, 4, 4, 5];  // h e l l o
        let text = decoder.decode(&ids).unwrap();
        
        assert_eq!(text, "hello");
    }
    
    #[test]
    fn test_decode_with_space() {
        let decoder = create_test_decoder();
        
        let ids = vec![2, 3, 6, 7, 5];  // h e Ġ w o
        let text = decoder.decode(&ids).unwrap();
        
        assert_eq!(text, "he wo");
    }
    
    #[test]
    fn test_decode_with_special_tokens() {
        let decoder = create_test_decoder();
        
        let ids = vec![0, 2, 3, 1];  // BOS h e EOS
        let text = decoder.decode_with_special(&ids, true).unwrap();
        
        assert_eq!(text, "he");
    }
    
    #[test]
    fn test_decode_empty() {
        let decoder = create_test_decoder();
        
        let text = decoder.decode(&[]).unwrap();
        assert_eq!(text, "");
    }
    
    #[test]
    fn test_unknown_token_id_error() {
        let decoder = create_test_decoder();
        
        let ids = vec![999];
        let result = decoder.decode(&ids);
        
        assert!(matches!(result, Err(DecodeError::UnknownTokenId { .. })));
    }
    
    #[test]
    fn test_invalid_utf8_error() {
        let decoder = create_test_decoder();
        
        // Create invalid UTF-8 sequence
        let tokens = vec!["<0xFF>".to_string(), "<0xFE>".to_string()];
        let bytes = decoder.from_byte_level(&tokens).unwrap();
        let result = decoder.bytes_to_utf8(&bytes);
        
        assert!(matches!(result, Err(DecodeError::InvalidUtf8)));
    }
    
    #[test]
    fn test_round_trip() {
        let decoder = create_test_decoder();
        
        // Encode manually
        let text = "hello";
        let ids = vec![2, 3, 4, 4, 5];  // h e l l o
        
        // Decode
        let decoded = decoder.decode(&ids).unwrap();
        
        assert_eq!(decoded, text);
    }
}
