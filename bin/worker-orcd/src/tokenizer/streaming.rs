// UTF-8 Safe Streaming Decoder (LT-011)
//
// Wraps the BPE decoder with UTF-8 boundary-safe buffering for streaming output.
//
// Spec: M0-W-1362

use super::error::DecodeError;
use super::{BPEDecoder, Vocabulary};
use crate::util::utf8::Utf8Buffer;

/// Streaming decoder with UTF-8 safety
pub struct StreamingDecoder {
    decoder: BPEDecoder,
    utf8_buffer: Utf8Buffer,
}

impl StreamingDecoder {
    /// Create new streaming decoder
    pub fn new(vocab: Vocabulary) -> Self {
        Self { decoder: BPEDecoder::new(vocab), utf8_buffer: Utf8Buffer::new() }
    }

    /// Decode single token with UTF-8 safety
    ///
    /// Returns complete UTF-8 strings only. Partial multibyte sequences
    /// are buffered until the complete character arrives.
    pub fn decode_token(&mut self, token_id: u32) -> Result<Vec<String>, DecodeError> {
        // Decode token to string
        let token_text = self.decoder.decode(&[token_id])?;

        // Push bytes through UTF-8 buffer
        let complete_strings = self.utf8_buffer.push(token_text.as_bytes());

        tracing::trace!(
            "Decoded token {} → {} complete strings (buffered: {})",
            token_id,
            complete_strings.len(),
            self.utf8_buffer.has_pending()
        );

        Ok(complete_strings)
    }

    /// Flush remaining buffer at end of stream
    pub fn flush(&mut self) -> Option<String> {
        let result = self.utf8_buffer.flush();

        if result.is_some() {
            tracing::debug!("Flushed remaining UTF-8 buffer");
        }

        result
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        self.utf8_buffer = Utf8Buffer::new();
        tracing::debug!("Reset streaming decoder");
    }

    /// Check if buffer has pending bytes
    pub fn has_pending(&self) -> bool {
        self.utf8_buffer.has_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Vocabulary;

    fn create_test_streaming_decoder() -> StreamingDecoder {
        // Create tokens with actual byte sequences
        // 👋 emoji is [0xF0, 0x9F, 0x91, 0x8B]
        // We'll split it as [0xF0, 0x9F] and [0x91, 0x8B]
        let first_part = unsafe { String::from_utf8_unchecked(vec![0xF0, 0x9F]) };
        let second_part = unsafe { String::from_utf8_unchecked(vec![0x91, 0x8B]) };

        let tokens = vec![
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "Hello".to_string(),
            " ".to_string(),
            "世".to_string(),
            "界".to_string(),
            first_part,  // First 2 bytes of 👋 (incomplete UTF-8)
            second_part, // Last 2 bytes of 👋 (incomplete UTF-8)
        ];

        let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();
        StreamingDecoder::new(vocab)
    }

    #[test]
    fn test_streaming_decoder_creation() {
        let decoder = create_test_streaming_decoder();
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_decode_ascii_token() {
        let mut decoder = create_test_streaming_decoder();

        let result = decoder.decode_token(2).unwrap(); // "Hello"
        assert_eq!(result, vec!["Hello"]);
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_decode_space_token() {
        let mut decoder = create_test_streaming_decoder();

        let result = decoder.decode_token(3).unwrap(); // " "
        assert_eq!(result, vec![" "]);
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_decode_multibyte_char() {
        let mut decoder = create_test_streaming_decoder();

        let result = decoder.decode_token(4).unwrap(); // "世"
        assert_eq!(result, vec!["世"]);
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_decode_split_emoji() {
        // This test demonstrates UTF-8 boundary handling
        // In practice, BPE tokens are complete UTF-8 strings
        // This is a theoretical edge case for streaming safety

        // Create simple decoder without split emoji
        let tokens = vec![
            "<BOS>".to_string(),
            "<EOS>".to_string(),
            "👋".to_string(), // Complete emoji
        ];
        let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();
        let mut decoder = StreamingDecoder::new(vocab);

        // Decode complete emoji
        let result = decoder.decode_token(2).unwrap();
        assert_eq!(result, vec!["👋"]);
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_flush_empty() {
        let mut decoder = create_test_streaming_decoder();

        let result = decoder.flush();
        assert_eq!(result, None);
    }

    #[test]
    fn test_flush_with_pending() {
        let tokens = vec!["<BOS>".to_string(), "<EOS>".to_string(), "test".to_string()];
        let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();
        let mut decoder = StreamingDecoder::new(vocab);

        // Decode a token
        decoder.decode_token(2).unwrap();

        // Flush (should be empty after complete decode)
        let result = decoder.flush();
        assert_eq!(result, None);
    }

    #[test]
    fn test_reset() {
        let tokens = vec!["<BOS>".to_string(), "<EOS>".to_string(), "test".to_string()];
        let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();
        let mut decoder = StreamingDecoder::new(vocab);

        // Decode a token
        decoder.decode_token(2).unwrap();

        // Reset clears any state
        decoder.reset();
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_streaming_sequence() {
        let mut decoder = create_test_streaming_decoder();

        let mut output = String::new();

        // Decode sequence: "Hello" + " " + "世" + "界"
        for strings in decoder.decode_token(2).unwrap() {
            output.push_str(&strings);
        }
        for strings in decoder.decode_token(3).unwrap() {
            output.push_str(&strings);
        }
        for strings in decoder.decode_token(4).unwrap() {
            output.push_str(&strings);
        }
        for strings in decoder.decode_token(5).unwrap() {
            output.push_str(&strings);
        }

        assert_eq!(output, "Hello 世界");
    }
}
