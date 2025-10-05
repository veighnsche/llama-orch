//! UTF-8 boundary-safe buffer for streaming tokens
//!
//! This module provides a buffer that handles partial UTF-8 multibyte sequences,
//! ensuring that only complete, valid UTF-8 strings are emitted during streaming.
//!
//! # Spec References
//! - M0-W-1312: UTF-8 boundary safety

use std::str;

/// UTF-8 boundary-safe buffer
///
/// Buffers incoming bytes and only returns complete UTF-8 strings.
/// Partial multibyte sequences are held until the complete character arrives.
///
/// # Example
/// ```
/// use worker_tokenizer::util::utf8::Utf8Buffer;
///
/// let mut buffer = Utf8Buffer::new();
///
/// // Push bytes for "Hello 👋" where emoji is split
/// let bytes1 = b"Hello ";
/// let bytes2 = &[0xF0, 0x9F]; // First 2 bytes of 👋
/// let bytes3 = &[0x91, 0x8B]; // Last 2 bytes of 👋
///
/// let strings1 = buffer.push(bytes1); // Returns ["Hello "]
/// let strings2 = buffer.push(bytes2); // Returns [] (incomplete)
/// let strings3 = buffer.push(bytes3); // Returns ["👋"]
/// ```
pub struct Utf8Buffer {
    /// Internal byte buffer for partial sequences
    buffer: Vec<u8>,
}

impl Utf8Buffer {
    /// Create a new UTF-8 buffer
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Push bytes and return complete UTF-8 strings
    ///
    /// This method appends bytes to the internal buffer and attempts to
    /// decode complete UTF-8 strings. Partial multibyte sequences are
    /// retained until the complete character arrives.
    ///
    /// # Arguments
    /// * `bytes` - Bytes to push (may contain partial UTF-8 sequences)
    ///
    /// Vector of complete UTF-8 strings that can be safely emitted
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);

        let mut result = Vec::new();

        // Find the last valid UTF-8 boundary
        let valid_up_to = match str::from_utf8(&self.buffer) {
            Ok(s) => {
                // All bytes are valid UTF-8
                if !s.is_empty() {
                    result.push(s.to_string());
                }
                self.buffer.clear();
                return result;
            }
            Err(e) => e.valid_up_to(),
        };

        // Split at the last valid boundary
        if valid_up_to > 0 {
            let valid_bytes = self.buffer[..valid_up_to].to_vec();
            if let Ok(s) = str::from_utf8(&valid_bytes) {
                result.push(s.to_string());
            }
            self.buffer.drain(..valid_up_to);
        }

        result
    }

    /// Flush remaining bytes
    ///
    /// Call this at the end of the stream to get any buffered bytes.
    /// If the buffer contains invalid UTF-8, returns None and clears the buffer.
    ///
    /// # Returns
    /// * `Some(String)` - Remaining valid UTF-8 string
    /// * `None` - Buffer was empty or contained invalid UTF-8
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let result = str::from_utf8(&self.buffer).ok().map(|s| s.to_string());

        self.buffer.clear();
        result
    }

    /// Check if buffer has pending bytes
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }
}

impl Default for Utf8Buffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_ascii() {
        let mut buffer = Utf8Buffer::new();
        let result = buffer.push(b"Hello, world!");
        assert_eq!(result, vec!["Hello, world!"]);
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_complete_emoji() {
        let mut buffer = Utf8Buffer::new();
        let result = buffer.push("Hello 👋".as_bytes());
        assert_eq!(result, vec!["Hello 👋"]);
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_split_2byte_char() {
        let mut buffer = Utf8Buffer::new();

        // ñ is 2 bytes: 0xC3 0xB1
        let result1 = buffer.push(&[0xC3]); // First byte
        assert_eq!(result1, Vec::<String>::new());
        assert!(buffer.has_pending());

        let result2 = buffer.push(&[0xB1]); // Second byte
        assert_eq!(result2, vec!["ñ"]);
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_split_3byte_char() {
        let mut buffer = Utf8Buffer::new();

        // 世 is 3 bytes: 0xE4 0xB8 0x96
        let result1 = buffer.push(&[0xE4, 0xB8]); // First 2 bytes
        assert_eq!(result1, Vec::<String>::new());
        assert!(buffer.has_pending());

        let result2 = buffer.push(&[0x96]); // Last byte
        assert_eq!(result2, vec!["世"]);
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_split_4byte_char() {
        let mut buffer = Utf8Buffer::new();

        // 👋 is 4 bytes: 0xF0 0x9F 0x91 0x8B
        let result1 = buffer.push(&[0xF0, 0x9F]); // First 2 bytes
        assert_eq!(result1, Vec::<String>::new());
        assert!(buffer.has_pending());

        let result2 = buffer.push(&[0x91, 0x8B]); // Last 2 bytes
        assert_eq!(result2, vec!["👋"]);
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_multiple_chars_with_split() {
        let mut buffer = Utf8Buffer::new();

        // "Hello 👋 世界" with split in the middle of 👋
        let result1 = buffer.push(b"Hello ");
        assert_eq!(result1, vec!["Hello "]);

        let result2 = buffer.push(&[0xF0, 0x9F]); // First 2 bytes of 👋
        assert_eq!(result2, Vec::<String>::new());

        let result3 = buffer.push(&[0x91, 0x8B]); // Last 2 bytes of 👋
        assert_eq!(result3, vec!["👋"]);

        let result4 = buffer.push(" 世界".as_bytes());
        assert_eq!(result4, vec![" 世界"]);
    }

    #[test]
    fn test_flush_empty() {
        let mut buffer = Utf8Buffer::new();
        assert_eq!(buffer.flush(), None);
    }

    #[test]
    fn test_flush_with_complete_string() {
        let mut buffer = Utf8Buffer::new();
        buffer.push(&[0xF0, 0x9F]); // Partial emoji

        buffer.push(&[0x91, 0x8B]); // Complete emoji
        let result = buffer.flush();
        assert_eq!(result, None); // Already flushed by push
    }

    #[test]
    fn test_flush_with_partial_sequence() {
        let mut buffer = Utf8Buffer::new();
        buffer.push(&[0xF0, 0x9F]); // Partial emoji (incomplete)

        // Flush should return None for invalid UTF-8
        let result = buffer.flush();
        assert_eq!(result, None);
        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_empty_input() {
        let mut buffer = Utf8Buffer::new();
        let result = buffer.push(&[]);
        assert_eq!(result, Vec::<String>::new());
    }

    #[test]
    fn test_mixed_ascii_and_multibyte() {
        let mut buffer = Utf8Buffer::new();

        let input = "Hello 世界 👋 Test".as_bytes();
        let result = buffer.push(input);
        assert_eq!(result, vec!["Hello 世界 👋 Test"]);
    }

    #[test]
    fn test_consecutive_emoji() {
        let mut buffer = Utf8Buffer::new();

        let result = buffer.push("👋🌍🎉".as_bytes());
        assert_eq!(result, vec!["👋🌍🎉"]);
    }
}

// ---
// Built by Foundation-Alpha 🏗️
