//! Hex string validation applet
//!
//! Validates hexadecimal strings (digests, hashes, signatures).

use crate::error::{Result, ValidationError};

/// Validate hexadecimal string
///
/// # Rules
/// - Exact length match
/// - Only hex characters: `[0-9a-fA-F]+`
/// - No whitespace
/// - No null bytes
/// - Case-insensitive
///
/// # Arguments
/// * `s` - Hex string to validate
/// * `expected_len` - Expected length (e.g., 64 for SHA-256)
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_hex_string;
///
/// // Valid SHA-256 digest
/// let digest = "a".repeat(64);
/// assert!(validate_hex_string(&digest, 64).is_ok());
///
/// // Invalid
/// assert!(validate_hex_string("xyz", 64).is_err());  // Non-hex
/// assert!(validate_hex_string("abc", 64).is_err());  // Wrong length
/// ```
///
/// # Common Lengths
/// - SHA-256: 64 chars
/// - SHA-1: 40 chars
/// - MD5: 32 chars
///
/// # Errors
/// * `ValidationError::WrongLength` - Length doesn't match expected
/// * `ValidationError::InvalidHex` - Contains non-hex character
/// * `ValidationError::NullByte` - Contains null byte
pub fn validate_hex_string(s: &str, expected_len: usize) -> Result<()> {
    // Check length first (fast check, O(1) for UTF-8 strings)
    // This prevents processing of obviously invalid inputs
    if s.len() != expected_len {
        return Err(ValidationError::WrongLength {
            actual: s.len(),
            expected: expected_len,
        });
    }
    
    // Early return for empty strings (valid if expected_len is 0)
    if s.is_empty() {
        // Already validated by length check above
        return Ok(());
    }
    
    // PERFORMANCE PHASE 2: Single-pass validation
    // Combines null byte check and hex validation (2 iterations → 1 iteration)
    // Auth-min approved: Security-equivalent, same validation order
    
    for c in s.chars() {
        // SECURITY: Check null byte first (prevents C string truncation)
        if c == '\0' {
            return Err(ValidationError::NullByte);
        }
        
        // SECURITY: Validate hex characters (case-insensitive)
        // Only ASCII hex digits allowed: 0-9, a-f, A-F
        // is_ascii_hexdigit() is faster than manual range checks
        if !c.is_ascii_hexdigit() {
            return Err(ValidationError::InvalidHex { char: c });
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_hex_strings() {
        // SHA-256 (64 chars)
        let sha256 = "a".repeat(64);
        assert!(validate_hex_string(&sha256, 64).is_ok());
        
        // Mixed case
        assert!(validate_hex_string("AbCdEf0123456789", 16).is_ok());
        
        // All digits
        assert!(validate_hex_string("0123456789", 10).is_ok());
        
        // All letters
        assert!(validate_hex_string("abcdef", 6).is_ok());
        assert!(validate_hex_string("ABCDEF", 6).is_ok());
    }
    
    #[test]
    fn test_wrong_length_rejected() {
        assert!(matches!(
            validate_hex_string("abc", 64),
            Err(ValidationError::WrongLength { actual: 3, expected: 64 })
        ));
        
        assert!(matches!(
            validate_hex_string(&"a".repeat(65), 64),
            Err(ValidationError::WrongLength { actual: 65, expected: 64 })
        ));
    }
    
    #[test]
    fn test_non_hex_rejected() {
        assert!(matches!(
            validate_hex_string("xyz", 3),
            Err(ValidationError::InvalidHex { char: 'x' })
        ));
        
        assert!(matches!(
            validate_hex_string("abc 123", 7),
            Err(ValidationError::InvalidHex { char: ' ' })
        ));
        
        assert!(matches!(
            validate_hex_string("abcg", 4),
            Err(ValidationError::InvalidHex { char: 'g' })
        ));
    }
    
    #[test]
    fn test_null_byte_rejected() {
        assert_eq!(
            validate_hex_string("abc\0def", 7),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_common_hash_lengths() {
        // SHA-256: 64 hex chars
        let sha256 = "a".repeat(64);
        assert!(validate_hex_string(&sha256, 64).is_ok());
        
        // SHA-1: 40 hex chars
        let sha1 = "b".repeat(40);
        assert!(validate_hex_string(&sha1, 40).is_ok());
        
        // MD5: 32 hex chars
        let md5 = "c".repeat(32);
        assert!(validate_hex_string(&md5, 32).is_ok());
    }
    
    #[test]
    fn test_null_byte_positions() {
        // Null byte at start
        assert_eq!(
            validate_hex_string("\0abcdef", 7),
            Err(ValidationError::NullByte)
        );
        
        // Null byte at end
        assert_eq!(
            validate_hex_string("abcdef\0", 7),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_more_invalid_characters() {
        // Hyphen
        assert!(matches!(
            validate_hex_string("abc-def", 7),
            Err(ValidationError::InvalidHex { char: '-' })
        ));
        
        // Underscore
        assert!(matches!(
            validate_hex_string("abc_def", 7),
            Err(ValidationError::InvalidHex { char: '_' })
        ));
        
        // Colon
        assert!(matches!(
            validate_hex_string("abc:def", 7),
            Err(ValidationError::InvalidHex { char: ':' })
        ));
        
        // Unicode (é is 2 bytes in UTF-8, so length check fails first)
        // We expect WrongLength because "abcé" is 5 bytes, not 4
        let result = validate_hex_string("abcé", 4);
        assert!(result.is_err());
        // Either WrongLength (if length check first) or InvalidHex (if char check first)
        match result {
            Err(ValidationError::WrongLength { .. }) | Err(ValidationError::InvalidHex { .. }) => {},
            other => panic!("Expected WrongLength or InvalidHex, got: {:?}", other),
        }
    }
    
    #[test]
    fn test_empty_string() {
        assert!(matches!(
            validate_hex_string("", 64),
            Err(ValidationError::WrongLength { actual: 0, expected: 64 })
        ));
    }
    
    #[test]
    fn test_early_termination() {
        // Length check happens first
        let result = validate_hex_string("xyz", 64);
        assert!(matches!(result, Err(ValidationError::WrongLength { .. })));
        
        // Null byte checked before hex validation
        let result = validate_hex_string("abc\0def", 7);
        assert_eq!(result, Err(ValidationError::NullByte));
        
        // Stop on first invalid hex character
        let result = validate_hex_string("abcxyz", 6);
        assert!(matches!(result, Err(ValidationError::InvalidHex { char: 'x' })));
    }
    
    // ========== ROBUSTNESS TESTS ==========
    
    #[test]
    fn test_robustness_zero_length() {
        // Zero-length expected
        assert!(validate_hex_string("", 0).is_ok());
        
        // Non-empty when zero expected
        assert!(matches!(
            validate_hex_string("a", 0),
            Err(ValidationError::WrongLength { actual: 1, expected: 0 })
        ));
    }
    
    #[test]
    fn test_robustness_whitespace_rejection() {
        // Leading whitespace
        assert!(matches!(
            validate_hex_string(" abc", 4),
            Err(ValidationError::InvalidHex { char: ' ' })
        ));
        
        // Trailing whitespace
        assert!(matches!(
            validate_hex_string("abc ", 4),
            Err(ValidationError::InvalidHex { char: ' ' })
        ));
        
        // Internal whitespace
        assert!(matches!(
            validate_hex_string("ab cd", 5),
            Err(ValidationError::InvalidHex { char: ' ' })
        ));
        
        // Tab character
        assert!(matches!(
            validate_hex_string("ab\tcd", 5),
            Err(ValidationError::InvalidHex { char: '\t' })
        ));
        
        // Newline
        assert!(matches!(
            validate_hex_string("ab\ncd", 5),
            Err(ValidationError::InvalidHex { char: '\n' })
        ));
    }
    
    #[test]
    fn test_robustness_special_characters() {
        // Punctuation
        assert!(matches!(
            validate_hex_string("ab.cd", 5),
            Err(ValidationError::InvalidHex { char: '.' })
        ));
        
        assert!(matches!(
            validate_hex_string("ab,cd", 5),
            Err(ValidationError::InvalidHex { char: ',' })
        ));
        
        assert!(matches!(
            validate_hex_string("ab;cd", 5),
            Err(ValidationError::InvalidHex { char: ';' })
        ));
        
        // Brackets
        assert!(matches!(
            validate_hex_string("ab[cd", 5),
            Err(ValidationError::InvalidHex { char: '[' })
        ));
        
        assert!(matches!(
            validate_hex_string("ab]cd", 5),
            Err(ValidationError::InvalidHex { char: ']' })
        ));
    }
    
    #[test]
    fn test_robustness_hex_prefix_rejection() {
        // 0x prefix should be rejected
        assert!(matches!(
            validate_hex_string("0xabcd", 6),
            Err(ValidationError::InvalidHex { char: 'x' })
        ));
        
        // 0X prefix should be rejected
        assert!(matches!(
            validate_hex_string("0Xabcd", 6),
            Err(ValidationError::InvalidHex { char: 'X' })
        ));
    }
    
    #[test]
    fn test_robustness_case_insensitivity() {
        // All lowercase
        assert!(validate_hex_string("abcdef", 6).is_ok());
        
        // All uppercase
        assert!(validate_hex_string("ABCDEF", 6).is_ok());
        
        // Mixed case
        assert!(validate_hex_string("AbCdEf", 6).is_ok());
        assert!(validate_hex_string("aBcDeF", 6).is_ok());
        assert!(validate_hex_string("ABcdEF", 6).is_ok());
    }
    
    #[test]
    fn test_robustness_boundary_hex_characters() {
        // Just before 'a' in ASCII
        assert!(matches!(
            validate_hex_string("`", 1),
            Err(ValidationError::InvalidHex { char: '`' })
        ));
        
        // Just after 'f' in ASCII
        assert!(matches!(
            validate_hex_string("g", 1),
            Err(ValidationError::InvalidHex { char: 'g' })
        ));
        
        // Just before 'A' in ASCII
        assert!(matches!(
            validate_hex_string("@", 1),
            Err(ValidationError::InvalidHex { char: '@' })
        ));
        
        // Just after 'F' in ASCII
        assert!(matches!(
            validate_hex_string("G", 1),
            Err(ValidationError::InvalidHex { char: 'G' })
        ));
        
        // Just before '0' in ASCII
        assert!(matches!(
            validate_hex_string("/", 1),
            Err(ValidationError::InvalidHex { char: '/' })
        ));
        
        // Just after '9' in ASCII
        assert!(matches!(
            validate_hex_string(":", 1),
            Err(ValidationError::InvalidHex { char: ':' })
        ));
    }
    
    #[test]
    fn test_robustness_utf8_multibyte() {
        // 2-byte UTF-8 character (é = 0xC3 0xA9)
        let result = validate_hex_string("é", 1);
        assert!(result.is_err());
        // Either WrongLength (2 bytes != 1) or InvalidHex
        match result {
            Err(ValidationError::WrongLength { .. }) | Err(ValidationError::InvalidHex { .. }) => {},
            other => panic!("Expected WrongLength or InvalidHex, got: {:?}", other),
        }
        
        // 3-byte UTF-8 character (€ = 0xE2 0x82 0xAC)
        let result = validate_hex_string("€", 1);
        assert!(result.is_err());
        
        // 4-byte UTF-8 character (emoji 🚀 = 0xF0 0x9F 0x9A 0x80)
        let result = validate_hex_string("🚀", 1);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_robustness_control_characters() {
        // Bell (0x07)
        assert!(matches!(
            validate_hex_string("ab\x07cd", 5),
            Err(ValidationError::InvalidHex { char: '\x07' })
        ));
        
        // Backspace (0x08)
        assert!(matches!(
            validate_hex_string("ab\x08cd", 5),
            Err(ValidationError::InvalidHex { char: '\x08' })
        ));
        
        // Escape (0x1b)
        assert!(matches!(
            validate_hex_string("ab\x1bcd", 5),
            Err(ValidationError::InvalidHex { char: '\x1b' })
        ));
    }
    
    #[test]
    fn test_robustness_very_long_strings() {
        // Very long valid hex string (1024 bytes)
        let long_hex = "a".repeat(1024);
        assert!(validate_hex_string(&long_hex, 1024).is_ok());
        
        // Very long invalid hex string (should fail fast on first invalid char)
        let mut long_invalid = "a".repeat(1023);
        long_invalid.push('z');
        assert!(matches!(
            validate_hex_string(&long_invalid, 1024),
            Err(ValidationError::InvalidHex { char: 'z' })
        ));
    }
    
    #[test]
    fn test_robustness_all_valid_hex_digits() {
        // Test every valid hex digit individually
        for c in "0123456789abcdefABCDEF".chars() {
            let s = c.to_string();
            assert!(validate_hex_string(&s, 1).is_ok(), "Failed for char: {}", c);
        }
    }
    
    #[test]
    fn test_robustness_all_invalid_ascii() {
        // Test a sample of invalid ASCII characters
        let invalid_chars = "ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+={}[]|\\:;\"'<>,.?/~`";
        for c in invalid_chars.chars() {
            let s = c.to_string();
            let result = validate_hex_string(&s, 1);
            assert!(
                matches!(result, Err(ValidationError::InvalidHex { .. })),
                "Should reject char: {} ({:?})", c, c
            );
        }
    }
    
    #[test]
    fn test_robustness_null_byte_positions() {
        // Null byte at every position in a 5-char string
        for i in 0..5 {
            let mut s = "abcde".to_string();
            s.replace_range(i..i+1, "\0");
            assert_eq!(
                validate_hex_string(&s, 5),
                Err(ValidationError::NullByte),
                "Failed to detect null byte at position {}", i
            );
        }
    }
    
    #[test]
    fn test_robustness_char_count_vs_byte_count() {
        // ASCII: char count == byte count
        let ascii = "abcd";
        assert_eq!(ascii.len(), ascii.chars().count());
        assert!(validate_hex_string(ascii, 4).is_ok());
        
        // UTF-8: char count < byte count
        // "café" = 5 bytes, 4 chars
        let utf8 = "café";
        assert_eq!(utf8.len(), 5); // bytes
        assert_eq!(utf8.chars().count(), 4); // chars
        // Should be rejected due to length mismatch or invalid char
        assert!(validate_hex_string(utf8, 4).is_err());
        assert!(validate_hex_string(utf8, 5).is_err());
    }
    
    #[test]
    fn test_robustness_empty_with_nonzero_expected() {
        // Empty string with non-zero expected length
        for expected in 1..10 {
            assert!(matches!(
                validate_hex_string("", expected),
                Err(ValidationError::WrongLength { actual: 0, .. })
            ));
        }
    }
}
