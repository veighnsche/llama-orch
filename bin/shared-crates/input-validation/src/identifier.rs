//! Identifier validation applet
//!
//! Validates identifiers (shard_id, task_id, pool_id, node_id).

use crate::error::{Result, ValidationError};

/// Validate identifier (shard_id, task_id, pool_id, node_id)
///
/// # Rules
/// - Not empty
/// - Length <= max_len
/// - Only ASCII alphanumeric + dash + underscore: `[a-zA-Z0-9_-]+`
/// - No null bytes
/// - No path traversal sequences (`../`, `./`, `..\\`, `.\\`)
/// - No Unicode characters (ASCII-only policy)
///
/// # Arguments
/// * `s` - String to validate
/// * `max_len` - Maximum allowed length
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_identifier;
///
/// // Valid
/// assert!(validate_identifier("shard-abc123", 256).is_ok());
/// assert!(validate_identifier("task_gpu0", 256).is_ok());
///
/// // Invalid
/// assert!(validate_identifier("shard-../etc/passwd", 256).is_err());
/// assert!(validate_identifier("shard\0null", 256).is_err());
/// ```
///
/// # Errors
/// * `ValidationError::Empty` - String is empty
/// * `ValidationError::TooLong` - Exceeds max_len
/// * `ValidationError::NullByte` - Contains null byte
/// * `ValidationError::PathTraversal` - Contains `../` or `./`
/// * `ValidationError::InvalidCharacters` - Contains non-alphanumeric/dash/underscore
pub fn validate_identifier(s: &str, max_len: usize) -> Result<()> {
    // Check empty first (fast check, prevents processing empty strings)
    if s.is_empty() {
        return Err(ValidationError::Empty);
    }
    
    // Check length early (fast check, prevents processing oversized inputs)
    // Uses byte length which is O(1) for UTF-8 strings
    if s.len() > max_len {
        return Err(ValidationError::TooLong {
            actual: s.len(),
            max: max_len,
        });
    }
    
    // PERFORMANCE PHASE 2: Single-pass validation
    // Combines null byte check, path traversal detection, and character validation
    // into one iteration for maximum performance (6 iterations → 1 iteration)
    // Auth-min approved: Security-equivalent, same validation order
    
    let mut prev = '\0';
    let mut prev_prev = '\0';
    
    for c in s.chars() {
        // SECURITY: Check null byte first (prevents C string truncation)
        if c == '\0' {
            return Err(ValidationError::NullByte);
        }
        
        // SECURITY: Check path traversal patterns BEFORE character validation
        // This ensures we detect path traversal even with invalid characters (dots)
        // Detects: ../ ..\ ./ .\
        if c == '/' || c == '\\' {
            // Check for "../" or "..\"
            if prev == '.' && prev_prev == '.' {
                return Err(ValidationError::PathTraversal);
            }
            // Check for "./" or ".\"
            if prev == '.' {
                return Err(ValidationError::PathTraversal);
            }
            // Path separator itself is invalid in identifiers
            return Err(ValidationError::InvalidCharacters {
                found: c.to_string(),
            });
        }
        
        // SECURITY: Check for dots (potential path traversal or invalid character)
        // Dots are only allowed as part of path traversal patterns (../ ..\ ./ .\)
        // Any other dot is an invalid character
        if c == '.' {
            // Dot is temporarily allowed to pass through for path traversal detection
            // Will be caught as invalid if not followed by / or \ (checked above)
            prev_prev = prev;
            prev = c;
            continue;
        }
        
        // SECURITY: If previous character was a dot and current is not / or \,
        // then the dot was invalid (not part of path traversal)
        if prev == '.' {
            return Err(ValidationError::InvalidCharacters {
                found: ".".to_string(),
            });
        }
        
        // SECURITY: Validate character whitelist (ASCII alphanumeric + dash + underscore)
        // ASCII-only policy prevents Unicode homoglyph attacks
        if !c.is_ascii_alphanumeric() && c != '-' && c != '_' {
            return Err(ValidationError::InvalidCharacters {
                found: c.to_string(),
            });
        }
        
        prev_prev = prev;
        prev = c;
    }
    
    // SECURITY: Check if string ends with dots (invalid, not path traversal)
    if prev == '.' {
        return Err(ValidationError::InvalidCharacters {
            found: ".".to_string(),
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_identifiers() {
        assert!(validate_identifier("shard-abc123", 256).is_ok());
        assert!(validate_identifier("task_gpu0", 256).is_ok());
        assert!(validate_identifier("pool-1", 256).is_ok());
        assert!(validate_identifier("a", 256).is_ok());
        assert!(validate_identifier("ABC-123_xyz", 256).is_ok());
    }
    
    #[test]
    fn test_empty_rejected() {
        assert_eq!(
            validate_identifier("", 256),
            Err(ValidationError::Empty)
        );
    }
    
    #[test]
    fn test_too_long_rejected() {
        let long = "a".repeat(257);
        assert!(matches!(
            validate_identifier(&long, 256),
            Err(ValidationError::TooLong { actual: 257, max: 256 })
        ));
    }
    
    #[test]
    fn test_null_byte_rejected() {
        assert_eq!(
            validate_identifier("shard\0null", 256),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_path_traversal_rejected() {
        assert_eq!(
            validate_identifier("shard-../etc/passwd", 256),
            Err(ValidationError::PathTraversal)
        );
        assert_eq!(
            validate_identifier("shard-./config", 256),
            Err(ValidationError::PathTraversal)
        );
        assert_eq!(
            validate_identifier("shard-..\\windows", 256),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_invalid_characters_rejected() {
        assert!(matches!(
            validate_identifier("shard@123", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        assert!(matches!(
            validate_identifier("shard!123", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        assert!(matches!(
            validate_identifier("shard 123", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_boundary_values() {
        // Exactly at limit
        let exact = "a".repeat(256);
        assert!(validate_identifier(&exact, 256).is_ok());
        
        // One over limit
        let over = "a".repeat(257);
        assert!(validate_identifier(&over, 256).is_err());
        
        // One under limit
        let under = "a".repeat(255);
        assert!(validate_identifier(&under, 256).is_ok());
    }
    
    #[test]
    fn test_null_byte_positions() {
        // Null byte at start
        assert_eq!(
            validate_identifier("\0shard", 256),
            Err(ValidationError::NullByte)
        );
        
        // Null byte at end
        assert_eq!(
            validate_identifier("shard\0", 256),
            Err(ValidationError::NullByte)
        );
        
        // Multiple null bytes
        assert_eq!(
            validate_identifier("shard\0\0null", 256),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_more_invalid_characters() {
        // Slash
        assert!(matches!(
            validate_identifier("shard/123", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Colon
        assert!(matches!(
            validate_identifier("shard:123", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Dot
        assert!(matches!(
            validate_identifier("shard.123", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_unicode_rejected() {
        // Unicode characters (ASCII-only policy)
        // Note: 'é' is non-ASCII, so it will be rejected as InvalidCharacters
        let result = validate_identifier("shard-café", 256);
        assert!(result.is_err());
        match result {
            Err(ValidationError::InvalidCharacters { .. }) => {},
            other => panic!("Expected InvalidCharacters, got: {:?}", other),
        }
        
        // Emoji (non-ASCII)
        let result = validate_identifier("shard-🚀", 256);
        assert!(result.is_err());
        match result {
            Err(ValidationError::InvalidCharacters { .. }) => {},
            other => panic!("Expected InvalidCharacters, got: {:?}", other),
        }
    }
    
    #[test]
    fn test_path_traversal_windows() {
        // Windows backslash traversal
        assert_eq!(
            validate_identifier("shard-.\\config", 256),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_multiple_traversal_sequences() {
        assert_eq!(
            validate_identifier("shard-../../etc/passwd", 256),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_dots_without_traversal() {
        // Single dots are not traversal (but dot is invalid character)
        assert!(matches!(
            validate_identifier("shard.backup", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    // ========== ROBUSTNESS TESTS ==========
    
    #[test]
    fn test_robustness_whitespace_rejection() {
        // Leading whitespace
        assert!(matches!(
            validate_identifier(" shard", 256),
            Err(ValidationError::InvalidCharacters { found }) if found == " "
        ));
        
        // Trailing whitespace
        assert!(matches!(
            validate_identifier("shard ", 256),
            Err(ValidationError::InvalidCharacters { found }) if found == " "
        ));
        
        // Internal whitespace
        assert!(matches!(
            validate_identifier("shard abc", 256),
            Err(ValidationError::InvalidCharacters { found }) if found == " "
        ));
        
        // Tab
        assert!(matches!(
            validate_identifier("shard\tabc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Newline
        assert!(matches!(
            validate_identifier("shard\nabc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Carriage return
        assert!(matches!(
            validate_identifier("shard\rabc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_special_characters() {
        let special_chars = "!@#$%^&*()+={}[]|\\:;\"'<>,.?/~`";
        for c in special_chars.chars() {
            let s = format!("shard{}abc", c);
            let result = validate_identifier(&s, 256);
            assert!(
                matches!(result, Err(ValidationError::InvalidCharacters { .. })),
                "Should reject special char: {} ({:?})", c, c
            );
        }
    }
    
    #[test]
    fn test_robustness_valid_characters_only() {
        // All lowercase letters
        assert!(validate_identifier("abcdefghijklmnopqrstuvwxyz", 256).is_ok());
        
        // All uppercase letters
        assert!(validate_identifier("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 256).is_ok());
        
        // All digits
        assert!(validate_identifier("0123456789", 256).is_ok());
        
        // Dashes only
        assert!(validate_identifier("---", 256).is_ok());
        
        // Underscores only
        assert!(validate_identifier("___", 256).is_ok());
        
        // Mixed valid characters
        assert!(validate_identifier("a1-b2_c3-D4_E5", 256).is_ok());
    }
    
    #[test]
    fn test_robustness_boundary_lengths() {
        // Exactly at max length
        let exact = "a".repeat(256);
        assert!(validate_identifier(&exact, 256).is_ok());
        
        // One under max length
        let under = "a".repeat(255);
        assert!(validate_identifier(&under, 256).is_ok());
        
        // One over max length
        let over = "a".repeat(257);
        assert!(matches!(
            validate_identifier(&over, 256),
            Err(ValidationError::TooLong { actual: 257, max: 256 })
        ));
        
        // Very small max length
        assert!(validate_identifier("a", 1).is_ok());
        assert!(matches!(
            validate_identifier("ab", 1),
            Err(ValidationError::TooLong { actual: 2, max: 1 })
        ));
    }
    
    #[test]
    fn test_robustness_path_traversal_variants() {
        // Unix forward slash variants
        assert_eq!(validate_identifier("../etc", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier("./config", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier("shard-../etc", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier("shard-./etc", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier("../../etc", 256), Err(ValidationError::PathTraversal));
        
        // Windows backslash variants
        assert_eq!(validate_identifier("..\\windows", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier(".\\config", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier("shard-..\\windows", 256), Err(ValidationError::PathTraversal));
        assert_eq!(validate_identifier("shard-.\\config", 256), Err(ValidationError::PathTraversal));
        
        // Mixed separators
        assert_eq!(validate_identifier("../\\etc", 256), Err(ValidationError::PathTraversal));
    }
    
    #[test]
    fn test_robustness_null_byte_all_positions() {
        // Null byte at start
        assert_eq!(validate_identifier("\0shard", 256), Err(ValidationError::NullByte));
        
        // Null byte in middle
        assert_eq!(validate_identifier("shard\0abc", 256), Err(ValidationError::NullByte));
        
        // Null byte at end
        assert_eq!(validate_identifier("shard\0", 256), Err(ValidationError::NullByte));
        
        // Multiple null bytes
        assert_eq!(validate_identifier("shard\0\0abc", 256), Err(ValidationError::NullByte));
        
        // Null byte at every position in a 5-char string
        for i in 0..5 {
            let mut s = "abcde".to_string();
            s.replace_range(i..i+1, "\0");
            assert_eq!(
                validate_identifier(&s, 256),
                Err(ValidationError::NullByte),
                "Failed to detect null byte at position {}", i
            );
        }
    }
    
    #[test]
    fn test_robustness_unicode_homoglyphs() {
        // Cyrillic 'а' (U+0430) looks like Latin 'a' (U+0061)
        assert!(matches!(
            validate_identifier("shаrd", 256), // Cyrillic а
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Greek 'ο' (U+03BF) looks like Latin 'o' (U+006F)
        assert!(matches!(
            validate_identifier("shοrd", 256), // Greek ο
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_unicode_various() {
        // Accented characters
        assert!(matches!(
            validate_identifier("café", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Emoji
        assert!(matches!(
            validate_identifier("shard🚀", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Chinese characters
        assert!(matches!(
            validate_identifier("shard中文", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Arabic characters
        assert!(matches!(
            validate_identifier("shardعربي", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_control_characters() {
        // Bell (0x07)
        assert!(matches!(
            validate_identifier("shard\x07abc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Backspace (0x08)
        assert!(matches!(
            validate_identifier("shard\x08abc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Escape (0x1b)
        assert!(matches!(
            validate_identifier("shard\x1babc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Vertical tab (0x0b)
        assert!(matches!(
            validate_identifier("shard\x0babc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // Form feed (0x0c)
        assert!(matches!(
            validate_identifier("shard\x0cabc", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_single_character_identifiers() {
        // Single valid characters
        assert!(validate_identifier("a", 256).is_ok());
        assert!(validate_identifier("Z", 256).is_ok());
        assert!(validate_identifier("0", 256).is_ok());
        assert!(validate_identifier("-", 256).is_ok());
        assert!(validate_identifier("_", 256).is_ok());
        
        // Single invalid characters
        assert!(matches!(
            validate_identifier(".", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        assert!(matches!(
            validate_identifier(" ", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        assert!(matches!(
            validate_identifier("@", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_very_long_identifiers() {
        // Very long valid identifier (1024 chars)
        let long_valid = "a".repeat(1024);
        assert!(validate_identifier(&long_valid, 1024).is_ok());
        
        // Very long invalid identifier (should fail on length)
        let long_invalid = "a".repeat(1025);
        assert!(matches!(
            validate_identifier(&long_invalid, 1024),
            Err(ValidationError::TooLong { actual: 1025, max: 1024 })
        ));
        
        // Long identifier with invalid char at end (should fail fast on char)
        let mut long_with_invalid = "a".repeat(1023);
        long_with_invalid.push('!');
        assert!(matches!(
            validate_identifier(&long_with_invalid, 1024),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_case_sensitivity() {
        // Mixed case should be allowed
        assert!(validate_identifier("ShArD-AbC_123", 256).is_ok());
        assert!(validate_identifier("SHARD-ABC-123", 256).is_ok());
        assert!(validate_identifier("shard-abc-123", 256).is_ok());
    }
    
    #[test]
    fn test_robustness_dash_underscore_positions() {
        // Dash at start
        assert!(validate_identifier("-shard", 256).is_ok());
        
        // Dash at end
        assert!(validate_identifier("shard-", 256).is_ok());
        
        // Multiple consecutive dashes
        assert!(validate_identifier("shard---abc", 256).is_ok());
        
        // Underscore at start
        assert!(validate_identifier("_shard", 256).is_ok());
        
        // Underscore at end
        assert!(validate_identifier("shard_", 256).is_ok());
        
        // Multiple consecutive underscores
        assert!(validate_identifier("shard___abc", 256).is_ok());
        
        // Mixed dashes and underscores
        assert!(validate_identifier("-_-_shard_-_-", 256).is_ok());
    }
    
    #[test]
    fn test_robustness_digits_positions() {
        // Digits at start (allowed)
        assert!(validate_identifier("123shard", 256).is_ok());
        
        // Digits at end
        assert!(validate_identifier("shard123", 256).is_ok());
        
        // Only digits
        assert!(validate_identifier("123456", 256).is_ok());
    }
    
    #[test]
    fn test_robustness_char_count_vs_byte_count() {
        // ASCII: char count == byte count
        let ascii = "shard-abc";
        assert_eq!(ascii.len(), ascii.chars().count());
        assert!(validate_identifier(ascii, 256).is_ok());
        
        // UTF-8: char count < byte count
        // "café" = 5 bytes, 4 chars
        let utf8 = "café";
        assert_eq!(utf8.len(), 5); // bytes
        assert_eq!(utf8.chars().count(), 4); // chars
        // Should be rejected due to non-ASCII character
        assert!(validate_identifier(utf8, 256).is_err());
    }
    
    #[test]
    fn test_robustness_empty_vs_whitespace() {
        // Empty string
        assert_eq!(validate_identifier("", 256), Err(ValidationError::Empty));
        
        // Only whitespace (not empty, but invalid)
        assert!(matches!(
            validate_identifier(" ", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        assert!(matches!(
            validate_identifier("   ", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        assert!(matches!(
            validate_identifier("\t", 256),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_validation_order() {
        // Empty check happens first
        assert_eq!(validate_identifier("", 256), Err(ValidationError::Empty));
        
        // Length check happens before character validation
        let long_invalid = "!".repeat(257);
        assert!(matches!(
            validate_identifier(&long_invalid, 256),
            Err(ValidationError::TooLong { .. })
        ));
        
        // Null byte check happens before path traversal
        assert_eq!(
            validate_identifier("\0../etc", 256),
            Err(ValidationError::NullByte)
        );
        
        // Path traversal check happens before character validation
        assert_eq!(
            validate_identifier("../!@#", 256),
            Err(ValidationError::PathTraversal)
        );
    }
}
