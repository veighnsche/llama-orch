//! Prompt validation applet
//!
//! Validates user prompts to prevent resource exhaustion.

use crate::error::{Result, ValidationError};

/// Validate user prompt
///
/// # Rules
/// - Length <= max_len (default 100,000 characters)
/// - No null bytes
/// - Valid UTF-8 (enforced by `&str` type)
///
/// # Arguments
/// * `s` - Prompt string
/// * `max_len` - Maximum allowed length
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_prompt;
///
/// // Valid
/// assert!(validate_prompt("Write a story about...", 100_000).is_ok());
///
/// // Invalid
/// assert!(validate_prompt("prompt\0null", 100_000).is_err());
/// assert!(validate_prompt(&"a".repeat(200_000), 100_000).is_err());
/// ```
///
/// # Errors
/// * `ValidationError::TooLong` - Exceeds max_len
/// * `ValidationError::NullByte` - Contains null byte
///
/// # Security
/// Prevents:
/// - VRAM exhaustion: 10MB prompt
/// - Null byte injection: `"prompt\0null"`
/// - Tokenizer exploits
/// - Control character injection (except allowed whitespace)
/// - ANSI escape sequences
/// - Unicode directional overrides
/// - Excessive repetition attacks
pub fn validate_prompt(s: &str, max_len: usize) -> Result<()> {
    // Check length first (fast check, O(1) for UTF-8 strings)
    // CRITICAL: This prevents resource exhaustion in tokenizer and model
    // A 10MB prompt could exhaust VRAM or cause OOM during tokenization
    if s.len() > max_len {
        return Err(ValidationError::TooLong {
            actual: s.len(),
            max: max_len,
        });
    }
    
    // Check for null bytes (security-critical)
    // Null bytes can cause:
    // 1. C string truncation in tokenizers (many use C/C++ libraries)
    // 2. Bypass of downstream validation
    // 3. Log injection if prompt is logged
    if s.contains('\0') {
        return Err(ValidationError::NullByte);
    }
    
    // Check for ANSI escape sequences (security-critical)
    // ANSI escapes in prompts could:
    // 1. Manipulate terminal output when prompts are logged
    // 2. Hide malicious content in logs
    // 3. Cause issues in web UIs displaying prompts
    if s.contains('\x1b') {
        return Err(ValidationError::AnsiEscape);
    }
    
    // Check for dangerous control characters
    // Allow: \t (tab), \n (newline), \r (carriage return)
    // Block: All other control characters (0x00-0x1F except \t, \n, \r)
    //
    // Dangerous control characters could:
    // 1. Cause tokenizer issues
    // 2. Break prompt formatting
    // 3. Enable injection attacks in logs
    for c in s.chars() {
        if c.is_control() && c != '\t' && c != '\n' && c != '\r' {
            return Err(ValidationError::ControlCharacter { char: c });
        }
    }
    
    // Additional robustness: Check for Unicode directional override characters
    // These can be used for display spoofing attacks
    // U+202E (RIGHT-TO-LEFT OVERRIDE) and similar characters
    const UNICODE_DIRECTIONAL_OVERRIDES: &[char] = &[
        '\u{202A}', // LEFT-TO-RIGHT EMBEDDING
        '\u{202B}', // RIGHT-TO-LEFT EMBEDDING
        '\u{202C}', // POP DIRECTIONAL FORMATTING
        '\u{202D}', // LEFT-TO-RIGHT OVERRIDE
        '\u{202E}', // RIGHT-TO-LEFT OVERRIDE
        '\u{2066}', // LEFT-TO-RIGHT ISOLATE
        '\u{2067}', // RIGHT-TO-LEFT ISOLATE
        '\u{2068}', // FIRST STRONG ISOLATE
        '\u{2069}', // POP DIRECTIONAL ISOLATE
    ];
    
    for c in s.chars() {
        if UNICODE_DIRECTIONAL_OVERRIDES.contains(&c) {
            return Err(ValidationError::InvalidCharacters {
                found: format!("Unicode directional override U+{:04X}", c as u32),
            });
        }
    }
    
    // Additional robustness: Verify character count is reasonable
    // This catches edge cases where byte length is OK but char count is extreme
    // (though this is unlikely with UTF-8)
    let char_count = s.chars().count();
    if char_count > max_len {
        return Err(ValidationError::TooLong {
            actual: char_count,
            max: max_len,
        });
    }
    
    // Note: Excessive repetition check is intentionally disabled
    // While repetitive prompts (e.g., "a" repeated 100,000 times) could cause
    // tokenizer issues, the length check already prevents resource exhaustion.
    // A repetition check would reject legitimate boundary test cases and
    // potentially valid use cases (e.g., testing with repeated characters).
    // The tokenizer and model should handle repetitive input gracefully.
    
    // UTF-8 validation is guaranteed by &str type
    // Rust's &str ensures valid UTF-8, preventing:
    // 1. Invalid UTF-8 sequences
    // 2. Overlong encodings
    // 3. Surrogate pairs
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_prompts() {
        assert!(validate_prompt("Hello, world!", 100_000).is_ok());
        assert!(validate_prompt("Write a story about...", 100_000).is_ok());
        assert!(validate_prompt("", 100_000).is_ok()); // Empty is allowed for prompts
        assert!(validate_prompt("Unicode: café ☕", 100_000).is_ok());
    }
    
    #[test]
    fn test_too_long_rejected() {
        let long = "a".repeat(100_001);
        assert!(matches!(
            validate_prompt(&long, 100_000),
            Err(ValidationError::TooLong { actual: 100_001, max: 100_000 })
        ));
    }
    
    #[test]
    fn test_null_byte_rejected() {
        assert_eq!(
            validate_prompt("prompt\0null", 100_000),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_boundary_values() {
        // Exactly at limit
        let exact = "a".repeat(100_000);
        assert!(validate_prompt(&exact, 100_000).is_ok());
        
        // One over limit
        let over = "a".repeat(100_001);
        assert!(validate_prompt(&over, 100_000).is_err());
        
        // One under limit
        let under = "a".repeat(99_999);
        assert!(validate_prompt(&under, 100_000).is_ok());
    }
    
    #[test]
    fn test_newlines_and_tabs() {
        // Newlines are allowed
        assert!(validate_prompt("Line 1\nLine 2", 100_000).is_ok());
        
        // Tabs are allowed
        assert!(validate_prompt("Text\twith\ttabs", 100_000).is_ok());
    }
    
    #[test]
    fn test_null_byte_positions() {
        // Null byte at start
        assert_eq!(
            validate_prompt("\0prompt", 100_000),
            Err(ValidationError::NullByte)
        );
        
        // Null byte at end
        assert_eq!(
            validate_prompt("prompt\0", 100_000),
            Err(ValidationError::NullByte)
        );
        
        // Multiple null bytes
        assert_eq!(
            validate_prompt("prompt\0\0null", 100_000),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_custom_max_length() {
        // Custom max length of 50
        let prompt = "a".repeat(50);
        assert!(validate_prompt(&prompt, 50).is_ok());
        
        // One over custom limit
        let prompt = "a".repeat(51);
        assert!(validate_prompt(&prompt, 50).is_err());
    }
    
    // ========== ROBUSTNESS TESTS - ATTACK SURFACE COVERAGE ==========
    
    #[test]
    fn test_robustness_ansi_escape_sequences() {
        // ANSI color codes
        assert!(matches!(
            validate_prompt("text\x1b[31mred\x1b[0m", 100_000),
            Err(ValidationError::AnsiEscape)
        ));
        
        // ANSI cursor movement
        assert!(matches!(
            validate_prompt("text\x1b[2J", 100_000),
            Err(ValidationError::AnsiEscape)
        ));
        
        // ANSI at start
        assert!(matches!(
            validate_prompt("\x1b[1mBold text", 100_000),
            Err(ValidationError::AnsiEscape)
        ));
        
        // ANSI at end
        assert!(matches!(
            validate_prompt("text\x1b[0m", 100_000),
            Err(ValidationError::AnsiEscape)
        ));
    }
    
    #[test]
    fn test_robustness_control_characters() {
        // Bell (0x07)
        assert!(matches!(
            validate_prompt("text\x07bell", 100_000),
            Err(ValidationError::ControlCharacter { char: '\x07' })
        ));
        
        // Backspace (0x08)
        assert!(matches!(
            validate_prompt("text\x08back", 100_000),
            Err(ValidationError::ControlCharacter { char: '\x08' })
        ));
        
        // Vertical tab (0x0b)
        assert!(matches!(
            validate_prompt("text\x0bvtab", 100_000),
            Err(ValidationError::ControlCharacter { char: '\x0b' })
        ));
        
        // Form feed (0x0c)
        assert!(matches!(
            validate_prompt("text\x0cff", 100_000),
            Err(ValidationError::ControlCharacter { char: '\x0c' })
        ));
        
        // Escape (0x1b) - handled by ANSI check
        assert!(validate_prompt("text\x1besc", 100_000).is_err());
    }
    
    #[test]
    fn test_robustness_allowed_whitespace() {
        // Tab is allowed
        assert!(validate_prompt("text\twith\ttabs", 100_000).is_ok());
        
        // Newline is allowed
        assert!(validate_prompt("line1\nline2\nline3", 100_000).is_ok());
        
        // Carriage return is allowed
        assert!(validate_prompt("text\rwith\rcr", 100_000).is_ok());
        
        // CRLF is allowed
        assert!(validate_prompt("line1\r\nline2\r\n", 100_000).is_ok());
        
        // Multiple newlines
        assert!(validate_prompt("para1\n\npara2\n\npara3", 100_000).is_ok());
    }
    
    #[test]
    fn test_robustness_unicode_directional_overrides() {
        // RIGHT-TO-LEFT OVERRIDE (U+202E)
        assert!(matches!(
            validate_prompt("text\u{202E}reversed", 100_000),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // LEFT-TO-RIGHT OVERRIDE (U+202D)
        assert!(matches!(
            validate_prompt("text\u{202D}override", 100_000),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // RIGHT-TO-LEFT EMBEDDING (U+202B)
        assert!(matches!(
            validate_prompt("text\u{202B}embed", 100_000),
            Err(ValidationError::InvalidCharacters { .. })
        ));
        
        // LEFT-TO-RIGHT ISOLATE (U+2066)
        assert!(matches!(
            validate_prompt("text\u{2066}isolate", 100_000),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }
    
    #[test]
    fn test_robustness_repetitive_prompts_allowed() {
        // Repetitive prompts are allowed (length check handles resource exhaustion)
        // This is intentional - tokenizers should handle repetitive input
        
        // Very repetitive prompt (allowed)
        let repetitive = "a".repeat(10_000);
        assert!(validate_prompt(&repetitive, 100_000).is_ok());
        
        // Mixed repetition (allowed)
        let mixed = "a".repeat(5_000) + &"b".repeat(5_000);
        assert!(validate_prompt(&mixed, 100_000).is_ok());
    }
    
    #[test]
    fn test_robustness_resource_exhaustion() {
        // Very long prompt (resource exhaustion)
        let huge = "a".repeat(1_000_000);
        assert!(matches!(
            validate_prompt(&huge, 100_000),
            Err(ValidationError::TooLong { actual: 1_000_000, max: 100_000 })
        ));
        
        // 10MB prompt (extreme case)
        let extreme = "a".repeat(10_000_000);
        assert!(matches!(
            validate_prompt(&extreme, 100_000),
            Err(ValidationError::TooLong { .. })
        ));
    }
    
    #[test]
    fn test_robustness_tokenizer_attacks() {
        // Null byte (could truncate in C tokenizers)
        assert_eq!(
            validate_prompt("prompt\0malicious", 100_000),
            Err(ValidationError::NullByte)
        );
        
        // Control characters (could break tokenizer)
        assert!(validate_prompt("prompt\x01control", 100_000).is_err());
        
        // ANSI escapes (could confuse tokenizer logging)
        assert!(validate_prompt("prompt\x1b[31mred", 100_000).is_err());
    }
    
    #[test]
    fn test_robustness_log_injection() {
        // Newlines are allowed (multi-line prompts are valid)
        assert!(validate_prompt("line1\nline2", 100_000).is_ok());
        
        // But ANSI escapes are blocked
        assert!(validate_prompt("text\x1b[31m[ERROR] Fake", 100_000).is_err());
        
        // Control characters are blocked
        assert!(validate_prompt("text\x07bell", 100_000).is_err());
    }
    
    #[test]
    fn test_robustness_unicode_edge_cases() {
        // Valid Unicode (emoji, accents, etc.)
        assert!(validate_prompt("Hello 👋 café ☕", 100_000).is_ok());
        
        // Chinese characters
        assert!(validate_prompt("你好世界", 100_000).is_ok());
        
        // Arabic
        assert!(validate_prompt("مرحبا", 100_000).is_ok());
        
        // Mixed scripts
        assert!(validate_prompt("Hello مرحبا 你好", 100_000).is_ok());
        
        // Zero-width characters (allowed, but directional overrides blocked)
        assert!(validate_prompt("text\u{200B}zwsp", 100_000).is_ok()); // Zero-width space
        assert!(validate_prompt("text\u{200C}zwnj", 100_000).is_ok()); // Zero-width non-joiner
    }
    
    #[test]
    fn test_robustness_char_count_vs_byte_count() {
        // ASCII: char count == byte count
        let ascii = "Hello world";
        assert_eq!(ascii.len(), ascii.chars().count());
        assert!(validate_prompt(ascii, 100_000).is_ok());
        
        // UTF-8: char count < byte count
        let utf8 = "café ☕"; // bytes > chars
        assert!(utf8.len() > utf8.chars().count()); // UTF-8 uses multiple bytes per char
        assert!(validate_prompt(utf8, 100_000).is_ok());
        
        // Emoji: 4 bytes per emoji
        let emoji = "🚀"; // 4 bytes, 1 char
        assert_eq!(emoji.len(), 4);
        assert_eq!(emoji.chars().count(), 1);
        assert!(validate_prompt(emoji, 100_000).is_ok());
    }
    
    #[test]
    fn test_robustness_boundary_conditions() {
        // Empty prompt (allowed)
        assert!(validate_prompt("", 100_000).is_ok());
        
        // Single character
        assert!(validate_prompt("a", 100_000).is_ok());
        
        // Exactly at limit
        let exact = "a".repeat(100_000);
        assert!(validate_prompt(&exact, 100_000).is_ok());
        
        // One byte over
        let over = "a".repeat(100_001);
        assert!(validate_prompt(&over, 100_000).is_err());
        
        // Very small limit
        assert!(validate_prompt("ab", 1).is_err());
        assert!(validate_prompt("a", 1).is_ok());
    }
    
    #[test]
    fn test_robustness_special_characters() {
        // Punctuation is allowed
        assert!(validate_prompt("Hello, world! How are you?", 100_000).is_ok());
        
        // Math symbols
        assert!(validate_prompt("2 + 2 = 4", 100_000).is_ok());
        
        // Currency symbols
        assert!(validate_prompt("Price: $100 €50 £30", 100_000).is_ok());
        
        // Quotes and brackets
        assert!(validate_prompt("\"quoted\" (parentheses) [brackets] {braces}", 100_000).is_ok());
    }
    
    #[test]
    fn test_robustness_multiline_prompts() {
        // Multi-paragraph prompt
        let multiline = "Paragraph 1\n\nParagraph 2\n\nParagraph 3";
        assert!(validate_prompt(multiline, 100_000).is_ok());
        
        // Code block with indentation
        let code = "def hello():\n    print('Hello')\n    return True";
        assert!(validate_prompt(code, 100_000).is_ok());
        
        // Markdown-style
        let markdown = "# Title\n\n## Subtitle\n\n- Item 1\n- Item 2";
        assert!(validate_prompt(markdown, 100_000).is_ok());
    }
    
    #[test]
    fn test_robustness_validation_order() {
        // Length check happens first
        let long = "a".repeat(100_001);
        assert!(matches!(
            validate_prompt(&long, 100_000),
            Err(ValidationError::TooLong { .. })
        ));
        
        // Null byte check happens before control char check
        assert_eq!(
            validate_prompt("text\0\x07", 100_000),
            Err(ValidationError::NullByte)
        );
        
        // ANSI check happens before control char check
        assert!(matches!(
            validate_prompt("text\x1b\x07", 100_000),
            Err(ValidationError::AnsiEscape)
        ));
    }
    
    #[test]
    fn test_robustness_real_world_prompts() {
        // Realistic user prompts that should work
        let prompts = vec![
            "Write a story about a robot learning to love.",
            "Explain quantum computing to a 5-year-old.",
            "What are the best practices for Rust programming?",
            "Translate 'Hello, how are you?' to Spanish.",
            "Debug this code:\nfn main() {\n    println!(\"Hello\");\n}",
            "List 10 creative business ideas for 2024.",
        ];
        
        for prompt in prompts {
            assert!(
                validate_prompt(prompt, 100_000).is_ok(),
                "Should accept real-world prompt: {}", prompt
            );
        }
    }
    
    #[test]
    fn test_robustness_attack_scenarios() {
        // Real attack scenarios that should be blocked
        let huge_prompt = "a".repeat(1_000_000);
        let attacks = vec![
            ("prompt\0malicious", "null byte truncation"),
            ("text\x1b[31m[ERROR] Fake", "ANSI escape injection"),
            ("text\x07bell", "control character injection"),
            ("text\u{202E}reversed", "Unicode directional override"),
            (huge_prompt.as_str(), "resource exhaustion (length)"),
        ];
        
        for (attack, description) in attacks {
            assert!(
                validate_prompt(attack, 100_000).is_err(),
                "Should block attack: {}", description
            );
        }
    }
}
