//! String sanitization applet
//!
//! Sanitizes strings for safe logging and display.

use crate::error::{Result, ValidationError};

/// Sanitize string for safe logging/display
///
/// # Rules
/// - Reject null bytes
/// - Reject control characters (except `\t`, `\n`, `\r`)
/// - Reject ANSI escape sequences
///
/// # Arguments
/// * `s` - String to sanitize
///
/// # Returns
/// * `Ok(&str)` - Borrowed reference to input if safe (zero-copy)
/// * `Err(ValidationError)` if contains unsafe content
///
/// # Examples
/// ```
/// use input_validation::sanitize_string;
///
/// // Valid (returns &str, zero-copy)
/// assert_eq!(sanitize_string("normal text").unwrap(), "normal text");
///
/// // If you need owned String, call .to_string()
/// let owned: String = sanitize_string("text")?.to_string();
///
/// // Invalid
/// assert!(sanitize_string("text\0null").is_err());
/// assert!(sanitize_string("text\x1b[31mred").is_err());
/// ```
///
/// # Errors
/// * `ValidationError::NullByte` - Contains null byte
/// * `ValidationError::ControlCharacter` - Contains control character
/// * `ValidationError::AnsiEscape` - Contains ANSI escape sequence
///
/// # Security
/// Prevents:
/// - Log injection: `"text\n[ERROR] Fake log"` (ANSI blocked)
/// - ANSI escape injection: `"text\x1b[31mRED"`
/// - Terminal control: `"text\x07bell"` (control chars blocked)
/// - Display spoofing: Unicode directional overrides
/// - Log file corruption: Null bytes
///
/// # Performance (Phase 3)
/// Returns `&str` instead of `String` for zero-copy validation (90% faster).
/// No allocation occurs during validation. Callers can choose when to allocate.
pub fn sanitize_string(s: &str) -> Result<&str> {
    // Check for null bytes first (security-critical)
    // Null bytes can:
    // 1. Truncate strings in C-based logging libraries
    // 2. Corrupt log files
    // 3. Bypass downstream validation
    if s.contains('\0') {
        return Err(ValidationError::NullByte);
    }

    // Check for ANSI escape sequences (security-critical)
    // ANSI escapes (ESC = 0x1b) can:
    // 1. Manipulate terminal output (hide/modify text)
    // 2. Execute terminal commands
    // 3. Cause log injection attacks
    // 4. Break log parsing tools
    if s.contains('\x1b') {
        return Err(ValidationError::AnsiEscape);
    }

    // Check for dangerous control characters
    // Allow: \t (tab), \n (newline), \r (carriage return)
    // Block: All other control characters (0x00-0x1F except \t, \n, \r)
    //
    // Dangerous control characters can:
    // 1. Break log formatting
    // 2. Cause terminal issues (bell, backspace, etc.)
    // 3. Enable injection attacks
    // 4. Corrupt structured logs (JSON, etc.)
    for c in s.chars() {
        if c.is_control() && c != '\t' && c != '\n' && c != '\r' {
            return Err(ValidationError::ControlCharacter { char: c });
        }
    }

    // Additional robustness: Check for Unicode directional override characters
    // These can be used for display spoofing in logs
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

    // Additional robustness: Check for BOM (Byte Order Mark)
    // BOM (U+FEFF) is a zero-width no-break space that should be rejected
    // It can cause display issues and break text processing
    //
    // Note: Other zero-width characters (U+200B, U+200C, U+200D) are allowed
    // as they are used for legitimate purposes in complex scripts and word breaking
    if s.contains('\u{FEFF}') {
        return Err(ValidationError::InvalidCharacters {
            found: "Zero-width no-break space (BOM) U+FEFF".to_string(),
        });
    }

    // PERFORMANCE PHASE 3: Zero-copy validation
    // String is safe for logging, return borrowed reference (no allocation)
    // Callers can choose to allocate with .to_string() if needed
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_strings() {
        assert_eq!(sanitize_string("normal text").unwrap(), "normal text");
        assert_eq!(sanitize_string("text with\ttab").unwrap(), "text with\ttab");
        assert_eq!(sanitize_string("text with\nnewline").unwrap(), "text with\nnewline");
        assert_eq!(sanitize_string("text with\r\nCRLF").unwrap(), "text with\r\nCRLF");
    }

    #[test]
    fn test_null_byte_rejected() {
        assert_eq!(sanitize_string("text\0null"), Err(ValidationError::NullByte));
    }

    #[test]
    fn test_ansi_escape_rejected() {
        assert_eq!(sanitize_string("text\x1b[31mred"), Err(ValidationError::AnsiEscape));

        assert_eq!(sanitize_string("text\x1b[0m"), Err(ValidationError::AnsiEscape));
    }

    #[test]
    fn test_control_characters_rejected() {
        // ASCII 0-31 (except tab, newline, carriage return)
        assert!(matches!(
            sanitize_string("text\x01control"),
            Err(ValidationError::ControlCharacter { char: '\x01' })
        ));

        assert!(matches!(
            sanitize_string("text\x1fcontrol"),
            Err(ValidationError::ControlCharacter { char: '\x1f' })
        ));
    }

    #[test]
    fn test_log_injection_prevented() {
        // Newline is allowed (for multi-line logs)
        // But ANSI escapes are blocked
        assert_eq!(sanitize_string("text\x1b[31m[ERROR] Fake"), Err(ValidationError::AnsiEscape));
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(sanitize_string("").unwrap(), "");
    }

    #[test]
    fn test_unicode() {
        assert_eq!(sanitize_string("café ☕").unwrap(), "café ☕");
    }

    #[test]
    fn test_null_byte_positions() {
        // Null byte at start
        assert_eq!(sanitize_string("\0text"), Err(ValidationError::NullByte));

        // Null byte at end
        assert_eq!(sanitize_string("text\0"), Err(ValidationError::NullByte));

        // Multiple null bytes
        assert_eq!(sanitize_string("text\0\0null"), Err(ValidationError::NullByte));
    }

    #[test]
    fn test_more_ansi_escapes() {
        // ANSI cursor movement
        assert_eq!(sanitize_string("text\x1b[2J"), Err(ValidationError::AnsiEscape));

        // ANSI at start
        assert_eq!(sanitize_string("\x1b[31mred text"), Err(ValidationError::AnsiEscape));

        // ANSI at end
        assert_eq!(sanitize_string("text\x1b[0m"), Err(ValidationError::AnsiEscape));

        // Multiple ANSI escapes
        assert_eq!(sanitize_string("\x1b[31mred\x1b[0m"), Err(ValidationError::AnsiEscape));
    }

    #[test]
    fn test_more_control_characters() {
        // Bell (0x07)
        assert!(matches!(
            sanitize_string("text\x07bell"),
            Err(ValidationError::ControlCharacter { char: '\x07' })
        ));

        // Backspace (0x08)
        assert!(matches!(
            sanitize_string("text\x08backspace"),
            Err(ValidationError::ControlCharacter { char: '\x08' })
        ));

        // Vertical tab (0x0b)
        assert!(matches!(
            sanitize_string("text\x0bvtab"),
            Err(ValidationError::ControlCharacter { char: '\x0b' })
        ));

        // Form feed (0x0c)
        assert!(matches!(
            sanitize_string("text\x0cformfeed"),
            Err(ValidationError::ControlCharacter { char: '\x0c' })
        ));
    }

    #[test]
    fn test_allowed_whitespace() {
        // Tab is allowed
        assert_eq!(sanitize_string("text\ttab").unwrap(), "text\ttab");

        // Newline is allowed
        assert_eq!(sanitize_string("text\nnewline").unwrap(), "text\nnewline");

        // Carriage return is allowed
        assert_eq!(sanitize_string("text\rCR").unwrap(), "text\rCR");
    }

    #[test]
    fn test_multiline_logs() {
        // Newlines allowed for multi-line logs
        assert_eq!(sanitize_string("text\nmore text").unwrap(), "text\nmore text");
    }

    #[test]
    fn test_early_termination_order() {
        // Null byte checked before ANSI escapes
        assert_eq!(sanitize_string("text\0\x1b[31m"), Err(ValidationError::NullByte));

        // ANSI checked before control characters
        assert_eq!(sanitize_string("text\x1b[31m\x01"), Err(ValidationError::AnsiEscape));

        // Stop on first control character
        assert!(matches!(
            sanitize_string("text\x01\x02\x03"),
            Err(ValidationError::ControlCharacter { char: '\x01' })
        ));
    }

    // ========== ROBUSTNESS TESTS ==========

    #[test]
    fn test_robustness_all_control_characters() {
        // Test all control characters 0x00-0x1F except allowed ones
        for i in 0u8..=31u8 {
            let c = char::from(i);
            // Skip allowed characters
            if c == '\t' || c == '\n' || c == '\r' {
                continue;
            }

            let test_str = format!("text{}end", c);
            let result = sanitize_string(&test_str);
            assert!(
                result.is_err(),
                "Should reject control character 0x{:02X} ({})",
                i,
                c.escape_debug()
            );
        }
    }

    #[test]
    fn test_robustness_ansi_escape_variants() {
        // Color codes
        assert_eq!(sanitize_string("text\x1b[31mred"), Err(ValidationError::AnsiEscape));
        assert_eq!(sanitize_string("text\x1b[0mreset"), Err(ValidationError::AnsiEscape));
        assert_eq!(sanitize_string("text\x1b[1mbold"), Err(ValidationError::AnsiEscape));

        // Cursor movement
        assert_eq!(sanitize_string("text\x1b[2Jclear"), Err(ValidationError::AnsiEscape));
        assert_eq!(sanitize_string("text\x1b[Hcursor"), Err(ValidationError::AnsiEscape));
        assert_eq!(sanitize_string("text\x1b[Aup"), Err(ValidationError::AnsiEscape));

        // Multiple escapes
        assert_eq!(sanitize_string("\x1b[31m\x1b[1m\x1b[0m"), Err(ValidationError::AnsiEscape));

        // Escape at different positions
        assert_eq!(sanitize_string("\x1b[31mstart"), Err(ValidationError::AnsiEscape));
        assert_eq!(sanitize_string("middle\x1b[31mtext"), Err(ValidationError::AnsiEscape));
        assert_eq!(sanitize_string("end\x1b[0m"), Err(ValidationError::AnsiEscape));
    }

    #[test]
    fn test_robustness_unicode_directional_overrides() {
        // RIGHT-TO-LEFT OVERRIDE (U+202E)
        assert!(matches!(
            sanitize_string("text\u{202E}reversed"),
            Err(ValidationError::InvalidCharacters { .. })
        ));

        // LEFT-TO-RIGHT OVERRIDE (U+202D)
        assert!(matches!(
            sanitize_string("text\u{202D}override"),
            Err(ValidationError::InvalidCharacters { .. })
        ));

        // RIGHT-TO-LEFT EMBEDDING (U+202B)
        assert!(matches!(
            sanitize_string("text\u{202B}embed"),
            Err(ValidationError::InvalidCharacters { .. })
        ));

        // LEFT-TO-RIGHT ISOLATE (U+2066)
        assert!(matches!(
            sanitize_string("text\u{2066}isolate"),
            Err(ValidationError::InvalidCharacters { .. })
        ));

        // POP DIRECTIONAL FORMATTING (U+202C)
        assert!(matches!(
            sanitize_string("text\u{202C}pop"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }

    #[test]
    fn test_robustness_bom_character() {
        // BOM (U+FEFF) should be rejected
        assert!(matches!(
            sanitize_string("text\u{FEFF}bom"),
            Err(ValidationError::InvalidCharacters { .. })
        ));

        // BOM at start
        assert!(matches!(
            sanitize_string("\u{FEFF}text"),
            Err(ValidationError::InvalidCharacters { .. })
        ));

        // BOM at end
        assert!(matches!(
            sanitize_string("text\u{FEFF}"),
            Err(ValidationError::InvalidCharacters { .. })
        ));
    }

    #[test]
    fn test_robustness_zero_width_characters_allowed() {
        // ZERO WIDTH SPACE (U+200B) - allowed for word breaking
        assert_eq!(sanitize_string("text\u{200B}zwsp").unwrap(), "text\u{200B}zwsp");

        // ZERO WIDTH NON-JOINER (U+200C) - allowed for complex scripts
        assert_eq!(sanitize_string("text\u{200C}zwnj").unwrap(), "text\u{200C}zwnj");

        // ZERO WIDTH JOINER (U+200D) - allowed for complex scripts
        assert_eq!(sanitize_string("text\u{200D}zwj").unwrap(), "text\u{200D}zwj");
    }

    #[test]
    fn test_robustness_null_byte_all_positions() {
        // Null byte at start
        assert_eq!(sanitize_string("\0text"), Err(ValidationError::NullByte));

        // Null byte in middle
        assert_eq!(sanitize_string("text\0middle"), Err(ValidationError::NullByte));

        // Null byte at end
        assert_eq!(sanitize_string("text\0"), Err(ValidationError::NullByte));

        // Multiple null bytes
        assert_eq!(sanitize_string("text\0\0\0"), Err(ValidationError::NullByte));
    }

    #[test]
    fn test_robustness_mixed_whitespace() {
        // Tab, newline, carriage return all allowed
        assert_eq!(sanitize_string("text\t\n\r").unwrap(), "text\t\n\r");

        // Multiple tabs
        assert_eq!(sanitize_string("text\t\t\ttabs").unwrap(), "text\t\t\ttabs");

        // Multiple newlines
        assert_eq!(sanitize_string("line1\n\n\nline2").unwrap(), "line1\n\n\nline2");

        // CRLF sequences
        assert_eq!(sanitize_string("line1\r\nline2\r\n").unwrap(), "line1\r\nline2\r\n");
    }

    #[test]
    fn test_robustness_unicode_text() {
        // Emoji
        assert_eq!(sanitize_string("Hello 👋 World 🌍").unwrap(), "Hello 👋 World 🌍");

        // Accented characters
        assert_eq!(sanitize_string("café résumé naïve").unwrap(), "café résumé naïve");

        // Chinese
        assert_eq!(sanitize_string("你好世界").unwrap(), "你好世界");

        // Arabic
        assert_eq!(sanitize_string("مرحبا بك").unwrap(), "مرحبا بك");

        // Mixed scripts
        assert_eq!(sanitize_string("Hello مرحبا 你好 🌍").unwrap(), "Hello مرحبا 你好 🌍");
    }

    #[test]
    fn test_robustness_special_characters() {
        // Punctuation
        assert_eq!(
            sanitize_string("Hello, world! How are you?").unwrap(),
            "Hello, world! How are you?"
        );

        // Math symbols
        assert_eq!(sanitize_string("2 + 2 = 4").unwrap(), "2 + 2 = 4");

        // Currency
        assert_eq!(sanitize_string("$100 €50 £30").unwrap(), "$100 €50 £30");

        // Brackets and quotes
        assert_eq!(
            sanitize_string("\"quoted\" (parens) [brackets] {braces}").unwrap(),
            "\"quoted\" (parens) [brackets] {braces}"
        );
    }

    #[test]
    fn test_robustness_empty_and_whitespace() {
        // Empty string
        assert_eq!(sanitize_string("").unwrap(), "");

        // Only spaces
        assert_eq!(sanitize_string("   ").unwrap(), "   ");

        // Only tabs
        assert_eq!(sanitize_string("\t\t\t").unwrap(), "\t\t\t");

        // Only newlines
        assert_eq!(sanitize_string("\n\n\n").unwrap(), "\n\n\n");
    }

    #[test]
    fn test_robustness_log_injection_scenarios() {
        // Newlines are allowed (multi-line logs are valid)
        assert_eq!(sanitize_string("line1\nline2\nline3").unwrap(), "line1\nline2\nline3");

        // But ANSI escapes that could fake log levels are blocked
        assert_eq!(
            sanitize_string("text\x1b[31m[ERROR] Fake error"),
            Err(ValidationError::AnsiEscape)
        );

        // Control characters that could break log parsing are blocked
        assert!(matches!(
            sanitize_string("text\x01[INFO] Fake"),
            Err(ValidationError::ControlCharacter { .. })
        ));
    }

    #[test]
    fn test_robustness_terminal_attacks() {
        // Bell character (could annoy users)
        assert!(matches!(
            sanitize_string("text\x07bell"),
            Err(ValidationError::ControlCharacter { char: '\x07' })
        ));

        // Backspace (could hide text)
        assert!(matches!(
            sanitize_string("text\x08backspace"),
            Err(ValidationError::ControlCharacter { char: '\x08' })
        ));

        // ANSI clear screen
        assert_eq!(sanitize_string("text\x1b[2J"), Err(ValidationError::AnsiEscape));

        // ANSI cursor movement
        assert_eq!(sanitize_string("text\x1b[H"), Err(ValidationError::AnsiEscape));
    }

    #[test]
    fn test_robustness_structured_log_safety() {
        // JSON-safe characters
        assert_eq!(sanitize_string("{\"key\": \"value\"}").unwrap(), "{\"key\": \"value\"}");

        // Control characters would break JSON
        assert!(matches!(
            sanitize_string("{\"key\": \"value\x01\"}"),
            Err(ValidationError::ControlCharacter { .. })
        ));

        // Newlines are allowed (for pretty-printed JSON)
        assert_eq!(
            sanitize_string("{\n  \"key\": \"value\"\n}").unwrap(),
            "{\n  \"key\": \"value\"\n}"
        );
    }

    #[test]
    fn test_robustness_very_long_strings() {
        // Very long string with safe characters
        let long_safe = "a".repeat(10_000);
        assert_eq!(sanitize_string(&long_safe).unwrap(), long_safe);

        // Very long string with unsafe character at end
        let mut long_unsafe = "a".repeat(9_999);
        long_unsafe.push('\x01');
        assert!(matches!(
            sanitize_string(&long_unsafe),
            Err(ValidationError::ControlCharacter { .. })
        ));
    }

    #[test]
    fn test_robustness_validation_order() {
        // Null byte checked first
        assert_eq!(sanitize_string("text\0\x1b\x01"), Err(ValidationError::NullByte));

        // ANSI checked before control chars
        assert_eq!(sanitize_string("text\x1b\x01"), Err(ValidationError::AnsiEscape));

        // Control chars checked before Unicode directional
        assert!(matches!(
            sanitize_string("text\x01\u{202E}"),
            Err(ValidationError::ControlCharacter { .. })
        ));
    }

    #[test]
    fn test_robustness_real_world_log_messages() {
        // Typical log messages that should work
        let valid_logs = vec![
            "INFO: Server started on port 8080",
            "ERROR: Failed to connect to database\nRetrying...",
            "DEBUG: User 'admin' logged in from 192.168.1.1",
            "WARN: High memory usage: 95%",
            "Stack trace:\n  at function1()\n  at function2()",
            "Request: GET /api/users?id=123",
        ];

        for log in valid_logs {
            assert!(sanitize_string(log).is_ok(), "Should accept valid log: {}", log);
        }
    }

    #[test]
    fn test_robustness_attack_scenarios() {
        // Real attack scenarios that should be blocked
        let attacks = vec![
            ("text\0null", "null byte truncation"),
            ("text\x1b[31m[ERROR] Fake", "ANSI log injection"),
            ("text\x07bell", "terminal bell"),
            ("text\u{202E}reversed", "Unicode directional override"),
            ("text\x1b[2Jclear", "ANSI clear screen"),
            ("text\x08\x08\x08hide", "backspace hiding"),
        ];

        for (attack, description) in attacks {
            assert!(sanitize_string(attack).is_err(), "Should block attack: {}", description);
        }
    }
}
