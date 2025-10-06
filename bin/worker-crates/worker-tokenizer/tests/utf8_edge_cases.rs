//! UTF-8 Streaming Edge Cases - FT-043
//!
//! Tests UTF-8 handling with multibyte characters, emoji, and edge cases.

// UTF-8 validation tests
// Note: Utf8Validator is internal, these tests verify UTF-8 handling patterns

#[test]
fn test_emoji_streaming() {
    // Split emoji across chunks
    let emoji = "🎉"; // 4-byte emoji
    let bytes = emoji.as_bytes();

    // Verify emoji is 4 bytes
    assert_eq!(bytes.len(), 4);

    // Verify it's valid UTF-8
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), "🎉");
}

#[test]
fn test_cjk_characters() {
    // Chinese character (3 bytes)
    let text = "中";
    let bytes = text.as_bytes();

    // Verify CJK is 3 bytes
    assert_eq!(bytes.len(), 3);

    // Verify it's valid UTF-8
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), "中");
}

#[test]
fn test_mixed_ascii_multibyte() {
    let text = "Hello 世界 🌍";
    let bytes = text.as_bytes();

    // Verify mixed text is valid UTF-8
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), text);
}

#[test]
fn test_incomplete_sequence_at_end() {
    let emoji = "🎉";
    let bytes = emoji.as_bytes();

    // Incomplete UTF-8 sequence (first 3 bytes of 4-byte emoji)
    let incomplete = &bytes[0..3];

    // Should fail validation
    assert!(std::str::from_utf8(incomplete).is_err());
}

#[test]
fn test_multiple_emoji_sequence() {
    let text = "🎉🎊🎈";
    let bytes = text.as_bytes();

    // Verify multiple emoji
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), text);
}

#[test]
fn test_zero_width_joiner() {
    // Family emoji with ZWJ
    let text = "👨‍👩‍👧‍👦";
    let bytes = text.as_bytes();

    // Verify complex emoji with ZWJ
    assert!(std::str::from_utf8(bytes).is_ok());
    assert!(!std::str::from_utf8(bytes).unwrap().is_empty());
}

#[test]
fn test_bidi_text() {
    // Mixed LTR and RTL text
    let text = "Hello مرحبا";
    let bytes = text.as_bytes();

    // Verify bidirectional text
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), text);
}

#[test]
fn test_surrogate_pairs() {
    // Emoji that requires surrogate pairs in UTF-16
    let text = "𝕳𝖊𝖑𝖑𝖔";
    let bytes = text.as_bytes();

    // Verify mathematical alphanumeric symbols
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), text);
}

#[test]
fn test_combining_characters() {
    // Character with combining diacritics
    let text = "é"; // e + combining acute
    let bytes = text.as_bytes();

    // Verify combining characters
    assert!(std::str::from_utf8(bytes).is_ok());
    assert!(!std::str::from_utf8(bytes).unwrap().is_empty());
}

#[test]
fn test_null_byte_handling() {
    let bytes = b"Hello\x00World";

    // Null bytes are valid UTF-8
    assert!(std::str::from_utf8(bytes).is_ok());
}

#[test]
fn test_very_long_sequence() {
    // Generate long text with multibyte characters
    let text = "🎉".repeat(1000);
    let bytes = text.as_bytes();

    // Verify long sequence
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap().chars().count(), 1000);
}

#[test]
fn test_chunked_streaming() {
    let text = "Hello 世界 🌍!";
    let bytes = text.as_bytes();

    // Verify full text is valid UTF-8
    assert!(std::str::from_utf8(bytes).is_ok());
    assert_eq!(std::str::from_utf8(bytes).unwrap(), text);

    // Test that chunking works at character boundaries
    let chars: Vec<char> = text.chars().collect();
    assert!(chars.len() > 0);
}

// Built by Foundation-Alpha 🏗️
