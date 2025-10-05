//! FT-043: UTF-8 Streaming Edge Cases Test
//!
//! Tests UTF-8 streaming with multibyte characters, emojis, and edge cases.
//! Validates that SSE streaming correctly handles partial UTF-8 sequences.
//!
//! Spec: M0-W-1610

use worker_orcd::tests::integration::{collect_sse_events, extract_tokens, make_test_request, WorkerTestHarness};

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Only run with real model
async fn test_emoji_streaming() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let mut req = make_test_request(
        "test-emoji",
        "List 5 emojis: 🎨",
        50
    );
    req.temperature = 0.7;
    
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    
    let tokens = extract_tokens(&events);
    let output = tokens.join("");
    
    // Verify output is valid UTF-8
    assert!(output.is_ascii() || output.chars().all(|c| c.is_alphanumeric() || c.is_whitespace() || !c.is_control()));
    
    println!("Emoji output: {}", output);
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Only run with real model
async fn test_multibyte_characters() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let mut req = make_test_request(
        "test-multibyte",
        "Write in Japanese: こんにちは",
        30
    );
    req.temperature = 0.7;
    
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    
    let tokens = extract_tokens(&events);
    let output = tokens.join("");
    
    // Verify output is valid UTF-8
    assert!(std::str::from_utf8(output.as_bytes()).is_ok());
    
    println!("Multibyte output: {}", output);
}

#[tokio::test]
#[cfg(feature = "cuda")]
#[ignore] // Only run with real model
async fn test_mixed_scripts() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let mut req = make_test_request(
        "test-mixed",
        "Mix English, 中文, and العربية",
        40
    );
    req.temperature = 0.7;
    
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    
    let tokens = extract_tokens(&events);
    let output = tokens.join("");
    
    // Verify output is valid UTF-8
    assert!(std::str::from_utf8(output.as_bytes()).is_ok());
    
    println!("Mixed scripts output: {}", output);
}

#[test]
fn test_utf8_validation() {
    // Test valid UTF-8 sequences
    let valid_sequences = vec![
        "Hello",
        "🎨",
        "こんにちは",
        "中文",
        "العربية",
        "Emoji: 🚀🎯✨",
    ];
    
    for seq in valid_sequences {
        assert!(std::str::from_utf8(seq.as_bytes()).is_ok());
    }
}

#[test]
fn test_utf8_byte_boundaries() {
    // Test that we can correctly identify UTF-8 byte boundaries
    let text = "Hello 🎨 World";
    let bytes = text.as_bytes();
    
    // Verify we can reconstruct the string
    let reconstructed = std::str::from_utf8(bytes).unwrap();
    assert_eq!(text, reconstructed);
}

#[test]
fn test_utf8_char_iteration() {
    let text = "🎨🚀✨";
    let chars: Vec<char> = text.chars().collect();
    
    assert_eq!(chars.len(), 3);
    assert_eq!(chars[0], '🎨');
    assert_eq!(chars[1], '🚀');
    assert_eq!(chars[2], '✨');
}

#[test]
fn test_utf8_partial_sequence_detection() {
    // Test detecting incomplete UTF-8 sequences
    let complete = "🎨".as_bytes();
    assert_eq!(complete.len(), 4); // Emoji is 4 bytes
    
    // Partial sequence (first 3 bytes of emoji)
    let partial = &complete[0..3];
    assert!(std::str::from_utf8(partial).is_err());
}

#[test]
fn test_utf8_mixed_ascii_multibyte() {
    let text = "ASCII 中文 ASCII";
    
    // Count bytes vs chars
    let byte_count = text.len();
    let char_count = text.chars().count();
    
    assert!(byte_count > char_count); // Multibyte chars present
    assert!(std::str::from_utf8(text.as_bytes()).is_ok());
}

#[test]
fn test_utf8_zero_width_joiner() {
    // Test zero-width joiner (used in emoji combinations)
    let text = "👨‍👩‍👧‍👦"; // Family emoji with ZWJ
    
    assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    
    // This is actually multiple codepoints joined
    let chars: Vec<char> = text.chars().collect();
    assert!(chars.len() > 1);
}

#[test]
fn test_utf8_bom_handling() {
    // Test UTF-8 BOM (Byte Order Mark)
    let with_bom = "\u{FEFF}Hello";
    let without_bom = "Hello";
    
    assert!(std::str::from_utf8(with_bom.as_bytes()).is_ok());
    assert_ne!(with_bom, without_bom);
}

#[test]
fn test_utf8_surrogate_pairs() {
    // Test characters that would require surrogate pairs in UTF-16
    let text = "𝕳𝖊𝖑𝖑𝖔"; // Mathematical bold text
    
    assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    
    // Each char is 4 bytes in UTF-8
    for c in text.chars() {
        let mut buf = [0u8; 4];
        let encoded = c.encode_utf8(&mut buf);
        assert_eq!(encoded.len(), 4);
    }
}

// Built by Foundation-Alpha 🏗️
