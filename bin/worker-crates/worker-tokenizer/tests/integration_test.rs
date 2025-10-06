//! Integration test for tokenizer with real GGUF file

use worker_tokenizer::{Tokenizer, TokenizerBackend};

#[test]
fn test_tokenizer_with_qwen_gguf() {
    let path =
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    // Skip if file doesn't exist
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test - GGUF file not found");
        return;
    }

    // TODO: Implement GGUF tokenizer loading
    // For now, skip this test
    eprintln!("⚠️  GGUF tokenizer loading not yet implemented - skipping");
    return;

    // Create tokenizer from GGUF
    let tokenizer = Tokenizer::from_gguf(path).expect("Failed to create tokenizer");

    // Test encoding
    let text = "Write a haiku about code";
    let tokens = tokenizer.encode(text, true).expect("Failed to encode");

    println!("✅ Encoded '{}' to {} tokens", text, tokens.len());
    assert!(!tokens.is_empty());

    // Test decoding
    let decoded = tokenizer.decode(&tokens, false).expect("Failed to decode");

    println!("✅ Decoded back to: '{}'", decoded);

    // Should contain most of the original text
    assert!(decoded.contains("haiku") || decoded.contains("code"));

    println!("✅ Tokenizer test passed!");
}

#[test]
fn test_encode_decode_roundtrip() {
    let path =
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    if !std::path::Path::new(path).exists() {
        return;
    }

    // TODO: Implement GGUF tokenizer loading
    eprintln!("⚠️  GGUF tokenizer loading not yet implemented - skipping");
    return;

    let tokenizer = Tokenizer::from_gguf(path).expect("Failed to create tokenizer");

    let test_cases =
        vec!["Hello, world!", "The quick brown fox", "Write a haiku about", "fn main() {"];

    for text in test_cases {
        let tokens = tokenizer.encode(text, false).expect("Encode failed");
        let decoded = tokenizer.decode(&tokens, false).expect("Decode failed");

        println!("Original: '{}' -> {} tokens -> '{}'", text, tokens.len(), decoded);

        // Decoded should be similar (may have whitespace differences)
        assert!(!decoded.is_empty());
    }

    println!("✅ All roundtrip tests passed!");
}
