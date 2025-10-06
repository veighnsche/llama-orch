//! Phi-3 Tokenizer Conformance Tests - LT-032
//!
//! Validates that the BPE tokenizer correctly handles Phi-3 vocabulary.
//! Similar to Qwen conformance tests but for Phi-3-specific patterns.

use worker_tokenizer::{BPEDecoder, BPEEncoder, MergeTable, Vocabulary};

/// Create a Phi-3-like tokenizer for testing
fn create_phi3_tokenizer() -> (BPEEncoder, BPEDecoder) {
    // Phi-3 uses a 32K vocabulary (smaller than Qwen's 151K)
    // Note: BPE uses byte-level encoding with Ġ for spaces
    let tokens = vec![
        "<s>".to_string(),
        "</s>".to_string(),
        "<unk>".to_string(),
        "H".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
        " ".to_string(),
        "Ġ".to_string(), // Byte-level space
        "w".to_string(),
        "r".to_string(),
        "d".to_string(),
        "!".to_string(),
        "He".to_string(),
        "ll".to_string(),     // Merged token
        "lo".to_string(),     // Merged token
        "llo".to_string(),    // Merged token
        "Hello".to_string(),  // Merged token
        " w".to_string(),     // Merged token
        "Ġw".to_string(),     // Byte-level space + w
        "wo".to_string(),     // Merged token
        "or".to_string(),     // Merged token
        "rl".to_string(),     // Merged token
        "ld".to_string(),     // Merged token
        "world".to_string(),  // Merged token
        "Ġworld".to_string(), // Byte-level space + world
    ];

    let vocab = Vocabulary::new(tokens, 0, 1, Some(2)).unwrap();

    let merges = vec![
        "H e".to_string(),
        "l l".to_string(),
        "l o".to_string(),
        "ll o".to_string(),
        "He llo".to_string(),
        " w".to_string(),
        "w o".to_string(),
        "o r".to_string(),
        "r l".to_string(),
        "l d".to_string(),
    ];

    let merge_table = MergeTable::new(merges).unwrap();

    let encoder = BPEEncoder::new(vocab.clone(), merge_table);
    let decoder = BPEDecoder::new(vocab);

    (encoder, decoder)
}

#[test]
fn test_phi3_hello_world() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "Hello world!";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_single_character() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "H";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_merged_token() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "He";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_with_space() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "Hello world";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_with_punctuation() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "Hello!";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_empty_string() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_repeated_characters() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "HHH";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_deterministic_encoding() {
    let (encoder, _) = create_phi3_tokenizer();

    let text = "Hello";
    let ids1 = encoder.encode(text).unwrap();
    let ids2 = encoder.encode(text).unwrap();

    assert_eq!(ids1, ids2);
}

#[test]
fn test_phi3_round_trip_multiple() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let texts = vec!["H", "He", "Hello", "Hello world", "Hello world!"];

    for text in texts {
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();
        assert_eq!(decoded, text, "Round-trip failed for: {}", text);
    }
}

#[test]
fn test_phi3_special_tokens() {
    let (_, decoder) = create_phi3_tokenizer();

    // Test BOS token decodes successfully
    let ids = vec![0]; // BOS token ID
    let decoded = decoder.decode(&ids);
    assert!(decoded.is_ok(), "BOS token should decode");

    // Test EOS token decodes successfully
    let ids = vec![1]; // EOS token ID
    let decoded = decoder.decode(&ids);
    assert!(decoded.is_ok(), "EOS token should decode");
}

#[test]
fn test_phi3_unknown_token_handling() {
    let (_, decoder) = create_phi3_tokenizer();

    // Test UNK token decodes successfully
    let ids = vec![2]; // UNK token ID
    let decoded = decoder.decode(&ids);
    assert!(decoded.is_ok(), "UNK token should decode");
}

#[test]
fn test_phi3_long_text() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "Hello world! Hello world!";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded, text);
}

#[test]
fn test_phi3_token_ids_valid() {
    let (encoder, _) = create_phi3_tokenizer();

    let text = "Hello";
    let ids = encoder.encode(text).unwrap();

    // All token IDs should be valid (< vocab size of 26)
    for id in ids {
        assert!(id < 26, "Token ID {} exceeds vocab size", id);
    }
}

#[test]
fn test_phi3_encoding_not_empty() {
    let (encoder, _) = create_phi3_tokenizer();

    let text = "Hello";
    let ids = encoder.encode(text).unwrap();

    assert!(!ids.is_empty(), "Encoding should not be empty");
}

#[test]
fn test_phi3_decoding_preserves_length() {
    let (encoder, decoder) = create_phi3_tokenizer();

    let text = "Hello world!";
    let ids = encoder.encode(text).unwrap();
    let decoded = decoder.decode(&ids).unwrap();

    assert_eq!(decoded.len(), text.len());
}

#[test]
fn test_phi3_merge_priority() {
    let (encoder, _) = create_phi3_tokenizer();

    // "Hello" should use merged tokens efficiently
    let text = "Hello";
    let ids = encoder.encode(text).unwrap();

    // Should use merged tokens, not individual characters
    assert!(ids.len() < text.len(), "Should use merged tokens");
}

#[test]
fn test_phi3_consistency_across_calls() {
    let (encoder1, decoder1) = create_phi3_tokenizer();
    let (encoder2, decoder2) = create_phi3_tokenizer();

    let text = "Hello world!";

    let ids1 = encoder1.encode(text).unwrap();
    let ids2 = encoder2.encode(text).unwrap();

    assert_eq!(ids1, ids2, "Encoders should be consistent");

    let decoded1 = decoder1.decode(&ids1).unwrap();
    let decoded2 = decoder2.decode(&ids2).unwrap();

    assert_eq!(decoded1, decoded2, "Decoders should be consistent");
}

// Built by Llama-Beta 🦙
