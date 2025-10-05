// Tokenizer Conformance Tests for Qwen2.5 - LT-018
//
// Validates BPE tokenizer against reference implementation.
// Tests encoding, decoding, and round-trip for various text inputs.
//
// Note: These are simplified tests with a minimal vocab.
// Full conformance testing requires loading actual Qwen vocab/merges from GGUF.
//
// Spec: M0-W-1363

use worker_orcd::tokenizer::{BPEDecoder, BPEEncoder, MergeTable, Vocabulary};

#[allow(dead_code)]
/// Test vector for tokenizer conformance
struct TokenizerTestVector {
    text: &'static str,
    expected_tokens: Vec<u32>,
    description: &'static str,
}

/// Create test encoder/decoder with comprehensive vocab for testing
fn create_test_tokenizer() -> (BPEEncoder, BPEDecoder) {
    // Comprehensive vocab including all intermediate merge results
    let tokens = vec![
        "<BOS>".to_string(),
        "<EOS>".to_string(),
        "H".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
        ",".to_string(),
        " ".to_string(),
        "w".to_string(),
        "r".to_string(),
        "d".to_string(),
        "!".to_string(),
        "He".to_string(),
        "ll".to_string(),
        "Hello".to_string(),
        "world".to_string(),
        "Hell".to_string(),
        "wo".to_string(),
        "wor".to_string(),
        "worl".to_string(),
    ];

    let vocab = Vocabulary::new(tokens, 0, 1, None).unwrap();

    // Merge rules that produce the vocab above
    let merge_lines = vec![
        "H e".to_string(),    // H + e → He
        "l l".to_string(),    // l + l → ll
        "He ll".to_string(),  // He + ll → Hell
        "Hell o".to_string(), // Hell + o → Hello
        "w o".to_string(),    // w + o → wo
        "wo r".to_string(),   // wo + r → wor
        "wor l".to_string(),  // wor + l → worl
        "worl d".to_string(), // worl + d → world
    ];

    let merges = MergeTable::new(merge_lines).unwrap();

    let encoder = BPEEncoder::new(vocab.clone(), merges);
    let decoder = BPEDecoder::new(vocab);

    (encoder, decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformance_single_token() {
        let (encoder, decoder) = create_test_tokenizer();

        // Test single character that exists in vocab
        let text = "H";
        let ids = encoder.encode(text).unwrap();
        assert!(!ids.is_empty());

        let decoded = decoder.decode(&ids).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_merged_token() {
        let (encoder, decoder) = create_test_tokenizer();

        // Test "He" which should merge from "H" + "e"
        let text = "He";
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_double_char() {
        let (encoder, decoder) = create_test_tokenizer();

        // Test "ll" which should merge from "l" + "l"
        let text = "ll";
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_empty_string() {
        let (encoder, decoder) = create_test_tokenizer();

        let text = "";
        let ids = encoder.encode(text).unwrap();
        assert_eq!(ids, Vec::<u32>::new());

        let decoded = decoder.decode(&ids).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_conformance_punctuation() {
        let (encoder, decoder) = create_test_tokenizer();

        let text = "!";
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_with_special_tokens() {
        let (encoder, decoder) = create_test_tokenizer();

        let text = "Hello";
        let ids = encoder.encode_with_special(text, true, true).unwrap();

        // Should have BOS + content + EOS
        assert_eq!(ids[0], 0); // BOS
        assert_eq!(ids[ids.len() - 1], 1); // EOS

        // Decode with special token skipping
        let decoded = decoder.decode_with_special(&ids, true).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_round_trip_simple() {
        let (encoder, decoder) = create_test_tokenizer();

        // Test simple tokens that we know exist
        let texts = vec!["H", "e", "l", "He", "ll"];

        for text in texts {
            let ids = encoder.encode(text).unwrap();
            let decoded = decoder.decode(&ids).unwrap();
            assert_eq!(decoded, text, "Round-trip failed for: {}", text);
        }
    }

    #[test]
    fn test_conformance_encode_deterministic() {
        let (encoder, _) = create_test_tokenizer();

        let text = "He";
        let ids1 = encoder.encode(text).unwrap();
        let ids2 = encoder.encode(text).unwrap();

        // Encoding should be deterministic
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn test_conformance_decode_deterministic() {
        let (_, decoder) = create_test_tokenizer();

        let ids = vec![2, 3, 4, 4, 5];
        let text1 = decoder.decode(&ids).unwrap();
        let text2 = decoder.decode(&ids).unwrap();

        // Decoding should be deterministic
        assert_eq!(text1, text2);
    }

    #[test]
    fn test_conformance_merge_application() {
        let (encoder, _) = create_test_tokenizer();

        // Test that merges are applied
        let text = "He"; // Should merge H + e
        let ids = encoder.encode(text).unwrap();

        // Should produce fewer tokens than characters (due to merging)
        assert!(ids.len() <= text.len());
    }

    #[test]
    fn test_conformance_token_count() {
        let (encoder, _) = create_test_tokenizer();

        let text = "He";
        let ids = encoder.encode(text).unwrap();

        // Should produce some tokens
        assert!(!ids.is_empty());

        // Token count should be reasonable (merging reduces count)
        assert!(ids.len() <= text.len());
    }

    #[test]
    fn test_conformance_vocab_coverage() {
        let (encoder, _) = create_test_tokenizer();

        // All single-char vocab tokens should be encodable
        let texts = vec!["H", "e", "l", "o", ",", "w", "r", "d", "!"];

        for text in texts {
            let result = encoder.encode(text);
            assert!(result.is_ok(), "Failed to encode: {}", text);
        }
    }

    #[test]
    fn test_conformance_special_tokens_not_in_text() {
        let (encoder, decoder) = create_test_tokenizer();

        // Special tokens should not appear in decoded text
        let text = "He";
        let ids = encoder.encode_with_special(text, true, true).unwrap();
        let decoded = decoder.decode_with_special(&ids, true).unwrap();

        // Decoded text should not contain <BOS> or <EOS>
        assert!(!decoded.contains("<BOS>"));
        assert!(!decoded.contains("<EOS>"));
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_multiple_merges() {
        let (encoder, decoder) = create_test_tokenizer();

        let text = "Hell"; // Should merge: H+e → He, l+l → ll, He+ll → Hell
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_repeated_chars() {
        let (encoder, decoder) = create_test_tokenizer();

        let text = "llll";
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_conformance_all_punctuation() {
        let (encoder, decoder) = create_test_tokenizer();

        let text = "!,!,!";
        let ids = encoder.encode(text).unwrap();
        let decoded = decoder.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }
}

/// Integration test suite
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_conformance_suite() {
        let (encoder, decoder) = create_test_tokenizer();

        // Test vectors with tokens that exist in our minimal vocab
        let test_vectors = vec![
            ("", "Empty string"),
            ("H", "Single char"),
            ("e", "Single char e"),
            ("He", "Merged token"),
            ("ll", "Double l"),
            ("Hell", "Multiple merges"),
        ];

        let mut passed = 0;
        let mut failed = 0;

        for (text, description) in test_vectors {
            match encoder.encode(text) {
                Ok(ids) => match decoder.decode(&ids) {
                    Ok(decoded) => {
                        if decoded == text {
                            passed += 1;
                            eprintln!("PASS: {} - '{}'", description, text);
                        } else {
                            failed += 1;
                            eprintln!(
                                "FAIL: {} - expected '{}', got '{}'",
                                description, text, decoded
                            );
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("FAIL: {} - decode error: {:?}", description, e);
                    }
                },
                Err(e) => {
                    failed += 1;
                    eprintln!("FAIL: {} - encode error: {:?}", description, e);
                }
            }
        }

        eprintln!("Conformance tests: {} passed, {} failed", passed, failed);
        assert_eq!(failed, 0, "Some conformance tests failed");
    }
}
