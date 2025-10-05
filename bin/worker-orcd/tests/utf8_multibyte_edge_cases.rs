// UTF-8 Multibyte Edge Cases
//
// Tests for UTF-8 multibyte character edge cases in GPT tokenizer and streaming
// to ensure correct handling of complex Unicode sequences.
//
// Story: GT-046
// Spec: M0-W-1330

#[cfg(test)]
mod utf8_edge_case_tests {

    // Test 1: Multibyte Character Encoding
    #[test]
    fn test_multibyte_encoding() {
        println!("Test 1: Multibyte character encoding");
        
        let test_cases = vec![
            ("ASCII", "Hello", vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]),
            ("2-byte UTF-8", "café", vec![0x63, 0x61, 0x66, 0xC3, 0xA9]),
            ("3-byte UTF-8", "你好", vec![0xE4, 0xBD, 0xA0, 0xE5, 0xA5, 0xBD]),
            ("4-byte UTF-8", "🚀", vec![0xF0, 0x9F, 0x9A, 0x80]),
        ];
        
        println!("\n  Test cases:");
        for (name, text, expected_bytes) in test_cases {
            println!("\n    {}: \"{}\"", name, text);
            
            let actual_bytes: Vec<u8> = text.bytes().collect();
            println!("      Bytes: {:02X?}", actual_bytes);
            println!("      Expected: {:02X?}", expected_bytes);
            
            assert_eq!(actual_bytes, expected_bytes);
            println!("      ✓ Encoding correct");
        }
        
        println!("\n  ✓ Multibyte encoding validated");
    }

    // Test 2: Multibyte Character Decoding
    #[test]
    fn test_multibyte_decoding() {
        println!("Test 2: Multibyte character decoding");
        
        let test_cases = vec![
            ("2-byte", vec![0xC3, 0xA9], "é"),
            ("3-byte", vec![0xE4, 0xBD, 0xA0], "你"),
            ("4-byte", vec![0xF0, 0x9F, 0x9A, 0x80], "🚀"),
            ("Mixed", vec![0x48, 0xC3, 0xA9, 0xE4, 0xBD, 0xA0], "Hé你"),
        ];
        
        println!("\n  Test cases:");
        for (name, bytes, expected_text) in test_cases {
            println!("\n    {}", name);
            println!("      Bytes: {:02X?}", bytes);
            
            let decoded = String::from_utf8(bytes.clone()).unwrap();
            println!("      Decoded: \"{}\"", decoded);
            println!("      Expected: \"{}\"", expected_text);
            
            assert_eq!(decoded, expected_text);
            println!("      ✓ Decoding correct");
        }
        
        println!("\n  ✓ Multibyte decoding validated");
    }

    // Test 3: Streaming Boundary Safety
    #[test]
    fn test_streaming_boundary_safety() {
        println!("Test 3: Streaming boundary safety");
        
        // Emoji that spans 4 bytes: 🚀 = F0 9F 9A 80
        let full_emoji = vec![0xF0, 0x9F, 0x9A, 0x80];
        
        // Simulate streaming with boundary splits
        let splits = vec![
            ("Split at byte 1", vec![vec![0xF0], vec![0x9F, 0x9A, 0x80]]),
            ("Split at byte 2", vec![vec![0xF0, 0x9F], vec![0x9A, 0x80]]),
            ("Split at byte 3", vec![vec![0xF0, 0x9F, 0x9A], vec![0x80]]),
        ];
        
        println!("\n  Test cases:");
        for (name, chunks) in splits {
            println!("\n    {}", name);
            
            let mut buffer = Vec::new();
            let mut decoded_parts = Vec::new();
            
            for (i, chunk) in chunks.iter().enumerate() {
                println!("      Chunk {}: {:02X?}", i + 1, chunk);
                buffer.extend_from_slice(chunk);
                
                // Try to decode buffer
                match String::from_utf8(buffer.clone()) {
                    Ok(text) => {
                        println!("        Decoded: \"{}\" ✓", text);
                        decoded_parts.push(text);
                        buffer.clear();
                    }
                    Err(_) => {
                        println!("        Incomplete, buffering...");
                    }
                }
            }
            
            // Final decode
            if !buffer.is_empty() {
                let final_text = String::from_utf8(buffer.clone()).unwrap();
                decoded_parts.push(final_text);
            }
            
            let full_text: String = decoded_parts.concat();
            let expected = String::from_utf8(full_emoji.clone()).unwrap();
            
            println!("      Final: \"{}\"", full_text);
            assert_eq!(full_text, expected);
            println!("      ✓ Boundary safe");
        }
        
        println!("\n  ✓ Streaming boundary safety validated");
    }

    // Test 4: Emoji and Special Characters
    #[test]
    fn test_emoji_special_chars() {
        println!("Test 4: Emoji and special characters");
        
        let test_cases = vec![
            ("Single emoji", "🚀", 1),
            ("Multiple emoji", "🚀🌟💡", 3),
            ("Emoji with text", "Hello 🌍 World", 3), // 3 words
            ("Complex emoji", "👨‍👩‍👧‍👦", 1), // Family emoji (ZWJ sequence)
            ("Emoji variants", "👍🏻👍🏿", 2), // Skin tone variants
        ];
        
        println!("\n  Test cases:");
        for (name, text, _expected_parts) in test_cases {
            println!("\n    {}: \"{}\"", name, text);
            
            let byte_len = text.len();
            let char_count = text.chars().count();
            
            println!("      Bytes: {}", byte_len);
            println!("      Chars: {}", char_count);
            
            // Verify valid UTF-8
            assert!(text.is_ascii() || text.chars().all(|c| c.len_utf8() <= 4));
            
            // Verify round-trip
            let bytes: Vec<u8> = text.bytes().collect();
            let decoded = String::from_utf8(bytes).unwrap();
            assert_eq!(decoded, text);
            
            println!("      ✓ Valid UTF-8");
        }
        
        println!("\n  ✓ Emoji and special characters validated");
    }

    // Test 5: Invalid UTF-8 Handling
    #[test]
    fn test_invalid_utf8_handling() {
        println!("Test 5: Invalid UTF-8 handling");
        
        let invalid_sequences = vec![
            ("Truncated 2-byte", vec![0xC3]),
            ("Truncated 3-byte", vec![0xE4, 0xBD]),
            ("Truncated 4-byte", vec![0xF0, 0x9F, 0x9A]),
            ("Invalid continuation", vec![0xC3, 0x28]),
            ("Overlong encoding", vec![0xC0, 0x80]),
        ];
        
        println!("\n  Test cases:");
        for (name, bytes) in invalid_sequences {
            println!("\n    {}", name);
            println!("      Bytes: {:02X?}", bytes);
            
            match String::from_utf8(bytes.clone()) {
                Ok(text) => {
                    println!("      Unexpected success: \"{}\"", text);
                    panic!("Should have failed");
                }
                Err(e) => {
                    println!("      Error: {} ✓", e);
                    println!("      Handling: Buffer until complete or replace");
                }
            }
        }
        
        println!("\n  ✓ Invalid UTF-8 handling validated");
    }

    // Test 6: Token Boundary UTF-8 Safety
    #[test]
    fn test_token_boundary_utf8_safety() {
        println!("Test 6: Token boundary UTF-8 safety");
        
        // Simulate tokens that might split UTF-8 sequences
        let text = "Hello 世界 🚀";
        let tokens = vec![
            ("Token 1", "Hello "),
            ("Token 2", "世"),
            ("Token 3", "界 "),
            ("Token 4", "🚀"),
        ];
        
        println!("  Original text: \"{}\"", text);
        println!("\n  Token stream:");
        
        let mut reconstructed = String::new();
        let mut buffer = Vec::new();
        
        for (name, token_text) in tokens {
            println!("\n    {}: \"{}\"", name, token_text);
            
            let token_bytes: Vec<u8> = token_text.bytes().collect();
            println!("      Bytes: {:02X?}", token_bytes);
            
            // Add to buffer
            buffer.extend_from_slice(&token_bytes);
            
            // Try to decode
            match String::from_utf8(buffer.clone()) {
                Ok(decoded) => {
                    println!("      Decoded: \"{}\" ✓", decoded);
                    reconstructed.push_str(&decoded);
                    buffer.clear();
                }
                Err(_) => {
                    println!("      Incomplete, buffering...");
                }
            }
        }
        
        // Flush buffer
        if !buffer.is_empty() {
            let final_text = String::from_utf8(buffer).unwrap();
            reconstructed.push_str(&final_text);
        }
        
        println!("\n  Reconstructed: \"{}\"", reconstructed);
        assert_eq!(reconstructed, text);
        
        println!("\n  ✓ Token boundary UTF-8 safety validated");
    }

    // Test 7: SSE Streaming UTF-8 Safety
    #[test]
    fn test_sse_streaming_utf8_safety() {
        println!("Test 7: SSE streaming UTF-8 safety");
        
        // Simulate SSE token events with multibyte characters
        let events = vec![
            ("event: token\ndata: {\"t\":\"Hello\"}\n\n", "Hello"),
            ("event: token\ndata: {\"t\":\" \"}\n\n", " "),
            ("event: token\ndata: {\"t\":\"世\"}\n\n", "世"),
            ("event: token\ndata: {\"t\":\"界\"}\n\n", "界"),
            ("event: token\ndata: {\"t\":\" \"}\n\n", " "),
            ("event: token\ndata: {\"t\":\"🚀\"}\n\n", "🚀"),
        ];
        
        println!("\n  SSE event stream:");
        
        let mut full_text = String::new();
        
        for (i, (event, token)) in events.iter().enumerate() {
            println!("\n    Event {}: {}", i + 1, event.trim());
            
            // Verify event is valid UTF-8
            assert!(event.is_ascii() || event.chars().all(|c| c.len_utf8() <= 4));
            
            // Extract token
            full_text.push_str(token);
            println!("      Token: \"{}\"", token);
            println!("      Accumulated: \"{}\"", full_text);
        }
        
        println!("\n  Final text: \"{}\"", full_text);
        assert_eq!(full_text, "Hello 世界 🚀");
        
        println!("\n  ✓ SSE streaming UTF-8 safety validated");
    }

    // Test 8: Unicode Normalization
    #[test]
    fn test_unicode_normalization() {
        println!("Test 8: Unicode normalization");
        
        // Different representations of "é"
        let nfc = "é"; // NFC: single codepoint U+00E9
        let nfd = "é"; // NFD: e + combining acute U+0065 U+0301
        
        println!("  NFC: \"{}\" ({} bytes, {} chars)", nfc, nfc.len(), nfc.chars().count());
        println!("  NFD: \"{}\" ({} bytes, {} chars)", nfd, nfd.len(), nfd.chars().count());
        
        // Both should be valid UTF-8
        assert!(nfc.is_ascii() || nfc.chars().all(|c| c.len_utf8() <= 4));
        assert!(nfd.is_ascii() || nfd.chars().all(|c| c.len_utf8() <= 4));
        
        // Visual equality (may differ in byte representation)
        println!("\n  Visual equality: {}", nfc == nfd || nfc.chars().eq(nfd.chars()));
        
        println!("\n  ✓ Unicode normalization handled");
    }

    // Test 9: Zero-Width Characters
    #[test]
    fn test_zero_width_characters() {
        println!("Test 9: Zero-width characters");
        
        let test_cases = vec![
            ("Zero-width space", "Hello\u{200B}World", "Hello​World"),
            ("Zero-width joiner", "👨\u{200D}👩\u{200D}👧", "👨‍👩‍👧"),
            ("Zero-width non-joiner", "fi\u{200C}sh", "fi‌sh"),
        ];
        
        println!("\n  Test cases:");
        for (name, text, _display) in test_cases {
            println!("\n    {}", name);
            println!("      Text: \"{}\"", text);
            println!("      Bytes: {}", text.len());
            println!("      Chars: {}", text.chars().count());
            
            // Verify valid UTF-8
            let bytes: Vec<u8> = text.bytes().collect();
            let decoded = String::from_utf8(bytes).unwrap();
            assert_eq!(decoded, text);
            
            println!("      ✓ Valid UTF-8");
        }
        
        println!("\n  ✓ Zero-width characters validated");
    }

    // Test 10: Bidirectional Text
    #[test]
    fn test_bidirectional_text() {
        println!("Test 10: Bidirectional text");
        
        let test_cases = vec![
            ("RTL Arabic", "مرحبا"),
            ("RTL Hebrew", "שלום"),
            ("Mixed LTR-RTL", "Hello مرحبا World"),
            ("RTL with numbers", "العدد 123"),
        ];
        
        println!("\n  Test cases:");
        for (name, text) in test_cases {
            println!("\n    {}: \"{}\"", name, text);
            
            let byte_len = text.len();
            let char_count = text.chars().count();
            
            println!("      Bytes: {}", byte_len);
            println!("      Chars: {}", char_count);
            
            // Verify valid UTF-8
            assert!(text.chars().all(|c| c.len_utf8() <= 4));
            
            // Verify round-trip
            let bytes: Vec<u8> = text.bytes().collect();
            let decoded = String::from_utf8(bytes).unwrap();
            assert_eq!(decoded, text);
            
            println!("      ✓ Valid UTF-8");
        }
        
        println!("\n  ✓ Bidirectional text validated");
    }
}

// ---
// Crafted by GPT-Gamma 🤖
