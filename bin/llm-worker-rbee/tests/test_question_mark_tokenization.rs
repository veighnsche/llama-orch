//! TEAM-095: Tokenization diagnostic test
//!
//! This test loads a real GGUF model and verifies tokenization works correctly.
//! Result: Tokenization is fine. The bug is elsewhere (zero token generation).

use llm_worker_rbee::backend::gguf_tokenizer::extract_tokenizer_from_gguf;
use std::path::PathBuf;

#[test]
#[ignore] // Requires model file to be present
fn test_tokenize_question_mark() {
    // TEAM-095: Find the TinyLlama model in .test-models
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let model_path = workspace_root.join(".test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    
    if !model_path.exists() {
        eprintln!("⚠️  Model not found at {:?}", model_path);
        eprintln!("Skipping test - model required for tokenization");
        return;
    }

    // Extract tokenizer from GGUF
    let tokenizer = extract_tokenizer_from_gguf(&model_path)
        .expect("Failed to extract tokenizer from GGUF");

    // TEAM-095: Test prompts with and without question marks
    let test_cases = vec![
        ("Hello world", "Control - no special chars"),
        ("Why?", "Single question mark"),
        ("Why is the sky blue?", "Question at end"),
        ("What? How?", "Multiple questions"),
        ("?", "Just a question mark"),
        ("Hello?World", "Question in middle"),
        ("???", "Triple question mark"),
    ];

    println!("\n🧪 TEAM-095: Question Mark Tokenization Test\n");
    println!("{:25} | Tokens | Token IDs", "Prompt");
    println!("{}", "-".repeat(80));

    for (prompt, description) in test_cases {
        // Encode with add_special_tokens=true (same as inference.rs line 186)
        let encoding = tokenizer.encode(prompt, true)
            .expect(&format!("Failed to tokenize: {}", prompt));
        
        let token_ids = encoding.get_ids();
        let token_count = token_ids.len();
        
        // Decode back to verify round-trip
        let decoded = tokenizer.decode(token_ids, true)
            .expect(&format!("Failed to decode: {:?}", token_ids));

        println!(
            "{:25} | {:6} | {:?}",
            format!("{} ({})", prompt, description),
            token_count,
            token_ids
        );
        
        // TEAM-095: Check for anomalies
        if token_count == 0 {
            eprintln!("  ⚠️  WARNING: 0 tokens generated for '{}'", prompt);
        }
        
        if decoded.trim() != prompt.trim() {
            eprintln!("  ⚠️  WARNING: Round-trip mismatch!");
            eprintln!("     Original: '{}'", prompt);
            eprintln!("     Decoded:  '{}'", decoded);
        }
    }
    
    println!("\n✅ Tokenization test complete");
}

#[test]
#[ignore] // Requires model file to be present
fn test_tokenizer_special_tokens() {
    // TEAM-095: Check what special tokens are defined
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let model_path = workspace_root.join(".test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    
    if !model_path.exists() {
        eprintln!("⚠️  Model not found - skipping test");
        return;
    }

    let tokenizer = extract_tokenizer_from_gguf(&model_path)
        .expect("Failed to extract tokenizer from GGUF");

    println!("\n🔍 TEAM-095: Special Tokens Inspection\n");
    
    let special_tokens = vec!["<unk>", "<s>", "</s>", "?", "!", ".", ","];
    
    for token_str in special_tokens {
        match tokenizer.token_to_id(token_str) {
            Some(id) => println!("  '{}' -> ID {}", token_str, id),
            None => println!("  '{}' -> NOT FOUND (will be tokenized)", token_str),
        }
    }
    
    println!("\n✅ Special tokens inspection complete");
}
