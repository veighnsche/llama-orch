//! Integration tests for real GGUF parsing
//!
//! Tests with actual GGUF files to verify the parser works correctly.

use worker_gguf::GGUFMetadata;

#[test]
fn test_parse_real_qwen_file() {
    let path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Skip if file doesn't exist
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test - GGUF file not found: {}", path);
        return;
    }
    
    let metadata = GGUFMetadata::from_file(path).expect("Failed to parse GGUF file");
    
    // Verify architecture
    assert_eq!(metadata.architecture().unwrap(), "qwen2");
    
    // Verify config values
    assert_eq!(metadata.vocab_size().unwrap(), 151936);
    assert_eq!(metadata.hidden_dim().unwrap(), 896);
    assert_eq!(metadata.num_layers().unwrap(), 24);
    assert_eq!(metadata.num_heads().unwrap(), 14);
    assert_eq!(metadata.num_kv_heads().unwrap(), 2);
    assert_eq!(metadata.context_length().unwrap(), 32768);
    
    // Verify GQA detection
    assert!(metadata.is_gqa());
    
    println!("✅ Successfully parsed Qwen2.5-0.5B GGUF file!");
    println!("   Architecture: {}", metadata.architecture().unwrap());
    println!("   Vocab size: {}", metadata.vocab_size().unwrap());
    println!("   Hidden dim: {}", metadata.hidden_dim().unwrap());
    println!("   Layers: {}", metadata.num_layers().unwrap());
    println!("   Heads: {} (KV: {})", metadata.num_heads().unwrap(), metadata.num_kv_heads().unwrap());
    println!("   Context: {}", metadata.context_length().unwrap());
}

#[test]
fn test_invalid_file() {
    let result = GGUFMetadata::from_file("/nonexistent/file.gguf");
    assert!(result.is_err());
}

#[test]
fn test_missing_keys() {
    let path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    if !std::path::Path::new(path).exists() {
        return;
    }
    
    let metadata = GGUFMetadata::from_file(path).unwrap();
    
    // All required keys should exist
    assert!(metadata.architecture().is_ok());
    assert!(metadata.vocab_size().is_ok());
    assert!(metadata.hidden_dim().is_ok());
    assert!(metadata.num_layers().is_ok());
    assert!(metadata.num_heads().is_ok());
    assert!(metadata.context_length().is_ok());
}
