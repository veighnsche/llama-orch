//! Integration tests for model-loader
//!
//! Tests end-to-end workflows combining multiple components
//!
//! Note: These tests use validate_bytes() instead of load_and_validate()
//! because path validation requires relative paths within allowed_root,
//! which is complex to set up in tests. The validate_bytes() path tests
//! the same validation logic without filesystem complexity.

use model_loader::{LoadError, ModelLoader};

#[test]
fn test_full_validation_workflow_success() {
    let loader = ModelLoader::new();
    
    // Create valid GGUF bytes
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
    bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
    
    // Compute hash
    let hash = model_loader::validation::hash::compute_hash(&bytes);
    
    // Validate with all checks
    let result = loader.validate_bytes(&bytes, Some(&hash));
    
    // Should succeed
    assert!(result.is_ok(), "Validation failed: {:?}", result.err());
}

#[test]
fn test_full_validation_workflow_hash_mismatch() {
    let loader = ModelLoader::new();
    
    // Create valid GGUF bytes
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46];
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    
    // Use wrong hash
    let wrong_hash = "0".repeat(64);
    
    let result = loader.validate_bytes(&bytes, Some(&wrong_hash));
    
    // Should fail with hash mismatch
    assert!(matches!(result, Err(LoadError::HashMismatch { .. })));
}

#[test]
fn test_full_validation_workflow_invalid_gguf() {
    let loader = ModelLoader::new();
    
    // Create invalid GGUF bytes (wrong magic)
    let bytes = vec![0x00, 0x00, 0x00, 0x00];
    
    let result = loader.validate_bytes(&bytes, None);
    
    // Should fail with invalid format
    assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
}

#[test]
fn test_validate_bytes_workflow() {
    let loader = ModelLoader::new();
    
    // Valid GGUF bytes
    let mut bytes = vec![0x47, 0x47, 0x55, 0x46];
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    
    // Validate without hash
    let result = loader.validate_bytes(&bytes, None);
    assert!(result.is_ok());
    
    // Validate with correct hash
    let hash = model_loader::validation::hash::compute_hash(&bytes);
    let result = loader.validate_bytes(&bytes, Some(&hash));
    assert!(result.is_ok());
    
    // Validate with wrong hash
    let wrong_hash = "0".repeat(64);
    let result = loader.validate_bytes(&bytes, Some(&wrong_hash));
    assert!(matches!(result, Err(LoadError::HashMismatch { .. })));
}

#[test]
fn test_error_messages_are_actionable() {
    // Test that error messages provide useful context
    
    // Buffer overflow error
    let error = LoadError::BufferOverflow {
        offset: 100,
        length: 8,
        available: 105,
    };
    let msg = error.to_string();
    assert!(msg.contains("100"));
    assert!(msg.contains("8"));
    assert!(msg.contains("105"));
    
    // Tensor count exceeded
    let error = LoadError::TensorCountExceeded {
        count: 50000,
        max: 10000,
    };
    let msg = error.to_string();
    assert!(msg.contains("50000"));
    assert!(msg.contains("10000"));
    
    // String too long
    let error = LoadError::StringTooLong {
        length: 20_000_000,
        max: 10_000_000,
    };
    let msg = error.to_string();
    assert!(msg.contains("20000000"));
    assert!(msg.contains("10000000"));
}

#[test]
fn test_multiple_validations_in_sequence() {
    // Test validating multiple byte arrays to ensure no state leakage
    let loader = ModelLoader::new();
    
    for i in 0..5 {
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46];
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend(vec![i as u8; 100]); // Different content
        
        let result = loader.validate_bytes(&bytes, None);
        assert!(result.is_ok(), "Failed to validate iteration {}: {:?}", i, result.err());
    }
}

#[test]
fn test_concurrent_validation() {
    // Test that ModelLoader can be used from multiple threads
    use std::sync::Arc;
    use std::thread;
    
    let loader = Arc::new(ModelLoader::new());
    let mut handles = vec![];
    
    for i in 0..4 {
        let loader_clone = Arc::clone(&loader);
        
        let handle = thread::spawn(move || {
            let mut bytes = vec![0x47, 0x47, 0x55, 0x46];
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.extend(vec![i as u8; 100]);
            
            loader_clone.validate_bytes(&bytes, None)
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok());
    }
}
