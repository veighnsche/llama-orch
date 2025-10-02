//! Property-based tests for model-loader
//!
//! These tests verify security properties using proptest.
//! Each property is tested with 1000 random inputs.

use model_loader::{LoadError, ModelLoader};
use proptest::prelude::*;

// Property 1: Parser never panics on any input
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn property_parser_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..10000)) {
        let loader = ModelLoader::new();
        
        // Parser should never panic, only return error
        let _ = loader.validate_bytes(&bytes, None);
        // If we reach here, no panic occurred ✓
    }
}

// Property 2: Valid GGUF always accepted
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    #[test]
    fn property_valid_gguf_accepted(
        tensor_count in 1u64..100,
        metadata_count in 0u64..10,
    ) {
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
        bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        bytes.extend_from_slice(&tensor_count.to_le_bytes());
        bytes.extend_from_slice(&metadata_count.to_le_bytes());
        
        let loader = ModelLoader::new();
        let result = loader.validate_bytes(&bytes, None);
        
        // Valid GGUF should always be accepted
        prop_assert!(result.is_ok());
    }
}

// Property 3: Bounds checks always hold
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn property_bounds_checks_hold(
        bytes in prop::collection::vec(any::<u8>(), 0..100),
        offset in 0usize..200,
    ) {
        use model_loader::validation::gguf::parser;
        
        let result_u32 = parser::read_u32(&bytes, offset);
        let result_u64 = parser::read_u64(&bytes, offset);
        
        // If offset + size > len, must return error
        if offset.saturating_add(4) > bytes.len() {
            prop_assert!(result_u32.is_err());
        }
        
        if offset.saturating_add(8) > bytes.len() {
            prop_assert!(result_u64.is_err());
        }
    }
}

// Property 4: String length validation
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn property_string_length_validated(str_len in 0u64..1_000_000) {
        use model_loader::validation::gguf::{parser, limits};
        
        // Create bytes with string length header
        let mut bytes = vec![];
        bytes.extend_from_slice(&str_len.to_le_bytes());
        
        // Add enough data for the string
        if str_len <= 10000 {
            bytes.extend(vec![b'a'; str_len as usize]);
        }
        
        let result = parser::read_string(&bytes, 0);
        
        // Strings longer than MAX_STRING_LEN must be rejected
        if str_len > limits::MAX_STRING_LEN as u64 {
            let is_too_long = matches!(result, Err(LoadError::StringTooLong { length: _, max: _ }));
            prop_assert!(is_too_long);
        }
    }
}

// Property 5: Tensor count limits
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn property_tensor_count_limited(tensor_count in 0u64..100_000) {
        use model_loader::validation::gguf::limits;
        
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Magic
        bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        bytes.extend_from_slice(&tensor_count.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
        
        let loader = ModelLoader::new();
        let result = loader.validate_bytes(&bytes, None);
        
        // Tensor counts > MAX_TENSORS must be rejected
        if tensor_count > limits::MAX_TENSORS as u64 {
            let is_exceeded = matches!(result, Err(LoadError::TensorCountExceeded { count: _, max: _ }));
            prop_assert!(is_exceeded);
        }
    }
}

// Property 6: Hash verification correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    #[test]
    fn property_hash_verification_correct(bytes in prop::collection::vec(any::<u8>(), 0..1000)) {
        use model_loader::validation::hash;
        
        // Compute correct hash
        let correct_hash = hash::compute_hash(&bytes);
        
        // Verification with correct hash should succeed
        prop_assert!(hash::verify_hash(&bytes, &correct_hash).is_ok());
        
        // Verification with wrong hash should fail
        let wrong_hash = "0".repeat(64);
        if wrong_hash != correct_hash {
            prop_assert!(hash::verify_hash(&bytes, &wrong_hash).is_err());
        }
    }
}

// Property 7: Invalid magic number always rejected
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn property_invalid_magic_rejected(magic in any::<u32>()) {
        // Skip valid magic
        prop_assume!(magic != 0x46554747);
        
        let mut bytes = vec![];
        bytes.extend_from_slice(&magic.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // Version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
        
        let loader = ModelLoader::new();
        let result = loader.validate_bytes(&bytes, None);
        
        // Invalid magic must be rejected
        prop_assert!(result.is_err());
    }
}

// Property 8: Version validation
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn property_version_validated(version in any::<u32>()) {
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // Valid magic
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata count
        
        let loader = ModelLoader::new();
        let result = loader.validate_bytes(&bytes, None);
        
        // Only versions 2 and 3 are valid
        if version != 2 && version != 3 {
            prop_assert!(result.is_err());
        } else {
            prop_assert!(result.is_ok());
        }
    }
}
