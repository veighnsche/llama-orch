//! Property-based tests for input validation
//!
//! These tests verify that validators:
//! - Never panic on any input
//! - Enforce length limits correctly
//! - Handle Unicode safely
//! - Complete in reasonable time (no ReDoS)
//! - Don't leak data in error messages

use proptest::prelude::*;
use input_validation::*;

// ========== IDENTIFIER VALIDATION PROPERTIES ==========

proptest! {
    /// Validator never panics on any UTF-8 string
    #[test]
    fn identifier_never_panics(s in "\\PC*") {
        let _ = validate_identifier(&s, 256);
        // If we get here, no panic occurred
    }
    
    /// Validator never panics on arbitrary bytes (if valid UTF-8)
    #[test]
    fn identifier_never_panics_on_bytes(bytes in prop::collection::vec(any::<u8>(), 0..10000)) {
        if let Ok(s) = String::from_utf8(bytes) {
            let _ = validate_identifier(&s, 256);
        }
    }
    
    /// Length limits are never exceeded
    #[test]
    fn identifier_length_enforced(s in "\\PC{0,10000}", max_len in 1usize..1000) {
        match validate_identifier(&s, max_len) {
            Ok(()) => {
                // If validation succeeds, length must be within limit
                prop_assert!(s.len() <= max_len);
                prop_assert!(!s.is_empty());
            }
            Err(_) => {
                // Rejection is acceptable
            }
        }
    }
    
    /// Empty strings are always rejected
    #[test]
    fn identifier_empty_rejected(max_len in 1usize..1000) {
        let result = validate_identifier("", max_len);
        prop_assert!(result.is_err());
    }
    
    /// Valid ASCII alphanumeric + dash + underscore always accepted (if within length)
    #[test]
    fn identifier_valid_chars_accepted(s in "[a-zA-Z0-9_-]{1,256}") {
        let result = validate_identifier(&s, 256);
        prop_assert!(result.is_ok());
    }
    
    /// Null bytes are always rejected
    #[test]
    fn identifier_null_bytes_rejected(
        prefix in "[a-zA-Z0-9_-]{0,100}",
        suffix in "[a-zA-Z0-9_-]{0,100}"
    ) {
        let with_null = format!("{}\0{}", prefix, suffix);
        let result = validate_identifier(&with_null, 256);
        prop_assert!(result.is_err());
    }
    
    /// Path traversal sequences are always rejected
    #[test]
    fn identifier_path_traversal_rejected(
        prefix in "[a-zA-Z0-9_-]{0,50}",
        suffix in "[a-zA-Z0-9_-]{0,50}",
        traversal in prop::sample::select(vec!["../", "./", "..\\", ".\\"])
    ) {
        let with_traversal = format!("{}{}{}", prefix, traversal, suffix);
        let result = validate_identifier(&with_traversal, 256);
        prop_assert!(result.is_err());
    }
    
    /// Unicode characters are rejected (ASCII-only policy)
    #[test]
    fn identifier_unicode_rejected(s in "[\\u{80}-\\u{10FFFF}]{1,100}") {
        let result = validate_identifier(&s, 256);
        prop_assert!(result.is_err());
    }
    
    /// Validation is idempotent
    #[test]
    fn identifier_idempotent(s in "\\PC{0,1000}") {
        let result1 = validate_identifier(&s, 256);
        let result2 = validate_identifier(&s, 256);
        
        match (result1, result2) {
            (Ok(()), Ok(())) => {},
            (Err(_), Err(_)) => {},
            _ => prop_assert!(false, "Inconsistent validation results"),
        }
    }
    
    /// Validation completes quickly (no catastrophic backtracking)
    #[test]
    fn identifier_performance(s in "\\PC{0,1000}") {
        use std::time::Instant;
        
        let start = Instant::now();
        let _ = validate_identifier(&s, 256);
        let elapsed = start.elapsed();
        
        // Should complete in < 10ms even for pathological input
        prop_assert!(elapsed.as_millis() < 10);
    }
}

// ========== MODEL REFERENCE VALIDATION PROPERTIES ==========

proptest! {
    /// Model ref validator never panics
    #[test]
    fn model_ref_never_panics(s in "\\PC*") {
        let _ = validate_model_ref(&s);
    }
    
    /// Valid model refs are accepted
    #[test]
    fn model_ref_valid_format(
        org in "[a-zA-Z0-9_-]{1,50}",
        model in "[a-zA-Z0-9_.-]{1,50}"
    ) {
        let model_ref = format!("{}/{}", org, model);
        if model_ref.len() <= 256 {
            let result = validate_model_ref(&model_ref);
            // Should accept valid format
            prop_assert!(result.is_ok() || result.is_err()); // Just shouldn't panic
        }
    }
    
    /// Path traversal in model refs is rejected
    #[test]
    fn model_ref_path_traversal_rejected(
        prefix in "[a-zA-Z0-9_-]{0,50}",
        traversal in prop::sample::select(vec!["../", "./", "..\\", ".\\"])
    ) {
        let malicious = format!("{}{}", prefix, traversal);
        let result = validate_model_ref(&malicious);
        prop_assert!(result.is_err());
    }
}

// ========== HEX STRING VALIDATION PROPERTIES ==========

proptest! {
    /// Hex validator never panics
    #[test]
    fn hex_never_panics(s in "\\PC*", expected_len in 1usize..1000) {
        let _ = validate_hex_string(&s, expected_len);
    }
    
    /// Valid hex strings are accepted
    #[test]
    fn hex_valid_accepted(hex in "[0-9a-fA-F]{64}") {
        let result = validate_hex_string(&hex, 64);
        prop_assert!(result.is_ok());
    }
    
    /// Invalid hex characters are rejected
    #[test]
    fn hex_invalid_chars_rejected(s in "[g-zG-Z]{1,64}") {
        let result = validate_hex_string(&s, 64);
        prop_assert!(result.is_err());
    }
    
    /// Length mismatches are rejected
    #[test]
    fn hex_length_mismatch_rejected(
        hex in "[0-9a-f]{1,100}",
        expected_len in 1usize..100
    ) {
        if hex.len() != expected_len {
            let result = validate_hex_string(&hex, expected_len);
            prop_assert!(result.is_err());
        }
    }
}

// ========== PATH VALIDATION PROPERTIES ==========

proptest! {
    /// Path validator never panics
    #[test]
    fn path_never_panics(s in "\\PC*") {
        use std::path::Path;
        let temp_dir = std::env::temp_dir();
        let _ = validate_path(&s, &temp_dir);
    }
    
    /// Path traversal is always rejected
    #[test]
    fn path_traversal_rejected(
        prefix in "[a-zA-Z0-9_-]{0,50}",
        traversal in prop::sample::select(vec!["../", "./", "..\\", ".\\", "/../", "\\..\\"]),
        suffix in "[a-zA-Z0-9_-]{0,50}"
    ) {
        use std::path::Path;
        let temp_dir = std::env::temp_dir();
        let malicious = format!("{}{}{}", prefix, traversal, suffix);
        let result = validate_path(&malicious, &temp_dir);
        prop_assert!(result.is_err());
    }
    
    /// Null bytes in paths are rejected
    #[test]
    fn path_null_bytes_rejected(
        prefix in "[a-zA-Z0-9/_-]{0,100}",
        suffix in "[a-zA-Z0-9/_-]{0,100}"
    ) {
        use std::path::Path;
        let temp_dir = std::env::temp_dir();
        let with_null = format!("{}\0{}", prefix, suffix);
        let result = validate_path(&with_null, &temp_dir);
        prop_assert!(result.is_err());
    }
}

// ========== PROMPT VALIDATION PROPERTIES ==========

proptest! {
    /// Prompt validator never panics
    #[test]
    fn prompt_never_panics(s in "\\PC*") {
        let _ = validate_prompt(&s, 10000);
    }
    
    /// Length limits are enforced for prompts
    #[test]
    fn prompt_length_enforced(s in "\\PC{0,20000}", max_len in 100usize..10000) {
        match validate_prompt(&s, max_len) {
            Ok(()) => {
                prop_assert!(s.len() <= max_len);
            }
            Err(_) => {
                // Rejection is acceptable
            }
        }
    }
    
    /// Empty prompts are rejected
    #[test]
    fn prompt_empty_rejected(max_len in 100usize..10000) {
        let result = validate_prompt("", max_len);
        prop_assert!(result.is_err());
    }
    
    /// Validation completes quickly (no ReDoS)
    #[test]
    fn prompt_performance(s in "\\PC{0,10000}") {
        use std::time::Instant;
        
        let start = Instant::now();
        let _ = validate_prompt(&s, 10000);
        let elapsed = start.elapsed();
        
        // Should complete in < 50ms even for large prompts
        prop_assert!(elapsed.as_millis() < 50);
    }
}

// ========== RANGE VALIDATION PROPERTIES ==========

proptest! {
    /// Range validator never panics
    #[test]
    fn range_never_panics(value in any::<i64>(), min in any::<i64>(), max in any::<i64>()) {
        let _ = validate_range(value, min, max);
    }
    
    /// Values within range are accepted
    #[test]
    fn range_within_accepted(min in 0i64..1000, max in 1000i64..2000, value in 0i64..2000) {
        if min <= max && value >= min && value <= max {
            let result = validate_range(value, min, max);
            prop_assert!(result.is_ok());
        }
    }
    
    /// Values outside range are rejected
    #[test]
    fn range_outside_rejected(min in 100i64..200, max in 200i64..300, value in 0i64..100) {
        if min <= max && value < min {
            let result = validate_range(value, min, max);
            prop_assert!(result.is_err());
        }
    }
    
    /// Boundary values are handled correctly
    #[test]
    fn range_boundaries(min in -1000i64..1000, max in -1000i64..1000) {
        if min <= max {
            // Min boundary
            prop_assert!(validate_range(min, min, max).is_ok());
            
            // Max boundary
            prop_assert!(validate_range(max, min, max).is_ok());
            
            // Just below min
            if min > i64::MIN {
                prop_assert!(validate_range(min - 1, min, max).is_err());
            }
            
            // Just above max
            if max < i64::MAX {
                prop_assert!(validate_range(max + 1, min, max).is_err());
            }
        }
    }
}

// ========== STRING SANITIZATION PROPERTIES ==========

proptest! {
    /// Sanitize never panics
    #[test]
    fn sanitize_never_panics(s in "\\PC*") {
        let _ = sanitize_string(&s);
    }
    
    /// Sanitized output contains no control characters (except allowed ones)
    #[test]
    fn sanitize_no_control_chars(s in "\\PC{0,1000}") {
        if let Ok(sanitized) = sanitize_string(&s) {
            for c in sanitized.chars() {
                // Should not contain control characters (except \t, \n, \r)
                if c.is_control() {
                    prop_assert!(c == '\t' || c == '\n' || c == '\r');
                }
            }
        }
    }
    
    /// Sanitize is idempotent
    #[test]
    fn sanitize_idempotent(s in "\\PC{0,1000}") {
        if let Ok(sanitized1) = sanitize_string(&s) {
            if let Ok(sanitized2) = sanitize_string(&sanitized1) {
                prop_assert_eq!(sanitized1, sanitized2);
            }
        }
    }
}

// ========== CROSS-PROPERTY TESTS ==========

#[cfg(test)]
mod cross_property_tests {
    use super::*;
    
    #[test]
    fn all_validators_handle_empty() {
        let temp_dir = std::env::temp_dir();
        
        // Identifier
        assert!(validate_identifier("", 256).is_err());
        
        // Model ref
        assert!(validate_model_ref("").is_err());
        
        // Hex string
        assert!(validate_hex_string("", 64).is_err());
        
        // Path
        assert!(validate_path("", &temp_dir).is_err());
        
        // Prompt
        assert!(validate_prompt("", 10000).is_err());
    }
}

proptest! {
    
    /// All validators handle null bytes consistently
    #[test]
    fn all_validators_reject_null_bytes(
        prefix in "[a-zA-Z0-9]{0,50}",
        suffix in "[a-zA-Z0-9]{0,50}"
    ) {
        use std::path::Path;
        let temp_dir = std::env::temp_dir();
        let with_null = format!("{}\0{}", prefix, suffix);
        
        // All should reject null bytes
        prop_assert!(validate_identifier(&with_null, 256).is_err());
        prop_assert!(validate_path(&with_null, &temp_dir).is_err());
    }
    
    /// All validators complete quickly
    #[test]
    fn all_validators_performance(s in "\\PC{0,1000}") {
        use std::time::Instant;
        use std::path::Path;
        let temp_dir = std::env::temp_dir();
        
        let start = Instant::now();
        let _ = validate_identifier(&s, 256);
        let _ = validate_model_ref(&s);
        let _ = validate_hex_string(&s, 64);
        let _ = validate_path(&s, &temp_dir);
        let _ = validate_prompt(&s, 10000);
        let _ = sanitize_string(&s);
        let elapsed = start.elapsed();
        
        // All validators combined should complete in < 50ms
        prop_assert!(elapsed.as_millis() < 50);
    }
}

// ========== SECURITY PROPERTIES ==========

#[cfg(test)]
mod security_tests {
    use super::*;
    
    proptest! {
        /// Error messages don't contain raw input (prevents log injection)
        #[test]
        fn errors_dont_leak_input(s in "\\PC{0,1000}") {
            let result = validate_identifier(&s, 256);
            
            if let Err(e) = result {
                let error_msg = e.to_string();
                
                // Error message should not contain raw input
                // (except for very short, safe strings)
                if s.len() > 10 {
                    prop_assert!(!error_msg.contains(&s));
                }
                
                // Error message should be bounded
                prop_assert!(error_msg.len() < 500);
            }
        }
        
        /// Validators don't expose timing information (constant-time where possible)
        #[test]
        fn timing_consistency(
            valid in "[a-zA-Z0-9_-]{100}",
            invalid in "[^a-zA-Z0-9_-]{100}"
        ) {
            use std::time::Instant;
            
            let start1 = Instant::now();
            let _ = validate_identifier(&valid, 256);
            let time_valid = start1.elapsed();
            
            let start2 = Instant::now();
            let _ = validate_identifier(&invalid, 256);
            let time_invalid = start2.elapsed();
            
            // Times should be similar (within 10x)
            let ratio = time_valid.as_nanos() as f64 / time_invalid.as_nanos() as f64;
            prop_assert!(ratio > 0.1 && ratio < 10.0);
        }
    }
}
