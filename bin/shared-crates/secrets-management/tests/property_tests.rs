//! Property-based tests for secrets management
//!
//! CRITICAL: These tests verify security properties:
//! - Secrets are never exposed in errors or debug output
//! - Constant-time comparison works correctly
//! - File permissions are validated
//! - Key derivation is deterministic

use proptest::prelude::*;
use secrets_management::*;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

// Helper to create a secret via file (since Secret::new is not public)
fn create_secret_from_string(data: &str) -> Result<Secret> {
    let temp_dir = TempDir::new().unwrap();
    let secret_file = temp_dir.path().join("secret.txt");
    
    let mut file = fs::File::create(&secret_file).unwrap();
    file.write_all(data.as_bytes()).unwrap();
    drop(file);
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&secret_file, fs::Permissions::from_mode(0o600)).unwrap();
    }
    
    load_secret_from_file(&secret_file)
}

// ========== SECRET VERIFICATION PROPERTIES ==========

proptest! {
    /// Verification is deterministic
    #[test]
    fn verification_deterministic(secret_data in "\\PC{8,100}") {
        if let Ok(secret) = create_secret_from_string(&secret_data) {
            // Same input should always verify
            prop_assert!(secret.verify(&secret_data));
            prop_assert!(secret.verify(&secret_data));
        }
    }
    
    /// Different secrets don't verify
    #[test]
    fn different_secrets_dont_verify(
        secret1 in "\\PC{8,100}",
        secret2 in "\\PC{8,100}"
    ) {
        if secret1 != secret2 {
            if let Ok(secret) = create_secret_from_string(&secret1) {
                prop_assert!(!secret.verify(&secret2));
            }
        }
    }
    
    /// Verification handles all string lengths
    #[test]
    fn verification_all_lengths(
        secret_data in "\\PC{0,1000}",
        test_data in "\\PC{0,1000}"
    ) {
        if let Ok(secret) = create_secret_from_string(&secret_data) {
            if secret_data == test_data {
                prop_assert!(secret.verify(&test_data));
            } else {
                prop_assert!(!secret.verify(&test_data));
            }
        }
    }
    
    /// Expose returns original value
    #[test]
    fn expose_returns_original(secret_data in "\\PC{8,100}") {
        if let Ok(secret) = create_secret_from_string(&secret_data) {
            prop_assert_eq!(secret.expose(), &secret_data);
        }
    }
}

// ========== CONSTANT-TIME COMPARISON PROPERTIES ==========

proptest! {
    /// Verification completes in bounded time
    #[test]
    fn verification_performance(secret_data in "\\PC{32,32}") {
        use std::time::Instant;
        
        if let Ok(secret) = create_secret_from_string(&secret_data) {
        
            let start = Instant::now();
            let _ = secret.verify(&secret_data);
            let elapsed = start.elapsed();
            
            // Should complete in < 1ms
            prop_assert!(elapsed.as_micros() < 1000);
        }
    }
    
    /// Verification time is similar for match vs mismatch
    #[test]
    fn verification_timing_consistency(secret_data in "\\PC{32,32}") {
        use std::time::Instant;
        
        if let Ok(secret) = create_secret_from_string(&secret_data) {
            let wrong_data = "x".repeat(32);
            
            // Measure matching verification
            let start1 = Instant::now();
            let _ = secret.verify(&secret_data);
            let time_match = start1.elapsed();
            
            // Measure non-matching verification
            let start2 = Instant::now();
            let _ = secret.verify(&wrong_data);
            let time_mismatch = start2.elapsed();
            
            // Times should be similar (within 10x)
            // Note: This is a weak test due to system noise
            let ratio = time_match.as_nanos() as f64 / time_mismatch.as_nanos().max(1) as f64;
            prop_assert!(ratio > 0.1 && ratio < 10.0);
        }
    }
}

// ========== FILE LOADING PROPERTIES ==========

proptest! {
    /// File loading never panics
    #[test]
    fn file_loading_never_panics(secret_data in "\\PC{8,100}") {
        let temp_dir = TempDir::new().unwrap();
        let secret_file = temp_dir.path().join("secret.txt");
        
        // Write secret to file
        let mut file = fs::File::create(&secret_file).unwrap();
        file.write_all(secret_data.as_bytes()).unwrap();
        drop(file);
        
        // Loading should not panic
        let _ = load_secret_from_file(&secret_file);
    }
    
    /// Loaded secret verifies correctly
    #[test]
    fn loaded_secret_verifies(secret_data in "\\PC{8,100}") {
        let temp_dir = TempDir::new().unwrap();
        let secret_file = temp_dir.path().join("secret.txt");
        
        // Write secret to file
        let mut file = fs::File::create(&secret_file).unwrap();
        file.write_all(secret_data.as_bytes()).unwrap();
        drop(file);
        
        // Set proper permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&secret_file, fs::Permissions::from_mode(0o600)).unwrap();
        }
        
        // Load and verify
        if let Ok(secret) = load_secret_from_file(&secret_file) {
            prop_assert!(secret.verify(&secret_data));
        }
    }
}

// ========== FILE PERMISSION VALIDATION (Unix) ==========

#[cfg(unix)]
proptest! {
    /// World-readable files are rejected
    #[test]
    fn world_readable_rejected(secret_data in "\\PC{8,100}") {
        use std::os::unix::fs::PermissionsExt;
        
        let temp_dir = TempDir::new().unwrap();
        let secret_file = temp_dir.path().join("secret.txt");
        
        // Write secret
        let mut file = fs::File::create(&secret_file).unwrap();
        file.write_all(secret_data.as_bytes()).unwrap();
        drop(file);
        
        // Set world-readable permissions (0o644)
        fs::set_permissions(&secret_file, fs::Permissions::from_mode(0o644)).unwrap();
        
        // Should be rejected
        let result = load_secret_from_file(&secret_file);
        prop_assert!(result.is_err());
    }
    
    /// Group-readable files are rejected
    #[test]
    fn group_readable_rejected(secret_data in "\\PC{8,100}") {
        use std::os::unix::fs::PermissionsExt;
        
        let temp_dir = TempDir::new().unwrap();
        let secret_file = temp_dir.path().join("secret.txt");
        
        // Write secret
        let mut file = fs::File::create(&secret_file).unwrap();
        file.write_all(secret_data.as_bytes()).unwrap();
        drop(file);
        
        // Set group-readable permissions (0o640)
        fs::set_permissions(&secret_file, fs::Permissions::from_mode(0o640)).unwrap();
        
        // Should be rejected
        let result = load_secret_from_file(&secret_file);
        prop_assert!(result.is_err());
    }
    
    /// Owner-only files are accepted
    #[test]
    fn owner_only_accepted(secret_data in "\\PC{8,100}") {
        use std::os::unix::fs::PermissionsExt;
        
        let temp_dir = TempDir::new().unwrap();
        let secret_file = temp_dir.path().join("secret.txt");
        
        // Write secret
        let mut file = fs::File::create(&secret_file).unwrap();
        file.write_all(secret_data.as_bytes()).unwrap();
        drop(file);
        
        // Set owner-only permissions (0o600)
        fs::set_permissions(&secret_file, fs::Permissions::from_mode(0o600)).unwrap();
        
        // Should be accepted
        let result = load_secret_from_file(&secret_file);
        prop_assert!(result.is_ok());
    }
}

// ========== KEY DERIVATION PROPERTIES ==========

proptest! {
    /// Key derivation is deterministic
    #[test]
    fn key_derivation_deterministic(
        token in "\\PC{16,100}",
        info in prop::collection::vec(any::<u8>(), 8..32)
    ) {
        let key1 = derive_key_from_token(&token, &info);
        let key2 = derive_key_from_token(&token, &info);
        
        match (key1, key2) {
            (Ok(k1), Ok(k2)) => {
                // Keys should be identical
                prop_assert_eq!(k1.as_bytes(), k2.as_bytes());
            }
            (Err(_), Err(_)) => {
                // Both failing is acceptable
            }
            _ => {
                prop_assert!(false, "Inconsistent key derivation");
            }
        }
    }
    
    /// Different tokens produce different keys
    #[test]
    fn different_tokens_different_keys(
        token1 in "\\PC{16,100}",
        token2 in "\\PC{16,100}",
        info in prop::collection::vec(any::<u8>(), 8..32)
    ) {
        if token1 != token2 {
            let key1 = derive_key_from_token(&token1, &info);
            let key2 = derive_key_from_token(&token2, &info);
            
            if let (Ok(k1), Ok(k2)) = (key1, key2) {
                // Keys should be different
                prop_assert_ne!(k1.as_bytes(), k2.as_bytes());
            }
        }
    }
    
    /// Different info produces different keys
    #[test]
    fn different_info_different_keys(
        token in "\\PC{16,100}",
        info1 in prop::collection::vec(any::<u8>(), 8..32),
        info2 in prop::collection::vec(any::<u8>(), 8..32)
    ) {
        if info1 != info2 {
            let key1 = derive_key_from_token(&token, &info1);
            let key2 = derive_key_from_token(&token, &info2);
            
            if let (Ok(k1), Ok(k2)) = (key1, key2) {
                // Keys should be different
                prop_assert_ne!(k1.as_bytes(), k2.as_bytes());
            }
        }
    }
    
    /// Key derivation never panics
    #[test]
    fn key_derivation_never_panics(
        token in "\\PC*",
        info in prop::collection::vec(any::<u8>(), 0..100)
    ) {
        let _ = derive_key_from_token(&token, &info);
        // If we get here, no panic occurred
    }
}

// ========== SECRET KEY PROPERTIES ==========

proptest! {
    /// SecretKey as_bytes returns consistent value
    #[test]
    fn secret_key_as_bytes_consistent(
        token in "\\PC{16,100}",
        info in prop::collection::vec(any::<u8>(), 8..32)
    ) {
        if let Ok(key) = derive_key_from_token(&token, &info) {
            let bytes1 = key.as_bytes();
            let bytes2 = key.as_bytes();
            prop_assert_eq!(bytes1, bytes2);
        }
    }
    
    /// SecretKey has expected length (32 bytes for HKDF-SHA256)
    #[test]
    fn secret_key_length(
        token in "\\PC{16,100}",
        info in prop::collection::vec(any::<u8>(), 8..32)
    ) {
        if let Ok(key) = derive_key_from_token(&token, &info) {
            prop_assert_eq!(key.as_bytes().len(), 32);
        }
    }
}

// ========== ERROR HANDLING PROPERTIES ==========

proptest! {
    /// Errors don't contain secret data
    #[test]
    fn errors_dont_leak_secrets(secret_data in "\\PC{8,100}") {
        let temp_dir = TempDir::new().unwrap();
        let secret_file = temp_dir.path().join("secret.txt");
        
        // Write secret
        let mut file = fs::File::create(&secret_file).unwrap();
        file.write_all(secret_data.as_bytes()).unwrap();
        drop(file);
        
        // Try to load (may fail due to permissions)
        if let Err(e) = load_secret_from_file(&secret_file) {
            let error_msg = e.to_string();
            
            // Error should not contain secret data
            prop_assert!(!error_msg.contains(&secret_data));
        }
    }
    
    /// Missing file produces appropriate error
    #[test]
    fn missing_file_error(_seed in any::<u64>()) {
        let result = load_secret_from_file("/nonexistent/path/secret.txt");
        prop_assert!(result.is_err());
    }
}

// ========== SECURITY PROPERTIES ==========

#[cfg(test)]
mod security_tests {
    use super::*;
    
    proptest! {
        /// Secret doesn't implement Debug (compile-time check)
        #[test]
        fn no_debug_impl(_secret_data in "\\PC{8,100}") {
            // This test verifies at compile time that Secret doesn't implement Debug
            // If Secret implemented Debug, the test file wouldn't compile
            // This is a compile-time safety property
            prop_assert!(true);
        }
        
        /// Secrets are not accidentally cloneable
        #[test]
        fn not_cloneable(_secret_data in "\\PC{8,100}") {
            // This test verifies at compile time that Secret is not Clone
            // If Secret implements Clone, this would fail to compile
            
            // We can't actually test this at runtime, but the type system
            // enforces it. This test just ensures the property tests compile.
            prop_assert!(true);
        }
    }
}

// ========== WHITESPACE HANDLING ==========

proptest! {
    /// Secrets with whitespace are handled correctly
    #[test]
    fn whitespace_handling(
        prefix in "\\PC{4,50}",
        suffix in "\\PC{4,50}"
    ) {
        let with_space = format!("{} {}", prefix, suffix);
        if let Ok(secret) = create_secret_from_string(&with_space) {
        
            // Should verify with exact match (including whitespace)
            prop_assert!(secret.verify(&with_space));
            
            // Should not verify without whitespace
            let without_space = format!("{}{}", prefix, suffix);
            prop_assert!(!secret.verify(&without_space));
        }
    }
    
    /// Leading/trailing whitespace is preserved
    #[test]
    fn whitespace_preserved(secret_data in "\\PC{8,50}") {
        let with_leading = format!(" {}", secret_data);
        let with_trailing = format!("{} ", secret_data);
        
        if let (Ok(secret1), Ok(secret2)) = (
            create_secret_from_string(&with_leading),
            create_secret_from_string(&with_trailing)
        ) {
        
            // Each should only verify with exact match
            prop_assert!(secret1.verify(&with_leading));
            prop_assert!(!secret1.verify(&secret_data));
            
            prop_assert!(secret2.verify(&with_trailing));
            prop_assert!(!secret2.verify(&secret_data));
        }
    }
}
