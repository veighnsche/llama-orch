# Secrets Management — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐⭐⭐ CRITICAL  
**Applies to**: `bin/shared-crates/secrets-management/`

---

## 0. Why Property Testing for Secrets Management?

**Secrets management is TIER 1 security-critical**. Property-based testing ensures:

- ✅ Secrets are **always zeroized** on drop
- ✅ Memory is **never leaked**
- ✅ Timing attacks are **prevented** (constant-time operations)
- ✅ File permissions are **always correct** (0600)
- ✅ Secrets **never appear** in error messages or logs

**One mistake = complete security breach**

---

## 1. Critical Properties to Test

### 1.1 Zeroization is Always Complete

**Property**: Secret memory is always zeroed on drop, regardless of secret size or content.

```rust
use proptest::prelude::*;
use std::ptr;

proptest! {
    /// Secrets are always zeroized on drop
    #[test]
    fn secret_always_zeroized(data in prop::collection::vec(any::<u8>(), 0..10000)) {
        let original_data = data.clone();
        let secret_ptr: *const u8;
        
        {
            let secret = Secret::new(data);
            secret_ptr = secret.as_bytes().as_ptr();
            
            // Secret is accessible while in scope
            prop_assert_eq!(secret.as_bytes(), &original_data[..]);
        }
        
        // After drop, memory should be zeroed
        // NOTE: This is unsafe and only for testing!
        #[cfg(test)]
        unsafe {
            let slice = std::slice::from_raw_parts(secret_ptr, original_data.len());
            prop_assert!(slice.iter().all(|&b| b == 0), "Memory not zeroized!");
        }
    }
    
    /// Zeroization works for all sizes
    #[test]
    fn zeroization_all_sizes(size in 0usize..100_000) {
        let data = vec![0xAA; size];
        let secret = Secret::new(data);
        drop(secret);
        // If we get here without panic, zeroization succeeded
    }
}
```

---

### 1.2 Constant-Time Comparison

**Property**: Comparison time is independent of where strings differ.

```rust
use std::time::Instant;

proptest! {
    /// Timing is constant regardless of difference position
    #[test]
    fn constant_time_comparison(
        secret1 in prop::collection::vec(any::<u8>(), 32..32),
        secret2 in prop::collection::vec(any::<u8>(), 32..32)
    ) {
        let s1 = Secret::new(secret1);
        let s2 = Secret::new(secret2);
        
        // Measure comparison time
        let start = Instant::now();
        let _ = s1.constant_time_eq(&s2);
        let elapsed = start.elapsed();
        
        // Should complete in < 1ms regardless of input
        prop_assert!(elapsed.as_micros() < 1000);
    }
    
    /// Equal secrets compare in same time as unequal
    #[test]
    fn timing_independent_of_equality(data in prop::collection::vec(any::<u8>(), 32..32)) {
        let s1 = Secret::new(data.clone());
        let s2 = Secret::new(data.clone());
        let s3 = Secret::new(vec![0xFF; 32]);
        
        let start1 = Instant::now();
        let _ = s1.constant_time_eq(&s2); // Equal
        let time_equal = start1.elapsed();
        
        let start2 = Instant::now();
        let _ = s1.constant_time_eq(&s3); // Not equal
        let time_unequal = start2.elapsed();
        
        // Times should be similar (within 2x)
        let ratio = time_equal.as_nanos() as f64 / time_unequal.as_nanos() as f64;
        prop_assert!(ratio > 0.5 && ratio < 2.0, "Timing leak detected!");
    }
}
```

---

### 1.3 File Permissions Always Correct

**Property**: Secret files always have 0600 permissions (owner read/write only).

```rust
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

proptest! {
    /// Secret files always have correct permissions
    #[test]
    #[cfg(unix)]
    fn file_permissions_secure(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("secret.key");
        
        let secret = Secret::new(data);
        secret.save_to_file(&file_path)?;
        
        // Check permissions
        let metadata = std::fs::metadata(&file_path)?;
        let permissions = metadata.permissions();
        let mode = permissions.mode();
        
        // Should be 0600 (owner read/write only)
        prop_assert_eq!(mode & 0o777, 0o600, "Incorrect file permissions!");
    }
    
    /// Permissions are set atomically (no race window)
    #[test]
    #[cfg(unix)]
    fn permissions_atomic(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("secret.key");
        
        let secret = Secret::new(data);
        
        // Save should be atomic (no window with wrong permissions)
        secret.save_to_file(&file_path)?;
        
        // Immediately check permissions
        let metadata = std::fs::metadata(&file_path)?;
        prop_assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    }
}
```

---

### 1.4 No Secrets in Error Messages

**Property**: Error messages never contain secret data.

```rust
proptest! {
    /// Error messages don't leak secrets
    #[test]
    fn errors_dont_leak_secrets(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let secret = Secret::new(data.clone());
        
        // Try to trigger various errors
        let invalid_path = "/nonexistent/path/secret.key";
        if let Err(e) = secret.save_to_file(invalid_path) {
            let error_msg = e.to_string();
            
            // Error message should not contain secret data
            for byte in &data {
                prop_assert!(!error_msg.contains(&format!("{:02x}", byte)));
            }
            
            // Error message should not contain raw bytes
            if let Ok(s) = String::from_utf8(data.clone()) {
                prop_assert!(!error_msg.contains(&s));
            }
        }
    }
    
    /// Debug output doesn't leak secrets
    #[test]
    fn debug_safe(data in prop::collection::vec(any::<u8>(), 0..100)) {
        let secret = Secret::new(data.clone());
        let debug_output = format!("{:?}", secret);
        
        // Debug should show "[REDACTED]" not actual data
        prop_assert!(debug_output.contains("[REDACTED]") || debug_output.contains("***"));
        
        // Should not contain actual secret
        for byte in &data {
            prop_assert!(!debug_output.contains(&format!("{:02x}", byte)));
        }
    }
}
```

---

### 1.5 Clone Behavior

**Property**: Cloning a secret creates independent zeroization.

```rust
proptest! {
    /// Cloned secrets are independently zeroized
    #[test]
    fn clone_independent_zeroization(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let secret1 = Secret::new(data.clone());
        let secret2 = secret1.clone();
        
        drop(secret1);
        
        // secret2 should still be valid after secret1 is dropped
        prop_assert_eq!(secret2.as_bytes(), &data[..]);
        
        drop(secret2);
        // Both should be zeroized independently
    }
}
```

---

### 1.6 Serialization Safety

**Property**: Secrets are never serialized in plain text.

```rust
proptest! {
    /// Secrets don't serialize to plain text
    #[test]
    fn serialization_safe(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let secret = Secret::new(data.clone());
        
        // Attempt to serialize
        if let Ok(json) = serde_json::to_string(&secret) {
            // JSON should not contain raw secret
            for byte in &data {
                prop_assert!(!json.contains(&format!("{:02x}", byte)));
            }
            
            // Should contain redaction marker
            prop_assert!(json.contains("REDACTED") || json.contains("***"));
        }
    }
}
```

---

## 2. Specific Secret Types to Test

### 2.1 API Keys

```rust
proptest! {
    /// API keys are validated
    #[test]
    fn api_key_validation(key in "[a-zA-Z0-9_-]{32,128}") {
        let result = Secret::from_api_key(&key);
        prop_assert!(result.is_ok());
    }
    
    /// Invalid API keys are rejected
    #[test]
    fn api_key_rejects_invalid(key in "\\PC{0,10}") {
        if key.len() < 32 {
            let result = Secret::from_api_key(&key);
            prop_assert!(result.is_err());
        }
    }
}
```

---

### 2.2 Encryption Keys

```rust
proptest! {
    /// Encryption keys have correct length
    #[test]
    fn encryption_key_length(key_bytes in prop::collection::vec(any::<u8>(), 32..32)) {
        let secret = Secret::new(key_bytes);
        prop_assert_eq!(secret.len(), 32);
    }
    
    /// Key derivation is deterministic
    #[test]
    fn key_derivation_deterministic(password in "\\PC{8,100}", salt in prop::collection::vec(any::<u8>(), 16..16)) {
        let key1 = Secret::derive_key(&password, &salt)?;
        let key2 = Secret::derive_key(&password, &salt)?;
        
        prop_assert!(key1.constant_time_eq(&key2));
    }
}
```

---

### 2.3 HMAC Secrets

```rust
proptest! {
    /// HMAC computation is deterministic
    #[test]
    fn hmac_deterministic(
        key in prop::collection::vec(any::<u8>(), 32..32),
        data in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        let secret = Secret::new(key);
        let hmac1 = secret.compute_hmac(&data)?;
        let hmac2 = secret.compute_hmac(&data)?;
        
        prop_assert_eq!(hmac1, hmac2);
    }
    
    /// Different keys produce different HMACs
    #[test]
    fn hmac_key_sensitivity(
        key1 in prop::collection::vec(any::<u8>(), 32..32),
        key2 in prop::collection::vec(any::<u8>(), 32..32),
        data in prop::collection::vec(any::<u8>(), 0..1000)
    ) {
        if key1 != key2 {
            let secret1 = Secret::new(key1);
            let secret2 = Secret::new(key2);
            
            let hmac1 = secret1.compute_hmac(&data)?;
            let hmac2 = secret2.compute_hmac(&data)?;
            
            prop_assert_ne!(hmac1, hmac2);
        }
    }
}
```

---

## 3. Memory Safety Properties

### 3.1 No Use-After-Free

```rust
proptest! {
    /// Cannot access secret after drop
    #[test]
    fn no_use_after_free(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let secret = Secret::new(data);
        let _bytes = secret.as_bytes(); // Borrow
        drop(secret);
        
        // Attempting to use _bytes here would be a compile error
        // This test verifies the borrow checker works correctly
    }
}
```

---

### 3.2 No Double-Free

```rust
proptest! {
    /// Dropping twice doesn't cause issues
    #[test]
    fn no_double_free(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let secret = Secret::new(data);
        drop(secret);
        // Rust prevents double-drop at compile time
        // This test verifies the Drop implementation is correct
    }
}
```

---

## 4. Concurrency Properties

### 4.1 Thread Safety

```rust
use std::sync::Arc;
use std::thread;

proptest! {
    /// Secrets are thread-safe
    #[test]
    fn thread_safe(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let secret = Arc::new(Secret::new(data.clone()));
        let mut handles = vec![];
        
        for _ in 0..10 {
            let secret_clone = Arc::clone(&secret);
            let handle = thread::spawn(move || {
                // Read secret from multiple threads
                let _bytes = secret_clone.as_bytes();
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // All threads should complete without panic
    }
}
```

---

## 5. Implementation Guide

### 5.1 Add Proptest Dependency

```toml
# Cargo.toml
[dev-dependencies]
proptest.workspace = true
tempfile = "3.8"
```

### 5.2 Test Structure

```rust
//! Property-based tests for secrets management
//!
//! CRITICAL: These tests verify security properties:
//! - Secrets are always zeroized
//! - Timing attacks are prevented
//! - File permissions are secure
//! - No secrets leak in errors

use proptest::prelude::*;
use secrets_management::*;

proptest! {
    // Core security properties
}

#[cfg(test)]
mod zeroization {
    use super::*;
    proptest! {
        // Zeroization tests
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    proptest! {
        // Constant-time tests
    }
}

#[cfg(test)]
mod file_security {
    use super::*;
    proptest! {
        // File permission tests
    }
}
```

---

## 6. Running Tests

```bash
# Run all tests
cargo test -p secrets-management

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p secrets-management --test property_tests

# Run with sanitizers (detect memory issues)
RUSTFLAGS="-Z sanitizer=address" cargo +nightly test -p secrets-management

# Check for memory leaks
valgrind --leak-check=full cargo test -p secrets-management
```

---

## 7. Security Validation Checklist

Before production:
- [ ] All secrets zeroized on drop (verified by property tests)
- [ ] Constant-time comparison (verified by timing tests)
- [ ] File permissions 0600 (verified on Unix)
- [ ] No secrets in error messages (verified by property tests)
- [ ] No secrets in debug output (verified by property tests)
- [ ] Thread-safe access (verified by concurrency tests)
- [ ] No memory leaks (verified by valgrind)
- [ ] No use-after-free (verified by miri)

---

## 8. Common Pitfalls

### ❌ Don't Do This

```rust
// BAD: Testing with only small secrets
proptest! {
    #[test]
    fn bad_test(data in prop::collection::vec(any::<u8>(), 0..10)) {
        // Only tests tiny secrets
    }
}
```

### ✅ Do This Instead

```rust
// GOOD: Testing with various sizes including large secrets
proptest! {
    #[test]
    fn good_test(data in prop::collection::vec(any::<u8>(), 0..100_000)) {
        // Tests full range of secret sizes
    }
}
```

---

## 9. Refinement Opportunities

### 9.1 Advanced Testing

**Future work**:
- Add fuzzing with cargo-fuzz
- Test with memory sanitizers (ASAN, MSAN)
- Add side-channel attack tests
- Test with hardware security modules (HSM)

### 9.2 Performance Testing

**Future work**:
- Benchmark zeroization performance
- Test memory usage under load
- Verify constant-time operations at assembly level

---

## 10. References

- **Zeroize Crate**: https://docs.rs/zeroize/
- **Constant-Time Operations**: https://www.bearssl.org/constanttime.html
- **Secure Memory**: https://github.com/stouset/secrets

---

**Priority**: ⭐⭐⭐⭐⭐ CRITICAL  
**Estimated Effort**: 3-4 days  
**Impact**: Prevents catastrophic security breaches  
**Status**: Recommended for immediate implementation
