# Input Validation — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐⭐⭐ CRITICAL  
**Applies to**: `bin/shared-crates/input-validation/`

---

## 0. Why Property Testing for Input Validation?

**Input validation is the #1 attack vector** in most applications. Property-based testing with `proptest` helps ensure:

- ✅ Validators **never panic** on malicious input
- ✅ Length limits are **always enforced**
- ✅ Unicode handling is **safe** (no buffer overflows)
- ✅ Regex patterns don't cause **ReDoS** (Regular Expression Denial of Service)
- ✅ Error messages don't **leak sensitive data**

**Traditional unit tests**: Test known good/bad inputs  
**Property tests**: Test **millions of random inputs** to find edge cases

---

## 1. Critical Properties to Test

### 1.1 Never Panic (Most Important)

**Property**: Validators must never panic, regardless of input.

```rust
use proptest::prelude::*;

proptest! {
    /// Validator never panics on any UTF-8 string
    #[test]
    fn validate_model_name_never_panics(s in "\\PC*") {
        let _ = validate_model_name(&s);
        // If we get here, no panic occurred
    }
    
    /// Validator never panics on arbitrary bytes
    #[test]
    fn validate_never_panics_on_bytes(bytes in prop::collection::vec(any::<u8>(), 0..10000)) {
        if let Ok(s) = String::from_utf8(bytes) {
            let _ = validate_model_name(&s);
        }
    }
}
```

---

### 1.2 Length Limits Always Enforced

**Property**: If validation succeeds, output length ≤ max length.

```rust
proptest! {
    /// Length limits are never exceeded
    #[test]
    fn length_limits_enforced(s in "\\PC{0,10000}") {
        const MAX_LEN: usize = 256;
        
        match validate_with_max_length(&s, MAX_LEN) {
            Ok(validated) => {
                prop_assert!(validated.len() <= MAX_LEN);
                prop_assert!(validated.chars().count() <= MAX_LEN);
            }
            Err(_) => {
                // Rejection is acceptable
                prop_assert!(s.len() > MAX_LEN || !is_valid_format(&s));
            }
        }
    }
    
    /// Empty strings are handled correctly
    #[test]
    fn empty_string_handling(min_len in 0usize..100) {
        let result = validate_with_min_length("", min_len);
        if min_len == 0 {
            prop_assert!(result.is_ok());
        } else {
            prop_assert!(result.is_err());
        }
    }
}
```

---

### 1.3 Unicode Safety

**Property**: Unicode characters are handled safely (no buffer overflows).

```rust
proptest! {
    /// Unicode characters don't cause buffer issues
    #[test]
    fn unicode_safe(s in "[\\u{0}-\\u{10FFFF}]{0,1000}") {
        let _ = validate_model_name(&s);
        
        // If validation succeeds, string is still valid UTF-8
        if let Ok(validated) = validate_model_name(&s) {
            prop_assert!(validated.is_char_boundary(0));
            prop_assert!(validated.is_char_boundary(validated.len()));
        }
    }
    
    /// Emoji and special characters are handled
    #[test]
    fn emoji_handling(emoji_count in 0usize..100) {
        let s = "😀".repeat(emoji_count);
        let result = validate_model_name(&s);
        
        // Should either accept or reject, but never panic
        prop_assert!(result.is_ok() || result.is_err());
    }
    
    /// Null bytes are rejected
    #[test]
    fn null_bytes_rejected(s in "\\PC{0,100}") {
        let with_null = format!("{}\0{}", s, s);
        let result = validate_model_name(&with_null);
        
        // Must reject strings with null bytes
        prop_assert!(result.is_err());
    }
}
```

---

### 1.4 Regex DoS Protection

**Property**: Validation completes in reasonable time (no catastrophic backtracking).

```rust
use std::time::{Duration, Instant};

proptest! {
    /// Validation completes within timeout
    #[test]
    fn no_regex_dos(s in "\\PC{0,1000}") {
        let start = Instant::now();
        let _ = validate_model_name(&s);
        let elapsed = start.elapsed();
        
        // Should complete in < 100ms even for pathological input
        prop_assert!(elapsed < Duration::from_millis(100));
    }
    
    /// Repeated patterns don't cause exponential blowup
    #[test]
    fn repeated_patterns_safe(pattern in "[a-z]{1,10}", repeat in 1usize..100) {
        let s = pattern.repeat(repeat);
        let start = Instant::now();
        let _ = validate_model_name(&s);
        let elapsed = start.elapsed();
        
        // Linear time complexity
        prop_assert!(elapsed < Duration::from_millis(10 * repeat as u64));
    }
}
```

---

### 1.5 Error Messages Don't Leak Data

**Property**: Error messages never contain user input (prevents XSS, log injection).

```rust
proptest! {
    /// Error messages don't contain raw input
    #[test]
    fn error_messages_safe(s in "\\PC{0,1000}") {
        if let Err(e) = validate_model_name(&s) {
            let error_msg = e.to_string();
            
            // Error message should not contain raw input
            prop_assert!(!error_msg.contains(&s));
            
            // Error message should be bounded
            prop_assert!(error_msg.len() < 500);
            
            // No control characters in error messages
            prop_assert!(!error_msg.chars().any(|c| c.is_control()));
        }
    }
}
```

---

### 1.6 Idempotence

**Property**: Validating twice gives same result.

```rust
proptest! {
    /// Validation is idempotent
    #[test]
    fn validation_idempotent(s in "\\PC{0,1000}") {
        let result1 = validate_model_name(&s);
        let result2 = validate_model_name(&s);
        
        match (result1, result2) {
            (Ok(v1), Ok(v2)) => prop_assert_eq!(v1, v2),
            (Err(_), Err(_)) => {}, // Both errors is fine
            _ => prop_assert!(false, "Inconsistent validation results"),
        }
    }
    
    /// Validating validated input always succeeds
    #[test]
    fn double_validation(s in "\\PC{0,1000}") {
        if let Ok(validated) = validate_model_name(&s) {
            let result = validate_model_name(&validated);
            prop_assert!(result.is_ok());
        }
    }
}
```

---

## 2. Specific Validators to Test

### 2.1 Model Name Validation

```rust
proptest! {
    /// Model names follow expected format
    #[test]
    fn model_name_format(
        org in "[a-zA-Z0-9_-]{1,50}",
        model in "[a-zA-Z0-9_-]{1,50}"
    ) {
        let name = format!("{}/{}", org, model);
        let result = validate_model_name(&name);
        
        // Valid format should be accepted
        prop_assert!(result.is_ok());
    }
    
    /// Invalid characters are rejected
    #[test]
    fn model_name_rejects_invalid(s in "[^a-zA-Z0-9/_-]+") {
        let result = validate_model_name(&s);
        prop_assert!(result.is_err());
    }
}
```

---

### 2.2 ORCH-ID Validation

```rust
proptest! {
    /// ORCH-ID format is strictly enforced
    #[test]
    fn orch_id_format(
        prefix in "orch-(job|model|pool|node)",
        uuid in "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ) {
        let orch_id = format!("{}-{}", prefix, uuid);
        let result = validate_orch_id(&orch_id);
        prop_assert!(result.is_ok());
    }
    
    /// Invalid ORCH-IDs are rejected
    #[test]
    fn orch_id_rejects_invalid(s in "\\PC{0,100}") {
        if !s.starts_with("orch-") {
            let result = validate_orch_id(&s);
            prop_assert!(result.is_err());
        }
    }
}
```

---

### 2.3 VRAM Size Validation

```rust
proptest! {
    /// VRAM sizes are within bounds
    #[test]
    fn vram_size_bounds(vram_mb in 0usize..1_000_000) {
        let result = validate_vram_size(vram_mb);
        
        if vram_mb == 0 {
            prop_assert!(result.is_err()); // Zero VRAM invalid
        } else if vram_mb > 500_000 {
            prop_assert!(result.is_err()); // > 500GB unreasonable
        } else {
            prop_assert!(result.is_ok());
        }
    }
    
    /// VRAM calculations don't overflow
    #[test]
    fn vram_mb_to_bytes(vram_mb in 0usize..1_000_000) {
        if let Ok(validated) = validate_vram_size(vram_mb) {
            let bytes = validated.saturating_mul(1024).saturating_mul(1024);
            prop_assert!(bytes <= usize::MAX);
        }
    }
}
```

---

### 2.4 GPU Index Validation

```rust
proptest! {
    /// GPU indices are reasonable
    #[test]
    fn gpu_index_bounds(index in any::<u32>()) {
        let result = validate_gpu_index(index);
        
        if index < 256 {
            prop_assert!(result.is_ok()); // 0-255 is reasonable
        } else {
            prop_assert!(result.is_err()); // > 255 GPUs unreasonable
        }
    }
}
```

---

## 3. Implementation Guide

### 3.1 Add Proptest Dependency

```toml
# Cargo.toml
[dev-dependencies]
proptest.workspace = true
```

### 3.2 Create Test File

```bash
# Create property test file
touch tests/property_tests.rs
```

### 3.3 Test Structure

```rust
//! Property-based tests for input validation
//!
//! These tests verify that validators:
//! - Never panic on any input
//! - Enforce length limits
//! - Handle Unicode safely
//! - Complete in reasonable time
//! - Don't leak data in error messages

use proptest::prelude::*;
use input_validation::*;

proptest! {
    // Add tests here
}

#[cfg(test)]
mod model_names {
    use super::*;
    
    proptest! {
        // Model name specific tests
    }
}

#[cfg(test)]
mod orch_ids {
    use super::*;
    
    proptest! {
        // ORCH-ID specific tests
    }
}
```

---

## 4. Running Property Tests

```bash
# Run all tests (including property tests)
cargo test -p input-validation

# Run only property tests
cargo test -p input-validation --test property_tests

# Run with more cases (default is 256)
PROPTEST_CASES=10000 cargo test -p input-validation

# Run with specific seed (for reproducibility)
PROPTEST_SEED=12345 cargo test -p input-validation

# Generate minimal failing case
cargo test -p input-validation -- --nocapture
```

---

## 5. Expected Test Coverage

**Minimum requirements**:
- ✅ 10+ property tests
- ✅ All public validators covered
- ✅ Never-panic tests for all inputs
- ✅ Length limit tests
- ✅ Unicode safety tests
- ✅ Performance tests (no ReDoS)

**Target**: 20+ property tests covering all validation paths

---

## 6. Common Pitfalls

### ❌ Don't Do This

```rust
// BAD: Testing with only ASCII
proptest! {
    #[test]
    fn bad_test(s in "[a-z]+") {  // Only lowercase ASCII
        validate(&s);
    }
}
```

### ✅ Do This Instead

```rust
// GOOD: Testing with full Unicode range
proptest! {
    #[test]
    fn good_test(s in "\\PC*") {  // All printable Unicode
        validate(&s);
    }
}
```

---

## 7. Integration with BDD Tests

Property tests **complement** BDD tests:

**BDD tests** (Gherkin):
- Test business requirements
- Test known scenarios
- Human-readable specifications

**Property tests**:
- Test mathematical properties
- Find unknown edge cases
- Exhaustive input coverage

**Both are needed** for comprehensive testing!

---

## 8. Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/input-validation-ci.yml
- name: Run property tests
  run: |
    PROPTEST_CASES=1000 cargo test -p input-validation --test property_tests
    
- name: Check for test failures
  if: failure()
  run: |
    echo "Property test found a failing case!"
    echo "Check test output for minimal reproducing input"
```

---

## 9. Refinement Opportunities

### 9.1 Advanced Testing

**Future work**:
- Add fuzzing integration (cargo-fuzz)
- Test concurrent validation (multiple threads)
- Add performance benchmarks
- Test memory usage under load

### 9.2 Custom Strategies

**Future work**:
- Create custom generators for domain-specific formats
- Add weighted random generation (more likely to hit edge cases)
- Implement stateful property testing

---

## 10. Success Metrics

**Before property testing**:
- Unknown edge cases
- Potential panics on malicious input
- Unclear validation behavior

**After property testing**:
- ✅ Confidence in validator robustness
- ✅ No panics on any input
- ✅ Clear property guarantees
- ✅ Better error handling

---

## 11. References

- **Proptest Book**: https://proptest-rs.github.io/proptest/
- **Property-Based Testing**: https://hypothesis.works/articles/what-is-property-based-testing/
- **Input Validation OWASP**: https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html

---

**Priority**: ⭐⭐⭐⭐⭐ CRITICAL  
**Estimated Effort**: 2-3 days  
**Impact**: Prevents entire class of vulnerabilities  
**Status**: Recommended for immediate implementation
