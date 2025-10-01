# Input Validation — Worker VRAM Residency Requirements

**Consumer**: `bin/worker-orcd-crates/vram-residency`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies what `vram-residency` expects from the `input-validation` crate for secure input sanitization and validation.

**Context**: `vram-residency` handles user-controlled inputs (shard IDs, digests, GPU device indices) that must be validated to prevent injection attacks, path traversal, and other security issues.

**Reference**: 
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security spec (IV-001 to IV-005)
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations

---

## 1. Use Cases

### 1.1 Shard ID Validation

**Purpose**: Validate shard IDs are safe for use in logs, filenames, and API responses.

**Requirements** (from `20_security.md` IV-003):
- Max length: 256 characters
- Allowed characters: alphanumeric + dash + underscore
- No null bytes
- No path traversal sequences (`../`, `./`)
- No control characters

**Example inputs**:
```
✅ Valid:   "shard-abc123", "shard_gpu0_model1", "shard-0"
❌ Invalid: "shard/../etc/passwd", "shard\0null", "shard-" + "a" * 300
```

---

### 1.2 Digest String Validation

**Purpose**: Validate SHA-256 digest strings are properly formatted hex.

**Requirements** (from `20_security.md` IV-004):
- Exactly 64 characters (SHA-256 = 32 bytes = 64 hex chars)
- Only hexadecimal characters (0-9, a-f, A-F)
- No whitespace, no null bytes
- Case-insensitive

**Example inputs**:
```
✅ Valid:   "abc123def456789012345678901234567890123456789012345678901234"
❌ Invalid: "abc123", "xyz123...", "abc123\0def", "abc 123"
```

---

### 1.3 GPU Device Index Validation

**Purpose**: Validate GPU device indices are within valid range.

**Requirements** (from `20_security.md` IV-002):
- Must be non-negative integer
- Must be within available GPU count (0 to N-1)
- No overflow or wraparound

**Example inputs**:
```
✅ Valid:   0, 1, 2, 3 (if 4 GPUs available)
❌ Invalid: -1, 999, usize::MAX
```

---

### 1.4 String Sanitization

**Purpose**: Prevent null bytes and control characters in all string inputs.

**Requirements** (from `20_security.md` IV-005):
- No null bytes (`\0`)
- No control characters (ASCII 0-31 except whitespace)
- No ANSI escape sequences (prevent log injection)

**Example inputs**:
```
✅ Valid:   "normal string", "string with spaces"
❌ Invalid: "string\0null", "string\x1b[31mred", "string\r\n"
```

---

## 2. Required API

### 2.1 Identifier Validation

**Validate shard IDs, model refs, etc.**:
```rust
use input_validation::{validate_identifier, ValidationError};

// Validate shard ID
validate_identifier(&shard_id, 256)?;
```

**Function signature**:
```rust
pub fn validate_identifier(
    s: &str,
    max_len: usize
) -> Result<(), ValidationError>;
```

**Validation rules**:
1. Length <= max_len
2. Not empty
3. Only alphanumeric, dash, underscore: `[a-zA-Z0-9_-]+`
4. No null bytes
5. No path traversal sequences (`../`, `./`, `..\\`)

**Error cases**:
- `ValidationError::TooLong { actual, max }`
- `ValidationError::Empty`
- `ValidationError::InvalidCharacters { found }`
- `ValidationError::NullByte`
- `ValidationError::PathTraversal`

---

### 2.2 Hex String Validation

**Validate digest strings**:
```rust
use input_validation::validate_hex_string;

// Validate SHA-256 digest (64 hex chars)
validate_hex_string(&digest, 64)?;
```

**Function signature**:
```rust
pub fn validate_hex_string(
    s: &str,
    expected_len: usize
) -> Result<(), ValidationError>;
```

**Validation rules**:
1. Length == expected_len
2. Only hex characters: `[0-9a-fA-F]+`
3. No whitespace
4. No null bytes

**Error cases**:
- `ValidationError::WrongLength { actual, expected }`
- `ValidationError::InvalidHex { char }`
- `ValidationError::NullByte`

---

### 2.3 Integer Range Validation

**Validate GPU device indices**:
```rust
use input_validation::validate_range;

// Validate GPU device index
validate_range(gpu_device, 0, available_gpus)?;
```

**Function signature**:
```rust
pub fn validate_range<T: PartialOrd + Display>(
    value: T,
    min: T,
    max: T
) -> Result<(), ValidationError>;
```

**Validation rules**:
1. min <= value < max
2. No overflow

**Error cases**:
- `ValidationError::OutOfRange { value, min, max }`

---

### 2.4 String Sanitization

**Check for null bytes and control characters**:
```rust
use input_validation::sanitize_string;

// Sanitize before logging
let safe_string = sanitize_string(&user_input)?;
```

**Function signature**:
```rust
pub fn sanitize_string(s: &str) -> Result<String, ValidationError>;
```

**Sanitization rules**:
1. Reject null bytes (`\0`)
2. Reject control characters (ASCII 0-31 except `\t`, `\n`, `\r`)
3. Reject ANSI escape sequences (`\x1b[`)
4. Return sanitized string or error

**Error cases**:
- `ValidationError::NullByte`
- `ValidationError::ControlCharacter { char }`
- `ValidationError::AnsiEscape`

---

## 3. Error Type

```rust
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("string too long: {actual} chars (max {max})")]
    TooLong { actual: usize, max: usize },
    
    #[error("string is empty")]
    Empty,
    
    #[error("invalid characters found: {found}")]
    InvalidCharacters { found: String },
    
    #[error("null byte found in string")]
    NullByte,
    
    #[error("path traversal sequence detected")]
    PathTraversal,
    
    #[error("wrong length: {actual} (expected {expected})")]
    WrongLength { actual: usize, expected: usize },
    
    #[error("invalid hex character: {char}")]
    InvalidHex { char: char },
    
    #[error("value out of range: {value} (expected {min}..{max})")]
    OutOfRange { value: String, min: String, max: String },
    
    #[error("control character found: {char:?}")]
    ControlCharacter { char: char },
    
    #[error("ANSI escape sequence detected")]
    AnsiEscape,
}
```

---

## 4. Usage in vram-residency

### 4.1 Shard ID Validation

```rust
use input_validation::validate_identifier;

impl VramManager {
    pub fn seal_model(
        &mut self,
        model_bytes: &[u8],
        gpu_device: u32,
        shard_id: String,
    ) -> Result<SealedShard> {
        // Validate shard ID
        validate_identifier(&shard_id, 256)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Proceed with sealing...
    }
}
```

---

### 4.2 Digest Validation

```rust
use input_validation::validate_hex_string;

impl SealedShard {
    pub fn verify(&self, current_digest: &str) -> Result<()> {
        // Validate digest format
        validate_hex_string(current_digest, 64)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Compare digests
        if self.digest != current_digest {
            return Err(VramError::SealVerificationFailed);
        }
        
        Ok(())
    }
}
```

---

### 4.3 GPU Device Validation

```rust
use input_validation::validate_range;

impl VramManager {
    pub fn seal_model(
        &mut self,
        model_bytes: &[u8],
        gpu_device: u32,
    ) -> Result<SealedShard> {
        // Validate GPU device index
        let available_gpus = get_gpu_count()?;
        validate_range(gpu_device, 0, available_gpus)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Proceed with sealing...
    }
}
```

---

### 4.4 Logging Sanitization

```rust
use input_validation::sanitize_string;

impl VramManager {
    fn log_seal_operation(&self, shard: &SealedShard) {
        // Sanitize before logging
        let safe_shard_id = sanitize_string(&shard.shard_id)
            .unwrap_or_else(|_| "[invalid]".to_string());
        
        tracing::info!(
            shard_id = %safe_shard_id,
            gpu_device = %shard.gpu_device,
            vram_bytes = %shard.vram_bytes,
            "Model sealed in VRAM"
        );
    }
}
```

---

## 5. Security Properties

### 5.1 Prevent Path Traversal

**Reject traversal sequences**:
```rust
// ❌ REJECTED
validate_identifier("shard-../../../etc/passwd", 256)?;
// Error: PathTraversal

// ❌ REJECTED
validate_identifier("shard-./config", 256)?;
// Error: PathTraversal

// ✅ ACCEPTED
validate_identifier("shard-abc123", 256)?;
```

---

### 5.2 Prevent Null Byte Injection

**Reject null bytes**:
```rust
// ❌ REJECTED
validate_identifier("shard\0null", 256)?;
// Error: NullByte

// ✅ ACCEPTED
validate_identifier("shard-normal", 256)?;
```

---

### 5.3 Prevent Log Injection

**Reject ANSI escapes and control characters**:
```rust
// ❌ REJECTED
sanitize_string("shard\x1b[31mred")?;
// Error: AnsiEscape

// ❌ REJECTED
sanitize_string("shard\r\nFAKE LOG ENTRY")?;
// Error: ControlCharacter

// ✅ ACCEPTED
sanitize_string("shard-normal")?;
```

---

### 5.4 Prevent Integer Overflow

**Validate ranges**:
```rust
// ❌ REJECTED
validate_range(usize::MAX, 0, 4)?;
// Error: OutOfRange

// ✅ ACCEPTED
validate_range(2, 0, 4)?;
```

---

## 6. Testing Requirements

### 6.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_identifier_rejects_traversal() {
        assert!(validate_identifier("shard-../etc/passwd", 256).is_err());
        assert!(validate_identifier("shard-./config", 256).is_err());
    }
    
    #[test]
    fn test_identifier_rejects_null_bytes() {
        assert!(validate_identifier("shard\0null", 256).is_err());
    }
    
    #[test]
    fn test_identifier_accepts_valid() {
        assert!(validate_identifier("shard-abc123", 256).is_ok());
        assert!(validate_identifier("shard_gpu0", 256).is_ok());
    }
    
    #[test]
    fn test_hex_string_validates_length() {
        assert!(validate_hex_string("abc", 64).is_err());  // Too short
        assert!(validate_hex_string(&"a".repeat(64), 64).is_ok());
    }
    
    #[test]
    fn test_hex_string_rejects_non_hex() {
        assert!(validate_hex_string("xyz123", 6).is_err());
        assert!(validate_hex_string("abc 123", 7).is_err());
    }
    
    #[test]
    fn test_sanitize_rejects_ansi() {
        assert!(sanitize_string("text\x1b[31mred").is_err());
    }
    
    #[test]
    fn test_sanitize_rejects_control_chars() {
        assert!(sanitize_string("text\r\nfake").is_err());
    }
}
```

---

### 6.2 Property Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_valid_identifiers_never_panic(
        s in "[a-zA-Z0-9_-]{1,256}"
    ) {
        let _ = validate_identifier(&s, 256);
        // Should never panic
    }
    
    #[test]
    fn test_hex_strings_validated(
        s in "[0-9a-fA-F]{64}"
    ) {
        assert!(validate_hex_string(&s, 64).is_ok());
    }
}
```

---

## 7. Performance Considerations

### 7.1 Validation Overhead

**Expected performance**:
- Identifier validation: O(n) where n = string length
- Hex validation: O(n)
- Range validation: O(1)
- Sanitization: O(n)

**Optimization**:
- Use `str::chars()` for UTF-8 safety
- Early return on first invalid character
- No allocations for validation (only for sanitization)

---

### 7.2 Caching

**Not needed**: Validation is cheap enough to run on every input.

---

## 8. Dependencies

**Required crates**:
- `thiserror` — Error types
- `regex` (optional) — For complex pattern matching

**No external dependencies preferred** — Use standard library where possible.

---

## 9. Implementation Priority

### Phase 1: M0 Essentials
1. ✅ `validate_identifier()` with path traversal detection
2. ✅ `validate_hex_string()` for digests
3. ✅ `validate_range()` for GPU indices
4. ✅ `sanitize_string()` for logging
5. ✅ `ValidationError` type

### Phase 2: Hardening
6. ⬜ Property tests with proptest
7. ⬜ Fuzzing with cargo-fuzz
8. ⬜ Performance benchmarks
9. ⬜ Integration with vram-residency

---

## 10. Open Questions

**Q1**: Should we support Unicode in identifiers?  
**A**: No, ASCII-only for M0. Simplifies validation and prevents encoding attacks.

**Q2**: Should we normalize case (lowercase all identifiers)?  
**A**: No, preserve case. Validation is case-sensitive.

**Q3**: Should we support custom validation rules per field?  
**A**: Defer to post-M0. Fixed rules sufficient for M0.

---

## 11. References

**Specifications**:
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security requirements (IV-001 to IV-005)
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Issue #10 (model_ref validation)

**Standards**:
- OWASP Input Validation Cheat Sheet
- CWE-20 — Improper Input Validation
- CWE-22 — Path Traversal
- CWE-117 — Log Injection

---

**End of Requirements Document**
