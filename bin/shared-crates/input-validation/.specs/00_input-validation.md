# Input Validation SPEC — Centralized Input Sanitization (VALID-1xxx)

**Status**: Draft  
**Security Tier**: TIER 2 (High-Importance)  
**Last Updated**: 2025-10-01

---

## 0. Scope & Goals

This specification defines requirements for the `input-validation` shared crate, which provides centralized validation and sanitization for all user-controlled inputs across llama-orch services.

**Key objectives**:
- Prevent injection attacks (SQL, command, log, path traversal)
- Enforce length limits and character whitelists
- Sanitize strings before logging or storage
- Provide reusable validation primitives for all services

**Reference documents**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #9, #10, #18
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Vulnerability #12
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — IV-001 to IV-005

**Consumers**:
- `bin/worker-orcd-crates/vram-residency` — Shard IDs, digests, GPU indices
- `bin/queen-rbee` — model_ref, task_id, prompts
- `bin/pool-managerd` — pool_id, node_id, paths
- `libs/catalog-core` — Model references, file paths
- All services — General string sanitization

---

## 1. Threat Model

### 1.1 Attack Vectors

**Injection Attacks**:
- **SQL Injection**: `model_ref: "'; DROP TABLE models; --"`
- **Command Injection**: `model_ref: "model.gguf; rm -rf /"`
- **Log Injection**: `model_ref: "model\n[ERROR] Fake log entry"`
- **Path Traversal**: `model_ref: "file:../../../../etc/passwd"`
- **ANSI Escape Injection**: `task_id: "task\x1b[31mRED"`

**Resource Exhaustion**:
- **Length Attacks**: 10MB prompt → VRAM exhaustion
- **Integer Overflow**: `max_tokens: usize::MAX` → infinite loop

**Encoding Attacks**:
- **Null Byte Injection**: `shard_id: "shard\0null"` → C string truncation
- **Unicode Exploits**: Homoglyph attacks, normalization bypasses
- **Control Characters**: `\r\n` → log parsing confusion

### 1.2 Assets to Protect

**Primary Assets**:
1. **System integrity** — Prevent command execution, file access
2. **Log integrity** — Prevent log injection, ANSI escape attacks
3. **Data integrity** — Prevent SQL injection, path traversal
4. **Service availability** — Prevent resource exhaustion

---

## 2. Validation Requirements (RFC-2119)

### 2.1 Identifier Validation

**VALID-1001**: Identifiers (shard_id, task_id, pool_id, node_id) MUST be validated before use.

**VALID-1002**: Identifier validation MUST enforce:
- Maximum length (configurable, default 256 characters)
- Non-empty
- Character whitelist: `[a-zA-Z0-9_-]+`
- No null bytes (`\0`)
- No path traversal sequences (`../`, `./`, `..\\`, `.\\`)
- No control characters (ASCII 0-31 except whitespace)

**VALID-1003**: Identifier validation MUST return descriptive errors indicating which rule failed.

---

### 2.2 Model Reference Validation

**VALID-1010**: Model references MUST be validated before use in catalog, logs, or API calls.

**VALID-1011**: Model reference validation MUST enforce:
- Maximum length: 512 characters
- Non-empty
- Character whitelist: `[a-zA-Z0-9\-_/:\.]+` (alphanumeric, dash, underscore, slash, colon, dot)
- No null bytes
- No path traversal sequences
- No shell metacharacters (`;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`)

**VALID-1012**: Model references with `file:` prefix MUST have path component validated separately.

---

### 2.3 Hex String Validation

**VALID-1020**: Hex strings (digests, hashes) MUST be validated before use.

**VALID-1021**: Hex string validation MUST enforce:
- Exact length (specified by caller, e.g., 64 for SHA-256)
- Only hexadecimal characters: `[0-9a-fA-F]+`
- No whitespace
- No null bytes

**VALID-1022**: Hex validation MUST be case-insensitive.

---

### 2.4 Filesystem Path Validation

**VALID-1030**: Filesystem paths MUST be validated before use.

**VALID-1031**: Path validation MUST enforce:
- Canonicalization (resolve `..` and symlinks)
- Containment check (path must be within allowed root directory)
- No null bytes
- No path traversal sequences

**VALID-1032**: Path validation MUST reject:
- Absolute paths outside allowed root
- Symlinks pointing outside allowed root
- Paths containing `..` after canonicalization

---

### 2.5 Prompt Validation

**VALID-1040**: User prompts MUST be validated before inference.

**VALID-1041**: Prompt validation MUST enforce:
- Maximum length: 100,000 characters (configurable)
- No null bytes
- Valid UTF-8 encoding

**VALID-1042**: Prompt validation MAY reject:
- Excessive whitespace (>50% of content)
- Repeated characters (>1000 consecutive identical chars)

---

### 2.6 Integer Range Validation

**VALID-1050**: Integer parameters (max_tokens, GPU indices, timeouts) MUST be validated.

**VALID-1051**: Range validation MUST enforce:
- Value within specified min/max bounds
- No overflow or wraparound

**VALID-1052**: Range validation MUST use inclusive lower bound, exclusive upper bound: `min <= value < max`.

---

### 2.7 String Sanitization

**VALID-1060**: Strings MUST be sanitized before logging or displaying.

**VALID-1061**: Sanitization MUST remove or reject:
- Null bytes (`\0`)
- Control characters (ASCII 0-31 except `\t`, `\n`, `\r`)
- ANSI escape sequences (`\x1b[`)
- Unicode directional overrides (U+202E, etc.)

**VALID-1062**: Sanitization SHOULD preserve original string if possible, or return error if unsafe.

---

## 3. API Specification

### 3.1 Identifier Validation

```rust
/// Validate identifier (shard_id, task_id, pool_id, node_id)
///
/// # Arguments
/// * `s` - String to validate
/// * `max_len` - Maximum allowed length
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_identifier;
///
/// // Valid
/// assert!(validate_identifier("shard-abc123", 256).is_ok());
/// assert!(validate_identifier("task_gpu0", 256).is_ok());
///
/// // Invalid
/// assert!(validate_identifier("shard/../etc", 256).is_err());
/// assert!(validate_identifier("shard\0null", 256).is_err());
/// ```
pub fn validate_identifier(s: &str, max_len: usize) -> Result<(), ValidationError>;
```

**Validation rules**:
1. `s.is_empty()` → `ValidationError::Empty`
2. `s.len() > max_len` → `ValidationError::TooLong { actual, max }`
3. `s.contains('\0')` → `ValidationError::NullByte`
4. `s.contains("../") || s.contains("..\\")` → `ValidationError::PathTraversal`
5. Invalid chars → `ValidationError::InvalidCharacters { found }`

---

### 3.2 Model Reference Validation

```rust
/// Validate model reference
///
/// # Arguments
/// * `s` - Model reference string
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_model_ref;
///
/// // Valid
/// assert!(validate_model_ref("meta-llama/Llama-3.1-8B").is_ok());
/// assert!(validate_model_ref("hf:org/repo").is_ok());
/// assert!(validate_model_ref("file:models/model.gguf").is_ok());
///
/// // Invalid
/// assert!(validate_model_ref("model; rm -rf /").is_err());
/// assert!(validate_model_ref("model\n[ERROR]").is_err());
/// ```
pub fn validate_model_ref(s: &str) -> Result<(), ValidationError>;
```

**Validation rules**:
1. Length <= 512
2. Not empty
3. No null bytes
4. No shell metacharacters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`
5. Character whitelist: `[a-zA-Z0-9\-_/:\.]+`
6. No path traversal sequences

---

### 3.3 Hex String Validation

```rust
/// Validate hexadecimal string
///
/// # Arguments
/// * `s` - Hex string to validate
/// * `expected_len` - Expected length (e.g., 64 for SHA-256)
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_hex_string;
///
/// // Valid SHA-256 digest
/// let digest = "a".repeat(64);
/// assert!(validate_hex_string(&digest, 64).is_ok());
///
/// // Invalid
/// assert!(validate_hex_string("xyz", 64).is_err());  // Non-hex
/// assert!(validate_hex_string("abc", 64).is_err());  // Wrong length
/// ```
pub fn validate_hex_string(s: &str, expected_len: usize) -> Result<(), ValidationError>;
```

**Validation rules**:
1. `s.len() != expected_len` → `ValidationError::WrongLength { actual, expected }`
2. Non-hex char → `ValidationError::InvalidHex { char }`
3. Null byte → `ValidationError::NullByte`

---

### 3.4 Filesystem Path Validation

```rust
/// Validate filesystem path
///
/// # Arguments
/// * `path` - Path to validate
/// * `allowed_root` - Root directory path must be within
///
/// # Returns
/// * `Ok(PathBuf)` - Canonicalized path if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_path;
/// use std::path::PathBuf;
///
/// let allowed = PathBuf::from("/var/lib/llorch/models");
///
/// // Valid
/// let path = validate_path("model.gguf", &allowed)?;
///
/// // Invalid
/// assert!(validate_path("../../../etc/passwd", &allowed).is_err());
/// ```
pub fn validate_path(
    path: impl AsRef<Path>,
    allowed_root: &Path
) -> Result<PathBuf, ValidationError>;
```

**Validation rules**:
1. Canonicalize path (resolve `..`, symlinks)
2. Check `canonical.starts_with(allowed_root)`
3. Reject if outside allowed root

---

### 3.5 Prompt Validation

```rust
/// Validate user prompt
///
/// # Arguments
/// * `s` - Prompt string
/// * `max_len` - Maximum allowed length (default 100,000)
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_prompt;
///
/// // Valid
/// assert!(validate_prompt("Write a story about...", 100_000).is_ok());
///
/// // Invalid
/// assert!(validate_prompt("prompt\0null", 100_000).is_err());
/// assert!(validate_prompt(&"a".repeat(200_000), 100_000).is_err());
/// ```
pub fn validate_prompt(s: &str, max_len: usize) -> Result<(), ValidationError>;
```

**Validation rules**:
1. Length <= max_len
2. No null bytes
3. Valid UTF-8 (enforced by `&str` type)

---

### 3.6 Integer Range Validation

```rust
/// Validate integer is within range
///
/// # Arguments
/// * `value` - Value to validate
/// * `min` - Minimum allowed value (inclusive)
/// * `max` - Maximum allowed value (exclusive)
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(ValidationError)` with specific failure reason
///
/// # Examples
/// ```
/// use input_validation::validate_range;
///
/// // Valid
/// assert!(validate_range(2, 0, 4).is_ok());
///
/// // Invalid
/// assert!(validate_range(5, 0, 4).is_err());
/// assert!(validate_range(usize::MAX, 0, 100).is_err());
/// ```
pub fn validate_range<T: PartialOrd + Display>(
    value: T,
    min: T,
    max: T
) -> Result<(), ValidationError>;
```

**Validation rules**:
1. `value < min || value >= max` → `ValidationError::OutOfRange { value, min, max }`

---

### 3.7 String Sanitization

```rust
/// Sanitize string for safe logging/display
///
/// # Arguments
/// * `s` - String to sanitize
///
/// # Returns
/// * `Ok(String)` - Sanitized string if safe
/// * `Err(ValidationError)` if contains unsafe content
///
/// # Examples
/// ```
/// use input_validation::sanitize_string;
///
/// // Valid
/// assert_eq!(sanitize_string("normal text")?, "normal text");
///
/// // Invalid
/// assert!(sanitize_string("text\0null").is_err());
/// assert!(sanitize_string("text\x1b[31mred").is_err());
/// ```
pub fn sanitize_string(s: &str) -> Result<String, ValidationError>;
```

**Sanitization rules**:
1. Reject null bytes
2. Reject control characters (except `\t`, `\n`, `\r`)
3. Reject ANSI escape sequences
4. Return sanitized copy or error

---

## 4. Error Types

### 4.1 ValidationError Enum

```rust
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("string is empty")]
    Empty,
    
    #[error("string too long: {actual} chars (max {max})")]
    TooLong { actual: usize, max: usize },
    
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
    
    #[error("shell metacharacter found: {char}")]
    ShellMetacharacter { char: char },
    
    #[error("path outside allowed directory: {path}")]
    PathOutsideRoot { path: String },
    
    #[error("I/O error: {0}")]
    Io(String),
}
```

---

## 5. Security Properties

### 5.1 Injection Prevention

**SQL Injection**:
```rust
// ❌ BLOCKED
validate_model_ref("'; DROP TABLE models; --")?;
// Error: ShellMetacharacter { char: ';' }
```

**Command Injection**:
```rust
// ❌ BLOCKED
validate_model_ref("model.gguf; rm -rf /")?;
// Error: ShellMetacharacter { char: ';' }
```

**Log Injection**:
```rust
// ❌ BLOCKED
validate_model_ref("model\n[ERROR] Fake log")?;
// Error: ControlCharacter { char: '\n' }
```

---

### 5.2 Path Traversal Prevention

**Directory Traversal**:
```rust
// ❌ BLOCKED
validate_identifier("shard-../../../etc/passwd", 256)?;
// Error: PathTraversal

// ❌ BLOCKED
validate_path("../../../etc/passwd", &allowed_root)?;
// Error: PathOutsideRoot
```

---

### 5.3 Null Byte Prevention

**C String Truncation**:
```rust
// ❌ BLOCKED
validate_identifier("shard\0null", 256)?;
// Error: NullByte
```

---

### 5.4 ANSI Escape Prevention

**Terminal Injection**:
```rust
// ❌ BLOCKED
sanitize_string("text\x1b[31mRED")?;
// Error: AnsiEscape
```

---

### 5.5 Resource Exhaustion Prevention

**Length Attacks**:
```rust
// ❌ BLOCKED
validate_prompt(&"a".repeat(200_000), 100_000)?;
// Error: TooLong { actual: 200000, max: 100000 }
```

**Integer Overflow**:
```rust
// ❌ BLOCKED
validate_range(usize::MAX, 0, 4096)?;
// Error: OutOfRange
```

---

## 6. Implementation Guidelines

### 6.1 Performance

**Optimization strategies**:
- Early return on first invalid character
- Use `str::chars()` for UTF-8 safety (not `str::bytes()`)
- No allocations for validation (only for sanitization)
- Avoid regex where simple character checks suffice

**Expected performance**:
- Identifier validation: O(n) where n = string length
- Hex validation: O(n)
- Range validation: O(1)
- Path validation: O(n) + filesystem I/O

---

### 6.2 Error Messages

**Actionable errors**:
```rust
// ✅ GOOD: Specific error
ValidationError::InvalidCharacters { found: "!@#" }

// ❌ BAD: Vague error
ValidationError::Invalid
```

**No sensitive data in errors**:
```rust
// ✅ GOOD: No data leak
ValidationError::TooLong { actual: 1000, max: 512 }

// ❌ BAD: Leaks content
ValidationError::Invalid { content: "secret data..." }
```

---

### 6.3 Unicode Handling

**ASCII-only for identifiers**:
```rust
// Identifiers: ASCII only (simplifies validation)
validate_identifier("shard-abc123", 256)?;  // ✅
validate_identifier("shard-café", 256)?;    // ❌ Non-ASCII
```

**UTF-8 for prompts**:
```rust
// Prompts: Full UTF-8 support
validate_prompt("Write a story about café", 100_000)?;  // ✅
```

---

## 7. Testing Requirements

### 7.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_identifier_rejects_traversal() {
        assert_eq!(
            validate_identifier("shard-../etc/passwd", 256),
            Err(ValidationError::PathTraversal)
        );
    }
    
    #[test]
    fn test_identifier_rejects_null_bytes() {
        assert_eq!(
            validate_identifier("shard\0null", 256),
            Err(ValidationError::NullByte)
        );
    }
    
    #[test]
    fn test_identifier_accepts_valid() {
        assert!(validate_identifier("shard-abc123", 256).is_ok());
        assert!(validate_identifier("shard_gpu0", 256).is_ok());
    }
    
    #[test]
    fn test_model_ref_rejects_shell_metacharacters() {
        assert!(validate_model_ref("model; rm -rf /").is_err());
        assert!(validate_model_ref("model | cat").is_err());
        assert!(validate_model_ref("model && ls").is_err());
    }
    
    #[test]
    fn test_model_ref_rejects_log_injection() {
        assert!(validate_model_ref("model\n[ERROR] Fake").is_err());
        assert!(validate_model_ref("model\r\nFake").is_err());
    }
    
    #[test]
    fn test_hex_validates_length() {
        assert!(validate_hex_string("abc", 64).is_err());
        assert!(validate_hex_string(&"a".repeat(64), 64).is_ok());
    }
    
    #[test]
    fn test_hex_rejects_non_hex() {
        assert!(validate_hex_string("xyz123", 6).is_err());
        assert!(validate_hex_string("abc 123", 7).is_err());
    }
    
    #[test]
    fn test_path_rejects_traversal() {
        let allowed = PathBuf::from("/var/lib/llorch");
        assert!(validate_path("../../../etc/passwd", &allowed).is_err());
    }
    
    #[test]
    fn test_sanitize_rejects_ansi() {
        assert_eq!(
            sanitize_string("text\x1b[31mred"),
            Err(ValidationError::AnsiEscape)
        );
    }
    
    #[test]
    fn test_sanitize_rejects_control_chars() {
        assert!(sanitize_string("text\r\nfake").is_err());
    }
    
    #[test]
    fn test_range_validates_bounds() {
        assert!(validate_range(2, 0, 4).is_ok());
        assert!(validate_range(5, 0, 4).is_err());
        assert!(validate_range(-1, 0, 4).is_err());
    }
}
```

---

### 7.2 Property Tests

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
    
    #[test]
    fn test_ranges_validated(
        value in 0u32..100,
        min in 0u32..50,
        max in 50u32..100
    ) {
        let result = validate_range(value, min, max);
        if value >= min && value < max {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    }
}
```

---

### 7.3 Fuzzing

```rust
// fuzz/fuzz_targets/validate_identifier.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use input_validation::validate_identifier;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = validate_identifier(s, 256);
        // Should never panic
    }
});

// fuzz/fuzz_targets/validate_model_ref.rs
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = validate_model_ref(s);
        // Should never panic
    }
});
```

---

## 8. Integration with Services

### 8.1 queen-rbee

**Validate task creation**:
```rust
use input_validation::{validate_model_ref, validate_prompt};

pub async fn create_task(
    Json(body): Json<TaskRequest>
) -> Result<impl IntoResponse> {
    // Validate model_ref
    validate_model_ref(&body.model_ref)
        .map_err(|e| ErrO::InvalidParams(e.to_string()))?;
    
    // Validate prompt
    validate_prompt(&body.prompt, 100_000)
        .map_err(|e| ErrO::InvalidParams(e.to_string()))?;
    
    // Proceed...
}
```

---

### 8.2 vram-residency

**Validate shard IDs and digests**:
```rust
use input_validation::{validate_identifier, validate_hex_string};

impl VramManager {
    pub fn seal_model(
        &mut self,
        shard_id: String,
        model_bytes: &[u8],
    ) -> Result<SealedShard> {
        // Validate shard ID
        validate_identifier(&shard_id, 256)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Compute digest
        let digest = compute_sha256(model_bytes);
        
        // Validate digest format
        validate_hex_string(&digest, 64)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Proceed with sealing...
    }
}
```

---

### 8.3 catalog-core

**Validate model references and paths**:
```rust
use input_validation::{validate_model_ref, validate_path};

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self> {
        // Validate model ref format
        validate_model_ref(s)?;
        
        if let Some(p) = s.strip_prefix("file:") {
            let path = PathBuf::from(p);
            let allowed_root = PathBuf::from("/var/lib/llorch/models");
            
            // Validate path
            let validated = validate_path(&path, &allowed_root)?;
            return Ok(ModelRef::File { path: validated });
        }
        
        // Parse other formats...
    }
}
```

---

## 9. Clippy Configuration

### 9.1 TIER 2 Configuration

```rust
// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::missing_errors_doc)]
```

**Rationale**: Input validation is security-critical; any panic could be exploited for DoS.

---

## 10. Dependencies

**Required crates**:
- `thiserror` — Error types

**Optional crates**:
- `regex` — For complex pattern matching (avoid if possible)
- `unicode-normalization` — For Unicode handling (if needed)

**Prefer standard library** where possible to minimize dependencies.

---

## 11. Performance Benchmarks

### 11.1 Target Performance

**Validation overhead**:
- Identifier validation: < 1μs for typical inputs (< 100 chars)
- Model ref validation: < 2μs for typical inputs (< 200 chars)
- Hex validation: < 1μs for 64-char strings
- Path validation: < 10μs (includes filesystem I/O)

**Benchmark suite**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_validate_identifier(c: &mut Criterion) {
    c.bench_function("validate_identifier", |b| {
        b.iter(|| validate_identifier(black_box("shard-abc123"), 256))
    });
}

criterion_group!(benches, bench_validate_identifier);
criterion_main!(benches);
```

---

## 12. Traceability

**Code**:
- `bin/shared-crates/input-validation/src/lib.rs` — Main implementation
- `bin/shared-crates/input-validation/src/identifier.rs` — Identifier validation
- `bin/shared-crates/input-validation/src/model_ref.rs` — Model ref validation
- `bin/shared-crates/input-validation/src/path.rs` — Path validation
- `bin/shared-crates/input-validation/src/sanitize.rs` — String sanitization

**Tests**:
- `bin/shared-crates/input-validation/tests/` — Integration tests
- `bin/shared-crates/input-validation/fuzz/` — Fuzz targets

**Related specs**:
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — IV-001 to IV-005
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #9, #10, #18
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Vulnerability #12

---

## 13. Refinement Opportunities

**Post-M0 enhancements**:
- Unicode normalization (NFC, NFKC)
- Homoglyph detection
- Configurable validation profiles per service
- Validation caching for repeated inputs
- Custom validation rules via callbacks
- Validation metrics (track rejection rates)

---

**End of Specification**
