# Input Validation — Consumer Expectations

**Status**: Draft  
**Purpose**: Documents what other crates expect from `input-validation`  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document catalogs the expectations and dependencies that other llama-orch crates have on `input-validation`. It serves as a contract specification to guide implementation priorities and API stability.

**Consuming crates**:
- `bin/worker-orcd-crates/vram-residency` — Shard IDs, digests, GPU indices
- `bin/worker-orcd-crates/api` — Request validation (prompts, params)
- `bin/worker-orcd-crates/model-loader` — Model paths, GGUF validation
- `bin/queen-rbee` — Task requests (model_ref, task_id, prompts)
- `bin/pool-managerd` — Pool/node IDs, configuration paths
- `libs/catalog-core` — Model references, file paths
- `bin/shared-crates/secrets-management` — File path validation
- `bin/shared-crates/audit-logging` — Event data sanitization

---

## 1. Core Validation Primitives

### 1.1 Identifier Validation

**Required by**: All services (queen-rbee, pool-managerd, worker-orcd, vram-residency)

**Expected function**:
```rust
pub fn validate_identifier(s: &str, max_len: usize) -> Result<(), ValidationError>;
```

**Use cases**:
- **vram-residency**: Validate `shard_id` before sealing (max 256 chars)
- **queen-rbee**: Validate `task_id`, `session_id`, `correlation_id` (max 256 chars)
- **pool-managerd**: Validate `pool_id`, `node_id`, `worker_id` (max 256 chars)
- **worker-orcd**: Validate `job_id`, `handle_id` (max 256 chars)

**Validation rules expected**:
1. Not empty
2. Length <= max_len
3. Only alphanumeric + dash + underscore: `[a-zA-Z0-9_-]+`
4. No null bytes
5. No path traversal sequences (`../`, `./`)
6. No control characters

**Error cases expected**:
- `ValidationError::Empty`
- `ValidationError::TooLong { actual, max }`
- `ValidationError::InvalidCharacters { found }`
- `ValidationError::NullByte`
- `ValidationError::PathTraversal`

---

### 1.2 Model Reference Validation

**Required by**: queen-rbee, catalog-core, pool-managerd

**Expected function**:
```rust
pub fn validate_model_ref(s: &str) -> Result<(), ValidationError>;
```

**Use cases**:
- **queen-rbee**: Validate `model_ref` in task creation (SECURITY_AUDIT #10)
- **catalog-core**: Validate model references before catalog operations
- **pool-managerd**: Validate model references in pool configuration

**Validation rules expected**:
1. Length <= 512 chars
2. Not empty
3. Character whitelist: `[a-zA-Z0-9\-_/:\.]+`
4. No null bytes
5. No shell metacharacters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`
6. No path traversal sequences

**Attack scenarios prevented**:
- SQL injection: `"'; DROP TABLE models; --"`
- Command injection: `"model.gguf; rm -rf /"`
- Log injection: `"model\n[ERROR] Fake log"`
- Path traversal: `"file:../../../../etc/passwd"`

---

### 1.3 Hex String Validation

**Required by**: vram-residency, catalog-core, audit-logging

**Expected function**:
```rust
pub fn validate_hex_string(s: &str, expected_len: usize) -> Result<(), ValidationError>;
```

**Use cases**:
- **vram-residency**: Validate SHA-256 digests (64 hex chars)
- **catalog-core**: Validate model checksums
- **audit-logging**: Validate event IDs, signatures

**Validation rules expected**:
1. Exact length match
2. Only hex characters: `[0-9a-fA-F]+`
3. No whitespace
4. No null bytes
5. Case-insensitive

**Common lengths**:
- SHA-256: 64 chars
- SHA-1: 40 chars
- MD5: 32 chars

---

### 1.4 Filesystem Path Validation

**Required by**: catalog-core, model-loader, secrets-management, pool-managerd

**Expected function**:
```rust
pub fn validate_path(
    path: impl AsRef<Path>,
    allowed_root: &Path
) -> Result<PathBuf, ValidationError>;
```

**Use cases**:
- **catalog-core**: Validate model file paths (SECURITY_AUDIT #9)
- **model-loader**: Validate model loading paths
- **secrets-management**: Validate key file paths
- **pool-managerd**: Validate configuration file paths

**Validation rules expected**:
1. Canonicalize path (resolve `..`, symlinks)
2. Verify path is within `allowed_root`
3. No null bytes
4. Return canonicalized PathBuf

**Attack scenarios prevented**:
- Directory traversal: `"../../../../etc/passwd"`
- Symlink escape: `"/var/lib/llorch/models/../../etc/passwd"`

---

### 1.5 Prompt Validation

**Required by**: worker-orcd (api crate), queen-rbee

**Expected function**:
```rust
pub fn validate_prompt(s: &str, max_len: usize) -> Result<(), ValidationError>;
```

**Use cases**:
- **worker-orcd/api**: Validate prompts in Execute endpoint (SECURITY_AUDIT #12)
- **queen-rbee**: Validate prompts in task creation

**Validation rules expected**:
1. Length <= max_len (default 100,000 chars)
2. No null bytes
3. Valid UTF-8 (enforced by `&str` type)

**Attack scenarios prevented**:
- VRAM exhaustion: 10MB prompt
- Null byte injection: `"prompt\0null"`
- Tokenizer exploits

---

### 1.6 Integer Range Validation

**Required by**: worker-orcd (api, vram-residency), queen-rbee

**Expected function**:
```rust
pub fn validate_range<T: PartialOrd + Display>(
    value: T,
    min: T,
    max: T
) -> Result<(), ValidationError>;
```

**Use cases**:
- **vram-residency**: Validate GPU device indices (0 to N-1)
- **worker-orcd/api**: Validate `max_tokens` parameter (0 to 4096)
- **queen-rbee**: Validate `ctx` (context length), `deadline_ms`

**Validation rules expected**:
1. `min <= value < max` (inclusive lower, exclusive upper)
2. No overflow

**Attack scenarios prevented**:
- Integer overflow: `max_tokens: usize::MAX`
- Invalid GPU index: `gpu_device: 999`

---

### 1.7 String Sanitization

**Required by**: All services (for logging), audit-logging

**Expected function**:
```rust
pub fn sanitize_string(s: &str) -> Result<String, ValidationError>;
```

**Use cases**:
- **All services**: Sanitize before logging user input
- **audit-logging**: Sanitize event data before storage
- **vram-residency**: Sanitize shard IDs before logging

**Sanitization rules expected**:
1. Reject null bytes
2. Reject control characters (except `\t`, `\n`, `\r`)
3. Reject ANSI escape sequences (`\x1b[`)
4. Return sanitized string or error

**Attack scenarios prevented**:
- Log injection: `"text\n[ERROR] Fake log"`
- ANSI escape injection: `"text\x1b[31mRED"`
- Terminal control: `"text\r\nfake"`

---

## 2. Error Type Expectations

### 2.1 ValidationError Enum

**Required by**: All consumers

**Expected variants**:
```rust
pub enum ValidationError {
    Empty,
    TooLong { actual: usize, max: usize },
    InvalidCharacters { found: String },
    NullByte,
    PathTraversal,
    WrongLength { actual: usize, expected: usize },
    InvalidHex { char: char },
    OutOfRange { value: String, min: String, max: String },
    ControlCharacter { char: char },
    AnsiEscape,
    ShellMetacharacter { char: char },
    PathOutsideRoot { path: String },
    Io(String),
}
```

**Expected traits**:
- `Debug` — For error logging
- `Display` — For user-facing error messages (via `thiserror`)
- `Error` — Standard error trait
- `PartialEq`, `Eq` — For testing

**Error message expectations**:
- Specific (not vague)
- Actionable (tell user what's wrong)
- No sensitive data (don't leak input content)

---

## 3. Performance Expectations

### 3.1 Validation Overhead

**Expected performance**:
- Identifier validation: < 1μs for typical inputs (< 100 chars)
- Model ref validation: < 2μs for typical inputs (< 200 chars)
- Hex validation: < 1μs for 64-char strings
- Path validation: < 10μs (includes filesystem I/O)
- Prompt validation: < 10μs for typical prompts (< 1000 chars)

**Performance requirements**:
- O(n) complexity for string validation (where n = length)
- O(1) complexity for range validation
- No allocations during validation (only for sanitization)
- Early return on first invalid character

---

### 3.2 No Panics

**Critical requirement**: Validation functions MUST NEVER panic.

**Rationale**:
- Used in request handlers (panic = DoS)
- Used in security-critical paths
- TIER 2 Clippy enforcement denies panics

**Expected behavior**:
```rust
// ✅ CORRECT: Returns Result
validate_identifier(untrusted_input, 256)?;

// ❌ FORBIDDEN: Panics
validate_identifier(untrusted_input, 256).unwrap();
```

---

## 4. Integration Patterns

### 4.1 vram-residency Integration

**Expected usage**:
```rust
use input_validation::{validate_identifier, validate_hex_string, validate_range};

impl VramManager {
    pub fn seal_model(
        &mut self,
        shard_id: String,
        gpu_device: u32,
        model_bytes: &[u8],
    ) -> Result<SealedShard> {
        // Validate shard ID
        validate_identifier(&shard_id, 256)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Validate GPU device
        let gpu_count = get_gpu_count()?;
        validate_range(gpu_device, 0, gpu_count)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Compute and validate digest
        let digest = compute_sha256(model_bytes);
        validate_hex_string(&digest, 64)
            .map_err(|e| VramError::InvalidInput(e.to_string()))?;
        
        // Proceed with sealing...
    }
}
```

---

### 4.2 queen-rbee Integration

**Expected usage**:
```rust
use input_validation::{validate_model_ref, validate_prompt, validate_identifier};

pub async fn create_task(
    Json(body): Json<TaskRequest>
) -> Result<impl IntoResponse> {
    // Validate task ID
    validate_identifier(&body.task_id, 256)
        .map_err(|e| ErrO::InvalidParams(e.to_string()))?;
    
    // Validate model_ref (SECURITY_AUDIT #10)
    validate_model_ref(&body.model_ref)
        .map_err(|e| ErrO::InvalidParams(e.to_string()))?;
    
    // Validate prompt (SECURITY_AUDIT #12)
    validate_prompt(&body.prompt, 100_000)
        .map_err(|e| ErrO::InvalidParams(e.to_string()))?;
    
    // Proceed with task creation...
}
```

---

### 4.3 catalog-core Integration

**Expected usage**:
```rust
use input_validation::{validate_model_ref, validate_path};

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self> {
        // Validate model ref format
        validate_model_ref(s)?;
        
        if let Some(p) = s.strip_prefix("file:") {
            let path = PathBuf::from(p);
            let allowed_root = PathBuf::from("/var/lib/llorch/models");
            
            // Validate path (SECURITY_AUDIT #9)
            let validated = validate_path(&path, &allowed_root)
                .map_err(|e| CatalogError::InvalidRef(e.to_string()))?;
            
            return Ok(ModelRef::File { path: validated });
        }
        
        // Parse other formats...
    }
}
```

---

### 4.4 secrets-management Integration

**Expected usage**:
```rust
use input_validation::validate_path;

impl SecretKey {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let allowed_root = PathBuf::from("/etc/llorch/secrets");
        
        // Validate path before loading
        let validated_path = validate_path(&path, &allowed_root)
            .map_err(|e| SecretError::InvalidPath(e.to_string()))?;
        
        // Now safe to read
        let contents = fs::read_to_string(&validated_path)?;
        // ...
    }
}
```

---

### 4.5 audit-logging Integration

**Expected usage**:
```rust
use input_validation::sanitize_string;

impl AuditLogger {
    pub async fn emit(&self, event: AuditEvent) -> Result<()> {
        // Sanitize user-controlled fields before logging
        let safe_shard_id = sanitize_string(&event.shard_id)
            .unwrap_or_else(|_| "[invalid]".to_string());
        
        let safe_model_ref = sanitize_string(&event.model_ref)
            .unwrap_or_else(|_| "[invalid]".to_string());
        
        // Write to audit log
        self.write_event(&safe_shard_id, &safe_model_ref).await?;
        Ok(())
    }
}
```

---

## 5. Testing Expectations

### 5.1 Unit Test Coverage

**Expected test coverage**:
- All validation functions have positive and negative tests
- All error variants are tested
- Edge cases covered (empty, max length, boundary values)
- Attack scenarios from security audits are tested

**Example tests expected**:
```rust
#[test]
fn test_identifier_rejects_traversal() {
    assert!(validate_identifier("shard-../etc/passwd", 256).is_err());
}

#[test]
fn test_model_ref_rejects_shell_metacharacters() {
    assert!(validate_model_ref("model; rm -rf /").is_err());
}

#[test]
fn test_sanitize_rejects_ansi() {
    assert!(sanitize_string("text\x1b[31mred").is_err());
}
```

---

### 5.2 Property Testing

**Expected property tests**:
- Valid inputs never panic
- Valid inputs always return Ok
- Invalid inputs always return Err (never panic)
- Validation is deterministic (same input → same result)

**Example property tests expected**:
```rust
proptest! {
    #[test]
    fn valid_identifiers_never_panic(s in "[a-zA-Z0-9_-]{1,256}") {
        let _ = validate_identifier(&s, 256);
    }
    
    #[test]
    fn hex_strings_validated(s in "[0-9a-fA-F]{64}") {
        assert!(validate_hex_string(&s, 64).is_ok());
    }
}
```

---

### 5.3 Fuzzing

**Expected fuzz targets**:
- `validate_identifier` — Fuzz with arbitrary bytes
- `validate_model_ref` — Fuzz with arbitrary strings
- `validate_hex_string` — Fuzz with arbitrary strings
- `sanitize_string` — Fuzz with arbitrary bytes

**Fuzzing requirement**: No panics, no crashes, no hangs.

---

## 6. Documentation Expectations

### 6.1 Function Documentation

**Expected for each public function**:
- Purpose (what it validates)
- Arguments (with types and constraints)
- Returns (Ok/Err cases)
- Examples (valid and invalid inputs)
- Errors (which ValidationError variants)

**Example**:
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
/// // Valid
/// assert!(validate_identifier("shard-abc123", 256).is_ok());
///
/// // Invalid
/// assert!(validate_identifier("shard/../etc", 256).is_err());
/// ```
///
/// # Errors
/// * `ValidationError::Empty` - String is empty
/// * `ValidationError::TooLong` - Exceeds max_len
/// * `ValidationError::PathTraversal` - Contains `../` or `./`
pub fn validate_identifier(s: &str, max_len: usize) -> Result<(), ValidationError>;
```

---

### 6.2 Error Documentation

**Expected for ValidationError**:
- Each variant documented
- When it's returned
- How to fix it

---

## 7. Stability Expectations

### 7.1 API Stability

**Pre-1.0 status**: Breaking changes allowed.

**Expected breaking changes**:
- Add new validation functions
- Add new error variants
- Change validation rules (stricter or more lenient)
- Rename functions for clarity

**Migration support**: None required (pre-1.0).

---

### 7.2 Validation Rule Changes

**When validation rules change**:
- Document in CHANGELOG
- Add migration guide if needed
- Update all consumers

**Example**: If we decide to allow dots in identifiers:
```rust
// Before: [a-zA-Z0-9_-]+
// After:  [a-zA-Z0-9_\-.]+
```

---

## 8. Security Expectations

### 8.1 No Information Leakage

**Critical requirement**: Validation errors MUST NOT leak sensitive data.

**Examples**:
```rust
// ✅ GOOD: No data leak
ValidationError::TooLong { actual: 1000, max: 512 }

// ❌ BAD: Leaks content
ValidationError::Invalid { content: "secret-api-key-abc123" }
```

---

### 8.2 Timing-Safe Validation

**Not required for M0**, but consider for post-M0:
- Validation should not leak information via timing
- Especially important for secret/token validation

---

### 8.3 Denial of Service Prevention

**Critical requirement**: Validation MUST NOT be exploitable for DoS.

**Requirements**:
- O(n) complexity maximum
- No exponential backtracking (if using regex)
- Early termination on invalid input
- No unbounded loops

---

## 9. Dependency Expectations

### 9.1 Minimal Dependencies

**Expected dependencies**:
- `thiserror` — Error types (already in workspace)

**Optional dependencies**:
- `regex` — Only if absolutely necessary (prefer manual parsing)
- `unicode-normalization` — Only if Unicode support needed

**Rationale**: Minimize attack surface, reduce compile times.

---

### 9.2 No Async

**Requirement**: All validation functions MUST be synchronous.

**Rationale**:
- Validation is CPU-bound, not I/O-bound
- Simpler to use in both sync and async contexts
- No tokio dependency needed

**Exception**: Path validation may do filesystem I/O (canonicalization), but still synchronous.

---

## 10. Clippy Expectations

### 10.1 TIER 2 Configuration

**Expected Clippy lints**:
```rust
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::missing_errors_doc)]
```

**Rationale**: Input validation is security-critical; any panic is a DoS vector.

---

## 11. Implementation Priority

### Phase 1: M0 Essentials (Week 1)
1. ✅ `validate_identifier()` — Most widely used
2. ✅ `validate_hex_string()` — Needed by vram-residency
3. ✅ `validate_range()` — Simple, widely used
4. ✅ `ValidationError` enum — Foundation
5. ✅ Basic unit tests

### Phase 2: Security Critical (Week 1-2)
6. ✅ `validate_model_ref()` — Fixes SECURITY_AUDIT #10
7. ✅ `validate_path()` — Fixes SECURITY_AUDIT #9
8. ✅ `sanitize_string()` — Prevents log injection
9. ✅ Comprehensive unit tests
10. ✅ Property tests

### Phase 3: Worker Integration (Week 2)
11. ⬜ `validate_prompt()` — Fixes SECURITY_AUDIT #12
12. ⬜ Integration with vram-residency
13. ⬜ Integration with queen-rbee
14. ⬜ Integration with catalog-core

### Phase 4: Hardening (Week 3+)
15. ⬜ Fuzzing with cargo-fuzz
16. ⬜ Performance benchmarks
17. ⬜ Documentation polish
18. ⬜ Integration with all services

---

## 12. Open Questions

### Q1: Should we support Unicode in identifiers?
**Current**: ASCII-only (`[a-zA-Z0-9_-]+`)  
**Alternative**: Allow Unicode letters  
**Recommendation**: ASCII-only for M0 (simpler, safer)

### Q2: Should validation be configurable?
**Current**: Fixed rules  
**Alternative**: Configurable via builder pattern  
**Recommendation**: Fixed rules for M0, add configurability post-M0 if needed

### Q3: Should we cache validation results?
**Current**: No caching  
**Alternative**: Cache validated strings  
**Recommendation**: No caching for M0 (validation is cheap)

### Q4: Should we provide validation macros?
**Current**: Functions only  
**Alternative**: Compile-time validation macros  
**Recommendation**: Functions only for M0, consider macros post-M0

---

## 13. References

**Specifications**:
- `bin/shared-crates/input-validation/.specs/00_input-validation.md` — Main spec
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — IV-001 to IV-005
- `bin/worker-orcd/.specs/00_worker-orcd.md` — WORKER-4300-4305

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #9, #10, #18
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Vulnerability #12

**Standards**:
- OWASP Input Validation Cheat Sheet
- CWE-20 — Improper Input Validation
- CWE-22 — Path Traversal
- CWE-117 — Log Injection
- CWE-78 — OS Command Injection

---

**End of Expectations Document**
