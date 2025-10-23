# Input Validation — Security Specification

**Status**: Draft  
**Security Tier**: TIER 2 (High-Importance)  
**Last Updated**: 2025-10-01

---

## 0. Security Classification

### 0.1 Criticality Assessment

**Tier**: TIER 2 — High-Importance (Security-Critical)

**Rationale**:
- First line of defense against malicious input
- Used in all request handlers (panic = DoS)
- Prevents injection attacks across entire system
- Blocks path traversal and resource exhaustion
- Foundation for other security-critical crates

**Impact of compromise**:
- Remote code execution via command injection
- Data exfiltration via path traversal
- Service disruption via DoS (panics, resource exhaustion)
- Log poisoning via injection attacks
- Cross-service attacks via malformed identifiers

---

## 1. Threat Model

### 1.1 Adversary Capabilities

**External Attacker** (network access):
- Can send arbitrary strings to all validation functions
- Can attempt injection attacks (SQL, command, log, path)
- Can attempt resource exhaustion (length attacks, overflow)
- Can probe for validation bypasses
- Cannot directly access validation logic internals

**Compromised Service** (internal):
- Can call validation functions with malicious input
- Can attempt to bypass validation by not calling it
- Can try to exploit validation logic bugs
- Can attempt timing attacks to leak information

**Malicious Input Sources**:
- User prompts and task requests
- Model references and file paths
- Configuration values and identifiers
- API parameters and headers
- Log messages and event data

### 1.2 Assets to Protect

**Primary Assets**:
1. **System integrity** — Prevent command execution, arbitrary file access
2. **Service availability** — Prevent DoS via panics or resource exhaustion
3. **Log integrity** — Prevent log injection and ANSI escape attacks
4. **Data confidentiality** — Prevent path traversal and information leakage

**Secondary Assets**:
5. **Validation correctness** — Ensure no bypasses exist
6. **Performance** — Prevent validation from becoming bottleneck
7. **Error information** — Prevent sensitive data leakage in errors

---

## 2. Attack Surface Analysis

### 2.1 Injection Attack Surface

**Attack Vector**: Malicious strings containing special characters

**Vulnerable Functions**:
- `validate_model_ref()` — Shell metacharacters, SQL injection
- `validate_identifier()` — Path traversal, null bytes
- `sanitize_string()` — ANSI escapes, control characters

**Attack Scenarios**:

**SQL Injection**:
```rust
// Attack input
validate_model_ref("'; DROP TABLE models; --")?;

// Expected: Rejected with ShellMetacharacter error
// Risk if bypassed: Database compromise
```

**Command Injection**:
```rust
// Attack input
validate_model_ref("model.gguf; rm -rf /")?;

// Expected: Rejected with ShellMetacharacter error
// Risk if bypassed: Arbitrary command execution
```

**Log Injection**:
```rust
// Attack input
validate_model_ref("model\n[ERROR] Fake log entry")?;

// Expected: Rejected with ControlCharacter error
// Risk if bypassed: Log poisoning, SIEM evasion
```

**Path Traversal**:
```rust
// Attack input
validate_identifier("shard-../../../etc/passwd", 256)?;

// Expected: Rejected with PathTraversal error
// Risk if bypassed: Arbitrary file access
```

**ANSI Escape Injection**:
```rust
// Attack input
sanitize_string("text\x1b[31mRED\x1b[0m")?;

// Expected: Rejected with AnsiEscape error
// Risk if bypassed: Terminal hijacking, log obfuscation
```

**Mitigation**:
- Character whitelisting (not blacklisting)
- Explicit rejection of shell metacharacters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`
- Path traversal detection: `../`, `./`, `..\\`, `.\\`
- ANSI escape detection: `\x1b[`
- Null byte detection: `\0`

---

### 2.2 Resource Exhaustion Attack Surface

**Attack Vector**: Extremely long or malformed inputs

**Vulnerable Functions**:
- `validate_prompt()` — Length attacks
- `validate_identifier()` — Length attacks
- `validate_model_ref()` — Length attacks
- `validate_range()` — Integer overflow

**Attack Scenarios**:

**Length Attack (Memory Exhaustion)**:
```rust
// Attack input: 10MB prompt
let huge_prompt = "a".repeat(10_000_000);
validate_prompt(&huge_prompt, 100_000)?;

// Expected: Rejected with TooLong error
// Risk if bypassed: VRAM exhaustion, OOM crash
```

**Integer Overflow**:
```rust
// Attack input
validate_range(usize::MAX, 0, 4096)?;

// Expected: Rejected with OutOfRange error
// Risk if bypassed: Infinite loop, buffer overflow
```

**Algorithmic Complexity Attack**:
```rust
// Attack input: Pathological regex input (if using regex)
let evil = "a".repeat(1000) + "!";
validate_model_ref(&evil)?;

// Expected: O(n) validation, early termination
// Risk if bypassed: ReDoS (regex denial of service)
```

**Mitigation**:
- Strict length limits (configurable maximums)
- Early termination on invalid character
- O(n) complexity maximum (no exponential backtracking)
- No regex usage (avoid ReDoS)
- Saturating arithmetic for range checks

---

### 2.3 Encoding Attack Surface

**Attack Vector**: Special byte sequences and Unicode exploits

**Vulnerable Functions**:
- All validation functions (null bytes)
- `sanitize_string()` — Control characters, Unicode

**Attack Scenarios**:

**Null Byte Injection**:
```rust
// Attack input
validate_identifier("shard\0null", 256)?;

// Expected: Rejected with NullByte error
// Risk if bypassed: C string truncation, path traversal
```

**Unicode Normalization Bypass**:
```rust
// Attack input: Homoglyph attack
validate_identifier("shаrd-abc", 256)?;  // Cyrillic 'а' instead of 'a'

// Expected: Rejected with InvalidCharacters error (ASCII-only)
// Risk if bypassed: Identifier collision, confusion attacks
```

**Unicode Directional Override**:
```rust
// Attack input
sanitize_string("text\u{202E}reversed")?;

// Expected: Rejected (if implemented)
// Risk if bypassed: Display spoofing, log confusion
```

**UTF-8 Overlong Encoding**:
```rust
// Attack input: Overlong encoding of '/'
let overlong = b"\xC0\xAF";  // Overlong UTF-8 for '/'

// Expected: Invalid UTF-8, rejected by Rust's &str type
// Risk if bypassed: Path traversal via encoding bypass
```

**Mitigation**:
- ASCII-only for identifiers (no Unicode)
- UTF-8 validation enforced by Rust's `&str` type
- Explicit null byte detection
- Control character rejection
- Unicode directional override detection (post-M0)

---

### 2.4 Logic Bypass Attack Surface

**Attack Vector**: Edge cases and validation logic flaws

**Vulnerable Functions**:
- `validate_path()` — Symlink attacks, TOCTOU
- `validate_hex_string()` — Case sensitivity, length
- All functions — Empty string, boundary values

**Attack Scenarios**:

**Symlink Attack**:
```rust
// Attack: Create symlink inside allowed directory pointing outside
// /var/lib/llorch/models/evil -> /etc/passwd

let allowed = PathBuf::from("/var/lib/llorch/models");
validate_path("evil", &allowed)?;

// Expected: Canonicalize resolves symlink, rejects if outside root
// Risk if bypassed: Arbitrary file access
```

**TOCTOU (Time-of-Check-Time-of-Use)**:
```rust
// Attack: Change file after validation
let path = validate_path("model.gguf", &allowed)?;
// <Attacker replaces model.gguf with symlink>
let contents = fs::read(&path)?;

// Expected: Validation provides no TOCTOU protection (caller's responsibility)
// Risk: Race condition between validation and use
```

**Case Sensitivity Bypass**:
```rust
// Attack: Mixed case to bypass validation
validate_hex_string("ABC123", 6)?;

// Expected: Accepted (case-insensitive)
// Risk: None (intentional behavior)
```

**Empty String Edge Case**:
```rust
// Attack: Empty string
validate_identifier("", 256)?;

// Expected: Rejected with Empty error
// Risk if bypassed: Invalid identifiers in system
```

**Boundary Value Attack**:
```rust
// Attack: Exactly at limit
validate_identifier(&"a".repeat(256), 256)?;

// Expected: Accepted (exactly at limit)
// Risk: None (intentional behavior)

validate_identifier(&"a".repeat(257), 256)?;

// Expected: Rejected with TooLong error
// Risk if bypassed: Buffer overflow in downstream code
```

**Mitigation**:
- Path canonicalization (resolve symlinks)
- Containment check after canonicalization
- Explicit empty string rejection
- Boundary value testing (off-by-one)
- TOCTOU documented as caller's responsibility

---

### 2.5 Panic/DoS Attack Surface

**Attack Vector**: Inputs that cause panics or hangs

**Vulnerable Functions**:
- All validation functions (must never panic)

**Attack Scenarios**:

**Panic via Unwrap**:
```rust
// Vulnerable code (we must NOT do this)
fn validate_bad(s: &str) -> Result<()> {
    let first = s.chars().next().unwrap();  // ❌ Panics on empty string
    // ...
}

// Expected: All functions use ? or explicit error handling
// Risk if bypassed: DoS via panic
```

**Panic via Index**:
```rust
// Vulnerable code (we must NOT do this)
fn validate_bad(s: &str) -> Result<()> {
    let first = s[0];  // ❌ Panics on empty string or non-ASCII
    // ...
}

// Expected: Use .chars() or .get() with bounds checking
// Risk if bypassed: DoS via panic
```

**Panic via Integer Overflow**:
```rust
// Vulnerable code (we must NOT do this)
fn validate_bad(value: usize, max: usize) -> Result<()> {
    if value + 1 > max {  // ❌ Can overflow
        return Err(...);
    }
}

// Expected: Use saturating_add or checked_add
// Risk if bypassed: DoS via panic in debug mode
```

**Infinite Loop**:
```rust
// Vulnerable code (we must NOT do this)
fn validate_bad(s: &str) -> Result<()> {
    while s.contains("..") {  // ❌ Can loop forever
        s = s.replace("..", ".");
    }
}

// Expected: Single-pass validation with early termination
// Risk if bypassed: DoS via hang
```

**Mitigation**:
- TIER 2 Clippy: `#![deny(clippy::unwrap_used)]`
- TIER 2 Clippy: `#![deny(clippy::expect_used)]`
- TIER 2 Clippy: `#![deny(clippy::panic)]`
- TIER 2 Clippy: `#![warn(clippy::indexing_slicing)]`
- TIER 2 Clippy: `#![warn(clippy::integer_arithmetic)]`
- Fuzz testing to ensure no panics
- Property testing for all inputs

---

### 2.6 Information Leakage Attack Surface

**Attack Vector**: Error messages revealing sensitive data

**Vulnerable Functions**:
- All validation functions (error messages)

**Attack Scenarios**:

**Sensitive Data in Error**:
```rust
// Vulnerable code (we must NOT do this)
ValidationError::Invalid { 
    content: "secret-api-key-abc123"  // ❌ Leaks secret
}

// Expected: No input content in errors
ValidationError::TooLong { actual: 1000, max: 512 }  // ✅ Safe

// Risk if bypassed: Credential leakage via error logs
```

**Path Disclosure**:
```rust
// Vulnerable code (we must NOT do this)
ValidationError::PathOutsideRoot { 
    path: "/home/user/.ssh/id_rsa"  // ❌ Leaks filesystem structure
}

// Expected: Generic error or sanitized path
ValidationError::PathOutsideRoot { 
    path: "[redacted]"  // ✅ Safe
}

// Risk if bypassed: Information disclosure for further attacks
```

**Timing Attack**:
```rust
// Vulnerable code (we must NOT do this)
fn validate_secret(input: &str, expected: &str) -> bool {
    input == expected  // ❌ Early termination leaks length
}

// Expected: Not applicable for input validation (not comparing secrets)
// Risk: None for this crate (no secret comparison)
```

**Mitigation**:
- Error messages contain only metadata (lengths, types)
- No input content in errors
- No filesystem paths in errors (or sanitized)
- No timing-sensitive operations (not needed for validation)

---

### 2.7 Dependency Attack Surface

**Attack Vector**: Vulnerabilities in dependencies

**Current Dependencies**:
- `thiserror` — Error type derivation

**Attack Scenarios**:

**Compromised Dependency**:
```toml
# Risk: thiserror compromised
[dependencies]
thiserror = "1.0"  # What if this is malicious?
```

**Transitive Dependencies**:
```
input-validation
└── thiserror
    └── (no dependencies)
```

**Supply Chain Attack**:
- Typosquatting: `thiserr0r` instead of `thiserror`
- Version pinning attack: Force old vulnerable version
- Dependency confusion: Internal package name collision

**Mitigation**:
- Minimal dependencies (only `thiserror`)
- `thiserror` has zero dependencies (no transitive risk)
- Workspace-level dependency management
- Regular `cargo audit` in CI
- Dependency review before updates

---

## 3. Security Requirements (RFC-2119)

### 3.1 Panic Prevention

**SEC-VALID-001**: All validation functions MUST NEVER panic under any input.

**SEC-VALID-002**: All validation functions MUST use `?` operator or explicit error handling (no `.unwrap()`, `.expect()`).

**SEC-VALID-003**: All array/slice access MUST use `.get()` or bounds-checked iteration (no indexing `[]`).

**SEC-VALID-004**: All integer arithmetic MUST use saturating or checked operations (no unchecked `+`, `-`, `*`).

---

### 3.2 Injection Prevention

**SEC-VALID-010**: Model reference validation MUST reject shell metacharacters: `;`, `|`, `&`, `$`, `` ` ``, `\n`, `\r`.

**SEC-VALID-011**: Identifier validation MUST reject path traversal sequences: `../`, `./`, `..\\`, `.\\`.

**SEC-VALID-012**: All validation MUST reject null bytes (`\0`).

**SEC-VALID-013**: String sanitization MUST reject ANSI escape sequences (`\x1b[`).

**SEC-VALID-014**: String sanitization MUST reject control characters (ASCII 0-31 except `\t`, `\n`, `\r`).

---

### 3.3 Resource Protection

**SEC-VALID-020**: All string validation MUST enforce maximum length limits.

**SEC-VALID-021**: Validation complexity MUST be O(n) or better (no exponential backtracking).

**SEC-VALID-022**: Validation MUST terminate early on first invalid character.

**SEC-VALID-023**: Range validation MUST prevent integer overflow.

---

### 3.4 Path Security

**SEC-VALID-030**: Path validation MUST canonicalize paths (resolve `..` and symlinks).

**SEC-VALID-031**: Path validation MUST verify canonicalized path is within allowed root.

**SEC-VALID-032**: Path validation MUST reject paths outside allowed root after canonicalization.

---

### 3.5 Information Protection

**SEC-VALID-040**: Error messages MUST NOT contain input content.

**SEC-VALID-041**: Error messages MUST NOT leak filesystem paths (or sanitize them).

**SEC-VALID-042**: Error messages MUST be actionable without revealing sensitive data.

---

### 3.6 Encoding Safety

**SEC-VALID-050**: Identifier validation MUST be ASCII-only (reject non-ASCII).

**SEC-VALID-051**: All functions MUST rely on Rust's UTF-8 validation for `&str`.

**SEC-VALID-052**: Null byte detection MUST occur before other validation.

---

## 4. Secure Coding Guidelines

### 4.1 No Panics

**Principle**: Never panic, always return `Result`.

```rust
// ❌ BAD: Panics
pub fn validate_bad(s: &str) -> Result<()> {
    let first = s.chars().next().unwrap();
    // ...
}

// ✅ GOOD: Returns Result
pub fn validate_good(s: &str) -> Result<()> {
    let first = s.chars().next()
        .ok_or(ValidationError::Empty)?;
    // ...
}
```

---

### 4.2 Bounds Checking

**Principle**: Always check bounds before access.

```rust
// ❌ BAD: Unchecked indexing
let byte = bytes[offset];

// ✅ GOOD: Checked access
let byte = bytes.get(offset)
    .ok_or(ValidationError::InvalidInput)?;
```

---

### 4.3 Integer Safety

**Principle**: Use saturating or checked arithmetic.

```rust
// ❌ BAD: Can overflow
let total = a + b;

// ✅ GOOD: Saturating
let total = a.saturating_add(b);

// ✅ ALSO GOOD: Checked
let total = a.checked_add(b)
    .ok_or(ValidationError::Overflow)?;
```

---

### 4.4 Early Termination

**Principle**: Return error on first invalid character.

```rust
// ✅ GOOD: Early termination
for c in s.chars() {
    if !c.is_alphanumeric() && c != '-' && c != '_' {
        return Err(ValidationError::InvalidCharacters { 
            found: c.to_string() 
        });
    }
}
```

---

### 4.5 No Information Leakage

**Principle**: Errors contain only metadata, not content.

```rust
// ❌ BAD: Leaks content
return Err(ValidationError::Invalid { 
    content: s.to_string() 
});

// ✅ GOOD: Only metadata
return Err(ValidationError::TooLong { 
    actual: s.len(), 
    max: max_len 
});
```

---

## 5. Testing Requirements

### 5.1 Attack Scenario Tests

**Required tests for each attack vector**:

```rust
#[test]
fn test_sql_injection_blocked() {
    assert!(validate_model_ref("'; DROP TABLE models; --").is_err());
}

#[test]
fn test_command_injection_blocked() {
    assert!(validate_model_ref("model; rm -rf /").is_err());
}

#[test]
fn test_log_injection_blocked() {
    assert!(validate_model_ref("model\n[ERROR] Fake").is_err());
}

#[test]
fn test_path_traversal_blocked() {
    assert!(validate_identifier("shard-../etc/passwd", 256).is_err());
}

#[test]
fn test_ansi_escape_blocked() {
    assert!(sanitize_string("text\x1b[31mred").is_err());
}

#[test]
fn test_null_byte_blocked() {
    assert!(validate_identifier("shard\0null", 256).is_err());
}

#[test]
fn test_length_attack_blocked() {
    let huge = "a".repeat(1_000_000);
    assert!(validate_prompt(&huge, 100_000).is_err());
}

#[test]
fn test_integer_overflow_blocked() {
    assert!(validate_range(usize::MAX, 0, 100).is_err());
}
```

---

### 5.2 Panic Prevention Tests

**Required property tests**:

```rust
proptest! {
    #[test]
    fn validate_identifier_never_panics(s: String) {
        let _ = validate_identifier(&s, 256);
        // Should never panic
    }
    
    #[test]
    fn validate_model_ref_never_panics(s: String) {
        let _ = validate_model_ref(&s);
        // Should never panic
    }
    
    #[test]
    fn sanitize_string_never_panics(s: String) {
        let _ = sanitize_string(&s);
        // Should never panic
    }
}
```

---

### 5.3 Fuzzing

**Required fuzz targets**:

```rust
// fuzz/fuzz_targets/validate_identifier.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

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

// fuzz/fuzz_targets/validate_path.rs
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let allowed = PathBuf::from("/tmp");
        let _ = validate_path(s, &allowed);
        // Should never panic
    }
});
```

**Fuzzing requirements**:
- Run for minimum 1 hour per target
- No panics, no crashes, no hangs
- Coverage-guided fuzzing (libfuzzer)

---

## 6. Clippy Security Configuration

### 6.1 TIER 2 Configuration

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

**Rationale**: Input validation is security-critical; any panic is a DoS vector.

---

## 7. Incident Response

### 7.1 Validation Bypass Detected

**Detection**:
```rust
// If validation is bypassed, downstream code may detect it
if model_ref.contains(';') {
    // SECURITY INCIDENT: Validation bypass
    tracing::error!(
        security_event = "validation_bypass",
        input_type = "model_ref",
        "Validation bypass detected"
    );
}
```

**Response**:
1. Log security alert with context
2. Reject request immediately
3. Investigate validation logic for bug
4. Add regression test
5. Update validation rules if needed

---

### 7.2 Panic in Validation

**Detection**:
```
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value'
```

**Response**:
1. Identify panic location
2. Add test case reproducing panic
3. Fix with proper error handling
4. Run full fuzz suite
5. Deploy hotfix immediately

---

### 7.3 DoS via Validation

**Detection**:
```
High CPU usage in validation functions
Slow response times
```

**Response**:
1. Identify slow validation path
2. Add performance test
3. Optimize with early termination
4. Add complexity limits
5. Monitor validation performance

---

## 8. Security Review Checklist

### 8.1 Code Review

**Before merging**:
- [ ] All TIER 2 Clippy lints pass
- [ ] No `.unwrap()` or `.expect()` in code
- [ ] No `panic!()` in code
- [ ] All array access is bounds-checked
- [ ] All arithmetic uses saturating/checked operations
- [ ] No input content in error messages
- [ ] All attack scenarios have tests
- [ ] Property tests pass
- [ ] Fuzz tests run for 1+ hour without crashes

---

### 8.2 Security Testing

**Before release**:
- [ ] All injection attack tests pass
- [ ] All resource exhaustion tests pass
- [ ] All encoding attack tests pass
- [ ] All panic prevention tests pass
- [ ] Fuzzing completed (1+ hour per target)
- [ ] No security warnings from `cargo audit`

---

### 8.3 Documentation

**Before deployment**:
- [ ] Security spec reviewed and approved
- [ ] Attack surfaces documented
- [ ] Incident response procedures defined
- [ ] Integration examples include security notes

---

## 9. Known Limitations

### 9.1 TOCTOU (Time-of-Check-Time-of-Use)

**Limitation**: Path validation cannot prevent TOCTOU attacks.

**Example**:
```rust
// Validated path
let path = validate_path("model.gguf", &allowed)?;

// <Attacker replaces file here>

// Use path (may be different file now)
let contents = fs::read(&path)?;
```

**Mitigation**: Caller's responsibility to handle TOCTOU (e.g., open file atomically).

---

### 9.2 Unicode Normalization

**Limitation**: No Unicode normalization in M0.

**Example**:
```rust
// These are different strings but may look identical
"café"  // NFC normalized
"café"  // NFD normalized (separate combining accent)
```

**Mitigation**: ASCII-only for identifiers in M0. Add Unicode normalization post-M0 if needed.

---

### 9.3 Regex Denial of Service

**Limitation**: No regex used, so no ReDoS risk.

**Rationale**: Manual character checking is safer and faster than regex.

---

## 10. References

**Specifications**:
- `bin/shared-crates/input-validation/.specs/00_input-validation.md` — Main spec
- `bin/shared-crates/input-validation/.specs/10_expectations.md` — Consumer expectations

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #9, #10, #18
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Vulnerability #12

**Standards**:
- OWASP Input Validation Cheat Sheet
- CWE-20 — Improper Input Validation
- CWE-22 — Path Traversal
- CWE-78 — OS Command Injection
- CWE-117 — Log Injection
- CWE-400 — Resource Exhaustion

---

**End of Security Specification**
