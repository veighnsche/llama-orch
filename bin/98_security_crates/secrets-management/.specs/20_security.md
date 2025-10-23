# Secrets Management — Security Specification

**Status**: Draft  
**Security Tier**: TIER 1 (Critical Security)  
**Last Updated**: 2025-10-01

---

## 0. Security Classification

### 0.1 Criticality Assessment

**Tier**: TIER 1 — Critical Security (Maximum Security Enforcement)

**Rationale**:
- Handles all authentication credentials (API tokens, seal keys)
- Compromise enables complete system takeover
- Used by all services (queen-rbee, pool-managerd, worker-orcd)
- Foundation for authentication, encryption, and integrity
- Single point of failure for credential security

**Impact of compromise**:
- Complete authentication bypass (stolen API tokens)
- Cryptographic seal forgery (compromised seal keys)
- Data exfiltration (model theft via forged seals)
- Service impersonation (worker registration with stolen tokens)
- Lateral movement (credential reuse across services)
- Compliance violations (GDPR, SOC2, PCI-DSS)

---

## 1. Threat Model

### 1.1 Adversary Capabilities

**External Attacker** (network access):
- Can attempt to trigger secret logging via error injection
- Can probe for timing differences in verification
- Can attempt path traversal to read arbitrary files
- Can attempt to cause memory dumps to extract secrets
- Cannot directly access secret files (requires OS compromise)

**Compromised Service** (internal):
- Can call secrets-management API with malicious inputs
- Can attempt to bypass file permission validation
- Can attempt to extract secrets via error messages
- Can attempt timing attacks on verification
- Can attempt to prevent zeroization via panic injection

**Malicious Administrator** (host access):
- Can read secret files if permissions are wrong
- Can inspect process memory if secrets not zeroized
- Can view environment variables if secrets stored there
- Can inspect /proc filesystem for credential exposure
- Can trigger core dumps to extract secrets

**Supply Chain Attacker**:
- Can compromise dependencies (hkdf, zeroize, subtle)
- Can inject malicious code via compromised crates
- Can backdoor key derivation or verification logic

### 1.2 Assets to Protect

**Primary Assets**:
1. **API tokens** — Authentication credentials for all services
2. **Seal keys** — Cryptographic keys for VRAM shard integrity
3. **Worker tokens** — Worker registration credentials
4. **Derived keys** — Keys generated via HKDF from tokens

**Secondary Assets**:
5. **File paths** — Location of secret files (information disclosure)
6. **Key derivation parameters** — Domain separation strings
7. **Verification results** — Success/failure timing information

---

## 2. Attack Surface Analysis

### 2.1 Credential Exposure Attack Surface

**Attack Vector**: Secrets leaked via logs, errors, or process inspection

**Vulnerable Functions**:
- `Secret::load_from_file()` — May log file paths or errors
- `Secret::from_env()` — Exposes secrets in environment
- `SecretKey::derive_from_token()` — May log derivation parameters
- All error types — May contain secret values

**Attack Scenarios**:

**Environment Variable Exposure** (SEC-VULN-3):
```rust
// ❌ VULNERABLE: Visible in ps auxe
std::env::set_var("LLORCH_API_TOKEN", "secret-abc123");

// Attack
$ ps auxe | grep queen-rbee
# Output: LLORCH_API_TOKEN=secret-abc123

$ cat /proc/$(pidof queen-rbee)/environ
# LLORCH_API_TOKEN=secret-abc123
```

**Expected**: Never use environment variables for secrets
**Risk if bypassed**: Complete credential exposure

**Debug/Display Trait Leakage**:
```rust
// ❌ VULNERABLE: Accidental logging
let token = Secret::load_from_file("/etc/llorch/api-token")?;
tracing::info!("Loaded token: {:?}", token);  // Leaks if Debug implemented

// Expected: No Debug/Display traits on Secret types
// Risk if bypassed**: Secrets in logs, metrics, error traces
```

**Error Message Leakage**:
```rust
// ❌ VULNERABLE: Secret in error
return Err(SecretError::InvalidFormat(format!("bad token: {}", token)));

// ✅ SAFE: No secret in error
return Err(SecretError::InvalidFormat("expected 64 hex chars".to_string()));

// Risk if bypassed**: Secrets in error logs
```

**Memory Dump Exposure**:
```rust
// ❌ VULNERABLE: No zeroization
impl Drop for Secret {
    fn drop(&mut self) {
        // Secret remains in memory
    }
}

// ✅ SAFE: Zeroize on drop
impl Drop for Secret {
    fn drop(&mut self) {
        self.value.zeroize();
    }
}

// Risk if bypassed**: Secrets in core dumps, swap files
```

**Mitigation**:
- No Debug/Display/ToString/Serialize traits on secret types
- Error messages contain only metadata (lengths, paths)
- Zeroize on drop using `zeroize` crate with compiler fences
- Never use environment variables (or warn loudly)
- File paths logged, never values

---

### 2.2 File Permission Attack Surface

**Attack Vector**: World/group-readable secret files

**Vulnerable Functions**:
- `Secret::load_from_file()` — Must validate permissions
- `SecretKey::load_from_file()` — Must validate permissions

**Attack Scenarios**:

**World-Readable Secret File**:
```bash
# Vulnerable setup
echo "secret-token" > /etc/llorch/secrets/api-token
chmod 0644 /etc/llorch/secrets/api-token  # ❌ World-readable

# Attack
$ cat /etc/llorch/secrets/api-token
# secret-token

# Expected: Reject with PermissionsTooOpen error
# Risk if bypassed**: Any user can read secrets
```

**Group-Readable Secret File**:
```bash
# Vulnerable setup
chmod 0640 /etc/llorch/secrets/api-token  # ❌ Group-readable

# Attack (as group member)
$ cat /etc/llorch/secrets/api-token
# secret-token

# Expected: Reject with PermissionsTooOpen error
# Risk if bypassed**: Group members can read secrets
```

**Symlink to World-Readable File**:
```bash
# Attack
ln -s /etc/passwd /etc/llorch/secrets/api-token

# Expected: Canonicalize resolves symlink, validates target permissions
# Risk if bypassed**: Read arbitrary files
```

**Mitigation**:
- Validate file permissions before reading (Unix)
- Reject files with mode `0o077` bits set (world/group readable)
- Recommended permissions: `0600` (owner read/write only)
- Canonicalize paths to resolve symlinks
- Validate target file permissions after canonicalization

---

### 2.3 Path Traversal Attack Surface

**Attack Vector**: Malicious file paths accessing arbitrary files

**Vulnerable Functions**:
- `Secret::load_from_file()` — Path validation
- `SecretKey::load_from_file()` — Path validation

**Attack Scenarios**:

**Directory Traversal**:
```rust
// Attack input
let token = Secret::load_from_file("../../../etc/passwd")?;

// Expected: Canonicalize and reject if outside allowed directory
// Risk if bypassed**: Read arbitrary files
```

**Absolute Path Escape**:
```rust
// Attack input
let token = Secret::load_from_file("/etc/shadow")?;

// Expected: Validate path is within allowed directory
// Risk if bypassed**: Read arbitrary system files
```

**Symlink Escape**:
```rust
// Attack: Create symlink
// /etc/llorch/secrets/evil -> /etc/shadow

let token = Secret::load_from_file("/etc/llorch/secrets/evil")?;

// Expected: Canonicalize resolves symlink, validates target
// Risk if bypassed**: Read arbitrary files via symlink
```

**Null Byte Injection**:
```rust
// Attack input
let token = Secret::load_from_file("/etc/llorch/secrets/api-token\0/etc/passwd")?;

// Expected: Rust's Path type rejects null bytes
// Risk if bypassed**: C string truncation, read wrong file
```

**Mitigation**:
- Canonicalize all paths (resolve `..` and symlinks)
- Validate canonicalized path is within allowed directory (optional)
- Rust's `Path` type prevents null byte injection
- Explicit path validation before file operations

---

### 2.4 Timing Attack Surface

**Attack Vector**: Timing differences reveal secret information

**Vulnerable Functions**:
- `Secret::verify()` — Must use constant-time comparison

**Attack Scenarios**:

**Timing Attack on Verification**:
```rust
// ❌ VULNERABLE: Early termination
pub fn verify(&self, input: &str) -> bool {
    self.value == input  // Short-circuits on first mismatch
}

// Attack: Measure timing to guess token byte-by-byte
let mut guess = String::new();
for byte in 0..=255 {
    guess.push(byte as char);
    let start = Instant::now();
    token.verify(&guess);
    let duration = start.elapsed();
    // Longer duration = correct byte (matched further)
}

// ✅ SAFE: Constant-time comparison
pub fn verify(&self, input: &str) -> bool {
    use subtle::ConstantTimeEq;
    self.value.as_bytes().ct_eq(input.as_bytes()).into()
}

// Risk if bypassed**: Token recovery via timing side-channel
```

**Length Leakage**:
```rust
// ⚠️ ACCEPTABLE: Length check can short-circuit
if self.value.len() != input.len() {
    return false;  // Length is not secret
}

// Constant-time comparison of bytes
self.value.as_bytes().ct_eq(input.as_bytes()).into()
```

**Mitigation**:
- Use `subtle::ConstantTimeEq` for all secret comparisons
- Examine all bytes regardless of match status
- No short-circuit on first mismatch
- Length comparison can short-circuit (length is public)
- Prevents CWE-208 (Observable Timing Discrepancy)

---

### 2.5 Key Derivation Attack Surface

**Attack Vector**: Weak key derivation enables key recovery

**Vulnerable Functions**:
- `SecretKey::derive_from_token()` — HKDF implementation

**Attack Scenarios**:

**Weak KDF (SHA-256 only)**:
```rust
// ❌ VULNERABLE: Simple hash
pub fn derive_from_token(token: &str, domain: &[u8]) -> SecretKey {
    let hash = sha256(token.as_bytes());
    SecretKey(hash)
}

// Attack: Rainbow table attack, no salt
// Expected: Use HKDF with domain separation
// Risk if bypassed**: Key recovery via precomputation
```

**No Domain Separation**:
```rust
// ❌ VULNERABLE: Same key for all purposes
let seal_key = derive_from_token(&token, b"");
let enc_key = derive_from_token(&token, b"");
// seal_key == enc_key (bad!)

// ✅ SAFE: Domain separation
let seal_key = derive_from_token(&token, b"llorch-seal-key-v1");
let enc_key = derive_from_token(&token, b"llorch-encryption-v1");
// seal_key != enc_key (good!)

// Risk if bypassed**: Key reuse across contexts
```

**Insufficient Output Length**:
```rust
// ❌ VULNERABLE: 16 bytes (128-bit)
let mut key = [0u8; 16];
hkdf.expand(domain, &mut key)?;

// ✅ SAFE: 32 bytes (256-bit)
let mut key = [0u8; 32];
hkdf.expand(domain, &mut key)?;

// Risk if bypassed**: Brute-force attack feasible
```

**Mitigation**:
- Use HKDF-SHA256 (NIST SP 800-108 compliant)
- Always include domain separation string
- Output 32 bytes (256-bit keys)
- Deterministic derivation (same input → same output)
- No salt needed (token is high-entropy)

---

### 2.6 Systemd Credential Attack Surface

**Attack Vector**: Malicious systemd credential paths

**Vulnerable Functions**:
- `Secret::from_systemd_credential()` — Path construction
- `SecretKey::from_systemd_credential()` — Path construction

**Attack Scenarios**:

**Environment Variable Injection**:
```bash
# Attack: Malicious CREDENTIALS_DIRECTORY
export CREDENTIALS_DIRECTORY="/etc/shadow"

# Rust code
let token = Secret::from_systemd_credential("api_token")?;
# Reads /etc/shadow/api_token

# Expected: Validate CREDENTIALS_DIRECTORY format
# Risk if bypassed**: Read arbitrary files
```

**Path Traversal in Credential Name**:
```rust
// Attack input
let token = Secret::from_systemd_credential("../../../etc/passwd")?;

// Expected: Validate credential name (no path separators)
// Risk if bypassed**: Read arbitrary files
```

**Symlink in Credentials Directory**:
```bash
# Attack: Replace credentials directory with symlink
rm -rf /run/credentials/queen-rbee.service
ln -s /etc /run/credentials/queen-rbee.service

# Expected: Validate credentials directory is not symlink
# Risk if bypassed**: Read arbitrary files
```

**Mitigation**:
- Validate `$CREDENTIALS_DIRECTORY` is absolute path
- Validate credential name contains no path separators
- Canonicalize final path and validate
- Check credentials directory is not symlink (optional)

---

### 2.7 Panic/DoS Attack Surface

**Attack Vector**: Inputs that cause panics or hangs

**Vulnerable Functions**:
- All functions (must never panic)

**Attack Scenarios**:

**Panic via Unwrap**:
```rust
// ❌ VULNERABLE: Panics on error
pub fn load_from_file(path: &Path) -> SecretKey {
    let contents = fs::read_to_string(path).unwrap();  // Panics if file missing
    // ...
}

// ✅ SAFE: Returns Result
pub fn load_from_file(path: &Path) -> Result<SecretKey, SecretError> {
    let contents = fs::read_to_string(path)?;
    // ...
}

// Risk if bypassed**: DoS via panic
```

**Panic via Index**:
```rust
// ❌ VULNERABLE: Panics on empty string
let first_byte = hex_string.as_bytes()[0];

// ✅ SAFE: Bounds checking
let first_byte = hex_string.as_bytes().get(0)
    .ok_or(SecretError::InvalidFormat("empty string".to_string()))?;

// Risk if bypassed**: DoS via panic
```

**Panic via Integer Overflow**:
```rust
// ❌ VULNERABLE: Can overflow in debug mode
if bytes.len() + 1 > max_len {
    return Err(...);
}

// ✅ SAFE: Saturating arithmetic
if bytes.len().saturating_add(1) > max_len {
    return Err(...);
}

// Risk if bypassed**: DoS via panic in debug builds
```

**Infinite Loop**:
```rust
// ❌ VULNERABLE: Can loop forever
while contents.contains("  ") {
    contents = contents.replace("  ", " ");
}

// ✅ SAFE: Single-pass processing
let contents = contents.split_whitespace().collect::<Vec<_>>().join(" ");

// Risk if bypassed**: DoS via hang
```

**Mitigation**:
- TIER 1 Clippy: `#![deny(clippy::unwrap_used)]`
- TIER 1 Clippy: `#![deny(clippy::expect_used)]`
- TIER 1 Clippy: `#![deny(clippy::panic)]`
- TIER 1 Clippy: `#![deny(clippy::indexing_slicing)]`
- TIER 1 Clippy: `#![deny(clippy::integer_arithmetic)]`
- All functions return `Result`
- No unwrap/expect/panic in code

**Attack Scenarios**:

**Compromised Dependency**:
```toml
# Risk: zeroize compromised
[dependencies]
zeroize = "1.7"  # What if this is backdoored?
```

**Transitive Dependencies**:
```
secrets-management
├── zeroize (no dependencies)
├── thiserror (no dependencies)
├── hkdf
│   └── hmac
│       └── digest
│           └── crypto-common
├── sha2
│   └── digest
│       └── crypto-common
├── subtle (no dependencies)
├── hex (no dependencies)
└── tracing
    └── tracing-core
        └── once_cell
```

**Supply Chain Attack**:
- Typosquatting: `zer0ize` instead of `zeroize`
- Version pinning attack: Force old vulnerable version
- Dependency confusion: Internal package name collision

**Mitigation**:
- **Use battle-tested libraries**: Don't roll your own crypto
- **RustCrypto ecosystem**: `secrecy`, `zeroize`, `subtle`, `hkdf`, `sha2`, `hex`
- **Security audits**: RustCrypto crates are professionally audited
- **Minimal dependencies**: Most have zero or few transitive deps
- Workspace-level dependency management
- Regular `cargo audit` in CI
- Dependency review before updates
- Pin exact versions in Cargo.lock
- Monitor security advisories

**Why `secrecy` crate**:
- Industry-standard secret wrapper
- Automatic zeroization on drop
- No Debug/Display by default
- Used by thousands of production projects
- Better than rolling our own `Secret` type

---

## 3. Security Requirements (RFC-2119)

### 3.1 Panic Prevention

**SEC-SECRET-001**: All functions MUST NEVER panic under any input.

**SEC-SECRET-002**: All functions MUST use `?` operator or explicit error handling (no `.unwrap()`, `.expect()`, `.panic!()`).

**SEC-SECRET-003**: All array/slice access MUST use `.get()` or bounds-checked iteration (no indexing `[]`).

**SEC-SECRET-004**: All integer arithmetic MUST use saturating or checked operations (no unchecked `+`, `-`, `*`).

**SEC-SECRET-005**: Drop implementations MUST NEVER panic (critical for cleanup).

---

### 3.2 Credential Protection

**SEC-SECRET-010**: Secret types MUST use `secrecy::Secret<T>` wrapper (don't roll your own).

**SEC-SECRET-011**: Secret types MUST NOT implement `Debug`, `Display`, `ToString`, or `Serialize` traits.

**SEC-SECRET-012**: Error messages MUST NOT contain secret values (only metadata).

**SEC-SECRET-013**: Logging MUST NOT log secret values (only file paths, metadata).

**SEC-SECRET-014**: Secret types MUST implement `Drop` with zeroization using `zeroize` crate.

**SEC-SECRET-015**: Zeroization MUST use compiler fences to prevent optimization.

**SEC-SECRET-016**: Environment variable loading MUST emit security warnings.

---

### 3.3 File Security

**SEC-SECRET-020**: File permissions MUST be validated before reading (Unix).

**SEC-SECRET-021**: Files with mode `0o077` bits set MUST be rejected (world/group readable).

**SEC-SECRET-022**: Recommended file permissions: `0600` (owner read/write only).

**SEC-SECRET-023**: All file paths MUST be canonicalized (resolve `..` and symlinks).

**SEC-SECRET-024**: Path validation MUST occur before file operations.

---

### 3.4 Timing Safety

**SEC-SECRET-030**: Secret verification MUST use constant-time comparison.

**SEC-SECRET-031**: Verification MUST use `subtle::ConstantTimeEq` or equivalent.

**SEC-SECRET-032**: Verification MUST examine all bytes regardless of match status.

**SEC-SECRET-033**: Length comparison MAY short-circuit (length is not secret).

---

### 3.5 Key Derivation

**SEC-SECRET-040**: Key derivation MUST use HKDF-SHA256.

**SEC-SECRET-041**: Key derivation MUST include domain separation string.

**SEC-SECRET-042**: Derived keys MUST be 32 bytes (256-bit).

**SEC-SECRET-043**: Key derivation MUST be deterministic (same input → same output).

**SEC-SECRET-044**: Domain separation strings MUST be unique per use case.

---

### 3.6 Systemd Credentials

**SEC-SECRET-050**: Systemd credential names MUST be validated (no path separators).

**SEC-SECRET-051**: `$CREDENTIALS_DIRECTORY` MUST be validated as absolute path.

**SEC-SECRET-052**: Credential paths MUST be canonicalized and validated.

---

## 4. Secure Coding Guidelines

### 4.1 No Panics

**Principle**: Never panic, always return `Result`.

```rust
// ❌ BAD: Panics
pub fn load_from_file(path: &Path) -> SecretKey {
    let contents = fs::read_to_string(path).unwrap();
    // ...
}

// ✅ GOOD: Returns Result
pub fn load_from_file(path: &Path) -> Result<SecretKey, SecretError> {
    let contents = fs::read_to_string(path)?;
    // ...
}
```

---

### 4.2 No Secret Leakage

**Principle**: Secrets never appear in logs, errors, or debug output.

```rust
// ❌ BAD: Leaks secret
tracing::info!("Loaded token: {}", token.expose());

// ✅ GOOD: Logs path only
tracing::info!(path = %path.display(), "Secret loaded");

// ❌ BAD: Secret in error
return Err(SecretError::InvalidFormat(format!("bad: {}", token)));

// ✅ GOOD: Metadata only
return Err(SecretError::InvalidFormat("expected 64 hex chars".to_string()));
```

---

### 4.3 Zeroize on Drop

**Principle**: Always zeroize secrets on drop.

```rust
use zeroize::Zeroize;

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.0.zeroize();  // Compiler fence prevents optimization
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.value.zeroize();
    }
}
```

---

### 4.4 Constant-Time Comparison

**Principle**: Use constant-time comparison for secrets.

```rust
use subtle::ConstantTimeEq;

pub fn verify(&self, input: &str) -> bool {
    // Length check can short-circuit
    if self.value.len() != input.len() {
        return false;
    }
    
    // Constant-time comparison of bytes
    self.value.as_bytes().ct_eq(input.as_bytes()).into()
}
```

---

### 4.5 File Permission Validation

**Principle**: Validate permissions before reading.

```rust
#[cfg(unix)]
fn validate_permissions(path: &Path) -> Result<(), SecretError> {
    use std::os::unix::fs::PermissionsExt;
    
    let metadata = std::fs::metadata(path)?;
    let mode = metadata.permissions().mode();
    
    // Check if world or group readable
    if mode & 0o077 != 0 {
        return Err(SecretError::PermissionsTooOpen {
            path: path.display().to_string(),
            mode,
        });
    }
    
    Ok(())
}
```

---

### 4.6 Path Canonicalization

**Principle**: Canonicalize paths before use.

```rust
pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, SecretError> {
    let path = path.as_ref();
    
    // Canonicalize to resolve .. and symlinks
    let canonical = path.canonicalize()
        .map_err(|_| SecretError::FileNotFound(path.display().to_string()))?;
    
    // Validate permissions
    validate_permissions(&canonical)?;
    
    // Read file
    let contents = std::fs::read_to_string(&canonical)?;
    // ...
}
```

---

## 5. Testing Requirements

### 5.1 Attack Scenario Tests

**Required tests for each attack vector**:

```rust
#[test]
fn test_reject_world_readable_file() {
    let file = create_temp_file_with_mode(0o644);
    let result = Secret::load_from_file(&file);
    assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
}

#[test]
fn test_reject_group_readable_file() {
    let file = create_temp_file_with_mode(0o640);
    let result = Secret::load_from_file(&file);
    assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
}

#[test]
fn test_reject_path_traversal() {
    let result = Secret::load_from_file("../../../etc/passwd");
    assert!(result.is_err());
}

#[test]
fn test_timing_safe_verification() {
    let secret = Secret { value: "correct-token".to_string() };
    
    // Measure timing for early mismatch
    let start = Instant::now();
    let _ = secret.verify("wrong-token-aaaa");
    let early_duration = start.elapsed();
    
    // Measure timing for late mismatch
    let start = Instant::now();
    let _ = secret.verify("correct-tokez");
    let late_duration = start.elapsed();
    
    // Timing variance should be < 10%
    let variance = (early_duration.as_nanos() as f64 - late_duration.as_nanos() as f64).abs()
        / early_duration.as_nanos() as f64;
    assert!(variance < 0.1, "Timing variance too high: {}", variance);
}

#[test]
fn test_no_debug_trait() {
    // Should not compile
    // let secret = Secret { value: "test".to_string() };
    // println!("{:?}", secret);
}

#[test]
fn test_zeroize_on_drop() {
    let key_ptr = {
        let key = SecretKey([42u8; 32]);
        key.0.as_ptr()
    };
    // After drop, memory should be zeroed
    // (Hard to test reliably, but zeroize crate provides this)
}

#[test]
fn test_error_no_secret_leakage() {
    let result = Secret::load_from_file("/nonexistent");
    let error_msg = format!("{}", result.unwrap_err());
    
    // Error should not contain secret values
    assert!(!error_msg.contains("secret"));
    assert!(!error_msg.contains("token"));
}

#[test]
fn test_hkdf_deterministic() {
    let key1 = SecretKey::derive_from_token("test-token", b"domain-v1").unwrap();
    let key2 = SecretKey::derive_from_token("test-token", b"domain-v1").unwrap();
    
    assert_eq!(key1.as_bytes(), key2.as_bytes());
}

#[test]
fn test_hkdf_domain_separation() {
    let key1 = SecretKey::derive_from_token("test-token", b"domain-v1").unwrap();
    let key2 = SecretKey::derive_from_token("test-token", b"domain-v2").unwrap();
    
    assert_ne!(key1.as_bytes(), key2.as_bytes());
}
```

---

### 5.2 Panic Prevention Tests

**Required property tests**:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn secret_load_never_panics(path: String) {
        let _ = Secret::load_from_file(&path);
        // Should never panic
    }
    
    #[test]
    fn secret_verify_never_panics(s1: String, s2: String) {
        let secret = Secret { value: s1 };
        let _ = secret.verify(&s2);
        // Should never panic
    }
    
    #[test]
    fn key_derive_never_panics(token: String, domain: Vec<u8>) {
        let _ = SecretKey::derive_from_token(&token, &domain);
        // Should never panic
    }
}
```

---

### 5.3 Fuzzing

**Required fuzz targets**:

```rust
// fuzz/fuzz_targets/load_from_file.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = Secret::load_from_file(s);
        // Should never panic
    }
});

// fuzz/fuzz_targets/verify.rs
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let secret = Secret { value: "test".to_string() };
        let _ = secret.verify(s);
        // Should never panic
    }
});

// fuzz/fuzz_targets/derive_from_token.rs
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = SecretKey::derive_from_token(s, b"domain");
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

### 6.1 TIER 1 Configuration

```rust
// Critical security crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]
```

**Rationale**: Secrets-management is TIER 1 critical; any panic or leak is a security incident.

---

## 7. Incident Response

### 7.1 Secret Leakage Detected

**Detection**:
```bash
# Check logs for secret values
grep -r "secret-" /var/log/llorch/
grep -r "LLORCH_API_TOKEN" /var/log/llorch/

# Check process environment
ps auxe | grep llorch
```

**Response**:
1. Rotate all affected secrets immediately
2. Audit all logs for secret exposure
3. Identify leak source (code, config, deployment)
4. Add regression test
5. Update security training

---

### 7.2 Panic in Secrets-Management

**Detection**:
```
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value'
```

**Response**:
1. Identify panic location
2. Add test case reproducing panic
3. Fix with proper error handling
4. Run full fuzz suite
5. Deploy hotfix immediately (CRITICAL)

---

### 7.3 Timing Attack Detected

**Detection**:
```
Statistical analysis shows timing correlation with input
```

**Response**:
1. Identify non-constant-time comparison
2. Replace with `subtle::ConstantTimeEq`
3. Add timing variance test
4. Rotate affected secrets
5. Deploy hotfix immediately

---

### 7.4 File Permission Bypass

**Detection**:
```
Secret loaded from world-readable file
```

**Response**:
1. Identify validation bypass
2. Add test case reproducing bypass
3. Fix validation logic
4. Audit all secret files
5. Deploy hotfix immediately

---

## 8. Security Review Checklist

### 8.1 Code Review

**Before merging**:
- [ ] All TIER 1 Clippy lints pass
- [ ] No `.unwrap()`, `.expect()`, or `.panic!()` in code
- [ ] No `Debug`, `Display`, `ToString`, or `Serialize` on secret types
- [ ] All array access is bounds-checked
- [ ] All arithmetic uses saturating/checked operations
- [ ] All secrets zeroized on drop
- [ ] All verification uses constant-time comparison
- [ ] All file permissions validated (Unix)
- [ ] All paths canonicalized
- [ ] No secret values in error messages
- [ ] No secret values in logs
- [ ] All attack scenarios have tests
- [ ] Property tests pass
- [ ] Fuzz tests run for 1+ hour without crashes

---

### 8.2 Security Testing

**Before release**:
- [ ] All credential exposure tests pass
- [ ] All file permission tests pass
- [ ] All path traversal tests pass
- [ ] All timing attack tests pass
- [ ] All panic prevention tests pass
- [ ] Fuzzing completed (1+ hour per target)
- [ ] No security warnings from `cargo audit`
- [ ] Manual code review by security team

---

### 8.3 Documentation

**Before deployment**:
- [ ] Security spec reviewed and approved
- [ ] Attack surfaces documented
- [ ] Incident response procedures defined
- [ ] Integration examples include security notes
- [ ] Migration guide from environment variables
- [ ] File permission setup documented

---

## 9. Known Limitations

### 9.1 Platform-Specific Permission Validation

**Limitation**: File permission validation only works on Unix.

**Example**:
```rust
#[cfg(unix)]
fn validate_permissions(path: &Path) -> Result<()> {
    // Validates permissions
}

#[cfg(not(unix))]
fn validate_permissions(path: &Path) -> Result<()> {
    tracing::warn!("Permission validation not available on this platform");
    Ok(())
}
```

**Mitigation**: Document limitation, recommend Unix for production.

---

### 9.2 No Automatic Rotation

**Limitation**: Secrets must be rotated manually (requires restart).

**Example**:
```rust
// Secrets loaded once at startup
let token = Secret::load_from_file("/etc/llorch/api-token")?;

// <Token file updated>

// Old token still in memory (no automatic reload)
```

**Mitigation**: Document manual rotation procedure, add graceful rotation post-M0.

---

### 9.3 No Hardware Security Module Support

**Limitation**: No TPM/HSM support in M0.

**Mitigation**: File-based secrets sufficient for M0, add HSM support post-M0 if needed.

---

### 9.4 Zeroization Best-Effort

**Limitation**: Zeroization cannot guarantee secrets never written to swap.

**Example**:
```rust
// Zeroize on drop
impl Drop for Secret {
    fn drop(&mut self) {
        self.value.zeroize();  // Clears memory
    }
}

// But: OS may have already swapped memory to disk
```

**Mitigation**: Disable swap on production systems, use encrypted swap, or use `mlock()` (post-M0).

---

## 10. References

**Specifications**:
- `00_secrets_management.md` — Main specification
- `10_expectations.md` — Consumer expectations
- `11_worker_vram_residency.md` — vram-residency seal key requirements

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #3 (token in environment)
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issues #1-#20
- `.docs/security/SECURITY_OVERSEER_SUMMARY.md` — Overall security posture

**Standards**:
- NIST SP 800-108 — Key derivation using HKDF
- FIPS 140-2 — Cryptographic module security
- RFC 5869 — HMAC-based Extract-and-Expand Key Derivation Function (HKDF)
- CWE-208 — Observable Timing Discrepancy
- CWE-312 — Cleartext Storage of Sensitive Information
- CWE-522 — Insufficiently Protected Credentials
- CWE-798 — Use of Hard-coded Credentials
- OWASP — Cryptographic Storage Cheat Sheet

---

**End of Security Specification**
