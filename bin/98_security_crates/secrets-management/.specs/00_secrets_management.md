# Secrets Management — Specification

**Status**: Draft  
**Version**: 0.1.0  
**Last Updated**: 2025-10-01  
**Owners**: @llama-orch-maintainers

---

## 0. Document Overview

This specification defines the **secrets-management** shared crate, which provides secure credential loading, storage, and management for the llama-orch orchestration system. This crate is **security-critical** (Tier 1) and implements defense-in-depth against credential exposure, timing attacks, and memory dumps.

**Primary Consumers**:
- `bin/queen-rbee` — API token loading
- `bin/pool-managerd` — API token loading
- `bin/worker-orcd-crates/vram-residency` — Seal key management
- All services requiring secure credential handling

**Related Documents**:
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Security requirements (Issues #1-#20)
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #3 (token in environment)
- `.specs/12_auth-min-hardening.md` — Authentication integration
- `bin/worker-orcd-crates/vram-residency/.specs/11_worker_vram_residency.md` — Consumer requirements

---

## 1. Purpose & Scope

### 1.1 Purpose (SM-PURPOSE-1001)

The secrets-management crate **MUST** provide:

1. **Secure loading** of secrets from files, systemd credentials, and secret managers
2. **Memory protection** with automatic zeroization on drop
3. **Logging safety** preventing secrets from appearing in logs, metrics, or traces
4. **Timing-safe operations** for credential verification
5. **File permission validation** to prevent world-readable secrets
6. **Key derivation** for generating cryptographic keys from base secrets

### 1.2 Scope (SM-SCOPE-1002)

**In Scope**:
- Loading secrets from files (primary method)
- Loading from systemd `LoadCredential` (production deployment)
- Loading from environment variables (with security warnings)
- HKDF-based key derivation for cryptographic keys
- File permission validation (Unix)
- Secure memory cleanup (zeroize on drop)
- Timing-safe credential verification

**Out of Scope** (Future):
- HashiCorp Vault integration (post-M0)
- AWS Secrets Manager integration (post-M0)
- Azure Key Vault integration (post-M0)
- Automatic credential rotation (post-M0)
- Certificate/key pair management (separate crate)
- Multi-tenancy or per-user secrets

---

## 2. Threat Model

### 2.1 Threats Mitigated (SM-THREAT-2001)

The secrets-management crate **MUST** defend against:

1. **Credential Exposure in Process Listings** (SEC-VULN-3)
   - Environment variables visible via `ps auxe` and `/proc/PID/environ`
   - Mitigation: Load from files, never from environment by default

2. **Credential Exposure in Logs** (SEC-VULN-LOG)
   - Secrets accidentally logged via Debug/Display traits
   - Mitigation: No Debug/Display implementation on secret types

3. **Credential Exposure in Memory Dumps** (CI-006)
   - Secrets remain in memory after use
   - Mitigation: Zeroize on drop using compiler fences

4. **Timing Attacks on Verification** (SEC-AUTH-2001)
   - Token comparison timing leaks reveal token prefixes
   - Mitigation: Constant-time comparison for all verification

5. **File Permission Vulnerabilities** (SEC-FILE-PERM)
   - World-readable or group-readable secret files
   - Mitigation: Validate permissions before loading (Unix)

6. **Directory Traversal** (SEC-VULN-9, SEC-VULN-18)
   - Attacker-controlled paths access arbitrary files
   - Mitigation: Validate and canonicalize all file paths

### 2.2 Out of Scope Threats (SM-THREAT-2002)

**Not Mitigated** (require external controls):
- Physical access to disk (use disk encryption)
- Root/admin compromise (use least privilege)
- Side-channel attacks beyond timing (use secure hardware)
- Network interception (use TLS/mTLS)

---

## 3. Core Types & API

### 3.1 SecretKey Type (SM-TYPE-3001)

**Purpose**: Opaque wrapper for cryptographic keys (32 bytes) used in HMAC, encryption, etc.

**Requirements**:

**SM-TYPE-3001-R1**: `SecretKey` **MUST** be a newtype wrapping `[u8; 32]`

**SM-TYPE-3001-R2**: `SecretKey` **MUST NOT** implement `Debug`, `Display`, `Clone`, or `Serialize`

**SM-TYPE-3001-R3**: `SecretKey` **MUST** implement `Drop` with zeroization

**SM-TYPE-3001-R4**: `SecretKey` **MUST** provide `as_bytes(&self) -> &[u8; 32]` for internal use

**SM-TYPE-3001-R5**: `as_bytes()` **MUST** be `pub(crate)` or module-private, never public

**Reference Implementation**:
```rust
use zeroize::Zeroize;

pub struct SecretKey([u8; 32]);

impl SecretKey {
    /// Access key bytes (private - only for HMAC/crypto operations)
    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

// NO Debug, Display, Clone, Serialize
```

**Test Requirements**:
- **SM-TEST-3001-T1**: Verify `SecretKey` does not implement `Debug` (compile-time)
- **SM-TEST-3001-T2**: Verify memory is zeroed after drop (best-effort test)

---

### 3.2 Secret Type (SM-TYPE-3002)

**Purpose**: Opaque wrapper for string-based secrets (API tokens, passwords)

**Requirements**:

**SM-TYPE-3002-R1**: `Secret` **MUST** wrap a `String` with secure cleanup

**SM-TYPE-3002-R2**: `Secret` **MUST NOT** implement `Debug`, `Display`, or `Clone`

**SM-TYPE-3002-R3**: `Secret` **MUST** provide `expose(&self) -> &str` for controlled access

**SM-TYPE-3002-R4**: `Secret` **MUST** provide `verify(&self, input: &str) -> bool` with timing-safe comparison

**SM-TYPE-3002-R5**: `verify()` **MUST** use constant-time comparison (no short-circuit)

**Reference Implementation**:
```rust
use zeroize::Zeroize;

pub struct Secret {
    value: String,
}

impl Secret {
    /// Verify input matches secret (timing-safe)
    pub fn verify(&self, input: &str) -> bool {
        use subtle::ConstantTimeEq;
        self.value.as_bytes().ct_eq(input.as_bytes()).into()
    }
    
    /// Expose secret value (use sparingly)
    pub fn expose(&self) -> &str {
        &self.value
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.value.zeroize();
    }
}
```

**Test Requirements**:
- **SM-TEST-3002-T1**: Verify timing-safe comparison (measure variance)
- **SM-TEST-3002-T2**: Verify `verify()` returns correct results
- **SM-TEST-3002-T3**: Verify memory is zeroed after drop

---

### 3.3 SecretError Type (SM-TYPE-3003)

**Purpose**: Error type for secret loading and validation failures

**Requirements**:

**SM-TYPE-3003-R1**: `SecretError` **MUST** implement `std::error::Error`

**SM-TYPE-3003-R2**: Error messages **MUST NOT** contain secret values

**SM-TYPE-3003-R3**: Error messages **MAY** contain file paths and permission modes

**Reference Implementation**:
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SecretError {
    #[error("secret file not found: {0}")]
    FileNotFound(String),
    
    #[error("file permissions too open: {path} (mode: {mode:o}, expected 0600)")]
    PermissionsTooOpen { path: String, mode: u32 },
    
    #[error("invalid secret format: {0}")]
    InvalidFormat(String),
    
    #[error("systemd credential not found: {0}")]
    SystemdCredentialNotFound(String),
    
    #[error("path validation failed: {0}")]
    PathValidationFailed(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("key derivation error: {0}")]
    KeyDerivation(String),
}
```

---

## 4. Secret Loading API

### 4.1 Load from File (SM-LOAD-4001)

**Purpose**: Load secrets from filesystem (primary method)

**Requirements**:

**SM-LOAD-4001-R1**: `SecretKey::load_from_file(path: impl AsRef<Path>)` **MUST** be provided

**SM-LOAD-4001-R2**: File permissions **MUST** be validated before reading (Unix)

**SM-LOAD-4001-R3**: Files with mode `0o077` bits set **MUST** be rejected (world/group readable)

**SM-LOAD-4001-R4**: File contents **MUST** be trimmed of leading/trailing whitespace

**SM-LOAD-4001-R5**: Empty files (after trim) **MUST** return `SecretError::InvalidFormat`

**SM-LOAD-4001-R6**: File **MUST** contain exactly 32 bytes (after hex/base64 decode) or 64 hex chars

**SM-LOAD-4001-R7**: Path **MUST** be canonicalized to prevent traversal

**Reference Implementation**:
```rust
impl SecretKey {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, SecretError> {
        let path = path.as_ref();
        
        // Canonicalize to prevent traversal
        let canonical = path.canonicalize()
            .map_err(|_| SecretError::FileNotFound(path.display().to_string()))?;
        
        // Validate permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(&canonical)?;
            let mode = metadata.permissions().mode();
            
            if mode & 0o077 != 0 {
                return Err(SecretError::PermissionsTooOpen {
                    path: canonical.display().to_string(),
                    mode,
                });
            }
        }
        
        // Read and validate
        let contents = std::fs::read_to_string(&canonical)?;
        let trimmed = contents.trim();
        
        if trimmed.is_empty() {
            return Err(SecretError::InvalidFormat("empty file".to_string()));
        }
        
        // Decode hex or base64
        let bytes = if trimmed.len() == 64 {
            hex::decode(trimmed)
                .map_err(|_| SecretError::InvalidFormat("invalid hex".to_string()))?
        } else {
            return Err(SecretError::InvalidFormat("expected 64 hex chars".to_string()));
        };
        
        if bytes.len() != 32 {
            return Err(SecretError::InvalidFormat("expected 32 bytes".to_string()));
        }
        
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        
        tracing::info!(path = %canonical.display(), "Secret key loaded from file");
        Ok(SecretKey(key))
    }
}
```

**Test Requirements**:
- **SM-TEST-4001-T1**: Load valid key file (0600 permissions)
- **SM-TEST-4001-T2**: Reject world-readable file (0644)
- **SM-TEST-4001-T3**: Reject group-readable file (0640)
- **SM-TEST-4001-T4**: Reject empty file
- **SM-TEST-4001-T5**: Reject file with wrong length
- **SM-TEST-4001-T6**: Reject file with invalid hex
- **SM-TEST-4001-T7**: Reject path traversal attempts (`../../etc/passwd`)

---

### 4.2 Load from Systemd Credential (SM-LOAD-4002)

**Purpose**: Load secrets from systemd `LoadCredential` (production deployment)

**Requirements**:

**SM-LOAD-4002-R1**: `SecretKey::from_systemd_credential(name: &str)` **MUST** be provided

**SM-LOAD-4002-R2**: Credential path **MUST** be `/run/credentials/<service>/<name>`

**SM-LOAD-4002-R3**: Service name **MUST** be detected from `$CREDENTIALS_DIRECTORY` environment variable

**SM-LOAD-4002-R4**: If `$CREDENTIALS_DIRECTORY` is not set, **MUST** return `SecretError::SystemdCredentialNotFound`

**SM-LOAD-4002-R5**: File validation **MUST** follow same rules as `load_from_file()`

**Reference Implementation**:
```rust
impl SecretKey {
    pub fn from_systemd_credential(name: &str) -> Result<Self, SecretError> {
        let creds_dir = std::env::var("CREDENTIALS_DIRECTORY")
            .map_err(|_| SecretError::SystemdCredentialNotFound(
                "CREDENTIALS_DIRECTORY not set".to_string()
            ))?;
        
        let path = PathBuf::from(creds_dir).join(name);
        
        if !path.exists() {
            return Err(SecretError::SystemdCredentialNotFound(
                format!("credential not found: {}", name)
            ));
        }
        
        Self::load_from_file(path)
    }
}
```

**Systemd Service Example**:
```ini
[Service]
LoadCredential=seal_key:/etc/llorch/secrets/worker-seal-key
# Key available at /run/credentials/worker-orcd.service/seal_key
```

**Test Requirements**:
- **SM-TEST-4002-T1**: Load credential when `$CREDENTIALS_DIRECTORY` is set
- **SM-TEST-4002-T2**: Fail when `$CREDENTIALS_DIRECTORY` is not set
- **SM-TEST-4002-T3**: Fail when credential file doesn't exist

---

### 4.3 Derive from Token (SM-LOAD-4003)

**Purpose**: Derive cryptographic keys from API tokens using HKDF

**Requirements**:

**SM-LOAD-4003-R1**: `SecretKey::derive_from_token(token: &str, domain: &[u8])` **MUST** be provided

**SM-LOAD-4003-R2**: Derivation **MUST** use HKDF-SHA256

**SM-LOAD-4003-R3**: Domain separation string **MUST** be included as HKDF info parameter

**SM-LOAD-4003-R4**: Output **MUST** be exactly 32 bytes

**SM-LOAD-4003-R5**: Derivation **MUST** be deterministic (same token + domain → same key)

**SM-LOAD-4003-R6**: Token **MUST NOT** be logged or exposed

**Reference Implementation**:
```rust
use hkdf::Hkdf;
use sha2::Sha256;

impl SecretKey {
    pub fn derive_from_token(token: &str, domain: &[u8]) -> Result<Self, SecretError> {
        if token.is_empty() {
            return Err(SecretError::InvalidFormat("empty token".to_string()));
        }
        
        // HKDF-SHA256 with domain separation
        let hkdf = Hkdf::<Sha256>::new(None, token.as_bytes());
        let mut key = [0u8; 32];
        hkdf.expand(domain, &mut key)
            .map_err(|e| SecretError::KeyDerivation(e.to_string()))?;
        
        tracing::info!(domain = %String::from_utf8_lossy(domain), "Secret key derived from token");
        Ok(SecretKey(key))
    }
}
```

**Domain Separation Examples**:
```rust
// vram-residency seal key
let seal_key = SecretKey::derive_from_token(&worker_token, b"llorch-seal-key-v1")?;

// Future: encryption key
let enc_key = SecretKey::derive_from_token(&worker_token, b"llorch-encryption-v1")?;
```

**Test Requirements**:
- **SM-TEST-4003-T1**: Verify deterministic derivation (same inputs → same output)
- **SM-TEST-4003-T2**: Verify different domains produce different keys
- **SM-TEST-4003-T3**: Verify output is 32 bytes
- **SM-TEST-4003-T4**: Reject empty token

---

### 4.4 Load Secret (String) from File (SM-LOAD-4004)

**Purpose**: Load string-based secrets (API tokens) from files

**Requirements**:

**SM-LOAD-4004-R1**: `Secret::load_from_file(path: impl AsRef<Path>)` **MUST** be provided

**SM-LOAD-4004-R2**: File validation **MUST** follow same rules as `SecretKey::load_from_file()`

**SM-LOAD-4004-R3**: Contents **MUST** be trimmed but otherwise unmodified

**SM-LOAD-4004-R4**: Empty files (after trim) **MUST** return error

**Reference Implementation**:
```rust
impl Secret {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, SecretError> {
        let path = path.as_ref();
        
        // Canonicalize and validate permissions (same as SecretKey)
        let canonical = path.canonicalize()
            .map_err(|_| SecretError::FileNotFound(path.display().to_string()))?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(&canonical)?;
            let mode = metadata.permissions().mode();
            
            if mode & 0o077 != 0 {
                return Err(SecretError::PermissionsTooOpen {
                    path: canonical.display().to_string(),
                    mode,
                });
            }
        }
        
        let contents = std::fs::read_to_string(&canonical)?;
        let trimmed = contents.trim();
        
        if trimmed.is_empty() {
            return Err(SecretError::InvalidFormat("empty file".to_string()));
        }
        
        tracing::info!(path = %canonical.display(), "Secret loaded from file");
        Ok(Secret {
            value: trimmed.to_string(),
        })
    }
}
```

**Test Requirements**:
- **SM-TEST-4004-T1**: Load valid token file
- **SM-TEST-4004-T2**: Reject world-readable file
- **SM-TEST-4004-T3**: Reject empty file
- **SM-TEST-4004-T4**: Trim whitespace correctly

---

### 4.5 Load from Environment (Deprecated) (SM-LOAD-4005)

**Purpose**: Load secrets from environment variables (NOT RECOMMENDED)

**Requirements**:

**SM-LOAD-4005-R1**: `Secret::from_env(var_name: &str)` **MAY** be provided for backward compatibility

**SM-LOAD-4005-R2**: Function **MUST** emit a `tracing::warn!` about security risk

**SM-LOAD-4005-R3**: Warning **MUST** mention visibility in process listings

**SM-LOAD-4005-R4**: Function **SHOULD** be marked `#[deprecated]` in future versions

**Reference Implementation**:
```rust
impl Secret {
    /// Load from environment variable (NOT RECOMMENDED - visible in process listing)
    pub fn from_env(var_name: &str) -> Result<Self, SecretError> {
        tracing::warn!(
            env_var = %var_name,
            "Loading secret from environment variable (NOT RECOMMENDED - visible in process listing and /proc)"
        );
        
        let value = std::env::var(var_name)
            .map_err(|_| SecretError::FileNotFound(format!("env var not set: {}", var_name)))?;
        
        if value.is_empty() {
            return Err(SecretError::InvalidFormat("empty value".to_string()));
        }
        
        Ok(Secret { value })
    }
}
```

**Migration Path**:
```rust
// OLD (vulnerable)
let token = std::env::var("LLORCH_API_TOKEN")?;

// NEW (secure)
let token = Secret::load_from_file("/etc/llorch/api-token")?;
```

---

## 5. Security Properties

### 5.1 Memory Safety (SM-SEC-5001)

**Requirements**:

**SM-SEC-5001-R1**: All secret types **MUST** implement `Drop` with zeroization

**SM-SEC-5001-R2**: Zeroization **MUST** use `zeroize` crate with compiler fences

**SM-SEC-5001-R3**: `Drop` implementation **MUST NOT** panic

**SM-SEC-5001-R4**: Zeroization **MUST** overwrite all bytes of secret data

**Reference**:
```rust
use zeroize::Zeroize;

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.0.zeroize();
        // Compiler fence prevents optimization
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.value.zeroize();
    }
}
```

**Rationale**: Prevents secrets from remaining in memory dumps, swap files, or core dumps.

---

### 5.2 Logging Safety (SM-SEC-5002)

**Requirements**:

**SM-SEC-5002-R1**: Secret types **MUST NOT** implement `Debug` trait

**SM-SEC-5002-R2**: Secret types **MUST NOT** implement `Display` trait

**SM-SEC-5002-R3**: Secret types **MUST NOT** implement `ToString` trait

**SM-SEC-5002-R4**: Secret types **MUST NOT** implement `Serialize` trait

**SM-SEC-5002-R5**: Error messages **MUST NOT** contain secret values

**SM-SEC-5002-R6**: Tracing logs **MUST NOT** log secret values (only paths, metadata)

**Forbidden Patterns**:
```rust
// ❌ FORBIDDEN
tracing::info!("Loaded key: {:?}", secret_key);
println!("Token: {}", secret);
format!("Secret: {:?}", secret);

// ✅ ALLOWED
tracing::info!(path = %path.display(), "Secret loaded");
```

---

### 5.3 Timing Safety (SM-SEC-5003)

**Requirements**:

**SM-SEC-5003-R1**: `Secret::verify()` **MUST** use constant-time comparison

**SM-SEC-5003-R2**: Comparison **MUST NOT** short-circuit on first mismatch

**SM-SEC-5003-R3**: Comparison **MUST** examine all bytes regardless of match status

**SM-SEC-5003-R4**: Length comparison **MAY** short-circuit (length is not secret)

**Reference Implementation**:
```rust
use subtle::ConstantTimeEq;

pub fn verify(&self, input: &str) -> bool {
    // Length check can short-circuit (length is public)
    if self.value.len() != input.len() {
        return false;
    }
    
    // Constant-time comparison of bytes
    self.value.as_bytes().ct_eq(input.as_bytes()).into()
}
```

**Test Requirements**:
- **SM-TEST-5003-T1**: Measure timing variance for matching vs non-matching inputs
- **SM-TEST-5003-T2**: Verify no statistical difference in timing

---

### 5.4 File Permission Validation (SM-SEC-5004)

**Requirements**:

**SM-LOAD-5004-R1**: File permissions **MUST** be validated on Unix systems

**SM-LOAD-5004-R2**: Files with world-readable bit (0o004) **MUST** be rejected

**SM-LOAD-5004-R3**: Files with group-readable bit (0o040) **MUST** be rejected

**SM-LOAD-5004-R4**: Recommended permissions: `0600` (owner read/write only)

**SM-LOAD-5004-R5**: Validation **MUST** occur before reading file contents

**SM-LOAD-5004-R6**: On non-Unix systems, validation **MAY** be skipped with warning

**Reference Implementation**:
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

#[cfg(not(unix))]
fn validate_permissions(path: &Path) -> Result<(), SecretError> {
    tracing::warn!(
        path = %path.display(),
        "File permission validation not available on this platform"
    );
    Ok(())
}
```

**Setup Instructions**:
```bash
# Create secret file with correct permissions
sudo mkdir -p /etc/llorch/secrets
sudo chmod 0700 /etc/llorch/secrets
sudo touch /etc/llorch/secrets/api-token
sudo chmod 0600 /etc/llorch/secrets/api-token
sudo chown queen-rbee:queen-rbee /etc/llorch/secrets/api-token
```

---

### 5.5 Path Validation (SM-SEC-5005)

**Requirements**:

**SM-SEC-5005-R1**: All file paths **MUST** be canonicalized before use

**SM-SEC-5005-R2**: Canonicalization **MUST** resolve symlinks and `..` sequences

**SM-SEC-5005-R3**: Paths **SHOULD** be validated against allowed directories (optional)

**SM-SEC-5005-R4**: Path traversal attempts **MUST** be rejected

**Reference Implementation**:
```rust
fn validate_path(path: &Path) -> Result<PathBuf, SecretError> {
    // Canonicalize to resolve .. and symlinks
    let canonical = path.canonicalize()
        .map_err(|_| SecretError::PathValidationFailed(
            format!("cannot canonicalize: {}", path.display())
        ))?;
    
    // Optional: Check against allowed root
    // let allowed_root = PathBuf::from("/etc/llorch/secrets");
    // if !canonical.starts_with(&allowed_root) {
    //     return Err(SecretError::PathValidationFailed(
    //         "path outside allowed directory".to_string()
    //     ));
    // }
    
    Ok(canonical)
}
```

**Test Requirements**:
- **SM-TEST-5005-T1**: Reject `../../etc/passwd`
- **SM-TEST-5005-T2**: Reject symlinks to unauthorized locations (optional)
- **SM-TEST-5005-T3**: Accept valid paths within allowed directory

---

## 6. Integration Requirements

### 6.1 queen-rbee Integration (SM-INTEG-6001)

**Current State**: Uses environment variable (vulnerable)

**Required Changes**:

**SM-INTEG-6001-R1**: Replace `std::env::var("LLORCH_API_TOKEN")` with `Secret::load_from_file()`

**SM-INTEG-6001-R2**: Add configuration for token file path

**SM-INTEG-6001-R3**: Default path: `/etc/llorch/secrets/api-token`

**SM-INTEG-6001-R4**: Support `$LLORCH_API_TOKEN_FILE` environment variable for path override

**Migration Example**:
```rust
// OLD (vulnerable)
let token = std::env::var("LLORCH_API_TOKEN").ok();

// NEW (secure)
use secrets_management::Secret;

let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
    .unwrap_or_else(|_| "/etc/llorch/secrets/api-token".to_string());

let token = Secret::load_from_file(&token_path)
    .map_err(|e| format!("Failed to load API token: {}", e))?;
```

---

### 6.2 pool-managerd Integration (SM-INTEG-6002)

**Current State**: No authentication (critical vulnerability)

**Required Changes**:

**SM-INTEG-6002-R1**: Add `secrets-management` dependency

**SM-INTEG-6002-R2**: Load API token from file (same pattern as queen-rbee)

**SM-INTEG-6002-R3**: Implement Bearer token validation middleware

**SM-INTEG-6002-R4**: Apply middleware to all endpoints except `/health`

---

### 6.3 vram-residency Integration (SM-INTEG-6003)

**Consumer Requirements**: See `bin/worker-orcd-crates/vram-residency/.specs/11_worker_vram_residency.md`

**Required API**:

**SM-INTEG-6003-R1**: `SecretKey::load_from_file()` for seal key loading

**SM-INTEG-6003-R2**: `SecretKey::derive_from_token()` for key derivation from worker token

**SM-INTEG-6003-R3**: `SecretKey::from_systemd_credential()` for production deployment

**SM-INTEG-6003-R4**: `SecretKey::as_bytes()` for HMAC-SHA256 computation

**Usage Example**:
```rust
use secrets_management::SecretKey;
use hmac::{Hmac, Mac};
use sha2::Sha256;

// Load seal key
let seal_key = if let Some(key_file) = &config.seal_key_file {
    SecretKey::load_from_file(key_file)?
} else if let Some(token) = &config.worker_api_token {
    SecretKey::derive_from_token(token, b"llorch-seal-key-v1")?
} else {
    SecretKey::from_systemd_credential("seal_key")?
};

// Compute HMAC seal signature
let message = format!("{}|{}|{}|{}", 
    shard_id, digest, sealed_at_unix, gpu_device
);

let mut mac = Hmac::<Sha256>::new_from_slice(seal_key.as_bytes())
    .expect("HMAC can take key of any size");
mac.update(message.as_bytes());
let signature = mac.finalize().into_bytes();
```

---

## 7. Dependencies

### 7.1 Required Dependencies (SM-DEPS-7001)

**SM-DEPS-7001-R1**: `zeroize` — Secure memory cleanup

**SM-DEPS-7001-R2**: `thiserror` — Error types

**SM-DEPS-7001-R3**: `hkdf` — Key derivation (HKDF-SHA256)

**SM-DEPS-7001-R4**: `sha2` — SHA-256 for HKDF

**SM-DEPS-7001-R5**: `subtle` — Constant-time comparison

**SM-DEPS-7001-R6**: `hex` — Hex decoding for key files

**Cargo.toml**:
```toml
[dependencies]
zeroize = { version = "1.7", features = ["derive"] }
thiserror = "1.0"
hkdf = "0.12"
sha2 = "0.10"
subtle = "2.5"
hex = "0.4"
tracing = "0.1"
```

### 7.2 Optional Dependencies (SM-DEPS-7002)

**Future Integrations** (post-M0):
- `vaultrs` — HashiCorp Vault client
- `aws-sdk-secretsmanager` — AWS Secrets Manager
- `azure_security_keyvault` — Azure Key Vault

---

## 8. Testing Requirements

### 8.1 Unit Tests (SM-TEST-8001)

**Required Test Coverage**:

1. **SecretKey Loading**
   - Load valid key file (0600 permissions)
   - Reject world-readable file (0644)
   - Reject group-readable file (0640)
   - Reject empty file
   - Reject invalid hex
   - Reject wrong length

2. **SecretKey Derivation**
   - Deterministic derivation
   - Different domains produce different keys
   - Output is 32 bytes
   - Reject empty token

3. **Secret Loading**
   - Load valid token file
   - Reject world-readable file
   - Trim whitespace correctly
   - Reject empty file

4. **Secret Verification**
   - Correct verification (match/no-match)
   - Timing-safe comparison (measure variance)
   - Length mismatch short-circuits

5. **Systemd Credentials**
   - Load when `$CREDENTIALS_DIRECTORY` is set
   - Fail when not set
   - Fail when credential doesn't exist

6. **Path Validation**
   - Reject `../../etc/passwd`
   - Accept valid paths
   - Canonicalize correctly

7. **Memory Safety**
   - Verify zeroization on drop (best-effort)
   - No Debug/Display implementation (compile-time)

---

### 8.2 Integration Tests (SM-TEST-8002)

**Required Integration Tests**:

1. **queen-rbee Integration**
   - Load token from file
   - Verify authentication works
   - Reject invalid token

2. **pool-managerd Integration**
   - Load token from file
   - Verify authentication works
   - Reject invalid token

3. **vram-residency Integration**
   - Load seal key from file
   - Derive seal key from token
   - Compute HMAC signature
   - Verify signature

---

### 8.3 Security Tests (SM-TEST-8003)

**Required Security Tests**:

1. **Timing Attack Resistance**
   - Measure comparison timing variance
   - Verify no statistical difference

2. **Permission Validation**
   - Reject world-readable files
   - Reject group-readable files
   - Accept owner-only files

3. **Path Traversal**
   - Reject `../` sequences
   - Reject symlinks to unauthorized locations

4. **Memory Safety**
   - Verify no secrets in logs
   - Verify no Debug output
   - Verify zeroization (best-effort)

---

## 9. Configuration

### 9.1 File Locations (SM-CONFIG-9001)

**Recommended Paths**:

| Environment | Path | Permissions |
|-------------|------|-------------|
| Production | `/etc/llorch/secrets/api-token` | 0600 |
| Production | `/etc/llorch/secrets/worker-seal-key` | 0600 |
| Systemd | `/run/credentials/<service>/<name>` | 0600 (automatic) |
| Development | `~/.config/llorch/api-token` | 0600 |
| Testing | `/tmp/llorch-test-<uuid>/secrets/` | 0600 |

### 9.2 Environment Variables (SM-CONFIG-9002)

**Supported Variables**:

| Variable | Purpose | Recommended |
|----------|---------|-------------|
| `LLORCH_API_TOKEN_FILE` | Path to API token file | ✅ YES |
| `WORKER_SEAL_KEY_FILE` | Path to seal key file | ✅ YES |
| `CREDENTIALS_DIRECTORY` | Systemd credentials directory | ✅ YES (systemd) |
| `LLORCH_API_TOKEN` | API token value | ❌ NO (vulnerable) |
| `WORKER_SEAL_KEY` | Seal key value | ❌ NO (vulnerable) |

---

## 10. Implementation Phases

### 10.1 Phase 1: M0 Essentials (SM-PHASE-10001)

**Status**: ✅ Partially Complete

**Required for M0**:
1. ✅ `SecretKey` type with zeroize on drop
2. ✅ `Secret` type with zeroize on drop
3. ✅ `SecretError` type
4. ⬜ `SecretKey::load_from_file()` with permission validation
5. ⬜ `SecretKey::derive_from_token()` with HKDF
6. ⬜ `Secret::load_from_file()` with permission validation
7. ⬜ `Secret::verify()` with timing-safe comparison
8. ⬜ No Debug/Display/Serialize traits
9. ⬜ Unit tests for all core functionality

### 10.2 Phase 2: Production Hardening (SM-PHASE-10002)

**Status**: ⬜ Not Started

**Required for Production**:
1. ⬜ `SecretKey::from_systemd_credential()` support
2. ⬜ File permission validation on all platforms
3. ⬜ Path canonicalization and validation
4. ⬜ Comprehensive security tests
5. ⬜ Integration with queen-rbee (replace env var)
6. ⬜ Integration with pool-managerd (add auth)
7. ⬜ Integration with vram-residency (seal keys)

### 10.3 Phase 3: Advanced Features (SM-PHASE-10003)

**Status**: ⬜ Future

**Post-M0 Enhancements**:
1. ⬜ HashiCorp Vault integration
2. ⬜ AWS Secrets Manager integration
3. ⬜ Azure Key Vault integration
4. ⬜ Automatic credential rotation
5. ⬜ Key versioning and rollover
6. ⬜ Hardware security module (HSM) support

---

## 11. Acceptance Criteria

### 11.1 Functional Acceptance (SM-ACCEPT-11001)

**Criteria**:
- ✅ All Phase 1 features implemented
- ✅ All unit tests passing
- ✅ All integration tests passing
- ✅ queen-rbee loads token from file (not env)
- ✅ pool-managerd loads token from file
- ✅ vram-residency loads seal key from file or derives from token

### 11.2 Security Acceptance (SM-ACCEPT-11002)

**Criteria**:
- ✅ No secrets in logs (verified by audit)
- ✅ No Debug/Display on secret types (compile-time check)
- ✅ Timing-safe comparison (measured variance < 5%)
- ✅ File permissions validated (Unix)
- ✅ Path traversal rejected (tested)
- ✅ Memory zeroized on drop (best-effort test)

### 11.3 Documentation Acceptance (SM-ACCEPT-11003)

**Criteria**:
- ✅ This specification complete
- ✅ API documentation (rustdoc)
- ✅ Integration examples for all consumers
- ✅ Migration guide from environment variables
- ✅ Security best practices documented

---

## 12. Open Questions

**Q1**: Should we support hardware-based keys (TPM, HSM)?  
**A**: Defer to post-M0. File-based keys sufficient for M0.

**Q2**: Should keys be rotatable without restart?  
**A**: Defer to post-M0. Manual restart acceptable for M0.

**Q3**: Should we support multiple key versions (for rotation)?  
**A**: Defer to post-M0. Single key sufficient for M0.

**Q4**: Should we validate paths against allowed directories?  
**A**: Optional for M0. Canonicalization is sufficient.

**Q5**: Should we support base64 encoding in addition to hex?  
**A**: Defer to post-M0. Hex is sufficient for M0.

---

## 13. References

**Security Audits**:
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issues #1-#20
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #3
- `.docs/security/SECURITY_OVERSEER_SUMMARY.md` — Overall security posture

**Specifications**:
- `.specs/12_auth-min-hardening.md` — Authentication requirements
- `bin/worker-orcd-crates/vram-residency/.specs/11_worker_vram_residency.md` — Consumer requirements
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security requirements (CI-005, CI-006)

**Standards**:
- NIST SP 800-108 — Key derivation using HKDF
- FIPS 140-2 — Cryptographic module security
- RFC 5869 — HMAC-based Extract-and-Expand Key Derivation Function (HKDF)

---

**End of Specification**
