# Secrets Management — Worker VRAM Residency Requirements

**Consumer**: `bin/worker-orcd-crates/vram-residency`  
**Status**: Draft  
**Last Updated**: 2025-10-01

---

## 0. Overview

This document specifies what `vram-residency` expects from the `secrets-management` crate for secure seal key handling.

**Context**: `vram-residency` implements cryptographic seal signatures (HMAC-SHA256) to prove sealed shard integrity. The seal secret key is security-critical and must be handled with extreme care.

**Reference**: 
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security spec
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations

---

## 1. Use Case: Seal Secret Key Management

### 1.1 Purpose

**Seal keys** are used to generate HMAC-SHA256 signatures that cryptographically attest sealed shard integrity:

```rust
// Seal signature covers (shard_id, digest, sealed_at, gpu_device)
let message = format!("{}|{}|{}|{}", 
    shard.shard_id, 
    shard.digest, 
    sealed_at_unix_timestamp,
    shard.gpu_device
);

let signature = hmac_sha256(&seal_key, message.as_bytes());
```

**Security requirements** (from `20_security.md`):
- **CI-005**: Seal secret keys MUST be derived from worker API token or hardware ID
- **CI-006**: Seal secret keys MUST NOT be logged, exposed in API, or written to disk
- Keys MUST be zeroized on drop (prevent memory dumps)
- Keys MUST be loaded from secure sources (not environment variables)

---

## 2. Required API

### 2.1 Key Loading

**Load from file** (primary method):
```rust
use secrets_management::SecretKey;

// Load seal key from file
let seal_key = SecretKey::load_from_file(
    "/etc/llorch/secrets/worker-seal-key"
)?;
```

**Expected behavior**:
- Read file contents
- Validate file permissions (must be 0600 or stricter)
- Trim whitespace
- Return opaque `SecretKey` type
- Fail fast if file missing or unreadable

**Error cases**:
- `SecretError::FileNotFound` — Key file doesn't exist
- `SecretError::PermissionsTooOpen` — File is world-readable
- `SecretError::InvalidFormat` — Key is not valid (wrong length, encoding)

---

### 2.2 Key Derivation

**Derive from worker token** (alternative method):
```rust
// Derive seal key from worker API token
let seal_key = SecretKey::derive_from_token(
    &worker_api_token,
    b"llorch-seal-key-v1"  // Domain separation
)?;
```

**Expected behavior**:
- Use HKDF-SHA256 or similar KDF
- Include domain separation string
- Return 32-byte key suitable for HMAC-SHA256
- Deterministic (same token → same key)

---

### 2.3 Systemd Credential Support

**Load from systemd credentials** (production deployment):
```rust
// Load from systemd LoadCredential
let seal_key = SecretKey::from_systemd_credential("seal_key")?;
```

**Expected behavior**:
- Read from `/run/credentials/<service>/seal_key`
- Validate file exists and is readable
- Return opaque `SecretKey` type

**Systemd service example**:
```ini
[Service]
LoadCredential=seal_key:/etc/llorch/secrets/worker-seal-key
# Key not visible in process listing or /proc
```

---

### 2.4 Key Access

**Get key bytes for HMAC**:
```rust
impl SecretKey {
    // Private method, only accessible within vram-residency
    pub(crate) fn as_bytes(&self) -> &[u8; 32];
}
```

**Security requirements**:
- Key bytes MUST be private (not public)
- No `Debug` or `Display` implementation
- No serialization (no `Serialize` trait)
- No cloning (single owner)

---

### 2.5 Secure Cleanup

**Zeroize on drop**:
```rust
impl Drop for SecretKey {
    fn drop(&mut self) {
        // Zeroize key material
        self.0.zeroize();
    }
}
```

**Expected behavior**:
- Overwrite key bytes with zeros
- Use `zeroize` crate for compiler-fence protection
- Prevent key material in memory dumps
- MUST NOT panic in Drop

---

## 3. Type Requirements

### 3.1 SecretKey Type

```rust
pub struct SecretKey([u8; 32]);

impl SecretKey {
    // Load from file
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, SecretError>;
    
    // Derive from token
    pub fn derive_from_token(token: &str, domain: &[u8]) -> Result<Self, SecretError>;
    
    // Load from systemd
    pub fn from_systemd_credential(name: &str) -> Result<Self, SecretError>;
    
    // Access key bytes (private)
    pub(crate) fn as_bytes(&self) -> &[u8; 32];
}

// No Debug, Display, Clone, Serialize
impl Drop for SecretKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
```

---

### 3.2 Error Type

```rust
#[derive(Debug, Error)]
pub enum SecretError {
    #[error("secret file not found: {0}")]
    FileNotFound(String),
    
    #[error("file permissions too open: {0} (expected 0600)")]
    PermissionsTooOpen(String),
    
    #[error("invalid secret format")]
    InvalidFormat,
    
    #[error("systemd credential not found: {0}")]
    SystemdCredentialNotFound(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

---

## 4. Security Properties

### 4.1 Never Log Keys

**MUST NOT**:
```rust
// ❌ FORBIDDEN
tracing::info!("Seal key: {:?}", seal_key);
eprintln!("Debug: key={:?}", seal_key);
format!("Key: {:?}", seal_key);
```

**Implementation**:
- No `Debug` trait implementation
- No `Display` trait implementation
- No `ToString` implementation

---

### 4.2 Never Expose in API

**MUST NOT**:
```rust
// ❌ FORBIDDEN
pub struct SealedShard {
    pub seal_key: SecretKey,  // Never expose
}

// ❌ FORBIDDEN
#[derive(Serialize)]
pub struct Config {
    seal_key: SecretKey,  // Never serialize
}
```

---

### 4.3 Prevent Memory Dumps

**Zeroize on drop**:
```rust
use zeroize::Zeroize;

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.0.zeroize();
        // Compiler fence prevents optimization
    }
}
```

---

### 4.4 File Permission Validation

**Validate before loading**:
```rust
pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, SecretError> {
    let metadata = fs::metadata(&path)?;
    let permissions = metadata.permissions();
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = permissions.mode();
        
        // Check if world-readable or group-readable
        if mode & 0o077 != 0 {
            return Err(SecretError::PermissionsTooOpen(
                format!("{:o}", mode)
            ));
        }
    }
    
    // Now safe to read
    let contents = fs::read_to_string(&path)?;
    // ...
}
```

---

## 5. Usage in vram-residency

### 5.1 Initialization

```rust
// In VramManager::new()
use secrets_management::SecretKey;

pub struct VramManager {
    seal_key: SecretKey,
    // ...
}

impl VramManager {
    pub fn new(config: &WorkerConfig) -> Result<Self> {
        // Load seal key from config
        let seal_key = if let Some(key_file) = &config.seal_key_file {
            SecretKey::load_from_file(key_file)?
        } else if let Some(token) = &config.worker_api_token {
            SecretKey::derive_from_token(token, b"llorch-seal-key-v1")?
        } else {
            SecretKey::from_systemd_credential("seal_key")?
        };
        
        Ok(Self {
            seal_key,
            total_vram: config.total_vram,
            used_vram: 0,
        })
    }
}
```

---

### 5.2 Seal Signature Computation

```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

impl VramManager {
    fn compute_seal_signature(&self, shard: &SealedShard) -> Vec<u8> {
        let message = format!("{}|{}|{}|{}", 
            shard.shard_id, 
            shard.digest, 
            shard.sealed_at.duration_since(UNIX_EPOCH).unwrap().as_secs(),
            shard.gpu_device
        );
        
        let mut mac = HmacSha256::new_from_slice(self.seal_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        
        mac.finalize().into_bytes().to_vec()
    }
}
```

---

### 5.3 Seal Verification

```rust
use subtle::ConstantTimeEq;

impl VramManager {
    pub fn verify_seal(&self, shard: &SealedShard) -> Result<()> {
        let expected = self.compute_seal_signature(shard);
        
        // Timing-safe comparison
        if expected.ct_eq(&shard.signature).into() {
            Ok(())
        } else {
            Err(VramError::SealVerificationFailed)
        }
    }
}
```

---

## 6. Configuration

### 6.1 Environment Variables

**Supported**:
```bash
# Path to seal key file
WORKER_SEAL_KEY_FILE=/etc/llorch/secrets/worker-seal-key

# Or derive from worker token
WORKER_API_TOKEN=<token>
```

**NOT supported** (security violation):
```bash
# ❌ FORBIDDEN: Key in environment
WORKER_SEAL_KEY=<key_bytes>
```

---

### 6.2 File Locations

**Recommended paths**:
```
/etc/llorch/secrets/worker-seal-key       # Production
/run/credentials/worker-orcd/seal_key     # Systemd
~/.config/llorch/seal-key                 # Development
```

**File permissions**:
```bash
# Create with correct permissions
sudo mkdir -p /etc/llorch/secrets
sudo chmod 0700 /etc/llorch/secrets
sudo touch /etc/llorch/secrets/worker-seal-key
sudo chmod 0600 /etc/llorch/secrets/worker-seal-key
sudo chown worker-orcd:worker-orcd /etc/llorch/secrets/worker-seal-key
```

---

## 7. Testing Requirements

### 7.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_key_not_logged() {
        let key = SecretKey::derive_from_token("test", b"domain")?;
        let debug_str = format!("{:?}", key);
        // Should not compile (no Debug trait)
    }
    
    #[test]
    fn test_key_zeroized_on_drop() {
        let key_bytes = {
            let key = SecretKey::derive_from_token("test", b"domain")?;
            let ptr = key.as_bytes().as_ptr();
            // Read bytes before drop
            unsafe { std::slice::from_raw_parts(ptr, 32).to_vec() }
        };
        // After drop, memory should be zeroed
        // (Hard to test reliably, but zeroize crate provides this)
    }
    
    #[test]
    fn test_file_permissions_validated() {
        // Create file with wrong permissions
        let path = "/tmp/test-key-world-readable";
        fs::write(path, "test-key")?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o644))?;
        
        let result = SecretKey::load_from_file(path);
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen(_))));
    }
}
```

---

## 8. Dependencies

**Required crates**:
- `zeroize` — Secure memory cleanup
- `thiserror` — Error types
- `sha2` — For HKDF (key derivation)
- `hkdf` — Key derivation function

**Optional**:
- `subtle` — Constant-time comparison (if needed)

---

## 9. Implementation Priority

### Phase 1: M0 Essentials
1. ✅ `SecretKey` type with zeroize on drop
2. ✅ `load_from_file()` with permission validation
3. ✅ `derive_from_token()` with HKDF
4. ✅ No Debug/Display/Serialize traits
5. ✅ Error types

### Phase 2: Production Hardening
6. ⬜ `from_systemd_credential()` support
7. ⬜ File permission validation on all platforms
8. ⬜ Comprehensive unit tests
9. ⬜ Integration with vram-residency

---

## 10. Open Questions

**Q1**: Should we support hardware-based keys (TPM, HSM)?  
**A**: Defer to post-M0. File-based keys sufficient for M0.

**Q2**: Should keys be rotatable without restart?  
**A**: Defer to post-M0. Manual restart acceptable for M0.

**Q3**: Should we support multiple key versions (for rotation)?  
**A**: Defer to post-M0. Single key sufficient for M0.

---

## 11. References

**Specifications**:
- `bin/worker-orcd-crates/vram-residency/.specs/20_security.md` — Security requirements
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md` — Consumer expectations
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issue #3 (credential exposure)

**Standards**:
- NIST SP 800-108 — Key derivation
- FIPS 140-2 — Cryptographic module security

---

**End of Requirements Document**
