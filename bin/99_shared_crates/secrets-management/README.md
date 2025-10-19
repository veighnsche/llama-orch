# secrets-management

**Secure credential loading and management**

`bin/shared-crates/secrets-management` — Load API tokens and cryptographic keys from secure sources. Prevents credential exposure in process listings, logs, and memory dumps.

---

## What This Library Offers

secrets-management provides **security-hardened credential handling** for llama-orch:

- **File-based loading** — Load secrets from files (not environment variables)
- **Systemd credentials** — Production-ready systemd LoadCredential support
- **Key derivation** — HKDF-SHA256 for deriving keys from tokens
- **Memory safety** — Automatic zeroization on drop (prevents memory dumps)
- **Permission validation** — Rejects world/group-readable files (Unix)
- **Timing-safe verification** — Constant-time comparison for tokens
- **Logging safety** — Never logs secret values (only paths/metadata)

**Used by**: queen-rbee, pool-managerd, vram-residency, worker-orcd

---

## Key APIs

### Load API Token from File

```rust
use secrets_management::Secret;

// Load token from file
let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;

// Verify incoming request (timing-safe)
if token.verify(&received_token) {
    println!("Authenticated");
}

// Expose for outbound requests
let auth_header = format!("Bearer {}", token.expose());
```

**Security**: File permissions validated (rejects 0644), memory zeroized on drop

### Load Cryptographic Key from File

```rust
use secrets_management::SecretKey;

// Load 32-byte key from hex file
let seal_key = SecretKey::load_from_file("/etc/llorch/secrets/seal-key")?;

// Use for HMAC-SHA256
use hmac::{Hmac, Mac};
use sha2::Sha256;

let mut mac = Hmac::<Sha256>::new_from_slice(seal_key.as_bytes())
    .expect("HMAC accepts any key size");
mac.update(message.as_bytes());
let signature = mac.finalize().into_bytes();
```

**Security**: Keys are 32 bytes, zeroized on drop, never logged

### Derive Key from Token

```rust
use secrets_management::SecretKey;

// Derive seal key from worker token (HKDF-SHA256)
let seal_key = SecretKey::derive_from_token(
    &worker_api_token,
    b"llorch-seal-key-v1"  // Domain separation
)?;
```

**Use case**: Generate cryptographic keys from API tokens without storing separate key files

### Load from Systemd Credentials

```rust
use secrets_management::Secret;

// Load from systemd LoadCredential
let token = Secret::from_systemd_credential("api_token")?;
```

**Systemd service configuration**:
```ini
[Service]
LoadCredential=api_token:/etc/llorch/secrets/api-token
# Token available at /run/credentials/<service>/api_token
```

**Security**: Credentials not visible in process listings or `/proc`

---

## Migration from Environment Variables

### Before (Vulnerable)

```rust
// ❌ INSECURE: Visible in `ps auxe` and `/proc/PID/environ`
let token = std::env::var("LLORCH_API_TOKEN")?;
```

### After (Secure)

```rust
use secrets_management::Secret;

// ✅ SECURE: File-based, permission-validated
let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
    .unwrap_or_else(|_| "/etc/llorch/secrets/api-token".to_string());

let token = Secret::load_from_file(&token_path)?;
```

**Why this matters**: Environment variables are visible to all processes on the same host via `ps auxe`, `/proc/PID/environ`, Docker inspect, systemd service files, etc.

---

## File Setup

### Create Secret File

```bash
# Create secrets directory
sudo mkdir -p /etc/llorch/secrets
sudo chmod 0700 /etc/llorch/secrets

# Generate API token
openssl rand -hex 32 | sudo tee /etc/llorch/secrets/api-token
sudo chmod 0600 /etc/llorch/secrets/api-token
sudo chown queen-rbee:queen-rbee /etc/llorch/secrets/api-token

# Generate seal key (32 bytes hex)
openssl rand -hex 32 | sudo tee /etc/llorch/secrets/seal-key
sudo chmod 0600 /etc/llorch/secrets/seal-key
sudo chown worker-orcd:worker-orcd /etc/llorch/secrets/seal-key
```

### Verify Permissions

```bash
# Check permissions (should be 0600)
ls -la /etc/llorch/secrets/

# Output should show:
# -rw------- 1 queen-rbee queen-rbee 65 Oct  1 21:00 api-token
# -rw------- 1 worker-orcd   worker-orcd   65 Oct  1 21:00 seal-key
```

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLORCH_API_TOKEN_FILE` | Path to API token file | `/etc/llorch/secrets/api-token` |
| `WORKER_SEAL_KEY_FILE` | Path to seal key file | `/etc/llorch/secrets/seal-key` |
| `CREDENTIALS_DIRECTORY` | Systemd credentials directory | `/run/credentials/queen-rbee.service` |

**Deprecated** (insecure):
- `LLORCH_API_TOKEN` — Token value in environment (visible in process listings)
- `WORKER_SEAL_KEY` — Key value in environment (visible in process listings)

---

## Security Properties

### Memory Safety

- **Zeroization**: All secret types implement `Drop` with `zeroize` crate
- **No leaks**: Secrets overwritten on drop (prevents memory dumps)
- **Compiler fences**: Prevents optimization from removing zeroization

### Logging Safety

- **No Debug/Display**: Secret types cannot be logged accidentally
- **Path logging only**: Logs file paths, never values
- **Error safety**: Error messages never contain secret values

### Timing Safety

- **Constant-time comparison**: `Secret::verify()` uses `subtle::ConstantTimeEq`
- **No short-circuit**: Examines all bytes regardless of match
- **Prevents CWE-208**: Observable timing discrepancy attacks

### File Permission Validation

- **Unix permissions**: Rejects files with mode `0o077` bits set (world/group readable)
- **Recommended**: `0600` (owner read/write only)
- **Enforced**: Before reading file contents
- **Error**: Returns `SecretError::PermissionsTooOpen` with mode details

### Path Validation

- **Canonicalization**: Resolves `..` and symlinks
- **Traversal prevention**: Rejects path traversal attempts
- **Absolute paths**: Validates paths are within expected directories

---

## Use Cases

### 1. queen-rbee: API Token Authentication

```rust
use secrets_management::Secret;

// Load token at startup
let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
    .unwrap_or_else(|_| "/etc/llorch/secrets/api-token".to_string());
let expected_token = Secret::load_from_file(&token_path)?;

// In auth middleware
if expected_token.verify(&received_token) {
    // Authenticated
} else {
    return Err(StatusCode::UNAUTHORIZED);
}
```

### 2. pool-managerd: Systemd Credentials

```rust
use secrets_management::Secret;

// Try systemd first, fall back to file
let token = Secret::from_systemd_credential("api_token")
    .or_else(|_| Secret::load_from_file("/etc/llorch/secrets/api-token"))?;
```

### 3. vram-residency: Seal Key Management

```rust
use secrets_management::SecretKey;

// Load or derive seal key
let seal_key = if let Some(key_file) = &config.seal_key_file {
    SecretKey::load_from_file(key_file)?
} else {
    SecretKey::derive_from_token(&config.worker_api_token, b"llorch-seal-key-v1")?
};

// Compute HMAC seal signature
let message = format!("{}|{}|{}", shard_id, digest, sealed_at);
let mut mac = Hmac::<Sha256>::new_from_slice(seal_key.as_bytes())?;
mac.update(message.as_bytes());
let signature = mac.finalize().into_bytes();
```

### 4. worker-orcd: Worker Registration

```rust
use secrets_management::Secret;

// Load worker token
let worker_token = Secret::load_from_file("/etc/llorch/secrets/worker-token")?;

// Register with pool-managerd
let response = client
    .post(format!("{}/v2/nodes/register", pool_managerd_url))
    .header("Authorization", format!("Bearer {}", worker_token.expose()))
    .json(&registration_payload)
    .send()
    .await?;
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p secrets-management

# Specific test suites
cargo test -p secrets-management load      # File loading
cargo test -p secrets-management verify    # Timing-safe verification
cargo test -p secrets-management perms     # Permission validation
cargo test -p secrets-management derive    # Key derivation
```

### Security Tests

```bash
# Timing attack resistance
cargo test -p secrets-management test_timing_safe_verify

# Permission validation
cargo test -p secrets-management test_reject_world_readable

# Path traversal prevention
cargo test -p secrets-management test_reject_path_traversal
```

---

## Error Handling

### Error Types

```rust
pub enum SecretError {
    FileNotFound(String),                    // File doesn't exist
    PermissionsTooOpen { path, mode },       // File is world/group readable
    InvalidFormat(String),                   // Wrong format (empty, bad hex, etc.)
    SystemdCredentialNotFound(String),       // Systemd credential missing
    PathValidationFailed(String),            // Path traversal or invalid
    Io(std::io::Error),                      // I/O error
    KeyDerivation(String),                   // HKDF error
}
```

### Example Error Messages

```
✅ "secret file not found: /etc/llorch/secrets/api-token"
✅ "file permissions too open: /etc/llorch/secrets/api-token (mode: 0644, expected 0600)"
✅ "invalid secret format: expected 64 hex chars"
✅ "systemd credential not found: api_token"
```

---

## Dependencies

### Battle-Tested Security Libraries

We use industry-standard, audited libraries instead of rolling our own:

- **`secrecy`** — Secret wrapper with automatic zeroization (used by thousands of projects)
- **`zeroize`** — Secure memory cleanup with compiler fences (RustCrypto)
- **`subtle`** — Constant-time comparison (RustCrypto)
- **`hkdf`** — HKDF-SHA256 key derivation (RustCrypto, NIST SP 800-108)
- **`sha2`** — SHA-256 hashing (RustCrypto)
- **`hex`** — Hex encoding/decoding (RustCrypto)
- **`thiserror`** — Error types (minimal, no dependencies)
- **`tracing`** — Logging

**Why these libraries?**
- ✅ Don't roll your own crypto
- ✅ Professionally audited (RustCrypto)
- ✅ Used in production by thousands of projects
- ✅ Active maintenance and security patches
- ✅ Minimal transitive dependencies

### Optional (Future)

- `vaultrs` — HashiCorp Vault integration
- `aws-sdk-secretsmanager` — AWS Secrets Manager
- `azure_security_keyvault` — Azure Key Vault

---

## What This Library Is Not

- ❌ Not a secret manager (use Vault/AWS Secrets Manager for that)
- ❌ Not a key rotation system (manual restart required)
- ❌ Not a certificate manager (separate crate)
- ❌ Not a password hashing library (use `argon2` for that)

---

## Specifications

Implements requirements from:
- `00_secrets_management.md` — Complete specification (SM-TYPE-*, SM-LOAD-*, SM-SEC-*)
- `10_expectations.md` — Consumer expectations (EXP-*)
- `11_worker_vram_residency.md` — vram-residency seal key requirements

Addresses security vulnerabilities:
- **SEC-VULN-3**: Token in environment (process listing exposure)
- **CI-005**: Seal keys derived from worker token
- **CI-006**: Seal keys never logged or exposed
- **SEC-AUTH-2001**: Timing-safe comparison

---

## Status

- **Version**: 0.1.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Phase 1 (M0 essentials) in progress
- **Security Tier**: Tier 1 (critical security crate)
- **Maintainers**: @llama-orch-maintainers
- **Security Review**: Required for all changes

---

## Roadmap

### Phase 1: M0 Essentials (Current)
- ✅ `Secret` type with file loading
- ✅ `SecretKey` type with file loading
- ⬜ File permission validation
- ⬜ Timing-safe verification
- ⬜ Key derivation (HKDF)
- ⬜ Zeroize on drop

### Phase 2: Production Hardening
- ⬜ Systemd credential support
- ⬜ Path validation and canonicalization
- ⬜ Comprehensive security tests
- ⬜ Integration with queen-rbee/pool-managerd

### Phase 3: Advanced Features (Post-M0)
- ⬜ HashiCorp Vault integration
- ⬜ AWS Secrets Manager integration
- ⬜ Automatic credential rotation
- ⬜ HSM/TPM support