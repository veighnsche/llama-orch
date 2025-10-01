# Secrets Management — Consumer Expectations

**Status**: Draft  
**Version**: 0.1.0  
**Last Updated**: 2025-10-01  
**Owners**: @llama-orch-maintainers

---

## 0. Document Overview

This document specifies what **consumers** of the `secrets-management` crate expect from its API, organized by consumer type. Each expectation is mapped to specific requirements in the main specification (`00_secrets_management.md`).

**Purpose**: Ensure the secrets-management crate meets the needs of all its consumers without over-engineering or under-delivering.

---

## 1. Consumer Overview

### 1.1 Current Consumers (EXP-CONSUMER-1001)

**Active consumers requiring migration**:

1. **orchestratord** (`bin/orchestratord`)
   - Current: Loads API token from `LLORCH_API_TOKEN` environment variable
   - Need: Load from file with permission validation
   - Priority: **P0** (critical security vulnerability)

2. **pool-managerd** (`bin/pool-managerd`)
   - Current: Loads API token from `LLORCH_API_TOKEN` environment variable
   - Need: Load from file with permission validation
   - Priority: **P0** (critical security vulnerability)

3. **auth-min** (`bin/shared-crates/auth-min`)
   - Current: Uses environment variable for bind policy enforcement
   - Need: Helper to check if token file exists for startup validation
   - Priority: **P1** (operational improvement)

### 1.2 Future Consumers (EXP-CONSUMER-1002)

**Planned consumers** (post-M0):

1. **vram-residency** (`bin/worker-orcd-crates/vram-residency`)
   - Need: Load seal keys from file or derive from worker token
   - Priority: **M0+1** (worker-orcd implementation)

2. **worker-orcd** (`bin/worker-orcd`)
   - Need: Load worker API token for registration with pool-managerd
   - Priority: **M0+1** (worker-orcd implementation)

3. **audit-logging** (`bin/shared-crates/audit-logging`)
   - Need: Load audit log signing keys (future)
   - Priority: **Post-M0** (audit log signing)

---

## 2. orchestratord Expectations

### 2.1 Current State (EXP-ORCH-2001)

**Current Implementation** (`bin/orchestratord/src/app/auth_min.rs:41`):
```rust
let expected_token = std::env::var("LLORCH_API_TOKEN").ok().filter(|t| !t.is_empty());
```

**Security Issue**: Environment variables visible in process listings (`ps auxe`, `/proc/PID/environ`)

**Also Used In**:
- `bin/orchestratord/src/api/nodes.rs:26` — Duplicate auth logic
- `bin/orchestratord/src/clients/pool_manager.rs:49` — Outbound client auth

### 2.2 Expected API (EXP-ORCH-2002)

**What orchestratord expects from secrets-management**:

**EXP-ORCH-2002-R1**: Load API token from file path

**EXP-ORCH-2002-R2**: Support configurable file path via environment variable

**EXP-ORCH-2002-R3**: Validate file permissions (reject world/group readable)

**EXP-ORCH-2002-R4**: Trim whitespace from token

**EXP-ORCH-2002-R5**: Provide timing-safe verification method

**EXP-ORCH-2002-R6**: Never log token value (only path/fingerprint)

**EXP-ORCH-2002-R7**: Clear token from memory on drop

**EXP-ORCH-2002-R8**: Expose token value for Bearer header construction

**Reference Implementation**:
```rust
use secrets_management::Secret;

// Load token from file
let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
    .unwrap_or_else(|_| "/etc/llorch/secrets/api-token".to_string());

let token = Secret::load_from_file(&token_path)
    .map_err(|e| format!("Failed to load API token: {}", e))?;

// Verify incoming request
if token.verify(&received_token) {
    // Authenticated
}

// Or expose for outbound requests
let auth_header = format!("Bearer {}", token.expose());
```

**Maps to Spec Requirements**:
- SM-TYPE-3002 — `Secret` type
- SM-LOAD-4004 — `Secret::load_from_file()`
- SM-SEC-5002 — Logging safety
- SM-SEC-5003 — Timing-safe verification
- SM-SEC-5004 — File permission validation

### 2.3 Migration Path (EXP-ORCH-2003)

**Step 1**: Add secrets-management dependency
```toml
[dependencies]
secrets-management = { path = "../shared-crates/secrets-management" }
```

**Step 2**: Update `app/auth_min.rs`
```rust
// OLD
let expected_token = std::env::var("LLORCH_API_TOKEN").ok().filter(|t| !t.is_empty());

// NEW
use secrets_management::Secret;

fn load_api_token() -> Result<Secret, String> {
    let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
        .unwrap_or_else(|_| "/etc/llorch/secrets/api-token".to_string());
    
    Secret::load_from_file(&token_path)
        .map_err(|e| format!("Failed to load API token: {}", e))
}

// In middleware
let expected_token = load_api_token()?;
if expected_token.verify(&received_token) {
    // Authenticated
}
```

**Step 3**: Update `clients/pool_manager.rs`
```rust
// OLD
let api_token = std::env::var("LLORCH_API_TOKEN").ok();

// NEW
let api_token = load_api_token().ok().map(|s| s.expose().to_string());
```

**Step 4**: Remove duplicate logic in `api/nodes.rs`

**Step 5**: Update documentation and deployment guides

### 2.4 Backward Compatibility (EXP-ORCH-2004)

**Requirement**: Support gradual migration without breaking existing deployments

**Strategy**:
```rust
fn load_api_token() -> Result<Secret, String> {
    // Try file first (new method)
    if let Ok(token_file) = std::env::var("LLORCH_API_TOKEN_FILE") {
        return Secret::load_from_file(&token_file)
            .map_err(|e| format!("Failed to load from file: {}", e));
    }
    
    // Fall back to environment variable (deprecated)
    if let Ok(token_value) = std::env::var("LLORCH_API_TOKEN") {
        tracing::warn!(
            "Loading API token from LLORCH_API_TOKEN environment variable (DEPRECATED). \
             Use LLORCH_API_TOKEN_FILE instead for better security."
        );
        return Secret::from_env("LLORCH_API_TOKEN")
            .map_err(|e| format!("Failed to load from env: {}", e));
    }
    
    Err("No API token configured (set LLORCH_API_TOKEN_FILE or LLORCH_API_TOKEN)".to_string())
}
```

**Deprecation Timeline**:
- v0.1.0: Support both methods, warn on environment variable
- v0.2.0: Deprecate environment variable method
- v1.0.0: Remove environment variable support

---

## 3. pool-managerd Expectations

### 3.1 Current State (EXP-POOL-3001)

**Current Implementation** (`bin/pool-managerd/src/api/auth.rs:33`):
```rust
let expected_token = std::env::var("LLORCH_API_TOKEN")
    .ok()
    .filter(|t| !t.is_empty())
    .ok_or_else(|| {
        tracing::error!("LLORCH_API_TOKEN not configured");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
```

**Security Issue**: Same as orchestratord (environment variable exposure)

### 3.2 Expected API (EXP-POOL-3002)

**What pool-managerd expects from secrets-management**:

**EXP-POOL-3002-R1**: Identical API to orchestratord (consistency)

**EXP-POOL-3002-R2**: Load API token from file path

**EXP-POOL-3002-R3**: Support systemd credentials for production deployment

**EXP-POOL-3002-R4**: Validate file permissions

**EXP-POOL-3002-R5**: Timing-safe verification

**Reference Implementation**:
```rust
use secrets_management::Secret;

// Load token (try systemd first, then file)
fn load_api_token() -> Result<Secret, String> {
    // Try systemd credential first (production)
    if let Ok(token) = Secret::from_systemd_credential("api_token") {
        return Ok(token);
    }
    
    // Fall back to file
    let token_path = std::env::var("LLORCH_API_TOKEN_FILE")
        .unwrap_or_else(|_| "/etc/llorch/secrets/api-token".to_string());
    
    Secret::load_from_file(&token_path)
        .map_err(|e| format!("Failed to load API token: {}", e))
}
```

**Systemd Service Configuration**:
```ini
[Service]
LoadCredential=api_token:/etc/llorch/secrets/api-token
Environment=CREDENTIALS_DIRECTORY=/run/credentials/pool-managerd.service
```

**Maps to Spec Requirements**:
- SM-TYPE-3002 — `Secret` type
- SM-LOAD-4004 — `Secret::load_from_file()`
- SM-LOAD-4002 — Systemd credential support (via `Secret` variant)

### 3.3 Migration Path (EXP-POOL-3003)

**Step 1**: Add secrets-management dependency

**Step 2**: Update `api/auth.rs`
```rust
// OLD
let expected_token = std::env::var("LLORCH_API_TOKEN")...

// NEW
use secrets_management::Secret;

// Load once at startup
let expected_token = load_api_token()?;

// In middleware
if expected_token.verify(&received_token) {
    // Authenticated
}
```

**Step 3**: Update tests to use file-based tokens

**Step 4**: Update deployment documentation

---

## 4. auth-min Expectations

### 4.1 Current State (EXP-AUTH-4001)

**Current Implementation** (`bin/shared-crates/auth-min/src/policy.rs:140`):
```rust
let token = std::env::var("LLORCH_API_TOKEN").ok().filter(|t| !t.is_empty());
```

**Used For**: Startup bind policy enforcement (refuse non-loopback bind without token)

### 4.2 Expected API (EXP-AUTH-4002)

**What auth-min expects from secrets-management**:

**EXP-AUTH-4002-R1**: Check if token is configured (without loading it)

**EXP-AUTH-4002-R2**: Validate token meets minimum length requirements

**EXP-AUTH-4002-R3**: Support both file and environment variable sources

**Reference Implementation**:
```rust
use secrets_management::Secret;

pub fn enforce_startup_bind_policy(bind_addr: &str) -> Result<()> {
    if is_loopback_addr(bind_addr) {
        return Ok(()); // Loopback always OK
    }
    
    // Non-loopback: token MUST be configured
    let token_configured = std::env::var("LLORCH_API_TOKEN_FILE").is_ok()
        || std::env::var("LLORCH_API_TOKEN").is_ok();
    
    if !token_configured {
        return Err(AuthError::BindPolicyViolation(
            "Refusing to bind non-loopback without token configured".to_string()
        ));
    }
    
    Ok(())
}
```

**Alternative**: Load token at startup to validate it exists
```rust
pub fn enforce_startup_bind_policy(bind_addr: &str) -> Result<()> {
    if is_loopback_addr(bind_addr) {
        return Ok(());
    }
    
    // Try to load token (validates it exists and is readable)
    let _token = load_api_token()
        .map_err(|e| AuthError::BindPolicyViolation(format!(
            "Token required for non-loopback bind but failed to load: {}", e
        )))?;
    
    Ok(())
}
```

**Maps to Spec Requirements**:
- SM-LOAD-4004 — `Secret::load_from_file()`
- SM-LOAD-4005 — `Secret::from_env()` (deprecated)

### 4.3 Integration Notes (EXP-AUTH-4003)

**Decision**: auth-min should NOT depend on secrets-management

**Rationale**:
- auth-min is lower-level (used by secrets-management tests)
- Circular dependency risk
- Bind policy only needs to check if token exists, not load it

**Recommendation**: Keep current implementation, update to check for `LLORCH_API_TOKEN_FILE` in addition to `LLORCH_API_TOKEN`

---

## 5. vram-residency Expectations

### 5.1 Requirements (EXP-VRAM-5001)

**Source**: `bin/worker-orcd-crates/vram-residency/.specs/11_worker_vram_residency.md`

**What vram-residency expects from secrets-management**:

**EXP-VRAM-5001-R1**: Load 32-byte seal keys from file

**EXP-VRAM-5001-R2**: Derive seal keys from worker API token using HKDF

**EXP-VRAM-5001-R3**: Load from systemd credentials

**EXP-VRAM-5001-R4**: Access key bytes for HMAC-SHA256 computation

**EXP-VRAM-5001-R5**: Zeroize keys on drop

**EXP-VRAM-5001-R6**: Never log or expose keys

**EXP-VRAM-5001-R7**: Validate file permissions (0600)

### 5.2 Expected API (EXP-VRAM-5002)

**Reference Implementation**:
```rust
use secrets_management::SecretKey;
use hmac::{Hmac, Mac};
use sha2::Sha256;

pub struct VramManager {
    seal_key: SecretKey,
    // ...
}

impl VramManager {
    pub fn new(config: &WorkerConfig) -> Result<Self> {
        // Load seal key (multiple sources)
        let seal_key = if let Some(key_file) = &config.seal_key_file {
            // Load from file
            SecretKey::load_from_file(key_file)?
        } else if let Some(token) = &config.worker_api_token {
            // Derive from worker token
            SecretKey::derive_from_token(token, b"llorch-seal-key-v1")?
        } else {
            // Load from systemd credential
            SecretKey::from_systemd_credential("seal_key")?
        };
        
        Ok(Self {
            seal_key,
            total_vram: config.total_vram,
            used_vram: 0,
        })
    }
    
    fn compute_seal_signature(&self, shard: &SealedShard) -> Vec<u8> {
        let message = format!("{}|{}|{}|{}", 
            shard.shard_id, 
            shard.digest, 
            shard.sealed_at.duration_since(UNIX_EPOCH).unwrap().as_secs(),
            shard.gpu_device
        );
        
        let mut mac = Hmac::<Sha256>::new_from_slice(self.seal_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        
        mac.finalize().into_bytes().to_vec()
    }
}
```

**Maps to Spec Requirements**:
- SM-TYPE-3001 — `SecretKey` type
- SM-LOAD-4001 — `SecretKey::load_from_file()`
- SM-LOAD-4002 — `SecretKey::from_systemd_credential()`
- SM-LOAD-4003 — `SecretKey::derive_from_token()`
- SM-SEC-5001 — Memory safety (zeroize on drop)

### 5.3 Configuration (EXP-VRAM-5003)

**Environment Variables**:
```bash
# Option 1: Load from file
WORKER_SEAL_KEY_FILE=/etc/llorch/secrets/worker-seal-key

# Option 2: Derive from worker token
WORKER_API_TOKEN=<token>

# Option 3: Systemd credential (production)
# (Automatic via LoadCredential)
```

**File Setup**:
```bash
# Generate seal key
openssl rand -hex 32 > /etc/llorch/secrets/worker-seal-key
chmod 0600 /etc/llorch/secrets/worker-seal-key
chown worker-orcd:worker-orcd /etc/llorch/secrets/worker-seal-key
```

---

## 6. worker-orcd Expectations

### 6.1 Requirements (EXP-WORKER-6001)

**What worker-orcd expects from secrets-management**:

**EXP-WORKER-6001-R1**: Load worker API token for registration with pool-managerd

**EXP-WORKER-6001-R2**: Support file-based loading

**EXP-WORKER-6001-R3**: Support systemd credentials

**EXP-WORKER-6001-R4**: Expose token for Bearer header construction

### 6.2 Expected API (EXP-WORKER-6002)

**Reference Implementation**:
```rust
use secrets_management::Secret;

pub struct WorkerConfig {
    worker_api_token: Secret,
    pool_managerd_url: String,
}

impl WorkerConfig {
    pub fn from_env() -> Result<Self> {
        // Load worker token
        let worker_api_token = if let Ok(token) = Secret::from_systemd_credential("worker_token") {
            token
        } else {
            let token_path = std::env::var("WORKER_API_TOKEN_FILE")
                .unwrap_or_else(|_| "/etc/llorch/secrets/worker-token".to_string());
            Secret::load_from_file(&token_path)?
        };
        
        Ok(Self {
            worker_api_token,
            pool_managerd_url: std::env::var("POOL_MANAGERD_URL")?,
        })
    }
    
    pub fn auth_header(&self) -> String {
        format!("Bearer {}", self.worker_api_token.expose())
    }
}

// Usage in registration
async fn register_with_pool_manager(config: &WorkerConfig) -> Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v2/nodes/register", config.pool_managerd_url))
        .header("Authorization", config.auth_header())
        .json(&registration_payload)
        .send()
        .await?;
    
    // ...
}
```

**Maps to Spec Requirements**:
- SM-TYPE-3002 — `Secret` type
- SM-LOAD-4004 — `Secret::load_from_file()`
- SM-LOAD-4002 — Systemd credential support

---

## 7. Cross-Cutting Expectations

### 7.1 Error Handling (EXP-CROSS-7001)

**All consumers expect**:

**EXP-CROSS-7001-R1**: Clear error messages (without exposing secrets)

**EXP-CROSS-7001-R2**: Distinguish between file not found, permission errors, format errors

**EXP-CROSS-7001-R3**: Include file path in errors (for debugging)

**Example Error Messages**:
```
✅ GOOD: "secret file not found: /etc/llorch/secrets/api-token"
✅ GOOD: "file permissions too open: /etc/llorch/secrets/api-token (mode: 0644, expected 0600)"
✅ GOOD: "invalid secret format: expected 64 hex chars"

❌ BAD: "failed to load secret"
❌ BAD: "error: abc123def456..." (leaks secret value)
```

### 7.2 Logging (EXP-CROSS-7002)

**All consumers expect**:

**EXP-CROSS-7002-R1**: Log successful loads (with path, not value)

**EXP-CROSS-7002-R2**: Log failures (with error, not value)

**EXP-CROSS-7002-R3**: Warn when using deprecated methods (environment variables)

**Example Logs**:
```
✅ GOOD: "Secret loaded from file: /etc/llorch/secrets/api-token"
✅ GOOD: "Secret key derived from token (domain: llorch-seal-key-v1)"
✅ GOOD: "Failed to load secret: file not found"

❌ BAD: "Loaded token: abc123..." (leaks secret)
❌ BAD: "Token verification failed for: xyz789..." (leaks secret)
```

### 7.3 Testing (EXP-CROSS-7003)

**All consumers expect**:

**EXP-CROSS-7003-R1**: Test helpers for creating temporary secret files

**EXP-CROSS-7003-R2**: Test fixtures with known secrets

**EXP-CROSS-7003-R3**: Mock systemd credential directory

**Example Test Helper**:
```rust
#[cfg(test)]
pub mod test_helpers {
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;
    
    pub struct TestSecretFile {
        pub dir: TempDir,
        pub path: PathBuf,
    }
    
    impl TestSecretFile {
        pub fn new(content: &str) -> Self {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("test-secret");
            
            fs::write(&path, content).unwrap();
            
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&path).unwrap().permissions();
                perms.set_mode(0o600);
                fs::set_permissions(&path, perms).unwrap();
            }
            
            Self { dir, path }
        }
    }
}
```

### 7.4 Documentation (EXP-CROSS-7004)

**All consumers expect**:

**EXP-CROSS-7004-R1**: Rustdoc examples for common use cases

**EXP-CROSS-7004-R2**: Migration guide from environment variables

**EXP-CROSS-7004-R3**: Deployment guide (file permissions, systemd)

**EXP-CROSS-7004-R4**: Security best practices

---

## 8. Non-Functional Expectations

### 8.1 Performance (EXP-PERF-8001)

**All consumers expect**:

**EXP-PERF-8001-R1**: Loading secrets is fast (< 10ms for local files)

**EXP-PERF-8001-R2**: Verification is constant-time (no timing leaks)

**EXP-PERF-8001-R3**: No unnecessary allocations or copies

**EXP-PERF-8001-R4**: Zeroization is efficient (no performance impact)

### 8.2 Reliability (EXP-RELIABLE-8002)

**All consumers expect**:

**EXP-RELIABLE-8002-R1**: No panics (all errors returned as `Result`)

**EXP-RELIABLE-8002-R2**: Drop never panics (critical for cleanup)

**EXP-RELIABLE-8002-R3**: Thread-safe (can be used in async contexts)

**EXP-RELIABLE-8002-R4**: No global state (except environment variables)

### 8.3 Security (EXP-SECURITY-8003)

**All consumers expect**:

**EXP-SECURITY-8003-R1**: Secrets never logged or displayed

**EXP-SECURITY-8003-R2**: Timing-safe comparison (no timing leaks)

**EXP-SECURITY-8003-R3**: Memory zeroized on drop (no memory leaks)

**EXP-SECURITY-8003-R4**: File permissions validated (no world-readable files)

**EXP-SECURITY-8003-R5**: Path traversal prevented (no `../` attacks)

---

## 9. Priority Matrix

### 9.1 Feature Priority (EXP-PRIORITY-9001)

| Feature | orchestratord | pool-managerd | vram-residency | worker-orcd | Priority |
|---------|---------------|---------------|----------------|-------------|----------|
| `Secret::load_from_file()` | ✅ Required | ✅ Required | ❌ Not needed | ✅ Required | **P0** |
| `Secret::verify()` | ✅ Required | ✅ Required | ❌ Not needed | ❌ Not needed | **P0** |
| `Secret::expose()` | ✅ Required | ✅ Required | ❌ Not needed | ✅ Required | **P0** |
| `SecretKey::load_from_file()` | ❌ Not needed | ❌ Not needed | ✅ Required | ❌ Not needed | **P1** |
| `SecretKey::derive_from_token()` | ❌ Not needed | ❌ Not needed | ✅ Required | ❌ Not needed | **P1** |
| `SecretKey::from_systemd_credential()` | ⚠️ Nice to have | ⚠️ Nice to have | ✅ Required | ⚠️ Nice to have | **P1** |
| `Secret::from_env()` (deprecated) | ⚠️ Backward compat | ⚠️ Backward compat | ❌ Not needed | ❌ Not needed | **P2** |
| File permission validation | ✅ Required | ✅ Required | ✅ Required | ✅ Required | **P0** |
| Timing-safe comparison | ✅ Required | ✅ Required | ❌ Not needed | ❌ Not needed | **P0** |
| Zeroize on drop | ✅ Required | ✅ Required | ✅ Required | ✅ Required | **P0** |

### 9.2 Implementation Phases (EXP-PRIORITY-9002)

**Phase 1 (M0)**: Core functionality for orchestratord/pool-managerd
- `Secret` type with `load_from_file()`, `verify()`, `expose()`
- File permission validation
- Timing-safe comparison
- Zeroize on drop
- Error types with clear messages

**Phase 2 (M0+1)**: Support for vram-residency/worker-orcd
- `SecretKey` type with `load_from_file()`, `derive_from_token()`
- Systemd credential support
- HKDF key derivation
- Additional tests

**Phase 3 (Post-M0)**: Advanced features
- Vault/AWS Secrets Manager integration
- Automatic rotation
- Key versioning

---

## 10. Acceptance Criteria

### 10.1 Consumer Acceptance (EXP-ACCEPT-10001)

**orchestratord**:
- ✅ Loads token from file (not environment)
- ✅ Validates file permissions
- ✅ Uses timing-safe verification
- ✅ No token values in logs
- ✅ All tests passing

**pool-managerd**:
- ✅ Loads token from file (not environment)
- ✅ Validates file permissions
- ✅ Uses timing-safe verification
- ✅ No token values in logs
- ✅ All tests passing

**vram-residency**:
- ✅ Loads seal keys from file
- ✅ Derives seal keys from token
- ✅ Computes HMAC signatures correctly
- ✅ Keys zeroized on drop
- ✅ All tests passing

**worker-orcd**:
- ✅ Loads worker token from file
- ✅ Registers with pool-managerd successfully
- ✅ All tests passing

### 10.2 Integration Acceptance (EXP-ACCEPT-10002)

**End-to-End Tests**:
- ✅ orchestratord authenticates pool-managerd requests
- ✅ pool-managerd authenticates orchestratord requests
- ✅ worker-orcd registers with pool-managerd
- ✅ vram-residency seals shards with HMAC signatures
- ✅ All services start with file-based tokens
- ✅ All services refuse to start with bad permissions

---

## 11. Open Questions

**Q1**: Should we support token rotation without restart?  
**A**: Defer to post-M0. Manual restart acceptable for M0.

**Q2**: Should we cache loaded secrets or reload on each access?  
**A**: Load once at startup, cache in memory. No automatic reload.

**Q3**: Should we support multiple secret sources with fallback?  
**A**: Yes, try systemd → file → environment (with warnings).

**Q4**: Should we provide a CLI tool for generating/validating secrets?  
**A**: Defer to post-M0. Use `openssl rand -hex 32` for now.

**Q5**: Should we integrate with system keyrings (e.g., gnome-keyring)?  
**A**: Defer to post-M0. File-based is sufficient for server deployments.

---

## 12. References

**Consumer Specifications**:
- `bin/worker-orcd-crates/vram-residency/.specs/11_worker_vram_residency.md` — Seal key requirements
- `.specs/12_auth-min-hardening.md` — Authentication requirements
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #3

**Current Implementations**:
- `bin/orchestratord/src/app/auth_min.rs` — Current token loading
- `bin/pool-managerd/src/api/auth.rs` — Current token loading
- `bin/shared-crates/auth-min/src/policy.rs` — Bind policy enforcement

**Main Specification**:
- `00_secrets_management.md` — Complete specification

---

**End of Consumer Expectations Document**
