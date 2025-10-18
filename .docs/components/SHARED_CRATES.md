# Shared Crates Library

**Location:** `bin/shared-crates/`  
**Purpose:** Reusable security, validation, and utility libraries  
**Last Updated:** 2025-10-18 (TEAM-096 documentation)

## Overview

Collection of production-ready shared libraries providing security, validation, logging, and utility functionality across the rbee ecosystem. These crates are **already implemented** and ready to use.

---

## Security Crates

### 1. auth-min ✅ PRODUCTION-READY
**Location:** `bin/shared-crates/auth-min/`  
**Status:** ✅ Security-hardened, production-ready  
**Purpose:** Minimal authentication primitives

**Features:**
- ✅ Timing-safe token comparison (prevents CWE-208)
- ✅ Token fingerprinting (SHA-256, safe for logs)
- ✅ Bearer token parsing (RFC 6750 compliant)
- ✅ Bind policy enforcement (loopback detection)

**API:**
```rust
use auth_min::{timing_safe_eq, token_fp6, parse_bearer};

// Timing-safe comparison
if timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
    // Authenticated
}

// Safe fingerprint for logs
let fp = token_fp6(&token); // "a3f2c1"
tracing::info!(identity = %format!("token:{}", fp), "auth success");

// Parse Bearer token
let token = parse_bearer(Some("Bearer secret-token"));
```

**Use Cases:**
- ✅ queen-rbee ↔ rbee-hive authentication
- ✅ rbee-hive ↔ llm-worker-rbee authentication
- ✅ API token validation
- ✅ Audit logging (safe fingerprints)

**Dependencies:** `sha2`, `http`, `hex`

---

### 2. jwt-guardian ✅ PRODUCTION-READY
**Location:** `bin/shared-crates/jwt-guardian/`  
**Status:** ✅ Enterprise-grade JWT security  
**Purpose:** JWT lifecycle management with revocation

**Features:**
- ✅ RS256/ES256 signature validation
- ✅ Clock-skew tolerance (±5 min)
- ✅ Redis-backed revocation lists
- ✅ Short-lived tokens (15 min)
- ✅ Algorithm whitelist (no HS256 confusion)
- ✅ Audience validation

**API:**
```rust
use jwt_guardian::{JwtValidator, ValidationConfig, RevocationList};

// Create validator
let config = ValidationConfig::default()
    .with_issuer("llama-orch")
    .with_audience("api")
    .with_clock_skew(300);

let validator = JwtValidator::new(public_key_pem, config)?;

// Validate token
let claims = validator.validate(&token)?;

// Check revocation (optional)
let revocation = RevocationList::connect("redis://localhost").await?;
if revocation.is_revoked(&claims.jti).await? {
    return Err(JwtError::TokenRevoked);
}
```

**Use Cases:**
- ✅ User authentication (future)
- ✅ Service-to-service auth
- ✅ API gateway integration
- ✅ Token revocation on logout/compromise

**Dependencies:** `jsonwebtoken`, `redis` (optional), `sha2`, `chrono`

---

### 3. secrets-management ✅ PRODUCTION-READY
**Location:** `bin/shared-crates/secrets-management/`  
**Status:** ✅ Security-hardened credential handling  
**Purpose:** Secure credential loading and management

**Features:**
- ✅ File-based secret loading (not env vars)
- ✅ Systemd LoadCredential support
- ✅ HKDF-SHA256 key derivation
- ✅ Automatic memory zeroization
- ✅ Permission validation (rejects 0644)
- ✅ Timing-safe verification
- ✅ Logging safety (never logs secrets)

**API:**
```rust
use secrets_management::{Secret, SecretKey};

// Load API token from file
let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;

// Verify incoming request (timing-safe)
if token.verify(&received_token) {
    // Authenticated
}

// Load cryptographic key
let seal_key = SecretKey::load_from_file("/etc/llorch/secrets/seal-key")?;

// Derive key from token (HKDF)
let derived_key = SecretKey::derive_from_token(
    &api_token,
    b"llorch-seal-key-v1"
)?;

// Systemd credentials
let token = Secret::from_systemd_credential("api_token")?;
```

**Use Cases:**
- ✅ API token storage
- ✅ Encryption key management
- ✅ HMAC signing keys
- ✅ Production credential loading

**Dependencies:** `secrecy`, `zeroize`, `subtle`, `hkdf`, `sha2`

---

## Validation & Safety Crates

### 4. input-validation ✅ PRODUCTION-READY
**Location:** `bin/shared-crates/input-validation/`  
**Status:** ✅ BDD-tested validation library  
**Purpose:** Input sanitization and validation

**Features:**
- ✅ Log injection prevention
- ✅ Path traversal prevention
- ✅ Command injection prevention
- ✅ SQL injection prevention
- ✅ Property-based testing (proptest)
- ✅ Performance benchmarks

**API:**
```rust
use input_validation::{validate_log_message, validate_path, validate_command};

// Prevent log injection
let safe_msg = validate_log_message(user_input)?;

// Prevent path traversal
let safe_path = validate_path(user_path)?;

// Prevent command injection
let safe_cmd = validate_command(user_command)?;
```

**Use Cases:**
- ✅ User input sanitization
- ✅ Log message validation
- ✅ File path validation
- ✅ Command argument validation

**Dependencies:** `thiserror`

---

### 5. deadline-propagation ✅ IMPLEMENTED
**Location:** `bin/shared-crates/deadline-propagation/`  
**Status:** ✅ Request timeout propagation  
**Purpose:** Deadline propagation across service boundaries

**Features:**
- ✅ HTTP header-based deadline propagation
- ✅ Timeout calculation
- ✅ Deadline inheritance
- ✅ Cancellation support

**API:**
```rust
use deadline_propagation::{Deadline, propagate_deadline};

// Parse deadline from request
let deadline = Deadline::from_header(req.headers())?;

// Propagate to downstream service
let client_req = propagate_deadline(client_req, &deadline);

// Check if deadline exceeded
if deadline.is_exceeded() {
    return Err(TimeoutError);
}
```

**Use Cases:**
- ✅ queen-rbee → rbee-hive → llm-worker-rbee timeout propagation
- ✅ Request cancellation
- ✅ Distributed tracing

**Dependencies:** `http`, `tracing`

---

## Audit & Logging Crates

### 6. audit-logging ✅ PRODUCTION-READY
**Location:** `bin/shared-crates/audit-logging/`  
**Status:** ✅ Tamper-evident audit logging  
**Purpose:** Secure audit trail with hash chains

**Features:**
- ✅ Tamper-evident hash chains
- ✅ Log injection prevention (uses input-validation)
- ✅ Disk space monitoring
- ✅ Structured JSON logs
- ✅ HMAC signatures (optional)
- ✅ Ed25519 signatures (optional, platform mode)

**API:**
```rust
use audit_logging::{AuditLogger, AuditEvent};

// Create logger
let logger = AuditLogger::new("/var/log/llorch/audit.log")?;

// Log event
logger.log(AuditEvent {
    action: "worker.spawn",
    actor: "queen-rbee",
    resource: "worker-abc123",
    outcome: "success",
    metadata: json!({"model": "tinyllama"}),
})?;
```

**Use Cases:**
- ✅ Security audit trails
- ✅ Compliance logging
- ✅ Forensic analysis
- ✅ Tamper detection

**Dependencies:** `sha2`, `input-validation`, `hmac` (optional), `ed25519-dalek` (optional)

---

## Utility Crates

### 7. gpu-info ✅ ACTIVE
**Location:** `bin/shared-crates/gpu-info/`  
**Status:** ✅ Cross-platform GPU detection  
**Purpose:** GPU detection and information

**Features:**
- ✅ CUDA detection (NVML)
- ✅ Metal detection (macOS)
- ✅ CPU fallback
- ✅ Device enumeration
- ✅ VRAM information

**API:**
```rust
use gpu_info::{detect_gpus, Backend};

// Detect available GPUs
let gpus = detect_gpus()?;

for gpu in gpus {
    println!("Backend: {:?}, Device: {}, VRAM: {} MB", 
        gpu.backend, gpu.device_id, gpu.vram_mb);
}
```

**Use Cases:**
- ✅ Backend selection (cuda, metal, cpu)
- ✅ Device enumeration
- ✅ Capability detection

**Dependencies:** Platform-specific (nvml, metal-rs)

---

### 8. model-catalog ✅ ACTIVE
**Location:** `bin/shared-crates/model-catalog/`  
**Status:** ✅ SQLite model tracking  
**Purpose:** Model download tracking

**Features:**
- ✅ SQLite persistence
- ✅ Model metadata storage
- ✅ Download tracking
- ✅ Async/await support

**API:**
```rust
use model_catalog::ModelCatalog;

let catalog = ModelCatalog::new(Path::new(".rbee/models/catalog.db")).await?;

// Register model
catalog.register_model(&ModelInfo {
    reference: "TinyLlama".to_string(),
    provider: "hf".to_string(),
    local_path: "/path/to/model.gguf".to_string(),
    size_bytes: 1024,
    downloaded_at: now(),
}).await?;

// Find model
let model = catalog.find_model("TinyLlama", "hf").await?;
```

**Use Cases:**
- ✅ Model download tracking
- ✅ Local path resolution
- ✅ Disk usage monitoring

**Dependencies:** `sqlx`, `tokio`

---

### 9. hive-core ✅ ACTIVE
**Location:** `bin/shared-crates/hive-core/`  
**Status:** ✅ Shared hive types  
**Purpose:** Common types for rbee-hive

**Features:**
- ✅ Worker state types
- ✅ Pool management types
- ✅ Shared constants

---

## Deprecated/Unused Crates

### ⚠️ narration-core + narration-macros
**Status:** ❌ DEPRECATED - Use `tracing` instead  
**Reason:** Replaced by standard `tracing` crate

---

## Integration Guide

### For New Components

When building new components, use these crates instead of rolling your own:

1. **Authentication?** → Use `auth-min` or `jwt-guardian`
2. **Secrets?** → Use `secrets-management`
3. **User input?** → Use `input-validation`
4. **Audit trail?** → Use `audit-logging`
5. **Timeouts?** → Use `deadline-propagation`
6. **GPU detection?** → Use `gpu-info`

### Adding to Cargo.toml

```toml
[dependencies]
auth-min = { path = "../shared-crates/auth-min" }
jwt-guardian = { path = "../shared-crates/jwt-guardian" }
secrets-management = { path = "../shared-crates/secrets-management" }
input-validation = { path = "../shared-crates/input-validation" }
audit-logging = { path = "../shared-crates/audit-logging" }
deadline-propagation = { path = "../shared-crates/deadline-propagation" }
gpu-info = { path = "../shared-crates/gpu-info" }
```

---

## Current Usage Matrix

| Crate | queen-rbee | rbee-hive | llm-worker-rbee | rbee-keeper |
|-------|-----------|-----------|-----------------|-------------|
| **auth-min** | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet |
| **jwt-guardian** | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet |
| **secrets-management** | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet |
| **input-validation** | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet |
| **audit-logging** | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet |
| **deadline-propagation** | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet | 🔴 Not yet |
| **gpu-info** | ❌ N/A | ✅ Used | ✅ Used | ❌ N/A |
| **model-catalog** | ❌ N/A | ✅ Used | ❌ N/A | ❌ N/A |
| **hive-core** | ❌ N/A | ✅ Used | ❌ N/A | ❌ N/A |

**Legend:**
- ✅ Used - Currently integrated
- 🔴 Not yet - Ready to use, not integrated
- ❌ N/A - Not applicable to this component

---

## Recommended Integration Priority

### Phase 1 - Security Hardening
1. **auth-min** → Add to queen-rbee, rbee-hive, llm-worker-rbee
   - Implement Bearer token authentication
   - Use timing-safe comparison
   - Add token fingerprinting to logs

2. **secrets-management** → Add to all components
   - Replace env var secrets with file-based loading
   - Add systemd credential support
   - Implement memory zeroization

3. **input-validation** → Add to all HTTP endpoints
   - Validate all user inputs
   - Prevent log injection
   - Sanitize file paths

### Phase 2 - Operational Excellence
4. **audit-logging** → Add to queen-rbee, rbee-hive
   - Log worker spawn/shutdown
   - Log authentication events
   - Log configuration changes

5. **deadline-propagation** → Add to request chain
   - Propagate timeouts from queen → hive → worker
   - Implement request cancellation
   - Add distributed tracing

### Phase 3 - Enterprise Features
6. **jwt-guardian** → Add to queen-rbee
   - Implement user authentication
   - Add token revocation
   - Support API gateway integration

---

## Testing

Each crate has comprehensive tests:

```bash
# Security crates
cargo test -p auth-min
cargo test -p jwt-guardian
cargo test -p secrets-management

# Validation crates
cargo test -p input-validation
cargo test -p deadline-propagation

# Audit crates
cargo test -p audit-logging

# Utility crates
cargo test -p gpu-info
cargo test -p model-catalog
```

---

## Documentation

Each crate has detailed README with:
- ✅ API documentation
- ✅ Usage examples
- ✅ Security considerations
- ✅ Testing instructions
- ✅ Integration guide

**Read the READMEs:**
- `bin/shared-crates/auth-min/README.md`
- `bin/shared-crates/jwt-guardian/README.md`
- `bin/shared-crates/secrets-management/README.md`
- `bin/shared-crates/input-validation/README.md`
- `bin/shared-crates/audit-logging/README.md`
- `bin/shared-crates/deadline-propagation/README.md`

---

## Security Review

All security crates have been:
- ✅ Reviewed for timing attacks
- ✅ Reviewed for memory safety
- ✅ Reviewed for log injection
- ✅ Property-tested with proptest
- ✅ Benchmarked for performance

**Security audit commands in each README.**

---

**Created by:** Various teams  
**Documented by:** TEAM-096 | 2025-10-18  
**Status:** ✅ Production-ready, awaiting integration
